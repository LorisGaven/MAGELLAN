'''

'''

import hydra
import numpy as np
import os
import pickle
import random
import torch

from collections import deque
from lamorel import Caller, lamorel_init
from tqdm import tqdm
from transformers import set_seed

from environment import VectorizedEnv, generate_goals, list_to_dict
from goal_sampler import RandomGoalSampler, OnlineGoalSampler, EKOnlineGoalSampler, MAGELLANGoalSampler
from initializer import SequentialInitializer, WeightsLoaderInitializer, PeftInitializer
from models import LogScoringModuleFn, ValueHeadModuleFn, SRHeadModuleFn
from updater import SACUpdater
from utils.generate_prompt import generate_prompt
from utils.logs_utils import save_logs, save_goal_sampler
from utils.replay_buffer import NStepReplayBuffer
from utils.scoring_utils import scores_stacking
from utils.tests import test_policy, test_lp

# Initialize Lamorel
lamorel_init()

def collect_trajectories(train_envs, agent, goal_sampler, buffer, nb_steps, nb_envs, state=None):
    
    data = {
        "ep_len": [],
        "ep_ret": [],
        "goals": [],
        "possible_actions": [],
        "actions": [],
        "prompts": [],
        "ep_done": 0
    }
    
    if state is None:
        results = [train_envs.reset_at(i, goal_sampler.sample()) for i in range(nb_envs)]
        observations, infos = zip(*results)
        infos = list_to_dict(infos)
        observations = list(observations)
        
        # For SR training and logs
        initial_states = [generate_prompt(_o, _g) for _o, _g in zip(observations, infos['goal'])]
        
        ep_ret, ep_len = np.zeros(nb_envs), np.zeros(nb_envs)
    else:
        observations, infos = state['observations'], state['infos']
        ep_ret, ep_len = state['ep_ret'], state['ep_len']
        initial_states = state['initial_states']
    
    for _ in tqdm(range(nb_steps // nb_envs), ascii=" " * 9 + ">", ncols=100):
            
        possible_actions = infos["possible_actions"]
        prompts = [generate_prompt(_o, _g) for _o, _g in zip(observations, infos['goal'])]
        output = agent.custom_module_fns(['score'],
                                          contexts=prompts,
                                          candidates=possible_actions,
                                          require_grad=False,
                                          peft_adapter='default')
        scores = scores_stacking([_o['score'] for _o in output])
        proba_dist = torch.distributions.Categorical(logits=scores)
        sampled_actions = proba_dist.sample()
        actions_id = sampled_actions.cpu().numpy()
        
        actions_command = []
        for j in range(len(actions_id)):
            command = possible_actions[j][int(actions_id[j])]
            actions_command.append(command)
            
        data["possible_actions"].append(possible_actions)
        data["actions"].append(actions_command)
        data["prompts"].append(prompts)
            
        observations, rewards, dones, _, infos = train_envs.step(actions_command)

        for i in range(nb_envs):
            buffer.add(prompts[i], actions_command[i], rewards[i], generate_prompt(observations[i], infos['goal'][i]), dones[i], possible_actions[i])
            ep_ret[i] += rewards[i]
            ep_len[i] += 1
            if dones[i]:
                data["ep_len"].append(ep_len[i])
                data["ep_ret"].append(ep_ret[i])
                data["ep_done"] += 1
                ep_len[i], ep_ret[i] = 0, 0
                data["goals"].append(initial_states[i])
                    
                # Reset the environment
                observation, info = train_envs.reset_at(i, goal_sampler.sample())
                observations[i] = observation
                initial_states[i] = generate_prompt(observations[i], info['goal'])

                for key, value in info.items():
                    infos[key][i] = value
                
    state = {
        "observations": observations,
        "infos": infos,
        "ep_ret": ep_ret,
        "ep_len": ep_len,
        "initial_states": initial_states
    }
        
    return data, state

def reset_history():
    return {
        "ep_len": [],
        "ep_ret": [],
        "goals": [],
        "policy_loss": [],
        "value_loss": [],
        "alpha_loss": [],
        "alpha": [],
        "entropy": [],
        "possible_actions": [],
        "actions": [],
        "prompts": [],
        "sr": [],
        "sr_delayed": [],
        "lp": [],
        "update_ep": [],
        "keys": []
    }
    

@hydra.main(config_path='config', config_name='config')
def main(config_args):
    
    ##################
    # INITIALIZATION #
    ##################
        
    # Set random seed
    seed = config_args.rl_script_args.seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    set_seed(seed)
    
    loading_path = config_args.rl_script_args.loading_path
    if loading_path is not None and loading_path.split("/")[-1].startswith("seed"):
        subdirs = [int(folder) for folder in os.listdir(loading_path) if os.path.isdir(os.path.join(loading_path, folder)) and folder.isdigit()]
        loading_path = os.path.join(loading_path, str(max(subdirs)))
        
    # Load environment
    train_envs = VectorizedEnv(config_args.rl_script_args.number_envs,
                               not config_args.rl_script_args.adaptation_test, seed)
    nb_test_envs = 256
    test_envs = VectorizedEnv(nb_test_envs, False, seed)
    eval_envs = VectorizedEnv(nb_test_envs, not config_args.rl_script_args.adaptation_test, seed)
    
    train_goals = generate_goals(train_envs.envs[0], seed, config_args.rl_script_args.goals_distribution,
                                 config_args.rl_script_args.adaptation_test)
    test_goals = generate_goals(test_envs.envs[0], seed, config_args.rl_script_args.goals_distribution, False)
    eval_goals = train_goals
        
    # Load agent
    use_magellan = config_args.rl_script_args.goal_sampler == "magellan"
    if use_magellan:
        module_functions = {
            'score': LogScoringModuleFn(config_args.lamorel_args.llm_args.model_type,
                                        config_args.lamorel_args.llm_args.pre_encode_inputs),
            'critic': ValueHeadModuleFn(config_args.lamorel_args.llm_args.model_type,
                                        config_args.lamorel_args.llm_args.pre_encode_inputs,
                                        name='critic'),
            'critic_target': ValueHeadModuleFn(config_args.lamorel_args.llm_args.model_type,
                                               config_args.lamorel_args.llm_args.pre_encode_inputs,
                                               name='critic_target'),
            'sr': SRHeadModuleFn(config_args.lamorel_args.llm_args.model_type,
                                 config_args.lamorel_args.llm_args.pre_encode_inputs,
                                 name='sr', adapters=config_args.magellan_args.sr_adapters,
                                 train_llm=config_args.magellan_args.train_llm),
            'delayed': SRHeadModuleFn(config_args.lamorel_args.llm_args.model_type,
                                      config_args.lamorel_args.llm_args.pre_encode_inputs,
                                      name='delayed', adapters='delayed',
                                      train_llm=config_args.magellan_args.train_llm)
        }
    else:
        module_functions = {
            'score': LogScoringModuleFn(config_args.lamorel_args.llm_args.model_type,
                                        config_args.lamorel_args.llm_args.pre_encode_inputs),
            'critic': ValueHeadModuleFn(config_args.lamorel_args.llm_args.model_type,
                                        config_args.lamorel_args.llm_args.pre_encode_inputs,
                                        name='critic'),
            'critic_target': ValueHeadModuleFn(config_args.lamorel_args.llm_args.model_type,
                                               config_args.lamorel_args.llm_args.pre_encode_inputs,
                                               name='critic_target'),
        }
    
    agent = Caller(config_args.lamorel_args,
                   custom_updater=SACUpdater(config_args.lamorel_args.llm_args.model_type,
                                             config_args.rl_script_args.minibatch_size,
                                             config_args.rl_script_args.gradient_batch_size,
                                             config_args.rl_script_args.goal_sampler,
                                             use_magellan,
                                             loading_path),
                   custom_model_initializer=SequentialInitializer([
                        PeftInitializer(config_args.lamorel_args.llm_args.model_type,
                                        config_args.lamorel_args.llm_args.model_path,
                                        config_args.rl_script_args.use_lora,
                                        config_args.lamorel_args.llm_args.load_in_4bit,
                                        config_args.rl_script_args.lora_r,
                                        config_args.rl_script_args.lora_alpha,
                                        config_args.lamorel_args.llm_args.pre_encode_inputs),
                        WeightsLoaderInitializer(loading_path)
                    ]),
                    custom_module_functions=module_functions
            )
    
    # Load goal sampler
    if config_args.rl_script_args.goal_sampler == "random":
        print("Using random goal sampler.")
        goal_sampler = RandomGoalSampler(train_goals['goals'])
    elif config_args.rl_script_args.goal_sampler == "online":
        print("Using Online-ALP goal sampler.")
        goal_sampler = OnlineGoalSampler(train_goals['goals'], config_args.srdiff_args)
    elif config_args.rl_script_args.goal_sampler == "ek_online":
        print("Using EK-Online-ALP bucket goal sampler.")
        goal_sampler = EKOnlineGoalSampler(train_goals, config_args.srdiff_args)
    elif use_magellan:
        print("Using MAGELLAN goal sampler.")
        goal_sampler = MAGELLANGoalSampler(train_goals['goals'], agent, config_args.magellan_args)
    else:
        raise ValueError(f"Goal sampler {config_args.rl_script_args.goal_sampler} not recognized.")
    
    if loading_path is not None:
        goal_sampler.load(loading_path + "/goal_sampler.pkl")

    # Initialize buffers
    if loading_path is not None:
        with open(loading_path + "/replay_buffer.pkl", "rb") as f:
            rb = pickle.load(f)
    else:
        rb = NStepReplayBuffer(config_args.rl_script_args.buffer_size, config_args.rl_script_args.minibatch_size,
                               config_args.rl_script_args.n_steps, config_args.rl_script_args.gamma, 
                               config_args.rl_script_args.number_envs)
    
    if use_magellan:
        if loading_path is not None:
            with open(loading_path + "/goal_buffer.pkl", "rb") as f:
                goal_buffer = pickle.load(f)
            with open(loading_path + "/success_buffer.pkl", "rb") as f:
                success_buffer = pickle.load(f)
        else:
            goal_buffer = deque(maxlen=config_args.magellan_args.buffer_size)
            success_buffer = deque(maxlen=config_args.magellan_args.buffer_size)
        
    
    #################
    # TRAINING LOOP #
    #################
    
    history = reset_history()
    goal_sampler_update_results = None
    
    # Load experiment variables
    if loading_path is not None:
        with open(loading_path + "/test_results.pkl", "rb") as f:
            test_results = pickle.load(f)
        with open(loading_path + "/eval_results.pkl", "rb") as f:
            eval_results = pickle.load(f)
        ep = int(loading_path.split("/")[-1])
        nb_test = len(test_results)
    else:
        ep = 0
        nb_test = 0
        test_results = []
        eval_results = []
    
    state = None
    nb_updates = 0
    while ep < config_args.rl_script_args.num_episodes:
        
        # Test the agent
        if ep >= config_args.rl_script_args.test_freq * nb_test:
            result = test_policy(test_envs, test_goals, agent)
            if not config_args.rl_script_args.adaptation_test:
                eval_result = test_policy(eval_envs, eval_goals, agent)
            if config_args.rl_script_args.goal_sampler == "magellan":
                result.update(test_lp(test_goals, goal_sampler, config_args))
                if not config_args.rl_script_args.adaptation_test:
                    eval_result.update(test_lp(eval_goals, goal_sampler, config_args))
            nb_test += 1
            test_results.append((ep, result))
            if not config_args.rl_script_args.adaptation_test:
                eval_results.append((ep, eval_result))
            
                    
        # Collect trajectories
        data, state = collect_trajectories(train_envs, agent, goal_sampler, rb, 
                                    config_args.rl_script_args.update_freq,
                                    config_args.rl_script_args.number_envs, 
                                    state)
        ep += data['ep_done']
        
        # Update history
        history['ep_len'].extend(data['ep_len'])
        history['ep_ret'].extend(data['ep_ret'])
        history['goals'].extend(data['goals'])
        history['possible_actions'].extend(data['possible_actions'])
        history['actions'].extend(data['actions'])
        history['prompts'].extend(data['prompts'])
        
        if config_args.rl_script_args.goal_sampler == "magellan":
            goal_buffer.extend(data['goals'])
            success_buffer.extend(data['ep_ret'])
        
        save_model_and_history = (nb_updates % config_args.rl_script_args.save_freq == 1 or ep >= config_args.rl_script_args.num_episodes)
        saving_path = config_args.rl_script_args.output_dir + f"/{ep}"
        
        
        if len(rb) >= config_args.rl_script_args.minibatch_size:
            
            for _ in range(config_args.rl_script_args.nb_updates):
                
                collected_trajectories = rb.sample()
                    
                # Update the agent
                update_policy = config_args.rl_script_args.warmup_updates < nb_updates or loading_path is not None
                policy_update_results = agent.update(collected_trajectories['states'],
                                            collected_trajectories['possible_actions'],
                                            actions=collected_trajectories['actions'],
                                            rewards=collected_trajectories['rewards'],
                                            dones=collected_trajectories['dones'],
                                            next_states=collected_trajectories['next_states'],
                                            update_policy=update_policy,
                                            lr=config_args.rl_script_args.lr,
                                            a_lr=config_args.rl_script_args.a_lr,
                                            alpha=config_args.rl_script_args.alpha,
                                            gammas=collected_trajectories['gammas'],
                                            save_after_update=save_model_and_history,
                                            saving_path=saving_path,
                                            loading_path=loading_path,
                                            func='sac_update'
                                            )
                
                history['policy_loss'].append(np.mean([_r['policy_loss'] for _r in policy_update_results]))
                history['value_loss'].append(np.mean([_r['value_loss'] for _r in policy_update_results]))
                history['alpha_loss'].append(np.mean([_r['alpha_loss'] for _r in policy_update_results]))
                history['entropy'].append(np.mean([_r['entropy'] for _r in policy_update_results]))
                history['alpha'].append(policy_update_results[0]['alpha'])
                                                
                if use_magellan and len(goal_buffer) > 0:
                    # Update the SR estimator
                    p = np.arange(1, len(goal_buffer) + 1)
                    p = p / p.sum()
                    idx = np.random.choice(len(goal_buffer), size=config_args.magellan_args.batch_size, p=p)
                    goals = [goal_buffer[i] for i in idx]
                    success = [success_buffer[i] for i in idx]
                    agent.update([""] * len(goals),
                                [[""]] * len(success),
                                goals=goals,
                                success=success,
                                lr=config_args.rl_script_args.lr,
                                save_after_update=save_model_and_history,
                                saving_path=saving_path,
                                loading_path=loading_path,
                                func='sr_update',
                                adapters=config_args.magellan_args.sr_adapters
                            )
                
            # Update goal sampler state
            goal_sampler_update_results = goal_sampler.update(goals=data['goals'], returns=data['ep_ret'])
            nb_updates += 1
                
        print(f"{ep}/{config_args.rl_script_args.num_episodes} episodes done.")
            
        history['update_ep'].append(ep)
        if goal_sampler_update_results is not None:
            history['sr'].append(goal_sampler_update_results['sr'])
            history['sr_delayed'].append(goal_sampler_update_results['sr_delayed'])
            history['lp'].append(goal_sampler_update_results['lp'])
            history['keys'] = goal_sampler.keys
        
        # Save the logs  
        if save_model_and_history:
            if not os.path.exists(saving_path):
                os.makedirs(saving_path)
                
            save_goal_sampler(saving_path, goal_sampler, 
                              config_args.rl_script_args.goal_sampler)
                
            save_logs(saving_path, {
                "/history.pkl": history,
                "/test_results.pkl": test_results,
                "/eval_results.pkl": eval_results,
                "/replay_buffer.pkl": rb
            })
            
            if use_magellan:
                save_logs(saving_path, {
                    "/goal_buffer.pkl": goal_buffer,
                    "/success_buffer.pkl": success_buffer
                })
            
            history = reset_history()
    
    print("Training done.")
    agent.close()
                
if __name__ == "__main__":
    main()