import numpy as np
import torch

from collections import defaultdict
from utils.generate_prompt import generate_prompt
from utils.scoring_utils import scores_stacking

def test_policy(test_envs, test_goals, agent):
    all_grasp = [g for g in test_goals['grasp'] if g in test_goals['goals'].keys()]
    all_grow_plants = [g for g in test_goals['grow_plants'] if g in test_goals['goals'].keys()]
    all_grow_herbivores = [g for g in test_goals['grow_herbivores'] if g in test_goals['goals'].keys()]
    all_grow_carnivores = [g for g in test_goals['grow_carnivores'] if g in test_goals['goals'].keys()]
    # grasp = list(np.random.choice(all_grasp, 2048, replace=True))
    # grow_plants = list(np.random.choice(all_grow_plants, 2048, replace=True))
    # grow_herbivores = list(np.random.choice(all_grow_herbivores, 2048, replace=True))
    # grow_carnivores = list(np.random.choice(all_grow_carnivores, 2048, replace=True))
    grasp = list(np.random.choice(all_grasp, 64, replace=True))
    grow_plants = list(np.random.choice(all_grow_plants, 64, replace=True))
    grow_herbivores = list(np.random.choice(all_grow_herbivores, 64, replace=True))
    grow_carnivores = list(np.random.choice(all_grow_carnivores, 64, replace=True))
    goals = grasp + grow_plants + grow_herbivores + grow_carnivores
        
    observations, infos = test_envs.reset([test_goals['goals'][g] for g in goals])
    
    test_result = {
        'grasp': [],
        'grow_plants': [],
        'grow_herbivores': [],
        'grow_carnivores': []
    }
    
    terminated = [False for _ in range(len(goals))]
    while not all(terminated):
        possible_actions = infos["possible_actions"]
        prompts = [generate_prompt(_o, _g, 'LittleZoo') for _o, _g in zip(observations, infos['goal'])]

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
            
        observations, rewards, dones, _, infos = test_envs.step(actions_command)
        
        for i in range(len(goals)):
            if dones[i] and not terminated[i]:
                terminated[i] = True
                if goals[i] in grasp:
                    test_result['grasp'].append(rewards[i])
                elif goals[i] in grow_plants:
                    test_result['grow_plants'].append(rewards[i])
                elif goals[i] in grow_herbivores:
                    test_result['grow_herbivores'].append(rewards[i])
                elif goals[i] in grow_carnivores:
                    test_result['grow_carnivores'].append(rewards[i])
                else:
                    raise ValueError(f"Goal {goals[i]} not recognized.")
    
    for key in ['grasp', 'grow_plants', 'grow_herbivores', 'grow_carnivores']:
        # test_result[key + '_eval64'] = np.mean(test_result[key][:64])
        # test_result[key + '_eval128'] = np.mean(test_result[key][:128])
        # test_result[key + '_eval256'] = np.mean(test_result[key][:256])
        # test_result[key + '_eval512'] = np.mean(test_result[key][:512])
        # test_result[key + '_eval1024'] = np.mean(test_result[key][:1024])
        test_result[key] = np.mean(test_result[key])
            
    return test_result
        
        
def test_lp(test_goals, goal_sampler):
    result = {}
    for category in ('impossibles', 'grasp', 'grow_plants', 'grow_herbivores', 'grow_carnivores'):
        sr, sr_delayed, lp = goal_sampler.compute_lp(list(np.random.choice(test_goals[category], 64, replace=False)))
        result['estimated_sr_' + category] = np.mean(sr)
        result['estimated_sr_delayed_' + category] = np.mean(sr_delayed)
        result['estimated_lp_' + category] = np.mean(lp)
    return result



def list_to_dict(list_):
    dict_ = defaultdict(list)
    for d in list_:
        for key, value in d.items():
            dict_[key].append(value)
    return dict(dict_)