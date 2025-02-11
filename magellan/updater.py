'''

'''

import math
import os
import pickle
import torch
import torch.nn.functional as F
from accelerate import Accelerator
from collections import OrderedDict, deque
from lamorel import BaseUpdater
from tqdm import tqdm
from utils.scoring_utils import scores_stacking

class SACUpdater(BaseUpdater):
    
    def __init__(self, model_type, minibatch_size, gradient_batch_size, goal_sampler, magellan, 
                 loading_path, gradient_minibatch_size=None):
        super(SACUpdater, self).__init__()
        self._model_type = model_type
        self._minibatch_size = minibatch_size
        self._gradient_batch_size = gradient_batch_size
        self._gradient_minibatch_size = gradient_minibatch_size
        self._goal_sampler = goal_sampler
        self.magellan = magellan
        self.loading_path = loading_path

    def _get_trainable_params(self, model, return_with_names=False):
        if return_with_names:
            return filter(lambda p: p[1].requires_grad, model.named_parameters())
        else:
            return filter(lambda p: p.requires_grad, model.parameters())
    
    def _get_filtered_params(self, model, name_filter, return_with_names=False):
        if return_with_names:
            return filter(lambda p: name_filter(p[0]), model.named_parameters())
        else:
            return filter(lambda p: name_filter(p[0]), model.parameters())
        
    def _print_trainable_parameters(self, model_parameters, name):
        """
        Prints the number of trainable parameters in the model.
        """
        trainable_params = 0
        all_param = 0
        for n, param in model_parameters:
            print(n, param.requires_grad)
            all_param += param.numel()
            if param.requires_grad:
                trainable_params += param.numel()
        print(
            f"({name}) trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}"
        )

    def perform_update(self, contexts, candidates, _current_batch_ids, **kwargs):
        if not hasattr(self, '_accelerator'):
            self._accelerator = Accelerator()
        
        self._iterator_named_filtered_params = lambda name_filter: self._get_filtered_params(self._llm_module, name_filter, True)
        
        self._policy_parameters_filter = self._llm_module.module._module_functions['score'].get_parameters_name_filter()
        self._critic_parameters_filter = self._llm_module.module._module_functions['critic'].get_parameters_name_filter()
        self._critic_target_parameters_filter = self._llm_module.module._module_functions['critic_target'].get_parameters_name_filter()
        self._iterator_policy_params = (p for n, p in self._iterator_named_filtered_params(self._policy_parameters_filter))
        self._iterator_critic_params = (p for n, p in self._iterator_named_filtered_params(self._critic_parameters_filter))
        
        if self.magellan:
            self._sr_delayed_parameters_filter = self._llm_module.module._module_functions['delayed'].get_parameters_name_filter()
            self._sr_parameters_filter = self._llm_module.module._module_functions['sr'].get_parameters_name_filter()
            self._sr_trainable_params = lambda n: self._sr_parameters_filter(n) and \
                (self._llm_module.module._module_functions['sr']._adapters not in n or self._llm_module.module._module_functions['sr']._train_llm)
            self._iterator_sr_params = (p for n, p in self._iterator_named_filtered_params(self._sr_trainable_params))
        
        if self.magellan:
            self._all_params_filter = lambda n: self._policy_parameters_filter(n) or self._critic_parameters_filter(n) or self._critic_target_parameters_filter(n) or self._sr_parameters_filter(n) or self._sr_delayed_parameters_filter(n)
        else:
            self._all_params_filter = lambda n: self._policy_parameters_filter(n) or self._critic_parameters_filter(n) or self._critic_target_parameters_filter(n)
            
        if kwargs['func'] == 'update_buffer':
            self.update_buffer(kwargs['buff_size'])
            return
        
        elif kwargs['func'] == 'set_weights':
            self.set_weights(kwargs['idx'])
            return
        
        elif kwargs['func'] == 'sr_update':
            
            self._llm_module.module._modules['_LLM_model'].set_adapter(kwargs['adapters'])
            
            if not hasattr(self, 'optimizer_sr'):
                self.optimizer_sr = torch.optim.Adam(self._iterator_sr_params, lr=kwargs["lr"])
                
                if self.loading_path is not None and os.path.exists(self.loading_path + "/optimizer_sr.checkpoint"):
                    self.optimizer_sr.load_state_dict(torch.load(self.loading_path + "/optimizer_sr.checkpoint"))
                    
                self._print_trainable_parameters(self._iterator_named_filtered_params(self._sr_parameters_filter), "SR")
                self._print_trainable_parameters(self._iterator_named_filtered_params(self._sr_trainable_params), "SR Trainable")
                self._print_trainable_parameters(self._iterator_named_filtered_params(self._sr_delayed_parameters_filter), "Delayed SR")
            
            goals = kwargs['goals']
            success = torch.tensor(kwargs['success'], dtype=torch.float32)
                    
            self.optimizer_sr.zero_grad()
            
            gradient_accumulation_steps = math.ceil(self._minibatch_size / self._gradient_batch_size)
                        
            for accumulated_batch in tqdm(range(gradient_accumulation_steps)):
                _start_idx = accumulated_batch * self._gradient_batch_size
                _stop_idx = (accumulated_batch + 1) * self._gradient_batch_size
                
                _goals = goals[_start_idx:_stop_idx]
                _success = success[_start_idx:_stop_idx]
                
                _batch_size = len(_goals)
                
                if _batch_size <= 1:
                    continue
                
                # Use LLM to compute again action probabilities and value
                output = self._llm_module(['sr'], contexts=_goals,
                                        require_grad=True, minibatch_size=_batch_size,
                                        peft_adapter=kwargs['adapters'])
                sr = scores_stacking([_o['sr'] for _o in output]).squeeze()
                                        
                # Compute sr loss
                sr_loss = F.binary_cross_entropy_with_logits(sr, _success)
                
                # Compute final loss
                loss = sr_loss / gradient_accumulation_steps
                
                # Backward
                loss.backward()
    
            self.optimizer_sr.step()
                    
            if kwargs["save_after_update"] and self._accelerator.process_index == 1:
                torch.save(self.optimizer_sr.state_dict(), kwargs["saving_path"] + "/optimizer_sr.checkpoint")
            
        else: # sac_update
                        
            # Set default LoRA adapters
            self._llm_module.module._modules['_LLM_model'].set_adapter('default')
                        
            if not hasattr(self, 'policy_optimizer'):
                self.policy_optimizer = torch.optim.Adam(self._iterator_policy_params, lr=kwargs["lr"])
                self.critic_optimizer = torch.optim.Adam(self._iterator_critic_params, lr=kwargs["lr"])
                
                if self.loading_path is not None and os.path.exists(self.loading_path + "/policy_optimizer.checkpoint"):
                    self.policy_optimizer.load_state_dict(torch.load(self.loading_path + "/policy_optimizer.checkpoint"))
                    self.critic_optimizer.load_state_dict(torch.load(self.loading_path + "/critic_optimizer.checkpoint"))
                self._print_trainable_parameters(self._iterator_named_filtered_params(self._policy_parameters_filter), "Policy")
                self._print_trainable_parameters(self._iterator_named_filtered_params(self._critic_parameters_filter), "Critic")
                self._print_trainable_parameters(self._iterator_named_filtered_params(self._critic_target_parameters_filter), "Critic Target")
                
            if not hasattr(self, 'log_alpha'):
                self.target_entropy = lambda n: torch.log(torch.FloatTensor(1.0 / n)).unsqueeze(-1)
                if self.loading_path is not None and os.path.exists(self.loading_path + "/log_alpha.checkpoint") and os.path.exists(self.loading_path + "/a_optimizer.checkpoint"):
                    self.log_alpha = torch.load(self.loading_path + "/log_alpha.checkpoint")
                    if kwargs['alpha'] == 'auto':
                        self.a_optimizer = torch.optim.Adam([self.log_alpha], lr=kwargs["lr"])
                        self.a_optimizer.load_state_dict(torch.load(self.loading_path + "/a_optimizer.checkpoint"))
                else:
                    if kwargs['alpha'] == 'auto':
                        self.log_alpha = torch.tensor(-3.0, requires_grad=True)
                        self.a_optimizer = torch.optim.Adam([self.log_alpha], lr=kwargs["a_lr"])
                    else:
                        self.log_alpha = torch.log(torch.tensor(kwargs['alpha'], requires_grad=False))

            self.alpha = self.log_alpha.detach().exp()
                
            value_loss_log = 0
            policy_loss_log = 0
            alpha_loss_log = 0
            entropy_log = []
            
            current_process_buffer = {}
            for k in ["actions", "rewards", "next_states", "dones"]:
                if isinstance(kwargs[k], torch.Tensor):
                    current_process_buffer[k] = kwargs[k][_current_batch_ids]
                else:
                    current_process_buffer[k] = [kwargs[k][i] for i in _current_batch_ids]
            
            # CRITIC UPDATE
            
            self.critic_optimizer.zero_grad()

            gradient_accumulation_steps = math.ceil(self._minibatch_size / self._gradient_batch_size)
            
            for accumulated_batch in tqdm(range(gradient_accumulation_steps)):
                _start_idx = accumulated_batch * self._gradient_batch_size
                _stop_idx = (accumulated_batch + 1) * self._gradient_batch_size

                _states = contexts[_start_idx:_stop_idx]
                _possibles_actions = candidates[_start_idx:_stop_idx]
                _next_states = current_process_buffer['next_states'][_start_idx:_stop_idx]
                _actions = current_process_buffer['actions'][_start_idx:_stop_idx]
                _rewards = current_process_buffer['rewards'][_start_idx:_stop_idx]
                _dones = current_process_buffer['dones'][_start_idx:_stop_idx]
                _gammas = kwargs["gammas"][_start_idx:_stop_idx]
                
                if len(_states) <= 1:
                    continue
                                
                _batch_size = sum([len(pa) for pa in _possibles_actions])
                    
                # Compute current Q values
                prompts = [s + a for s, a in zip(_states, _actions)]
                output = self._llm_module(['critic'], contexts=prompts, require_grad=True, 
                                          minibatch_size=_batch_size, peft_adapter='default')
                q_values = scores_stacking([_o['critic'] for _o in output]).squeeze()
                
                with torch.no_grad():
                    # Compute actions probabilities and log probabilities for the next states
                    output = self._llm_module(['score'], contexts=_next_states, candidates=_possibles_actions,
                                              require_grad=False, minibatch_size=_batch_size, peft_adapter='default')
                    scores = scores_stacking([_o['score'] for _o in output]).squeeze()
                    action_log_probs = F.log_softmax(scores, dim=-1)
                    action_probs = F.softmax(scores, dim=-1)
                    mask = ~torch.isinf(scores)
                    
                    # Compute target Q values
                    max_len = max([len(pa) for pa in _possibles_actions])
                    possible_actions_padding = [pa if len(pa) == max_len else pa + [""] * (max_len - len(pa)) for pa in _possibles_actions]
                    prompts = [obs + possible_action for obs, possible_actions in zip(_next_states, possible_actions_padding) for possible_action in possible_actions]
                    output = self._llm_module(['critic_target'], contexts=prompts, require_grad=False, minibatch_size=_batch_size, peft_adapter='critic_target')
                    q_target_values = torch.stack([_o['critic_target'].squeeze() for _o in output]).view(-1, max_len)
                    q_target_values = action_probs * (q_target_values - self.alpha * action_log_probs.masked_fill(~mask, 0.0))
                    q_target_values = q_target_values.sum(-1)
                    next_q_values = _rewards + (1 - _dones) * _gammas * q_target_values
                                    
                value_loss = F.mse_loss(q_values, next_q_values)
                value_loss = value_loss / gradient_accumulation_steps
                
                value_loss_log += value_loss.item()
                entropy_log.append(torch.mean(-torch.sum(action_probs * action_log_probs, dim=-1)))
                
                # Backward
                value_loss.backward()
                
                # Free unused memory
                del value_loss, q_values, scores, action_log_probs, action_probs, next_q_values
                torch.cuda.empty_cache()
            
            self.critic_optimizer.step()            
                       
            # Update target networks with polyak averaging
            with (torch.no_grad()):
                for (n, p), (_, p_targ) in zip(
                        ((n, p) for n, p in self._iterator_named_filtered_params(self._critic_parameters_filter)),
                        ((n, p) for n, p in self._iterator_named_filtered_params(self._critic_target_parameters_filter))):
                    # NB: We use an in-place operations "mul_", "add_" to update target
                    # params, as opposed to "mul" and "add", which would make new tensors.
                    if self._critic_parameters_filter(n):
                        p_targ.data.mul_(0.995)
                        p_targ.data.add_(0.005 * p.data)
            
            # POLICY UPDATE
            
            if kwargs["update_policy"]:
            
                self.policy_optimizer.zero_grad()
                if kwargs['alpha'] == 'auto':
                    self.a_optimizer.zero_grad()
                
                gradient_accumulation_steps = math.ceil(self._minibatch_size / self._gradient_batch_size)
                
                for accumulated_batch in tqdm(range(gradient_accumulation_steps)):
                    _start_idx = accumulated_batch * self._gradient_batch_size
                    _stop_idx = (accumulated_batch + 1) * self._gradient_batch_size

                    _states = contexts[_start_idx:_stop_idx]
                    _possibles_actions = candidates[_start_idx:_stop_idx]
                    _next_states = current_process_buffer['next_states'][_start_idx:_stop_idx]
                    _actions = current_process_buffer['actions'][_start_idx:_stop_idx]
                    _rewards = current_process_buffer['rewards'][_start_idx:_stop_idx]
                    _dones = current_process_buffer['dones'][_start_idx:_stop_idx]
                    
                    if len(_states) <= 1:
                        continue
                    
                    _batch_size = sum([len(pa) for pa in _possibles_actions])
                    
                    # Compute current Q values
                    with torch.no_grad():
                        max_len = max([len(pa) for pa in _possibles_actions])
                        possible_actions_padding = [pa if len(pa) == max_len else pa + [""] * (max_len - len(pa)) for pa in _possibles_actions]
                        prompts = [obs + possible_action for obs, possible_actions in zip(_states, possible_actions_padding) for possible_action in possible_actions]
                        output = self._llm_module(['critic'], contexts=prompts, require_grad=False, minibatch_size=_batch_size, peft_adapter='default')
                        q_values = torch.stack([_o['critic'].squeeze() for _o in output]).view(-1, max_len)

                    # Compute probs and log probs for the current states
                    output = self._llm_module(['score'], contexts=_states, candidates=_possibles_actions,
                                            require_grad=True, minibatch_size=_batch_size, peft_adapter='default')
                    scores = scores_stacking([_o['score'] for _o in output]).squeeze()
                    mask = ~torch.isinf(scores)
                    action_log_probs = F.log_softmax(scores, dim=-1)
                    action_probs = F.softmax(scores, dim=-1)
                    
                    valid_count = torch.sum(mask, dim=-1)
                    
                    policy_loss = torch.sum(action_probs * (self.alpha * action_log_probs.masked_fill(~mask, 0.0) - q_values), dim=-1)
                    policy_loss = torch.mean(policy_loss)
                    policy_loss = policy_loss / gradient_accumulation_steps
                    
                    if kwargs['alpha'] == 'auto':           
                        alpha_loss = torch.sum(action_probs.detach() * (-self.log_alpha.exp() * (action_log_probs.masked_fill(~mask, 0.0) + self.target_entropy(valid_count)).detach()), dim=-1)
                        alpha_loss = torch.mean(alpha_loss)
                        alpha_loss = alpha_loss / gradient_accumulation_steps
                    else:
                        alpha_loss = torch.tensor(0.0)
                    
                    policy_loss_log += policy_loss.item()
                    alpha_loss_log += alpha_loss.item()
                        
                    # Backward and cleanup
                    policy_loss.backward()
                    del policy_loss, action_log_probs, action_probs, mask

                    if kwargs['alpha'] == 'auto': 
                        alpha_loss.backward()
                        del alpha_loss

                    del scores, q_values, possible_actions_padding, prompts
                    torch.cuda.empty_cache()
        
                self.policy_optimizer.step()
                if kwargs['alpha'] == 'auto':
                    self.a_optimizer.step()
                
                    
            if kwargs["save_after_update"] and self._accelerator.process_index == 1:
                print("Saving model...")
                model_state_dict = OrderedDict({
                        k: v for k, v in self._iterator_named_filtered_params(self._all_params_filter)
                    })
                
                if not os.path.exists(kwargs["saving_path"]):
                    os.makedirs(kwargs["saving_path"])
                
                torch.save(model_state_dict, kwargs["saving_path"] + "/model.checkpoint")
                torch.save(self.policy_optimizer.state_dict(), kwargs["saving_path"] + "/policy_optimizer.checkpoint")
                torch.save(self.critic_optimizer.state_dict(), kwargs["saving_path"] + "/critic_optimizer.checkpoint")
                torch.save(self.log_alpha, kwargs["saving_path"] + "/log_alpha.checkpoint")
                if kwargs['alpha'] == 'auto':
                    torch.save(self.a_optimizer.state_dict(), kwargs["saving_path"] + "/a_optimizer.checkpoint")
                
                if hasattr(self, 'weights_buffer'):
                    with open(kwargs["saving_path"] + "/weights_buffer.pkl", 'wb') as f:
                        pickle.dump(self.weights_buffer, f)
                
                print("Model saved")
            
            return {'value_loss': value_loss_log, 'policy_loss': policy_loss_log, 'alpha_loss': alpha_loss_log,
                    'alpha': self.alpha.item(), 'entropy': torch.mean(torch.stack(entropy_log)).item()}
        
        self._llm_module.module._modules['_LLM_model'].set_adapter('default')

    
    def update_buffer(self, buff_size):
        if not hasattr(self, 'weights_buffer'):
            if self.loading_path is not None and os.path.exists(self.loading_path + "/weights_buffer.pkl"):
                with open(self.loading_path + "/weights_buffer.pkl", 'rb') as f:
                    self.weights_buffer = pickle.load(f)
            else:
                self.weights_buffer = deque(maxlen=buff_size)

        weights = [p.data.detach().clone() for n, p in self._iterator_named_filtered_params(self._sr_parameters_filter)]
        
        self.weights_buffer.append(weights)
        
    def set_weights(self, idx):
        for (n, p), w in zip(self._iterator_named_filtered_params(self._sr_delayed_parameters_filter), self.weights_buffer[idx]):
            p.data.copy_(w)
    
    