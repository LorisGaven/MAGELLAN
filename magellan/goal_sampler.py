'''
    
'''

import numpy as np
import pickle
import torch
import torch.nn.functional as F
from collections import deque

class GoalSampler:
    
    def __init__(self, goals):
        self.goals = goals
        
    def sample(self):
        pass
    
    def update(self, **kwargs):
        pass
    
    def load(self, path):
        pass

    
class RandomGoalSampler(GoalSampler):
    
    def __init__(self, goals):
        super().__init__(goals)
        
    def sample(self):
        return list(self.goals.values())[np.random.randint(0, len(self.goals))]
    
    def update(self, **kwargs):
        return None
    
    def load(self, path):
        pass

            
class OnlineGoalSampler(GoalSampler):
    
    def __init__(self, goals, srdiff_args):
        super().__init__(goals)
        
        self.keys = list(self.goals.keys())
        self.keys2idx = {'\n'.join(k.split('\n')[:2]): i for i, k in enumerate(self.keys)}
        self.values = list(self.goals.values())
        self.epsilon_start = srdiff_args.epsilon_start
        self.epsilon_end = srdiff_args.epsilon_end
        self.epsilon_decay = srdiff_args.epsilon_decay
        self.epsilon = self.epsilon_start
        self.step = 0
        
        self.lp, self.sr, self.sr_delayed = np.zeros(len(self.goals)), np.zeros(len(self.goals)), np.zeros(len(self.goals))
        self.goals_success = [deque(maxlen=srdiff_args.buffer_size) for _ in range(len(self.goals))]
        
    def sample(self):
        
        sum_lp = np.sum(self.lp)       
        
        if np.random.rand() < self.epsilon or sum_lp == 0:
            return self.values[np.random.randint(0, len(self.goals))]
        else:
            p = self.lp / sum_lp
            return self.goals[np.random.choice(self.keys, p=p)]
    
    def update(self, **kwargs):

        self.epsilon = self.epsilon_end + (self.epsilon_start - self.epsilon_end) * np.exp(-1. * self.step / self.epsilon_decay)
        self.step += 1
        
        for g, r in zip(kwargs['goals'], kwargs['returns']):
            i = self.keys2idx['\n'.join(g.split('\n')[:2])]
            self.goals_success[i].append(r)
        
        self.compute_lp()
        
        return {'sr': self.sr, 'sr_delayed': self.sr_delayed, 'lp': self.lp}
    
    def load(self, path):
        
        with open(path, "rb") as f:
            data = pickle.load(f)
        
        self.epsilon = data['epsilon']
        self.step = data['step']
        
        if 'lp' in data:
            self.lp = data['lp']
            self.sr = data['sr']
            self.sr_delayed = data['sr_delayed']
            
        if 'goals_success' in data:
            self.goals_success = data['goals_success']
            
    def compute_lp(self):
        for i, buffer in enumerate(self.goals_success):
            if len(buffer) < 2:
                self.sr[i] = 0
                self.sr_delayed[i] = 0
            else:
                buffer_array = np.array(buffer)
                midpoint = len(buffer_array) // 2
                self.sr[i] = np.mean(buffer_array[midpoint:])
                self.sr_delayed[i] = np.mean(buffer_array[:midpoint])

        self.lp = np.abs(self.sr - self.sr_delayed)
        
class EKOnlineGoalSampler(GoalSampler):
    
    def __init__(self, goals, srdiff_args):
        super().__init__(goals)
        
        self.goals = goals['goals']
        self.keys = list(self.goals.keys())
        self.values = list(self.goals.values())
        self.epsilon_start = srdiff_args.epsilon_start
        self.epsilon_end = srdiff_args.epsilon_end
        self.epsilon_decay = srdiff_args.epsilon_decay
        self.epsilon = self.epsilon_start
        self.step = 0
        
        self.impossibles = goals['impossibles']
        self.grasp_goals = goals['grasp']
        self.grow_plants_goals = goals['grow_plants']
        self.grow_herbivores_goals = goals['grow_herbivores']
        self.grow_carnivores_goals = goals['grow_carnivores']
        
        self.lp, self.sr, self.sr_delayed = np.zeros(len(self.goals)), np.zeros(len(self.goals)), np.zeros(len(self.goals))
        self.lp_bucket = np.zeros(len(self.goals))
        self.goals_success = [deque(maxlen=srdiff_args.buffer_size) for _ in range(4)]
        
    def sample(self):
        
        sum_lp = np.sum(self.lp_bucket)  
        
        if np.random.rand() < self.epsilon or sum_lp == 0:
            bucket_idx = np.random.randint(0, 5)
        else:
            p = self.lp_bucket / sum_lp
            bucket_idx = np.random.choice(np.arange(len(self.goals_success)), p=p)
            
        if bucket_idx == 0:
            goal = self.grasp_goals[np.random.randint(0, len(self.grasp_goals))]
        elif bucket_idx == 1:
            goal = self.grow_plants_goals[np.random.randint(0, len(self.grow_plants_goals))]
        elif bucket_idx == 2:
            goal = self.grow_herbivores_goals[np.random.randint(0, len(self.grow_herbivores_goals))]
        elif bucket_idx == 3:
            goal = self.grow_carnivores_goals[np.random.randint(0, len(self.grow_carnivores_goals))]
        else:
            goal = self.impossibles[np.random.randint(0, len(self.impossibles))]
        
        return self.goals[goal]
        
    def update(self, **kwargs):
            
            self.epsilon = self.epsilon_end + (self.epsilon_start - self.epsilon_end) * np.exp(-1. * self.step / self.epsilon_decay)
            self.step += 1
            
            for g, r in zip(kwargs['goals'], kwargs['returns']):
                if g in self.grasp_goals:
                    self.goals_success[0].append(r)
                elif g in self.grow_plants_goals:
                    self.goals_success[1].append(r)
                elif g in self.grow_herbivores_goals:
                    self.goals_success[2].append(r)
                elif g in self.grow_carnivores_goals:
                    self.goals_success[3].append(r)
            
            self.compute_lp()
            
            return {'sr': self.sr, 'sr_delayed': self.sr_delayed, 'lp': self.lp}
        
    def load(self, path):
        
        with open(path, "rb") as f:
            data = pickle.load(f)
        
        self.epsilon = data['epsilon']
        self.step = data['step']
        
        if 'lp' in data:
            self.lp = data['lp']
            self.sr = data['sr']
            self.sr_delayed = data['sr_delayed']
            self.lp_bucket = data['lp_bucket']
            
        if 'goals_success' in data:
            self.goals_success = data['goals_success']
        
    def compute_lp(self):
        
        sr_bucket = np.zeros(4)
        sr_delayed_bucket = np.zeros(4)
        self.lp_bucket = np.zeros(4)
        for i, buffer in enumerate(self.goals_success):
            if len(buffer) >= 2:
                buffer_array = np.array(buffer)
                midpoint = len(buffer_array) // 2
                sr_bucket[i] = np.mean(buffer_array[midpoint:])
                sr_delayed_bucket[i] = np.mean(buffer_array[:midpoint])

        self.lp_bucket = np.abs(sr_bucket - sr_delayed_bucket)
        
        sr, sr_delayed, lp = [], [], []
        for g in self.keys:
            if g in self.grasp_goals:
                sr.append(sr_bucket[0])
                sr_delayed.append(sr_delayed_bucket[0])
                lp.append(self.lp_bucket[0])
            elif g in self.grow_plants_goals:
                sr.append(sr_bucket[1])
                sr_delayed.append(sr_delayed_bucket[1])
                lp.append(self.lp_bucket[1])
            elif g in self.grow_herbivores_goals:
                sr.append(sr_bucket[2])
                sr_delayed.append(sr_delayed_bucket[2])
                lp.append(self.lp_bucket[2])
            elif g in self.grow_carnivores_goals:
                sr.append(sr_bucket[3])
                sr_delayed.append(sr_delayed_bucket[3])
                lp.append(self.lp_bucket[3])
            else:
                sr.append(0.0)
                sr_delayed.append(0.0)
                lp.append(0.0)
                
        self.lp = np.array(lp)
        self.sr = np.array(sr)
        self.sr_delayed = np.array(sr_delayed)
        

class MAGELLANGoalSampler(GoalSampler):
    
    def __init__(self, goals, agent, magellan_args):
        super().__init__(goals)
        
        self.agent = agent
        self.N = magellan_args.N
        self.epsilon_start = magellan_args.epsilon_start
        self.epsilon_end = magellan_args.epsilon_end
        self.epsilon_decay = magellan_args.epsilon_decay
        self.epsilon = self.epsilon_start
        self.step = 0
        self.keys = list(self.goals.keys())
        self.values = list(self.goals.values())
        self.current_estimator_name = 'value' if magellan_args.use_value else 'sr'
        self.sr_adapters = magellan_args.sr_adapters
        
        self.recompute_freq = magellan_args.recompute_freq
        
        self.agent.update([""] * 8, [[""]] * 8, func='update_buffer', buff_size=int(self.N / self.recompute_freq + 1))
        self.sr, self.sr_delayed, self.lp = self.compute_lp(self.keys)
                
    def sample(self):
        
        sum_lp = np.sum(self.lp)       
        
        if np.random.rand() < self.epsilon or sum_lp == 0:
            return self.values[np.random.randint(0, len(self.goals))]
        else:
            p = self.lp / sum_lp
            return self.goals[np.random.choice(self.keys, p=p)]
        
    def update(self, **kwargs):
        
        self.epsilon = self.epsilon_end + (self.epsilon_start - self.epsilon_end) * np.exp(-1. * self.step / self.epsilon_decay)
        self.step += 1
        
        if self.step % self.recompute_freq == 0:
            self.agent.update([""] * 8, [[""]] * 8, func='update_buffer', buff_size=int(self.N / self.recompute_freq + 1))
            self.sr, self.sr_delayed, self.lp = self.compute_lp(self.keys)
        
        return {'sr': self.sr, 'sr_delayed': self.sr_delayed, 'lp': self.lp}
    
    def load(self, path):
        
        with open(path, "rb") as f:
            data = pickle.load(f)
        
        self.epsilon = data['epsilon']
        self.step = data['step']
        
    def compute_lp(self, goals):
        
        # Compute delayed sr
        self.agent.update([""] * 8, [[""]] * 8, func='set_weights', idx=0)
        output = self.agent.custom_module_fns(['delayed'], contexts=goals, require_grad=False, peft_adapter='delayed_adapters')
        sr_delayed = F.sigmoid(torch.stack([_o['delayed'][0] for _o in output]).squeeze()).numpy()
        
        # Compute current sr
        output = self.agent.custom_module_fns([self.current_estimator_name], contexts=goals, require_grad=False, peft_adapter=self.sr_adapters)
        sr = F.sigmoid(torch.stack([_o[self.current_estimator_name][0] for _o in output]).squeeze()).numpy()
        
        # Compute absolute lp
        lp = np.abs(sr - sr_delayed)
        
        # For numerical stability
        lp[lp < 0.01] = 0.0
        
        return sr, sr_delayed, lp