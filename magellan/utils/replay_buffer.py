'''

'''

import random
import torch
from collections import deque


class ReplayBuffer:

    def __init__(self, buffer_size, batch_size):
        self.memory = deque(maxlen=buffer_size)  
        self.batch_size = batch_size
    
    def add(self, state, action, reward, next_state, done, possible_actions, gamma):
        self.memory.append((state, action, reward, next_state, done, possible_actions, gamma))
    
    def sample(self):
        experiences = random.sample(self.memory, k=self.batch_size)

        states = [e[0] for e in experiences if e is not None]
        actions = [e[1] for e in experiences if e is not None]
        rewards = torch.FloatTensor([e[2] for e in experiences if e is not None])
        next_states = [e[3] for e in experiences if e is not None]
        dones = torch.ByteTensor([e[4] for e in experiences if e is not None])
        possible_actions = [e[5] for e in experiences if e is not None]
        gammas = torch.FloatTensor([e[6] for e in experiences if e is not None])
  
        return {"states": states, "actions": actions, "rewards": rewards, "next_states": next_states, 
                "dones": dones, "possible_actions": possible_actions, "gammas": gammas}

    def __len__(self):
        return len(self.memory)
    
class NStepReplayBuffer(ReplayBuffer):
    
    def __init__(self, buffer_size, batch_size, n_steps, gamma, nb_envs):
        super(NStepReplayBuffer, self).__init__(buffer_size, batch_size)
        self.n_steps = n_steps
        self.gamma = gamma
        self.n_step_buffer = [deque(maxlen=n_steps) for _ in range(nb_envs)]
        self.env_idx = 0
        
    def add(self, state, action, reward, next_state, done, possible_actions):
        self.n_step_buffer[self.env_idx].append((state, action, reward, next_state, done, possible_actions))
        
        # Flush remaining transitions when done=True, even if buffer has less than n steps
        if done:
            while len(self.n_step_buffer[self.env_idx]) > 0:
                state, action, _, _, _, possible_actions = self.n_step_buffer[self.env_idx][0]
                reward = self.compute_discounted_return()
                _, _, _, next_state, done, _ = self.n_step_buffer[self.env_idx][-1]
                super().add(state, action, reward, next_state, done, possible_actions, self.gamma**len(self.n_step_buffer[self.env_idx]))
                self.n_step_buffer[self.env_idx].popleft()
                
        # When n-step buffer is filled, calculate n-step reward and store in replay buffer
        elif len(self.n_step_buffer[self.env_idx]) == self.n_steps:
            state, action, reward, _, _, possible_actions = self.n_step_buffer[self.env_idx][0]
            reward = self.compute_discounted_return()
            _, _, _, next_state, done, _ = self.n_step_buffer[self.env_idx][-1]
            super().add(state, action, reward, next_state, done, possible_actions, self.gamma**self.n_steps)
    
        # Move to next environment
        self.env_idx = (self.env_idx + 1) % len(self.n_step_buffer)

            
    def sample(self):
        return super().sample()
        
    def __len__(self):
        return len(self.memory)
    
    def compute_discounted_return(self):
        return sum([self.gamma**i * exp[2] for i, exp in enumerate(self.n_step_buffer[self.env_idx])])
