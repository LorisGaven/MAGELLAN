'''

'''

import numpy as np
from collections import defaultdict, OrderedDict
from little_zoo import LittleZoo

class VectorizedEnv():
    
    def __init__(self, num_envs, train, seed=None):
        self.envs = [LittleZoo(train = train, seed = seed + i * 100) for i in range(num_envs)]
    
    def reset(self, goals=None):
        if goals is not None:
            results = [env.reset(goal) for env, goal in zip(self.envs, goals)]
        else:
            results = [env.reset() for env in self.envs]
        observations, infos = zip(*results)
        return list(observations), list_to_dict(infos)
    
    def reset_at(self, index, goal):
        observation, infos = self.envs[index].reset(goal)
        return observation, infos
    
    def step(self, actions):
        results = [env.step(action) for env, action in zip(self.envs, actions)]
        observations, rewards, dones, _, infos = zip(*results)
        return list(observations), rewards, dones, False, list_to_dict(infos)
    

def generate_goals(env, seed, distribution, filter_test):
    
    # Filter goals randomly to match the distribution
    def sample_goals(goal_list, num_samples):
        if len(goal_list) <= num_samples:
            return goal_list
        return list(np.random.choice(goal_list, num_samples, replace=False))
    
    np.random.seed(seed)
    
    furnitures = env.env_params['categories']['furniture'][:6]
    plants = env.env_params['categories']['plant'][:6]
    herbivores = env.env_params['categories']['herbivore'][:6]
    carnivores = env.env_params['categories']['carnivore'][:6]
    supplies = env.env_params['categories']['supply']
    
    furnitures_test = env.env_params['categories']['furniture'][6:]
    plants_test = env.env_params['categories']['plant'][6:]
    herbivores_test = env.env_params['categories']['herbivore'][6:]
    carnivores_test = env.env_params['categories']['carnivore'][6:]
    
    objects = furnitures + plants + herbivores + carnivores + supplies
    objects_test = furnitures_test + plants_test + herbivores_test + carnivores_test
    
    all_goals = OrderedDict()
    goals = OrderedDict()

    def get_name(obj):
        if obj in plants + plants_test:
            return obj + ' seed'
        elif obj in herbivores + carnivores + herbivores_test + carnivores_test:
            return 'baby ' + obj
        else:
            return obj

    impossibles = []
    grasp = []
    grow_plants = []
    grow_herbivores = []
    grow_carnivores = []
    
    if env.train:
        
        # Train goals
        for e1 in objects:
            e1_name = get_name(e1)
            for e2 in objects:
                e2_name = get_name(e2)
                for e3 in objects:
                    e3_name = get_name(e3)
                    for e4 in objects:
                        e4_name = get_name(e4)
                        for o in objects:
                            for t in ('Grasp', 'Grow'):
                                g = f'Goal: {t} {o}\n'
                                g += f'You see: {e1_name}, {e2_name}, {e3_name}, {e4_name}\n'
                                g += 'You are standing on: nothing\n'
                                g += 'Inventory (0/2): empty\n'
                                g += 'Action: '
                                
                                # Impossible goals
                                if (o not in (e1, e2, e3, e4) or t == 'Grow' and (o in furnitures + supplies or 'water' not in (e1, e2, e3, e4) or o in herbivores + carnivores and e1 not in plants and e2 not in plants and e3 not in plants and e4 not in plants or o in carnivores and e1 not in herbivores and e2 not in herbivores and e3 not in herbivores and e4 not in herbivores)):
                                    impossibles.append(g)
                                elif t == 'Grasp':
                                    grasp.append(g)
                                elif o in plants:
                                    grow_plants.append(g)
                                elif o in herbivores:
                                    grow_herbivores.append(g)
                                elif o in carnivores:
                                    grow_carnivores.append(g)
                                else:
                                    raise ValueError(f'Invalid goal: {g}')
                                
                                all_goals[g] = (t + ' ' + o, e1, e2, e3, e4)
        

        # Apply distribution
        goals = OrderedDict()
        impossibles = {g: all_goals.pop(g) for g in sample_goals(impossibles, distribution[0])}
        goals.update(impossibles)
        grasp = {g: all_goals.pop(g) for g in sample_goals(grasp, distribution[1])}
        goals.update(grasp)
        grow_plants = {g: all_goals.pop(g) for g in sample_goals(grow_plants, distribution[2])}
        goals.update(grow_plants)
        grow_herbivores = {g: all_goals.pop(g) for g in sample_goals(grow_herbivores, distribution[3])}
        goals.update(grow_herbivores)
        grow_carnivores = {g: all_goals.pop(g) for g in sample_goals(grow_carnivores, distribution[4])}
        goals.update(grow_carnivores)
        impossibles = list(impossibles.keys())
        grasp = list(grasp.keys())
        grow_plants = list(grow_plants.keys())
        grow_herbivores = list(grow_herbivores.keys())
        grow_carnivores = list(grow_carnivores.keys())
                                
    else:
        
        # Test goals
        for e2 in objects:
            e2_name = get_name(e2)
            for e3 in objects:
                e3_name = get_name(e3)
                for e4 in objects:
                    e4_name = get_name(e4)
                    for o in objects_test:
                        e1 = o
                        e1_name = get_name(e1)
                        for t in ('Grasp', 'Grow'):
                            g = f'Goal: {t} {o}\n'
                            g += f'You see: {e1_name}, {e2_name}, {e3_name}, {e4_name}\n'
                            g += 'You are standing on: nothing\n'
                            g += 'Inventory (0/2): empty\n'
                            g += 'Action: '
                            
                            # Impossible goals
                            if o not in (e1, e2, e3, e4) or t == 'Grow' and (o in furnitures_test + supplies or 'water' not in (e1, e2, e3, e4) or o in herbivores_test + carnivores_test and e1 not in plants and e2 not in plants and e3 not in plants and e4 not in plants or o in carnivores_test and e1 not in herbivores and e2 not in herbivores and e3 not in herbivores and e4 not in herbivores):
                                impossibles.append(g)
                            elif t == 'Grasp':
                                grasp.append(g)
                            elif o in plants_test:
                                grow_plants.append(g)
                            elif o in herbivores_test:
                                grow_herbivores.append(g)
                            elif o in carnivores_test:
                                grow_carnivores.append(g)
                            else:
                                raise ValueError('Invalid object')
                            
                            all_goals[g] = (t + ' ' + o, e1, e2, e3, e4)
            
        if filter_test:
            # Apply distribution
            goals = OrderedDict()
            impossibles = {g: all_goals.pop(g) for g in sample_goals(impossibles, distribution[0])}
            goals.update(impossibles)
            grasp = {g: all_goals.pop(g) for g in sample_goals(grasp, distribution[1])}
            goals.update(grasp)
            grow_plants = {g: all_goals.pop(g) for g in sample_goals(grow_plants, distribution[2])}
            goals.update(grow_plants)
            grow_herbivores = {g: all_goals.pop(g) for g in sample_goals(grow_herbivores, distribution[3])}
            goals.update(grow_herbivores)
            grow_carnivores = {g: all_goals.pop(g) for g in sample_goals(grow_carnivores, distribution[4])}
            goals.update(grow_carnivores)
            impossibles = list(impossibles.keys())
            grasp = list(grasp.keys())
            grow_plants = list(grow_plants.keys())
            grow_herbivores = list(grow_herbivores.keys())
            grow_carnivores = list(grow_carnivores.keys())
        else:
            goals = all_goals
        
                            
    return {'goals': goals, 'impossibles': impossibles, 'grasp': grasp, 'grow_plants': grow_plants, 
            'grow_herbivores': grow_herbivores, 'grow_carnivores': grow_carnivores}
    
    
def list_to_dict(list_):
    dict_ = defaultdict(list)
    for d in list_:
        for key, value in d.items():
            dict_[key].append(value)
    return dict(dict_)