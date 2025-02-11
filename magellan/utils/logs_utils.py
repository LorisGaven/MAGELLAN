'''

'''

import os
import pickle

def save_logs(saving_path, logs):
    if not os.path.exists(saving_path):
        os.makedirs(saving_path)
    for path, log in logs.items():
        with open(saving_path + path, "wb") as f:
            pickle.dump(log, f)
            
def save_goal_sampler(saving_path, goal_sampler, goal_sampler_name):
    if goal_sampler_name == "magellan":
        with open(saving_path + "/goal_sampler.pkl", "wb") as f:
            pickle.dump({
                "epsilon": goal_sampler.epsilon,
                "step": goal_sampler.step,
            }, f)
    elif goal_sampler_name == "online":
        with open(saving_path + "/goal_sampler.pkl", "wb") as f:
            pickle.dump({
                "epsilon": goal_sampler.epsilon,
                "step": goal_sampler.step,
                "sr": goal_sampler.sr,
                "sr_delayed": goal_sampler.sr_delayed,
                "lp": goal_sampler.lp,
                "goals_success": goal_sampler.goals_success,
            }, f)
    elif goal_sampler_name == "ek_online":
        with open(saving_path + "/goal_sampler.pkl", "wb") as f:
            pickle.dump({
                "epsilon": goal_sampler.epsilon,
                "step": goal_sampler.step,
                "sr": goal_sampler.sr,
                "sr_delayed": goal_sampler.sr_delayed,
                "lp": goal_sampler.lp,
                "lp_bucket": goal_sampler.lp_bucket,
                "goals_success": goal_sampler.goals_success,
            }, f)
    elif goal_sampler_name == "random":
        pass
    else:
        raise ValueError(f"Invalid sampler name: {goal_sampler_name}")