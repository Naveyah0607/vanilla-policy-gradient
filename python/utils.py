import random
import numpy as np
import torch.cuda


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)


def get_device():
    return 'cuda' if torch.cuda.is_available() else 'cpu'


def apply_reward_to_go(raw_reward):
    rtg_rewards = np.array([np.sum(raw_reward[i:]) for i in range(len(raw_reward))])
    rtg_rewards = (rtg_rewards - np.mean(rtg_rewards)) / (np.std(rtg_rewards) + np.finfo(np.float32).eps)
    return torch.tensor(rtg_rewards, dtype=torch.float32, device=get_device())


def apply_discount(raw_reward, gamma=0.99):
    discounted_rtg_reward = np.array([np.sum([gamma ** j * raw_reward[i+j] for j in range(len(raw_reward)-i)]) for i in range(len(raw_reward))])
    discounted_rtg_reward = (discounted_rtg_reward - np.mean(discounted_rtg_reward)) / (np.std(discounted_rtg_reward) + np.finfo(np.float32).eps)
    return torch.tensor(discounted_rtg_reward, dtype=torch.float32, device=get_device())


# Util function to apply reward-return (cumulative reward) on a list of instant-reward (from eq 6)
def apply_return(raw_reward):
    # Compute r_reward (as a list) from raw_reward
    r_reward = [np.sum(raw_reward) for _ in raw_reward]
    return torch.tensor(r_reward, dtype=torch.float32, device=get_device())