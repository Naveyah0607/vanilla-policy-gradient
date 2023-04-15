import pickle
import gymnasium as gym
import torch
from gymnasium.utils.save_video import save_video
from torch import nn
from torch.optim import Adam
from torch.distributions import Categorical
from utils import *



class PGTrainer:
    def __init__(self, params):
        self.params = params
        self.env = gym.make(self.params['env_name'])
        self.agent = Agent(env=self.env, params=self.params)
        self.actor_policy = PGPolicy(input_size=self.env.observation_space.shape[0], output_size=self.env.action_space.n, hidden_dim=self.params['hidden_dim']).to(get_device())
        self.optimizer = Adam(params=self.actor_policy.parameters(), lr=self.params['lr'])

    def run_training_loop(self):
        list_ro_reward = list()

        for ro_idx in range(self.params['n_rollout']):
            trajectory = self.agent.collect_trajectory(policy=self.actor_policy)
            loss = self.estimate_loss_function(trajectory)
            self.update_policy(loss)
            # Calculate the average reward for each trajectory
            n_rollouts = self.params['n_rollout'] 
            rollout_rewards = [sum(sum(r) for r in trajectory['reward'][i*self.params['n_rollout']:(i+1)*self.params['n_rollout']]) for i in range(n_rollouts)]
            avg_ro_reward = sum(rollout_rewards)/self.params['n_trajectory_per_rollout'];

            print(f'End of rollout {ro_idx}: Average trajectory reward is {avg_ro_reward: 0.2f}')
            # Append average rollout reward into a list
            list_ro_reward.append(avg_ro_reward)
        # Save avg-rewards as pickle files
        pkl_file_name = self.params['exp_name'] + '.pkl'
        with open(pkl_file_name, 'wb') as f:
            pickle.dump(list_ro_reward, f)
        # Save a video of the trained agent playing
        self.generate_video()
        # Close environment
        self.env.close()

    def estimate_loss_function(self, trajectory):
          loss_functions = {
            (True, True): lambda r: apply_discount(apply_reward_to_go(r)),
            (True, False): apply_reward_to_go,
            (False, True): apply_discount,
            (False, False): apply_return,
         }

          loss = []
          for t_idx in range(self.params['n_trajectory_per_rollout']):
        # Get the rewards for the current trajectory
              reward = torch.tensor(trajectory['reward'][t_idx], dtype=torch.float32)

        # Get the log-probs for the current trajectory
              log_prob = trajectory['log_prob'][t_idx]

        # Compute the loss based on the flags using switch
              loss_fn = loss_functions[(self.params['reward_to_go'], self.params['reward_discount'])]
              loss.append(-1 * (loss_fn(reward.tolist()) * log_prob).mean())

          loss = torch.stack(loss).mean()
          return loss




    def update_policy(self, loss):
        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()

    def generate_video(self, max_frame=1000):
        self.env = gym.make(self.params['env_name'], render_mode='rgb_array_list')
        obs, _ = self.env.reset()
        for _ in range(max_frame):
            action_idx, log_prob = self.actor_policy(torch.tensor(obs, dtype=torch.float32, device=get_device()))
            obs, reward, terminated, truncated, info = self.env.step(self.agent.action_space[action_idx.item()])
            if terminated or truncated:
                break
        save_video(frames=self.env.render(), video_folder=self.params['env_name'][:-3], fps=self.env.metadata['render_fps'], step_starting_index=0, episode_index=0)

class PGPolicy(nn.Module):
    def __init__(self, input_size, output_size, hidden_dim):
        super(PGPolicy, self).__init__()
        self.policy_net = nn.Sequential(
        nn.Linear(input_size, hidden_dim),
        nn.ReLU(),
        nn.Linear(hidden_dim, hidden_dim),
        nn.ReLU(),
        nn.Linear(hidden_dim, output_size),
        nn.Softmax(dim=-1)
        )


    def forward(self, obs):
        # Forward pass of policy net
        policy_dist = Categorical(self.policy_net(obs))
        action_index = policy_dist.sample()
        log_prob = policy_dist.log_prob(action_index)
        return action_index, log_prob


class Agent:
    def __init__(self, env, params=None):
        self.env = env
        self.params = params
        self.action_space = [action for action in range(self.env.action_space.n)]

    def collect_trajectory(self, policy):
        obs, _ = self.env.reset(seed=self.params['rng_seed'])
        rollout_buffer = list()
        for _ in range(self.params['n_trajectory_per_rollout']):
            trajectory_buffer = {'log_prob': list(), 'reward': list()}
            while True:
                # Convert observation to tensor
                obs_tensor = torch.tensor(obs, dtype=torch.float32)
                action_idx, log_prob = policy.forward(obs_tensor)

# Execute chosen action in the environment
                action = self.action_space[action_idx.item()]
                obs, reward, terminated, truncated, info = self.env.step(action)


                # Save log-prob and reward into the buffer
                trajectory_buffer['log_prob'].append(log_prob)
                trajectory_buffer['reward'].append(reward)
                # Check for termination criteria
                if terminated or truncated:
                    obs, _ = self.env.reset(seed=self.params['rng_seed'])
                    rollout_buffer.append(trajectory_buffer)
                    break
        rollout_buffer = self.serialize_trajectory(rollout_buffer)
        return rollout_buffer

    # Converts a list-of-dictionary into dictionary-of-list
    @staticmethod
    def serialize_trajectory(rollout_buffer):
        serialized_buffer = {'log_prob': list(), 'reward': list()}
        for trajectory_buffer in rollout_buffer:
            serialized_buffer['log_prob'].append(torch.stack(trajectory_buffer['log_prob']))
            serialized_buffer['reward'].append(trajectory_buffer['reward'])
        return serialized_buffer