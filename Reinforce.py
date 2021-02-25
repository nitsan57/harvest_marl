import ray
import os
from ray.tune.registry import register_env

import gym
import pickle
import numpy as np


class Reinforce:
    def __init__(self, folder_name, trainer, config, env_name, num_agents=1, env=None):

        self.num_agents = num_agents
        self.env_name = env_name
        self.env = env(num_agents)

        ray.shutdown()
        ray.init(include_dashboard=False, ignore_reinit_error=True)
        if env:
            register_env(env_name, lambda config: env(num_agents))
            # gym.envs.register(
            #     id=env_name,
            #     entry_point=env(num_agents))

        self.folder_name = folder_name
        self.check_point_default_path = self.folder_name+"/model_checkpoint.pkl"

        self.config = self.config_rllib(config)

        self.agent = trainer(config=config, env=env_name)

    def config_rllib(self, config):
        policies = {}
        agent_list = [str(i) for i in range(self.num_agents)]
        for i in range(self.num_agents):
            policies[str(i)] = (None, self.env.observation_space,
                                self.env.action_space, {"gamma": 0.85})

        config["multiagent"] = {
            "policies": policies,
            "policy_mapping_fn":
            lambda agent_id:
                # Randomly choose from car policies
                np.random.choice(agent_list)
        }
        return config

    def train(self, num_epochs):
        for n in range(num_epochs):
            result = self.agent.train()

            if n % 10 == 0:
                self.save_checkpoint(n)

            s = "{:3d} rewards: min: {:6.2f}, mean:{:6.2f}, max:{:6.2f}, path_len: {:6.2f}"

            # if verbose:

            print(s.format(
                n + 1,
                result["episode_reward_min"],
                result["episode_reward_mean"],
                result["episode_reward_max"],
                result["episode_len_mean"]
            ))
        return self.agent

    def run_policy(self, n_step):
        env = self.env
        obs = env.reset()
        sum_reward = 0
        actions = {}
        episode_reward = 0
        for step in range(n_step):
            for agent_id, agent_obs in obs.items():
                policy_id = self.agent.config['multiagent']['policy_mapping_fn'](
                    agent_id)
                actions[agent_id] = self.agent.compute_action(
                    agent_obs, policy_id=policy_id)
            env.render()
            obs, reward, done, info = env.step(actions)
            done = done['__all__']
            # sum up reward for all agents
            episode_reward += sum(reward.values())
            if done:
                print("cumulative reward", episode_reward)
                obs = env.reset()
                episode_reward = 0
        self.env.close_viewer()

    def save_checkpoint(self, current_epoch):
        save_path = self.agent.save(self.folder_name)
        dictionary_data = {"epoch": current_epoch, "last_save_path": save_path}
        a_file = open(self.check_point_default_path, "wb")
        pickle.dump(dictionary_data, a_file)
        a_file.close()
        print("saved checkpoint from epoch: {}".format(current_epoch))

    def load_checkpoint(self, check_point_file=None):
        check_point_path = self.check_point_default_path
        if check_point_file is not None:
            check_point_path = check_point_file
        a_file = open(check_point_path, "rb")
        output = pickle.load(a_file)
        a_file.close()
        current_epoch = output['epoch']
        save_path = output['last_save_path']
        print("loaded checkpoint from epoch: {}".format(current_epoch))
        self.agent.restore(save_path)
        return self.agent
