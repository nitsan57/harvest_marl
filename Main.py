# from multitaxienv.taxi_environment import TaxiEnv
import gym
from MultiTaxiEnvgit.multitaxienv.taxi_environment import TaxiEnv
import ray
import ray.rllib.agents.ppo as ppo
import ray.rllib.agents.dqn as dqn
import numpy as np
import os
from harvest_map_env import HarvestMap

from ray.tune.registry import register_env
from Reinforce import Reinforce

# env = TaxiEnv(num_taxis=2, num_passengers=2, max_fuel=None,
#               taxis_capacity=None, collision_sensitive_domain=False,
#               fuel_type_list=None, option_to_stand_by=True)

# custom_map = [
#     '+---------------+',
#     '| : :X| :F: : : |',
#     '|X: : | : | :X| |',
#     '| : : : : : : | |',
#     '| :X:F| :X| : :X|',
#     '+---------------+',
# ]

# # TaxiEnv(domain_map=custom_map)
# # env = TaxiEnv
# # select_env = "multi-taxi"
# # register_env(select_env, lambda config: env())
# # env.reset()
# # env.render()
# # env.step([0, 1])
# # env.render()
# # exit()


def init_ppo(name, num_agents=1, env=None):

    select_env = name

    config = ppo.DEFAULT_CONFIG.copy()
    config["log_level"] = "WARN"
    # config["framework"] = "torch"

    reinforce = Reinforce(r'train_data/'+name+'_ppo', ppo.PPOTrainer, config,
                          select_env, num_agents, env)
    return reinforce


def init_dqn(name, num_agents=1, env=None):

    select_env = name

    config = dqn.DEFAULT_CONFIG.copy()
    config["log_level"] = "WARN"
    # config["framework"] = "torch"
    reinforce = Reinforce(r'train_data/'+name+'_dqn', dqn.DQNTrainer,
                          config, select_env, num_agents, env=env)
    return reinforce


def main():
    env_name = "Harvest-v0"
    env = HarvestMap
    # register_env(env_name, lambda config: HarvestMap(1))
    # obs = env.reset()
    # x = env.get_map_with_agents()

    # pong = "Taxi-v3"
    # taxi = "Taxi-v3"
    # env = gym.make(pong)

    reinforce = init_ppo(env_name,  2, env)

    # # reinforce.load_checkpoint(r'train_data/ppo/checkpoint_1/checkpoint-1')
    # agent = reinforce.train(1)

    agent = reinforce.load_checkpoint()
    reinforce.run_policy(2000)


if __name__ == "__main__":
    main()
