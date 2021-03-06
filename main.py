import argparse
import os
import sys
import yaml
import gym
from agents.dqn_agent import DQN
from agents.double_dqn_agent import DoubleDQN

parser = argparse.ArgumentParser()
parser.add_argument("--config", type=str, help='path to config file')

try:
    args = parser.parse_args()
except IOError as msg:
    parser.error(str(msg))

with open('configs/' + args.config, 'r') as f:
    params = yaml.safe_load(f)

#Make environment
env = gym.make(params["env"])

#Select agent
if params["agent"] == "DQN":
    agent = DQN(env, params)
elif params["agent"] == "DoubleDQN":
    agent = DoubleDQN(env, params)

#Tell agent to train with environment and parameters.
agent.train()
