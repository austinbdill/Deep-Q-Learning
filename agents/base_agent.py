import gym
import torch
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from collections import deque
from utils.replay_buffer import ReplayBuffer
from networks.networks import CNN
from skimage.color import rgb2gray
from skimage.transform import resize
from skimage.util import crop

class Agent(object):
    
    def __init__(self, env, params):

        self.env = env
        self.params = params
        self.epsilon = self.params["epsilon_start"]
        self.replay_buffer = ReplayBuffer(int(self.params["buffer_size"]))
       
        self.n_actions = self.env.action_space.n
        
        if torch.cuda.is_available():  
            self.device = "cuda:0" 
        else:  
            self.device = "cpu" 
            
        self.Q = CNN(self.n_actions).to(self.device)
        self.Q_target = CNN(self.n_actions).to(self.device)
        
        self.optimizer = optim.RMSprop(self.Q.parameters(), lr=2.5e-4)
        
    def update_target(self):
        self.Q_target.load_state_dict(self.Q.state_dict())
        
    def update_epsilon(self):
        self.epsilon -= (1/self.params["anneal_period"])*(self.params["epsilon_start"]-self.params["epsilon_end"])
        self.epsilon = max(self.params["epsilon_end"], self.epsilon)
    
    def extract_state(self, prev_frame, curr_frame):
        diff_frame = rgb2gray(curr_frame) - rgb2gray(prev_frame)
        #frame = rgb2gray(frame)
        frame = crop(diff_frame, ((25, 10), (0,0)))
        frame = resize(frame, (84, 84))
        state = torch.from_numpy(frame).unsqueeze(0).unsqueeze(0).float()
        return state
            
    def epsilon_greedy_action(self, state):
        sample = np.random.random()
        if sample > self.epsilon:
            with torch.no_grad():
                a = self.Q(state).max(1)[1].view(1, 1).detach()
        else:
            a = torch.tensor([[np.random.randint(0, self.n_actions)]], device=self.device, dtype=torch.long)
        return a
    
    def greedy_action(self, state):
        sample = np.random.random()
        if sample > 0.05:
            with torch.no_grad():
                a = self.Q(state).max(1)[1].view(1, 1).detach()
        else:
            a = torch.tensor([[np.random.randint(0, self.n_actions)]], device=self.device, dtype=torch.long)
        return a
     
    def evaluate_policy(self):
        rewards = []
        print("TESTING...")
        for e in range(self.params["eval_episodes"]):
            #Reset environment and get initial state
            frame = self.env.reset()
            #prev_frames.append(frame)
            state = self.extract_state(frame, frame)
            total_reward = 0.0
            for t in range(self.params["max_steps"]):
                action = self.greedy_action(state.to(self.device))
                next_frame, reward, done, _ = self.env.step(action.item())
                total_reward += reward
                #prev_frames.append(next_frame)
                next_state = self.extract_state(frame, next_frame)
                frame = next_frame
                state = next_state
                if done:
                    break
            print("Testing episode", e, "completed in", t, "steps.")
            print("Reward achieved:", total_reward)
            rewards.append(total_reward)
        print("Average Reward: ", np.average(rewards))
        
    def record_video(self):
        env = gym.wrappers.Monitor(self.env, 'results',video_callable=lambda episode_id: True,force = True)
        #Reset environment and get initial state
        frame = env.reset()
        #prev_frames.append(frame)
        state = self.extract_state(frame, frame)
        done = False
        while not done:
            action = self.greedy_action(state.to(self.device))
            next_frame, reward, done, _ = env.step(action.item())
            next_state = self.extract_state(frame, next_frame)
            frame = next_frame
            state = next_state
      
    def optimize_model(self):
        pass
    
    def train(self):
        pass
      
