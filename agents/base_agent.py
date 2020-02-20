import gym
import torch
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from collections import deque
from utils.replay_buffer import ReplayBuffer
from networks.networks import CNN

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
        
        self.optimizer = optim.Adam(self.Q.parameters())
        
    def update_target(self):
        self.Q_target.load_state_dict(self.Q.state_dict())
        
    def update_epsilon(self):
        self.epsilon -= (1/self.params["anneal_period"])*(self.params["epsilon_start"]-self.params["epsilon_end"])
        self.epsilon = max(self.params["epsilon_end"], self.epsilon)
        
    def extract_state(self, frames):
        state = np.zeros((105, 80, 4))
        for i, frame in enumerate(frames):
            frame = np.mean(frame, -1).astype(np.uint8)
            frame = frame[::2, ::2]
            state[:, :, i] = frame
        state = state.transpose(2, 0, 1) / 255.0
        state = torch.from_numpy(state).float().unsqueeze(0)
        return state
            
    def epsilon_greedy_action(self, state):
        sample = np.random.random()
        if sample > self.epsilon:
            with torch.no_grad():
                a = self.Q(state).max(1)[1].view(1, 1)
        else:
            a = torch.tensor([[np.random.randint(0, self.n_actions)]], device=self.device, dtype=torch.long)
        return a
    
    def greedy_action(self, state):
        sample = np.random.random()
        if sample > 0.05:
            with torch.no_grad():
                a = self.Q(state).max(1)[1].view(1, 1)
        else:
            a = torch.tensor([[np.random.randint(0, self.n_actions)]], device=self.device, dtype=torch.long)
        return a
     
    def evaluate_policy(self):
        rewards = []
        print("TESTING...")
        for e in range(self.params["eval_episodes"]):
            prev_frames = deque(maxlen=4)
            for _ in range(3):
                prev_frames.append(np.zeros((210, 160, 3)))
            #Reset environment and get initial state
            frame = self.env.reset()
            prev_frames.append(frame)
            state = self.extract_state(prev_frames)
            total_reward = 0.0
            for t in range(self.params["max_steps"]):
                action = self.greedy_action(state.to(self.device))
                next_frame, reward, done, _ = self.env.step(action.item())
                total_reward += reward
                prev_frames.append(next_frame)
                next_state = self.extract_state(prev_frames)
                state = next_state
                if done:
                    break
            print("Testing episode", e, "completed in", t, "steps.")
            print("Reward achieved:", total_reward)
            rewards.append(total_reward)
        print("Average Reward: ", np.average(rewards))
        
    def record_video(self):
        env = gym.wrappers.Monitor(self.env, 'results',video_callable=lambda episode_id: True,force = True)
        prev_frames = deque(maxlen=4)
        for _ in range(3):
            prev_frames.append(np.zeros((210, 160, 3)))
        #Reset environment and get initial state
        frame = env.reset()
        prev_frames.append(frame)
        state = self.extract_state(prev_frames)
        done = False
        while not done:
            action = self.greedy_action(state.to(self.device))
            next_frame, reward, done, _ = env.step(action.item())
            prev_frames.append(next_frame)
            next_state = self.extract_state(prev_frames)
            state = next_state
      
    def optimize_model(self):
        pass
    
    def train(self):
        pass
      
