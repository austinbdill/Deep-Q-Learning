import time
import torch
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from collections import deque
from utils.replay_buffer import ReplayBuffer
from networks.networks import CNN

class DQN(object):
    
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
            frame = np.mean(frame, -1)
            frame = frame[::2, ::2]
            state[:, :, i] = frame
        state = state.transpose(2, 0, 1)
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
        with torch.no_grad():
            a = self.Q(state).max(1)[1].view(1, 1)
        return a
    
    def optimize_model(self):
        if len(self.replay_buffer) > self.params["batch_size"]:
            self.optimizer.zero_grad()
            replays = self.replay_buffer.sample(self.params["batch_size"])
            state_batch = torch.cat([rep[0] for rep in replays]).to(self.device)
            action_batch = torch.cat([rep[1] for rep in replays]).to(self.device)
            next_state_batch = torch.cat([rep[2] for rep in replays]).to(self.device)
            reward_batch = torch.cat([rep[3] for rep in replays]).to(self.device)
            done_batch = [rep[4] for rep in replays]
            
            non_final_mask = torch.tensor(tuple(map(lambda d: not d, done_batch)), device=self.device, dtype=torch.bool)
            non_final_next_states = torch.stack([s for s, d in zip(next_state_batch, non_final_mask) if d])
            
            state_action_values = self.Q(state_batch).gather(1, action_batch)
            target_state_action_values = self.Q(non_final_next_states).max(1)[0].detach()
            y = reward_batch
            y[not done_batch] = y[not done_batch] + self.params["gamma"] * target_state_action_values
            
            loss = F.mse_loss(state_action_values, y)
            loss.backward()
            self.optimizer.step()
     
    def evaluate_policy(self):
        rewards = []
        print("EPISLON: ", self.epsilon)
        for e in range(self.params["eval_episodes"]):
            prev_frames = deque(maxlen=4)
            for _ in range(3):
                prev_frames.append(np.zeros((210, 160, 3)))
            #Reset environment and get initial state
            frame = self.env.reset()
            #self.env.render()
            #time.sleep(0.1)
            prev_frames.append(frame)
            state = self.extract_state(prev_frames)
            done = False
            total_reward = 0.0
            for t in range(self.params["max_eval_steps"]):
                action = self.greedy_action(state.to(self.device))
                next_frame, reward, done, _ = self.env.step(action.item())
                total_reward += reward
                prev_frames.append(next_frame)
                next_state = self.extract_state(prev_frames)
                state = next_state
                #self.env.render()
                #time.sleep(0.1)
                if done:
                    break
            rewards.append(total_reward)
        print("TESTING: ", np.average(rewards))
            
    def train(self):
        for e in range(self.params["train_episodes"]):
            prev_frames = deque(maxlen=4)
            for _ in range(3):
                prev_frames.append(np.zeros((210, 160, 3)))
            #Reset environment and get initial state
            print("episode: ", e)
            frame = self.env.reset()
            prev_frames.append(frame)
            state = self.extract_state(prev_frames)
            for t in range(self.params["max_train_steps"]):
                action = self.epsilon_greedy_action(state.to(self.device))
                next_frame, reward, done, _ = self.env.step(action.item())
                reward = torch.tensor([[np.sign(reward)]])
                prev_frames.append(next_frame)
                next_state = self.extract_state(prev_frames)
                self.replay_buffer.push(state, action, next_state, reward, done)
                state = next_state
                self.optimize_model()
                self.update_epsilon()
                if done:
                    break
            if e % self.params["update_period"] == 0:
                self.update_target()
            if e % self.params["eval_period"] == 0:
                self.evaluate_policy()
      
