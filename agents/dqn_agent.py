import time
import torch
import numpy as np
import torch.nn.functional as F
from collections import deque
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from agents.base_agent import Agent

class DQN(Agent):
    
    def __init__(self, env, params):
        super().__init__(env, params)
    
    def optimize_model(self):
        if len(self.replay_buffer) > self.params["burn_in"]:
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
            target_state_action_values = self.Q_target(non_final_next_states).max(1)[0].detach()
            y = reward_batch
            y[not done_batch] = y[not done_batch] + self.params["gamma"] * target_state_action_values
            
            loss = F.mse_loss(state_action_values, y)
            loss.backward()
            self.optimizer.step()
            
    def train(self):
        for e in range(self.params["train_episodes"]):
            prev_frames = deque(maxlen=4)
            for _ in range(3):
                prev_frames.append(np.zeros((210, 160, 3)))
            #Reset environment and get initial state
            frame = self.env.reset()
            prev_frames.append(frame)
            state = self.extract_state(prev_frames)
            done = False
            total_reward = 0.0
            t = 0
            while not done:
                action = self.epsilon_greedy_action(state.to(self.device))
                next_frame, reward, done, _ = self.env.step(action.item())
                total_reward += reward
                reward = torch.tensor([[reward]])
                prev_frames.append(next_frame)
                next_state = self.extract_state(prev_frames)
                self.replay_buffer.push(state, action, next_state, reward, done)
                state = next_state
                self.optimize_model()
                self.update_epsilon()
                t += 1
            print("Training episode", e, "completed in", t, "steps.")
            print("Reward achieved:", total_reward)
            print("Epsilon:", self.epsilon)
            if (e+1) % self.params["update_period"] == 0:
                self.update_target()
            if (e+1) % self.params["eval_period"] == 0:
                self.evaluate_policy()
                self.record_video()
      
