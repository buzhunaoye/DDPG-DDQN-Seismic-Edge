import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import time

from utils.scheduler import LinearSchedule
from utils.policy import GATPolicy
from components.buffer import ReplayBuffer

class GATAgent:
    def __init__(self, env):
        super(GATAgent, self).__init__()
        self.env = env
        self.n_state = self.env.get_state_size()
        self.n_action = self.env.get_action_size()

        # TODO: hyper-parameters should be fine-tuned
        self.buffer_size = 50000 # 500 episodes * 100 nodes
        self.batch_size = 256
        self.lr = 0.0005
        self.gamma = 0.99
        self.epsilon_start = 1.0  # 一开始多探索
        self.epsilon_finish = 0.05  # 收敛后少量探索
        self.epsilon_time_length = 5000  # 或根据总步数调，比如 10000
        self.epsilon_schedule = LinearSchedule(self.epsilon_start, self.epsilon_finish, self.epsilon_time_length)
        self.target_update_interval = 2000 # update target network every 20 episodes
        self.grad_norm_clip = 10 # avoid gradient explode

        self.net = GATPolicy(3, self.n_action, self.env.max_num_nodes, self.env.M)
        self.target_net = GATPolicy(3, self.n_action, self.env.max_num_nodes, self.env.M)

        self.learn_step_counter = 0
        self.buffer = ReplayBuffer(self.buffer_size, self.batch_size, self.env)
        self.params = list(self.net.parameters())
        self.optimizer = torch.optim.RMSprop(params=self.params, lr=self.lr)

    def choose_action(self, state, avail_action, t=0, evaluate=False):
        task_idx, task_info, dev_info = state
        adj = self.env.adjs[self.env.ID]

        task_idx = torch.from_numpy(np.array(task_idx, dtype=np.float32)).unsqueeze(0)
        task_info = torch.from_numpy(np.array(task_info, dtype=np.float32)).unsqueeze(0)
        dev_info = torch.from_numpy(np.array(dev_info, dtype=np.float32)).unsqueeze(0)
        adj = torch.from_numpy(np.array(adj, dtype=np.float32)).unsqueeze(0)

        action_value = self.net.forward(task_idx, task_info, dev_info, adj).squeeze()
        action_value[avail_action == 0] = -1e9

        # 评估：纯 greedy
        if evaluate:
            action = torch.max(action_value, dim=0)[1].item()
            return action

        # 训练：ε-greedy
        epsilon = self.epsilon_schedule.eval(t)

        if np.random.rand() < epsilon:
            valid_indices = np.where(avail_action > 0)[0]
            action = int(np.random.choice(valid_indices))
        else:
            action = torch.max(action_value, dim=0)[1].item()

        return action

    def learn(self):

        #update target parameters
        if self.learn_step_counter % self.target_update_interval ==0:
            self.target_net.load_state_dict(self.net.state_dict())
        self.learn_step_counter+=1

        # sample from replay buffer
        batch_state, batch_action, batch_reward, batch_next_state, batch_avail_action, batch_IDs = self.buffer.sample()
        idx, x, y = batch_state
        target_idx, target_x, target_y = batch_next_state
        adj = np.array([self.env.adjs[ID] for ID in batch_IDs])
        batch_action = torch.LongTensor(batch_action.astype(int))
        batch_reward = torch.FloatTensor(batch_reward)
        batch_avail_action = torch.FloatTensor(batch_avail_action)
        q = torch.gather(self.net(torch.FloatTensor(idx), torch.FloatTensor(x), torch.FloatTensor(y), torch.FloatTensor(adj)), dim=1, index=batch_action.unsqueeze(1))
        q_next = self.target_net(torch.FloatTensor(target_idx), torch.FloatTensor(target_x), torch.FloatTensor(target_y), torch.FloatTensor(adj)).detach()
        q_next[batch_avail_action == 0] = -9999999
        q_target = batch_reward.view(self.batch_size, 1) + self.gamma * q_next.max(1)[0].view(self.batch_size, 1)

        loss = F.mse_loss(q, q_target)

        # update parameters
        self.optimizer.zero_grad()
        grad_norm = torch.nn.utils.clip_grad_norm_(self.net.parameters(), self.grad_norm_clip)
        if grad_norm > 0:
            print("grad_norm:", grad_norm)
        loss.backward()
        self.optimizer.step()


    def store_transition(self, state, action, reward, next_state, avail_action):
        self.buffer.store(state, action, reward, next_state, avail_action)

    
    def save_models(self, path):
        if not os.path.exists(path):
            os.makedirs(path)
        torch.save(self.net.state_dict(), "{}/net.th".format(path))
        torch.save(self.optimizer.state_dict(), "{}/opt.th".format(path))

    def load_models(self, path):
        # 加载模型文件（与 save_models 保存的文件名一致）
        checkpoint_path = os.path.join(path, 'model.pth')
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"模型文件不存在：{checkpoint_path}")

        checkpoint = torch.load(checkpoint_path)
        # 加载网络权重
        self.gat_net.load_state_dict(checkpoint['model_state_dict'])
        # 加载优化器状态（保证续训时优化器参数不重置）
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        # 设置网络为训练模式（重要！避免 BatchNorm/ Dropout 失效）
        self.gat_net.train()
        print(f"模型加载成功：{checkpoint_path}")

    def load_models(self, path):
        self.net.load_state_dict(torch.load("{}/net.th".format(path), map_location=lambda storage, loc: storage))
        self.optimizer.load_state_dict(torch.load("{}/opt.th".format(path), map_location=lambda storage, loc: storage))