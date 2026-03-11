import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import time

from utils.scheduler import LinearSchedule
from utils.policy import MLPPolicy
from components.buffer import ReplayBuffer


class MLPAgent:
    def __init__(self, env):
        super(MLPAgent, self).__init__()
        self.env = env
        self.n_state = self.env.get_state_size()
        self.n_action = self.env.get_action_size()

        # ---------- 超参数（沿用你原来的，只改 epsilon 部分） ----------
        self.buffer_size = 50000          # 500 episodes * 100 nodes
        self.batch_size = 256
        self.lr = 0.0005
        self.gamma = 0.99

        # epsilon 原来是 0.0 -> 0.99（而且逻辑反了），这里改成标准写法：
        # epsilon = 随机动作概率，从 1.0 衰减到 0.05
        self.epsilon_start = 1.0
        self.epsilon_finish = 0.05
        self.epsilon_time_length = 5000   # 可以按总步数再调
        self.epsilon_schedule = LinearSchedule(
            self.epsilon_start,
            self.epsilon_finish,
            self.epsilon_time_length
        )

        self.target_update_interval = 2000  # update target network every 20 episodes
        self.grad_norm_clip = 10            # avoid gradient explode

        # Q 网络 + target 网络（保持你原来的结构）
        self.net = MLPPolicy(self.n_state, self.n_action)
        self.target_net = MLPPolicy(self.n_state, self.n_action)
        self.target_net.load_state_dict(self.net.state_dict())

        self.learn_step_counter = 0
        self.buffer = ReplayBuffer(self.buffer_size, self.batch_size, self.env)
        self.params = list(self.net.parameters())
        self.optimizer = torch.optim.RMSprop(params=self.params, lr=self.lr)

    # ------------------------------------------------------------------
    #  选动作（修正后的 ε-greedy）
    # ------------------------------------------------------------------
    def choose_action(self, state, avail_action, t=0, evaluate=False):
        """
        state: 环境返回的原始状态 (task_idx, task_info, dev_info)
        avail_action: 0/1 向量，1 表示可选
        t: 总步数，用于 epsilon 退火
        evaluate: True 时为评估模式（纯 greedy）
        """
        # 编码状态
        inputs = torch.FloatTensor(self.env.encode_state(state)).unsqueeze(0)
        action_value = self.net.forward(inputs).squeeze()   # [n_action]

        # mask 不可行动作
        action_value[avail_action == 0] = -1e9

        # 评估模式：纯 greedy
        if evaluate:
            action = torch.max(action_value, dim=0)[1].item()
            return action

        # 训练模式：标准 ε-greedy（epsilon = 随机动作概率）
        epsilon = self.epsilon_schedule.eval(t)

        if np.random.rand() < epsilon:
            # 随机从可行动作里选一个
            valid_indices = np.where(avail_action > 0)[0]
            action = int(np.random.choice(valid_indices))
        else:
            # 选 Q 最大的动作
            action = torch.max(action_value, dim=0)[1].item()

        return action

    # ------------------------------------------------------------------
    #  DQN 学习（完全沿用你原来的逻辑，保持 encode_batch_state）
    # ------------------------------------------------------------------
    def learn(self):

        # update target parameters
        if self.learn_step_counter % self.target_update_interval == 0:
            self.target_net.load_state_dict(self.net.state_dict())
        self.learn_step_counter += 1

        # sample from replay buffer
        batch_state, batch_action, batch_reward, batch_next_state, batch_avail_action, batch_IDs = self.buffer.sample()

        # 这里用你原来的 encode_batch_state / decode_batch_state 设计
        batch_state = torch.FloatTensor(self.env.encode_batch_state(batch_state))
        batch_next_state = torch.FloatTensor(self.env.encode_batch_state(batch_next_state))
        batch_action = torch.LongTensor(batch_action.astype(int))
        batch_reward = torch.FloatTensor(batch_reward)
        batch_avail_action = torch.FloatTensor(batch_avail_action)

        # calculate loss
        q = torch.gather(self.net(batch_state), dim=1, index=batch_action.unsqueeze(1))
        q_next = self.target_net(batch_next_state).detach()
        q_next[batch_avail_action == 0] = -1e9
        q_target = batch_reward.view(self.batch_size, 1) + \
                   self.gamma * q_next.max(1)[0].view(self.batch_size, 1)
        loss = F.mse_loss(q, q_target)

        # update parameters
        self.optimizer.zero_grad()
        grad_norm = torch.nn.utils.clip_grad_norm_(self.net.parameters(), self.grad_norm_clip)
        if grad_norm > 0:
            print("grad_norm:", grad_norm)
        loss.backward()
        self.optimizer.step()

    # ------------------------------------------------------------------
    #  存经验
    # ------------------------------------------------------------------
    def store_transition(self, state, action, reward, next_state, avail_action):
        self.buffer.store(state, action, reward, next_state, avail_action)

    # ------------------------------------------------------------------
    #  保存 / 加载模型
    # ------------------------------------------------------------------
    def save_models(self, path):
        if not os.path.exists(path):
            os.makedirs(path)
        torch.save(self.net.state_dict(), f"{path}/net.th")
        torch.save(self.target_net.state_dict(), f"{path}/target_net.th")
        torch.save(self.optimizer.state_dict(), f"{path}/opt.th")

    def load_models(self, path):
        self.net.load_state_dict(
            torch.load(f"{path}/net.th", map_location=lambda storage, loc: storage)
        )
        self.target_net.load_state_dict(
            torch.load(f"{path}/target_net.th", map_location=lambda storage, loc: storage)
        )
        self.optimizer.load_state_dict(
            torch.load(f"{path}/opt.th", map_location=lambda storage, loc: storage)
        )
