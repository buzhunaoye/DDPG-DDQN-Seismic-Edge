import numpy as np
import time
import os

import pandas as pd  # ✅ 新增：用于把数据写到 Excel

from env.environment import Environment
from components.episodebuffer import Episode
from modules.agents.ACOagent import ToMAgent


def save(dirname, filename, data):
    """简单保存函数，把列表写成文本方便画图"""
    if not os.path.exists(dirname):
        os.makedirs(dirname)
    with open(os.path.join(dirname, filename), "w") as f:
        f.write(str(data))


def train():
    start_time = time.time()
    env = Environment()
    agent = ToMAgent(env)

    episodes = 5001

    dvr_list = []
    reward_list = []
    t = 0                      # 全局步数，用来算 epsilon

    base_dir = f"./saved/off/tom/{start_time}"

    # ✅ 新增：每 100 轮平均值记录 + Excel 路径
    avg_records_100 = []   # 每条记录: {"episode_end": ..., "avg_reward_100": ..., "avg_dvr_100": ...}
    avg_excel_path = os.path.join(base_dir, "avg_100_log.xlsx")

    for i in range(episodes):
        state = env.reset(seed=int(start_time) + i)
        ep_reward = 0.0
        done = False

        # ToM 用的是 episode 级别的 buffer，这里整局收集
        states = []
        actions = []
        rewards = []
        next_states = []
        avail_actions = []

        # 每一局开始前重置 ToM 的隐藏状态
        agent.init_hidden()

        while not done:
            avail_action = env.get_avail_actions()
            action = agent.choose_action(state, avail_action, t)

            next_state, reward, done = env.step(action)

            # 收集这一步的数据（用原始 state）
            states.append(state)
            actions.append(action)
            rewards.append(reward)
            next_states.append(next_state)
            avail_actions.append(avail_action)

            ep_reward += reward
            state = next_state
            t += 1

            if done:
                # 一局结束：算 dvr + 打印 + 记录（保持原逻辑）
                dvr_rate = env.get_metric()
                dvr_list.append(dvr_rate)
                reward_list.append(ep_reward)

                print(
                    f"episode: {i} , "
                    f"reward: {ep_reward:.3f} , "
                    f"dvr: {dvr_rate:.4f}"
                )

                # 构造一条 Episode 丢进 ToM 的 ReplayBuffer（保持原逻辑）
                episode = Episode(env)
                episode.update(
                    states, actions, rewards, next_states, avail_actions, env.ID
                )
                agent.buffer.insert_an_episode(episode)

                # buffer 里 episode 足够多就学习（保持原逻辑）
                if agent.buffer.can_sample():
                    agent.learn()

        # ✅ 新增：每 100 轮，计算最近 100 轮的平均值并写入 Excel
        # i 从 0 开始，所以 (i+1) 是当前已完成的 episode 数（100, 200, 300...）
        if (i + 1) % 100 == 0 and len(reward_list) >= 100:
            window_rewards = reward_list[-100:]
            window_dvrs = dvr_list[-100:]

            avg_reward_100 = float(np.mean(window_rewards))
            avg_dvr_100 = float(np.mean(window_dvrs))

            avg_records_100.append(
                {
                    "episode_end": i + 1,         # 截止到第几轮（100/200/300...）
                    "avg_reward_100": avg_reward_100,
                    "avg_dvr_100": avg_dvr_100,
                }
            )

            # 确保目录存在
            if not os.path.exists(base_dir):
                os.makedirs(base_dir)

            # 覆盖写入：文件里永远是“到目前为止每 100 轮的平均值”
            df = pd.DataFrame(avg_records_100)
            df.to_excel(avg_excel_path, index=False)

            print(
                f"[AVG-100] 截止 episode {i+1}："
                f"最近100轮平均 reward={avg_reward_100:.3f}, "
                f"平均 dvr={avg_dvr_100:.4f}"
            )

        # ===== 下面两段是你原来的逻辑，完全不动 =====
        # 定期用 buffer 里的 ID 更新图
        if i > 0 and i % 500 == 0:
            env.update_adjs(set(agent.buffer.get_IDs()))

        # 定期保存模型和日志
        if i % 500 == 0:
            model_dir = os.path.join(base_dir, str(i))
            agent.save_models(model_dir)

            save(base_dir, "dvr.txt", dvr_list)
            save(base_dir, "ep_reward.txt", reward_list)


if __name__ == "__main__":
    train()
