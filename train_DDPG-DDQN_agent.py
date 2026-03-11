import os
import time
import numpy as np
import pandas as pd  # ✅ 新增：写 Excel 用

from env.environment import Environment
from modules.agents.DDPGDDQNagent import MLPAgent


def save(dirname, filename, data):
    """简单的保存函数，把列表写到指定目录的 txt 文件里"""
    if not os.path.exists(dirname):
        os.makedirs(dirname)
    with open(os.path.join(dirname, filename), "w") as f:
        f.write(str(data))


def train():
    start_time = time.time()
    env = Environment()
    agent = MLPAgent(env)

    # 先跑少一点测试，确认没问题再加大
    episodes = 5001
    t = 0                      # 全局步数
    dvr_list = []
    reward_list = []

    # 保存目录（和你原来的 ./saved/off/mlp/xxx 一致）
    base_dir = f"./saved/off/mlp/{start_time}"

    # ✅ 新增：每 100 轮的「平均值」记录列表 & Excel 路径
    avg_records_100 = []  # 里面存 dict: {episode_end, avg_reward_100, avg_dvr_100}
    excel_path = os.path.join(base_dir, "avg_100_log.xlsx")

    for i in range(episodes):
        state = env.reset(seed=int(start_time) + i)
        ep_reward = 0.0
        done = False

        while not done:
            avail_action = env.get_avail_actions()
            action = agent.choose_action(state, avail_action, t)

            next_state, reward, done = env.step(action)
            agent.store_transition(state, action, reward, next_state, avail_action)
            ep_reward += reward

            if agent.buffer.can_sample():
                agent.learn()

            state = next_state
            t += 1

            if done:
                # 一局结束后同时记录 reward 和 dvr（保持你原来的逻辑）
                dvr_rate = env.get_metric()
                dvr_list.append(dvr_rate)
                reward_list.append(ep_reward)
                print(
                    f"episode: {i} , "
                    f"reward: {ep_reward:.3f} , "
                    f"dvr: {dvr_rate:.4f}"
                )

        # ✅ 新增：每 100 轮计算一次「最近 100 轮的平均值」，并写入 Excel
        # i 从 0 开始，所以 (i+1) 是第几轮（100, 200, 300...）
        if (i + 1) % 100 == 0 and len(reward_list) >= 100:
            # 最近 100 个 episode 的窗口（[-100:] 保证不影响前面的性能/逻辑）
            window_rewards = reward_list[-100:]
            window_dvrs = dvr_list[-100:]

            avg_reward_100 = float(np.mean(window_rewards))
            avg_dvr_100 = float(np.mean(window_dvrs))

            avg_records_100.append(
                {
                    "episode_end": i + 1,          # 截止到第几轮（100, 200, 300...）
                    "avg_reward_100": avg_reward_100,
                    "avg_dvr_100": avg_dvr_100,
                }
            )

            # 确保目录存在
            if not os.path.exists(base_dir):
                os.makedirs(base_dir)

            # 覆盖写入 Excel：内容始终是「到目前为止每 100 轮的平均值」
            df = pd.DataFrame(avg_records_100)
            df.to_excel(excel_path, index=False)

            print(
                f"[AVG-100] 截止 episode {i+1}："
                f"最近100轮平均 reward={avg_reward_100:.3f}, "
                f"平均 dvr={avg_dvr_100:.4f}"
            )

        # 定期更新图（保持你原来的逻辑）
        if i > 0 and i % 500 == 0 and len(agent.buffer.IDs) > 0:
            env.update_adjs(set(agent.buffer.IDs))

        # 定期保存模型和日志（保持你原来的逻辑）
        if i % 500 == 0:
            # 模型按 episode 分子目录存
            model_dir = os.path.join(base_dir, str(i))
            agent.save_models(model_dir)

            # dvr / reward 曲线放在 base_dir 下
            save(base_dir, "dvr.txt", dvr_list)
            save(base_dir, "ep_reward.txt", reward_list)


if __name__ == "__main__":
    train()
