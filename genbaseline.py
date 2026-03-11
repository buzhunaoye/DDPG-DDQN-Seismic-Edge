import os
import sys
import time
import numpy as np
import pandas as pd

from env.environment import Environment
from modules.agents.baselines import LocalAgent, EdgeAgent, CloudAgent, RandomAgent, GreedyAgent


class TeeLogger:
    def __init__(self, filepath: str):
        self.filepath = filepath
        self._stdout = sys.stdout
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        self._f = open(filepath, "w", encoding="utf-8")

    def write(self, msg):
        self._stdout.write(msg)
        self._f.write(msg)

    def flush(self):
        self._stdout.flush()
        self._f.flush()

    def close(self):
        self._f.close()


def make_agent(name: str):
    name = name.lower()
    if name == "local":
        return LocalAgent()
    if name == "edge":
        return EdgeAgent()
    if name == "cloud":
        return CloudAgent()
    if name == "random":
        return RandomAgent()
    if name == "greedy":
        return GreedyAgent()
    raise ValueError(f"Unknown baseline algorithm: {name}")


def run_one_episode(algo: str, seed: int):
    env = Environment()
    agent = make_agent(algo)

    state = env.reset(seed)
    ep_reward = 0.0

    while not env.done:
        avail_action = env.get_avail_actions()
        action = agent.choose_action(state, avail_action, evaluate=True)
        state, reward, done = env.step(action)
        ep_reward += float(reward)

    dvr_rate = float(env.get_metric())
    return dvr_rate, float(ep_reward)


def eval_mean(algo: str, seeds):
    dvr_list, rew_list = [], []
    for s in seeds:
        dvr, rew = run_one_episode(algo, int(s))
        dvr_list.append(dvr)
        rew_list.append(rew)
    return float(np.mean(dvr_list)), float(np.mean(rew_list))


def generate_baseline_50pts(
    algorithms=("greedy", "random", "local", "edge", "cloud"),
    total_rounds=5000,
    window=100,
    episodes_per_point=100,
    base_seed=None,
    out_dir="outputs",
):
    assert total_rounds % window == 0
    points = total_rounds // window

    if base_seed is None:
        base_seed = int(time.time())

    os.makedirs(out_dir, exist_ok=True)

    rows = []
    for algo in algorithms:
        algo_hash = abs(hash(algo)) % 10000
        for i in range(1, points + 1):
            step = i * window

            # ✅ 修复：保证 seed 永远在 [0, 2**32 - 1]
            seeds = [
                int(np.uint32(base_seed + algo_hash * 1_000_000 + step * 10_000 + j))
                for j in range(episodes_per_point)
            ]

            dvr_mean, rew_mean = eval_mean(algo, seeds)

            rows.append()
