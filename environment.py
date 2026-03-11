import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import networkx as nx
import random


class Constant:
    c_max = 5e10
    r_max = 8e6
    s_max = 5e10


def normalize(data):
    assert isinstance(data, np.ndarray)
    if np.max(data) == np.min(data):
        return np.zeros_like(data)
    return (data - np.min(data)) / (np.max(data) - np.min(data))


class Environment:
    def __init__(self):
        self.ID = 0
        self.adjs = {}
        self.min_num_nodes = 100
        self.max_num_nodes = 100
        self.min_num_edges = 250
        self.max_num_edges = 250
        self.M = 5

    # ----------- 生成 DAG（保持你原来的逻辑） -----------
    def generate_dag(self, num_nodes, num_edges):
        G = nx.DiGraph()
        G.add_nodes_from(range(num_nodes))
        while G.number_of_edges() < num_edges:
            a, b = np.random.randint(0, num_nodes, size=2)
            if a != b and not G.has_edge(a, b):
                G.add_edge(a, b)
                if not nx.is_directed_acyclic_graph(G):
                    G.remove_edge(a, b)
        return G

    def generate_tolerance(self, G, queue):
        lower, upper = 1, 6
        mu, sigma = 4, 1
        X = stats.truncnorm((lower - mu) / sigma, (upper - mu) / sigma, loc=mu, scale=sigma)

        tolerance = np.zeros(len(queue))
        for task_idx in queue:
            tolerance[task_idx] = X.rvs()
            deps = list(G.predecessors(task_idx))
            if len(deps) > 0:
                tolerance[task_idx] += max([tolerance[dep] for dep in deps])

        sorted_tolerance = np.array([tolerance[task_idx] for task_idx in queue])
        return sorted_tolerance

    def generate_task(self):
        self.G = self.generate_dag(self.num_nodes, self.num_edges)
        self.queue = np.array(list(nx.topological_sort(self.G)))

        self.data_size = np.random.uniform(8e5, 1.6e6, self.num_nodes)
        self.cpu_cycles = np.random.uniform(2e8, 2e9, self.num_nodes)
        self.tolerance = self.generate_tolerance(self.G, self.queue)

    def init_cluster(self):
        self.local_cpu_cycles = np.random.uniform(1e8, 2e8)
        self.local_storage = np.random.uniform(1e8, 2e8)

        self.edge_cpu_cycles = np.zeros(self.M)
        self.edge_trans_rate = np.zeros(self.M)
        self.edge_storage = np.zeros(self.M)

        delta_edge_cpu_cycles = (2e9 - 1e9) / self.M
        delta_edge_trans_rate = (4e6 - 2e6) / self.M
        delta_edge_storage = (2e9 - 1e9) / self.M

        for i in range(self.M):
            self.edge_cpu_cycles[i] = np.random.uniform(1e9 + i * delta_edge_cpu_cycles,
                                                        1e9 + (i + 1) * delta_edge_cpu_cycles)
            self.edge_trans_rate[i] = np.random.uniform(1e6 + i * delta_edge_trans_rate,
                                                        1e6 + (i + 1) * delta_edge_trans_rate)
            self.edge_storage[i] = np.random.uniform(1e9 + i * delta_edge_storage,
                                                     1e9 + (i + 1) * delta_edge_storage)

        self.cloud_trans_rate = np.random.uniform(2.4e6, 4.8e6)
        self.cloud_fixed_time = 3

    def reset(self, seed=0):
        random.seed(seed)
        np.random.seed(seed)

        self.ID += 1
        self.num_nodes = random.choice(range(self.min_num_nodes, self.max_num_nodes + 5, 5))
        self.num_edges = random.choice(range(self.min_num_edges, self.max_num_edges + 10, 10))

        self.generate_task()
        self.init_cluster()
        return self.get_state()

    def get_state(self):
        task_idx = np.zeros(self.max_num_nodes)
        task_info_padding = np.stack([
            np.pad(normalize(self.data_size), (0, self.max_num_nodes - self.num_nodes)),
            np.pad(normalize(self.cpu_cycles), (0, self.max_num_nodes - self.num_nodes)),
            np.pad(normalize(self.tolerance), (0, self.max_num_nodes - self.num_nodes))
        ], axis=1)

        dev_cpu_cycles = np.append(np.append(self.local_cpu_cycles, self.edge_cpu_cycles), Constant.c_max)
        dev_trans_rate = np.append(np.append(Constant.r_max, self.edge_trans_rate), self.cloud_trans_rate)
        dev_storage = np.append(np.append(self.local_storage, self.edge_storage), Constant.s_max)
        dev_info = np.stack([normalize(dev_cpu_cycles), normalize(dev_trans_rate), normalize(dev_storage)], axis=1)

        return (task_idx, task_info_padding, dev_info)

    # ----------------- 关键：计算 DAG 分层 depth（真正分层） -----------------
    def _dag_depth(self, G):
        topo = list(nx.topological_sort(G))
        depth = {v: 0 for v in topo}
        for v in topo:
            preds = list(G.predecessors(v))
            if preds:
                depth[v] = max(depth[p] + 1 for p in preds)
        return depth

    # ----------------- 选局部子图：保证跨多个层，避免全挤一条线 -----------------
    def _select_subdag_nodes_layered(self, G, k=30, seed=0):
        rng = np.random.default_rng(seed)
        depth = self._dag_depth(G)

        # 按层分桶
        buckets = {}
        for v, d in depth.items():
            buckets.setdefault(d, []).append(v)

        layers = sorted(buckets.keys())
        if not layers:
            return list(G.nodes)[:k]

        # 从多个层均匀采样（保证“像 DAG”）
        selected = []
        # 每层抽多少（至少覆盖尽可能多的层）
        per_layer = max(1, k // max(1, min(len(layers), 8)))  # 最多取 8 层，避免太分散
        for d in layers:
            rng.shuffle(buckets[d])
            selected.extend(buckets[d][:per_layer])
            if len(selected) >= k:
                break

        # 不够就从剩余里补齐
        if len(selected) < k:
            remaining = list(set(G.nodes) - set(selected))
            rng.shuffle(remaining)
            selected.extend(remaining[: (k - len(selected))])

        return selected[:k]

    # ----------------- 画依赖图 + 无依赖散点图（同一批节点） -----------------
    def plot_dependency_and_nodependency(self, k=30, seed=0,
                                        save_dep="dag_dep.png",
                                        save_nodep="dag_nodep.png"):
        G = self.G

        # ① 选同一批节点（跨层抽样，更像 DAG）
        nodes = self._select_subdag_nodes_layered(G, k=k, seed=seed)

        # ② 依赖子图
        H_dep = G.subgraph(nodes).copy()

        # ③ 传递约简：减少“多余边”，更清爽、更像论文示意图
        #    注意：transitive_reduction 只适用于 DAG
        if nx.is_directed_acyclic_graph(H_dep) and H_dep.number_of_edges() > 0:
            try:
                H_dep_draw = nx.transitive_reduction(H_dep)
            except Exception:
                H_dep_draw = H_dep
        else:
            H_dep_draw = H_dep

        # ④ 用 depth 分层布局（稳定、论文友好）
        depth_sub = self._dag_depth(H_dep_draw)
        for v in H_dep_draw.nodes:
            H_dep_draw.nodes[v]["layer"] = depth_sub.get(v, 0)

        pos_dep = nx.multipartite_layout(H_dep_draw, subset_key="layer", align="vertical", scale=2.0)

        # ---------- 画依赖 DAG ----------
        plt.figure(figsize=(11, 6), dpi=260)
        nx.draw_networkx_nodes(H_dep_draw, pos_dep, node_size=520, alpha=0.95)
        nx.draw_networkx_edges(
            H_dep_draw, pos_dep,
            arrows=True,
            arrowstyle='-|>',
            arrowsize=14,
            width=1.1,
            alpha=0.70,
            connectionstyle="arc3,rad=0.08"
        )
        nx.draw_networkx_labels(H_dep_draw, pos_dep, font_size=8)

        plt.title(f"Dependency-aware DAG (subgraph) |V|={H_dep_draw.number_of_nodes()}, |E|={H_dep_draw.number_of_edges()}",
                  fontsize=11)
        plt.axis("off")
        plt.tight_layout()
        plt.savefig(save_dep, dpi=300, bbox_inches="tight")
        print(f"[OK] Saved dependency DAG to: {save_dep}")
        try:
            plt.show(block=True)
        except Exception:
            pass
        plt.close()

        # ---------- 无依赖：同节点，散点布局 ----------
        H_nodep = nx.DiGraph()
        H_nodep.add_nodes_from(nodes)

        # 散点：random_layout（固定 seed 可复现）
        pos_nodep = nx.random_layout(H_nodep, seed=seed)  # 0~1 的散点
        # 放大一点，避免堆叠
        pos_nodep = {n: (float(p[0]) * 10.0, float(p[1]) * 10.0) for n, p in pos_nodep.items()}

        plt.figure(figsize=(8, 6), dpi=260)
        nx.draw_networkx_nodes(H_nodep, pos_nodep, node_size=520, alpha=0.95)
        nx.draw_networkx_labels(H_nodep, pos_nodep, font_size=8)
        plt.title(f"Dependency-free tasks (same nodes, no edges) |V|={H_nodep.number_of_nodes()}, |E|=0",
                  fontsize=11)
        plt.axis("off")
        plt.tight_layout()
        plt.savefig(save_nodep, dpi=300, bbox_inches="tight")
        print(f"[OK] Saved no-dependency scatter to: {save_nodep}")
        try:
            plt.show(block=True)
        except Exception:
            pass
        plt.close()


def main():
    env = Environment()
    env.reset(seed=0)

    # ✅ 保存到桌面（把 LENOVO 换成你的 Windows 用户名；你这里就是 LENOVO）
    env.plot_dependency_and_nodependency(
        k=30,
        seed=42,
        save_dep=r"C:\Users\LENOVO\Desktop\dag_dep.png",
        save_nodep=r"C:\Users\LENOVO\Desktop\dag_nodep.png"
    )

if __name__ == "__main__":
    main()

