"""表格型 RL 算法（第 2 章 2.02~2.07），全部基于 vla_ladder.envs.GridWorld 的模型/交互接口。"""

from __future__ import annotations

import numpy as np

from vla_ladder.envs.gridworld import policy_to_matrix


def policy_model(env, pi):
    """策略矩阵 pi (S, A) -> r_pi (S,), P_pi (S, S)"""
    pi = policy_to_matrix(pi, env.n_actions)
    return (pi * env.R).sum(1), np.einsum("sa,sat->st", pi, env.P)


def evaluate_closed_form(env, pi, gamma):
    """贝尔曼方程闭式解 v = (I - γP_pi)^{-1} r_pi"""
    r_pi, P_pi = policy_model(env, pi)
    return np.linalg.solve(np.eye(env.n_states) - gamma * P_pi, r_pi)


def evaluate_iterative(env, pi, gamma, tol=1e-8, max_iter=10000):
    r_pi, P_pi = policy_model(env, pi)
    v = np.zeros(env.n_states)
    for k in range(max_iter):
        v_new = r_pi + gamma * P_pi @ v
        if np.max(np.abs(v_new - v)) < tol:
            return v_new, k + 1
        v = v_new
    return v, max_iter


def q_from_v(env, v, gamma):
    return env.R + gamma * env.P @ v


def value_iteration(env, gamma, tol=1e-6, max_iter=10000):
    v = np.zeros(env.n_states)
    for _ in range(max_iter):
        q = q_from_v(env, v, gamma)
        v_new = q.max(1)
        if np.max(np.abs(v_new - v)) < tol:
            return v_new, q.argmax(1)
        v = v_new
    return v, q.argmax(1)


def policy_iteration(env, gamma, max_iter=100):
    pi = np.zeros(env.n_states, dtype=int)
    for _ in range(max_iter):
        v = evaluate_closed_form(env, pi, gamma)
        pi_new = q_from_v(env, v, gamma).argmax(1)
        if np.array_equal(pi_new, pi):
            break
        pi = pi_new
    return v, pi


def epsilon_greedy(q, eps):
    S, A = q.shape
    pi = np.full((S, A), eps / A)
    pi[np.arange(S), q.argmax(1)] += 1 - eps
    return pi


def mc_epsilon_greedy(env, gamma, eps, n_episodes=1000, length=200, seed=0):
    rng = np.random.default_rng(seed)
    q = np.zeros((env.n_states, env.n_actions))
    ret_sum, ret_cnt = np.zeros_like(q), np.zeros_like(q)
    for _ in range(n_episodes):
        pi = epsilon_greedy(q, eps)
        s = env.reset(int(rng.integers(env.n_states)))
        traj = []
        for _ in range(length):
            a = rng.choice(env.n_actions, p=pi[s])
            ns, r = env.step(a)
            traj.append((s, a, r))
            s = ns
        G = 0.0
        for s_t, a_t, r_t in reversed(traj):
            G = gamma * G + r_t
            ret_sum[s_t, a_t] += G
            ret_cnt[s_t, a_t] += 1
            q[s_t, a_t] = ret_sum[s_t, a_t] / ret_cnt[s_t, a_t]
    return q


def td_control(env, gamma, method="qlearning", alpha=0.1, eps=0.1, n_episodes=500, length=100, seed=0):
    """Sarsa / Q-learning（ε-greedy 行为策略）。返回 (q, 每回合总奖励)"""
    rng = np.random.default_rng(seed)
    q = np.zeros((env.n_states, env.n_actions))

    def act(row):
        return int(rng.integers(len(row))) if rng.random() < eps else int(np.argmax(row))

    rewards = []
    for _ in range(n_episodes):
        s = env.reset(int(rng.integers(env.n_states)))
        a, total = act(q[s]), 0.0
        for _ in range(length):
            ns, r = env.step(a)
            na = act(q[ns])
            target = r + gamma * (q[ns, na] if method == "sarsa" else q[ns].max())
            q[s, a] -= alpha * (q[s, a] - target)
            total += r
            s, a = ns, na
        rewards.append(total)
    return q, rewards
