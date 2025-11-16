import simpy
import random
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO

RANDOM_SEED = 42
max_queue = 128


# ======================================================
# Job + arrivals
# ======================================================

class Job:
    def __init__(self, jid, arrival_time, service_time, weight=1.0):
        self.id = jid
        self.arrival = arrival_time
        self.service = service_time
        self.remaining = service_time
        self.weight = weight
        self.vruntime = 0.0
        self.start = None
        self.finish = None


def job_generator(env, arrival_rate, service_rate, queue, max_jobs=None):
    jid = 0
    while max_jobs is None or jid < max_jobs:
        inter = random.expovariate(arrival_rate)
        yield env.timeout(inter)
        st = random.expovariate(service_rate)
        if len(queue) < max_queue:
            queue.append(Job(jid, env.now, st))
        jid += 1


# ======================================================
# Core CFS Environment (10-D state)
# ======================================================

class CFSEnv:
    """
    Rich 10-D latency-focused state:
        0: queue_len
        1: mean_remaining
        2: var_vruntime
        3: mean_wait
        4: max_remaining
        5: min_remaining
        6: max_wait
        7: std_remaining
        8: std_wait
        9: entropy_vruntime
    """

    def __init__(self, arrival_rate=0.9, service_rate=1.0,
                 sim_time=10000, alpha=0.6, beta=0.4):
        self.arrival_rate = arrival_rate
        self.service_rate = service_rate
        self.sim_time = sim_time
        self.alpha = alpha
        self.beta = beta

        self.time_slice = 0.1
        self.state_dim = 10
        self.reset()

    def reset(self):
        random.seed(RANDOM_SEED)
        self.env = simpy.Environment()
        self.queue = []
        self.results = []
        self.env.process(job_generator(
            self.env, self.arrival_rate, self.service_rate, self.queue
        ))
        return self._get_state()

    def _entropy(self, x):
        """Compute entropy of positive float array."""
        if len(x) == 0:
            return 0.0
        x = x - x.min()
        total = x.sum()
        if total <= 0:
            return 0.0
        p = x / total
        p = p[p > 0]
        return float(-(p * np.log(p)).sum())

    def _get_state(self):
        if not self.queue:
            return np.zeros(self.state_dim, dtype=np.float32)

        rem = np.array([j.remaining for j in self.queue], dtype=np.float32)
        vr  = np.array([j.vruntime for j in self.queue], dtype=np.float32)
        waits = np.array([self.env.now - j.arrival for j in self.queue], dtype=np.float32)

        # 10-D feature vector
        state = np.array([
            float(len(self.queue)),     # queue_len
            float(rem.mean()),          # mean_remaining
            float(vr.var()),            # var_vruntime
            float(waits.mean()),        # mean_wait
            float(rem.max()),           # max_remaining
            float(rem.min()),           # min_remaining
            float(waits.max()),         # max_wait
            float(rem.std()),           # std_remaining
            float(waits.std()),         # std_wait
            self._entropy(vr),          # entropy_vruntime
        ], dtype=np.float32)

        return state

    def step(self, action_idx):
        if not self.queue:
            self.env.run(until=self.env.now + 1e-3)
            return self._get_state(), 0.0, False

        action_idx = max(0, min(int(action_idx), len(self.queue) - 1))
        job = self.queue[action_idx]

        if job.start is None:
            job.start = self.env.now

        delta = min(self.time_slice, job.remaining)
        self.env.run(until=self.env.now + delta)
        job.remaining -= delta
        job.vruntime += delta / job.weight

        terminated = (self.env.now >= self.sim_time)

        if job.remaining <= 0:
            job.finish = self.env.now
            self.results.append(job)
            self.queue.remove(job)

        reward = self._compute_reward()
        return self._get_state(), reward, terminated

    def _compute_reward(self):
        if not self.results:
            return 0.0

        lat = np.array([j.finish - j.arrival for j in self.results], dtype=np.float32)
        avg = float(lat.mean())
        p95 = float(np.percentile(lat, 95))
        return -(self.alpha*(avg/10.0) + self.beta*(p95/30.0))


# ======================================================
# Gymnasium wrapper
# ======================================================

class CFSEnvGym(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, alpha=0.6, beta=0.4):
        super().__init__()
        self.core = CFSEnv(alpha=alpha, beta=beta)
        self.action_space = spaces.Discrete(max_queue)
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(10,), dtype=np.float32
        )
        self.reward_skip = 10
        self.max_steps = 20000
        self.steps = 0

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        obs = self.core.reset()
        self.steps = 0
        return obs.astype(np.float32), {}

    def step(self, action):
        self.steps += 1
        total_reward = 0.0
        terminated = False

        for _ in range(self.reward_skip):
            if len(self.core.queue) == 0:
                self.core.env.run(until=self.core.env.now + 1e-3)
                obs = self.core._get_state()
                reward = 0.0
            else:
                act = min(int(action), len(self.core.queue) - 1)
                obs, reward, terminated = self.core.step(act)

            total_reward += reward
            if terminated:
                break

        truncated = (self.steps >= self.max_steps)

        return (obs.astype(np.float32),
                float(total_reward),
                terminated,
                truncated,
                {})


# ======================================================
# Training
# ======================================================

if __name__ == "__main__":
    print("=== PPO RL (rich 10-D state) ===")

    env = CFSEnvGym()

    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        learning_rate=1e-4,
        gamma=0.995,
        n_steps=2048,
        batch_size=128,
        clip_range=0.2,
        seed=RANDOM_SEED
    )

    model.learn(total_timesteps=300_000)

    # ========= Evaluation =========
    eval_env = CFSEnvGym()
    obs, info = eval_env.reset()
    terminated = False
    truncated = False

    while not (terminated or truncated):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = eval_env.step(action)

    lat = np.array(
        [j.finish - j.arrival for j in eval_env.core.results],
        dtype=np.float32
    )

    print("avg =", float(lat.mean()))
    print("p95 =", float(np.percentile(lat, 95)))
    print("completed =", len(eval_env.core.results))
