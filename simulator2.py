import simpy
import random
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecEnvWrapper
import matplotlib.pyplot as plt

RANDOM_SEED = 42

# =====================================================
# JOB + ARRIVAL PROCESS
# =====================================================

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

MAX_QUEUE = 128
LATENCY_BOOST_THRESHOLD = 1.0


def job_generator(env, arrival_rate, service_rate, queue, max_jobs=None):
    """
    Poisson arrivals, exponential service times.
    New jobs are appended to `queue`.
    """
    jid = 0
    while max_jobs is None or jid < max_jobs:
        inter = random.expovariate(arrival_rate)
        yield env.timeout(inter)
        st = random.expovariate(service_rate)
        if len(queue) < MAX_QUEUE:
            queue.append(Job(jid, env.now, st))
        jid += 1


# =====================================================
# CORE SIMPY ENV
# =====================================================

class CFSEnv:
    """
    SimPy-based scheduling environment with different state representations
    and latency-only reward.
    """
    def __init__(self,
                 arrival_rate=0.9,
                 service_rate=1.0,
                 sim_time=10_000,
                 alpha=0.6,
                 beta=0.4,
                 gamma=0.0,
                 state_mode="minimal"):
        
        self.arrival_rate = arrival_rate
        self.service_rate = service_rate
        self.sim_time = sim_time
        self.time_slice = 0.1

        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.state_mode = state_mode

        if state_mode == "minimal":
            self.state_dim = 4
        elif state_mode == "extended":
            self.state_dim = 6
        elif state_mode == "rich":
            self.state_dim = 10
        else:
            raise ValueError("Unknown state_mode")

        self.reset()

    def reset(self):
        random.seed(RANDOM_SEED)
        self.env = simpy.Environment()
        self.queue = []
        self.results = []
        self.env.process(job_generator(self.env,
                                       self.arrival_rate,
                                       self.service_rate,
                                       self.queue))
        return self._get_state()

    def _get_state(self):
        eps = 1e-8
        if not self.queue:
            return np.zeros(self.state_dim, dtype=np.float32)

        rems = np.array([j.remaining for j in self.queue], dtype=np.float32)
        vrs  = np.array([j.vruntime for j in self.queue], dtype=np.float32)
        waits = np.array([self.env.now - j.arrival for j in self.queue], dtype=np.float32)

        if self.state_mode == "minimal":
            state = np.array([
                float(len(self.queue)),
                float(rems.mean()),
                float(vrs.var()),
                float(waits.mean())
            ], dtype=np.float32)

        elif self.state_mode == "extended":
            first = self.queue[0]
            vr_spread = float(vrs.max() - vrs.min())
            norm_rem = float(first.remaining / (rems.max() + eps))
            is_short = 1.0 if first.remaining < LATENCY_BOOST_THRESHOLD else 0.0

            state = np.array([
                float(len(self.queue)),
                float(rems.mean()),
                vr_spread,
                norm_rem,
                float(waits.mean()),
                is_short
            ], dtype=np.float32)
        else:
            vr_spread = float(vrs.max() - vrs.min())
            short_ratio = float((rems < LATENCY_BOOST_THRESHOLD).mean())

            if self.env.now > 0:
                throughput_so_far = float(len(self.results) / (self.env.now + eps))
            else:
                throughput_so_far = 0.0

            state = np.array([
                float(len(self.queue)),
                float(rems.mean()),
                float(rems.var()),
                float(rems.min()),
                float(rems.max()),
                vr_spread,
                float(waits.mean()),
                float(waits.max()),
                short_ratio,
                throughput_so_far
            ], dtype=np.float32)

        return state

    def step(self, action_idx):
        if not self.queue:
            self.env.run(until=self.env.now + 1e-3)
            return self._get_state(), 0.0, False
        
        action_idx = int(action_idx)
        action_idx = max(0, min(action_idx, len(self.queue) - 1))

        job = self.queue[action_idx]
        if job.start is None:
            job.start = self.env.now

        delta = min(self.time_slice, job.remaining)
        self.env.run(until=self.env.now + delta)
        job.remaining -= delta
        job.vruntime += delta / job.weight

        done = (self.env.now >= self.sim_time)
        if job.remaining <= 0:
            job.finish = self.env.now
            self.results.append(job)
            self.queue.remove(job)

        reward = self._compute_reward()
        return self._get_state(), reward, done

    def _compute_reward(self):
        if not self.results:
            return 0.0
        
        latencies = np.array(
            [j.finish - j.arrival for j in self.results],
            dtype=np.float32
        )
        avg = float(latencies.mean())
        p95 = float(np.percentile(latencies, 95))
        reward = -(self.alpha * (avg / 10.0) + self.beta * (p95 / 30.0))
        return reward


# =====================================================
# CRITICAL: OBS CAST WRAPPER (TO PYTHON LISTS)
# =====================================================

class ObsCastWrapper(VecEnvWrapper):
    """
    VecEnv wrapper that guarantees observations are plain
    Python nested lists of floats, not numpy scalars.
    """

    def reset(self):
        obs = self.venv.reset()
        # obs is typically a np.ndarray of shape (n_env, obs_dim)
        obs = np.asarray(obs, dtype=np.float32)
        return obs.tolist()

    def step_wait(self):
        obs, rewards, dones, infos = self.venv.step_wait()
        obs = np.asarray(obs, dtype=np.float32)
        return obs.tolist(), rewards, dones, infos


# =====================================================
# GYM WRAPPER
# =====================================================

class CFSEnvGym(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, **kwargs):
        super().__init__()
        self.core = CFSEnv(**kwargs)
        self.state_dim = self.core.state_dim
        self.action_space = spaces.Discrete(64)
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(self.state_dim,), dtype=np.float32
        )
        self.reward_skip = 10

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        state = self.core.reset().astype(np.float32)
        return state, {}

    def step(self, action):
        action = int(action)
        total_reward = 0.0
        terminated = False

        for _ in range(self.reward_skip):
            state, r, done = self.core.step(action)
            total_reward += r
            if done:
                terminated = True
                break
        
        truncated = False
        return state.astype(np.float32), float(total_reward), terminated, truncated, {}


# =====================================================
# MAIN EXPERIMENT
# =====================================================

if __name__ == "__main__":
    print("=== PPO RL: State Ablation (Latency-only Reward) ===")

    state_modes = ["minimal", "extended", "rich"]
    all_results = []

    for mode in state_modes:
        print(f"\n--- Training PPO with state_mode = '{mode}' ---")

        def make_env():
            return CFSEnvGym(
                arrival_rate=0.9,
                service_rate=1.0,
                sim_time=10_000,
                alpha=0.6,
                beta=0.4,
                gamma=0.0,
                state_mode=mode,
            )

        raw_vec = DummyVecEnv([make_env])
        vec = ObsCastWrapper(raw_vec)  # <- cast obs to list-of-floats

        model = PPO(
            "MlpPolicy", vec,
            verbose=1,
            learning_rate=1e-4,
            gamma=0.995,
            n_steps=2048,
            batch_size=128,
            clip_range=0.2,
            seed=RANDOM_SEED
        )

        model.learn(total_timesteps=50_000)

        # Evaluate
        eval_env = make_env()
        obs, _ = eval_env.reset()
        terminated = False
        truncated = False

        while not (terminated or truncated):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, _ = eval_env.step(action)

        latencies = np.array(
            [j.finish - j.arrival for j in eval_env.core.results],
            dtype=np.float32
        )
        avg = float(latencies.mean())
        p95 = float(np.percentile(latencies, 95))

        result = {
            "state_mode": mode,
            "avg_latency": avg,
            "p95": p95,
        }
        all_results.append(result)
        print("→ RL result:", result)

    # Plot
    modes = [r["state_mode"] for r in all_results]
    avg_lat = [r["avg_latency"] for r in all_results]
    p95_lat = [r["p95"] for r in all_results]

    plt.figure(figsize=(8, 4))
    plt.plot(modes, avg_lat, marker="o", label="Average latency")
    plt.plot(modes, p95_lat, marker="o", label="p95 latency")
    plt.xlabel("State vector mode")
    plt.ylabel("Latency")
    plt.title("Latency vs State Representation (RL)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig("latency_vs_state_mode.png", dpi=300)
    print("\nSaved plot as latency_vs_state_mode.png")
