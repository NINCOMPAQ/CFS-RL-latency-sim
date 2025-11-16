import simpy
import random
import numpy as np
import statistics
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO

RANDOM_SEED = 42
max_queue = 128


# =====================================================
# JOB + ARRIVALS
# =====================================================

class Job:
    def __init__(self, id, arrival_time, service_time, weight=1.0):
        self.id = id
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
            j = Job(jid, env.now, st)
            queue.append(j)
        jid += 1


# =====================================================
# BASELINE CFS SELECTION
# =====================================================

def cfs_select(queue):
    return min(range(len(queue)), key=lambda i: queue[i].vruntime)


# =====================================================
# LATENCY-BOOSTED HEURISTIC
# =====================================================

LATENCY_BOOST_THRESHOLD = 1.0
LATENCY_BOOST_FACTOR = 2.0


def latency_boost_select(queue, threshold=LATENCY_BOOST_THRESHOLD, boost=LATENCY_BOOST_FACTOR):
    if not queue:
        return None

    def effective_vruntime(job):
        weight = job.weight
        if job.remaining <= threshold:
            weight *= boost
        return job.vruntime / weight

    return min(range(len(queue)), key=lambda i: effective_vruntime(queue[i]))


# =====================================================
# BASELINE SIMULATION (NO RL)
# =====================================================

def server(env, queue, results, time_slice=0.05):
    while True:
        if not queue:
            yield env.timeout(1e-3)
            continue

        i = cfs_select(queue)
        job = queue[i]
        if job.start is None:
            job.start = env.now

        delta = min(time_slice, job.remaining)
        yield env.timeout(delta)

        job.remaining -= delta
        job.vruntime += delta / job.weight

        if job.remaining <= 0:
            job.finish = env.now
            queue.pop(i)
            results.append(job)


def run_sim(arrival_rate=0.9, service_rate=1.0, sim_time=10000,
            policy='cfs', max_jobs=None):
    random.seed(RANDOM_SEED)
    env = simpy.Environment()
    queue = []
    results = []

    env.process(job_generator(env, arrival_rate, service_rate, queue, max_jobs))
    env.process(server(env, queue, results))
    env.run(until=sim_time)

    latencies = [(j.finish - j.arrival) for j in results if j.finish is not None]
    if not latencies:
        return {}

    avg = statistics.mean(latencies)
    p95 = float(np.percentile(latencies, 95))
    std = float(np.std(latencies))
    lfi = avg / (avg + std) if (avg + std) > 0 else 1.0
    throughput = len(results) / sim_time

    return {
        'policy': policy,
        'avg_latency': avg,
        'p95': p95,
        'throughput': throughput,
        'completed': len(results),
        'lfi': lfi
    }


# =====================================================
# RL ENVIRONMENT CORE (4-D STATE)
# =====================================================

class CFSEnv:
    """
    Core simulator used by the RL wrapper.
    State (4-D):
        [queue_len, mean_remaining, var_vruntime, avg_wait]
    """
    def __init__(self, arrival_rate=0.9, service_rate=1.0, sim_time=10000,
                 alpha=0.6, beta=0.3, gamma=0.1):
        self.arrival_rate = arrival_rate
        self.service_rate = service_rate
        self.sim_time = sim_time
        self.time_slice = 0.1

        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

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

    def _get_state(self):
        if not self.queue:
            return np.zeros(4, dtype=np.float32)

        rems = np.array([j.remaining for j in self.queue], dtype=np.float32)
        vrs = np.array([j.vruntime for j in self.queue], dtype=np.float32)
        waits = np.array([self.env.now - j.arrival for j in self.queue], dtype=np.float32)

        return np.array([
            float(len(self.queue)),
            float(rems.mean()),
            float(vrs.var()),
            float(waits.mean())
        ], dtype=np.float32)

    def step(self, action_idx):
        """
        Take one scheduling decision.
        Returns: (obs, reward, terminated)
        """
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

        terminated = (self.env.now >= self.sim_time)

        if job.remaining <= 0:
            job.finish = self.env.now
            self.results.append(job)
            self.queue.remove(job)

        reward = self._compute_reward()
        return self._get_state(), float(reward), terminated

    def _compute_reward(self):
        """
        Latency + fairness reward (same structure as your original).
        """
        if not self.results:
            return 0.0

        lat = np.array([j.finish - j.arrival for j in self.results], dtype=np.float32)
        avg = float(lat.mean())
        p95 = float(np.percentile(lat, 95))
        std = float(np.std(lat))
        lfi = avg / (avg + std) if (avg + std) > 0 else 1.0

        # Same form as your original reward
        return -(self.alpha * avg / 10.0 + self.beta * p95 / 30.0) + self.gamma * (lfi * 5.0)


# =====================================================
# GYMNASIUM WRAPPER (SB3 COMPATIBLE)
# =====================================================

class CFSEnvGym(gym.Env):
    """
    Gymnasium-compatible wrapper around CFSEnv.

    reset() -> (obs, info)
    step()  -> (obs, reward, terminated, truncated, info)
    """
    metadata = {"render_modes": []}

    def __init__(self, arrival_rate=0.9, service_rate=1.0, sim_time=10000,
                 max_queue=64, alpha=0.6, beta=0.3, gamma=0.1):
        super().__init__()

        self.core = CFSEnv(arrival_rate, service_rate, sim_time, alpha, beta, gamma)
        self.max_queue = max_queue

        self.action_space = spaces.Discrete(max_queue)
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(4,), dtype=np.float32
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

        # Multi-step skip to accumulate reward
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

        return (
            obs.astype(np.float32),
            float(total_reward),
            terminated,
            truncated,
            {}
        )


# =====================================================
# MAIN
# =====================================================

if __name__ == "__main__":

    # ---------------- Baseline CFS ----------------
    print("=== Baseline CFS ===")
    base = run_sim()
    print(base)

    # ------------- Latency-Boost Heuristic --------
    print("\n=== CFS with Latency Boost Heuristic ===")
    heuristics = [
        {"threshold": 0.5, "boost": 2.0},
        {"threshold": 1.0, "boost": 2.0},
        {"threshold": 1.0, "boost": 3.0},
    ]

    for h in heuristics:
        sim_time = 10000
        random.seed(RANDOM_SEED)
        env = simpy.Environment()
        queue, results = [], []

        env.process(job_generator(env, 0.9, 1.0, queue))

        def server_boost(env, queue, results, time_slice=0.05, threshold=1.0, boost=2.0):
            while True:
                if not queue:
                    yield env.timeout(1e-3)
                    continue
                i = latency_boost_select(queue, threshold, boost)
                job = queue[i]
                if job.start is None:
                    job.start = env.now
                delta = min(time_slice, job.remaining)
                yield env.timeout(delta)
                job.remaining -= delta
                job.vruntime += delta / job.weight
                if job.remaining <= 0:
                    job.finish = env.now
                    queue.pop(i)
                    results.append(job)

        env.process(server_boost(env, queue, results,
                                 threshold=h["threshold"], boost=h["boost"]))
        env.run(until=sim_time)

        lat = [j.finish - j.arrival for j in results]
        avg = float(np.mean(lat))
        p95 = float(np.percentile(lat, 95))
        throughput = len(results) / sim_time
        std = float(np.std(lat))
        lfi = avg / (avg + std) if (avg + std) > 0 else 1.0

        print(
            f"T={h['threshold']} B={h['boost']} | "
            f"avg={avg:.3f}, p95={p95:.3f}, th={throughput:.3f}, "
            f"completed={len(results)}, lfi={lfi:.3f}"
        )

    # ------------- PPO RL TRAINING ----------------
    reward_configs = [
        {'alpha': 0.6, 'beta': 0.3, 'gamma': 0.1},
        {'alpha': 0.7, 'beta': 0.2, 'gamma': 0.1},
        {'alpha': 0.5, 'beta': 0.4, 'gamma': 0.1},
        {'alpha': 0.6, 'beta': 0.3, 'gamma': 0.2},
        {'alpha': 0.4, 'beta': 0.5, 'gamma': 0.1},
        {'alpha': 0.5, 'beta': 0.3, 'gamma': 0.2},
    ]

    for rc in reward_configs:
        print(f"\n=== PPO training α={rc['alpha']} β={rc['beta']} γ={rc['gamma']} ===")

        env_gym = CFSEnvGym(
            alpha=rc['alpha'], beta=rc['beta'], gamma=rc['gamma']
        )

        model = PPO(
            "MlpPolicy",
            env_gym,
            verbose=0,
            learning_rate=1e-4,
            gamma=0.995,
            n_steps=2048,
            batch_size=128,
            clip_range=0.2,
            seed=RANDOM_SEED
        )

        model.learn(total_timesteps=500_000)

        # ---- Evaluate this config ----
        eval_env = CFSEnvGym(
            alpha=rc['alpha'], beta=rc['beta'], gamma=rc['gamma']
        )
        obs, info = eval_env.reset()
        terminated = False
        truncated = False

        while not (terminated or truncated):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = eval_env.step(action)

        latencies = [j.finish - j.arrival for j in eval_env.core.results]
        avg = float(np.mean(latencies))
        p95 = float(np.percentile(latencies, 95))
        throughput = len(eval_env.core.results) / eval_env.core.sim_time
        std = float(np.std(latencies))
        lfi = avg / (avg + std) if (avg + std) > 0 else 1.0

        print({
            'policy': f"PPO_{rc}",
            'avg': avg,
            'p95': p95,
            'throughput': throughput,
            'completed': len(eval_env.core.results),
            'lfi': lfi
        })
