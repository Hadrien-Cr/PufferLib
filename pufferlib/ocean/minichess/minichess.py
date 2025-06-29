import numpy as np
import gymnasium

import pufferlib
from pufferlib.ocean.minichess import binding


class MiniChess(pufferlib.PufferEnv):
    def __init__(self, num_envs=1, render_mode=None, report_interval=128,
             buf=None, seed=0):

        self.single_observation_space = gymnasium.spaces.Box(low=0, high=1,
            shape=(5*5 + 6,), dtype=np.float32)
        self.single_action_space = gymnasium.spaces.Discrete(1225)
        self.report_interval = report_interval
        self.render_mode = render_mode
        self.num_agents = num_envs

        super().__init__(buf=buf)
        self.c_envs = binding.vec_init(self.observations, self.actions, self.rewards,
            self.terminals, self.truncations, num_envs, seed)

    def reset(self, seed=None):
        self.tick = 0
        if seed is None:
            binding.vec_reset(self.c_envs, 0)
        else:
            binding.vec_reset(self.c_envs, seed)
        return self.observations, []

    def step(self, actions):
        self.actions[:] = actions
        binding.vec_step(self.c_envs)
        self.tick += 1

        info = []
        # if self.tick % self.report_interval == 0:
        #     log = binding.vec_log(self.c_envs)
        #     if log['episode_length'] > 0:
        #         info.append(log)

        return (self.observations, self.rewards,
            self.terminals, self.truncations, info)

    def render(self):
        binding.vec_render(self.c_envs, 0)

    def close(self):
        binding.vec_close(self.c_envs)


if __name__ == '__main__':
    import time
    num_envs = 128
    atn_cache = 1000
    timeout = 10

    env = MiniChess(num_envs=num_envs)
    env.reset()
    tick = 0
    actions = np.random.randint(
        0,
        env.single_action_space.n + 1,
        (atn_cache, num_envs),
    )

    start = time.time()
    while time.time() - start < timeout:
        atn = actions[tick % atn_cache]         
        env.step(atn)
        tick += 1
        # env.render()

    print(f'SPS: {num_envs * tick / (time.time() - start)}')

