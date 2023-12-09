import os
import time
from copy import deepcopy
import numpy as np
import torch
import collections
import logging
import tqdm
from omegaconf import OmegaConf

from diffusion_policy.policy.base_lowdim_policy import BaseLowdimPolicy
from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.env_runner.base_lowdim_runner import BaseLowdimRunner

from diffusion_policy.env.raisim.raisim_env import RaisimEnv

module_logger = logging.getLogger(__name__)

class ObservationBuffer:
    def __init__(self, n_obs_steps, n_envs, obs_dim, device):
        self.n_obs_steps = n_obs_steps
        self.n_envs = n_envs
        self.obs_shape = obs_dim
        self.obs_deque = [collections.deque([np.zeros(obs_dim)] * self.n_obs_steps, maxlen=self.n_obs_steps) for i in range(self.n_envs)]
        self.device = device
    
    def append(self, obs):
        for i in range(self.n_envs):
            self.obs_deque[i].append(deepcopy(obs[i]))
    
    def get_obs_seq(self):
        obs_seq = np.stack(self.obs_deque).astype(np.float32)
        obs_seq = {'obs': torch.from_numpy(obs_seq).to(self.device)}
        return obs_seq
    
    def reset(self, obs, dones):
        for i in range(self.n_envs):
            if dones[i]:
                self.obs_deque[i] = collections.deque([deepcopy(obs[i])] * self.n_obs_steps, maxlen=self.n_obs_steps)


class RaisimRunner(BaseLowdimRunner):
    def __init__(self,
            output_dir,
            env_cfg,
            obs_dim,
            max_steps,
            n_obs_steps,
            n_action_steps,
            dataset=None,
            tqdm_interval_sec=1.0,
        ):
        super().__init__(output_dir)

        resource_dir = os.path.dirname(os.path.realpath(__file__)) + "/../env/raisim/resources"
        env_cfg = OmegaConf.to_yaml(env_cfg)
        self.env = RaisimEnv(resource_dir, env_cfg)

        self.n_obs_steps = n_obs_steps
        self.obs_dim = obs_dim
        self.n_action_steps = n_action_steps
        self.max_steps = max_steps
        self.n_envs = self.env.num_envs
        self.dataset = dataset
        self.tqdm_interval_sec = tqdm_interval_sec


    def run(self, policy: BaseLowdimPolicy, eval_run=False):
        if eval_run:
            self.max_steps = 10000
        device = policy.device
        env = self.env

        # init
        ep_rewards = np.zeros((self.max_steps, self.n_envs))
        self.obs_buffer = ObservationBuffer(
            self.n_obs_steps, self.n_envs, self.obs_dim, device)
        done = False
        step_idx = 0
        
        # start rollout
        obs = env.reset()
        self.obs_buffer.reset(obs, [True] * self.n_envs)
        policy.reset()

        pbar = tqdm.tqdm(total=self.max_steps, desc=f"Eval RaisimRunner", 
            leave=False, mininterval=self.tqdm_interval_sec)

        while not done:
            obs_seq = self.obs_buffer.get_obs_seq()

            # run policy
            start = time.time()
            with torch.no_grad():
                action_dict = policy.predict_action(obs_seq)
            # print(f"policy time: {time.time() - start}")

            # device_transfer
            np_action_dict = dict_apply(action_dict,
                lambda x: x.detach().to('cpu').numpy())

            action = np_action_dict['action']
            action = np.transpose(action, (1, 0, 2))

            # step env
            for i in range(len(action)):
                # time.sleep(0.01)
                obs, reward, dones = env.step(action[i])
                self.obs_buffer.append(obs)
                ep_rewards[step_idx] += reward
                step_idx += 1

                # update obs deque
                self.obs_buffer.reset(obs, dones)

            
            if step_idx >= self.max_steps:
                done = True
            
            # if step_idx % 100 == 0:
            #     obs = env.reset()
            #     self.obs_buffer.reset(obs, [True] * self.n_envs)

            pbar.update(len(action))
        pbar.close()

        #TODO: split ep_rewards by dones

        # log
        log_data = {
            'test_mean_score': ep_rewards.sum(axis=0).mean(),
        }

        return log_data
