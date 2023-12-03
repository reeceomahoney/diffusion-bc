import os
import numpy as np
import torch
import collections
import logging
import tqdm
from omegaconf import OmegaConf

from diffusion_policy.policy.base_lowdim_policy import BaseLowdimPolicy
from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.env_runner.base_lowdim_runner import BaseLowdimRunner

from diffusion_policy.env.raisim.raisim_gym.env.RaisimGymVecEnv import RaisimGymVecEnv as VecEnv
from diffusion_policy.env.raisim.raisim_gym.env.bin.anymal_velocity_command import RaisimGymEnv

module_logger = logging.getLogger(__name__)

class RaisimRunner(BaseLowdimRunner):
    def __init__(self,
            output_dir,
            env_cfg,
            max_steps,
            n_obs_steps,
            n_action_steps,
            tqdm_interval_sec=5.0,
        ):
        super().__init__(output_dir)

        env_dir = os.path.dirname(os.path.realpath(__file__)) + "/../env/raisim"
        env_cfg = OmegaConf.to_yaml(env_cfg)
        self.env = VecEnv(RaisimGymEnv(env_dir + "/resources", env_cfg), normalize_ob=False)

        self.n_obs_steps = n_obs_steps
        self.n_action_steps = n_action_steps
        self.max_steps = max_steps
        self.n_envs = self.env.num_envs
        self.tqdm_interval_sec = tqdm_interval_sec


    def run(self, policy: BaseLowdimPolicy, eval_run=False):
        device = policy.device
        env = self.env
        ep_rewards = []

        # start rollout
        obs = env.reset()
        obs_deque = collections.deque([obs] * self.n_obs_steps, maxlen=self.n_obs_steps)
        policy.reset()

        pbar = tqdm.tqdm(total=self.max_steps, desc=f"Eval RaisimRunner", 
            leave=False, mininterval=self.tqdm_interval_sec)
        done = False
        step_idx = 0
        past_action = np.zeros((self.n_envs, 12)).astype(np.float32)
        while not done:
            obs_seq = np.stack(obs_deque)
            obs_seq = np.transpose(obs_seq, (1, 0, 2))
            np_obs_dict = {'obs': obs_seq.astype(np.float32),
                           'past_action': past_action}

            # device transfer
            obs_dict = dict_apply(np_obs_dict, 
                lambda x: torch.from_numpy(x).to(
                    device=device))

            # run policy
            with torch.no_grad():
                action_dict = policy.predict_action(obs_dict)

            # device_transfer
            np_action_dict = dict_apply(action_dict,
                lambda x: x.detach().to('cpu').numpy())

            action = np_action_dict['action']
            action = np.transpose(action, (1, 0, 2))

            # step env
            for i in range(len(action)):
                # time.sleep(0.01)
                obs, reward, dones = env.step(action[i])
                obs_deque.append(obs)
                ep_rewards.append(np.mean(reward))
                past_action = action[i]

                step_idx += 1
                done = np.any(dones)
                if done:
                    break
            
            if step_idx >= self.max_steps:
                done = True
            
            if eval_run:
                done = False

            pbar.update(len(action))
        pbar.close()

        # log
        log_data = {
            'test_mean_score': sum(ep_rewards),
        }

        return log_data
