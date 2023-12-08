# //----------------------------//
# // This file is part of RaiSim//
# // Copyright 2020, RaiSim Tech//
# //----------------------------//

import numpy as np
import platform
import os

from diffusion_policy.env.raisim.lib.raisim_env import RaisimWrapper

class RaisimEnv:

    def __init__(self, resource_dir, cfg, seed=0):
        if platform.system() == "Darwin":
            os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
        
        # initialize environment
        self.env = RaisimWrapper(resource_dir, cfg)
        self.env.setSeed(seed)

        # get environment information
        self.num_obs = self.env.getObDim()
        self.num_acts = self.env.getActionDim()

        # initialize variables
        self._observation = np.zeros([self.num_envs, self.num_obs], dtype=np.float32)
        self._joint_pos_err_history = np.zeros([self.num_envs, 2*12], dtype=np.float32)
        self._joint_vel_history = np.zeros([self.num_envs, 2*12], dtype=np.float32)
        self._contact_states = np.zeros([self.num_envs, 4], dtype=np.float32)
        self._reward = np.zeros(self.num_envs, dtype=np.float32)
        self._done = np.zeros(self.num_envs, dtype=bool)
        self.rewards = [[] for _ in range(self.num_envs)]
        self.count = 0.0
        self.mean = np.zeros(self.num_obs, dtype=np.float32)
        self.var = np.zeros(self.num_obs, dtype=np.float32)
        
        self._max_episode_steps = 1000

    def seed(self, seed=None):
        self.env.setSeed(seed)

    def turn_on_visualization(self):
        self.env.turnOnVisualization()

    def turn_off_visualization(self):
        self.env.turnOffVisualization()

    def start_video_recording(self, file_name):
        self.env.startRecordingVideo(file_name)

    def stop_video_recording(self):
        self.env.stopRecordingVideo()

    def step(self, action):
        self.env.step(action, self._reward, self._done)
        return self.observe(), self._reward.copy(), self._done.copy()

    def load_scaling(self, dir_name, iteration, count=1e5):
        mean_file_name = dir_name + "/mean" + str(iteration) + ".csv"
        var_file_name = dir_name + "/var" + str(iteration) + ".csv"
        self.count = count
        self.mean = np.loadtxt(mean_file_name, dtype=np.float32)[:48]
        self.var = np.loadtxt(var_file_name, dtype=np.float32)[:48]
        self.env.setObStatistics(self.mean, self.var, self.count)

    def save_scaling(self, dir_name, iteration):
        mean_file_name = dir_name + "/mean" + iteration + ".csv"
        var_file_name = dir_name + "/var" + iteration + ".csv"
        self.env.getObStatistics(self.mean, self.var, self.count)
        np.savetxt(mean_file_name, self.mean)
        np.savetxt(var_file_name, self.var)

    def observe(self, update_statistics=True):
        self.env.observe(self._observation, update_statistics)
        return self._observation[..., :33]
        # return np.concatenate([self._observation[..., :33], self._observation[..., 36:]], axis=-1)

    def reset(self, conditional_reset=False):
        self._reward = np.zeros(self.num_envs, dtype=np.float32)

        if not conditional_reset:
            self.env.reset()
        else:
            self.env.conditionalReset()
            return self.env.conditionalResetFlags()

        # return [True] * self.num_envs
        return self.observe()

    def close(self):
        self.env.close()

    def curriculum_callback(self):
        self.env.curriculumUpdate()

    def enable_early_termination(self):
        self.env.enableEarlyTermination()

    def disable_early_termination(self):
        self.env.disableEarlyTermination()

    def set_max_episode_length(self, time_in_seconds):
        self.env.setMaxEpisodeLength(time_in_seconds)

    @property
    def num_envs(self):
        return self.env.getNumOfEnvs()

    def get_reward_info(self):
        return self.env.rewardInfo()
    

    def get_joint_pos_err_history(self):
        self.wrapper.getJointPositionErrorHistory(self._joint_pos_err_history)
        return self._joint_pos_err_history
    
    def get_joint_vel_history(self):
        self.wrapper.getJointVelocityHistory(self._joint_vel_history)
        return self._joint_vel_history

    def get_contact_states(self):
        self.wrapper.getContactStates(self._contact_states)
        return self._contact_states
    
    def kill_server(self):
        self.wrapper.killServer()