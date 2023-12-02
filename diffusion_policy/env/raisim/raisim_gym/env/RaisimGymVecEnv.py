# //----------------------------//
# // This file is part of RaiSim//
# // Copyright 2020, RaiSim Tech//
# //----------------------------//

import numpy as np
import platform
import os


class RaisimGymVecEnv:

    def __init__(self, impl, normalize_ob=True, seed=0, clip_obs=10.):
        if platform.system() == "Darwin":
            os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
        self.normalize_ob = normalize_ob
        self.clip_obs = clip_obs
        self.wrapper = impl
        self.num_obs = self.wrapper.getObDim()
        self.num_acts = self.wrapper.getActionDim()
        self._observation = np.zeros([self.num_envs, self.num_obs], dtype=np.float32)
        self._joint_pos_err_history = np.zeros([self.num_envs, 2*12], dtype=np.float32)
        self._joint_vel_history = np.zeros([self.num_envs, 2*12], dtype=np.float32)
        self._contact_states = np.zeros([self.num_envs, 4], dtype=np.float32)
        self.actions = np.zeros([self.num_envs, self.num_acts], dtype=np.float32)
        self.log_prob = np.zeros(self.num_envs, dtype=np.float32)
        self._reward = np.zeros(self.num_envs, dtype=np.float32)
        self._done = np.zeros(self.num_envs, dtype=bool)
        self.rewards = [[] for _ in range(self.num_envs)]
        self.wrapper.setSeed(seed)
        self.count = 0.0
        self.mean = np.zeros(self.num_obs, dtype=np.float32)
        self.var = np.zeros(self.num_obs, dtype=np.float32)
        
        self._max_episode_steps = 1000

    def seed(self, seed=None):
        self.wrapper.setSeed(seed)

    def turn_on_visualization(self):
        self.wrapper.turnOnVisualization()

    def turn_off_visualization(self):
        self.wrapper.turnOffVisualization()

    def start_video_recording(self, file_name):
        self.wrapper.startRecordingVideo(file_name)

    def stop_video_recording(self):
        self.wrapper.stopRecordingVideo()

    def step(self, action):
        self.wrapper.step(action, self._reward, self._done)
        return self.observe(), self._reward.copy(), self._done.copy()

    def load_scaling(self, dir_name, iteration, count=1e5):
        mean_file_name = dir_name + "/mean" + str(iteration) + ".csv"
        var_file_name = dir_name + "/var" + str(iteration) + ".csv"
        self.count = count
        self.mean = np.loadtxt(mean_file_name, dtype=np.float32)[:48]
        self.var = np.loadtxt(var_file_name, dtype=np.float32)[:48]
        self.wrapper.setObStatistics(self.mean, self.var, self.count)

    def save_scaling(self, dir_name, iteration):
        mean_file_name = dir_name + "/mean" + iteration + ".csv"
        var_file_name = dir_name + "/var" + iteration + ".csv"
        self.wrapper.getObStatistics(self.mean, self.var, self.count)
        np.savetxt(mean_file_name, self.mean)
        np.savetxt(var_file_name, self.var)

    def observe(self, update_statistics=True):
        self.wrapper.observe(self._observation, update_statistics)
        return self._observation[..., :33]
        # return np.concatenate([self._observation[..., :33], self._observation[..., 36:]], axis=-1)

    def reset(self, conditional_reset=False):
        self._reward = np.zeros(self.num_envs, dtype=np.float32)

        if not conditional_reset:
            self.wrapper.reset()
        else:
            self.wrapper.conditionalReset()
            return self.wrapper.conditionalResetFlags()

        # return [True] * self.num_envs
        return self.observe()

    def close(self):
        self.wrapper.close()

    def curriculum_callback(self):
        self.wrapper.curriculumUpdate()

    def enable_early_termination(self):
        self.wrapper.enableEarlyTermination()

    def disable_early_termination(self):
        self.wrapper.disableEarlyTermination()

    def set_max_episode_length(self, time_in_seconds):
        self.wrapper.setMaxEpisodeLength(time_in_seconds)

    @property
    def num_envs(self):
        return self.wrapper.getNumOfEnvs()

    def get_reward_info(self):
        return self.wrapper.rewardInfo()
    
    def get_normalized_score(self, score):
        # dummy function
        return score / 100

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