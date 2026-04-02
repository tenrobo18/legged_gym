# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# 
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# Copyright (c) 2021 ETH Zurich, Nikita Rudin

from legged_gym import LEGGED_GYM_ROOT_DIR
import os

import isaacgym
from legged_gym.envs import *
from legged_gym.utils import  get_args, export_policy_as_jit, task_registry, Logger
from isaacgym import gymtorch, gymapi, gymutil

import numpy as np
import pandas as pd
import torch
import time
import bisect
import scipy
import matplotlib.pyplot as plt

#params
fixed = True

# slide joint log
# log_file = "~/legged_gym/legged_gym/scripts/data/log-2025_09_02_17_41_43_jointchirp_jointslide_10s_-0.15rad-0.15rad_2Hz-10Hz.csv"
# start_time = 34.9
# end_time = 44.8

#roll joint log
# log_file = "~/legged_gym/legged_gym/scripts/data/log-2025_09_02_20_19_49_jointchirp_jointroll_10s_-0.5rad-0.5rad_2Hz-10Hz.csv"
# start_time = 29.7
# end_time = 39.6

#pitch joint log
# log_file = "~/legged_gym/legged_gym/scripts/data/log-2025_09_02_20_23_21_jointchirp_jointpitch_10s_-0.5rad-0.5rad_2Hz-10Hz.csv"
# start_time = 31.8
# end_time = 41.7

#slide joint log, kd = 0
# log_file = "~/legged_gym/legged_gym/scripts/data/log-2025_09_05_11_14_55_jointchirp_jointslide_10s_-0.07rad-0.07rad_2Hz-10Hz_kd0.csv"
# start_time = 61.5
# end_time = 71.4

#roll joint log, kd = 0
# log_file = "~/legged_gym/legged_gym/scripts/data/log-2025_09_05_11_18_24_jointchirp_jointroll_10s_-0.5rad-0.5rad_2Hz-10Hz_kd0.csv"
# start_time = 43.2
# end_time = 53.1

#pitch joint log, kd = 0
# log_file = "~/legged_gym/legged_gym/scripts/data/log-2025_09_05_11_21_05_jointchirp_jointpitch_10s_-0.5rad-0.5rad_2Hz-10Hz_kd0.csv"
# start_time = 55.3
# end_time = 65.2

#slide joint log, kd = 0, 2026/04/02
# log_file = "~/legged_gym/legged_gym/scripts/data/log-2026_04_02_14_35_59_jointchirp_jointslide_10s_-0.06rad-0.06rad_2Hz-10Hz.csv"
# start_time = 58.881
# end_time = 68.781

#roll joint log, kd = 0, 2026/04/02
# log_file = "~/legged_gym/legged_gym/scripts/data/log-2026_04_02_14_40_44_jointchirp_jointroll_10s_-0.5rad-0.5rad_2Hz-10Hz.csv"
# start_time = 33.579
# end_time = 43.479

#pitch joint log, kd = 0, 2026/04/02
log_file = "~/legged_gym/legged_gym/scripts/data/log-2026_04_02_14_43_23_jointchirp_jointpitch_10s_-0.5rad-0.5rad_2Hz-10Hz.csv"
start_time = 45.82
end_time = 55.72


class RecordedPolicy:
    def __init__(self, df_):

        num_steps = len(df_) - 1
        self.df = df_
        self.start_time = self.df.index[0]
        self.end_time = self.df.index[-1]
        self.times = self.df.index.to_numpy(copy=True)
        self.q_curs = np.zeros((len(self.times), 3))
        self.dq_curs = np.zeros((len(self.times), 3))
        self.q_refs = np.zeros((len(self.times), 3))
        self.tau_curs = np.zeros((len(self.times), 3))
        self.tau_refs = np.zeros((len(self.times), 3))
        self.pos_curs = np.zeros((len(self.times), 3))
        self.quat_curs = np.zeros((len(self.times), 4))
        self.lin_vel_curs = np.zeros((len(self.times), 3))
        self.ang_vel_curs = np.zeros((len(self.times), 3))
        for i in range(3):
            self.q_curs[:, i] = self.df[[f"q{i}[rad]"]].to_numpy(copy=True).flatten()
            self.dq_curs[:, i] = self.df[[f"dq_cur{i}[rad/s]"]].to_numpy(copy=True).flatten()           
            self.q_refs[:, i] = self.df[[f"q_ref{i}[rad]"]].to_numpy(copy=True).flatten()
            self.tau_curs[:, i] = self.df[[f"tau_cur{i}[Nm]"]].to_numpy(copy=True).flatten()
            self.tau_refs[:, i] = self.df[[f"tau_ref{i}[Nm]"]].to_numpy(copy=True).flatten()
        self.pos_curs[:, 0] = self.df[[f"pose_cur_pos_x[m]"]].to_numpy(copy=True).flatten()
        self.pos_curs[:, 1] = self.df[[f"pose_cur_pos_y[m]"]].to_numpy(copy=True).flatten()
        self.pos_curs[:, 2] = self.df[[f"pose_cur_pos_z[m]"]].to_numpy(copy=True).flatten()
        self.quat_curs[:, 0] = self.df[[f"pose_cur_ori_x"]].to_numpy(copy=True).flatten()
        self.quat_curs[:, 1] = self.df[[f"pose_cur_ori_y"]].to_numpy(copy=True).flatten()
        self.quat_curs[:, 2] = self.df[[f"pose_cur_ori_z"]].to_numpy(copy=True).flatten()
        self.quat_curs[:, 3] = self.df[[f"pose_cur_ori_w"]].to_numpy(copy=True).flatten()
        self.lin_vel_curs[:, 0] = self.df[[f"twist_cur_robot_pos_x[m/s]"]].to_numpy(copy=True).flatten()
        self.lin_vel_curs[:, 1] = self.df[[f"twist_cur_robot_pos_y[m/s]"]].to_numpy(copy=True).flatten()
        self.lin_vel_curs[:, 2] = self.df[[f"twist_cur_robot_pos_z[m/s]"]].to_numpy(copy=True).flatten()
        self.ang_vel_curs[:, 0] = self.df[[f"twist_cur_robot_ang_x[rad/s]"]].to_numpy(copy=True).flatten()
        self.ang_vel_curs[:, 1] = self.df[[f"twist_cur_robot_ang_y[rad/s]"]].to_numpy(copy=True).flatten()
        self.ang_vel_curs[:, 2] = self.df[[f"twist_cur_robot_ang_z[rad/s]"]].to_numpy(copy=True).flatten()


    def get_q(self, time, qs):
        """
        Input : time (float32), qs(np.ndarray (len(self.times), 3))
        Output : q_ref (np.ndarray 3)
        """
        time_offested = time + self.start_time
        index = bisect.bisect_left(self.times, time_offested)
        if index == 0:
            return qs[0]
        elif index == len(qs):
            return qs[-1]
        else:
            time1 = self.times[index-1]
            time2 = self.times[index]
            ratio = (time_offested - time1) / (time2 - time1)
            q = qs[index-1] * (1 - ratio) + qs[index] * ratio
            return q

def getTorchWrapper(recorded_policy, num_envs, device):
    def recordedPolicy(time):
        nonlocal recorded_policy
        q_refs = torch.tensor(recorded_policy.get_q(time, recorded_policy.q_refs))
        return q_refs.unsqueeze(0).repeat(num_envs, 1).to(device)

    return recordedPolicy


def play(args):
    env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)
    # override some parameters for testing
    # env_cfg.env.num_envs = min(env_cfg.env.num_envs, 50)
    env_cfg.terrain.num_rows = 2
    env_cfg.terrain.num_cols = 2
    env_cfg.env.num_envs = env_cfg.terrain.num_rows * env_cfg.terrain.num_cols
    env_cfg.terrain.curriculum = False
    env_cfg.noise.add_noise = False
    env_cfg.domain_rand.randomize_friction = False
    env_cfg.domain_rand.push_robots = False
    env_cfg.terrain.mesh_type = 'plane'
    env_cfg.asset.terminate_after_contacts_on = []
    if fixed:
        env_cfg.asset.fix_base_link = True 

    # prepare environment
    env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)
    obs_scales = env_cfg.normalization.obs_scales
    obs = env.get_observations()

    # load data
    print(f"reading {log_file}")
    df = pd.read_csv(log_file, index_col=0)
    df.columns = df.columns.str.replace(' ', '') # remove spaces from column names
    df = df.loc[start_time:end_time] # remove all rows that are not in the time range
    num_steps = int((end_time - start_time) / env.dt)+1

    recorded_policy = RecordedPolicy(df)
    recorded_policy_wrapper = getTorchWrapper(recorded_policy, env_cfg.env.num_envs, env.device)

    isaac_q_curs = np.zeros((num_steps, 3))
    isaac_dq_curs = np.zeros((num_steps, 3))
    isaac_tau_curs = np.zeros((num_steps, 3))
    isaac_quat_curs = np.zeros((num_steps, 4))
    isaac_lin_vel_curs = np.zeros((num_steps, 3))
    isaac_ang_vel_curs = np.zeros((num_steps, 3))
    time_array = np.zeros((num_steps, 1))

    # set delay
    delay_s = 0.0
    
    start_offset_time = 0.0
    start_index = int(start_offset_time / env.dt)

    for i in range(start_index, num_steps):
        if i == 0:
            time_array[i] = start_time

            #q_curとdq_curを初期化
            env_ids = torch.arange(env.num_envs, device=env.device)
            
            q_cur_tensor = torch.tensor(recorded_policy.q_curs[0], device=env.device, dtype=env.dof_pos.dtype)
            dq_cur_tensor = torch.tensor(recorded_policy.dq_curs[0], device=env.device, dtype=env.dof_pos.dtype)
            env.dof_pos[env_ids] = q_cur_tensor.unsqueeze(0).repeat(env.num_envs, 1).to(env.device)
            env.dof_vel[env_ids] = dq_cur_tensor.unsqueeze(0).repeat(env.num_envs, 1).to(env.device)
            env_ids_int32 = env_ids.to(dtype=torch.int32)
            env.gym.set_dof_state_tensor_indexed(env.sim,
                                                 gymtorch.unwrap_tensor(env.dof_state),
                                                 gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))
            isaac_q_curs[i] = recorded_policy.q_curs[0]
            isaac_dq_curs[i] = recorded_policy.dq_curs[0]
            
            #root linkの姿勢, 速度を調整
            pos_cur_tensor = torch.tensor(recorded_policy.pos_curs[0], device=env.device, dtype=env.dof_pos.dtype)
            quat_cur_tensor = torch.tensor(recorded_policy.quat_curs[0], device=env.device, dtype=env.dof_pos.dtype)
            lin_vel_cur_tensor = torch.tensor(recorded_policy.lin_vel_curs[0], device=env.device, dtype=env.dof_pos.dtype)
            ang_vel_cur_tensor = torch.tensor(recorded_policy.ang_vel_curs[0], device=env.device, dtype=env.dof_pos.dtype)
            env.root_states[env_ids] = env.base_init_state
            if fixed:
                env.root_states[env_ids, 2] += 1.0 #upward robot for 1m  
            else:
                env.root_states[env_ids, 2] += pos_cur_tensor[2].unsqueeze(0).repeat(env.num_envs, 1).to(env.device)[2] - 0.7
                env.root_states[env_ids, 3:7] = quat_cur_tensor.unsqueeze(0).repeat(env.num_envs, 1).to(env.device)
                env.root_states[env_ids, 7:10] = lin_vel_cur_tensor.unsqueeze(0).repeat(env.num_envs, 1).to(env.device)
                env.root_states[env_ids, 10:13] = ang_vel_cur_tensor.unsqueeze(0).repeat(env.num_envs, 1).to(env.device)
            env.gym.set_actor_root_state_tensor_indexed(env.sim,
                                                gymtorch.unwrap_tensor(env.root_states),
                                                gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))
            
        else:
            time_array[i] = i * env.dt + recorded_policy.start_time
            recorded_actions_raw = recorded_policy_wrapper(i * env.dt) 
            actions = (recorded_actions_raw - env.default_dof_pos) / env_cfg.control.action_scale
            # actions[:] = 0.
            obs, _, rews, dones, infos = env.step(actions)
            isaac_q_curs[i] = (obs[0, 12:15] / obs_scales.dof_pos + env.default_dof_pos.squeeze()).cpu().numpy()
            isaac_dq_curs[i] = (obs[0, 15:18] / obs_scales.dof_vel).cpu().numpy()
            isaac_tau_curs[i] = env.torques[0, :].detach().cpu().numpy()
            isaac_lin_vel_curs[i] = env.base_lin_vel[0, :].detach().cpu().numpy()


    fig, axes = plt.subplots(nrows=4, ncols=3)
    for i in range(3):
        axes[0, i].title.set_text(f"q {i}")
        axes[0, i].plot(recorded_policy.times, recorded_policy.q_curs[:,i])
        axes[0, i].plot(recorded_policy.times, recorded_policy.q_refs[:,i])
        axes[0, i].plot(time_array, isaac_q_curs[:,i])
        axes[0, i].legend(["q_cur", "q_ref", "q_cur_isaac"])

        axes[1, i].title.set_text(f"dq_cur{i}")
        axes[1, i].plot(recorded_policy.times, recorded_policy.dq_curs[:,i])
        axes[1, i].plot(time_array, isaac_dq_curs[:,i])
        axes[1, i].legend(["dq_cur", "dq_cur_isaac"])

        axes[2, i].title.set_text(f"tau{i}")
        axes[2, i].plot(recorded_policy.times, recorded_policy.tau_curs[:,i])
        axes[2, i].plot(recorded_policy.times, recorded_policy.tau_refs[:,i])
        axes[2, i].plot(time_array, isaac_tau_curs[:,i])
        axes[2, i].legend(["tau_cur", "tau_ref", "tau_cur_isaac"])   

        axes[3, i].title.set_text(f"lin_vel{i}")
        axes[3, i].plot(recorded_policy.times, recorded_policy.lin_vel_curs[:,i])
        axes[3, i].plot(time_array, isaac_lin_vel_curs[:,i])
        axes[3, i].legend(["lin_vel_cur", "lin_vel_cur_isaac"])  
    plt.show()







if __name__ == '__main__':
    args = get_args()
    play(args)
