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
        for i in range(3):
            self.q_curs[:, i] = self.df[[f"q{i}[rad]"]].to_numpy(copy=True).flatten()
            self.dq_curs[:, i] = self.df[[f"dq_cur{i}[rad/s]"]].to_numpy(copy=True).flatten()           
            self.q_refs[:, i] = self.df[[f"q_ref{i}[rad]"]].to_numpy(copy=True).flatten()
            self.tau_curs[:, i] = self.df[[f"tau_cur{i}[Nm]"]].to_numpy(copy=True).flatten()
            self.tau_refs[:, i] = self.df[[f"tau_ref{i}[Nm]"]].to_numpy(copy=True).flatten()

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
    env_cfg.asset.fix_base_link = True 

    # prepare environment
    env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)
    obs = env.get_observations()

    # load data
    log_file = "~/legged_gym/legged_gym/scripts/data/log-2025_01_27_12_38_01_rl_parameter_mintension50N.csv"
    start_time = 32.
    end_time = 52.
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
            
            #root linkの位置を調整
            env.root_states[env_ids] = env.base_init_state
            env.root_states[env_ids, 2] += 1.0 #upward robot for 1m  
            env.gym.set_actor_root_state_tensor_indexed(env.sim,
                                                gymtorch.unwrap_tensor(env.root_states),
                                                gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))
            
        else:
            time_array[i] = i * env.dt + recorded_policy.start_time
            recorded_actions_raw = recorded_policy_wrapper(i * env.dt) 
            actions = (recorded_actions_raw - env.default_dof_pos) / env_cfg.control.action_scale
            # actions[:] = 0.
            obs, _, rews, dones, infos = env.step(actions)
            isaac_q_curs[i] = (obs[0, 12:15] + env.default_dof_pos.squeeze()).cpu().numpy()
            isaac_tau_curs[i] = env.torques[0, :].cpu().numpy()


    fig, axes = plt.subplots(nrows=2, ncols=3)
    for i in range(3):
        axes[0, i].title.set_text(f"q {i}")
        axes[0, i].plot(recorded_policy.times, recorded_policy.q_curs[:,i])
        axes[0, i].plot(recorded_policy.times, recorded_policy.q_refs[:,i])
        axes[0, i].plot(time_array, isaac_q_curs[:,i])
        axes[0, i].legend(["q_cur", "q_ref", "q_cur_isaac"])

        axes[1, i].title.set_text(f"tau{i}")
        axes[1, i].plot(recorded_policy.times, recorded_policy.tau_curs[:,i])
        axes[1, i].plot(recorded_policy.times, recorded_policy.tau_refs[:,i])
        axes[1, i].plot(time_array, isaac_tau_curs[:,i])
        axes[1, i].legend(["tau_cur", "tau_ref", "tau_cur_isaac"])     
    plt.show()







if __name__ == '__main__':
    args = get_args()
    play(args)
