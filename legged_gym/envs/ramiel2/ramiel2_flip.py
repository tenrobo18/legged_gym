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

from time import time
import numpy as np
import os

from isaacgym.torch_utils import *
from isaacgym import gymtorch, gymapi, gymutil

import torch
from typing import Tuple, Dict
from legged_gym.envs import MonoLeggedRobot

class Ramiel2Flip(MonoLeggedRobot):
    def compute_observations(self):
        """ Computes observations
        """
        # print("-------------------------")
        # print(self.base_quat.cpu().numpy()[0])
        # print(self.base_lin_vel.cpu().numpy()[0])
        # print(self.base_ang_vel.cpu().numpy()[0])
        # print(self.projected_gravity.cpu().numpy()[0])
        # print(self.commands.cpu().numpy()[0])
        # print(self.dof_pos.cpu().numpy()[0])
        # print(self.default_dof_pos.cpu().numpy()[0])
        # print(self.dof_vel.cpu().numpy()[0])
        # print(self.actions.cpu().numpy()[0])
        self.obs_buf = torch.cat((  self.base_lin_vel * self.obs_scales.lin_vel,
                                    self.base_ang_vel  * self.obs_scales.ang_vel,
                                    self.projected_gravity,
                                    self.commands[:, [0, 1, 2, 4]] * self.commands_scale,
                                    (self.dof_pos - self.default_dof_pos) * self.obs_scales.dof_pos,
                                    self.dof_vel * self.obs_scales.dof_vel,
                                    # self.actions,
                                    self.is_standing,
                                    # self.last_actions,
                                    ),dim=-1)
        self.privileged_obs_buf = torch.cat((  self.base_lin_vel * self.obs_scales.lin_vel,
                                                self.base_ang_vel  * self.obs_scales.ang_vel,
                                                self.projected_gravity,
                                                self.commands[:, [0, 1, 2, 4]] * self.commands_scale,
                                                (self.dof_pos - self.default_dof_pos) * self.obs_scales.dof_pos,
                                                self.dof_vel * self.obs_scales.dof_vel,
                                                # self.actions,
                                                self.is_standing,
                                                # self.last_actions,
                                                self.torques,
                                                self.root_states[:, 0:3] - self.env_origins[:], 
                                                self.root_states[:, 3:7],
                                                self.root_states[:, 7:10],
                                                self.root_states[:, 10:13],
                                                ),dim=-1)
        # print(self.obs_buf.cpu().numpy()[0])
        # add perceptive inputs if not blind
        if self.cfg.terrain.measure_heights:
            heights = torch.clip(self.root_states[:, 2].unsqueeze(1) - 0.5 - self.measured_heights, -1, 1.) * self.obs_scales.height_measurements
            self.obs_buf = torch.cat((self.obs_buf, heights), dim=-1)
        # add noise if needed
        if self.add_noise:
            self.obs_buf += (2 * torch.rand_like(self.obs_buf) - 1) * self.noise_scale_vec * self.noise_curriculum_weight[0]

    def _resample_commands(self, env_ids):
        """ Randommly select commands of some environments

        Args:
            env_ids (List[int]): Environments ids for which new commands are needed
        """
        self.tracking_error_sum[env_ids] = 0.0
        self.step_counter[env_ids] = 0
        self.commands[env_ids, 0] = torch_rand_float(self.command_ranges["lin_vel_x"][0], self.command_ranges["lin_vel_x"][1], (len(env_ids), 1), device=self.device).squeeze(1)
        self.commands[env_ids, 1] = torch_rand_float(self.command_ranges["lin_vel_y"][0], self.command_ranges["lin_vel_y"][1], (len(env_ids), 1), device=self.device).squeeze(1)
        self.commands[env_ids, 2] = torch_rand_float(self.command_ranges["ang_vel_yaw"][0], self.command_ranges["ang_vel_yaw"][1], (len(env_ids), 1), device=self.device).squeeze(1)
        self.commands[env_ids, 3] = torch_rand_float(self.command_ranges["heading"][0], self.command_ranges["heading"][1], (len(env_ids), 1), device=self.device).squeeze(1)
        self.commands[env_ids, 4] =  self.commands[env_ids, 4] + torch_rand_float(self.command_ranges["half_turns_times_diff"][0], self.command_ranges["half_turns_times_diff"][1], (len(env_ids), 1), device=self.device).squeeze(1)

        r = torch.empty(len(env_ids), device=self.device)

        is_no_yaw_env = r.uniform_(0.0, 1.0) <= 0.7
        no_yaw_env_ids = is_no_yaw_env.nonzero(as_tuple=False).flatten()
        self.commands[no_yaw_env_ids, 2] = 0.0

        is_only_yaw_env = ~is_no_yaw_env * (r.uniform_(0.0, 1.0) <= 0.3)
        only_yaw_env_ids = is_only_yaw_env.nonzero(as_tuple=False).flatten()
        self.commands[only_yaw_env_ids, :2] = 0.0

        # set small commands to zero
        self.is_standing[env_ids] = (r.uniform_(0.0, 1.0) <= 0.10).float().reshape(-1, 1)
        self.commands[(self.is_standing > 0.5).flatten(), :] = 0.0

        self.is_heading[env_ids] = (r.uniform_(0.0, 1.0) <= 0.5).float().reshape(-1, 1)

        # set top-up commands true, if (int) half_turns_time is even number
        self.is_top_up_command[env_ids] = (self.commands[env_ids, 4].to(torch.int) % 2 == 0)

    def _reset_root_states(self, env_ids):
        """ Resets ROOT states position and velocities of selected environmments
            Sets base position based on the curriculum
            Selects randomized base velocities within -0.5:0.5 except z[m/s, rad/s]
            Selects randomized base velocity z within 1:3 m/s
        Args:
            env_ids (List[int]): Environemnt ids
        """
        # base position
        if self.custom_origins:
            self.root_states[env_ids] = self.base_init_state
            self.root_states[env_ids, :3] += self.env_origins[env_ids]
            self.root_states[env_ids, :2] += torch_rand_float(-1., 1., (len(env_ids), 2), device=self.device) # xy position within 1m of the center
            self.root_states[env_ids, 2] += 0.05 + torch_rand_float(-0.1, 0.1, (len(env_ids), 1), device=self.device).squeeze(1) # z position 0.1m above the ground
        else:
            self.root_states[env_ids] = self.base_init_state
            self.root_states[env_ids, :3] += self.env_origins[env_ids]
        # base quaternion
        roll = torch.empty(len(env_ids), device=self.device).uniform_(-0.1, 0.1)
        pitch = torch.empty(len(env_ids), device=self.device).uniform_(-0.1, 0.1)
        yaw = torch.empty(len(env_ids), device=self.device).uniform_(-3.14, 3.14)
        self.root_states[env_ids, 3:7] = quat_mul(quat_from_euler_xyz(roll, pitch, yaw), self.root_states[env_ids, 3:7])
        # base velocities
        lin_vel_z_min = torch.full((len(env_ids),), -0.5, device=self.device)
        lin_vel_z_max_start = 0.5
        lin_vel_z_max_end = 3.
        lin_vel_z_max = torch.zeros(len(env_ids), device=self.device)
        #calc lin_vel_z_ave from rand_curriculum_weight
        if self.cfg.domain_rand.curriculum:
            lin_vel_z_max = lin_vel_z_max_start + (lin_vel_z_max_end - lin_vel_z_max_start) * self.rand_curriculum_weight[env_ids]
        self.root_states[env_ids, 7:9] = torch_rand_float(-0.5, 0.5, (len(env_ids), 2), device=self.device) # [7:9]: lin vel x, y, 
        self.root_states[env_ids, 9] = lin_vel_z_min + (lin_vel_z_max - lin_vel_z_min) * torch_rand_float(0., 1.0, (len(env_ids), 1), device=self.device).squeeze(1)  # 9: lin_vel z
        self.root_states[env_ids, 10:13] = torch_rand_float(-0.5, 0.5, (len(env_ids), 3), device=self.device) #[10:13]: ang vel
        env_ids_int32 = env_ids.to(dtype=torch.int32)
        self.gym.set_actor_root_state_tensor_indexed(self.sim,
                                                     gymtorch.unwrap_tensor(self.root_states),
                                                     gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))
        # reset half turns times
        self.commands[env_ids, 4] = 0 
        self.is_top_up_command[env_ids] = True

    def _get_noise_scale_vec(self, cfg):
        """ Sets a vector used to scale the noise added to the observations.
            [NOTE]: Must be adapted when changing the observations structure

        Args:
            cfg (Dict): Environment config file

        Returns:
            [torch.Tensor]: Vector of scales used to multiply a uniform distribution in [-1, 1]
        """
        noise_vec = torch.zeros_like(self.obs_buf[0])
        self.add_noise = self.cfg.noise.add_noise
        noise_scales = self.cfg.noise.noise_scales
        noise_level = self.cfg.noise.noise_level
        noise_vec[:3] = noise_scales.lin_vel * noise_level * self.obs_scales.lin_vel
        noise_vec[3:6] = noise_scales.ang_vel * noise_level * self.obs_scales.ang_vel
        noise_vec[6:9] = noise_scales.gravity * noise_level
        noise_vec[9:13] = 0. # commands
        noise_vec[13:(13+self.num_dof)] = noise_scales.dof_pos * noise_level * self.obs_scales.dof_pos
        noise_vec[(13+self.num_dof):(13+2*self.num_dof)] = noise_scales.dof_vel * noise_level * self.obs_scales.dof_vel
        noise_vec[(13+2*self.num_dof):(14+2*self.num_dof)] = 0. # is standing
        # if self.cfg.terrain.measure_heights:
        #    noise_vec[(13+2*self.num_dof):235] = noise_scales.height_measurements* noise_level * self.obs_scales.height_measurements
        return noise_vec
    

    #----------------------------------------
    def _init_buffers(self):
        """ Initialize torch tensors which will contain simulation states and processed quantities
        """
        # get gym GPU state tensors
        actor_root_state = self.gym.acquire_actor_root_state_tensor(self.sim)
        dof_state_tensor = self.gym.acquire_dof_state_tensor(self.sim)
        net_contact_forces = self.gym.acquire_net_contact_force_tensor(self.sim)
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)

        # create some wrapper tensors for different slices
        self.root_states = gymtorch.wrap_tensor(actor_root_state)
        self.dof_state = gymtorch.wrap_tensor(dof_state_tensor)
        self.dof_pos = self.dof_state.view(self.num_envs, self.num_dof, 2)[..., 0]
        self.dof_vel = self.dof_state.view(self.num_envs, self.num_dof, 2)[..., 1]
        self.base_quat = self.root_states[:, 3:7]

        self.contact_forces = gymtorch.wrap_tensor(net_contact_forces).view(self.num_envs, -1, 3) # shape: num_envs, num_bodies, xyz axis

        # initialize some data used later on
        self.common_step_counter = 0
        self.step_counter = torch.zeros(self.num_envs, dtype=torch.float, device=self.device, requires_grad=False)
        self.extras = {}
        self.noise_scale_vec = self._get_noise_scale_vec(self.cfg)
        self.gravity_vec = to_torch(get_axis_params(-1., self.up_axis_idx), device=self.device).repeat((self.num_envs, 1))
        self.forward_vec = to_torch([1., 0., 0.], device=self.device).repeat((self.num_envs, 1))
        self.torques = torch.zeros(self.num_envs, self.num_dof, dtype=torch.float, device=self.device, requires_grad=False)
        self.p_gains = torch.zeros(self.num_dof, dtype=torch.float, device=self.device, requires_grad=False)
        self.d_gains = torch.zeros(self.num_dof, dtype=torch.float, device=self.device, requires_grad=False)
        self.actions = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        self.last_actions = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        self.last_last_actions = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        self.dynprms = self.cfg.domain_rand.dynprm_range[0] + (self.cfg.domain_rand.dynprm_range[1] - self.cfg.domain_rand.dynprm_range[0]) * torch_rand_float(0., 1., (self.num_envs, 1), device=self.device)

        # self.actions_delay_range = [0.04-0.0025, 0.04+0.0075]
        self.actions_delay_range = self.cfg.commands.delay_range
        self.actions_history_length = int((self.actions_delay_range[1]+self.dt)/self.dt)
        self.actions_history = torch.zeros((self.actions_history_length, self.num_envs, self.num_actions), dtype= torch.float, device= self.device, requires_grad=False)
        self.current_actions_delay = torch_rand_float(self.actions_delay_range[0], self.actions_delay_range[1], (self.num_envs, 1), device= self.device).flatten()
        self.actions_delayed_frames = ((self.current_actions_delay / self.dt) + 1).to(int)
        self.external_forces = torch.zeros((self.num_envs, self.num_bodies, 3), device=self.device)
        self.external_torques = torch.zeros((self.num_envs, self.num_bodies, 3), device=self.device)

        self.last_dof_vel = torch.zeros_like(self.dof_vel)
        self.last_root_vel = torch.zeros_like(self.root_states[:, 7:13])
        self.commands = torch.zeros(self.num_envs, self.cfg.commands.num_commands, dtype=torch.float, device=self.device, requires_grad=False) # x vel, y vel, yaw vel, heading
        self.commands_scale = torch.tensor([self.obs_scales.lin_vel, self.obs_scales.lin_vel, self.obs_scales.ang_vel, self.obs_scales.half_turns_times_diff], device=self.device, requires_grad=False,) # TODO change this
        self.is_top_up_command = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device, requires_grad=False)
        self.reward_curriculum_weight = torch.ones(self.num_envs, dtype=torch.float, device=self.device, requires_grad=False)
        if self.cfg.rewards.curriculum:
            self.reward_curriculum_weight *= self.cfg.rewards.curriculum_offset
        self.noise_curriculum_weight = torch.ones(self.num_envs, dtype=torch.float, device=self.device, requires_grad=False)
        if self.cfg.noise.curriculum:
            self.noise_curriculum_weight *= self.cfg.noise.curriculum_offset
        self.rand_curriculum_weight = torch.ones(self.num_envs, dtype=torch.float, device=self.device, requires_grad=False)
        if self.cfg.domain_rand.curriculum:
            self.rand_curriculum_weight *= self.cfg.domain_rand.curriculum_offset
        self.feet_air_time = torch.zeros(self.num_envs, self.feet_indices.shape[0], dtype=torch.float, device=self.device, requires_grad=False)
        self.last_contacts = torch.zeros(self.num_envs, len(self.feet_indices), dtype=torch.bool, device=self.device, requires_grad=False)
        self.base_lin_vel = quat_rotate_inverse(self.base_quat, self.root_states[:, 7:10])
        self.base_ang_vel = quat_rotate_inverse(self.base_quat, self.root_states[:, 10:13])
        self.projected_gravity = quat_rotate_inverse(self.base_quat, self.gravity_vec)
        self.height_points = self._init_height_points()
        self.measured_heights = 0
        self.is_standing = torch.ones(self.num_envs, 1, dtype=torch.float, device=self.device, requires_grad=False)
        self.is_heading = torch.zeros(self.num_envs, 1, dtype=torch.float, device=self.device, requires_grad=False)
        self.tracking_error_sum = torch.zeros(self.num_envs, self.cfg.commands.num_commands, device=self.device, requires_grad=False)

        # joint positions offsets and PD gains
        self.default_dof_pos = torch.zeros(self.num_dof, dtype=torch.float, device=self.device, requires_grad=False)
        print(self.dof_names)
        for i in range(self.num_dof):
            name = self.dof_names[i]
            angle = self.cfg.init_state.default_joint_angles[name]
            self.default_dof_pos[i] = angle
            found = False
            for dof_name in self.cfg.control.stiffness.keys():
                if dof_name in name:
                    self.p_gains[i] = self.cfg.control.stiffness[dof_name]
                    self.d_gains[i] = self.cfg.control.damping[dof_name]
                    found = True
            if not found:
                self.p_gains[i] = 0.
                self.d_gains[i] = 0.
                if self.cfg.control.control_type in ["P", "V"]:
                    print(f"PD gain of joint {name} were not defined, setting them to zero")
        self.default_dof_pos = self.default_dof_pos.unsqueeze(0)

        # tendon strain, kd
        self.default_tendon_strains = torch.zeros(len(self.motor_idx), dtype=torch.float, device=self.device, requires_grad=False)
        if self.cfg.env.enable_tendon:
            for i in range(len(self.motor_idx)):
                motor_id = self.motor_idx[i]
                name = self.dof_names[motor_id]
                strain = self.cfg.init_state.default_tendon_strains[name]
                self.default_tendon_strains[motor_id] = strain
                for motor_name in self.cfg.control.kd_pull.keys():
                    if motor_name in name:
                        self.tendon_robot_model.set_kd_pull(i, self.cfg.control.kd_pull[motor_name])
                        self.tendon_robot_model.set_kd_loosen(i, self.cfg.control.kd_loosen[motor_name])
            self.default_tendon_strains = self.default_tendon_strains.unsqueeze(0)

        # tension_ref_history, strain_history, tendon_vel_motor_history
        self.tension_ref_history = torch.zeros((self.num_envs, len(self.motor_idx), self.cfg.control.tension_cur_net.input_steps), dtype=torch.float, device=self.device, requires_grad=False)
        self.tendon_strain_history = torch.zeros((self.num_envs, len(self.motor_idx), self.cfg.control.tension_cur_net.input_steps), dtype=torch.float, device=self.device, requires_grad=False)
        self.tendon_vel_motor_history = torch.zeros((self.num_envs, len(self.motor_idx), self.cfg.control.tension_cur_net.input_steps), dtype=torch.float, device=self.device, requires_grad=False)
