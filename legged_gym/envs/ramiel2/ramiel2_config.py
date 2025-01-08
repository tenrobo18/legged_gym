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

from legged_gym.envs.base.monolegged_robot_config import MonoLeggedRobotCfg, MonoLeggedRobotCfgPPO

class Ramiel2FlatCfg( MonoLeggedRobotCfg ):
    class env( MonoLeggedRobotCfg.env):
        num_envs = 2048
        num_observations = 18 + 1
        num_privileged_obs = 18 + 1 + 16
        # num_observations = 169
        num_actions = 3

    # class terrain( LeggedRobotCfg.terrain):
    #     measured_points_x = [-0.5, -0.4, -0.3, -0.2, -0.1, 0., 0.1, 0.2, 0.3, 0.4, 0.5] # 1mx1m rectangle (without center line)
    #     measured_points_y = [-0.5, -0.4, -0.3, -0.2, -0.1, 0., 0.1, 0.2, 0.3, 0.4, 0.5]

    class terrain(MonoLeggedRobotCfg.terrain):
        # mesh_type = 'plane'
        # measure_heights = False

        mesh_type = 'trimesh' # "heightfield" # none, plane, heightfield or trimesh
        horizontal_scale = 0.1 # [m]
        vertical_scale = 0.005 # [m]
        border_size = 25 # [m]
        curriculum = True 
        static_friction = 1.0
        dynamic_friction = 1.0
        restitution = 0.
        # rough terrain only:
        measure_heights = False
        measured_points_x = [-0.8, -0.7, -0.6, -0.5, -0.4, -0.3, -0.2, -0.1, 0., 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8] # 1mx1.6m rectangle (without center line)
        measured_points_y = [-0.5, -0.4, -0.3, -0.2, -0.1, 0., 0.1, 0.2, 0.3, 0.4, 0.5]
        selected = False # select a unique terrain type and pass all arguments
        terrain_kwargs = None # Dict of arguments for selected terrain
        max_init_terrain_level = 5 # starting curriculum state
        terrain_length = 8.
        terrain_width = 8.
        num_rows= 15 # number of terrain rows (levels)
        num_cols = 25 # number of terrain cols (types)
        # terrain types: [smooth slope, rough slope, stairs up, stairs down, discrete]
        terrain_proportions = [0.15, 0.15, 0.35, 0.15, 0.2]
        # trimesh only:
        slope_treshold = 0.75 # slopes above this threshold will be corrected to vertical surfaces

    class init_state( MonoLeggedRobotCfg.init_state ):
        pos = [0.0, 0.0, 1.0] # x,y,z [m]
        default_joint_angles = { # = target angles [rad] when action = 0.0
            'roll': 0.0,
            'pitch': 0.0,
            'slide': -0.2,
        }

    class commands:
        curriculum = False
        max_curriculum = 1.
        num_commands = 4 # default: lin_vel_x, lin_vel_y, ang_vel_yaw, heading (in heading mode ang_vel_yaw is recomputed from heading error)
        resampling_time = 7. # time before command are changed[s]
        heading_command = False # if true: compute ang vel command from heading error
        delay_range = [0.00, 0.04]
        class ranges:
            lin_vel_x = [-1.0, 1.0] # min max [m/s]
            lin_vel_y = [-0.8, 0.8]   # min max [m/s]
            ang_vel_yaw = [-1.2, 1.2]    # min max [rad/s]
            heading = [-3.14, 3.14]

    class control( MonoLeggedRobotCfg.control ):
        # PD Drive parameters:
        stiffness = {'roll': 50.0, 'pitch': 50.0, 'slide': 800.}  # [N*m/rad]
        damping = {'roll': 1.8, 'pitch': 1.8, 'slide': 32}  # [N*m*s/rad]
        # action scale: target angle = actionScale * action + defaultAngle
        action_scale = 0.2
        # decimation: Number of control action updates @ sim DT per policy DT
        decimation = 4

    class asset( MonoLeggedRobotCfg.asset ):
        file = '{LEGGED_GYM_ROOT_DIR}/resources/robots/ramiel2/urdf/ramiel2.urdf'
        name = "ramiel2"
        foot_name = 'leg_link'
        terminate_after_contacts_on = ['base_link']
        flip_visual_attachments = False
        self_collisions = 1 # 1 to disable, 0 to enable...bitwise filter

    class domain_rand:
        randomize_friction = True
        friction_range = [-0.4, 0.6]
        randomize_mass = True
        added_mass_rigid_body_indices = [0, 1, 2, 3]
        added_mass_rate = [0.1, 0.1, 0.1, 0.1]
        push_robots = True
        push_interval_s = 12
        max_push_vel_xy = 1.

    class rewards( MonoLeggedRobotCfg.rewards ):
        base_heightt_min = 0.5
        base_height_max = 1.2
        soft_dof_pos_limit = 0.95
        soft_dof_vel_limit = 0.5
        soft_torque_limit = 0.8
        max_contact_force = 300.
        only_positive_rewards = True # if true negative total rewards are clipped at zero (avoids early termination problems)
        curriculum = True
        curriculum_offset = 0.01
        curriculum_decay = 0.9999
        class scales( MonoLeggedRobotCfg.rewards.scales ):
            # termination = -200.0
            tracking_lin_vel = 1.0 # fix
            tracking_ang_vel = 0.5
            feet_air_time = 1.0
            action_rate = -0.05
            orientation = -10.0
            base_height_range = -10.0
            ang_vel_xyz = -0.03
            torques = -1.0e-6
            dof_acc = -1.0e-4
            dof_vel = -1.0e-2
            stand_still = 0.0
            stand_still_contact = 0.0
            stumble = -3.0
            dof_pos_limits = -100.0
            dof_vel_limits = -0.1
            torque_limits = -1e-3
            collision = -1.0
            foot_slippage = -0.1

    class noise:
        add_noise = True
        noise_level = 1.0 # scales other values
        curriculum = True
        curriculum_offset = 0.01
        curriculum_decay = 0.99998
        class noise_scales:
            dof_pos = 0.05
            dof_vel = 1.0
            lin_vel = 0.2
            ang_vel = 0.3
            gravity = 0.1
            height_measurements = 0.1


class Ramiel2FlatCfgPPO( MonoLeggedRobotCfgPPO ):
    seed = 1
    # runner_class_name = 'ActorCriticReccurent'
    # class policy:
    #     # only for 'ActorCriticRecurrent':
    #     rnn_type = 'lstm'
    #     rnn_hidden_size = 512
    #     rnn_num_layers = 1

    runner_class_name = 'OnPolicyRunner'
    class policy( MonoLeggedRobotCfgPPO.policy):
        init_noise_std = 1.0
        actor_hidden_dims = [256, 128, 64]
        critic_hidden_dims = [256, 128, 64]
        activation = 'elu' # can be elu, relu, selu, crelu, lrelu, tanh, sigmoid

    class algorithm( MonoLeggedRobotCfgPPO.algorithm):
        entropy_coef = 0.003
        learning_rate = 1.e-3
        clip_param = 0.2
        schedule = 'adaptive' # could be adaptive, fixed

    class runner( MonoLeggedRobotCfgPPO.runner ):
        run_name = ''
        experiment_name = 'flat_ramiel2'
        num_steps_per_env = 48 # per iteration
        max_iterations = 40000 # number of policy updates

