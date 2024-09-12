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

from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg, LeggedRobotCfgPPO

class MeviusFlatCfg( LeggedRobotCfg ):
    class env( LeggedRobotCfg.env):
        num_envs = 4096
        num_observations = 48
        # num_observations = 169
        num_actions = 12

    # class terrain( LeggedRobotCfg.terrain):
    #     measured_points_x = [-0.5, -0.4, -0.3, -0.2, -0.1, 0., 0.1, 0.2, 0.3, 0.4, 0.5] # 1mx1m rectangle (without center line)
    #     measured_points_y = [-0.5, -0.4, -0.3, -0.2, -0.1, 0., 0.1, 0.2, 0.3, 0.4, 0.5]

    class terrain(LeggedRobotCfg.terrain):
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
        num_rows= 10 # number of terrain rows (levels)
        num_cols = 20 # number of terrain cols (types)
        # terrain types: [smooth slope, rough slope, stairs up, stairs down, discrete]
        terrain_proportions = [0.2, 0.2, 0.35, 0.25, 0.0]
        # trimesh only:
        slope_treshold = 0.75 # slopes above this threshold will be corrected to vertical surfaces

    class init_state( LeggedRobotCfg.init_state ):
        pos = [0.0, 0.0, 0.33] # x,y,z [m]
        default_joint_angles = { # = target angles [rad] when action = 0.0
            'FR_collar': -0.1,
            'FR_hip': 0.8,
            'FR_knee': -1.4,
            'FL_collar': 0.1,
            'FL_hip': 0.8,
            'FL_knee': -1.4,
            'BR_collar': -0.1,
            'BR_hip': 1.0,
            'BR_knee': -1.4,
            'BL_collar': 0.1,
            'BL_hip': 1.0,
            'BL_knee': -1.4,
        }

    class commands:
        curriculum = False
        max_curriculum = 1.
        num_commands = 4 # default: lin_vel_x, lin_vel_y, ang_vel_yaw, heading (in heading mode ang_vel_yaw is recomputed from heading error)
        resampling_time = 10. # time before command are changed[s]
        heading_command = False # if true: compute ang vel command from heading error
        delay_range = [0.01-0.0025, 0.03+0.0075]
        class ranges:
            # first
            lin_vel_x = [-0.7, 0.7] # min max [m/s]
            lin_vel_y = [-0.5, 0.5]   # min max [m/s]
            ang_vel_yaw = [-0.7, 0.7]    # min max [rad/s]
            # second
            # lin_vel_x = [-1.0, 1.0] # min max [m/s]
            # lin_vel_y = [-1.0, 1.0]   # min max [m/s]
            # ang_vel_yaw = [-1.0, 1.0]    # min max [rad/s]
            heading = [-3.14, 3.14]

    class control( LeggedRobotCfg.control ):
        # PD Drive parameters:
        stiffness = {'collar': 50.0, 'hip': 50.0, 'knee': 30.}  # [N*m/rad]
        damping = {'collar': 2.0, 'hip': 2.0, 'knee': 0.2}  # [N*m*s/rad]
        # action scale: target angle = actionScale * action + defaultAngle
        action_scale = 0.5
        # decimation: Number of control action updates @ sim DT per policy DT
        decimation = 4

    class asset( LeggedRobotCfg.asset ):
        file = '{LEGGED_GYM_ROOT_DIR}/resources/robots/mevius/urdf/mevius.urdf'
        name = "mevius"
        foot_name = 'calf_link'
        terminate_after_contacts_on = ['base_link', 'scapula_link', 'thigh_link']
        flip_visual_attachments = False
        self_collisions = 1 # 1 to disable, 0 to enable...bitwise filter

    class domain_rand:
        randomize_friction = True
        # first
        friction_range = [-0.4, 0.6]
        # second
        # friction_range = [-0.5, 1.0]
        randomize_base_mass = True
        added_mass_range = [-2., 2.]
        push_robots = True
        push_interval_s = 15
        max_push_vel_xy = 1.

    class rewards( LeggedRobotCfg.rewards ):
        soft_dof_pos_limit = 0.8
        soft_dof_vel_limit = 0.8
        soft_torque_limit = 0.9
        max_contact_force = 250.
        only_positive_rewards = False # if true negative total rewards are clipped at zero (avoids early termination problems)
        curriculum = True
        curriculum_offset = 0.01
        curriculum_scale = 1e-7
        class scales( LeggedRobotCfg.rewards.scales ):
            # termination = -200.0
            tracking_lin_vel = 1.0
            tracking_ang_vel = 0.5
            action_rate = -0.1
            orientation = -10.0
            lin_vel_z = -10.0
            ang_vel_xy = -0.1
            torques = -1.0e-3
            feet_air_time = 0.001
            dof_acc = -1.0e-4
            dof_vel = -1.0e-7
            stand_still = -10.0
            dof_pos_limits = -10.0

    class noise:
        add_noise = True
        noise_level = 1.0 # scales other values
        class noise_scales:
            dof_pos = 0.05
            dof_vel = 1.5
            lin_vel = 0.2
            ang_vel = 0.3
            gravity = 0.1
            height_measurements = 0.1


class MeviusFlatCfgPPO( LeggedRobotCfgPPO ):
    seed = 1
    # runner_class_name = 'ActorCriticReccurent'
    # class policy:
    #     # only for 'ActorCriticRecurrent':
    #     rnn_type = 'lstm'
    #     rnn_hidden_size = 512
    #     rnn_num_layers = 1

    runner_class_name = 'OnPolicyRunner'
    class policy( LeggedRobotCfgPPO.policy):
        init_noise_std = 1.0
        actor_hidden_dims = [128, 64, 32]
        critic_hidden_dims = [128, 64, 32]
        activation = 'elu' # can be elu, relu, selu, crelu, lrelu, tanh, sigmoid

    class algorithm( LeggedRobotCfgPPO.algorithm):
        entropy_coef = 0.003
        learning_rate = 1.e-3
        clip_param = 0.2
        schedule = 'adaptive' # could be adaptive, fixed

    class runner( LeggedRobotCfgPPO.runner ):
        run_name = ''
        experiment_name = 'flat_mevius'
        max_iterations = 20000 # number of policy updates

