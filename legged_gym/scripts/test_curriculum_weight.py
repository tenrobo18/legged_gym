#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt

num_steps_per_env = 48
max_iterations = 40000
curriculum_offset = 0.01
curriculum_decay = 0.99997

curriculum_weights = []
curriculum_weight = curriculum_offset
for i in range( max_iterations):
    curriculum_weights.append(curriculum_weight)
    for j in range(num_steps_per_env):
        curriculum_weight = pow(curriculum_weight, curriculum_decay)

plt.plot(curriculum_weights)
plt.xlabel('Iterations')
plt.ylabel('Curriculum Weight')
plt.show()
