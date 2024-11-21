import numpy as np
import matplotlib.pyplot as plt
from BD_plotting_functions import *

plot_brownian_motion(1, 10000, 1)

dim = 1
N_particles = 1000
N_steps = 5000

while dim < 4:
    plot_msd_time(dim, N_particles, N_steps, delta_t=0.001)
    dim += 1

N_particle_array = [1, 10, 50, 100, 250, 500, 750, 1000, 1500, 2000, 5000]
mean_error_1D, mean_error_2D, mean_error_3D = plot_error_particles(3, N_particle_array, 2500)
