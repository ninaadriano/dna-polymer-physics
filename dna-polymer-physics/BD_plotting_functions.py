import numpy as np
import matplotlib.pyplot as plt
from BD_simulation import *


def plot_brownian_motion(dim, N_steps, num_particles, delta_t=0.001):
    i = 0
    while i < num_particles:
        r = BD_dimensionless_harmonic_motion(dim, N_steps, delta_t)
        if dim == 1:
            t = np.arange(0, (N_steps + 1) * delta_t, delta_t)
            x = r[0, :]
            plt.plot(t, x)
            plt.show()
        i += 1


def plot_msd_time(dim, N_particles, N_steps, delta_t=0.001):
    N_step_array, msd_array = BD_msd(dim, N_particles, N_steps, delta_t)
    time_array = delta_t * N_step_array
    theoretical_msd = []
    for i in range(len(time_array)):
        theoretical_msd.append(dim * (1 - np.exp(-2 * time_array[i])))
    plt.plot(N_step_array, msd_array, label='Simulated msd')
    plt.plot(theoretical_msd, label='Theoretical msd', color='r')
    plt.title(f'Dimension = {dim}', size=14)
    plt.xlabel(f'Number of steps, step size = {delta_t}', size=14)
    plt.ylabel('Mean squared displacement', size=14)
    plt.xticks(size=12)
    plt.yticks(size=12)
    plt.legend(fontsize=12)
    plt.show()

    error_array = abs(np.array(theoretical_msd) - np.array(msd_array))
    plt.hist(error_array, edgecolor='black', color='white')
    plt.title(f'Dimension = {dim}', size=14)
    plt.xlabel('Error', size=14)
    plt.ylabel('Frequency', size=14)
    plt.xticks(size=12)
    plt.yticks(size=12)
    plt.show()


def plot_error_particles(max_dim, N_particle_array, N_steps, delta_t=0.001):
    mean_error_1D = []
    mean_error_2D = []
    mean_error_3D = []
    dim = 1
    while dim <= max_dim:
        print('Dimension = ', dim)
        for N_particles in N_particle_array:
            print('N_particles = ', N_particles)
            N_step_array, msd_array = BD_msd(dim, N_particles, N_steps, delta_t)
            time_array = delta_t * N_step_array
            theoretical_msd = []
            for i in range(len(time_array)):
                theoretical_msd.append(dim * (1 - np.exp(-2 * time_array[i])))
            error_array = abs(np.array(theoretical_msd) - np.array(msd_array))
            mean_error = np.mean(error_array)
            if dim == 1:
                mean_error_1D.append(mean_error)
            elif dim == 2:
                mean_error_2D.append(mean_error)
            elif dim == 3:
                mean_error_3D.append(mean_error)
        dim += 1

    plt.plot(N_particle_array, mean_error_1D, color='g', label='1D')
    plt.plot(N_particle_array, mean_error_2D, color='orange', label='2D')
    plt.plot(N_particle_array, mean_error_3D, color='purple', label="3D")
    plt.xlabel('Number of particles', size=14)
    plt.ylabel('Mean error', size=14)
    plt.title('Mean error of MSD simulation compared to theory')
    plt.legend(fontsize=12)
    plt.show()

    return mean_error_1D, mean_error_2D, mean_error_3D
