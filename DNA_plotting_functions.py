from mpl_toolkits.mplot3d import axes3d
import matplotlib.pyplot as plt
import numpy as np
from DNA_simulation import *
from BD_simulation import *


def polymer_visualisation_3d(r):
    fig = plt.figure()


ax = fig.add_subplot(111, projection='3d')
X = r[:, 0]
Y = r[:, 1]
Z = r[:, 2]
ax.plot3D(X, Y, Z)
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')
plt.title("Polymer chain in 3D")
plt.show()


def plot_force_extension(p_array, e_array):
    plt.plot(e_array, p_array)
    plt.xlabel('e_tilde')
    plt.ylabel('p_tilde')
    plt.show()


def plot_changing_k_val(k_vals, p_vals, N_steps, N_particles, dim, method):
    if method == 'bd':
        r_initialised = (BD_dimensionless_harmonic_motion(dim, N_particles)).T
    else:
        r_initialised = np.genfromtxt('initialised_polymer.csv', delimiter=',')
    r_clean = r_initialised.copy()
    for k in k_vals:
        r_initialised = r_clean.copy()
        e_vals_k = []
        for p in p_vals:
            print(f"k = {k}, Percentage complete {100 * p / p_vals[-1]} %", end="\r")
            x, r_final = DNA_force_extension_simulation(p, k, N_particles, dim, N_steps, r_initialised)
            e = abs(np.linalg.norm(r_final[-1, :] - r_final[0, :]))
            e_vals_k.append(e)
        np.savetxt(f"extension_vals_k_{k}.csv", e_vals_k, delimiter=',')

    for k in k_vals:
        e_vals = np.genfromtxt(f"extension_vals_k_{k}.csv", delimiter=',')
        plt.plot(p_vals, e_vals, label=f'k = {k}')
    plt.title(f"Step number: {N_steps}; Particle number: {N_particles}")
    plt.xlabel('Value of non-dimensional force, p_tilde', size=14)
    plt.ylabel('Extension of the polymer chain', size=14)
    plt.legend(fontsize=12)
    plt.show()


def plot_vs_fjc(p_vals, k_vals, N_steps, N_particles):
    p_vals_fjc, fjc_extension = theoretical_fjc(p_vals, N_particles)
    e_k_min = np.genfromtxt(f"extension_vals_k_{k_vals[0]}.csv", delimiter=',')
    plt.plot(p_vals, e_k_min, color='b', label=f'k = {k_vals[0]}')
    plt.plot(p_vals_fjc, fjc_extension, color='r', label='FJC model')
    plt.title(f"Step number: {N_steps}; Particle number: {N_particles}")
    plt.xlabel('Value of non-dimensional force, p_tilde', size=14)
    plt.ylabel('Extension of the polymer chain', size=14)
    plt.legend(fontsize=12)
    plt.show()

    e_k_max = np.genfromtxt(f"extension_vals_k_{k_vals[-1]}.csv", delimiter=',')
    plt.plot(p_vals, e_k_max, color='b', label=f'k = {k_vals[-1]}')
    plt.plot(p_vals, fjc_extension, color='r', label='FJC model')
    plt.title(f"Step number: {N_steps}; Particle number: {N_particles}")
    plt.xlabel('Value of non-dimensional force, p_tilde', size=14)
    plt.ylabel('Extension of the polymer chain', size=14)
    plt.legend(fontsize=12)
    plt.show()


def plot_changing_N_val(N_particle_list, k, p_vals, dim, N_steps):
    e_vals_particles = []
    for N_particles in N_particle_list:
        r_initialised = (BD_dimensionless_harmonic_motion(dim, N_particles)).T
        e_vals_N = []
        for p in p_vals:
            print(f"N = {N_particles}, Percentage complete {100 * p / p_vals[-1]} %", end="\r")
            x, r_final = DNA_force_extension_simulation(p, k, N_particles, dim, N_steps, r_initialised)
            e = abs(np.linalg.norm(r_final[-1, :] - r_final[0, :]))
            e_vals_N.append(e)
        e_vals_particles.append(e_vals_N)

    np.savetxt("extension_vals_particles.csv", e_vals_particles, delimiter=",")
    all_e_vals_particles = np.genfromtxt("extension_vals_particles.csv", delimiter=",")

    for i in range(len(N_particle_list)):
        plt.plot(p_vals, all_e_vals_particles[i], label=f'N = {N_particle_list[i]}')
    plt.title(f"Step number: {N_steps}; k: {k}")
    plt.xlabel('Value of non-dimensional force, p_tilde', size=14)
    plt.ylabel('Extension of the polymer chain', size=14)
    plt.legend(fontsize=12)
    plt.show()

def plot_wlc_fjc_lambdaDNA(N_particles=308):
    p_vals_DNA, DNA_extension = get_lambdaDNA_data()
    p_vals = np.linspace(0.01, p_vals_DNA[-1], len(p_vals_DNA)).tolist()
    p_vals_fjc, fjc_extension = theoretical_fjc(p_vals, N_particles)
    e_vals = np.linspace(0.01, DNA_extension[-1], len(DNA_extension)).tolist()
    p_vals_wlc, wlc_extension = theoretical_wlc(e_vals, N_particles)
    plt.plot(p_vals_DNA, DNA_extension, color='r', label='LambdaDNA data')
    plt.plot(p_vals_fjc, fjc_extension, color='g', label='FJC model')
    plt.plot(p_vals_wlc, wlc_extension, color='b', label='WLC model')
    plt.title(f"Particle number: {N_particles}")
    plt.xlabel('Value of non-dimensional force, p_tilde', size=14)
    plt.ylabel('Extension of the polymer chain', size=14)
    plt.legend(fontsize=12)
    plt.show()
