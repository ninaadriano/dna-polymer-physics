from DNA_simulation import *
from DNA_plotting_functions import *
from BD_simulation import *
import numpy as np

N_particles = 100
dim = 3

r_initialised = (BD_dimensionless_harmonic_motion(dim, N_particles)).T
polymer_visualisation_3d(r_initialised)

r_g = DNA_radius_gyration(r_initialised)
r_g_fjc = np.sqrt(N_particles / 6)
print(r_g, r_g_fjc)

i = 0
while r_g < r_g_fjc:
    r_initialised = DNA_force_extension_simulation(0, 1, N_particles, dim, 1, r_initialised)
    chain_length = abs(np.linalg.norm(r_initialised[-1, :] - r_initialised[0, :]))
    r_g = DNA_radius_gyration(r_initialised)
    i += 1

np.savetxt("initialised_polymer.csv", r_initialised, delimiter=",")
polymer_visualisation_3d(r_initialised)

p_vals = np.arange(0.001, 101, 1)
k_vals = [1, 10, 100, 1000]
N_steps = 1000

plot_changing_k_val(k_vals, p_vals, N_steps, N_particles, dim, 'bd')
plot_vs_fjc(p_vals, k_vals, N_steps, N_particles)

plot_changing_k_val(k_vals, p_vals, N_steps, N_particles, dim, 'rg')
plot_vs_fjc(p_vals, k_vals, N_steps, N_particles)

k = 100
N_particle_list = [10, 100, 1000]
p_vals_particles = np.arange(1, 101, 3)
plot_changing_N_val(N_particle_list, k, p_vals_particles, dim, N_steps)

plot_wlc_fjc_lambdaDNA()
