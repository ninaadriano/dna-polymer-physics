import numpy as np


def func_r(r_1, r_2):
    r_dist = np.linalg.norm(r_1 - r_2)
    multiplier = 1 - 1 / r_dist
    f = multiplier * (r_1 - r_2)
    return f


def DNA_force_extension_simulation(p, k, N_particles, dim, N_steps, initial_positions, delta_t=0.001):
    x = np.random.uniform(0, 1, dim)
    p_vec = p * x
    r = initial_positions
    Std_BF = np.sqrt(2 * delta_t)

    i = 0
    while i < N_steps:
        r[0, :] = r[0, :] - delta_t * (k * func_r(r[0, :], r[1, :]) + p_vec) + Std_BF * np.random.normal(0, 1, dim)
        r[-1, :] = r[-1, :] - delta_t * (k * func_r(r[-1, :], r[-2, :]) - p_vec) + Std_BF * np.random.normal(0, 1, dim)
        for j in range(1, N_particles - 1):
            r[j, :] = r[j, :] - delta_t * (k * func_r(r[j, :], r[j - 1, :]) + k * func_r(r[j, :], r[j + 1,
                                                                                                  :])) + Std_BF * np.random.normal(
                0, 1, dim)
        i += 1
    if p == 0:
        return r

    return x, r


def DNA_radius_gyration(r):
    N_particles = np.shape(r)[0]
    r_com = np.zeros(3)

    for i in range(N_particles):
        r_com += r[i, :]
    r_com = 1 / (N_particles + 1) * r_com
    r_sq_sum = 0

    for i in range(N_particles):
        r_diff = r[i, :] - r_com
        r_diff_norm_sq = (np.linalg.norm(r_diff)) ** 2
        r_sq_sum += r_diff_norm_sq

    r_gyration = np.sqrt(1 / (N_particles + 1) * r_sq_sum)

    return r_gyration


def DNA_extension(p, k, N_particles, dim, N_steps, initial_positions, delta_t=0.001):
    x, r = DNA_force_extension_simulation(p, k, N_particles, dim, N_steps, initial_positions, delta_t)
    chain_extension = np.dot((r[-1, :] - r[0, :]), x)

    return chain_extension


def DNA_extension_mean(N_repeats, p, k, N_particles, dim, N_steps, initial_positions, delta_t=0.001):
    chain_extension_vals = []
    i = 0
    while i < N_repeats:
        chain_extension = DNA_extension(p, k, N_particles, dim, N_steps, initial_positions, delta_t)
        chain_extension_vals.append(chain_extension)
        i += 1

    mean_extension = np.mean(chain_extension_vals)
    return mean_extension


def theoretical_fjc(p_vals, N_particles):
    fjc_extension = []

    for p in p_vals:
        e = N_particles * (1 / np.tanh(p) - 1 / p)
        fjc_extension.append(e)

    return p_vals, fjc_extension


def get_lambdaDNA_data():
    l = 106
    k_b = 1.38
    T = 298
    p_dim = l / (k_b * T)
    e_dim = 1 / (l)

    data = np.genfromtxt("LambdaDNA_force_ext.csv", delimiter=",", skip_header=1)
    p_vals = p_dim * data[:, 1]
    e_vals = e_dim * data[:, 0]

    return p_vals, e_vals


def theoretical_wlc(e_vals, N_particles):
    wlc_forces = []

    for e in e_vals:
        wlc_force = 0.5 * (1 - e / N_particles) ** (-2) - 0.5 * (e / N_particles)
        wlc_forces.append(wlc_force)

    return wlc_forces, e_vals
