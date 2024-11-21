import matplotlib.pyplot as plt
import numpy as np


def BD_free_motion(dim, N_steps):  # dim: dimension # N_steps: number of time steps

    delta_t = 0.001  # Time step
    diff = 1.0000  # Diffusion coefficient
    Std_BF = np.sqrt(2 * diff * delta_t)
    r = np.zeros((dim, N_steps + 1))
    for i in np.arange(0, N_steps):
        # Standard deviation of Brownian force
        # Initial array of positions # Brownian Dynamics
        r[:, i + 1] = r[:, i] + Std_BF * np.random.normal(0, 1, dim)
    return r  # Result


def BD_dimensionless_harmonic_motion(dim, N_steps, delta_t=0.001):  # N_steps: number of time steps
    # dim: dimension Std_BF = np.sqrt(2*delta_t) #TODO: Check physical interpretation and change name
    r = np.zeros((dim, N_steps + 1))  # Initial array of positions
    for i in np.arange(0,N_steps): # Brownian Dynamics
        r[:, i + 1] = (1 - delta_t) * r[:, i] + Std_BF * np.random.normal(0, 1, dim)
    return r  # Result


def BD_msd(dim, N_particles, N_steps, delta_t=0.001):
    sq_disp_arrays_T = []
    j = 0
    while j < N_particles:
        r = BD_dimensionless_harmonic_motion(dim, N_steps, delta_t)
    sq_disp_array = []
    for i in range(N_steps):
        r_disp = np.linalg.norm(r[:, i])
    sq_disp_array.append(r_disp ** 2)
    sq_disp_arrays_T.append(sq_disp_array)
    j += 1
    N_step_array = np.arange(0, N_steps, 1)
    sq_disp_arrays = np.array(sq_disp_arrays_T).T
    msd_array = []
    for k in range(np.shape(sq_disp_arrays)[0]):
        msd_array.append(np.mean(sq_disp_arrays[k, :]))
    return N_step_array, msd_array
