import argparse
import os
import re

import equinox as eqx
import h5py
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import pdequinox as pdeqx
from mpl_toolkits.axes_grid1 import ImageGrid
from scipy.sparse.linalg import eigs

jax.config.update('jax_platform_name', 'cpu')

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--savedir', type=str, default='../results/expo5_level3_batch16_roll10_nrun2', help='directory to save the model')
    args = parser.parse_args()

    save_path = args.savedir
    pattern = r"expo(\d+)_level(\d+)_batch(\d+)_roll(\d+)_nrun(\d+)"
    match = re.search(pattern, save_path)

    if match:
        expo, level, batch_size, roll, n_run = map(int, match.groups())  # 转换为整数
        print(f"Loaded parameters: expo={expo}, level={level}, batch_size={batch_size}, n_rollout={roll}, n_run={n_run}")
    else:
        raise ValueError(f"No match found: {save_path}.")

    idim = 32; jdim = 64; dt = 0.5
    x_1d = jnp.arange(idim) * 2 * jnp.pi / idim
    y_1d = jnp.cos(jnp.pi * jnp.arange(jdim) / (jdim - 1))
    x, y = jnp.meshgrid(x_1d, y_1d, indexing="ij")

    norm_dynamics = jnp.load("./norm_dynamics.npz")
    max_uval = norm_dynamics['u_max']
    max_vval = norm_dynamics['v_max']
 
    jacobian_matrix_list = []

    for n in range(n_run):

        model_path = save_path + "/run%d" % n + "/channel_UNet.eqx"
        emulator = pdeqx.arch.ClassicUNet(
                2,
                2,
                2,
                hidden_channels=2**expo,
                num_levels=level,
                activation=jax.nn.relu,
                key=jax.random.PRNGKey(0),
                boundary_mode="dirichlet",
            )

        sup_1_trained_emu = eqx.tree_deserialise_leaves(model_path, emulator)

        def wrapped_fun(input_tensor):
            input_tensor = input_tensor.reshape((2, idim, jdim))
            input_tensor = input_tensor.at[0].set(input_tensor[0] / max_uval)
            input_tensor = input_tensor.at[1].set(input_tensor[1] / max_vval)
            output_tensor = sup_1_trained_emu(input_tensor)
            output_tensor = output_tensor.at[0].set(output_tensor[0] * max_uval)
            output_tensor = output_tensor.at[1].set(output_tensor[1] * max_vval)
            return jnp.ravel(output_tensor)

        input_stat = jnp.zeros((2, idim, jdim))
        jacobian = jax.jacobian(wrapped_fun)(input_stat.flatten())
        jacobian = np.array(jacobian)
        jacobian_matrix_list.append(jacobian)

        np.save(save_path + "/run%d" % n + '/jacobian.npy', jacobian)

    jacobian_matrix = np.mean(jacobian_matrix_list, axis=0)
    np.save(save_path + '/ensemble_jacobian.npy', jacobian_matrix)

    k = 50  # number of eigenvalues and eigenvectors to be calculated
    eigenvalues, eigenvectors = eigs(jacobian_matrix, k=k, which='LM') 

    A_eigenvalues = np.log(eigenvalues)/dt
    real_dist = np.abs(np.real(A_eigenvalues))
    sorted_idx = np.argsort(real_dist)
    A_eigenvalues = A_eigenvalues[sorted_idx]

    # reserve one of the conjugate eigenvalues
    neigs = 6
    eigs_idx = []
    i = 0
    while i < len(eigenvalues) and len(eigs_idx) < neigs:
        if i + 1 < len(eigenvalues) and np.isclose(eigenvalues[i], np.conj(eigenvalues[i + 1])):
            if np.imag(eigenvalues[i]) < 0:
                eigs_idx.append(i)
            else:
                eigs_idx.append(i + 1)
            i += 2
        else:
            eigs_idx.append(i)
            i += 1

    fig, ax = plt.subplots(1,2,figsize=(12, 6))

    with h5py.File("./stability_truth.mat", "r") as f:
        data = np.array(f["eigs"])

    data = np.hstack((data, data.conj()))
    ax[0].scatter(np.imag(A_eigenvalues), np.real(A_eigenvalues), edgecolors='black', facecolors='none', alpha=0.75)
    ax[0].scatter(np.imag(data), np.real(data), edgecolors='red', facecolors='red', alpha=0.75, marker='o', label='Ground Truth')
    ax[0].legend(loc='upper right', fontsize=10)
    xtemp = np.arange(-10, 10, 0.1)
    ax[0].fill_between(xtemp, 0, 100, color='red', alpha=0.25)
    ax[0].set_xlim(-1, 1)
    bound = max(np.abs(min(np.real(A_eigenvalues))-0.2),max(np.real(A_eigenvalues))+0.2)
    ax[0].set_ylim(-bound, bound)
    ax[0].set_xlabel(r'$f=\omega/2\pi$', fontsize=12)
    ax[0].set_ylabel(r'$\sigma$', fontsize=12)
    for i in range(neigs): 
        ax[0].text(np.imag(A_eigenvalues[eigs_idx[i]]) + 0.03, np.real(A_eigenvalues[eigs_idx[i]]), 'mode '+str(i), fontsize=10, color='black')
    ax[0].axvline(x=0,linestyle='--',color='black',alpha=0.5)

    ax[1].scatter(np.real(eigenvalues), np.imag(eigenvalues), edgecolors='black', facecolors='none', alpha=0.75)
    ax[1].scatter(np.real(np.exp(data*dt)), np.imag(np.exp(data*dt)), edgecolors='red', facecolors='red', alpha=0.75, marker='o', label='Ground Truth')
    ax[1].legend(loc='upper right', fontsize=10)
    theta = np.linspace(0.0,2.*np.pi,400);
    ax[1].plot(np.cos(theta),np.sin(theta), lw=0.2, color='r', ls='-')
    ax[1].axhline(y=0., xmin=0, xmax=1, lw=0.2, color='k', ls=':')
    ax[1].axvline(x=0., ymin=0, ymax=1, lw=0.2, color='k', ls=':')
    ax[1].set_aspect('equal', 'box')
    ax[1].set_xlabel(r'$\Re (\mu)$')
    ax[1].set_ylabel(r'$\Im (\mu)$')

    fig.savefig(save_path + "/eigenspectrum.png")

    fig = plt.figure(figsize=(9,neigs),dpi=100)
    grid = ImageGrid(fig, 111, nrows_ncols=(neigs,2), axes_pad=0.05, share_all=True)
    axs = grid.axes_column
    j = 0

    for icolumn in eigs_idx:
        for i in range(0, 2):
            vis_output = eigenvectors[:, icolumn].reshape([1, 2, idim, jdim])
            vis_output = np.real(vis_output[0, i, :, :])
            im = axs[i][j].contourf(x, y, vis_output)

            axs[i][j].set_xticks([])
            axs[i][j].set_yticks([])
            axs[i][j].set_aspect('equal')
            axs[i][j].set_xlim(0, 2*np.pi); axs[i][j].set_ylim(-1, 1)

        j+=1

    fig.savefig (save_path + "/eigenvectors.png", dpi=300)