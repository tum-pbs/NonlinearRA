import argparse
import os
import re

import h5py
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from scipy.linalg import cholesky, svd


# Fourier weights
def fourier_weights(N: int):
    w = (2 * np.pi / N) * np.ones(N)
    return w

# Chebyshev weights
def cheb_weights(N: int):
    
    θ = np.pi * np.arange(N) / (N - 1)
    x = np.cos(θ)
    w = np.zeros(N)
    v = np.ones(N - 2)

    if N % 2 != 0:  # N odd
        w[0] = 1 / ((N - 1)**2 - 1)
        w[-1] = w[0]
        for k in range(1, (N - 1) // 2):
            v -= 2 * np.cos(2 * k * θ[1:-1]) / (4 * k**2 - 1)
        v -= np.cos((N - 1) * θ[1:-1]) / ((N - 1)**2 - 1)
    else:  # N even
        w[0] = 1 / (N - 1)**2
        w[-1] = w[0]
        for k in range(1, (N - 2) // 2 + 1):
            v -= 2 * np.cos(2 * k * θ[1:-1]) / (4 * k**2 - 1)
    
    w[1:-1] = 2 * v / (N - 1)
    return w

def energy_weight_2d(Nx, Ny, Nd):
    Qx = np.diag(fourier_weights(Nx))
    Qy = np.diag(cheb_weights(Ny))
    I_nd = np.eye(Nd)
    Q = np.kron(I_nd, np.kron(Qy, Qx)) / (4 * np.pi)
    return Q

def adj(A, Q):

    return np.linalg.solve(Q, A.conj().T @ Q)

def normalize_basis(V, Q):

    V_normed = V.copy()
    for i in range(V.shape[1]):
        norm_val = np.sqrt(np.vdot(V[:, i], Q @ V[:, i]))
        V_normed[:, i] = V[:, i] / norm_val
    return V_normed

def eigen_dual(A, Q, log_sort=False):

    Aadj = adj(A, Q)
    
    if log_sort:
        eigvals, V = np.linalg.eig(A)
        sort_idx = np.argsort(np.imag(np.log(eigvals)))
        eigvals = eigvals[sort_idx]
        V = V[:, sort_idx]

        eigvals_adj, W = np.linalg.eig(Aadj)
        sort_idx_adj = np.argsort(-np.imag(np.log(eigvals_adj)))
        eigvals_adj = eigvals_adj[sort_idx_adj]
        W = W[:, sort_idx_adj]
        
        p = np.argsort(-np.real(np.log(eigvals)))
        p_bar = np.argsort(-np.real(np.log(eigvals_adj)))
    else:
        eigvals, V = np.linalg.eig(A)
        sort_idx = np.argsort(np.imag(eigvals))
        eigvals = eigvals[sort_idx]
        V = V[:, sort_idx]
        
        eigvals_adj, W = np.linalg.eig(Aadj)
        sort_idx_adj = np.argsort(-np.imag(eigvals_adj))
        eigvals_adj = eigvals_adj[sort_idx_adj]
        W = W[:, sort_idx_adj]
        
        p = np.argsort(-np.real(eigvals))
        p_bar = np.argsort(-np.real(eigvals_adj))

    # Reorder eigenvalues and eigenvectors according to the new indices
    eigvals = eigvals[p]
    V = V[:, p]
    eigvals_adj = eigvals_adj[p_bar]
    W = W[:, p_bar]

    # basis normalization: ensure each basis vector is normalized with respect to Q
    V = normalize_basis(V, Q)
    W = normalize_basis(W, Q)

    # bi-orthogonal normalization: ensure each pair (W[:,i], V[:,i]) satisfies W^H Q V = 1
    for i in range(V.shape[1]):
        V[:, i] = V[:, i] / (W[:, i].conj().T @ Q @ V[:, i])
    
    return eigvals, V, W

def opt_gain(A, Q, wspan, m):

    F = factorize_weights(Q) 
    Fi = np.linalg.inv(F)

    R = []
    I = np.eye(A.shape[0])
    for ω in wspan:
        M = np.linalg.inv(-1j * ω * I - A)  # Solve (-iωI - A)
        singular_values = svd(F @ M @ Fi, compute_uv=False)
        R.append(singular_values[:m]**2)
    R = np.array(R).T  # Convert list of arrays to a 2D matrix
    return R

def opt_gain_dmd(lmbda, V, Q, wspan, m):
    A_tilde = np.diag(lmbda)
    Q_tilde = V.conj().T @ Q @ V
    R = opt_gain(A_tilde, Q_tilde, wspan, m) 
    return R

def opt_forcing(A, Q, ω):

    F = factorize_weights(Q) 
    Fi = np.linalg.inv(F)
    I = np.eye(A.shape[0])
    M = np.linalg.inv(-1j * ω * I - A)
    U, singular_values, Vh = svd(F @ M @ Fi, full_matrices=False)
    Vh = Vh.conj().T
    Ψ = Fi @ U
    Φ = Fi @ Vh
    return Ψ, singular_values, Φ

def opt_forcing_dmd(lmbda, V, Q, ω):
    A_tilde = np.diag(lmbda)
    Q_tilde = V.conj().T @ Q @ V
    Ψ, σ, Φ = opt_forcing(A_tilde, Q_tilde, ω)
    Ψ = V @ Ψ
    Φ = V @ Φ
    return Ψ, σ, Φ
  
def factorize_weights(Q):
    # Check if Q is diagonal
    if np.allclose(Q, np.diag(np.diagonal(Q))):  # Equivalent to checking typeof(Q) <: Diagonal in Julia
        F = np.sqrt(Q.diagonal())  # Element-wise square root for diagonal elements
        F = np.diag(F)  # Convert back to a diagonal matrix
    else:
        # Hermitian part and Cholesky factorization
        Q_hermitian = np.triu(Q) + np.triu(Q, k=1).T.conj()  # Ensure Hermitian symmetry
        F = cholesky(Q_hermitian, lower=False)  # Use upper-triangular Cholesky factor
    return F

def energy(u):
    Q = energy_weight_2d(u.shape[1], u.shape[0], u.shape[2])
    e = np.sqrt(u.flatten(order='F').reshape(-1, 1).T @ Q @ u.flatten(order='F').reshape(-1, 1))
    return e

def load_mode(path):
    with h5py.File(path, 'r') as f:
        phi = np.array(f['phi'])
        psi = np.array(f['psi'])
    Nd, Nx, Ny = phi.shape
    phi = np.transpose(phi, (2, 1, 0))  # Transpose to (Ny, Nx, Nd)
    psi = np.transpose(psi, (2, 1, 0))  
    phi = phi / energy(phi)
    psi = psi / energy(psi)

    return phi, psi

def align_mode(Φ, Ψ, ϕtrue, ψtrue, l, j, Q, Ny, Nx):
    # extract the l-th mode from the j-th group
    start = l * (j - 1)
    end = l * j
    ϕdata = np.real(Φ[:, start:end])  # shape: (n, l)
    ψdata = np.real(Ψ[:, start:end])  # shape: (n, l)

    ϕtrue_flat = ϕtrue.ravel(order='F')
    ψtrue_flat = ψtrue.ravel(order='F')

    # least squares alignment
    Ac = ϕdata.T @ Q @ ϕdata
    bc = ϕdata.T @ Q @ ϕtrue_flat
    cϕ = np.linalg.solve(Ac, bc)
    ϕdata = ϕdata @ cϕ

    Ac = ψdata.T @ Q @ ψdata
    bc = ψdata.T @ Q @ ψtrue_flat
    cψ = np.linalg.solve(Ac, bc)
    ψdata = ψdata @ cψ

    ϕdata = ϕdata.reshape((Ny, Nx, 2), order='F')
    ψdata = ψdata.reshape((Ny, Nx, 2), order='F')

    return ϕdata, ψdata

if __name__ == "__main__":

    idim = 32
    jdim = 64
    dt = 0.5
    x_1d = np.arange(idim) * 2 * np.pi / idim
    y_1d = np.cos(np.pi * np.arange(jdim) / (jdim - 1))
    x, y = np.meshgrid(x_1d, y_1d, indexing="ij")

    Q = energy_weight_2d(idim, jdim, 2)
    F = factorize_weights(Q)
    Fi = np.linalg.inv(F)

    ω = 0.313
    wspan = np.arange(0, 0.55, 0.05)  # Frequency range

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

    if os.path.exists(save_path + '/resolvent_results.npz'):
        print("Loading results...")
        lmbda, U, singular_values, Vt, R_dmd_sqrt = np.load(save_path + '/resolvent_results.npz').values()
    else:
        jacobian_matrix = np.load(save_path + '/ensemble_jacobian.npy')

        ρ, W, Wadj = eigen_dual(jacobian_matrix, Q, log_sort=True)
        for i in range(jacobian_matrix.shape[0]):
            W[:, i] = W[:, i] / np.sqrt(W[:, i].conj().T @ W[:, i])
            Wadj[:, i] = Wadj[:, i] / np.sqrt(Wadj[:, i].conj().T @ Wadj[:, i])
            W[:, i] = W[:, i] / (Wadj[:, i].conj().T @ W[:, i])
        lmbda = np.log(ρ) / dt
        V = W
        W = Wadj
        r_project = 30

        # response and forcing modes
        U, singular_values, Vt = opt_forcing_dmd(lmbda[0:r_project],V[:,0:r_project],Q,ω)

        # gain curve
        R_dmd = opt_gain_dmd(lmbda[0:r_project], V[:,0:r_project], Q, wspan, 4)
        R_dmd_sqrt = np.sqrt(R_dmd.T)

        np.savez(save_path + '/resolvent_results.npz', lmbda=lmbda, U=U, singular_values=singular_values, Vt=Vt, R_dmd_sqrt=R_dmd_sqrt)

    # Aligning the modes with the truth data
    # refer to https://github.com/ben-herrmann/Data-driven-resolvent
    Phi_true, Psi_true = load_mode('./resolvent_truth.mat')
    Vt, U = align_mode(Vt, U, Phi_true, Psi_true, 30, 1, Q, jdim, idim)

    fig, ax = plt.subplots(2, 2, figsize=(13, 6))
    colorbars = []
    
    for i in range(2): # Response Modes

        vis_output = U[:,:,i].T
        lim = np.max(np.abs(vis_output))

        contour = ax[0, i].contourf(x, y, vis_output, levels=np.linspace(-lim,lim,15))
        
        ax[0, i].set_xlim(0, 2*np.pi); ax[0, i].set_ylim(-1, 1)
        ax[0, i].set_xticks([]); ax[0, i].set_yticks([])
        ax[0, i].set_aspect('equal')

        cbar = fig.colorbar(contour, ax=ax[0, i], orientation='horizontal')
        cbar.set_ticks(np.linspace(-lim, lim, 5))
        colorbars.append(cbar)

    for i in range(2): # Forcing Modes

        vis_output = Vt[:,:,i].T
        lim = np.max(np.abs(vis_output))

        contour = ax[1, i].contourf(x, y, vis_output, levels=np.linspace(-lim,lim,15))
        
        ax[1, i].set_xlim(0, 2*np.pi); ax[1, i].set_ylim(-1, 1)
        ax[1, i].set_xticks([]); ax[1, i].set_yticks([])
        ax[1, i].set_aspect('equal')
        
        cbar = fig.colorbar(contour, ax=ax[1, i], orientation='horizontal')
        cbar.set_ticks(np.linspace(-lim, lim, 5))
        colorbars.append(cbar)

    plt.tight_layout()
    plt.savefig(save_path + '/singular_vectors.png', dpi=300)

    plt.figure(figsize=(6, 4))
    colors = ['r', 'g', 'b', 'c']
    for i in range(R_dmd_sqrt.shape[1]):
        plt.plot(wspan, R_dmd_sqrt[:, i], '-', color=colors[i], linewidth=2, alpha=0.8)

    plt.xlim(0, 0.5)
    plt.grid(False)
    plt.xlabel(r"$\omega$")
    plt.ylabel(r"$\sigma_j$")

    line_legend = [
        Line2D([0], [0], color='k', linestyle='-', linewidth=2, alpha=0.8, label='NN Results')
    ]

    color_legend = [
        Line2D([0], [0], color=colors[0], lw=4, label='Gain Distribution Mode 1'),
        Line2D([0], [0], color=colors[1], lw=4, label='Gain Distribution Mode 2'),
        Line2D([0], [0], color=colors[2], lw=4, label='Gain Distribution Mode 3'),
        Line2D([0], [0], color=colors[3], lw=4, label='Gain Distribution Mode 4')
    ]

    legend = plt.legend(handles=color_legend, loc='upper left')
    plt.gca().add_artist(legend)

    plt.tight_layout()
    plt.savefig(save_path + '/gain_distribution.png', dpi=300)
