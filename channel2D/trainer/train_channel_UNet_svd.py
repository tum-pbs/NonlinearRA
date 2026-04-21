import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
import optax
import equinox as eqx
import trainax as tx
import pdequinox as pdeqx
import os
import imageio
import time
import argparse

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Train a PDE emulator using UNet.")
    parser.add_argument('--n_rollout', type=int, default=50, help='rollout steps')
    parser.add_argument('--n_iterations', type=int, default=500000, help='number of training iterations')
    parser.add_argument('--expo', type=int, default=5, help='channel exponent for network size')
    parser.add_argument('--batch_size', type=int, default=64, help='batch size')
    #parser.add_argument('--load_model', type=int, default=0, help='load pre-trained model from previous rollout')
    parser.add_argument('--n_runs', type=int, default=5, help='number of runs')
    parser.add_argument('--gpu', type=str, default='0', help='GPU id to use')
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    dataDir = '../dataset/'
    dataDirList = [os.path.join(dataDir, item) for item in sorted(os.listdir(dataDir))]

    trjs = []
    for myDir in dataDirList:
        files = sorted(os.listdir(myDir))

        trj = []
        for i in range(len(files)):
            data = np.load(os.path.join(myDir, files[i]))
            trj.append(data['data'])

        trjs.append(trj)

    trjs = np.array(trjs, dtype=np.float64)
    print(trjs.shape)
    # max_uval = jnp.max(trjs[:, :, 0, :, :])
    # max_vval = jnp.max(trjs[:, :, 1, :, :])
    # trjs = trjs.at[:, :, 0, :, :].set(trjs[:, :, 0, :, :] / max_uval)
    # trjs = trjs.at[:, :, 1, :, :].set(trjs[:, :, 1, :, :] / max_vval)
    # print("Max u value:", max_uval, "Max v value:", max_vval)
    # jnp.savez('norm_dynamics.npz', u_max=max_uval, v_max=max_vval)

    # SVD for dimension reduction
    if os.path.exists("./reduced_dynamics.npz"):
        reduced_dynamics = np.load("./reduced_dynamics.npz")
        U = reduced_dynamics['U']
        sig = reduced_dynamics['sig']
        Vh = reduced_dynamics['Vh']
    else:
        U, sig, Vh = np.linalg.svd(trjs.reshape(trjs.shape[0]*trjs.shape[1], -1), full_matrices=False)

        # energy = np.cumsum(sig**2) / np.sum(sig**2)
        # plt.figure(figsize=(6,4))
        # plt.plot(energy, marker='o')
        # plt.xlabel("Number of modes r")
        # plt.ylabel("Energy ratio")
        # plt.title("SVD Energy Spectrum")
        # plt.grid(True)
        # plt.tight_layout()
        # plt.show()

    r = 20
    trjs_reduced = U[:, :r] @ np.diag(sig[:r])
    trjs_reconstructed = (trjs_reduced @ Vh[:r, :]).reshape(trjs.shape)
    trjs_reduced = trjs_reduced.reshape(trjs.shape[0],trjs.shape[1], -1)
    trjs_reduced = np.expand_dims(trjs_reduced, axis=2)  # add channel dimension back
    
    # normalization
    max_val = np.max(trjs_reduced, axis=(0, 1, 2))
    min_val = np.min(trjs_reduced, axis=(0, 1, 2))
    trjs_reduced_norm = 2 * ((trjs_reduced - min_val) / (max_val - min_val)) - 1
    np.savez('reduced_dynamics.npz', trjs_reduced_norm=trjs_reduced_norm, max_val=max_val, min_val=min_val, U=U, sig=sig, Vh=Vh)
    trjs_reduced_norm = jnp.array(trjs_reduced_norm, dtype=jnp.float32)
    
    # Comparison of original and reconstruction
    # plt.figure(figsize=(12, 6))
    # plt.subplot(1, 3, 1)
    # plt.title("Original")
    # plt.imshow(trjs[0, 0, 0, :, :], aspect='auto')
    # plt.colorbar()
    # plt.subplot(1, 3, 2)
    # plt.title("Reconstructed")
    # plt.imshow(trjs_reconstructed[0, 0, 0, :, :], aspect='auto')
    # plt.colorbar()
    # plt.subplot(1, 3, 3)
    # plt.imshow(jnp.abs(trjs[0, 0, 0, :, :] - trjs_reconstructed[0, 0, 0, :, :]), aspect='auto')
    # plt.colorbar()
    # plt.show()

    idim = 32; jdim = 64
    x_1d = np.arange(idim) * 2 * np.pi / idim
    y_1d = np.cos(np.pi * np.arange(jdim) / (jdim - 1))
    x, y = np.meshgrid(x_1d, y_1d, indexing="ij")

    save_path = "../results/expo%d_batch%d_roll%d_nrun%d_svd/" % (args.expo, args.batch_size, args.n_rollout, args.n_runs)

    for i in range(args.n_runs):
        os.makedirs(save_path + "run%d" % i, exist_ok=True)
        model_path = save_path + "run%d" % i + "/channel_UNet.eqx"
        
        if os.path.exists(model_path):
            print("Model already exists for run %d, skipping..." % i)
            continue
        else:
            print("Model file not found, initializing a new model.")
            emulator = pdeqx.arch.MLP(
                num_spatial_dims=1,
                in_channels=1,
                out_channels=1,
                num_points=r,
                width_size=128,
                depth=4,
                activation=jax.nn.gelu,
                key=jax.random.PRNGKey(i)
            )

            start = time.time()
            sup_1_trainer = tx.trainer.SupervisedTrainer(
                trjs_reduced_norm,
                optimizer=optax.adam(
                    optax.warmup_cosine_decay_schedule(0.0, 3e-4, 3000, args.n_iterations)
                ),
                num_training_steps=args.n_iterations,
                batch_size=args.batch_size,
                num_rollout_steps=args.n_rollout,  # one-step supervised training (would have been the default)
            )

            sup_1_trained_emu, sup_1_loss_history = sup_1_trainer(emulator, jax.random.PRNGKey(42 + i))
            end = time.time()
            print(f"Training completed in {end - start:.2f} seconds.")
            plt.figure(figsize=(10, 5))
            plt.semilogy(sup_1_loss_history)
            plt.xlabel("Training step")
            plt.ylabel("Train Loss")
            plt.savefig(save_path + "/run%d" % i + "/loss_channel_UNet.png")
            plt.close()

            eqx.tree_serialise_leaves(model_path, sup_1_trained_emu)
