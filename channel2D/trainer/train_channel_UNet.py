import argparse
import os
import time

import equinox as eqx
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import optax
import pdequinox as pdeqx
import trainax as tx

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Train a PDE emulator using UNet.")
    parser.add_argument('--n_rollout', type=int, default=10, help='rollout steps')
    parser.add_argument('--n_iterations', type=int, default=500000, help='number of training iterations')
    parser.add_argument('--expo', type=int, default=5, help='channel exponent for network size')
    parser.add_argument('--level', type=int, default=3, help='number of levels of Unet')
    parser.add_argument('--batch_size', type=int, default=64, help='batch size')
    #parser.add_argument('--load_model', type=int, default=0, help='load pre-trained model from previous rollout')
    parser.add_argument('--n_runs', type=int, default=2, help='number of runs')
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

    max_uval = np.max(trjs[:, :, 0, :, :])
    max_vval = np.max(trjs[:, :, 1, :, :])
    trjs[:, :, 0, :, :] = trjs[:, :, 0, :, :] / max_uval
    trjs[:, :, 1, :, :] = trjs[:, :, 1, :, :] / max_vval
    print("Max u value:", max_uval, "Max v value:", max_vval)
    np.savez('norm_dynamics.npz', u_max=max_uval, v_max=max_vval)
    trjs = jnp.array(trjs, dtype=jnp.float32)

    idim = 32; jdim = 64
    x_1d = np.arange(idim) * 2 * np.pi / idim
    y_1d = np.cos(np.pi * np.arange(jdim) / (jdim - 1))
    x, y = np.meshgrid(x_1d, y_1d, indexing="ij")

    save_path = "../results/expo%d_level%d_batch%d_roll%d_nrun%d/" % (args.expo, args.level, args.batch_size, args.n_rollout, args.n_runs)

    for i in range(args.n_runs):
        os.makedirs(save_path + "run%d" % i, exist_ok=True)
        model_path = save_path + "run%d" % i + "/channel_UNet.eqx"
        
        if os.path.exists(model_path):
            print("Model already exists for run %d, skipping..." % i)
            continue
        else:
            print("Model file not found, initializing a new model.")
            emulator = pdeqx.arch.ClassicUNet(
                2,
                2,
                2,
                hidden_channels=2**args.expo,
                num_levels=args.level,
                activation=jax.nn.relu,
                key=jax.random.PRNGKey(0),
                boundary_mode="dirichlet",
            )

            start = time.time()
            sup_1_trainer = tx.trainer.SupervisedTrainer(
                trjs,
                optimizer=optax.adam(
                    optax.warmup_cosine_decay_schedule(0.0, 3e-4, 3000, args.n_iterations)
                ),
                num_training_steps=args.n_iterations,
                batch_size=args.batch_size,
                num_rollout_steps=args.n_rollout,  # one-step supervised training (would have been the default)
            )

            sup_1_trained_emu, sup_1_loss_history = sup_1_trainer(emulator, jax.random.PRNGKey(42))
            end = time.time()
            print(f"Training completed in {end - start:.2f} seconds.")
            plt.figure(figsize=(10, 5))
            plt.semilogy(sup_1_loss_history)
            plt.xlabel("Training step")
            plt.ylabel("Train Loss")
            plt.savefig(save_path + "/run%d" % i + "/loss_channel_UNet.png")
            plt.close()

            eqx.tree_serialise_leaves(model_path, sup_1_trained_emu)
