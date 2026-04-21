import jax
import jax.numpy as jnp
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
    parser.add_argument('--n_rollout', type=int, default=1, help='rollout steps')
    parser.add_argument('--n_iterations', type=int, default=100000, help='number of training iterations')
    parser.add_argument('--expo', type=int, default=5, help='channel exponent for network size')
    parser.add_argument('--batch_size', type=int, default=16, help='batch size')
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
            data = jnp.load(os.path.join(myDir, files[i]))
            trj.append(data['data'])

        trjs.append(trj)

    trjs = jnp.array(trjs)
    print(trjs.shape)
    max_uval = jnp.max(trjs[:, :, 0, :, :])
    max_vval = jnp.max(trjs[:, :, 1, :, :])
    trjs = trjs.at[:, :, 0, :, :].set(trjs[:, :, 0, :, :] / max_uval)
    trjs = trjs.at[:, :, 1, :, :].set(trjs[:, :, 1, :, :] / max_vval)
    print("Max u value:", max_uval, "Max v value:", max_vval)
    jnp.savez('norm_dynamics.npz', u_max=max_uval, v_max=max_vval)

    idim = 32; jdim = 64
    x_1d = jnp.arange(idim) * 2 * jnp.pi / idim
    y_1d = jnp.cos(jnp.pi * jnp.arange(jdim) / (jdim - 1))
    x, y = jnp.meshgrid(x_1d, y_1d, indexing="ij")

    save_path = "../results/expo%d_batch%d_roll%d_nrun%d/" % (args.expo, args.batch_size, args.n_rollout, args.n_runs)

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
                num_levels=3,
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
