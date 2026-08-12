import argparse
import os
import re

import equinox as eqx
import imageio
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import pdequinox as pdeqx


def rollout(stepper, n, *, include_init: bool = False):
    def scan_fn(u, _):
        u_next = stepper(u)
        return u_next, u_next

    def rollout_fn(u_0):
        _, trj = jax.lax.scan(scan_fn, u_0, None, length=n)

        if include_init:
            return jnp.concatenate([jnp.expand_dims(u_0, axis=0), trj], axis=0)

        return trj

    return rollout_fn

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--savedir', type=str, required=True, help='directory to save the model')
    args = parser.parse_args()

    save_path = args.savedir
    pattern = r"expo(\d+)_level(\d+)_batch(\d+)_roll(\d+)_nrun(\d+)"
    match = re.search(pattern, save_path)

    if match:
        expo, level, batch_size, roll, n_run = map(int, match.groups())  # 转换为整数
        print(f"Loaded parameters: expo={expo}, level={level}, batch_size={batch_size}, n_rollout={roll}, n_run={n_run}")
    else:
        raise ValueError(f"No match found: {save_path}.")

    files = sorted(os.listdir("../dataset/random1"))
    trj = []
    for i in range(len(files)):
        data = jnp.load(os.path.join("../dataset/random1", files[i]))
        trj.append(data['data'])
    gt_trj = jnp.array(trj)

    norm_dynamics = jnp.load("./norm_dynamics.npz")
    max_uval = norm_dynamics['u_max']
    max_vval = norm_dynamics['v_max']

    gt_trj = gt_trj.at[:,0,:,:].set(gt_trj[:,0,:,:]/max_uval)
    gt_trj = gt_trj.at[:,1,:,:].set(gt_trj[:,1,:,:]/max_vval)

    idim = 32; jdim = 64
    x_1d = jnp.arange(idim) * 2 * jnp.pi / idim
    y_1d = jnp.cos(jnp.pi * jnp.arange(jdim) / (jdim - 1))
    x, y = jnp.meshgrid(x_1d, y_1d, indexing="ij")

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

        roll = rollout(sup_1_trained_emu, gt_trj.shape[0] - 1, include_init=True)
        pred_trj = roll(gt_trj[0])

        output_dir = save_path + "/run%d" % n + "/frames_channel_UNet"
        os.makedirs(output_dir, exist_ok=True)
        var_names = ["u", "v"] 

        fig, axes = plt.subplots(3, 2, figsize=(15, 8))
        prediction_loss = jnp.zeros((799, 2))
        with imageio.get_writer(save_path + "/run%d" % n + '/video_channel_UNet.mp4', fps=12) as writer:
            for i in range(1, 800):
                fig.suptitle(f"Frame {i}", fontsize=16)
                colorbars = []

                for j in range(2):  # u, v
                    vmin = gt_trj[i, j].min()
                    vmax = gt_trj[i, j].max()
                    bound = max(abs(vmin),abs(vmax))
                    diff_vmin = (gt_trj[i, j] - pred_trj[i, j]).min()
                    diff_vmax = (gt_trj[i, j] - pred_trj[i, j]).max()

                    # GT
                    contour1 = axes[0][j].contourf(x, y, gt_trj[i, j], cmap='viridis', levels=jnp.linspace(-bound, bound, 15))
                    axes[0][j].set_title(f"{var_names[j]} GT")
                    axes[0][j].axis('off')
                    cbar1 = plt.colorbar(contour1, ax=axes[0][j], orientation='horizontal')
                    cbar1.set_ticks(jnp.linspace(-bound, bound, 5))
                    colorbars.append(cbar1)

                    # Pred
                    contour2 = axes[1][j].contourf(x, y, pred_trj[i, j], cmap='viridis', levels=jnp.linspace(-bound, bound, 15))
                    axes[1][j].set_title(f"{var_names[j]} Pred")
                    axes[1][j].axis('off')
                    cbar2 = plt.colorbar(contour2, ax=axes[1][j], orientation='horizontal')
                    cbar2.set_ticks(jnp.linspace(-bound, bound, 5))
                    colorbars.append(cbar2)

                    # Diff
                    contour3 = axes[2][j].contourf(x, y, gt_trj[i, j] - pred_trj[i, j], cmap='bwr', levels=jnp.linspace(diff_vmin, diff_vmax, 15))
                    axes[2][j].set_title(f"{var_names[j]} Diff")
                    axes[2][j].axis('off')
                    cbar3 = plt.colorbar(contour3, ax=axes[2][j], orientation='horizontal')
                    cbar3.set_ticks(jnp.linspace(diff_vmin, diff_vmax, 5))
                    colorbars.append(cbar3)

                    prediction_loss = prediction_loss.at[i-1, j].set(jnp.sum(jnp.abs(pred_trj[i, j] - gt_trj[i, j]))/jnp.sum(jnp.abs(gt_trj[i, j])))

                plt.tight_layout()
                plt.subplots_adjust(top=0.92)

                fig.canvas.draw()
                frame = np.asarray(fig.canvas.buffer_rgba())[..., :3]
                writer.append_data(frame)

                for cbar in colorbars: cbar.remove()
                for a in axes.flatten(): a.cla()

        plt.figure()
        plt.yscale('log')
        jnp.save(save_path + "/run%d" % n + "/predict_loss.npy", prediction_loss)
        plt.plot(prediction_loss[:, 0], label='U Err', linestyle='-')
        plt.plot(prediction_loss[:, 1], label='V Err', linestyle='--')
        plt.legend()
        plt.title('Variable Error w.r.t rollouts')
        plt.xlabel('No. of Rollouts')
        plt.ylabel('Error ratio')
        plt.savefig(save_path + "/run%d" % n + '/predict_loss.png')
