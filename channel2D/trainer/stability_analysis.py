import jax
import jax.numpy as jnp
import numpy as np
from scipy.sparse.linalg import eigs
import matplotlib.pyplot as plt
import argparse
import equinox as eqx
import pdequinox as pdeqx
import os
from mpl_toolkits.axes_grid1 import ImageGrid
import re

jax.config.update('jax_platform_name', 'cpu')

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--savedir', type=str, required=True, help='directory to save the model')
    args = parser.parse_args()

    save_path = args.savedir
    pattern = r"expo(\d+)_batch(\d+)_roll(\d+)"
    match = re.search(pattern, save_path)

    if match:
        expo, batch_size, roll = map(int, match.groups())  # 转换为整数
        print(f"Loaded parameters: expo={expo}, batch_size={batch_size}, n_rollout={roll}")
    else:
        raise ValueError(f"No match found: {save_path}.")

    n_runs = sum(os.path.isdir(os.path.join(save_path, item)) for item in os.listdir(save_path))

    idim = 32; jdim = 64; dt = 0.5
    x_1d = jnp.arange(idim) * 2 * jnp.pi / idim
    y_1d = jnp.cos(jnp.pi * jnp.arange(jdim) / (jdim - 1))
    x, y = jnp.meshgrid(x_1d, y_1d, indexing="ij")

    norm_dynamics = jnp.load("./norm_dynamics.npz")
    max_uval = norm_dynamics['u_max']
    max_vval = norm_dynamics['v_max']
 
    jacobian_matrix_list = []

    for n in range(n_runs):

        model_path = save_path + "/run%d" % n + "/channel_UNet.eqx"
        emulator = pdeqx.arch.ClassicUNet(
                2,
                2,
                2,
                hidden_channels=2**expo,
                num_levels=3,
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

        np.save(save_path + "/run%d" % n + '/jacobian_steady.npy', jacobian)

    jacobian_matrix = np.mean(jacobian_matrix_list, axis=0)
    np.save(save_path + '/ensemble_jacobian_steady.npy', jacobian_matrix)

    k = 50  # 要计算的特征值和特征向量的数量
    eigenvalues, eigenvectors = eigs(jacobian_matrix, k=k, which='LM') 

    A_eigenvalues = np.log(eigenvalues)/dt
    neigs = 6
    eigs_idx = []

    for i in range(0, len(A_eigenvalues)):
        print('Lambda',i, A_eigenvalues[i])
        if(len(eigs_idx) < neigs):
            if(np.imag(A_eigenvalues[i]) >= 0):
                eigs_idx.append(i)


    fig, ax = plt.subplots(1,2,figsize=(12, 6))

    data = np.array([-0.019798658959069355 - 0.3121002978186324*1j  , -0.07671991918992122 - 0.42418426496552025*1j  , -0.07804705800473531 - 0.9207866653748497*1j  , -0.07820059989902754 - 0.9209180558431305*1j  , -0.13990151188716707 - 0.8571705506276106*1j  , -0.14031673896721727 - 0.8575896788567664*1j  , -0.20190507564589866 - 0.7939981213312742*1j  , -0.20232062944722828 - 0.7941342362194213*1j  , -0.22134137408680932 - 0.6391251260375203*1j  , -0.22356174519947167 - 0.5344210537949373*1j  , -0.2600560828622944 - 0.33128906971465943*1j  , -0.262807784446977 - 0.731505017031149*1j  , -0.27109732218426247 - 0.7350309624258046*1j  , -0.31056470917193146 - 0.3689819093985278*1j  , -0.3332869681524319 - 0.6740092091394806*1j  , -0.3404796019408088 - 0.6461628187475942*1j  , -0.357525895861826 - 0.6818071524419025*1j  , -0.4056675051722942 - 0.6541856948258901*1j  , -0.45217753309648956 - 0.671133687019918*1j  , -0.5063106236654336 - 0.6703433239671569*1j  , -0.5627819209816707 - 0.6702774202411698*1j  , -0.621078230851464 - 0.6701227439691683*1j  , -0.6813057451407214 - 0.6693168162079051*1j  , -0.7439158696932301 - 0.6693944673299458*1j  , -0.808672173365158 - 0.6686325744337219*1j  , -0.8757219266489588 - 0.66884648634492*1j  , -0.9451606834253747 - 0.668153074169882*1j  , -1.016749432379796 - 0.6684431796927137*1j  , -1.0909437278877765 - 0.667810863473027*1j  , -1.167145421330843 - 0.6681218064663597*1j  , -1.2460767556782664 - 0.6675912908535451*1j  , -1.3272242554621756 - 0.6680501623364318*1j  , -1.4110972836344473 - 0.6664204203079334*1j  , -1.4935702998266018 - 0.6684307636410967*1j  , -1.5921976471053758 - 0.6769892655924513*1j  , -1.6753120148430292 - 0.645598959934808*1j  , -1.7184126261699832 - 0.6646638841728884*1j  , -1.8589357568493916 - 0.8248962127941583*1j  , -1.9412088070794888 - 0.5624387927605302*1j  , -1.9412472090742305 - 0.579025582537711*1j  , -2.261574096296604 - 0.4757984971702194*1j  , -2.2716543365033095 - 0.48557473290637343*1j  , -2.692159569687726 - 0.3952549480319237*1j  , -2.7033348128255157 - 0.40071158159855125*1j  , -3.287225489689509 - 0.3199684994960702*1j  , -3.2987207640575975 - 0.32336709837170735*1j  , -4.144146545280101 - 0.25074871914260966*1j  , -4.1565886801072764 - 0.2530717647988924*1j  , -5.444054456505698 - 0.1886475697384148*1j  , -5.458435670632356 - 0.19033137623138807*1j  , -7.555549923177403 - 0.1345131108146723*1j  , -7.5736987611665105 - 0.135780552926595*1j  , -11.319872972929858 - 0.08900303754844291*1j  , -11.344965719673507 - 0.08998503596140114*1j  , -19.041075002859518 - 0.052554832380610116*1j  , -19.081221950491877 - 0.05333659219064175*1j  , -38.9390275668578 - 0.02555971201719252*1j  , -39.016685987694515 - 0.026206240636792555*1j  , -120.6653089869348 - 0.008214868059230886*1j  , -120.90007962508503 - 0.008766514865467256*1j  , -1344.643671658063 - 0.000648506361332131*1j  , -1346.4476925395315 - 0.0010971209010600114*1j])
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

    fig.savefig(save_path + "/eigenspectrum_steady.png")

    fig = plt.figure(figsize=(9,neigs),dpi=100)
    grid = ImageGrid(fig, 111, nrows_ncols=(neigs,2), axes_pad=0.05, share_all=True,
                    # cbar_location="right", cbar_mode="single", cbar_size="1%", cbar_pad=0.1,
                    )
    axs = grid.axes_column
    jjj = 0 
    im2 =[]
    for i_index, icolumn in enumerate(eigs_idx):
        for iii in range(0,2):
            vis_output = eigenvectors[:,icolumn].reshape([1, 2, idim, jdim])
            vis_output = np.real(vis_output[0,iii,:,:])

            # vmin = -0.04
            # vmax = 0.04

            im2_temp = axs[iii][jjj].contourf(
                x, y, vis_output#, levels=np.linspace(vmin, vmax, 15), 
                #vmin=vmin, vmax=vmax, cmap=plt.cm.get_cmap("twilight_r")
                )

            im2.append(im2_temp)

            axs[iii][jjj].set_xticks([])
            axs[iii][jjj].set_yticks([])
            axs[iii][jjj].set_aspect('equal')
            axs[iii][jjj].set_xlim(0, 2*np.pi); axs[iii][jjj].set_ylim(-1, 1)

        jjj+=1

    # cbar = fig.colorbar(im2_temp, cax=grid.cbar_axes[0], orientation='vertical', ticks=[vmin, vmin/2., 0, vmax/2., vmax])
    # im2_temp.set_clim(vmin,vmax)

    # fig.tight_layout()
    fig.savefig (save_path + "/eigenvectors_steady.png", dpi=300)