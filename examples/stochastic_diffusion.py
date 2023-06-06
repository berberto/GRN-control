#!/bin/python

import os
import numpy as np
import sys
from grn_control.envs.sd import StochasticDiffusion
import matplotlib.pyplot as plt


def plot_simulations (rhos_by_sim, n_sims=10, subdir="sims", full=True):


    print(f"Plot {n_sims} simulations in \"{subdir}\"")

    # calculate exact solution of diffusion-degradation PDE
    # (propagator integrated on space only, to integrate in time
    _n_sims, _n_steps, _n_bins = rhos_by_sim.shape
    _times = sim.dt * np.arange(_n_steps)
    _x = sim.centres
    assert _n_bins == len(_x), "error in number of bins"

    _plot_dir = os.path.join(output_dir, subdir)
    os.makedirs(_plot_dir, exist_ok=True)

    _mean = np.mean(rhos_by_sim, axis=0).T

    for i in range(n_sims):
        _sim = rhos_by_sim[i].T           # data from stochastic simulation
        _data = _sim

        fig, ax = plt.subplots(1,2,figsize=(10,3))

        if full:
            ax[0].set_title("Concentration")
            cmap = "Greens"
            vmin = 0
            vmax = _data.max()
        else:
            ax[0].set_title("Fluctuations of the concentration")
            cmap = "bwr"
            # _data = _data - rho_vs_time_det  # subtract solution of the PDE
            _data = _data - _mean  # subtract average
            _delta = np.max(np.abs(_data))
            vmin = -_delta
            vmax = _delta


        ax[0].set_xlabel("Time")
        ax[0].set_ylabel("Position")
        ax[0].set_xlim([0,15])
        kwargs = {"vmin":vmin, "vmax":vmax,
                  "extent": [0,25,0,1],
                  "aspect":"auto", "origin": "lower",
                  "cmap": cmap
                }
        im = ax[0].imshow(_data, **kwargs)
        fig.colorbar(im,ax=ax[0])

        # # ... and snapshots of the profiles
        # ax[1].set_xlabel("Position")
        # ax[1].set_ylabel("Number density")
        # ax[1].plot(_x, _sim[:,0], label="0")
        # ax[1].plot(_x, _sim[:,_n_steps//10], label="T/10")
        # ax[1].plot(_x, _sim[:,_n_steps//3], label="T/3")
        # ax[1].plot(_x, _sim[:,-1], label = "T")
        # # ax[1].plot(_x, sim.number_density(_x)/sim.size, c='k', ls='--', label="average\nsteady\nstate")
        # ax[1].legend(title=f"T = {_n_steps * 5 * sim.dt}")
        

        # ... and time traces at various positions
        ax[1].set_xlabel("Time")
        ax[1].set_ylabel("Number density")
        ax[1].set_xlim([0,15])
        ax[1].plot(_times, _sim[0], label="0.0", c='C0', lw=0.5)
        # ax[1].plot(_times, rho_vs_time_det[0], c='C0', ls='--', lw=2)
        ax[1].plot(_times, _sim[int(0.2*_n_bins)], label="0.2", c='C1', lw=0.5)
        # ax[1].plot(_times, rho_vs_time_det[int(0.2*_n_bins)], c='C1', ls='--', lw=2)
        ax[1].plot(_times, _sim[int(0.4*_n_bins)], label="0.4", c='C2', lw=0.5)
        # ax[1].plot(_times, rho_vs_time_det[int(0.4*_n_bins)], c='C2', ls='--', lw=2)
        ax[1].plot(_times, _sim[int(0.6*_n_bins)], label = "0.6", c='C3', lw=0.5)
        # ax[1].plot(_times, rho_vs_time_det[int(0.6*_n_bins)], c='C3', ls='--', lw=2)
        ax[1].legend(title="Position")
        
        plt.savefig(os.path.join(_plot_dir,f"sim_{i}.png"), bbox_inches='tight')
        plt.close(fig)


if __name__ == "__main__":

    np.random.seed(1871)

    overwrite = False

    generate_data = True

    n_steps = 2500
    n_sims = 10

    kappa=float(sys.argv[1])
    lam=float(sys.argv[2])

    parameters = {
            "n_bins":100,
            "kappa":kappa,
            "lam":lam,
            # "burst_size":100,
            # "rate_factor":50,
            }


    for prod_r in np.array([0., 0.01, 0.02, 0.05, 0.10, 0.20, 0.50, 1.00])[::-1]:

        print(prod_r)

        np.random.seed(1871)

        output_dir = os.path.join("SD_outputs_newrun", f"kappa_{kappa:.2f}")
        # output_dir = os.path.join(output_dir, f"prodr_local_{prod_r:.2f}")
        output_dir = os.path.join(output_dir, f"prodr_{prod_r:.2f}")
        os.makedirs(output_dir, exist_ok=True)

        sim = StochasticDiffusion(np.array([]), **parameters)

        rho_file = os.path.join(output_dir,"rhos_by_sim.npy")
        xs_file = os.path.join(output_dir,"xs_merged_vs_time.pkl")

        try:
            if overwrite:
                raise FileNotFoundError

            print("Trying to import previus simulations...")
            rhos_by_sim = np.load(rho_file)

        except FileNotFoundError:
            print("Simulating anew...")

            rhos_by_sim = []
            xs_by_sim = []
            xs_merged_vs_time = n_steps*[np.array([])] # list of arrays of positions, stacked across simulations
            for n in range(n_sims):
                '''
                simulate a number of times
                for each simulation, store the density vector at all times
                '''
                sim.reset()
                rho_vs_time = []
                xs_vs_time = []
                for i in range(n_steps):
                    # rho, x = sim.step(prod=prod_r * np.exp( -(sim.centres - 0.5)**2/(2*.01) ) )
                    rho, x = sim.step(prod=prod_r * np.ones_like(sim.centres) )
                    rho_vs_time.append(rho/sim.size)
                    xs_vs_time.append(x)

                    # for every time step, merge the particles to those of previous simulations
                    xs_merged_vs_time[i] = np.append(xs_merged_vs_time[i], x)
                
                # save the whole history into the array of simulations
                rho_vs_time = np.array(rho_vs_time)
                rhos_by_sim.append(rho_vs_time)
                xs_by_sim.append(xs_vs_time)

            rhos_by_sim=np.array(rhos_by_sim)

            np.save(rho_file, rhos_by_sim)


        plot_simulations( rhos_by_sim, subdir="sims", full=False)
        plot_simulations( rhos_by_sim, subdir="sims_full", full=True)
