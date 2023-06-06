import os
from os.path import join
import numpy as np
import pickle
from grn_control.envs.poni import f_poni

import matplotlib.pyplot as plt
from matplotlib import use
import matplotlib as mpl
from matplotlib.colors import to_rgb, ListedColormap
use("Agg")

from poni_colors import poni_colors_dict, poni_colors_array, cmap, red, green


def plot(alg, subdir=None):

    savedir = alg.checkpoint_dir
    if subdir != None:
        savedir = os.path.join(alg.checkpoint_dir, subdir)
        os.makedirs(savedir, exist_ok=True)

    state_dynamics = np.load(join(savedir, "state_dynamics.npy"))
    action_dynamics = np.load(join(savedir, "action_dynamics.npy"))
    scores = np.load(join(savedir, "test_scores.npy"))
    test_info = pickle.load(open(join(savedir, "test_info.pkl"), "rb") )

    n_episodes, n_steps, _ = state_dynamics.shape

    state_costs = np.zeros(n_episodes)
    contr_costs = np.zeros(n_episodes)
    discounts = alg.agent.gamma**np.arange(n_steps)
    for i in range(n_episodes):

        state_cost = np.array(test_info[i]['state_cost'])
        contr_cost = np.array(test_info[i]['control_cost'])

        state_costs[i] = np.sum( state_cost ) # np.sum( discounts[:,None] * state_cost )
        contr_costs[i] = np.sum( contr_cost ) # np.sum( discounts[:,None] * contr_cost )

    costs = np.array([state_costs,contr_costs])
    np.save(os.path.join(savedir, "costs.npy"), costs)

    ### STATISTICS OF THE DYNAMICS ###
    state_mean = np.mean(state_dynamics, axis=0)
    state_low = np.percentile(state_dynamics, 10, axis=0)
    state_med = np.percentile(state_dynamics, 50, axis=0)
    state_high = np.percentile(state_dynamics, 90, axis=0)

    action_mean = np.mean(action_dynamics, axis=0)
    action_low = np.percentile(action_dynamics, 10, axis=0)
    action_med = np.percentile(action_dynamics, 50, axis=0)
    action_high = np.percentile(action_dynamics, 90, axis=0)

    ## FIRST PASSAGE AT REGION AROUND TARGET
    try:
        target = alg.env.target[1:]
        max_radius = 0.15
        distance = np.sqrt(np.sum((state_dynamics[:,:,1:] - target[None, None, :])**2, axis=2))
        FPT = []
        for d in distance:
            try:
                FPT.append(alg.env.dt * np.where(d < max_radius)[0][0])
            except:
                pass
        np.save(os.path.join(savedir, "FPT.npy"), np.array(FPT))
    except Exception as e:
        print("Error in computing FPT: ", e)


    ### IMPORT/COMPUTE CONTROL FIELD ALONG TRAJECTORY ###

    xmin, xmax = 0., 1.2
    ymin, ymax = 0., 1.2
    oli_, nkx_ = np.linspace(xmin, xmax, 50), np.linspace(ymin, ymax, 50)
    oli,  nkx  = np.meshgrid(oli_, nkx_)
    time_steps = np.array([0, 10, 20, 50, 80, 120, 160, 200, 400])

    n_points = len(oli_) * len(nkx_)
    n_steps = len(time_steps)
    control = np.zeros((n_steps, n_points, action_dynamics.shape[-1]))
    control_file = join(savedir, "control.npy")
    try:
        print("Trying to import control...")
        control_ = np.load(control_file)
        if control_.shape != control.shape:
            raise Exception
        else:
            control = control_
    except:
        print("Computing control field from scratch...")
        import torch as T
        from tqdm import tqdm
        for t, step in enumerate(time_steps):

            print("Computing control field for time step ", step)
            for n in tqdm(range(n_episodes)):
                pax = np.ones(n_points) * state_dynamics[n,step,0]
                irx = np.ones(n_points) * state_dynamics[n,step,3]
                genes = np.vstack([pax, oli.ravel(), nkx.ravel(), irx]).T
                try:
                    control_ = alg.test_actor(genes[:,alg.env.obs_dims])
                except:
                    control_ = alg.test_actor(genes)
                control[t] += control_/float(n_episodes)
        np.save(control_file, control)


    ######################## PLOTTING ###########################



    # =============== PLOT FPT HISTOGRAM ================

    print("Plot histogram of first passage times")
    try:
        FPT = np.load(os.path.join(savedir, "FPT.npy"))
        fig, ax = plt.subplots(figsize=(2,2))
        ax.set_xlabel("First passage time near target")
        ax.set_ylabel("Probability density")
        ax.hist(FPT, density=True, bins=30)
        plt.savefig(join(savedir, "FPT_hist.svg"), bbox_inches="tight")
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(2,2))
        ax.set_xlabel("First passage time near target")
        ax.set_ylabel("Cumulative density")
        FPT = np.sort(FPT)
        ps = 1. * np.arange(len(FPT)) / (len(FPT) - 1)
        ax.plot(FPT, ps)
        plt.savefig(join(savedir, "FPT_CDF.svg"), bbox_inches="tight")
        plt.close(fig)
    except Exception as e:
        print("Error in plotting FPT histogram: ", e)


    # =============== PLOT COST CONTRIBUTIONS ================

    print("Plot costs scatter plot")
    fig, ax = plt.subplots(figsize=(2,2))
    ax.set_xlabel("State cost")
    ax.set_ylabel("Control cost")
    ax.scatter(*costs, s=.3, linewidth=0)
    plt.savefig(join(savedir, "costs_scatter.svg"), bbox_inches="tight")
    plt.close(fig)


    # =========== PLOT STATISTICS OF TRAJECTORIES ============

    print("Plot optimally controlled dynamics")
    ts = alg.env.dt*np.arange(state_dynamics.shape[1])

    fig, axs = plt.subplots(2,1, figsize=(3,2.4), gridspec_kw={'hspace':0.3})

    # plot control
    ax = axs[0]
    ax.set_ylim([0,1.])
    ax.set_xlim([0,20])
    ax.set_xticks([0, 5, 10, 15, 20])
    ax.set_xticklabels([0, "", 10, "", 20])
    ax.set_yticks([0, 0.5, 1])
    ax.set_yticklabels([0, "", 1])
    colors = [
        green,
        red
    ]
    labels = ["GliA", "GliR"]
    for i, label, color in zip(range(2), labels, colors):
        high = action_high[:,i]
        med = action_med[:,i] # action_mean[:,i]
        low = action_low[:,i]
        ax.plot(ts, med, label=label, color=color, lw=1)
        ax.fill_between(ts,low,high,
                        color=color,
                        alpha=0.2,lw=0)
    
    # plot state
    ax = axs[1]
    ax.set_ylim([0,1.])
    ax.set_xlim([0,20])
    ax.set_xticks([0, 5, 10, 15, 20])
    ax.set_xticklabels([0, "", 10, "", 20])
    ax.set_yticks([0, 0.5, 1])
    ax.set_yticklabels([0, "", 1])
    colors=[
            poni_colors_dict["Pax"],
            poni_colors_dict["Olig"],
            poni_colors_dict["Nkx"],
            poni_colors_dict["Irx"]
    ]
    labels = ["Pax6", "Olig2", "Nkx2.2", "Irx3"]
    for i, label, color in zip(range(4), labels, colors):
        high = state_high[:,i]
        med = state_med[:,i] # state_mean[:,i]
        low = state_low[:,i]
        ax.plot(ts, med, label=label, color=color, lw=1)
        ax.fill_between(ts,low,high, alpha=0.2, color=color,lw=0)
    plt.savefig(join(savedir, "testing_split.svg"), bbox_inches="tight")



    # =========== PLOT CONTROL FIELD =============

    print("Plotting control field at all time steps")
    xx, yy = oli, nkx

    greens = cmap(green)
    reds = cmap(red)

    for t, step in enumerate(time_steps):

        if step != time_steps[-1]:
            continue

        # indices:
        #   control: [t, (x,y), 0-1] 
        #   state_dynamics and action_dynamics: [n, t, 0-3]

        pts_ON = state_dynamics[:, step, np.array([1,2])].copy()
        pts_AR = action_dynamics[:, step].copy()

        ### PLOT CONTROL **FIELD** ALONG TRAJECTORY ### 
        fig, axes = plt.subplots(ncols=2, sharex=True, sharey=True, figsize=(5,2))
        for i, (ax, title, cmap_, u_vals) in enumerate(\
                    zip(axes, ["GliA", "GliR"], [greens, reds], pts_AR.T)):
            ax.set_xlim([xmin, xmax])
            ax.set_ylim([ymin, ymax])
            ax.set_xticks([0,.5,1])
            ax.set_xticklabels(["0","1/2","1"])
            ax.set_yticks([0,.5,1])
            ax.set_yticklabels(["0","1/2","1"])
            ax.set_xlabel("Olig2")
            ax.set_ylabel("Nkx2.2")
            ax.text(1.1, 1.1, title, va="top", ha="right", fontsize=10,
                bbox=dict(boxstyle='round', facecolor='gainsboro',ec="gray"))
            ax.set(aspect='equal')

            for state in state_dynamics[:100]:
                ax.plot(state[:200:5,1],state[:200:5,2],c='gray',lw=0.05, zorder=-100)
            _med = np.percentile(state_dynamics,50,axis=0)
            ax.plot(_med[:180:5,1], _med[:180:5,2], c='gray', ls='--', lw=1, zorder=0)

            im = ax.scatter(*pts_ON.T, c=u_vals, cmap=cmap_, s=4,# alpha=0.3,
                            linewidths=.1, edgecolors='k',
                            vmin=0., vmax=1, zorder=2000)
            cbar = plt.colorbar(im, ax=ax, ticks=[0,1/2,1],
                orientation="horizontal",location="top",shrink=0.7)#, extend='max')
            cbar.ax.set_xticklabels(["0","1/2","1"])

            plt.savefig(join(savedir,   
                f"signals_coloured_points_{t}.svg"),
                bbox_inches="tight"
            )
        plt.close(fig)

        if "Oli" in alg.env_name:
            rep = pts_AR[:,1]
            oli = pts_ON[:,0]
            fig, ax = plt.subplots(figsize=(1.5,1))
            _xlim = [.5,1.2]
            ax.set_xlim(_xlim)
            ax.set_ylim([0,1])
            ax.set_xticks([.5,1])
            ax.set_yticks([0,.5,1])
            ax.set_xticklabels(["1/2","1"])
            ax.set_yticklabels(["0","","1"])

            # fit Hill function
            print("Fit Hill function to GliR vs Olig2")
            from scipy.optimize import minimize
            # hill function (decreasing)
            def fun (x, h, c):
                y = (x/c)**h
                return 1. / (1. + y)
            def distance(pars):
                h, c = pars
                return np.mean( (rep - fun(oli, h, c))**2 )
            opt = minimize(distance, np.array([2., .7]), bounds=[(0,None), (0,1)])
            h, c = opt['x']
            print(opt)
            _x = np.linspace(*_xlim, 50)
            _y = fun(_x, h, c)
            ax.plot(_x, _y, c=red, lw=2, label=f"h={h:.1f}, c={c:.1f}")

            ax.scatter(oli, rep, c=red, s=3,
                linewidths=.1, edgecolors='k')

            ax.legend(loc='best')
            plt.savefig(join(savedir,   
                f"GliR_vs_Olig2.svg"),
                bbox_inches="tight"
            )
            plt.close(fig)



