import os
import numpy as np
import pickle
from grn_control.envs.poni import f_poni
from scipy.stats import gaussian_kde as kde

import matplotlib.pyplot as plt
from matplotlib import use
use("Agg")

from poni_colors import poni_colors_dict, poni_colors_array, cmap, red, green


# def load_test_data (alg):

#     # axes are:
#     # 0 - episode; 1 - time; 2 - species; 3 - position/agent
#     state_dynamics = np.load(os.path.join(alg.checkpoint_dir, "state_dynamics.npy"))
#     action_dynamics = np.load(os.path.join(alg.checkpoint_dir, "action_dynamics.npy"))
#     scores = np.load(os.path.join(alg.checkpoint_dir, "test_scores.npy"))
#     test_info = pickle.load( open(os.path.join(alg.checkpoint_dir, "test_info.pkl"), "rb") )

#     return state_dynamics, action_dynamics, scores, test_info


def preprocess (alg):

    # axes are:
    # 0 - episode; 1 - time; 2 - species; 3 - position/agent
    state_dynamics = np.load(os.path.join(alg.checkpoint_dir, "state_dynamics.npy"))
    action_dynamics = np.load(os.path.join(alg.checkpoint_dir, "action_dynamics.npy"))
    scores = np.load(os.path.join(alg.checkpoint_dir, "test_scores.npy"))
    test_info = pickle.load( open(os.path.join(alg.checkpoint_dir, "test_info.pkl"), "rb") )

    # state_dynamics, action_dynamics, scores, test_info = load_test_data(alg)
    n_episodes, n_steps, _, n_agents = state_dynamics.shape

    state_costs = np.zeros(n_episodes)
    contr_costs = np.zeros(n_episodes)
    discounts = alg.agent.gamma**np.arange(n_steps)
    for i in range(n_episodes):

        state_cost = np.array(test_info[i]['state_cost'])
        contr_cost = np.array(test_info[i]['control_cost'])

        state_costs[i] = np.sum( discounts[:,None] * state_cost )
        contr_costs[i] = np.sum( discounts[:,None] * contr_cost )

    costs = np.array([state_costs,contr_costs])
    np.save(os.path.join(alg.checkpoint_dir, "costs.npy"), costs)

    ### IMPORT/COMPUTE CONTROL FIELD ALONG TRAJECTORY ###

    xmin, xmax = 0., 1.2
    ymin, ymax = 0., 1.2
    oli_, nkx_ = np.linspace(xmin, xmax, 50), np.linspace(ymin, ymax, 50)
    oli,  nkx  = np.meshgrid(oli_, nkx_)
    time_steps = np.array([0, 10, 20, 50, 80, 120, 160, 200, 400])

    n_points = len(oli_) * len(nkx_)
    control = np.zeros((n_steps, n_agents, n_points, alg.n_actions))
    value = np.zeros((n_steps, n_agents, n_points))
    velocity = np.zeros((n_steps, n_agents, n_points, 4))
    control_file = os.path.join(alg.checkpoint_dir, "control.npy")
    value_file = os.path.join(alg.checkpoint_dir, "value.npy")
    velocity_file = os.path.join(alg.checkpoint_dir, "velocity.npy")
    try:
        # always re-compute the control, value and velocity fields
        # raise FileNotFoundError
        print("Trying to import control...")
        control_ = np.load(control_file)
        value_ = np.load(value_file)
        velocity_ = np.load(velocity_file)
        if control_.shape != control.shape \
            and value_.shape != value.shape \
            and velocity_.shape != velocity.shape:
            # force the exception if the shapes of the imported array are not right
            raise FileNotFoundError("Shapes are messed up")
        else:
            control = control_
            value = value_
            velocity = velocity_
    except FileNotFoundError:
        print("Computing control, value and velocity fields from scratch...")
        import torch as T
        from tqdm import tqdm
        for t, step in enumerate(time_steps):
            # break

            print("Time step ", step)
            for a in tqdm(range(alg.n_agents)):

                for n in range(n_episodes):
                    pax = np.ones(n_points) * state_dynamics[n,step,0,a]
                    irx = np.ones(n_points) * state_dynamics[n,step,3,a]
                    signal = np.ones(n_points) * state_dynamics[n,step,4,a]
                    state_ = np.vstack([pax, oli.ravel(), nkx.ravel(), irx, signal]).T
                    if hasattr(alg.env, "d_memory"):
                        mem_ = np.tile(state_dynamics[n,step,5:,a], (n_points,1))
                        state_ = np.vstack([state_.T, mem_.T]).T
                    try:
                        # if it is partially observable, use its `obs_dims` attribute
                        control_ = alg.test_actor(state_[:,alg.env.obs_dims])
                        value_ = alg.test_critic(state_[:,alg.env.obs_dims], control_)
                    except:
                        # if not, then use the full state
                        control_ = alg.test_actor(state_)
                        value_ = alg.test_critic(state_, control_)
                    control[t, a] += control_/float(n_episodes)
                    value[t, a] += value_.squeeze()/float(n_episodes)
                    prod_, degr_ = f_poni(state_.T[:4], control_.T[:2])
                    velocity[t, a] += prod_.T - degr_.T

        np.save(control_file, control)
        np.save(value_file, value)
        np.save(velocity_file, velocity)

    return state_dynamics, action_dynamics,\
           scores, state_costs, contr_costs,\
           control, value, velocity


######################## PLOTTING ###########################

def plot(alg):

    state_dynamics, action_dynamics,\
    scores, state_costs, contr_costs,\
    control, value, velocity = preprocess(alg)

    times = np.arange(state_dynamics.shape[1]) * alg.env.dt

    full_dynamics = np.vstack((
        state_dynamics.transpose(2,1,0,3),
        action_dynamics.transpose(2,1,0,3))).transpose(2,1,0,3)

    ### STATISTICS OF THE DYNAMICS ###
    state_mean = np.mean(state_dynamics, axis=0)
    state_low = np.percentile(state_dynamics, 10, axis=0)
    state_med = np.percentile(state_dynamics, 50, axis=0)
    state_high = np.percentile(state_dynamics, 90, axis=0)

    action_mean = np.mean(action_dynamics, axis=0)
    action_low = np.percentile(action_dynamics, 10, axis=0)
    action_med = np.percentile(action_dynamics, 50, axis=0)
    action_high = np.percentile(action_dynamics, 90, axis=0)

    # =========== PLOT STATISTICS OF TRAJECTORIES ============

    print("Plot optimally controlled dynamics")
    
    agents = np.round_(np.linspace(0, alg.n_agents-1, 9)).astype(int)
    pos = ["0", "1/8", "1/4", "3/8", "1/2", "5/8", "3/4", "7/8", "1"]

    # rows = int(np.sqrt(len(agents)))
    # cols = len(agents)//rows

    # fig, axs = plt.subplots(rows, cols,
    #                 gridspec_kw={'width_ratios': cols*[1],
    #                              'height_ratios': rows*[1.4],
    #                              'wspace': 0.4,
    #                              'hspace': 0.5})
    # for a, ax in zip(agents, axs.ravel()):
    testing_dir = os.path.join(alg.checkpoint_dir, "testing")
    os.makedirs(testing_dir, exist_ok=True)

    fig5_dir = os.path.join(testing_dir, "fig5")
    os.makedirs(fig5_dir, exist_ok=True)

    for a, p in zip(agents, pos):

        fig, axs = plt.subplots(2,1, figsize=(1.5,1.2), gridspec_kw={'hspace':0.3})

        # ax.set_title(f"agent {a}")

        # plotting signal
        ax = axs[0]
        ax.set_ylim([0,2.])
        ax.set_xlim([0,20])
        ax.set_xticks([0, 5, 10, 15, 20])
        ax.set_xticklabels([])
        ax.set_yticks([0, 1, 2])
        ax.set_yticklabels([0, 1, 2])
        high = state_high[:,4,a]
        med = state_med[:,4,a] # state_mean[:,4,a]
        low = state_low[:,4,a]
        ax.plot(times,state_med[:,4,a], label="Shh", color='k')
        ax.fill_between(times,low,high, alpha=0.2, color='k',lw=0)

        # plot control
        colors = [
            green,
            red
        ]
        labels = ["GliA", "GliR"]
        for i, label, color in zip(range(2), labels, colors):
            high = action_high[:,i,a]
            med = action_med[:,i,a] # action_mean[:,i,a]
            low = action_low[:,i,a]
            ax.plot(times, med, label=label, color=color, lw=1)
            ax.fill_between(times,low,high,
                            color=color,
                            alpha=0.2,lw=0)
        ax.text(20, 2, p, va="top", ha="right", fontsize=10,
            bbox=dict(boxstyle='round', facecolor='gainsboro',ec="gray"))
        
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
            high = state_high[:,i,a]
            med = state_med[:,i,a] # state_mean[:,i,a]
            low = state_low[:,i,a]
            ax.plot(times, med, label=label, color=color, lw=1)
            ax.fill_between(times,low,high, alpha=0.2, color=color,lw=0)
        plt.savefig(os.path.join(fig5_dir, f"dv-position_{a:02d}.svg"),
                    bbox_inches="tight")


    '''

        SCATTER PLOTS

    '''

    # pos_id = agents[pos.index('1/2')]
    # pos_id = agents[pos.index('5/8')]
    pos_id = 45
    print("pos_id =", pos_id)


    pts_AR = action_dynamics[:, -1, [0,1], pos_id].T
    pts_ON = state_dynamics[:, -1, [1,2], pos_id].T

    fig, axes = plt.subplots(ncols=2, sharex=True, sharey=True, figsize=(5,2))
    for i, (ax, title, cmap_, u_vals) in enumerate(\
                zip(axes, ["GliA", "GliR"], [cmap(green), cmap(red)], pts_AR)):
        ax.set_xlim([0, 1.2])
        ax.set_ylim([0, 1.2])
        ax.set_xticks([0,.5,1])
        ax.set_xticklabels(["0","1/2","1"])
        ax.set_yticks([0,.5,1])
        ax.set_yticklabels(["0","1/2","1"])
        # ax.set_xscale("log")
        # ax.set_yscale("log")
        # ax.set_xlim([.0001, 1.2])
        # ax.set_ylim([.0001, 1.2])
        # ax.set_xticks([.0001,.01,1])
        # ax.set_xticklabels([r"$10^{{-4}}$",r"$10^{{-2}}$","1"])
        # ax.set_yticks([.0001,.01,1])
        # ax.set_yticklabels([r"$10^{{-4}}$",r"$10^{{-2}}$","1"])

        ax.set_xlabel("Olig2")
        ax.set_ylabel("Nkx2.2")
        ax.text(.1, 1.1, title, va="top", ha="left", fontsize=10,
            bbox=dict(boxstyle='round', facecolor='gainsboro',ec="gray"))
        ax.set(aspect='equal')

        # for state in state_dynamics[:100]:
        #     ax.plot(state[:200:5,1],state[:200:5,2],c='gray',lw=0.05, zorder=-100)
        # _med = np.percentile(state_dynamics,50,axis=0)
        # ax.plot(_med[:180:5,1], _med[:180:5,2], c='gray', ls='--', lw=1, zorder=0)
            
        im = ax.scatter(*pts_ON, c=u_vals, cmap=cmap_, s=4,# alpha=0.3,
                        linewidths=.1, edgecolors='k',
                        vmin=0., vmax=1, zorder=2000)

        cbar = plt.colorbar(im, ax=ax, ticks=[0,1/2,1],
            orientation="horizontal",location="top",shrink=0.7)#, extend='max')
        cbar.ax.set_xticklabels(["0","1/2","1"])
    plt.savefig(os.path.join(testing_dir,   
        f"signals_coloured_points.svg"),
        bbox_inches="tight"
    )
    plt.close(fig)


    rep = pts_AR[1]
    oli = pts_ON[0]
    fig, ax = plt.subplots(figsize=(1.5,1))
    _xlim = [.5,1.2]
    ax.set_xlim(_xlim)
    ax.set_ylim([0,1])
    ax.set_xticks([.5,1])
    ax.set_yticks([0,1,2])
    ax.set_xticklabels(["1/2","1"])
    ax.set_yticklabels(["0","","2"])

    # fit Hill function
    print("Fit Hill function to GliR vs Olig2")
    from scipy.optimize import minimize
    # hill function (decreasing)
    def fun (x, h, c):
        y = (x/c)**h
        return 2. / (1. + y)
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

    ax.legend(loc='best')#, title=r"fit, $1/(1 + ([Oli]/c)^h)$")
    plt.savefig(os.path.join(testing_dir,   
        f"GliR_vs_Olig2_{pos_id}.svg"),
        bbox_inches="tight"
    )
    plt.close(fig)

    exit()

    # # ===================================================================
    # # ============================ SNAPSHOTS ============================

    # snapshots_dir = os.path.join(alg.checkpoint_dir, "snapshots")

    # scatter_kw = {'color':'w', 'edgecolors':'k', 'linewidths':.5, 's':2}

    # try:
    #     os.makedirs(snapshots_dir, exist_ok=True)
    # except FileExistsError:
    #     raise FileExistsError("Snapshots directory already exists")

    # agents = np.round_(np.linspace(0, alg.n_agents-1, min(9,alg.n_agents))).astype(int)

    # # =========== PLOT CONTROL FIELD =============

    # print("Plotting control field at all time steps")
    # xx, yy = oli, nkx

    # greens = cmap(green)
    # reds = cmap(red)


    # for t, step in enumerate(time_steps):
    #     # break

    #     for a in agents:

    #         # indices:
    #         #   control: [t, a, (x,y), 0-1] 
    #         #   state_dynamics and action_dynamics: [n, t, 0-3, a]

    #         # estimate marginal density in olig2-nkx2.2 space
    #         pts_ON = state_dynamics[:, step, [1,2], a].copy()
    #         xx_, yy_ = np.meshgrid(np.linspace(xmin,xmax,500),np.linspace(ymin,ymax,500))
    #         density = kde(pts_ON.T).evaluate(np.vstack([xx_.ravel(), yy_.ravel()])).reshape(xx_.shape)
    #         density = density / np.max(density)
    #         # define white image with density as alpha channel
    #         alpha = np.zeros(xx_.shape + (4,))
    #         alpha[:,:,3] = density

    #         # extract activator and repressor fields at given time
    #         # (expectation over Pax-Irx conditioned on Oli-Nkx)
    #         act = control[t,a,:,0].copy().reshape(xx.shape)
    #         rep = control[t,a,:,1].copy().reshape(xx.shape)

    #         ### PLOT CONTROL **FIELD** ALONG TRAJECTORY ### 
    #         fig, axes = plt.subplots(ncols=2, sharex=True, sharey=True, figsize=(7.5,4))
    #         for i, (ax, vals, title, cmap) in enumerate(zip(axes, [act,rep], ["GliA", "GliR"], [greens, reds])):

    #             ax.set_xlim([xmin, xmax])
    #             ax.set_ylim([ymin, ymax])
    #             ax.set_xlabel("Olig 2")
    #             ax.set_ylabel("Nkx 2.2")
    #             ax.set_title(title)
    #             fig.suptitle(f"Time step {step}")
    #             ax.set(aspect='equal')

    #             kwargs = {'origin':'lower',
    #                       'extent': [xmin, xmax, ymin, ymax],
    #                       'vmin':0.,
    #                       'vmax':np.max(action_high[:,i])
    #                     }

    #             im = ax.imshow(vals, cmap=cmap, **kwargs)
    #             plt.colorbar(im, ax=ax, extend='max')
                
    #             # ax.imshow(alpha, **kwargs)
    #             # plt.savefig(f"{alg.checkpoint_dir}/signals_cloud_{t}.svg")

    #             ax.scatter(*pts_ON.T, color='k', s=2)
    #             plt.savefig(os.path.join(snapshots_dir, f"signals_points_a{a:03}_t{step:03}.svg"), bbox_inches='tight')

    #         plt.close(fig)


    # # =========== PLOT VELOCITY FIELD =============

    # print("Plotting velocity field at all time steps")
    # xx, yy = oli, nkx

    # for t, step in enumerate(time_steps):
    #     # break

    #     rows = int(np.sqrt(len(agents)))
    #     cols = len(agents)//rows

    #     fig, axs = plt.subplots(rows, cols,
    #                     gridspec_kw={'width_ratios': cols*[1],
    #                                  'height_ratios': rows*[1.4],
    #                                  'wspace': 0.4,
    #                                  'hspace': 0.5})

    #     for a, ax in zip(agents, axs.ravel()):
    #         pts_ON = state_dynamics[:, step, [1,2], a]

    #         ax.set_aspect('equal')
    #         ax.set_title(f"agent {a}", fontsize=10)
    #         ax.set_xlim([xmin,xmax])
    #         ax.set_ylim([ymin,ymax])
    #         vx, vy = velocity[t,a,:,1], velocity[t,a,:,2]
    #         vx = vx.reshape(xx.shape)
    #         vy = vy.reshape(yy.shape)

    #         speed = np.sqrt(vx**2 + vy**2)
    #         ax.streamplot(xx, yy, vx, vy, density=1.2, color='r', linewidth=.6*speed/speed.max(), arrowstyle='->', arrowsize=.4)
    #         ax.scatter(*pts_ON.T, **scatter_kw)

    #     plt.savefig(os.path.join(snapshots_dir, f"velocity_t{step:03}.svg"), bbox_inches='tight')
    #     plt.close(fig)


    # # =========== PLOT VALUE FUNCTION =============

    # print("Plotting value function at all time steps")
    # xx, yy = oli, nkx

    # for t, step in enumerate(time_steps):
    #     break

    #     rows = int(np.sqrt(len(agents)))
    #     cols = len(agents)//rows

    #     fig, axs = plt.subplots(rows, cols,
    #                     gridspec_kw={'width_ratios': cols*[1],
    #                                  'height_ratios': rows*[1],# rows*[1.4],
    #                                  'wspace': 0.,
    #                                  'hspace': 0.5})

    #     imshow_kw = {
    #         'cmap': 'plasma',
    #         'aspect': 'equal',
    #         'origin': 'lower',
    #         'extent': [xmin, xmax, ymin, ymax],
    #         'vmin': np.min(value),
    #         'vmax': np.max(value),
    #     }

    #     for a, ax in zip(agents, axs.ravel()):
    #         pts_ON = state_dynamics[:, step, [1,2], a]

    #         ax.set_title(f"agent {a}")
    #         ax.set_xlim([xmin,xmax])
    #         ax.set_ylim([ymin,ymax])
    #         v = value[t,a]
    #         vv = v.reshape(xx.shape)
    #         im = ax.imshow(vv, **imshow_kw)
    #         ax.scatter(*pts_ON.T, **scatter_kw)
    #     fig.colorbar(im, ax=axs.ravel())

    #     plt.savefig(os.path.join(snapshots_dir, f"value_t{step:03}.svg"), bbox_inches='tight')
    #     plt.close(fig)

    # ============================= HEATMAPS ================================
    
    def all_heatmaps (data, dims=None, eps=0.3, color_array=None):
        '''
        `data`: array (episode, time, species, agent)
        `duration`: duration of video in sec
        '''
        if dims is None:
            dims = np.arange(3)
        if color_array is None:
            color_array = poni_colors_array[dims]

        try:
            data_ = data[:,:,dims]
        except:
            print("Problem with \"dims\"")
            data_ = data[:,:,:3]
            color_array = poni_colors_array[:3]

        alpha = np.exp(data_/.3)
        alpha /= np.sum(alpha, axis=2)[:,:,None]
        colors = np.einsum('...tsa,sc->...tac', alpha, color_array)
        return colors

    def heatmap_time_agent (data, ax, dims=None, color_array=None, eps=0.3):
        '''
        `data`: array (episode, time, species, agent)
        '''
        # (episode, time, agent, channel)
        colors = all_heatmaps(data, dims=dims, color_array=color_array)
        colors = np.mean(colors, axis=0)
        ax.imshow(np.transpose(colors,(1,0,2)), aspect='auto', origin='lower')

    # ----------- Average over episodes, heatmap in (agent/space, time) ------------ 

    # dims_state, dims_signal, dims_memory = alg.env.get_dims()
    _trial = 0
    figsize=(1.5,1.2)

    '''

        TARGET

    '''

    print("Plotting target state")
    for i, (species, target) in enumerate(zip(["Pax", "Olig", "Nkx", "Irx"], alg.env.target)):
        fig, ax = plt.subplots(figsize=(.4,1.2))
        ax.set_xlim([0,1])
        ax.set_ylim([0,1])
        ax.set_xticks([0,0.5,1])
        ax.set_xticklabels(['0','','1'])
        ax.set_yticks([])
        ax.set_yticklabels([])
        ax.plot(target, np.linspace(0,1,len(target)),\
            c=poni_colors_dict[species], ls='--', lw=2)
        ax.plot(state_dynamics[_trial,-1,i],np.linspace(0,1,len(target)),\
            c=poni_colors_dict[species], lw=1)
        ax.plot([0,1],[1/3,1/3], c='k', ls='--', lw=1)
        ax.plot([0,1],[2/3,2/3], c='k', ls='--', lw=1)
        plt.savefig(os.path.join(fig5_dir,"target_"+species+".svg"),
            bbox_inches="tight")
        plt.close(fig)

    fig, ax = plt.subplots(figsize=figsize)
    ax.set_xlim([0,1])
    ax.set_ylim([0,1])
    ax.set_yticks([0,0.5,1])
    ax.set_yticklabels(['0','1/2','1'])
    ax.set_xticks([0,1/3,2/3,1])
    ax.set_xticklabels(['0','1/3','2/3','1'])
    for i, (species, target) in enumerate(zip(["Pax", "Olig", "Nkx", "Irx"], alg.env.target)):
        ax.plot(np.linspace(0,1,len(target)), target,\
            c=poni_colors_dict[species], lw=2)
        ax.plot([1/3,1/3],[0,1], c='k', ls='--', lw=1)
        ax.plot([2/3,2/3],[0,1], c='k', ls='--', lw=1)
    plt.savefig(os.path.join(fig5_dir,"target.svg"),
        bbox_inches="tight")
    plt.close(fig)



    '''
    
        HEATMAPS

    '''

    print("Plotting heatmaps for grn state")

    fig, ax = plt.subplots(figsize=figsize)
    ax.set_xlim([0,1.2])
    ax.set_xticks([0,.5,1])
    ax.set_xticklabels(['0','','1'])
    ax.set_yticks([0,.5,1])
    ax.set_yticklabels(['0','','1'])
    im = ax.imshow( state_dynamics[_trial,:,4].T,
            extent=[0, 25, 0, 1], vmin=0, vmax=1,
            cmap=cmap("black"), aspect="auto", origin="lower" )
    ax.plot([0,20],[1/3,1/3], c='k', ls='--', lw=1)
    ax.plot([0,20],[2/3,2/3], c='k', ls='--', lw=1)
    fig.colorbar(im)
    plt.savefig(os.path.join(fig5_dir,"heatmap_time-space_shh.svg"),
        bbox_inches="tight")
    plt.close(fig)


    print("Plotting heatmaps for grn state")

    fig, ax = plt.subplots(figsize=figsize)
    ax.set_xlim([0,20])
    ax.set_xticks([0,5,10,15,20])
    ax.set_xticklabels(['0','','10','','20'])
    ax.set_yticks([0,1/3,2/3,1])
    ax.set_yticklabels(['0','1/3','2/3','1'])
    im = ax.imshow(state_dynamics[_trial,:,0].T, cmap=cmap(poni_colors_dict["Pax"]),
        extent=[0, 25, 0, 1], vmin=0, vmax=1,
        aspect="auto", origin="lower" )
    ax.plot([0,20],[1/3,1/3], c='k', ls='--', lw=1)
    ax.plot([0,20],[2/3,2/3], c='k', ls='--', lw=1)
    fig.colorbar(im)
    plt.savefig(os.path.join(fig5_dir,"heatmap_time-space_pax.svg"),
        bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=figsize)
    ax.set_xlim([0,20])
    ax.set_xticks([0,5,10,15,20])
    ax.set_xticklabels(['0','','10','','20'])
    ax.set_yticks([0,1/3,2/3,1])
    ax.set_yticklabels(['0','1/3','2/3','1'])
    im = ax.imshow(state_dynamics[_trial,:,1].T, cmap=cmap(poni_colors_dict["Olig"]),
        extent=[0, 25, 0, 1], vmin=0, vmax=1,
        aspect="auto", origin="lower" )
    ax.plot([0,20],[1/3,1/3], c='k', ls='--', lw=1)
    ax.plot([0,20],[2/3,2/3], c='k', ls='--', lw=1)
    fig.colorbar(im)
    plt.savefig(os.path.join(fig5_dir,"heatmap_time-space_olig.svg"),
        bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=figsize)
    ax.set_xlim([0,20])
    ax.set_xticks([0,5,10,15,20])
    ax.set_xticklabels(['0','','10','','20'])
    ax.set_yticks([0,1/3,2/3,1])
    ax.set_yticklabels(['0','1/3','2/3','1'])
    im = ax.imshow(state_dynamics[_trial,:,2].T, cmap=cmap(poni_colors_dict["Nkx"]),
        extent=[0, 25, 0, 1], vmin=0, vmax=1,
        aspect="auto", origin="lower" )
    ax.plot([0,20],[1/3,1/3], c='k', ls='--', lw=1)
    ax.plot([0,20],[2/3,2/3], c='k', ls='--', lw=1)
    fig.colorbar(im)
    plt.savefig(os.path.join(fig5_dir,"heatmap_time-space_nkx.svg"),
        bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=figsize)
    ax.set_xlim([0,20])
    ax.set_xticks([0,5,10,15,20])
    ax.set_xticklabels(['0','','10','','20'])
    ax.set_yticks([0,1/3,2/3,1])
    ax.set_yticklabels(['0','1/3','2/3','1'])
    im = ax.imshow(state_dynamics[_trial,:,3].T, cmap=cmap(poni_colors_dict["Irx"]),
        extent=[0, 25, 0, 1], vmin=0, vmax=1,
        aspect="auto", origin="lower" )
    ax.plot([0,20],[1/3,1/3], c='k', ls='--', lw=1)
    ax.plot([0,20],[2/3,2/3], c='k', ls='--', lw=1)
    fig.colorbar(im)
    plt.savefig(os.path.join(fig5_dir,"heatmap_time-space_irx.svg"),
        bbox_inches="tight")
    plt.close(fig)

    print("Plotting heatmap for GliA")
    fig, ax = plt.subplots(figsize=figsize)
    ax.set_xlim([0,20])
    ax.set_xticks([0,5,10,15,20])
    ax.set_xticklabels(['0','','10','','20'])
    ax.set_yticks([0,1/3,2/3,1])
    ax.set_yticklabels(['0','1/3','2/3','1'])
    im = im = ax.imshow( action_dynamics[_trial,:,0].T,
            extent=[0, 25, 0, 1], vmin=0, vmax=2,
            cmap=cmap("green"), aspect="auto", origin="lower" )
    ax.plot([0,20],[1/3,1/3], c='k', ls='--', lw=1)
    ax.plot([0,20],[2/3,2/3], c='k', ls='--', lw=1)
    fig.colorbar(im)
    plt.savefig(os.path.join(fig5_dir,"heatmap_time-space_gliA.svg"),
        bbox_inches="tight")
    plt.close(fig)

    print("Plotting heatmap for GliR")
    fig, ax = plt.subplots(figsize=figsize)
    ax.set_xlim([0,20])
    ax.set_xticks([0,5,10,15,20])
    ax.set_xticklabels(['0','','10','','20'])
    ax.set_yticks([0,1/3,2/3,1])
    ax.set_yticklabels(['0','1/3','2/3','1'])
    im = im = ax.imshow( action_dynamics[_trial,:,1].T,
            extent=[0, 25, 0, 1], vmin=0, vmax=2,
            cmap=cmap("red"), aspect="auto", origin="lower" )
    ax.plot([0,20],[1/3,1/3], c='k', ls='--', lw=1)
    ax.plot([0,20],[2/3,2/3], c='k', ls='--', lw=1)
    fig.colorbar(im)
    plt.savefig(os.path.join(fig5_dir,"heatmap_time-space_gliR.svg"),
        bbox_inches="tight")
    plt.close(fig)

    print("Plotting heatmap for signal")
    fig, ax = plt.subplots(figsize=figsize)
    ax.set_xlim([0,20])
    ax.set_xticks([0,5,10,15,20])
    ax.set_xticklabels(['0','','10','','20'])
    ax.set_yticks([0,1/3,2/3,1])
    ax.set_yticklabels(['0','1/3','2/3','1'])
    im = ax.imshow( state_dynamics[_trial,:,4].T,
            extent=[0, 25, 0, 1], vmin=0, vmax=1,
            cmap=cmap("black"), aspect="auto", origin="lower" )
    ax.plot([0,20],[1/3,1/3], c='k', ls='--', lw=1)
    ax.plot([0,20],[2/3,2/3], c='k', ls='--', lw=1)
    fig.colorbar(im)
    plt.savefig(os.path.join(fig5_dir,"heatmap_time-space_shh.svg"),
        bbox_inches="tight")
    plt.close(fig)


    print("Plotting heatmap for memory variables")
    mem_colors = ["blueviolet", "darkorange", "brown"]
    for i, color in zip(range(state_dynamics.shape[2]-5), mem_colors):
        fig, ax = plt.subplots(figsize=figsize)
        ax.set_xlim([0,20])
        ax.set_xticks([])
        ax.set_xticklabels([])
        ax.set_yticks([])
        ax.set_yticklabels([])
        # im = ax.imshow(np.log10(np.mean(state_dynamics[:,:,-1], axis=0).T),
        #     aspect="auto", origin="lower")
        im = ax.imshow( state_dynamics[_trial,:,5+i].T,
                extent=[0, 25, 0, 1], vmin=0, # vmax=1,
                cmap=cmap(color), aspect="auto", origin="lower" )
        ax.plot([0,20],[1/3,1/3], c='k', ls='--', lw=1)
        ax.plot([0,20],[2/3,2/3], c='k', ls='--', lw=1)
        fig.colorbar(im)
        plt.savefig(os.path.join(testing_dir, f"heatmap_time-space_mem{i+1}.svg"),
            bbox_inches="tight")
        plt.close(fig)

    # ============================= VIDEOS =============================

    # ---------------- Video, heatmap in (episode/run, agent/space) ----------------
    from matplotlib.animation import FuncAnimation

    def video_agent_episode (data, dims=None, duration=10,
        dir=None, filename="video_agent_episode.mp4"):
        '''
        `data`: array (episode, time, species, agent)
        `duration`: duration of video in sec
        '''
        use('Agg')

        colors = all_heatmaps(data, dims=dims)
        
        fig, ax = plt.subplots()
        def _plot_frame (k):
            plt.cla()
            ax.set_title("Pattern dynamics")
            ax.set_xlabel("agent (equally spaced in [0,1])")
            ax.set_ylabel("episode")
            ax.imshow(colors[:,k], origin="lower")

        frames=range(data.shape[1])
        dt = duration*1000./data.shape[1]
        ani = FuncAnimation(fig, _plot_frame,
                            interval=dt,
                            frames=frames,
                            blit=False)
        if isinstance(dir, str):
            filename = os.path.join(dir, filename)
        ani.save(filename)

    print("Producing video of grn state")
    video_agent_episode(state_dynamics, dir=testing_dir,
        filename="video_agent_episode_poni.mp4")
