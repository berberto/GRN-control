import os
import numpy as np
import matplotlib.pyplot as plt
from bs1d import BS1D


Ds = [.05, .1, .2]
epss = [10.]
xis = [1.]
taus = [2., 5., 10., 20., np.inf]
def bs():
    for eps in epss:
        for tau in taus:
            for D in Ds:
                b = BS1D(D=D, eps=eps, tau=tau, alpha=1., xi=1.)
                print("\n=========================================================")
                print(b.output_dir)
                print("=========================================================\n")
                yield b
# '''
for b in bs():

    b.plot()

    simulate = False
    try:
        costs = np.load(os.path.join(b.output_dir, "costs.npy"))
        x_final = np.load(os.path.join(b.output_dir, "x_final.npy"))
    except:
        simulate = True

    if simulate:
        state_costs = []
        contr_costs = []
        x_final = []
        from tqdm import tqdm
        for n in tqdm(range(20000)):
            T = min(5*b.tau, 100)
            N = int(T/.01)
            ts, xs, us = b.simulate(seed=None,T=T,N=N)

            # sample an exponentially-distributed terminal time
            # (and the corresponding state)
            t_final = min(100, b.tau * (- np.log(1. - np.random.rand())) )
            id_final = int(t_final/.01)
            try:
                x_final.append(xs[id_final])
            except:
                x_final.append(xs[-1])

            if b.tau is not np.inf:
                discounts = np.exp(-ts/b.tau)
                _state_cost = np.sum(discounts * b.qX(xs))/np.sum(discounts)
                _contr_cost = np.sum(discounts * b.qU(us))/np.sum(discounts)
            else:
                _state_cost = np.mean(b.qX(xs))
                _contr_cost = np.mean(b.qU(us))
            state_costs.append(_state_cost)
            contr_costs.append(_contr_cost)

        x_final = np.array(x_final)
        costs = np.array([state_costs,contr_costs])
        np.save(os.path.join(b.output_dir, "costs.npy"), costs)
        np.save(os.path.join(b.output_dir, "x_final.npy"), x_final)

    costs_ = np.vstack([costs, b.Q(x_final)])
    costs_[0] = costs_[0] - b.Q(x_final)
    np.save(os.path.join(b.output_dir, "costs_triplet.npy"), costs_)

    means = np.mean(costs,axis=1)
    print(f"tau = {b.tau} \t--> {means}, ratio = {means[1]/means[0]}")

    try:
        FPT = np.load(os.path.join(b.output_dir, "FPT.npy"))
        if len(FPT) < 20000:
            raise Exception("FPT too short")
    except Exception as e:
        print(e)
        FPT = b.simulate_FPT(dt=0.01, size=20000)
        np.save(os.path.join(b.output_dir, "FPT.npy"), FPT)
    FPT = b.simulate_FPT(dt=0.01, size=20000)
    np.save(os.path.join(b.output_dir, "FPT.npy"), FPT)


    # histogram of costs

    fig, axs = plt.subplots(1,2,figsize=(5,2))
    plt.tight_layout()
    ax=axs[0]
    ax.set_xlabel(r"State cost / $C_{\rm rms}$")
    ax.set_ylabel(r"Control cost / $C_{\rm rms}$")
    ax.set_xlim([0,2.])
    ax.set_ylim([0,2.])
    H, xedges, yedges = np.histogram2d(*costs/np.sqrt(np.mean(costs**2)),
                                        bins=2*(np.linspace(0,2.,100),), density=True)
    im = ax.imshow(H.T, interpolation="nearest", cmap="Reds", origin="lower", 
            extent=(xedges[0],xedges[-1],yedges[0],yedges[-1]),
            aspect="equal")
    ax=axs[1]
    costs[1] = b.Q(x_final)
    ax.set_xlabel(r"Running cost $\tau^{-1}Q$ / $C_{\rm rms}$")
    ax.set_ylabel(r"Terminal cost $q$ / $C_{\rm rms}$")
    ax.set_xlim([0,2.])
    ax.set_ylim([0,2.])
    H, xedges, yedges = np.histogram2d(*costs/np.sqrt(np.mean(costs**2)),
                                        bins=2*(np.linspace(0,2.,100),), density=True)
    im = ax.imshow(H.T, interpolation="nearest", cmap="Reds", origin="lower", 
            extent=(xedges[0],xedges[-1],yedges[0],yedges[-1]),
            aspect="equal")

    fig.savefig(os.path.join(b.output_dir, "costs_histo.svg"))
    plt.close(fig)


    # histogram of state at (random) terminal time

    fig, ax = plt.subplots(figsize=(3,2))
    plt.tight_layout()
    ax.set_xlabel("First passage time at target")
    ax.set_ylabel("Probability density")
    # ax.set_xlim([0,10])
    im= ax.hist(FPT, bins=30, density=True)
    fig.savefig(os.path.join(b.output_dir, "FPT_histo.svg"))
    plt.close(fig)


    # histogram of state at (random) terminal time

    fig, ax = plt.subplots(figsize=(3,2))
    plt.tight_layout()
    ax.set_xlabel("State at terminal time")
    ax.set_ylabel("Probability density")
    ax.set_xlim([-1.5,1.5])
    ax.set_ylim([0,2.5])

    xs = np.linspace(-2.,2.,200)
    psisq = b.psi(xs)**2
    H, xedges, = np.histogram(x_final, bins=xs, density=True)

    ax.stairs(H, xs, fill=True, alpha=0.3, label="Terminal time state (data)")
    ax.plot(xs, psisq/np.sum(psisq * (xs[1] - xs[0])), label="Steady state (theory)")

    ax.legend(loc="upper left")
    fig.savefig(os.path.join(b.output_dir, "x_final_histo.svg"))
    plt.close(fig)
# '''

#
#   mean first passage times
#
mftp_dir = os.path.join("bs_control", "MFPTs_terminal")

# plot heatmap of mean first passage time at 1
_x = np.linspace(-1.5,1.,1000)
MFPTs = np.zeros((len(Ds), len(epss), len(taus), len(_x))) # from theory -- mean
MFPTs_sym = np.zeros((len(Ds), len(epss), len(taus), 2))   # from simulations -- mean and variance
os.makedirs(mftp_dir, exist_ok=True)
for i, D in enumerate(Ds):
    for j, eps in enumerate(epss):
        for k, tau in enumerate(taus):
            b = BS1D(D=D, eps=eps, tau=tau, alpha=1., xi=1.)
            tau = min(tau,1000)
            MFPTs[i,j,k] = b.MFPT(_x)/tau

            FPT_sym = np.load(os.path.join(b.output_dir, "FPT.npy"))
            MFPTs_sym[i,j,k,0] = np.nanmean(FPT_sym)/tau
            MFPTs_sym[i,j,k,1] = np.nanstd(FPT_sym)/tau/np.sqrt(len(FPT_sym))

        fig, ax = plt.subplots(figsize=(3,2))
        plt.tight_layout()
        _text="\n".join((
                rf"$D = {b.D:.2f}$",
                rf"$\epsilon = {b.eps:.0f}$",
            ))
        ax.text(.95,.95,_text, fontsize=14, transform=ax.transAxes,
            horizontalalignment="right", verticalalignment="top",
            bbox=dict(facecolor='white', alpha=0.8))
        ax.set_ylabel(rf"$\tau$")
        ax.set_xlabel("Initial state")
        ax.set_xticks([-1.5, -1, -.5, 0., .5, 1.])
        ax.set_xticklabels([-1.5, -1, -.5, 0., .5, 1.], fontsize=14)
        ax.set_yticks(list(range(len(taus))))
        ax.set_yticklabels([f"{tau:.0f}" if tau != np.inf \
            else r"$\infty$" for tau in taus],
            fontsize=14)
        im = ax.imshow(MFPTs[i,j], 
            extent=[np.min(_x), np.max(_x), -.5, len(taus)-.5],
            vmin=0, vmax=2,
            origin="lower",
            aspect="auto",
            cmap="bwr",
        )
        ax.plot([-1.,-1.],[-.5, len(taus)-.5], c='k',ls='--')
        cbar = fig.colorbar(im)
        fig.savefig(os.path.join(mftp_dir, f"MFPT_eps:{eps:.1f}_D:{D:.2f}.svg"),
            bbox_inches="tight")
        plt.close(fig)

np.save(os.path.join(mftp_dir, "MFPTs.npy"), MFPTs)


# plot MFPT from -1 to 1 for different tau values
# each plot contains a line for each value of D
# produce one plot for each value of eps
MFPTs = np.load(os.path.join(mftp_dir, "MFPTs.npy"))
for j, eps in enumerate(epss):
    fig, ax = plt.subplots(figsize=(3,2))
    # plt.tight_layout()
    ax.set_yscale("log")
    ax.set_xlabel(r"$\tau$")
    ax.set_ylabel(r"MFPT/$\tau$")
    ax.set_xticks(list(range(len(taus)-1)))
    ax.set_xticklabels([f"{tau:.0f}" if tau != np.inf \
        else r"$\infty$" for tau in np.array(taus)[:-1]])
    ax.set_yticks([2**n for n in range(-1,4)])
    # ax.set_yticklabels([rf"$2^{{{n}}}$" for n in range(-1,4)])
    ax.set_yticklabels(["1/2", "1", "2", "4", "8"])
    ax.plot(np.ones_like(taus[:-1]), c='k', ls='--')
    for i, (D, marker) in enumerate(zip(Ds, ["o", "s", "^", "v", ">"])):
        l = np.argmin((_x + 1)**2)
        ax.plot(MFPTs[i,j,:-1,l], label=f"{D:.2f}", marker=marker)
        ax.fill_between(
            list(range(len(taus)-1)),
            MFPTs[i,j,:-1,l] - MFPTs_sym[i,j,:-1,1],
            MFPTs[i,j,:-1,l] + MFPTs_sym[i,j,:-1,1],
            alpha=0.3, color=f"C{i}"
            )
    ax.legend(title=r"D")
    fig.savefig(os.path.join(mftp_dir, f"MFPT_vs_tau__eps:{eps:2.1f}.svg"),
        bbox_inches="tight")
    plt.close(fig)



#
#   first passage times distributions
#
mftp_dir = os.path.join("bs_control", "MFPTs_terminal")
os.makedirs(mftp_dir, exist_ok=True)
for k, tau in enumerate(taus):
    for j, eps in enumerate(epss):
        fig, ax = plt.subplots(figsize=(3,2))
        ax.set_ylabel("Cumulative density")
        ax.set_xlabel("First passage time at target")
        if tau != np.inf:
            ax.plot([tau, tau], [0,1], c='k', ls='--')
        for i, D in enumerate(Ds):
            b = BS1D(D=D, eps=eps, tau=tau, alpha=1., xi=1.)
            FPT_sym = np.sort(np.load(os.path.join(b.output_dir, "FPT.npy")))
            mean = np.mean(FPT_sym)
            ps = 1. * np.arange(len(FPT_sym)) / (len(FPT_sym) - 1)
            ax.plot(2*[mean], [0,1], c=f"C{i}", ls="--")
            ax.plot(FPT_sym, ps, c=f"C{i}", label=f"{D:.2f}")
        ax.legend(title=rf"$\tau$ = {tau:.0f}"+"\nD" if tau != np.inf else r"$\tau$ = $\infty$"+"\nD")
        fig.savefig(os.path.join(mftp_dir, f"FPT_histo_eps:{eps:.1f}_tau:{tau:3.1f}.svg"),
            bbox_inches="tight")
        plt.close(fig)