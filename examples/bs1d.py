import numpy as np
import os
import sys
import pickle
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from tqdm import tqdm



class BS1D (object):

    def __init__ (self, D=0.1, eps=10, xi=1., alpha=1.,
        terminal=True, tau=np.inf, overwrite=False):

        self.D = D
        self.eps = eps
        self.xi = xi
        if self.xi is not None:
            self.__q = lambda x: 0.5 * (x - self.xi)**2
            cost = f"sq-xi:{self.xi:.1f}"
        else:
            self.__q = lambda x: -x
            cost = f"max"

        self.terminal = terminal
        if self.terminal:
            cost += "_term"

        self.tau = tau
        self.alpha = alpha
        self.u_critical = 2*(self.alpha/3)**1.5

        self.output_dir = os.path.join("bs_control",
            f"{cost}_alpha:{self.alpha:.1f}_tau:{self.tau:03.1f}_eps:{self.eps:.1f}_D:{self.D:.2f}")
        os.makedirs(self.output_dir, exist_ok=True)
        if overwrite:
            from glob import glob
            pkl_files = glob(os.path.join(self.output_dir, "*.pkl"))
            for file in pkl_files:
                os.remove(file)

    def Q (self, x):
        '''
        Terminal cost
        '''
        if self.terminal:
            return self.__q(x)
        else:
            return 0.

    def qX (self, x):
        '''
        Running cost for the state (including contribution from terminal cost)
        '''
        return self.__q(x) + self.Q(x) / self.tau

    def qU (self, u):
        '''
        Running cost for the control
        '''
        return self.eps * 0.5 * u*u

    def V (self, x, u=None):
        '''
        Potential
        '''
        _xsq = x*x
        _V = ( 0.5 * _xsq - self.alpha ) * 0.5 * _xsq
        if u is not None:
            _V += - u * x
        return _V

    def f (self, x, u=None):
        '''
        Derivative of the potential
        '''
        _f = - x * (x*x - self.alpha)
        if u is not None:
            _f += u
        return _f

    def jacobian (self, x, u=None):
        '''
        Jacobian of the force, or Hessian of the potential
        '''
        return 3. * x**2 - self.alpha


    def __solve (self):
        '''
        Wrapper function for the solution of HJB
        '''
        if hasattr(self, "psi") and hasattr(self, "control") and hasattr(self, "drift"):
            # these attributes are set when the `solve_HJB' method is called
            return
        else:
            try:
                # try to load them from existing files
                self.drift = pickle.load(open(os.path.join(self.output_dir, "drift.pkl"), "rb"))
                self.control = pickle.load(open(os.path.join(self.output_dir, "control.pkl"), "rb"))
                self.psi = pickle.load(open(os.path.join(self.output_dir, "psi.pkl"), "rb"))
            except:
                # otherwise, solve anew
                _x = np.linspace(- 5.*self.alpha, 5.*self.alpha, 3000)
                self.solve_HJB(_x)

    def plot (self):
        
        self.__solve()

        _x = np.linspace(-1.5, 1.5, 100)
        psi = self.psi(_x)

        V_eff = - 2*self.D*np.log(psi)
        V_eff -= - 2*self.D*np.log(psi[np.argmin(_x**2)])
        V_eff = interp1d(_x, V_eff)

        # plot dynamical landscape
        fig, ax = plt.subplots()
        plt.tight_layout()
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_ylim([-1,1])
        ax.plot(_x, V_eff(_x), c="C1",lw=2)
        fig.savefig(os.path.join(self.output_dir, "landscape_closed_loop.svg"))
        plt.close(fig)

        def set_text (ax, *pos, **kwargs):
            _tau_str = f"{self.tau:.0f}" if self.tau != np.inf else r"\infty"
            _text="\n".join((
                    rf"$D = {self.D:.2f}$",
                    rf"$\epsilon = {self.eps:.0f}$",
                    rf"$\tau = {_tau_str}$",
                ))
            ax.text(*pos,_text, fontsize=14, transform=ax.transAxes,
                horizontalalignment="right", **kwargs)
            return

        # plot the decay rate of the square norm of the solution
        # asymptotically proportional to the minimum eigenvalue
        fig, ax = plt.subplots()
        plt.tight_layout()
        ax.set_xlabel("Iteration")
        ax.set_ylabel("Eigenvalue (norm decay rate)")
        ax.set_yscale("log")
        ax.plot(np.load(os.path.join(self.output_dir, "eigenvalue.npy")))
        fig.savefig(os.path.join(self.output_dir, "eigenvalue.svg"))
        plt.close(fig)

        # plot uncontrolled dynamics and potential
        fig, ax = plt.subplots()
        plt.tight_layout()
        ax1 = ax.twinx()
        ax.set_xlabel("Position")
        ax.set_ylabel("Drift", c="C0")
        ax1.set_ylabel("Potential", c="C1")
        ax.plot(_x, self.f(_x), c="C0")
        ax.plot([_x.min(), _x.max()], [0,0], c='k', ls='--')
        ax1.plot(_x, self.V(_x), c="C1")
        fig.savefig(os.path.join(self.output_dir, "landscape_uncontrolled.svg"), bbox_inches="tight")
        plt.close(fig)

        # plot optimally controlled dynamics and effective potential
        fig, ax = plt.subplots()
        plt.tight_layout()
        ax1 = ax.twinx()
        ax.set_xticks([])
        ax.set_yticks([])
        ax1.set_yticks([])
        ax.set_ylim([-2,2])
        ax1.set_ylim([-1,1])
        set_text(ax, .95,.95, verticalalignment="top")
        ax.plot(_x, self.drift(_x), c="C0")
        ax.plot(_x, self.control(_x), c="C0", ls="-.")
        ax.plot(_x, np.zeros_like(_x), c="k", ls='--')
        ax1.plot(_x, V_eff(_x), c="C1")
        fig.savefig(os.path.join(self.output_dir, "landscape_controlled_clean.svg"))
        plt.close(fig)

        # plot the effective potential parametrically, 
        # where the control is selected as the optimal one
        # for the position of interest
        _landscape = np.zeros(2*_x.shape)
        for i, x in enumerate(_x):
            _landscape[i] = self.V(_x, u=self.control(x))
        _delta = max( np.max(_landscape), np.max(-_landscape) )

        fig, ax = plt.subplots()
        plt.tight_layout()
        ax.set_xlabel("Where I evaluate the landscape")
        ax.set_ylabel("Where I pick the control")
        ax.imshow(_landscape, origin='lower',
            extent=2*[np.min(_x),np.max(_x)], 
            vmin=-_delta, vmax=+_delta,
            aspect='equal', cmap='bwr')
        ax.contour(*np.meshgrid(_x,_x),_landscape, 0, colors='k',linestyles='dashed')
        ax.plot([0,0],[np.min(_x),np.max(_x)], ls='--', c='k')
        fig.savefig(os.path.join(self.output_dir, "landscape_parametric_heatmap.svg"))
        plt.close(fig)

        fig, ax = plt.subplots()
        plt.tight_layout()
        ax.set_xlabel("Where I evaluate the landscape")
        ax.set_ylabel("Potential")
        _selected_x = [-1,-.75,-.5, .5, .75, 1]
        for i, x in enumerate(_selected_x):
            ax.plot(_x, _landscape[np.argmin((_x - x)**2)], label=f"{x:.2f}")
        ax.legend(title="Where I pick the control",loc='best')
        fig.savefig(os.path.join(self.output_dir, "landscape_parametric.svg"))
        plt.close(fig)

        # simulate/import trajectories
        _filename = os.path.join(self.output_dir,"sample_dynamics_ens.pkl")
        try:
            with open(_filename, "rb") as f:
                ts, xs_arr, cs_arr = pickle.load(f)
        except:
            xs_arr = []
            cs_arr = []
            for i in tqdm(range(100)):
                ts, xs, cs = self.simulate(seed=None)
                xs_arr.append(xs)
                cs_arr.append(cs)
            xs_arr = np.array(xs_arr)
            cs_arr = np.array(cs_arr)
            with open(_filename, "wb") as f:
                pickle.dump((ts, xs_arr, cs_arr), f)

        _skip=len(ts)//100
        ts = ts[::_skip]
        xs_arr = xs_arr[:,::_skip]
        cs_arr = cs_arr[:,::_skip]

        # plot statistics of trajectories
        # (percentiles at every time)
        fig, ax = plt.subplots()
        plt.tight_layout()
        ax1 = ax.twinx()
        ax.set_xticks([])
        ax.set_yticks([])
        ax1.set_yticks([])
        ax.set_ylim([-1.2,1.2])
        ax1.set_ylim([-1.,1.])
        set_text(ax, 0.95, 0.05, verticalalignment="bottom")
        ax.set_xlim([np.min(ts),np.max(ts)])
        ax.plot(ts, self.xi*np.ones_like(ts), c="k", ls=":")
        ax.plot(ts, np.zeros_like(ts), c="k", ls="--")
        ax1.fill_between([np.min(ts),np.max(ts)], 2*[-self.u_critical], 2*[self.u_critical], color="grey", alpha=0.2, lw=0)
        ax.fill_between(ts, np.percentile(xs_arr, 25, axis=0), np.percentile(xs_arr, 75, axis=0), color="C2", alpha=0.2, lw=0)
        ax.plot(ts, np.median(xs_arr, axis=0), c="C2")
        ax1.fill_between(ts, np.percentile(cs_arr, 25, axis=0), np.percentile(cs_arr, 75, axis=0), color="C3", alpha=0.2, lw=0)
        ax1.plot(ts, np.median(cs_arr,axis=0), c="C3")
        fig.savefig(os.path.join(self.output_dir, "sample_dynamics_ens.svg"))
        plt.close(fig)

        # plot individual trakectories stopped at exponentially distributed time
        fig, ax = plt.subplots()
        plt.tight_layout()
        ax1 = ax.twinx()
        ax.set_xticks([])
        ax.set_yticks([])
        ax1.set_yticks([])
        ax.set_ylim([-1.2,1.2])
        ax1.set_ylim([-1.,1.])
        set_text(ax, 0.95, 0.05, verticalalignment="bottom")
        ax.set_xlim([np.min(ts),np.max(ts)])
        ax.plot(ts, self.xi*np.ones_like(ts), c="k", ls=":")
        ax.plot(ts, np.zeros_like(ts), c="k", ls="--")
        ax1.fill_between([np.min(ts),np.max(ts)], 2*[-self.u_critical], 2*[self.u_critical], color="grey", alpha=0.2, lw=0)
        ts_abs = - self.tau * np.log(1 - np.random.rand(len(xs_arr))) # random stopping times
        ns_abs = (ts_abs / (ts[1] - ts[0])).astype(int)
        for i, n_abs in enumerate(ns_abs):
            ax.plot(ts[:n_abs], xs_arr[i,:n_abs], c="C2", alpha=0.2)
            ax.plot(ts[n_abs:], xs_arr[i,n_abs:], c="C2", alpha=0.01)
            ax1.plot(ts[:n_abs], cs_arr[i,:n_abs], c="C3", alpha=0.2)
            ax1.plot(ts[n_abs:], cs_arr[i,n_abs:], c="C3", alpha=0.01)
        fig.savefig(os.path.join(self.output_dir, "sample_dynamics_ens_1.svg"))
        plt.close(fig)


    def simulate (self, seed=25101917, T=15, N=10000):

        self.__solve()

        if seed is not None:
            np.random.seed(seed)

        ts = np.linspace(0,T,N)
        xs = np.zeros_like(ts)
        gs = np.random.randn(ts.shape[0])

        dt = ts[1] - ts[0]
        x = -1.
        for i, t in enumerate(ts):
            xs[i] = x
            x += self.drift(x)*dt + np.sqrt(2.*self.D*dt)*gs[i]

        return ts, xs, self.control(xs)


    def simulate_FPT (self, dt=.01, seed=25101917, size=10000):

        self.__solve()

        if seed is not None:
            np.random.seed(seed)

        x0 = -1.
        fpt=np.zeros(size)
        for n in tqdm(range(size)):
            i = 0
            x = x0
            while True:
                xp = x + self.drift(x)*dt + np.sqrt(2.*self.D*dt)*np.random.randn()
                crossed = (xp - self.xi) / (x0 - self.xi) <= 0
                if crossed:
                    fpt[n] = dt * i
                    break
                i += 1
                x = xp

        return fpt


    def videos (self):

        if os.path.exists(os.path.join(self.output_dir, "video_dynamical_landscape_reduced.mp4")):
            return
            
        import matplotlib.animation as animation

        _x = np.linspace(-1.5, 1.5, 100)
        ts, xs, us = self.simulate()


        n_skip = 50
        n_frames = len(ts)//n_skip
        duration = 20. # in secs
        interval = duration/n_frames*1000.
        
        fig, ax = plt.subplots(2, figsize=(6,6))
        plt.tight_layout()

        ax[0].set_xlim([np.min(_x),np.max(_x)])
        ax[0].set_ylim([-.4, 1])
        ax[0].set_ylabel("Landscape")
        ax[0].tick_params(left=False,
                          bottom=False,
                          labelleft=False,
                          labelbottom=False)
        lineV, = ax[0].plot([],[], lw=2)
        lineP, = ax[0].plot([],[], marker='o', markersize=10., color="C2")

        ax[1].set_ylim(np.min(ts), np.max(ts))
        ax[1].set_xlim([np.min(_x),np.max(_x)])
        ax[1].set_xlabel("Position")
        ax[1].set_ylabel("Time")
        ax[1].axvspan(-self.u_critical, self.u_critical, alpha=0.2, color="grey")
        lineX, = ax[1].plot([],[], lw=1, c="C2")
        lineU, = ax[1].plot([],[], lw=1, c="C3")


        def generator():
            V = self.V(_x, u=us[0])
            for k , (t, x, u) in enumerate(zip(ts, xs, us)):
                if k % n_skip == 0:
                    V = 0.9 * V + 0.1 * self.V(_x, u=u)
                    yield t, x, u, V
        

        Tdata, Xdata, Udata = [], [], []
        def frame(data):
            t, x, u, V = data

            Tdata.append(t)
            Xdata.append(x)
            Udata.append(u)

            lineX.set_data(Xdata, Tdata)
            lineU.set_data(Udata, Tdata)

            lineV.set_data(_x, V)
            lineP.set_data([x],[V[np.argmin((_x - x)**2)]])
            # lineP1.set_data([x],[V_eff[np.argmin((_x - x)**2)]])

            return lineV, lineP, lineX, lineU #, lineP1

        ani = animation.FuncAnimation(fig, frame, generator, interval=interval,
                blit=True, save_count=n_frames, cache_frame_data=False, repeat=False)
        ani.save(os.path.join(self.output_dir, "video_dynamical_landscape_reduced.mp4"))
        plt.close(fig)


    def MFPT (self, x):

        if not isinstance(x, np.ndarray):
            x = np.array([x])

        self.__solve()

        _x = np.linspace(-3., 1, 100000)
        dx = _x[1] - _x[0]
        psi = self.psi(_x)

        int_1 = np.cumsum(dx * psi)/psi
        int_2 = dx * np.cumsum( int_1[::-1] )[::-1]
        ids = [np.argmin((_x-p)**2) for p in x]
        _mfpt = int_2[ids] / self.D

        return _mfpt


    def solve_HJB (self, _x):
        '''
        Solve for the ground state of the associated imaginary-time
        Schrodinger equation with the method of filtering
        '''
        dx = _x[1] - _x[0]
        dt = 1.e-5

        print(f"dx = {dx:.2E}")
        print(f"dt = {dt:.2E}")

        grad = lambda f: np.gradient(f, _x, edge_order=2)
        lapl = lambda f: grad(grad(f))

        V = self.V(_x)
        gradV = - self.f(_x)
        laplV = grad(gradV)
        laplV[0] = laplV[2]
        laplV[1] = laplV[2]
        laplV[-1] = laplV[-3]
        laplV[-2] = laplV[-3]

        # # start from the solution where V is not present
        # psi = np.exp( - 0.5 * self.__q(_x) / np.sqrt(self.eps) / self.D )
        # psi /= np.power(2. * np.pi * np.sqrt(self.eps) * self.D, .25)

        # start from the solution where tau -> 0
        psi = np.exp( - 0.5 * V / self.D + _x )
        psi /= np.sqrt(np.sum(psi*psi*dx))

        converged = False
        mu = np.inf
        iteration = 0
        eigenvalue = []
        psi_min = np.zeros_like(psi)
        momentum = np.zeros_like(psi) # Nesterov's momentum
        mu_min = mu
        # For any value of tau solve as a minimum eigenvalue problem 
        # (associated Schrodinger equation) without non-linearity
        while not converged:

            laplPsi = lapl(psi)
            V_s = self.qX(_x) * 0.5 / (self.D * self.eps)
            V_s += gradV * gradV * 0.25 / self.D
            V_s += - 0.5 * laplV

            if self.tau != np.inf:
                V_s += 0.5 * V / self.D / self.tau
                V_s += np.log( psi + 1.e-08 ) / self.tau
                
            H_psi = - self.D * laplPsi + V_s * psi
            _psi = psi - dt * H_psi
            _norm_sq = np.sum(_psi * _psi * dx)

            if self.tau == np.inf:
                _psi = np.abs( _psi / np.sqrt(_norm_sq))
                # in this case H is linear, so we can write solutions
                # as superpositions of the eigenvalues, each evolving
                # "independently". The slowest mode is the ground state
                # and from the change in normalisation in imaginary time,
                # we can infer the least eigenvalue
                _mu = - 0.5 / dt * np.log(_norm_sq)
                # check convergence to a minimum in the rate of decay
                # (proportional to the minimum eigenvalue)
                threshold = 1.e-06
                criterion = (mu - _mu)/mu < threshold
                _message = f"Iter {iteration},   eval/loss {_mu:.3E}"
                _message += f",   rel.ch. {(mu - _mu)/mu:.3E}"

            else:
                # in this case H is non-linear, and we are looking for
                # its "0-energy" state.
                # we can check the convergence by looking at the square of
                # the whole equation.
                _mu = np.sum( H_psi * H_psi * dx ) / _norm_sq

                threshold = 1.e-03
                criterion = _mu < threshold
                _message = f"Iter {iteration},   eval/loss {_mu:.3E}"
                _message += f",   rel ch  {(mu - _mu)/mu:.3E}"
                _message += f",   norm sq {_norm_sq:.3E}"

            if criterion:
                converged = True

            eigenvalue.append(_mu)

            if iteration % 1000 == 0:
                print(_message)

            if _mu < mu_min and _mu > 0:
                psi_min = psi
                mu_min = _mu

            psi = _psi
            mu = _mu

            iteration += 1

        print(_message)

        gradLogPsi = np.gradient(psi, _x) / ( psi + 1.e-08 )
        u = gradV + 2 * self.D * gradLogPsi

        self.drift = interp1d(_x, u - gradV)
        self.control = interp1d(_x, u)
        self.psi = interp1d(_x, psi)

        pickle.dump(self.drift, 
            open(os.path.join(self.output_dir, "drift.pkl"), "wb"))
        pickle.dump(self.control, 
            open(os.path.join(self.output_dir, "control.pkl"), "wb"))
        pickle.dump(self.psi, 
            open(os.path.join(self.output_dir, "psi.pkl"), "wb"))
        np.save(os.path.join(self.output_dir, "eigenvalue.npy"), eigenvalue)
        


if __name__ == "__main__":

    Ds = [.05, .1, .2, .5]
    epss = [2., 5., 10., 20., 50.]
    xis = [1.] #, 0, None]
    taus = [2., 5., 10., 20., np.inf] # [np.inf, 20., 10., 5., 2., 1.]

    def bs():
        for eps in epss:
            for tau in taus:
                for xi in xis:
                    for D in Ds:
                        b = BS1D(D=D, alpha=1., eps=eps, xi=xi, tau=tau)
                        print("\n=========================================================")
                        print(b.output_dir)
                        print("=========================================================\n")
                        yield b

    for b in bs():
        b.plot()
        b.videos()
