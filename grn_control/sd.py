#!/bin/python

import os
import numpy as np
import sys


def f_signal (t, x, s, kappa=0.1, alpha=0., var=0.0001, x0=-.1, J0=.25, D=0.009):
    '''
    Time derivative of extracellular signal at position x

    '''
    _var = 2*D*t + var
    num = np.exp( - kappa * t - (x - x0)*(x - x0) / 2 / _var )
    den = np.sqrt(2. * np.pi * _var)
    return - alpha * s + J0 * num/den


class StochasticDiffusion (object):

    x0 = -.1

    def __init__ (self, X, n_bins=100, x_lim=np.array([0.,1.]),
            lam=0.15, kappa=1., source=None,
            rate_factor = 50, burst_size = 100,
            dt=0.01,
            verbose=False):
        '''
        Lagrangean implementation of stochastic diffusion with degradation in continuous space
        with a point-like source.

        Since particles are independent, we do not need to implement it in a Gillespie
        algorithm. The particles are then binned around given centres to calculate
        occupancies/densities. This makes the coding easier, and it's still rigorous.

        X: array (m,)
            Positions of the initial m particles.

        n_bins: int
            Number of bins (of equal size) (default = 100)

        x_lim: array (2,) or list of length 2
            Boundaries of the domain (default = [0,1])

        D: float
            Diffusion constant (default = 1)

        k: float
            Degradation constant (default = 1)

        source: None, float, array (n,) or callable -- UNUSED NOW
            The local production rate
            - None: assumed zero everywhere 
            - float: constant in time and space
            - array (must have same shape as 'centres'): constant
                defined locally (only function of space)
            - callable (must take two arguments, x and t, x can be numpy array):
                function of space x and time t
        '''

        # define and store the initial positions
        self._start = X

        # degradation rate, diffusion constant and decay length at stat. state
        self.kappa = kappa
        self.lam = lam
        self.D = self.kappa * self.lam**2
        
        # production rate
        ## production occurs in bursts of size 'burst_size'
        self.burst_size = burst_size
        ## the production of a burst happens with rate 'J0' * 'rate_factor'
        self.rate_factor = rate_factor
        self.J0 = 2.*self.kappa * lam * np.exp(np.abs(self.x0)/lam) # normalization constant
        ## a number density 'size' is expected at x = 0 at stationary state
        self.size = self.rate_factor * self.burst_size
        self.burst_rate = self.J0 * self.rate_factor

        self.n_bins = n_bins
        self.dx = (x_lim[1] - x_lim[0])/self.n_bins
        self.dt = dt
        self.centres = np.arange(*x_lim, self.dx) + self.dx/2 # centres of the cells/bins
        self.edges = np.linspace(*x_lim, self.n_bins+1)
        self.x_lim = x_lim

        # sets 'positions' to the initial array, finds the ids of bins (if any) and count
        self.reset()
        self.binning()

        if verbose:
            print("time discretization    : ", self.dt)
            print("global production rate : ", self.burst_rate)
            print("global degradation rate: ", self.kappa)
            print("diffusion constant     : ", self.D)
            print("decay length st. state : ", self.lam)
            print("position of the source : ", self.x0)
            print("max number density at origin :", self.size)
            print("burst size             : ", self.burst_size)


    def __len__ (self):
        return len(self.positions)
        
    def binning (self):
        # Ids of the cell where particles find themselves (-1 & `n_cells` if off-range)"
        cell_ids = np.digitize(self.positions, bins=self.edges) - 1
        # Ids of particles that are in the region of interest (on any of the cells)
        ids_in = np.where((cell_ids > -1) * (cell_ids < len(self.centres)))[0]
        # Number of particles on each of the cells
        counts = np.bincount(cell_ids[ids_in], minlength=self.n_bins)
        # Ids of the cell for those particles which lie in the region of interest
        self.cell_ids_in = cell_ids[ids_in]
        self.counts = counts
        self.ids_in = ids_in

    def step (self, prod=None, degr=None):
        '''
        Returns the density at the specified bins and the position of all the particles
        (anywhere, falling in any bin or not)

        Optional arguments:
            prod/degr: (n_bins,) array, specifying local production/degradation rates
        '''

        '''
        degradation
        in a time step dt, remove particles with probability `local_degr * dt`
        where `local_degr` is the uniform constant `kappa`, if it is not specified
        via an array `degr` (optional argument, which must be of the same shape as 
        the `centres` attribute -- 1D array of dimension `n_bins`)
        '''
        local_degr = self.kappa*self.dt * np.ones(len(self))    # homogeneous degradation rate
        if degr is not None:
            local_degr[self.ids_in] += np.take(degr, self.cell_ids_in)
        r = np.random.rand(len(self))
        ids_to_remove = np.where(r < local_degr)[0]
        x = np.delete(self.positions, ids_to_remove)

        # diffusion
        x = x + np.sqrt(2.*self.D*self.dt)*np.random.randn(len(x))

        '''
        production
        We inject particles at `x0`, in bursts of size `birst_size`, at a rate
        `burst_rate`. The total production rate, therefore is `burst_rate * burst_size`.
        '''
        r = np.random.rand()
        if r < self.burst_rate * self.dt:
            x = np.append(x, np.repeat(self.x0, self.burst_size))
        if prod is not None:
            assert prod.shape == self.centres.shape, "invalid shape for production rates optional parameter, `prod`"
            local_prod = prod * self.dt
            _copies = np.random.binomial(int(self.size / self.n_bins), local_prod)
            x = np.append(x, np.repeat(self.centres, _copies))

        self.positions = x
        self.binning()

        return self.counts/self.dx, self.positions

    def reset (self):
        self.positions = self._start
        self.binning()

    @property
    def prob_density(self):
        # at the stationary state
        def _density (x):
            return np.exp(-np.abs(x - sim.x0)/self.lam)/self.lam/2.
        return _density
    
    @property
    def number_density(self):
        # at the stationary state
        def _density (x):
            return self.size * np.exp(-np.abs(x)/self.lam)
        return _density

    @property
    def local_rate(self):
        def _rate(t, x, s):
            return f_signal (t,x,s,kappa=self.kappa,alpha=0,var=0.0001, J0=self.J0, D=self.D)
        return _rate