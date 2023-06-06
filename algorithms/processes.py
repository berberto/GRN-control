import numpy as np


class GaussianProcess():
    '''
    Base class for Gaussian processes in 1D

    Constructor requires only buffer_size (size of a buffer of 
    standard normal numbers used to advance the dynamics)

    Only accessible attribute is standard_gaussian.
    This returns a standard gaussian number from the buffer and
    refills the buffer if this is empty.
    This makes the simulation of the stochastic more efficient.

    Derived classes must implement the step() method
    for the specific process implemented

    '''
    def __init__(self, mu, buffer_size=10000):
        self.mu = np.array(mu)
        self._buffer_size = buffer_size
        self._refill_buffer_gaussian()
        self.x = None

    def __call__(self):
        _x = self.x
        self.step()
        return _x

    def _refill_buffer_gaussian (self):
        '''
        Stores an array of standard gaussian numbers to be used for generating
        steps in the Wiener process.
        '''
        self._buffer_gaussian = np.random.normal(size=(self._buffer_size, *self.mu.shape))
        self._counter = 0

    @property
    def standard_gaussian(self):
        if self._counter == self._buffer_size:
            self._refill_buffer_gaussian()
        aux = self._buffer_gaussian[self._counter]
        self._counter += 1
        return aux

    def step(self):
        pass

    def reset(self):
        pass




class Wiener(GaussianProcess):
    '''
    Implementation of the Wiener process (Brownian motion)

    Integration is performed à la Itô (so far irrelevant, because the noise
    is additive).

    A Wiener object is callable. A call performs one Itô integration
    step, and returns the position evaluated *before* taking the step.

    '''
    def __init__(self, mu, D=1., dt=0.01, x0=None, **kwargs):
        super(Wiener, self).__init__(mu, **kwargs)
        self.mu = np.array(mu)
        self.dt = dt
        self.D = D
        self.x0 = x0
        self.reset()

    def step(self):
        # self.standard_gaussian is a property of the GaussianProcess base
        # class, returning the first element of the buffer of gaussian numbers
        self.x += np.sqrt(2. * self.D * self.dt) * self.standard_gaussian

    def reset(self):
        self.x = self.x0 if self.x0 is not None else np.zeros_like(self.mu)



class OrnsteinUhlenbeck(GaussianProcess):
    r'''
    Implementation of the Ornstein-Uhlenbeck process (corresponding to
    the overdamped limit of the Langevin-Kramers process)
    
    Integration is performed à la Itô (so far irrelevant, because the noise
    is additive).

    An OrnsteinUhlenbeck object is callable. A call performs one Itô integration
    step, and returns the position evaluated *before* taking the step.

    '''
    def __init__(self, mu, omega=1., D=1., x0=None, dt=1e-2, **kwargs):
        super(OrnsteinUhlenbeck, self).__init__(mu, **kwargs)
        self.mu = np.array(mu)
        self.omega = omega
        self.D = D
        self.dt = dt
        self.x0 = x0
        self.reset()

    def step(self):
        # use the one of the elements of the buffer to advance the dynamics
        # (integration here is performed à la Itô)
        # self.standard_gaussian is a property of the GaussianProcess base
        # class, returning the first element of the buffer of gaussian numbers
        self.x += - self.omega * (self.x - self.mu) * self.dt \
                  + np.sqrt(2. * self.D * self.dt) * self.standard_gaussian

    def reset(self):
        self.x = self.x0 if self.x0 is not None else np.zeros_like(self.mu)




class LangevinKramers (GaussianProcess):
    r'''
    Implementation of the Langevin-Kramers dynamics.

    Integration is performed à la Itô (so far irrelevant, because the noise
    is additive).

    A LangevinKramers object is callable. A call performs one Itô integration
    step, and returns the 2-tuple (position, velocity) evaluated *before* 
    taking the step.

    '''
    def __init__(self, mu, omega=1., Dv=1., x0=None, v0=None, dt=0.01, **kwargs):
        super(LangevinKramers, self).__init__(mu, **kwargs)
        self.mu = np.array(mu)
        self.omega = omega
        self.Dv = Dv
        self.dt = dt
        self.x0 = x0
        self.v0 = v0
        self.reset()

    def __call__(self):
        _x, _v = self.x, self.v
        self.step()
        return _x, _v

    def step(self):
        # if the buffer does not have stored standard normal numbers, refill it
        # use the one of the elements of the buffer to advance the dynamics
        # (integration here is performed à la Itô)
        # self.standard_gaussian is a property of the GaussianProcess base
        # class, returning the first element of the buffer of gaussian numbers
        aux = self.v * self.dt
        self.v += - self.omega * self.v * self.dt \
                  +  np.sqrt(2. * self.Dv * self.dt) * self.standard_gaussian
        self.x = aux

    def reset(self):
        self.x = self.x0 if self.x0 is not None else np.zeros_like(self.mu)
        self.v = self.v0 if self.v0 is not None else np.zeros_like(self.mu)