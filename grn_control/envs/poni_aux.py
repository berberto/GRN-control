import numpy as np

default_pars = {
    "lambdaConc":1,
    "lambdaTime":1,
    "K_Pol_Pax": 4.8,
    "K_Pol_Oli": 47.8,
    "K_Pol_Nkx": 27.4,
    "K_Pol_Irx": 23.4,
    "K_Gli_Oli": 18.0,
    "K_Gli_Nkx": 373.,
    "K_Oli_Pax": 1.9,
    "K_Oli_Nkx": 27.1,
    "K_Nkx_Oli": 60.6,
    "K_Pax_Nkx": 4.8,
    "K_Nkx_Pax": 26.7,
    "K_Irx_Oli": 28.4,
    "K_Oli_Irx": 58.8,
    "K_Irx_Nkx": 47.1,
    "K_Nkx_Irx": 76.2,
    "alpha_Pax": 2.,
    "alpha_Oli": 2.,
    "alpha_Nkx": 2.,
    "alpha_Irx": 2.,
    "delta": 2.}
f_A = 10.
C_Pol = 0.8

def Hill (x):
    return x/(1. + x)

def f_poni (state, control, pars=default_pars):
    pax, oli, nkx, irx = state
    glia, glir = control

    # Pax
    # Repression by Olig
    aux_1 = 1./(1. + pars["K_Oli_Pax"] * oli)**2
    # Repression by Nkx
    aux_2 = 1./(1. + pars["K_Nkx_Pax"] * nkx)**2
    p_pax = pars["alpha_Pax"] * Hill(C_Pol * pars["K_Pol_Pax"] * aux_1 * aux_2)
    d_pax = pars["delta"] * pax

    # Olig
    # Activation by Gli
    aux_1 = 1. + f_A * pars["K_Gli_Oli"] * glia
    aux_1 /= 1. + pars["K_Gli_Oli"] * ( glia + glir )
    # Repression by Nkx
    aux_2 = 1./(1. + pars["K_Nkx_Oli"] * nkx)**2
    # Repression by Irx
    aux_3 = 1./(1. + pars["K_Irx_Oli"] * irx)**2
    p_oli = pars["alpha_Oli"] * Hill(C_Pol * pars["K_Pol_Oli"]  * aux_1 * aux_2 * aux_3)
    d_oli = pars["delta"] * oli

    # Nkx
    # Activation by Gli
    aux_1 = 1. + f_A * pars["K_Gli_Nkx"] * glia
    aux_1 /= 1. + pars["K_Gli_Nkx"] * ( glia + glir )
    # Repression by Pax
    aux_2 = 1./(1. + pars["K_Pax_Nkx"] * pax)**2
    # Repression by Olig
    aux_3 = 1./(1. + pars["K_Oli_Nkx"] * oli)**2
    # Repression by Irx
    aux_4 = 1./(1. + pars["K_Irx_Nkx"] * irx)**2
    p_nkx = pars["alpha_Nkx"] * Hill(C_Pol * pars["K_Pol_Nkx"]  * aux_1 * aux_2 * aux_3 * aux_4)
    d_nkx = pars["delta"] * nkx

    # Irx
    # Repression by Olig
    aux_1 = 1./(1. + pars["K_Oli_Irx"] * oli)**2
    # Repression by Nkx
    aux_2 = 1./(1. + pars["K_Nkx_Irx"] * nkx)**2
    p_irx = pars["alpha_Irx"] * Hill(C_Pol * pars["K_Pol_Irx"]  * aux_1 * aux_2)
    d_irx = pars["delta"] * irx
        
    return np.array([p_pax, p_oli, p_nkx, p_irx]), np.array([d_pax, d_oli, d_nkx, d_irx])



def sigmoid (y, eps=0.01):
    return 1./(1. + np.exp(-y/eps))

def poni_target (x, eps=0.01):
    if not isinstance(x, np.ndarray):
    	_x = np.array([x])
    else:
    	_x = x

    b1 = 1./3. # ventral boundary
    b2 = 2./3. # dorsal boundary

    _target = np.empty((4, len(_x)), dtype=float)
    _target[0] = sigmoid(_x - 0.5*(b1 + b2), eps=(b2-b1)/5)
    _target[1] = sigmoid(_x - b1, eps=eps)*sigmoid(b2 - _x, eps=eps)
    _target[2] = sigmoid(b1 - _x, eps=eps)
    _target[3] = sigmoid(_x - b2, eps=eps)

    return np.squeeze(_target)

def f_signal (t, x, s, kappa=0.1, lam=0.15, alpha=0., var=0.0001, x0=-.1):
    '''
    Time derivative of extracellular signal at position x

    '''
    D = kappa * lam*lam # 0.15^2
    J0 = kappa * 2 * lam * np.exp(np.abs(x0)/lam)
    _var = 2*D*t + var
    num = np.exp( - kappa * t - (x - x0)*(x - x0) / 2 / _var )
    den = np.sqrt(2. * np.pi * _var)
    return - alpha * s + J0 * num/den


def f_memory (s, ms, rates):
    assert ms.shape[0] == rates.shape[0], "input vectors must have the same shape"
    
    if len(ms.shape) == 1:
        ms = ms.reshape(-1,1)
        rates = rates.reshape(-1,1)

    ms_ = np.zeros_like(ms)
    ms_[0] = s
    ms_[1:] = ms[:-1]
    ms_ = rates*ms_

    return ms_ - ms

import matplotlib.colors as mcol
poni_colors_dict = {
    "Shh":"#000000",    # black
    "Ptch": "#8D8D8D",  # grey
    "GliFL": "#8900FF", # purple
    "GliA": "#00AD2F",  # green
    "GliR": "#D02B09",  # red
    "Pax": "#0040FF",   # blue
    "Olig": "#FF00EB",  # magenta
    "Nkx": "#02DDD3",   # cyan
    "Irx": "#DDB002"    # gold
}
poni_colors_array = np.array([mcol.to_rgb(poni_colors_dict[gene]) for gene in ['Pax', 'Olig', 'Nkx', 'Irx']])

