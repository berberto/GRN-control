from gym.envs.registration import register

max_episode_steps=1000

############################################################
#
#   PONI MDP
#
##########################################

# Nkx target

register(
    id='PONI-Nkx-v0', # random init, deterministic, Olig target
    entry_point='grn_control.envs:PONINetwork',
    max_episode_steps=max_episode_steps,
)

register(
    id='PONI-Nkx-v1', # random init, stochastic, Olig target
    entry_point='grn_control.envs:PONINetworkNoise',
    max_episode_steps=max_episode_steps,
)

register(
    id='PONI-Nkx-v2', # localized init, deterministic, Olig target
    entry_point='grn_control.envs:PONINetworkHard',
    max_episode_steps=max_episode_steps,
)

register(
    id='PONI-Nkx-v3', # localized init, stochastic, Olig target
    entry_point='grn_control.envs:PONINetworkHardNoise',
    max_episode_steps=max_episode_steps,
)

##########################################

# Olig target

register(
    id='PONI-Oli-v0', # random init, deterministic, Nkx target
    entry_point='grn_control.envs:PONINetwork_Olig',
    max_episode_steps=max_episode_steps,
)

register(
    id='PONI-Oli-v1', # random init, stochastic, Nkx target
    entry_point='grn_control.envs:PONINetworkNoise_Olig',
    max_episode_steps=max_episode_steps,
)

register(
    id='PONI-Oli-v2', # localized init, deterministic, Nkx target
    entry_point='grn_control.envs:PONINetworkHard_Olig',
    max_episode_steps=max_episode_steps,
)

register(
    id='PONI-Oli-v3', # localized init, stochastic, Nkx target
    entry_point='grn_control.envs:PONINetworkHardNoise_Olig',
    max_episode_steps=max_episode_steps,
)


############################################################
#
#   PONI with PARTIAL OBSERVABILITY (ONLY OLIG AND NKX)
#

##########################################

# Nkx target

register(
    id='PONI-partial-Nkx-v0', # random init, deterministic, Olig target
    entry_point='grn_control.envs:PONINetwork_partial',
    max_episode_steps=max_episode_steps,
)

register(
    id='PONI-partial-Nkx-v1', # random init, stochastic, Olig target
    entry_point='grn_control.envs:PONINetworkNoise_partial',
    max_episode_steps=max_episode_steps,
)

register(
    id='PONI-partial-Nkx-v2', # localized init, deterministic, Olig target
    entry_point='grn_control.envs:PONINetworkHard_partial',
    max_episode_steps=max_episode_steps,
)

register(
    id='PONI-partial-Nkx-v3', # localized init, stochastic, Olig target
    entry_point='grn_control.envs:PONINetworkHardNoise_partial',
    max_episode_steps=max_episode_steps,
)

##########################################

# Olig target

register(
    id='PONI-partial-Oli-v0', # random init, deterministic, Nkx target
    entry_point='grn_control.envs:PONINetwork_Olig_partial',
    max_episode_steps=max_episode_steps,
)

register(
    id='PONI-partial-Oli-v1', # random init, stochastic, Nkx target
    entry_point='grn_control.envs:PONINetworkNoise_Olig_partial',
    max_episode_steps=max_episode_steps,
)

register(
    id='PONI-partial-Oli-v2', # localized init, deterministic, Nkx target
    entry_point='grn_control.envs:PONINetworkHard_Olig_partial',
    max_episode_steps=max_episode_steps,
)

register(
    id='PONI-partial-Oli-v3', # localized init, stochastic, Nkx target
    entry_point='grn_control.envs:PONINetworkHardNoise_Olig_partial',
    max_episode_steps=max_episode_steps,
)


############################################################
#
#   PONI with EXTRACELLULAR SIGNAL
#

## FULL OBSERVABILITY OF THE STATE
# ENV = "PONI-signal-Nkx-v0" # Nkx target, stochastic, localized init
register(
    id='PONI-signal-Nkx-v0', # localized init, stochastic, Nkx target
    entry_point='grn_control.envs:PONINetwork_Signal_Nkx',
    max_episode_steps=max_episode_steps,
)
# ENV = "PONI-signal-Oli-v0" # Nkx target, stochastic, localized init
register(
    id='PONI-signal-Oli-v0', # localized init, stochastic, Oli target
    entry_point='grn_control.envs:PONINetwork_Signal_Oli',
    max_episode_steps=max_episode_steps,
)
## PARTIAL OBSERVABILITY OF THE STATE
# ENV = "PONI-signal-Nkx-v1" # Nkx target, stochastic, localized init
register(
    id='PONI-signal-Nkx-v1', # localized init, stochastic, Nkx target
    entry_point='grn_control.envs:PONINetwork_Signal_Nkx_partial',
    max_episode_steps=max_episode_steps,
)
# ENV = "PONI-signal-Oli-v1" # Nkx target, stochastic, localized init
register(
    id='PONI-signal-Oli-v1', # localized init, stochastic, Oli target
    entry_point='grn_control.envs:PONINetwork_Signal_Oli_partial',
    max_episode_steps=max_episode_steps,
)



max_episode_steps=500

# MULTI-AGENT!
# ENV = "PONI-pattern-v0" # Pattern target, stochastic, localized init
register(
    id='PONI-pattern-v0', # localized init, stochastic, pattern target
    entry_point='grn_control.envs:PONINetwork_Signal_Pattern',
    max_episode_steps=max_episode_steps,
)

# MULTI-AGENT!
# ENV = "PONI-pattern-v1" # Pattern target, stochastic, localized init
register(
    id='PONI-pattern-v1', # localized init, stochastic, pattern target
    entry_point='grn_control.envs:PONINetwork_Signal_Pattern_partial',
    max_episode_steps=max_episode_steps,
)

# MULTI-AGENT!   DYNAMIC SIGNAL
# ENV = "PONI-pattern-v2" # Pattern target, stochastic, localized init
register(
    id='PONI-pattern-v2', # localized init, stochastic, pattern target
    entry_point='grn_control.envs:PONINetwork_Diffusion_Pattern',
    max_episode_steps=max_episode_steps,
)

# MULTI-AGENT!   DYNAMIC SIGNAL
# ENV = "PONI-pattern-v3" # Pattern target, stochastic, localized init
register(
    id='PONI-pattern-v3', # localized init, stochastic, pattern target
    entry_point='grn_control.envs:PONINetwork_Diffusion_Pattern_partial',
    max_episode_steps=max_episode_steps,
)

# MULTI-AGENT!   DYNAMIC SIGNAL - WITH MEMORY VARIABLES
# ENV = "PONI-pattern-v3" # Pattern target, stochastic, localized init
register(
    id='PONI-pattern-v4', # localized init, stochastic, pattern target
    entry_point='grn_control.envs:PONINetwork_Diffusion_Memory',
    max_episode_steps=max_episode_steps,
)



register(
    id='PONI-pattern-v5', # localized init, stochastic, pattern target
    entry_point='grn_control.envs:PONINetwork_SD_Pattern',
    max_episode_steps=max_episode_steps,
)

register(
    id='PONI-pattern-v6', # localized init, stochastic, pattern target
    entry_point='grn_control.envs:PONINetwork_SD_Memory',
    max_episode_steps=max_episode_steps,
)

register(
    id='PONI-pattern-v7', # localized init, stochastic, pattern target
    entry_point='grn_control.envs:PONINetwork_SD_Memory_Feedback',
    max_episode_steps=max_episode_steps,
)