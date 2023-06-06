import os

n_agents = 81
d_memory = 2

versions = [6]#, 7]
kappas = [0.5]
lambdas = [0.15, 0.30]
sizes = [10000]

wds = [0.0]#, 1.e-6, 1.e-4]

seeds = [58915, 55052, 21178]#, 37855, 17750, 26328, 58516,  7511, 42313, 26898]

run = False

with open('poni-pattern-params.txt', 'w') as f:
	os.chdir("examples")
	for version in versions:
		env = f"PONI-pattern-v{version:d}"
		for kappa in kappas:
			for size in sizes:
				for lam in lambdas:
					for wd in wds:
						for seed in seeds:
								pars_str = f"{env}   {n_agents:d}   {kappa:.2f}   {size:06d}   {lam:.2f}   {d_memory:d}   {wd:.1e}   {seed:d} "
								print(pars_str)
								f.write(pars_str+"\n")
								if run:
									os.system(f"python poni_td3_pattern.py {pars_str}")
