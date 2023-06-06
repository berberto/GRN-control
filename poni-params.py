import os

targets = ['Nkx', 'Oli']

versions = [3]

alphas = [0.0001] # [0.0001]

betas = [0.00001] # [0.00001]

seeds = [58915, 55052, 21178, 37855, 17750]#, 26328, 58516,  7511, 42313, 26898]

gammas = [0.990]#, 0.995]

wds = [0.0, 0.0001]

run = False

with open('poni-params.txt', 'w') as f:
	os.chdir("examples")
	for target in targets:
		for version in versions:
			env = f"PONI-{target}-v{version:d}"
			for alpha, beta in zip(alphas,betas):
				# for beta in betas:
				for gamma in gammas:
					for wd in wds:
						for seed in seeds:
							pars_str = f"{env}   {alpha:.6f}   {beta:.6f}   {gamma:.3f}   {wd:.1e}   {seed:d}"
							print(pars_str)
							f.write(pars_str+"\n")
							if run:
								os.system(f"python poni_td3.py {pars_str}")
