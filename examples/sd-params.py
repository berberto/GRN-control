import numpy as np

kappas = [0.2, 0.5, 1.0, 2.0]
lambdas= [0.15, 0.3, 0.45]
sizes  = [1000, 2000, 5000, 10000, 50000]

kk, ll, ss = np.meshgrid(kappas, lambdas, sizes)

with open("sd-params.txt", "w") as f:
	for (k,l,s) in zip(kk.ravel(), ll.ravel(), ss.ravel()):
		f.write(f"{k}\t{l}\t{s}\n")


