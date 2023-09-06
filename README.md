# Reinforcement learning solution for optimal feedback in a GRN


[![DOI](https://zenodo.org/badge/638884312.svg)](https://zenodo.org/badge/latestdoi/638884312)


### Setup 

1. Clone the repository:
```bash
git clone https://github.com/berberto/GRN-control
cd GRN-control
```

2. Create and activate the *Conda* environment.
```bash
conda env create -f environment.yml
conda activate torch-rl
```

3. Install *PyTorch*, following the instructions [on the website](), as per your system. The code has been tested on Linux systems with CUDA 18 support:
```bash
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
```

4. Install the `grn_control` package.
```bash
make
```
This include all the necessary gym environments and the `algorithms` module, with various RL algorithms coded.


### Run

The main main codes for the GRN control, and for the 1D lanscape problems are in the `examples` directory.

0. Remember to activate the *Conda* environment:
```bash
conda activate torch-rl
```

1. Generate the parameters for the simulations
```bash
python poni-params.py             	# for the single-cell GRN control
python poni-pattern-params.py     	# for the multi-agent (patterning) GRN control
```
This produces two files (`poni-params.txt` and `poni-pattern-params.txt`) containing a list of strings with command-line input parameters to pass to the main scripts.

2. To run the code:

	a. You can copy any line on this file and run
	```bash
	cd examples/
	python poni_td3.py  [string of parameters]          	# for the single-cell GRN control
	python poni_td3_pattern.py   [string of parameters] 	# for the multi-agent (patterning) GRN control
	```
	The data in the main text are produced by running[^1]
	```bash
	python poni_td3.py  PONI-Nkx-v3   0.000010   0.000001   0.990   0.0e+00   21178
	python poni_td3_pattern.py PONI-pattern-v6   81   0.50   010000   0.15   2   0.0e+00   21178
	```

	b. If you have access to a cluster with *SLURM*, you can submit job arrays with parameters generated above, by running
	```bash
	sbatch poni.slurm					# for the single-cell GRN control
	sbatch poni-pattern.slurm			# for the multi-agent (patterning) GRN control
	```

[^1]: The seed for random number are given (last command-line input), but some low-level PyTorch routines used in the optimisation seem to depend on the specific installation.
