import matplotlib.pyplot as plt
import numpy as np


def eps_relu (x, eps=1.):
	assert eps > 0., "invalid 'eps' parameter"
	return eps*np.log(1. + np.exp(x / eps))

def eps_relu_target (x, target, tol, eps):
	'''
	x -> (d,) array
	target -> (d,) array
	'''
	delta = np.abs(x - target)
	return np.sum(eps_relu(delta - tol, eps=eps))

def eps_resqu_target (x, target, tol, eps):
	'''
	x -> (d,) array
	target -> (d,) array
	'''
	delta = (x - target)*(x - target)
	return np.sum(eps_relu(delta - tol**2, eps=eps))

def quadratic_target (x, target, tol, eps):
	'''
	x -> (d,) array
	target -> (d,) array
	'''
	delta_sq = (x - target)*(x - target)
	return np.sum(delta_sq)

def eps_relu_target_pattern (x, target, tol, eps):
	'''
	x -> (d, N) array
	target -> (d, N) array
	'''
	delta = np.abs(x - target)
	return np.sum(eps_relu(delta - tol, eps=eps), axis=0)

def eps_resqu_target_pattern (x, target, tol, eps):
	'''
	x -> (d, N) array
	target -> (d, N) array
	'''
	delta = (x - target)*(x - target)
	return np.sum(eps_relu(delta - tol**2, eps=eps), axis=0)

def quadratic_target_pattern (x, target, tol, eps):
	'''
	x -> (d, N) array
	target -> (d, N) array
	'''
	delta_sq = (x - target)*(x - target)
	return np.sum(delta_sq, axis=0)


if __name__ == "__main__":

	fig, axs = plt.subplots(1,3,figsize=(12,3))

	for cost, ax in zip([eps_relu_target, eps_resqu_target,quadratic_target], axs.ravel()):

		target = np.array([1., 0.])
		x = np.linspace(-2,2,100)
		xx, yy = np.meshgrid(x,x)
		tol = np.array([.4, .1])

		x_f, y_f = xx.flatten(), yy.flatten()

		zz = np.array([cost(np.array([x,y]), target, tol, 0.4) for x,y in zip(x_f,y_f)]).reshape(xx.shape)

		im = ax.contourf(xx, yy, zz, 20)
		fig.colorbar(im, ax=ax)

	plt.show()