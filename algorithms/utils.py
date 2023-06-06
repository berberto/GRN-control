import numpy as np
import matplotlib.pyplot as plt
from matplotlib import use
use('Agg')


def plot_avg_scores(scores, filename, win=100):
    if not isinstance(scores, np.ndarray):
        scores = np.ndarray(scores)
    assert len(scores.shape) in [1,2], \
           "'scores' should be a numpy array of rank 1 or 2"

    fig, ax = plt.subplots()
    ax.set_title("Learning curve")
    ax.set_xlabel("Episode")
    ax.set_ylabel(f"Score (avg over last {win} episodes)")

    variance = True
    if len(scores.shape) == 1:
        scores = scores.reshape((1, len(scores)))
        variance = False

    episodes = np.arange(len(scores[0]), dtype=int) + 1

    avg = np.zeros_like(scores)
    # for each run
    for run, run_scores in enumerate(scores):
        # calculate averages over last 'win' episodes
        avg[run] = np.array([np.mean(run_scores[i-win:i+1]) for i, score in enumerate(run_scores)])
        # plot the running average
        ax.plot(episodes, avg[run], lw=1)
    # it more than 1 run, calculate the variance and shade area within std dev from mean
    avg_runs = np.mean(avg, axis=0)
    if variance:
        std_runs = np.std(avg, axis=0)
        ax.fill_between(episodes, avg_runs - std_runs, avg_runs + std_runs, color='k', alpha=0.3)
    # plot the average over runs
    ax.plot(episodes, avg_runs, lw=4, color='k')
    plt.savefig(f"{filename}",bbox_inches='tight')



if __name__ == "__main__":

    filename = "test_plot.png"
    x = np.linspace(0, 2*np.pi, 100, endpoint=False)
    scores = np.sin(x)
    scores = np.tile(scores, (5,1))
    scores += np.random.normal(size=scores.shape) * .3
    plot_avg_scores(scores, filename)
