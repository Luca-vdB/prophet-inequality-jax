import matplotlib.pyplot as plt


def plot_iid_instance(realization, probability, label):
    fig, ax = plt.subplots()  # Create a figure and axes object
    ax.scatter(realization.tolist(), probability.tolist(), label=label, marker="x")
    ax.legend()
    ax.set_xscale('log')
    ax.set_xlabel("realization")
    ax.set_ylabel("probability")
    return fig 


def plot_non_iid_instance(realizations, probabilities, competitive_ratio=""):
    fig, ax = plt.subplots()  # Create a figure and axes object
    for i in range(realizations.shape[0]):
        ax.scatter(realizations[i].tolist(), probabilities[i].tolist(), marker="x", label=f"Box {i}")
    
    ax.legend()
    ax.set_xscale('log')
    ax.set_xlabel("realization")
    ax.set_ylabel("probability")
    ax.set_title(f"Instance for semi online i.i.d with approx factor {competitive_ratio:.6f}")
    ax.grid()
    return fig



