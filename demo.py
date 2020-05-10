import numpy as np

# Plotting:
import matplotlib  # noqa F401
import matplotlib.pyplot as plt

# Import class
from scipy.stats import multivariate_normal
from multivariate_laplace import multivariate_laplace


def standard_anlyse(clsx=multivariate_normal):
    """Standard analyse of the multivariate distribution (suitable for normal
       and laplace). Prints and plots typical analysis figures.

    Parameters
    ----------
    cslx :
        Instance for multivariate generation.
    """

    x, y = np.mgrid[-1:1:.01, -1:1:.01]
    pos = np.empty(x.shape + (2,))
    pos[:, :, 0] = x
    pos[:, :, 1] = y
    rv = clsx([0.5, -0.2], [[2.0, 0.3], [0.3, 0.5]])

    # Plotting PDF:
    fig, ax = plt.subplots()
    ax.set(xlabel='x', ylabel='y', title='2D plot of PDF')
    plt.contourf(x, y, rv.pdf(pos))
    fig.savefig(f"plots/{clsx.__class__.__name__}_PDF_2D.png")
    plt.show()

    # Assigning:
    pdf = rv.pdf(pos)
    cdf = rv.cdf(pos)

    # Debug printing
    print(pdf)
    print(pdf.shape)

    # Plot 1D
    point_idx_sample = 130
    fig, ax = plt.subplots()
    ax.plot(x, pdf[point_idx_sample])
    ax.set(xlabel='x', ylabel='y',
           title=f'1D sample at point {point_idx_sample}')
    ax.grid()
    fig.savefig(f"plots/{clsx.__class__.__name__}_1D.png")
    plt.show()

    # Plot CDF (1D)
    point_idx_sample = 130
    fig, ax = plt.subplots()
    ax.plot(x, cdf[point_idx_sample])
    ax.set(xlabel='x', ylabel='y', title=f'1D CDF at point {point_idx_sample}')
    ax.grid()
    fig.savefig(f"plots/{clsx.__class__.__name__}_CDF_1D.png")
    plt.show()

    # Plotting Samples:
    # parameters
    nsamp = 200
    seed = 1973
    np.random.seed(seed)
    mu = 5.0 * np.random.random(nsamp)
    sigma = np.array(np.diag(np.ones(nsamp)), dtype=np.float64)
    # Generate samples:
    rvs = clsx.rvs(mean=mu, cov=sigma, size=nsamp)
    # plotting
    fig, ax = plt.subplots()
    ax.set(xlabel='x', ylabel='y', title='2D plot of Samples')
    plt.contourf(x, y, rvs)
    fig.savefig(f"plots/{clsx.__class__.__name__}_RVS_2D.png")
    plt.show()


# Plot the Multivariate Normal (Gauss) distribution version
standard_anlyse(multivariate_normal)

# Plot the Multivariate Double Exponential (Laplace) distribution version
standard_anlyse(multivariate_laplace)
