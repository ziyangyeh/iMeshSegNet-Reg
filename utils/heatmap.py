import numpy as np


def gaussian(pt, mu_pt, H=1, sigma=0.5, axis=0):
    exponent = -((pt - mu_pt) ** 2).sum(axis) / (2 * sigma**2)
    result = H * np.exp(exponent)
    return result


if __name__ == "__main__":
    pt = np.random.randn(100, 3)
    print(pt)
    mu = np.array([np.inf, np.inf, np.inf])
    print(gaussian(pt, mu, axis=1))
    pt = np.array([1, 2, 3])
    mu = np.array([1, 2, 3.5])
    print(gaussian(pt, mu))
