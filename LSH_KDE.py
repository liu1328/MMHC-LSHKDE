import math
import numpy as np

def Gaussian_kernel(x, y, bandwidth):
    "Gaussian kernel"
    return np.exp(-1. * (np.linalg.norm((x - y), ord=2))**2 / (2 * bandwidth ** 2))


def kde(query, dataset, bandwidth):
    """By iterating over each data point in the dataset,
    the Gaussian_kernel function calculates the kernel density estimate between the query point and the data point,
    and then averages these estimates to return the final kernel density estimate"""
    return np.mean([Gaussian_kernel(query, dataset[i], bandwidth) for i in range(dataset.shape[0])])

#np.random.seed(10)

class GaussianLSH:
    """A Local Sensitive Hash (LSH) class that uses a Gaussian distribution to map data points to hash values."""
    def __init__(self, dimension, bandwidth):
        poisson_param = dimension * 1. / bandwidth
        self.reps = np.random.poisson(poisson_param)
        self.axes = np.random.randint(0, dimension, self.reps)
        self.thresholds = np.random.uniform(0, 1, self.reps)

    def hash(self, point):
        """This method takes a data point as input,
         and then maps the data point to a hash value (a tuple) based on each dimension of the data point, the selected dimension,
        and the corresponding threshold, and returns the hash value."""
        return tuple([point[self.axes[i]] < self.thresholds[i] for i in range(self.reps)])

class BinningGaussianLSH:
    """This is used to map data points to hashes."""
    def __init__(self, dimension, bandwidth):
        self.dimension = dimension
        delta = np.random.gamma(2, bandwidth, dimension)
        self.thresholds = []
        for d in range(dimension):
            bin_length = delta[d]
            shift = np.random.uniform(0, bin_length)
            t_list = []

            while shift < 1:
                t_list.append(shift)
                shift += bin_length
            self.thresholds.append(t_list)

    def hash(self, point):
        """This method takes a data point as input,
        and then maps the data point to a hash value (a tuple) based on each dimension of the data point, the selected dimension,
        and the corresponding threshold, and returns the hash value."""
        return tuple([len([t for t in self.thresholds[d] if t < point[d]]) for d in range(self.dimension)])


class FastGaussianKDE:
    def __init__(self, dataset, bandwidth, L):
        """
        dataset: A dataset where each row is a data point
        bandwidth: Bandwidth parameter in kernel density estimation
        L: The number of LSH hash functions
        """
        self.n_points = dataset.shape[0]
        self.dimension = dataset.shape[1]
        self.bandwidth = bandwidth
        self.L = L
        
        self.sizes = np.random.binomial(self.n_points, L * 1. / self.n_points, self.L)
        random_samples = [np.random.choice(self.n_points, self.sizes[j], replace=False) for j in range(self.L)]

        # The LSH hash function is generated based on the size of the bandwidth parameter.
        if bandwidth >= 1:
            self.lshs = [GaussianLSH(self.dimension, 2 * self.bandwidth) for i in range(self.L)]
        else:
            self.lshs = [BinningGaussianLSH(self.dimension, 2 * self.bandwidth) for i in range(self.L)]

        self.hashed_points = []
        for j in range(L):
            bins = defaultdict(list)
            for i in random_samples[j]:
                point = dataset[i, :]
            self.hashed_points.append(bins)

    def kde(self, query):
        estimators = []
        dimension = len(query)
        for j in range(self.L):
            # This hash value is used to determine which hash bucket the query point is in.
            query_hash = self.lshs[j].hash(query)
            bin_size = len(self.hashed_points[j][query_hash])

            # If there are no data points in the hash bucket, the estimate is 0.
            if bin_size == 0:
                estimators.append(0)
            else:
                point = self.hashed_points[j][query_hash][np.random.randint(bin_size)]
                estimators.append(Gaussian_kernel(query, point, 2*self.bandwidth) * bin_size * 1. / self.L)

        return np.mean(estimators)/(((2*math.pi)**(dimension/2))*bandwidth**dimension)

