from neurips23.filter.base import BaseFilterANN
from benchmark.datasets import DATASETS
import numpy as np
import random


# L2 distance
def distance(p1, p2):
    return np.linalg.norm(np.array(p1) - np.array(p2))

# Random projection
def random_projection(dim):
    return [random.gauss(0, 1) for _ in range(dim)]

# Hash function
def compute_hash(projection, point):
    return int(np.dot(projection, point))


class LSHANN(BaseFilterANN):

    def __init__(self, L, K) -> None:
        super().__init__()
        self.L = L # number of hash tables
        self.K = K # number of refence points for each hash table
        self.dim = None   # dimension of data point: set in fit()
        self.data = None  # training data: set in fit()
        self.hash_tables = None # hash tables: set in fit()

    def fit(self, dataset):
        ds = DATASETS[dataset]()
        data = ds.get_dataset() # list of list of floats
        dim = ds.d
        print(f"data size: {len(data)} dim: {dim}")
        print("train...")
        self.dim = dim
        self.data = data
        self.build_lsh(data)
        print("done.")

    
    # LSH preprocessing
    def build_lsh(self, data):
        L, K = self.L, self.K
        dim = len(data[0])
        hash_tables = []

        for i in range(L): # for each hash 
            print(f"building hash {i}...")
            projections = [random_projection(dim) for _ in range(K)] # create K reference points
            table = {}
            for point in data:
                # compute distance of point to K reference points 
                key = tuple(compute_hash(proj, point) for proj in projections)
                key = tuple(key) # Convert key to tuple to be hashable
                if key not in table:
                    table[key] = []
                table[key].append(point)
            hash_tables.append((projections, table))
            print(f"Hash {i} done: {len(table)} buckets")
        self.hash_tables = hash_tables
    
    # LSH query
    def query(self, query_point):
        potential_neighbors = set()

        for projections, table in self.hash_tables:
            # compute distance of query point to K reference points
            key = tuple(compute_hash(proj, query_point) for proj in projections)
            key = tuple(key)  # Convert to tuple
            if key in table:
                for pt in table[key]:
                    potential_neighbors.add(tuple(pt))
        print(f"find {len(potential_neighbors)}!")
        # If we don't find any potential neighbors in the hashed buckets
        # we'll consider all the data points as potential neighbors
        if not potential_neighbors:
            potential_neighbors = set(map(map, self.data))

        closest = min(potential_neighbors, key=lambda x: distance(x, query_point))
        return closest
    
    def sanity_check(self):
        print("quick test")
        noise_scale = 1e-5
        noise = noise_scale * np.random.randn(self.dim)
        print("data0: ", self.data[0])
        query0 = self.data[0] + noise
        closest = self.query(query0)
        distance0 = distance(self.data[0], closest)
        print("closest: ", closest)
        print("distance: ", distance0)


    def filtered_query(self, X, filter, k):
        """
        Carry out a batch query for k-NN of query set X with associated filter.
        Query X[i] has asks for k-NN in the index that pass all filters in filter[i].
        """
    
