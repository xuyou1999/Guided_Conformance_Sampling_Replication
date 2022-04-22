import collections

import feature_encodings
from feature_encodings import create_feature_encoding_from_index
from itertools import combinations
from tqdm import tqdm
import sys
import numpy as np
import random
import time




class FeatureBasedPartitioning:
    def __init__(self):
        pass

    # TODO properly include index file
    # returns indices of traces in log
    def partition(self, log, index_file=None, verbose=False):
        preprocessing_time = time.time()
        if verbose:
            print("Partitionining Log")
        data, feature_names = create_feature_encoding_from_index(log, considered_feature_types=feature_encodings.feature_types.ALL, index_name=index_file)
        #old method using indices based on log names
        #data, feature_names = create_feature_encoding(log, logname, considered_feature_types=feature_encodings.feature_types.ALL)
        partitioned_log = {}
        for idx, encoded_trace in enumerate(data):
            for i in range(len(feature_names)):
                if encoded_trace[i] != 0:
                    if feature_names[i] in partitioned_log:
                        partitioned_log[feature_names[i]].append(idx)
                    else:
                        partitioned_log[feature_names[i]] = [idx]

        del_list=[]

        #remove partitions with singular items
        for partition in partitioned_log.keys():
            if len(partitioned_log[partition]) <= 1:
                del_list.append(partition)
        for partition in del_list:
            if verbose:
                print(f"Pruning away {partition} from partitioned log")
            del partitioned_log[partition]

        if verbose:
            print(f"There are {len(partitioned_log)} features in the event log")

        return partitioned_log, time.time()-preprocessing_time


class SequenceBasedLogPreprocessor:
    """
    Computes candidate pairs of likely similar traces using MinHashing and Locality Sensitive Hashing (LSH).
    The (approximated) distance metric is the Jaccard-Index.
    """

    def __init__(self, log, k=3, b=10, r=10, p=5, verbose=False):
        """
        Initializes the partitioning/bucketing parameters for the MinHash and LSH algorithm.
        params:
            log: The log to process.
            k: The size of k-grams to consider for each trace.
            b: The number of bands to use for the LSH-algorithm.
            r: The number of rows per band for the LSH-algorithm.
        note:
            The resulting number of hash functions will be n = b * r,
            which yields a similarity threshold for candidate pairs of ~(1/b)^(1/r)
            with an expected error of O(1/sqrt(n).
        """
        start = time.time()

        self.p = p
        self.k = k
        self.b = b
        self.r = r
        self.n_sig = b * r
        self.hash_params = None
        self.sig_matrix = None
        self.hash_buckets = None

        self.traces = [self._get_activity_sequence(t) for t in log]
        self.trace_k_grams = [self._get_k_grams(t, k) for t in self.traces]

        self.time = time.time()

        # Compute a dict mapping any k-gram in the event log to a unique integer index
        k_gram_set = set.union(*self.trace_k_grams)
        self.k_gram_dict = dict(zip(list(k_gram_set), range(len(k_gram_set))))

        if verbose:
            print(f"There are {len(k_gram_set)} unique k-grams in the event log for p={self.p}")

        self.init_minhash_lsh(b, r, verbose)

        self.time = time.time() - self.time

        # TODO: Compute based on options set
        self.partitioned_log = collections.defaultdict(list)
        for trace_idx, trace_k_grams in enumerate(self.trace_k_grams):
            for k_gram in trace_k_grams:
                self.partitioned_log[k_gram] += [trace_idx]

        del_list=[]
        for partition in self.partitioned_log.keys():
            if len(self.partitioned_log[partition]) < p:
                del_list.append(partition)
        for partition in del_list:
            if verbose:
                print(f"Pruning away {partition} from partitioned log")
            del self.partitioned_log[partition]


        if verbose:
            print(f"Log preprocessing done. (Total time elapsed: {time.time()-start:.3f}s)")

    def init_minhash_lsh(self, b=10, r=10, verbose=False):
        """
        Initializes the MinHash signature matrix and hashes the traces into buckets.
        params:
            b: The number of bands to use.
            r: The number of rows per band to use.
        note:
            The resulting number of hash-functions is n_sig = b * r.
        """
        self.b = b
        self.r = r
        self.n_sig = b * r

        # Sets the random parameters (a,b) of the hash function h(x) = (a*x+b) mod n_k_grams
        # where n_sig is the number of hash function parameters to generate
        n_k_grams = len(self.k_gram_dict)
        self.hash_params = np.random.choice(self._get_primes(n_k_grams), size=[self.n_sig, 2])

        # Initialize signature matrix with all entries set to np.inf
        sig = np.full((self.n_sig, len(self.trace_k_grams)), np.inf)

        len_pbar = len(self.trace_k_grams)

        # Iterate over all sets of k-grams (one set per trace)
        with tqdm(total=len_pbar, desc="Initializing...", file=sys.stdout, disable=not verbose) as pbar:
            pbar.set_description("Initializing signature matrix...")
            for i, k_grams in enumerate(self.trace_k_grams):
                # Iterate over the k-grams (actually) appearing in the trace
                for k_gram in k_grams:
                    old_row = self.k_gram_dict[k_gram]

                    # Compute permuted row indices (for each of the n_sig signatures)
                    new_rows = (self.hash_params @ np.array([old_row, 1])) % n_k_grams

                    # Update minhash signature of trace j
                    sig[:, i] = np.minimum(sig[:, i], new_rows)

                pbar.update()

            # The minhash signature matrix of shape (n_sig, len(traces))
            self.sig_matrix = sig.astype(int)

            pbar.set_description("Computing LSH hash buckets...")
            n, d = self.sig_matrix.shape
            assert (n == b * r)
            self.hash_buckets = collections.defaultdict(set)
            bands = np.array_split(self.sig_matrix, b, axis=0)
            for i, band in enumerate(bands):
                for j in range(d):
                    # We add the [str(i)] to the id to only hash signatures to the same bucket
                    # within the same band i
                    band_id = tuple(list(band[:, j]) + [str(i)])
                    self.hash_buckets[band_id].add(j)
                pbar.update()

    def _get_activity_sequence(self, trace):
        """
        Transforms a trace to a sequence of activities.
        """
        return [*map(lambda x: x['concept:name'], trace)]

    def _get_k_grams(self, trace, k=3):
        """
        Returns the k-grams (aka k-shingles) of an activity sequence.
        """
        k_grams = set()
        for i in range(len(trace) - k + 1):
            k_grams.add(tuple(trace[i:i + k]))

        # Fallback for traces that are shorter than k
        if len(k_grams) == 0 and k > 1:
            return self._get_k_grams(trace, k - 1)
        return k_grams

    def get_ordered_k_grams(self, trace_index, k):
        """
        Returns the k-grams (aka k-shingles) of an activity sequence, in the correct order.
        """
        trace = self.traces[trace_index]
        k_grams = []
        for i in range(len(trace) - k + 1):
            k_grams.append(tuple(trace[i:i + k]))

        # Fallback for traces that are shorter than k
        if len(k_grams) == 0 and k > 1:
            return self.get_ordered_k_grams(trace_index, k - 1)
        return k_grams

    def get_ranged_k_grams(self, trace_index, start, end):
        """
        Returns the set of k-grams of a trace within the range [start, end).
        params:
            trace_index: Index of trace to get k-grams from.
            start: Start index of trace to get k-grams from (inclusive).
            end: End index of trace to get k-grams from (exclusive).
        note:
            If the given range is shorter than k, the subsequence [start, end) is returned instead.
        """

        if end <= start:
            print("Error: Invalid range.")
        elif end - start <= self.k:
            result = set()
            result.add(tuple(self.traces[trace_index][start:end]))
            return result
        else:
            k_grams = set()
            for i in range(start, end - self.k + 1):
                k_grams.add(tuple(self.traces[trace_index][i:i + self.k]))
            return k_grams

    def _get_primes(self, n):
        """
        Returns a list of primes < n.
        We will need this later to produce (good) hash functions.
        """
        sieve = [True] * n
        for i in range(3, int(n ** 0.5) + 1, 2):
            if sieve[i]:
                sieve[i * i::2 * i] = [False] * ((n - i * i - 1) // (2 * i) + 1)
        return [2] + [i for i in range(3, n, 2) if sieve[i]]

    def get_lsh_candidates(self, ref_sig):
        """
        Returns the LSH candidates for a given minhash-signature (as a set of indices).
        params:
            ref_signature: Reference minhash-signature to compute LSH candidates for.
        """
        candidates = set()
        ref_sig_bands = np.array_split(ref_sig, self.b, axis=0)
        for i, band in enumerate(ref_sig_bands):
            band_id = tuple(list(band) + [str(i)])
            candidates.update(self.hash_buckets[band_id])
        return candidates

    def get_candidate_pairs(self, init=False, b=20, r=5, verbose=False):
        """
        Returns a set of candidate pair indices (and the corresponding list of traces) with
        likely similar traces. The similarity threshold for candidate pairs is ~(1/b)^(1/r) with
        an expected error of O(1/sqrt(b*r).
        params:
            init: Whether to re-initialize the signature matrix with new parameters b and r.
            b: The number of buckets to use for the MinHash+LSH algorithm (when re-initializing).
            r: The number of rows per buckets for the MinHash+LSH algorithm (when re-initializing).
            verbose: Whether to output log messages.
        """
        if init:
            if self.b == b and self.r == r:
                print("Warning: Re-initializing signature matrix with the same parameters.")
            self.init_minhash_lsh(b, r, verbose)

        candidate_pairs = set()
        for bucket in tqdm(self.hash_buckets.values(), desc="Searching candidate pairs...", disable=not verbose):
            if len(bucket) > 1:
                for pair in combinations(bucket, 2):
                    candidate_pairs.add(pair)

        return self.traces, candidate_pairs

    def get_random_sample(self, sample_size, indices=None):
        """
        Returns a random sample of trace indices, either from the whole log or from a selected set of indices.
        params:
            sample_size: The size of the sample to produce.
            (optional) indices: The indices to sample from.
        """
        if indices:
            if len(indices) <= sample_size:
                return indices

            sample_indices = random.sample(indices, sample_size)
            # sample_indices = np.random.choice(indices, replace=False, size=sample_size)
        else:
            sample_indices = random.sample([*range(len(self.traces))], sample_size)
        return sample_indices

    def get_diverse_sample(self, sample_size, init=False, b=100, r=2, verbose=False):
        """
        Returns a diverse (w.r.t. Jaccard-Similarity) sample of traces (as a list of indices).
        The underlying sampling algorithm based on the FAST-pw technique by
        Miranda et al. (ICSE 2018).
        params:
            sample_size: Size of the sample to produce.
            init: Whether to re-initialize the signature matrix with the given parameters.
            b: Number of bands to use.
            r: Number of rows per band to use.
        note:
        The algorithm aims to always select the next trace s.t. its similarity threshold
        is below ~(1/b)^(1/r) (e.g., for b=100 and r=2: threshold=0.1) w.r.t. all previously
        selected traces. The expected error is O(1/sqrt(b*r)).
        """
        if init:
            if self.b == b and self.r == r:
                print("Warning: Re-initializing signature matrix with the same parameters.")
            self.init_minhash_lsh(b, r, verbose)

        if verbose:
            print("Computing hash buckets...")

        sample = []
        trace_ids = set(range(len(self.traces)))

        # First trace: Choose randomly
        first_trace_id = np.random.choice(list(trace_ids))
        c_sig = self.sig_matrix[:, first_trace_id]
        sample.append(first_trace_id)

        if verbose:
            print("Start sampling...")

        while len(sample) != sample_size:

            # Determine LSH Candidates
            candidates = self.get_lsh_candidates(c_sig)

            if len(candidates) == 0:
                print("TODO: Reset Minhash Signature")
                next_trace_id = np.random.choice(list(trace_ids))
                c_sig = self.sig_matrix[:, next_trace_id]
                sample.append(next_trace_id)
                c_sig_bands = np.array_split(c_sig, b, axis=0)
                for i, band in enumerate(c_sig_bands):
                    band_id = tuple(list(band) + [str(i)])
                    candidates.update(self.hash_buckets[band_id])

            # Candidates to remove: Similar and already selected traces
            sim_candidates = candidates.union(sample)
            div_candidates = trace_ids.difference(sim_candidates)

            # print(len(sim_candidates), len(div_candidates))
            # Select random non-similar trace
            sample_idx = np.random.choice(tuple(div_candidates))

            # Alternative: Determine trace with highest distance (results in some overhead)
            # idx = np.argmin(list(np.mean(c_sig == sig_matrix[:, x]) for x in div_candidates))
            # sample_idx = div_candidates[idx]
            sample.append(sample_idx)

            # Update cumulative signature
            c_sig = np.minimum(c_sig, self.sig_matrix[:, sample_idx])

        return sample

    def get_similar_sample(self, ref_trace_idx, sample_size=None, validate_sim=False, init=False, b=10, r=10, verbose=False):
        """
        Returns a sample of similar traces w.r.t. a reference trace (as a list of indices).
        params:
            ref_trace_idx: Index of reference trace.
            sample size: Max. number of traces to include in the sample.
            validate_sim: Whether to validate similarity estimations.
            init: Whether to re-initialize the signature matrix with the given parameters.
            b: Number of bands to use.
            r: Number of rows per bands.
            verbose: Whether to print log messages.
        note:
            Default params (b=10, r=10) select similar traces with a similarity threshold of ~79%
            with an expected error of 10%.
        """
        if init:
            if self.b == b and self.r == r:
                print("Warning: Re-initializing signature matrix with the same parameters.")
            self.init_minhash_lsh(b, r, verbose)

        sample_indices = []
        ref_sig = self.sig_matrix[:, ref_trace_idx]
        candidates = self.get_lsh_candidates(ref_sig)

        if sample_size and len(candidates) > sample_size:
            # Randomly select sample_size traces
            # TODO: Select most similar traces instead
            # sample_indices = np.random.choice(candidates, size=sample_size, replace=False)
            sample_indices = random.sample(candidates, sample_size)

        else:
            sample_indices.extend(candidates)

        return sample_indices

class FeatureVectorBasedPartitioning:

    def __init__(self):
        pass

    def angular_distance(self, u, v):
        """
        Returns the angular distance between two vectors u and v.
        Angular distance performs better than cosine similarity for smaller angles,
        see: https://math.stackexchange.com/questions/2874940/cosine-similarity-vs-angular-distance
        """
        norm = np.linalg.norm(u) * np.linalg.norm(v)
        cosine = u @ v / norm
        ang = np.arccos(cosine)
        return 1 - ang / np.pi

    def get_candidate_pairs(self, T, b, r, thresh):
        """
        Returns a set of pairs (indices) with the required similarity. Contains no false positives,
        but may omit false negatives.
        Params:
            T: (n_vectors,d) matrix of d-dimensional vectors from which similar pairs are to be found
            b: Number of bands
            r: Number of rows per band
            thresh: Float value [-1,1] determining the required similarity threshold (angular distance)
        """
        n_vectors, d = T.shape
        n_sig = b * r

        # Compute signature matrix
        # Each row represents one feature vector, multiply with matrix of random normal vectors (hyperplanes)
        # to determine on which side of each hyperplane each vector lies (sign of result)
        M = T @ np.random.randn(d, n_sig)
        S = np.where(M > 0, 1, 0)

        # Break signature matrix into bands
        # Note: In contrast to the JaccardSimilarityBasedPartitioning, each row corresponds to a feature vector (trace)
        S = np.split(S, b, axis=1)

        # The idea is now to hash the binary row vectors (for one band) of length r by interpreting
        # the row as a binary number (e.g., (1,0,1) -> 5) and using this as the hash value

        # Column vector to convert binary vector to integer
        binary_column = 2 ** np.arange(r).reshape(-1, 1)

        # Convert each band into a single integer, i.e., convert band matrices to band columns
        S = np.hstack([M @ binary_column for M in S])

        # Every value in the matrix represents a hash bucket assignment, i.e.,
        # each row i contains the buckets the feature vector i (trace) has been hashed into
        d = collections.defaultdict(set)
        with np.nditer(S, flags=['multi_index']) as it:
            for x in it:
                # Add row i (multi_index[0]) to bucket x
                d[int(x)].add(it.multi_index[0])

        # For every bucket, find all pairs. These are the LSH pairs
        candidate_pairs = set()
        for k, v in d.items():
            if len(v) > 1:
                for pair in combinations(v, 2):
                    candidate_pairs.add(tuple(sorted(pair)))

        # For each candidate pair, compute actual similarity
        # We will obtain no false positives, but may have some false negatives (missed candidate pairs)
        lsh_pairs = set()
        for i, j in candidate_pairs:
            if self.angular_distance(T[i], T[j]) > thresh:
                lsh_pairs.add((i, j))

        return lsh_pairs