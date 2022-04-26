import random
import sys
import time
from collections import defaultdict

from pm4py import get_variants
from pm4py.algo.conformance.alignments.petri_net import algorithm as alignments
from pm4py.objects.log import obj as log_implementation
from pm4py.objects.petri_net.utils import align_utils as utils
from tqdm import tqdm

import Coocurrence
import LogIndexing
import feature_encodings
from ExploreExploitDecision import should_explore_greedy
from LogIndexing import SequenceBasedLogPreprocessor
from feature_encodings import create_feature_encoding_from_index


class Sample:
    """
    A sample of traces from a log.
    """
    #TODO decide on what to return in what way
    def __init__(self):
        self.fitness = -1.0
        self.correlations = {}
        self.traces = []  # contains index of sampled trace in log
        self.trace_deviations = 0
        self.total_deviations = 0
        self.activity_deviations = defaultdict(int)
        self.alignments = []
        self.times = {}
        self.correlation_changes = []
        self.first_positive_correlation_at = -1
        self.alignments_calculated = 0

    def __repr__(self):
        return "Sample Fitness:" + str(self.fitness) + "\nCorrelations:\n" + str(
            self.correlations) + "\nTraces:\n" + str(self.traces) + "\nAlignments:\n" + str(self.alignments)


class ConstantList:
    """
    Implements a list that returns the same value for each index (e.g., cost)
    """

    def __init__(self, value):
        self.value = value

    def __getitem__(self, item):
        return self.value


def init_alignment_params(model):
    """
    Sets common parameters for alignment computation using unit cost function.
    """
    alignment_params = {}
    model_cost_function = dict()
    sync_cost_function = dict()
    for t in model.transitions:
        if t.label is not None:
            model_cost_function[t] = 1
            sync_cost_function[t] = 0
        else:
            model_cost_function[t] = 0

    # TODO check if standard costs can be used here
    # dirty dirty hack to set a global cost for log-moves
    utils.STD_MODEL_LOG_MOVE_COST = 1
    # Set cost for each log-only-move to 1, does not work with current normalization any more
    # trace_cost_function = ConstantList(1)

    alignment_params[alignments.Parameters.PARAM_MODEL_COST_FUNCTION] = model_cost_function
    alignment_params[alignments.Parameters.PARAM_SYNC_COST_FUNCTION] = sync_cost_function
    # alignment_params[alignments.Parameters.PARAM_TRACE_COST_FUNCTION] = trace_cost_function

    return alignment_params


class LogSampler:
    def __init__(self, use_cache=True, alignment_cache={}, prep_time=0.0, verbose=False):
        # manually define alignment cost function
        self.alignment_params = {}
        self.use_cache = use_cache
        self.alignment_cache = alignment_cache

        # initialize result container
        self.sample = Sample()
        self.sample.times["partitioning"] = prep_time
        self.sample.times["alignment"] = 0.0
        self.sample.times["sampling"] = 0.0

        self.verbose = verbose

    def _update_sample(self, model, initial_marking, final_marking):
        # finally, add correlations and global fitness value to result object
        if hasattr(self, "knowledge_base"):
            self.sample.correlations = self.knowledge_base

        shortest_path = alignments.apply_trace(log_implementation.Trace(), model, initial_marking,
                                               final_marking, parameters=self.alignment_params)["cost"]
        total_costs = 0.0
        upper_bound_total_costs = 0.0
        for i in range(len(self.sample.traces)):
            total_costs += self.sample.alignments[i]["cost"]
            upper_bound_total_costs += int(shortest_path) + len(self.sample.traces[i])
        self.sample.fitness = 1 - (total_costs / upper_bound_total_costs)

    def _calculate_alignment(self, trace, model, initial_marking, final_marking):
        if self.verbose:
            print("Analyzing sampled trace ")
            print(" > Calculating alignments...")

        # calculate alignment
        # TODO replace with proper trace classifier
        event_representation = ""

        # check if alignment has been calculated already for the given trace, if so grab it, otherwise calculate it
        if self.use_cache:
            for event in trace:
                event_representation = event_representation + " >> " + event["concept:name"]

            # Second condition checks whether an alignment has been actually precomputed for the given trace
            if event_representation in self.alignment_cache and self.alignment_cache[event_representation]:
                alignment = self.alignment_cache[event_representation]
            else:
                if self.verbose:
                    print("Trace not found in precomputed cache!")
                alignment = alignments.apply(trace, model, initial_marking, final_marking,
                                             parameters=self.alignment_params)
                self.alignment_cache[event_representation] = alignment
        else:
            alignment = alignments.apply(trace, model, initial_marking, final_marking,
                                         parameters=self.alignment_params)

        return alignment

    def _update_deviation_statistics(self, deviation_found, activity):
        self.sample.total_deviations += 1
        if not deviation_found:
            self.sample.trace_deviations += 1
            deviation_found = True

        if activity is not None:
            self.sample.activity_deviations[activity] += 1

        return deviation_found


class GuidedLogSampler(LogSampler):
    def __init__(self, partitioned_log={}, window_size=3, n_gram_size=3,
                 use_cache=True, alignment_cache={}, index_file=None, prep_time=0.0, verbose=False):
        super().__init__(use_cache, alignment_cache, prep_time, verbose);

        # TODO add self.log and verbose as field? because used throughout exploit and explore

        self.partitioned_log = partitioned_log
        self.knowledge_base = {}

        # TODO window size and n_gram_size as parameter
        # TODO assert n gram size constant between here and log partitionining
        self.window_size = window_size
        self.n_gram_size = n_gram_size

        # index file for use in feature construction
        self.index_file = index_file

    def _sum_positive_correlations(self, condition):
        distribution = {}
        feature_sum = 0.0

        # feature may be event-level feature, trace-level feature or k-gram
        for feature in self.knowledge_base.keys():
            if condition(self.knowledge_base[feature]):
                feature_sum += self.knowledge_base[feature].correlation
                distribution[feature] = self.knowledge_base[feature].correlation

        return feature_sum, distribution

    # convert positive correlations into probability distribution proportional to their correlation
    def _convert_to_probability_distribution(self, distribution, feature_sum):
        for correlating_feature in distribution.keys():
            distribution[correlating_feature] /= feature_sum

    def _choose_feature(self, distribution):
        chosen_feature = random.choices([x for x in distribution.keys()], [x for x in distribution.values()], k=1)[0]
        if self.verbose:
            print("EXPLOITATION yields: " + str(chosen_feature) + " Prob:" + str(
                distribution[chosen_feature]) + ", Knowledge:" + str(self.knowledge_base[chosen_feature]))

        return chosen_feature

    def _prepare_sampling(self, keys, model):
        for key in keys:
            self.knowledge_base[key] = Coocurrence.coocurence_rule()

        self.alignment_params = init_alignment_params(model)

    def _calculate_alignment(self, trace, model, initial_marking, final_marking):
        alignment_time = time.time()

        alignment = super()._calculate_alignment(trace, model, initial_marking, final_marking)
        self.sample.alignments.append(alignment)

        alignment_time = time.time() - alignment_time
        self.sample.times["alignment"] += alignment_time

        if self.verbose:
            print(" > %s" % alignment["alignment"])
            print(" > Cost: %d" % alignment["cost"])

            print(" > Updating knowledge base")

        return alignment

    def _update_correlation_coefficients(self):
        change_in_correlation = 0.0
        positive_correlation = False
        for feature in self.knowledge_base.keys():
            prior = self.knowledge_base[feature].correlation
            self.knowledge_base[feature].update_correlation()
            posterior = self.knowledge_base[feature].correlation
            change_in_correlation += abs(posterior - prior)
            if self.knowledge_base[feature].correlation > 0:
                positive_correlation = True
        self.sample.correlation_changes.append(change_in_correlation)
        if self.sample.first_positive_correlation_at < 0 and positive_correlation:
            self.sample.first_positive_correlation_at = len(self.sample.traces)

    def _add_non_occurring_to_sets(self, potential_features, deviating_features, conforming_features):
        # potential_features may be features or k-grams
        features_unrelated_to_deviations = []
        features_unrelated_to_conforming = []
        for potential_feature in potential_features:
            if potential_feature not in deviating_features:
                features_unrelated_to_deviations.append(potential_feature)
            if potential_feature not in conforming_features:
                features_unrelated_to_conforming.append(potential_feature)

        return features_unrelated_to_deviations, features_unrelated_to_conforming

    def _increase_counters(self, deviating_features, conforming_features, features_unrelated_to_conforming,
                           features_unrelated_to_deviations):
        # increase counters of all features depending on their (non)-cooccurrence with conformance/deviations

        for feature in deviating_features:
            if feature not in self.knowledge_base:
                continue
            self.knowledge_base[feature].add_deviating()

        for feature in conforming_features:
            if feature not in self.knowledge_base:
                continue
            self.knowledge_base[feature].add_conforming()

        for feature in features_unrelated_to_conforming:
            if feature not in self.knowledge_base:
                continue
            self.knowledge_base[feature].add_unrelated_to_conforming()

        for feature in features_unrelated_to_deviations:
            if feature not in self.knowledge_base:
                continue
            self.knowledge_base[feature].add_unrelated_to_deviating()


# TODO should we save log and/or model during creation?
class FeatureGuidedLogSampler(GuidedLogSampler):
    def __init__(self, log, use_cache=True, preprocessing_time=None, index_file=None, verbose=False):
        super().__init__(partitioned_log={},
                         window_size=3,
                         n_gram_size=3,
                         use_cache=use_cache,
                         alignment_cache={},
                         prep_time=preprocessing_time,
                         index_file=index_file,
                         verbose=verbose)
        self.partitioned_log, _ = LogIndexing.FeatureBasedPartitioning().partition(log, index_file=index_file)

    def explore(self, log):
        sampled_trace = random.choice([idx for idx in [*range(len(log))] if log[idx] not in self.sample.traces])

        if self.verbose:
            print("EXPLORATION")
            print(" > " + str(log[sampled_trace]))
        return sampled_trace

    def exploit(self, log):
        """
         construct distribution out of positive correlations and pick a trace corresponding to the chosen feature
         return random trace, if no positive correlation is detected (less likely the later we are in sampling process)
         ignore empty partitions
        """

        # first pass over features - get sum of all features with positive correlation
        feature_sum, distribution = self._sum_positive_correlations(lambda c: c.correlation > 0.0)

        # if no correlation is known, keep exploring
        if feature_sum == 0.0:
            return self.explore(log)

        # convert positive correlations into probability distribution proportional to their correlation
        self._convert_to_probability_distribution(distribution, feature_sum)

        # TODO replace with mapping of trace to features (i.e inverted inverted index, for fast removal from
        #  partitions) - this will blow up for larger sample sizes
        sampled_trace = None

        while sampled_trace is None or log[sampled_trace] in self.sample.traces:
            # choose a feature using distribtution
            chosen_feature = self._choose_feature(distribution)

            # choose a trace from those that contain selected feature
            potential_traces = self.partitioned_log[chosen_feature]
            sampled_trace = random.choice(potential_traces)

        if self.verbose:
            print(" > " + str(log[sampled_trace]))

        return sampled_trace

    def construct_sample(self, log, model, initial_marking, final_marking, sample_size):
        start_time = time.time()
        # stop right away if sample size is larger than log size
        if len(log) <= sample_size:
            print("Sample size larger than log. Returning complete log")
            return log

        # Initializing some stuff
        self._prepare_sampling(self.partitioned_log, model)

        # Sampling process
        pbar = tqdm(list(range(sample_size)), desc=" > Sampling...", file=sys.stdout, disable=False)
        for i in pbar:
            if (self.verbose):
                print("Sampling " + str(len(self.sample.traces) + 1) + "/" + str(sample_size))
            sampled_trace = None

            # decide between exploration and exploitation
            if should_explore_greedy(0.8):
                # exploration - pick a random trace
                sampled_trace = self.explore(log)
            else:
                # exploitation - convert positive correlations into distribution and pick trace correspondign to chosen feature
                sampled_trace = self.exploit(log)

            # add sample to output set, remove trace from log and partitioned log
            self.sample.traces.append(log[sampled_trace])

            # check trace wrt property of interest - here alignments - and update knowledge base accordingly
            # c_time = time.time()
            self._check_for_property(log[sampled_trace], model, initial_marking, final_marking)
            if self.verbose:
                print(" > Updated knowledge base after trace analysis(only positive correlations):")
                for feature in self.knowledge_base.keys():
                    if self.knowledge_base[feature].correlation > 0.0:
                        print("      " + str(feature) + " : " + str(self.knowledge_base[feature]))
                print()

        # finally, add correlations and global fitness value to result object
        self._update_sample(model, initial_marking, final_marking)
        self.sample.times["sampling"] = (time.time() - start_time) - self.sample.times["alignment"]

        return self.sample

    def _check_for_property(self, trace, model, initial_marking, final_marking):
        alignment = self._calculate_alignment(trace, model, initial_marking, final_marking)
        deviation_contexts, deviating_n_grams = self.__get_deviation_context(trace, alignment)

        deviating_features = []
        conforming_features = []

        # add event-level features to list of deviating or conforming sets based on their marking
        deviating_e_features, conforming_e_features, non_conformance_in_trace = self.__assign_event_level_features(
            trace, deviation_contexts)
        deviating_features.extend(deviating_e_features)
        conforming_features.extend(conforming_e_features)

        # add n-gram features
        deviating_n_gram_features, conforming_n_gram_features = self.__assign_n_gram_features(trace, deviating_n_grams)
        deviating_features.extend(deviating_n_gram_features)
        conforming_features.extend(conforming_n_gram_features)

        # add trace-level features
        deviating_t_features, conforming_t_features = self.__assign_trace_level_features(trace, non_conformance_in_trace)
        deviating_features.extend(deviating_t_features)
        conforming_features.extend(conforming_t_features)

        # add non-occurring features to their sets
        features_unrelated_to_deviations, features_unrelated_to_conforming = self._add_non_occurring_to_sets(
            self.partitioned_log.keys(),
            deviating_features,
            conforming_features)

        self._increase_counters(deviating_features, conforming_features, features_unrelated_to_deviations,
                                features_unrelated_to_conforming)
        self._update_correlation_coefficients()

    def __get_deviation_context(self, trace, alignment):
        deviation_contexts = []
        deviation_found = False
        deviating_n_grams = []

        for LMstep in alignment["alignment"]:
            # Sync move
            if LMstep[0] == LMstep[1]:
                deviation_contexts.append(False)

            # Move on log only, deviation
            elif LMstep[1] == '>>':
                deviation_contexts.append(True)
                for i in range(max(0, len(deviation_contexts) - self.window_size), len(deviation_contexts)):
                    deviation_contexts[i] = True

                # TODO NGRAMS
                deviating_n_grams.extend(self.get_deviation_context_n_grams(trace, len(deviation_contexts) - 1))
                deviation_found = self._update_deviation_statistics(deviation_found, LMstep[0])

            # Move on model only
            elif LMstep[0] == '>>':
                # Hidden transition, no deviation
                if LMstep[1] is None:
                    pass
                # No hidden transition, deviation
                else:
                    for i in range(max(0, len(deviation_contexts) - self.window_size), len(deviation_contexts)):
                        deviation_contexts[i] = True

                    # TODO NGRAMS
                    deviating_n_grams.extend(self.get_deviation_context_n_grams(trace, len(deviation_contexts) - 1))
                    deviation_found = self._update_deviation_statistics(deviation_found, LMstep[1])
            else:
                print("Should not happen.")

        return deviation_contexts, deviating_n_grams

    def __assign_event_level_features(self, trace, deviation_contexts):
        deviating_features = []
        conforming_features = []

        non_conformance_in_trace = False
        for idx, is_deviating in enumerate(deviation_contexts):
            current_event = {trace[idx]}
            _, features = create_feature_encoding_from_index([current_event],
                                                             considered_feature_types=feature_encodings.feature_types.EVENT_LEVEL,
                                                             index_name=self.index_file)
            if is_deviating:
                deviating_features.extend(features)
                non_conformance_in_trace = True
            else:
                conforming_features.extend(features)

        return deviating_features, conforming_features, non_conformance_in_trace

    def __assign_n_gram_features(self, trace, deviating_n_grams):
        deviating_features = []
        conforming_features = []

        for i in range(0, max(len(trace) - self.n_gram_size, 1)):
            subtrace = None
            if len(trace) < self.n_gram_size:
                subtrace = trace
            else:
                subtrace = trace[i:i + self.n_gram_size]
            events = [*map(lambda x: x['concept:name'], subtrace)]
            events_string = "#".join(events)
            if events_string in deviating_n_grams:
                deviating_features.append(events_string)
            else:
                conforming_features.append(events_string)

        return deviating_features, conforming_features

    def __assign_trace_level_features(self, trace, non_conformance_in_trace):
        deviating_features = []
        conforming_features = []

        # add trace-level feature to set, either conforming or deviating, depending on existence of non-conformance
        _, features = create_feature_encoding_from_index([trace],
                                                         considered_feature_types=feature_encodings.feature_types.TRACE_LEVEL,
                                                         index_name=self.index_file)
        if non_conformance_in_trace:
            deviating_features.extend(features)
        else:
            conforming_features.extend(features)

        return deviating_features, conforming_features

    def get_deviation_context_n_grams(self, trace, deviation_index):
        if self.window_size < self.n_gram_size:
            return []

        if len(trace) < self.n_gram_size:
            events = [*map(lambda x: x['concept:name'], trace)]
            return ["#".join(events)]

        if deviation_index < self.n_gram_size:
            return []

        start = max(deviation_index - self.window_size + 1, 0)
        end = max(deviation_index - self.n_gram_size + 1, 0)

        deviating_n_grams = []
        for i in range(start, end + 1):
            subtrace = trace[i:i + self.n_gram_size]
            events = [*map(lambda x: x['concept:name'], subtrace)]
            events_string = "#".join(events)
            deviating_n_grams.append(events_string)
        return deviating_n_grams


class SequenceGuidedLogSampler(GuidedLogSampler):
    """
    Implements sampling based on k-grams in the activity sequence, which are correlated with deviations.
    """

    # TODO properly include index_file, i.e talk to Lam, if this is actually needed here?
    def __init__(self, log, k=3, b=10, r=10, p=3, batch_size=10, window_size=5, use_cache=True,
                 index_file=None, verbose=False):
        self.log_manager = SequenceBasedLogPreprocessor(log, k, b, r, p, verbose)

        super().__init__(partitioned_log=self.log_manager.partitioned_log,
                         window_size=window_size,
                         n_gram_size=k,
                         use_cache=use_cache,
                         alignment_cache={},
                         index_file=index_file,
                         prep_time=self.log_manager.time,
                         verbose=verbose)

        self.log = log
        self.batch_size = batch_size
        self.available_trace_ids = set(range(len(log)))

    def explore(self):
        """
        Randomly samples traces to explore the search space.
        """
        # TODO: Explore with diverse sample
        sampled_indices = self.log_manager.get_random_sample(sample_size=self.batch_size,
                                                             indices=[*self.available_trace_ids])

        if self.verbose:
            print(f"EXPLORATION: Sampled {len(sampled_indices)} traces.")

        return sampled_indices

    def exploit(self):
        """
        Samples traces based on the distribution of (previously observed) positive correlations with particular k-grams.
        Returns random sample if no positive correlation is detected (less likely the later we are in sampling process).
        """
        # First pass over features - get sum of all features with positive correlation
        feature_sum, distribution = self._sum_positive_correlations(
            lambda c: c.correlation > 0.0 and len(c.trace_indices) > 0)

        # if no correlation is known, keep exploring
        if feature_sum == 0.0:
            return self.explore()

        # convert positive correlations into probability distribution proportional to their correlation
        self._convert_to_probability_distribution(distribution, feature_sum)

        # choose a feature using distribtution
        chosen_feature = self._choose_feature(distribution)

        # Find traces similar to those with selected feature (activity sequence)
        # Note: This yields globally similar traces, not necessarily those with the same features

        # Select previously sampled trace with chosen feature
        reference_trace_indices = list(self.knowledge_base[chosen_feature].trace_indices)
        # reference_trace_indices = self.partitioned_log[chosen_feature]
        # candidate_trace_indices = set(self.sample_ids)
        assert (len(reference_trace_indices) != 0)

        # Randomly select a reference trace and find similar traces to it
        # If no similar traces can be found, continue with next reference trace
        # TODO: Find traces that are similar to all reference traces (e.g., by candidate index intersection)
        # ref_idx = random.choice(tuple(reference_trace_indices))
        random.shuffle(reference_trace_indices)
        for ref_idx in reference_trace_indices:
            candidate_indices = self.log_manager.get_similar_sample(ref_trace_idx=ref_idx)
            candidate_indices = [idx for idx in candidate_indices if idx in self.available_trace_ids]
            if len(candidate_indices) == 0:
                continue
            sampled_indices = self.log_manager.get_random_sample(sample_size=self.batch_size,
                                                                 indices=candidate_indices)

            if self.verbose:
                print(f" > Sampled {len(sampled_indices)} similar traces.")
            return sampled_indices
        # candidate_trace_indices = candidate_trace_indices.intersection(candidate_indices)
        if self.verbose:
            print(" > No candidate traces found for exploitation. Falling back to exploration.")
        return self.explore()

    def construct_sample(self, log, model, initial_marking, final_marking, sample_size):
        """
        Constructs a sample based on an exploration vs. exploitation guided sampling strategy.
        """
        # stop right away if sample size is larger than log size
        if len(log) <= sample_size:
            print("Sample size larger than log. Returning complete log")
            return log

        start_time = time.time()

        # Initializing some stuff
        self._prepare_sampling(self.log_manager.k_gram_dict.keys(), model)

        # Sampling process
        pbar = tqdm(list(range(sample_size)), desc=" > Sampling...", file=sys.stdout, disable=False)
        for i in pbar:
            if self.verbose:
                print("Sampling " + str(len(self.sample.traces)) + "/" + str(sample_size))
            sampled_trace = None

            # decide between exploration and exploitation
            if should_explore_greedy(0.8):
                # exploration - pick a random trace
                sampled_indices = self.explore()
            else:
                # exploitation - convert positive correlations into distribution and pick trace correspondign to chosen feature
                sampled_indices = self.exploit()

            # add sample to output set, remove trace from log and partitioned log
            # TODO is something missing here?

            # Remove sampled ids from set of not-yet-sampled ids
            self.available_trace_ids.difference_update(sampled_indices)

            # Extend sample
            self.sample.traces.extend([log[i] for i in sampled_indices])

            # check trace wrt property of interest - here alignments - and update knowledge base accordingly
            self._check_for_property(sampled_indices, model, initial_marking, final_marking)

            if self.verbose:
                print(" > Updated knowledge base after trace analysis(only positive correlations):")
                for feature in self.knowledge_base.keys():
                    if self.knowledge_base[feature].correlation > 0.0:
                        print("      " + str(feature) + " : " + str(self.knowledge_base[feature]))
                print()

        # finally, add correlations and global fitness value to result object
        self._update_sample(model, initial_marking, final_marking)
        self.sample.times["sampling"] = (time.time() - start_time) - self.sample.times["alignment"]

        return self.sample

    def _check_for_property(self, trace_ids, model, initial_marking, final_marking):
        """
        Analyzes the sampled traces (i.e., computes alignments and determines preceding k-grams in case of deviations)
        and updates the knowledge base accordingly.
        """

        # Calculate alignment
        # pbar = tqdm(trace_ids, desc=" > Analyzing sampled traces", file=sys.stdout, disable=not self.verbose)
        for trace_id in trace_ids:
            # pbar.set_description(" > Calculating alignments...")
            alignment = self._calculate_alignment(self.log[trace_id], model, initial_marking, final_marking)
            deviation_points, deviation_found = self.__get_deviation_contexts(alignment)
            deviating_k_grams, conforming_k_grams = self.__assign_k_grams(trace_id, deviation_points)

            # Add non-occurring k-grams to their sets
            k_grams_unrelated_to_deviations, k_grams_unrelated_to_conforming = self._add_non_occurring_to_sets(
                self.log_manager.k_gram_dict.keys(),
                deviating_k_grams,
                conforming_k_grams)
            self._increase_counters(deviating_k_grams, conforming_k_grams, k_grams_unrelated_to_deviations,
                                    k_grams_unrelated_to_conforming)

            self._update_correlation_coefficients()

    def __get_deviation_contexts(self, alignment):
        # LMstep[0] corresponds to an event in the trace and LMstep[1] corresponds to a transition in the model
        # The following cases are possible:
        # 1. Sync move (LMstep[0] == LMstep[1]): Both trace and model advance in the same way
        # 2. Move on log (LMstep[1] == '>>'): A move in the log that could not be mimicked by the model: deviation
        # 3. Move on model (LMstep[0] == '>>'):
        #   3.1 with hidden transition (LMstep[1] == None): OK, no deviation
        #   3.2 without hidden transition (LMstep[1] != None): not fit, deviation between log and model
        # pbar.set_description(" > Updating k-gram-deviation-statistics...")

        deviation_points = []
        trace_idx = 0
        deviation_found = False

        for LMstep in alignment["alignment"]:
            # Sync move
            if LMstep[0] == LMstep[1]:
                pass
            # Move on log only, deviation
            elif LMstep[1] == '>>':
                deviation_points.append(trace_idx)
                deviation_found = self._update_deviation_statistics(deviation_found, LMstep[0])

            # Move on model only
            elif LMstep[0] == '>>':
                # Hidden transition, no deviation
                if LMstep[1] is None:
                    pass
                # No hidden transition, deviation
                else:
                    deviation_points.append(trace_idx)
                    deviation_found = self._update_deviation_statistics(deviation_found, LMstep[1])

            # Increment pointer to current position in trace
            if LMstep[0] != '>>':
                trace_idx += 1

        return deviation_points, deviation_found

    def __determine_indices(self, deviation_points):
        # Determine indices in the trace from which to retrieve the relevant k-grams from
        deviating_indices = set()
        for d in deviation_points:
            start = max(d - self.window_size + 1, 0)
            end = max(d - self.n_gram_size + 1, 0)
            deviating_indices.update([*range(start, end + 1)])

        return deviating_indices

    def __assign_k_grams(self, trace_id, deviation_points):
        deviating_indices = self.__determine_indices(deviation_points)
        k_grams = self.log_manager.get_ordered_k_grams(trace_id, self.n_gram_size)

        # Add k-grams to list of deviating or conforming sets based on their marking
        deviating_k_grams = []
        conforming_k_grams = []

        non_conformance_in_trace = False
        for idx, kgram in enumerate(k_grams):
            # Add trace to knowledge base
            self.knowledge_base[kgram].add_trace_index(trace_id)

            if idx in deviating_indices:
                deviating_k_grams.append(kgram)
                non_conformance_in_trace = True
            else:
                conforming_k_grams.append(kgram)

        return deviating_k_grams, conforming_k_grams


class NaiveLogSampler(LogSampler):
    '''
        (Kind of) Abstract class for 'naive' LogSamplers
        Requires implementation of sampling strategy
    '''

    def __init__(self, use_cache=True, alignment_cache={}, verbose=False):
        super().__init__(use_cache, alignment_cache, 0.0, verbose)

    def construct_sample(self, log, model, initial_marking, final_marking, sample_size, calculate_alignments=True):
        sampled_traces = self._compute_traces_in_sample(log, sample_size)

        self.alignment_params = init_alignment_params(model)
        if calculate_alignments:
            self._compute_deviations(sampled_traces, model, initial_marking, final_marking)

        self._update_sample(model, initial_marking, final_marking)
        return self.sample

    def _compute_traces_in_sample(self, log, sample_size):
        raise NotImplementedError("Please Implement this method")

    def _compute_deviations(self, sampled_traces, model, initial_marking, final_marking):
        alignment_t = time.time()

        pbar = tqdm(sampled_traces, desc=" > Sampling...", file=sys.stdout, disable=False)
        for trace in pbar:
            alignment = self._calculate_alignment(trace, model, initial_marking, final_marking)
            self.sample.alignments.append(alignment)

            deviation_found = False

            for LMstep in alignment["alignment"]:
                # Sync move
                if LMstep[0] == LMstep[1]:
                    pass
                # Move on log only, deviation
                elif LMstep[1] == '>>':
                    deviation_found = self._update_deviation_statistics(deviation_found, LMstep[0])

                # Move on model only
                elif LMstep[0] == '>>':
                    # Hidden transition, no deviation
                    if LMstep[1] is None:
                        pass
                    # No hidden transition, deviation
                    else:
                        deviation_found = self._update_deviation_statistics(deviation_found, LMstep[1])

        self.sample.times["alignment"] = time.time() - alignment_t


class RandomLogSampler(NaiveLogSampler):
    def __init__(self, use_cache=True, alignment_cache={}, verbose=False):
        super().__init__(use_cache, alignment_cache, verbose)

    def _compute_traces_in_sample(self, log, sample_size):
        sampling_t = time.time()
        sampled_traces = random.sample(log, sample_size)
        self.sample.traces.extend(sampled_traces)
        self.sample.times["sampling"] = time.time() - sampling_t

        return sampled_traces


class LongestTraceVariantLogSampler(NaiveLogSampler):
    def __init__(self, use_cache=True, alignment_cache={}, verbose=False):
        super().__init__(use_cache, alignment_cache, verbose)

    def _compute_traces_in_sample(self, log, sample_size):
        sampling_t = time.time()

        # TODO replace with get_variant_as_tuple(log)
        trace_variants = get_variants(log)
        variant_list = [variant[0] for variant in trace_variants.values()]

        sorted_trace_variants = sorted(variant_list, key=len)
        sorted_trace_variants.reverse()

        sampled_traces = sorted_trace_variants[:sample_size]

        self.sample.traces.extend(sampled_traces)
        self.sample.times["sampling"] = time.time() - sampling_t

        return sampled_traces
