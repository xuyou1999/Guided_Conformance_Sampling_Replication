import random
import time
from collections import defaultdict

from pm4py import get_variants
from pm4py.algo.conformance.alignments.petri_net import algorithm as alignments
from pm4py.objects.log import obj as log_implementation

import Coocurrence
import feature_encodings
from LogIndexing import SequenceBasedLogPreprocessor, CombinedLogPartitioning
from feature_encodings import create_feature_encoding
from tqdm import tqdm

import sys


class Sample:
    """
    A sample of traces from a log.
    """

    def __init__(self):
        self.fitness = -1.0
        self.correlations = {}
        self.traces = []
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
    Sets common parameters for alignment computation.
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

    # Set cost for each log-only-move to 1
    trace_cost_function = ConstantList(1)

    alignment_params[alignments.Parameters.PARAM_MODEL_COST_FUNCTION] = model_cost_function
    alignment_params[alignments.Parameters.PARAM_SYNC_COST_FUNCTION] = sync_cost_function
    alignment_params[alignments.Parameters.PARAM_TRACE_COST_FUNCTION] = trace_cost_function

    return alignment_params


class FeatureGuidedLogSampler:
    def __init__(self, use_cache=True, alignment_cache={}, preprocessing_time=None):
        self.partitioned_log = {}
        self.knowledge_base = {}

        # TODO window size and n_gram_size as parameter
        # TODO assert n gram size constant between here and log partitionining
        self.window_size = 3
        self.n_gram_size = 3

        # manually define alignment cost function
        self.use_cache = use_cache
        self.alignment_cache = alignment_cache
        self.alignment_params = {}

        # initialize result container
        self.sample = Sample()
        if preprocessing_time:
            self.sample.times["partitioning"] = preprocessing_time
        self.sample.times["alignment"] = 0.0
        self.sample.times["sampling"] = 0.0

    def explore(self, log, verbose=False):
        #e_time = time.time()
        #print("Explore")
        sampled_trace = random.choice(log)

        # dirty probably inefficient hack - keep sampling until we have a trace that has not been sampled yet
        # TODO optimize
        while sampled_trace in self.sample.traces:
            sampled_trace = random.choice(log)
        if verbose:
            print("EXPLORATION")
            print(" > " + str(sampled_trace))
        #print(f"Exploration-time : {time.time()-e_time}")
        return sampled_trace

    def exploit(self, log, verbose=False):
        #e_time = time.time()
        # construct distribution out of positive correlations and pick a trace corresponding to the chosen feature
        # return random trace, if no positive correlation is detected (less likely the later we are in sampling process)
        # ignore empty partitions

        # first pass over features - get sum of all features with positive correlation
        distribution = {}
        feature_sum = 0.0
        for feature in self.knowledge_base.keys():
            if self.knowledge_base[feature].correlation > 0.0:
            #    traces_in_partition = self.partitioned_log[feature]
            #    partition_contains_unsampled_trace = False
            #    if len(self.partitioned_log[feature]) > len(self.sample.traces):
            #        partition_contains_unsampled_trace = True
            #    else:
            #        for trace in traces_in_partition:
            #            if trace not in self.sample.traces:
            #                partition_contains_unsampled_trace = True
            #                break
            #    if partition_contains_unsampled_trace:
                feature_sum += self.knowledge_base[feature].correlation
                distribution[feature] = self.knowledge_base[feature].correlation

        # if no correlation is known, keep exploring
        if feature_sum == 0.0:
            return self.explore(log)

        # convert positive correlations into probability distribution proportional to their correlation
        for correlating_feature in distribution.keys():
            distribution[correlating_feature] /= feature_sum

        # TODO replace with mapping of trace to features (i.e inverted inverted index, for fast removal from
        #  partitions) - this will blow up for larger sample sizes
        # choose a feature using distribtution
        chosen_feature = random.choices([x for x in distribution.keys()], [x for x in distribution.values()], k=1)[0]
        if verbose:
            print("EXPLOITATION yields: " + str(chosen_feature) + " Prob:" + str(
                distribution[chosen_feature]) + ", Knowledge:" + str(self.knowledge_base[chosen_feature]))

        # choose a trace from those that contain selected feature
        potential_traces = self.partitioned_log[chosen_feature]
        # for trace in self.sample.traces:
        #    if trace in potential_traces:
        #        potential_traces.remove(trace)
        sampled_trace = random.choice(potential_traces)
        while sampled_trace in self.sample.traces or sampled_trace is None:
            # choose a feature using distribtution
            chosen_feature = random.choices([x for x in distribution.keys()], [x for x in distribution.values()], k=1)[0]
            if verbose:
                print("EXPLOITATION yields: " + str(chosen_feature) + " Prob:" + str(
                    distribution[chosen_feature]) + ", Knowledge:" + str(self.knowledge_base[chosen_feature]))

            # choose a trace from those that contain selected feature
            potential_traces = self.partitioned_log[chosen_feature]
            #for trace in self.sample.traces:
            #    if trace in potential_traces:
            #        potential_traces.remove(trace)
            sampled_trace = random.choice(potential_traces)
        if verbose:
            print(" > " + str(sampled_trace))
        #print(f"Exploitation-time : {time.time()-e_time}")
        return sampled_trace

    def construct_sample(self, log_name, log, model, initial_marking, final_marking, partitioned_log, sample_size,
                         verbose=False):
        start_time = time.time()
        # stop right away if sample size is larger than log size
        if len(log) <= sample_size:
            print("Sample size larger than log. Returning complete log")
            return log

        # Initializing some stuff
        self.partitioned_log = partitioned_log
        for key in self.partitioned_log:
            self.knowledge_base[key] = Coocurrence.coocurence_rule()

        # print(len(self.knowledge_base), len(self.partitioned_log))

        self.alignment_params = init_alignment_params(model)

        # TODO decide on exploit vs explore, right now we have 80/20 split
        learning_rate = 0.8

        # Sampling process
        pbar = tqdm(list(range(sample_size)), desc=" > Sampling...", file=sys.stdout, disable=False)
        for i in pbar:
            if (verbose):
                print("Sampling " + str(len(self.sample.traces) + 1) + "/" + str(sample_size))
            sampled_trace = None

            # decide between exploration and exploitation
            decision = random.random()
            if decision > learning_rate:
                # exploration - pick a random trace
                sampled_trace = self.explore(log)
            else:
                # exploitation - convert positive correlations into distribution and pick trace correspondign to chosen feature
                sampled_trace = self.exploit(log)

            # add sample to output set, remove trace from log and partitioned log
            self.sample.traces.append(sampled_trace)
            #r_time = time.time()
            #self.remove_trace(sampled_trace, log_name)
            #print(f"Removing trace: {time.time()-r_time}")

            # check trace wrt property of interest - here alignments - and update knowledge base accordingly
            #c_time = time.time()
            self._check_for_property(log_name, sampled_trace, model, initial_marking, final_marking)
            #print(f"Property Checking: {time.time() - c_time}")
            #print()
            if (verbose):
                print(" > Updated knowledge base after trace analysis(only positive correlations):")
                for feature in self.knowledge_base.keys():
                    if self.knowledge_base[feature].correlation > 0.0:
                        print("      " + str(feature) + " : " + str(self.knowledge_base[feature]))
                print()

        # finally, add correlations and global fitness value to result object
        self.sample.correlations = self.knowledge_base
        fitness = -1.0

        shortest_path = alignments.apply_trace(log_implementation.Trace(), model, initial_marking,
                                               final_marking, parameters=self.alignment_params)["cost"]
        total_costs = 0.0
        upper_bound_total_costs = 0.0
        for i in range(len(self.sample.traces)):
            total_costs += self.sample.alignments[i]["cost"]
            upper_bound_total_costs += int(shortest_path) + len(self.sample.traces[i])
        self.sample.fitness = 1 - (total_costs / upper_bound_total_costs)

        # get sampling time, as total time minus time spent for calculation of alignments
        self.sample.times["sampling"] = (time.time() - start_time) - self.sample.times["alignment"]
        return self.sample

    def _check_for_property(self, log_name, trace, model, initial_marking, final_marking, verbose=False):
        if verbose:
            print("Analyzing sampled trace ")
            print(" > Calculating alignments...")

        # calculate alignment
        # TODO replace with proper trace classifier
        event_representation = ""
        alignment = None

        # check if alignment has been calculated already for the given trace, if so grab it, otherwise calculate it
        alignment_time = time.time()
        if self.use_cache:
            for event in trace:
                event_representation = event_representation + " >> " + event["concept:name"]

            # Second condition checks whether an alignment has been actually precomputed for the given trace
            if event_representation in self.alignment_cache and self.alignment_cache[event_representation]:
                alignment = self.alignment_cache[event_representation]
            else:
                if verbose:
                    print("Trace not found in precomputed cache!")
                alignment = alignments.apply(trace, model, initial_marking, final_marking,
                                             parameters=self.alignment_params)
                self.alignment_cache[event_representation] = alignment
        else:
            alignment = alignments.apply(trace, model, initial_marking, final_marking,
                                         parameters=self.alignment_params)

        alignment_time = time.time() - alignment_time
        self.sample.times["alignment"] = self.sample.times["alignment"] + alignment_time

        self.sample.alignments.append(alignment)

        if verbose:
            print(" > %s" % alignment["alignment"])
            print(" > Cost: %d" % alignment["cost"])

            print(" > Updating knowledge base")
        idx = 0
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
                    # print(i)
                    deviation_contexts[i] = True

                #TODO NGRAMS
                deviating_n_grams.extend(self.get_deviation_context_n_grams(trace, len(deviation_contexts)-1))


                # Update deviation statistics
                self.sample.total_deviations += 1
                if not deviation_found:
                    self.sample.trace_deviations += 1
                    deviation_found = True
                self.sample.activity_deviations[LMstep[0]] += 1

            # Move on model only
            elif LMstep[0] == '>>':
                # Hidden transition, no deviation
                if LMstep[1] == None:
                    pass
                # No hidden transition, deviation
                else:
                    for i in range(max(0, len(deviation_contexts) - self.window_size), len(deviation_contexts)):
                        deviation_contexts[i] = True

                    #TODO NGRAMS
                    deviating_n_grams.extend(self.get_deviation_context_n_grams(trace, len(deviation_contexts)-1))

                    # Update deviation statistics
                    self.sample.total_deviations += 1
                    if not deviation_found:
                        self.sample.trace_deviations += 1
                        deviation_found = True
                    self.sample.activity_deviations[LMstep[1]] += 1

            else:
                print("Should not happen.")

        # add event-level features to list of deviating or conforming sets based on their marking
        deviating_features = []
        conforming_features = []
        non_conformance_in_trace = False
        for idx, is_deviating in enumerate(deviation_contexts):
            current_event = {trace[idx]}
            data, features = create_feature_encoding([current_event], log_name, considered_feature_types=feature_encodings.feature_types.EVENT_LEVEL)
            if is_deviating:
                deviating_features.extend(features)
                non_conformance_in_trace = True
            else:
                conforming_features.extend(features)

        #add n-gram features
        for i in range (0, max(len(trace)-self.n_gram_size, 1)):
            subtrace = None
            if len(trace)<self.n_gram_size:
                subtrace = trace
            else:
                subtrace = trace[i:i+self.n_gram_size]
            events = [*map(lambda x: x['concept:name'], subtrace)]
            events_string = "#".join(events)
            if events_string in deviating_n_grams:
                deviating_features.append(events_string)
            else:
                conforming_features.append(events_string)



        # add trace-level feature to set, either conforming or deviating, depending on existence of non-conformance
        data, features = create_feature_encoding([trace], log_name, considered_feature_types=feature_encodings.feature_types.TRACE_LEVEL)
        if non_conformance_in_trace:
            deviating_features.extend(features)
        else:
            conforming_features.extend(features)


        # add non-occuring features to their sets
        features_unrelated_to_deviations = []
        features_unrelated_to_conforming = []
        for potential_feature in self.partitioned_log.keys():
            if potential_feature not in deviating_features:
                features_unrelated_to_deviations.append(potential_feature)
            if potential_feature not in conforming_features:
                features_unrelated_to_conforming.append(potential_feature)

        # increase counters of all features depending on their (non)-cooccurence with conformance/deviations
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

        # update correlation coefficients
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

    def get_deviation_context_n_grams(self, trace, deviation_index):
        if self.window_size < self.n_gram_size:
            return []

        if len(trace)<self.n_gram_size:
            events = [*map(lambda x: x['concept:name'], trace)]
            return ["#".join(events)]

        if deviation_index< self.n_gram_size:
            return []

        start = min(deviation_index-self.window_size,0)
        end = min(deviation_index-self.n_gram_size,0)

        deviating_n_grams = []
        for i in range (start, end+1):
            subtrace = trace[start:start+self.n_gram_size]
            events = [*map(lambda x: x['concept:name'], subtrace)]
            events_string = "#".join(events)
            deviating_n_grams.append(events_string)
        return deviating_n_grams

    def remove_trace(self, trace, log_name, verbose=False):
        if verbose:
            print("Removing sampled trace from partitioned log")
        # get all features of sampled trace and remove pointer to trace from partitioned log
        # first event-level features
        data, features = create_feature_encoding([trace], log_name, considered_feature_types=feature_encodings.feature_types.EVENT_LEVEL)
        for feature in features:
            if feature in self.partitioned_log:
                # print(self.partitioned_log[feature])
                self.partitioned_log[feature].remove(trace)

                # if partition is samples completely remove it completely, it cant be sampled anymore
                # also mark feature as empty for correlation updates - this way we keep the correlation information for potential analysis
                # if len(self.partitioned_log[feature]) == 0:
                #    if verbose:
                #        print(" > Partition " + str(feature) + " is empty. Removing from Log and Knowledge base")
                #    del self.partitioned_log[feature]

        # then trace-level features
        data, features = create_feature_encoding([trace], log_name, considered_feature_types=feature_encodings.feature_types.TRACE_LEVEL)
        for feature in features:
            if feature in self.partitioned_log:
                self.partitioned_log[feature].remove(trace)
                # if partition is samples completely remove it completely, it cant be sampled anymore
                # also mark feature as empty for correlation updates - this way we keep the correlation information for potential analysis
                # if len(self.partitioned_log[feature]) == 0:
                #    if verbose:
                #        print("Partition " + str(feature) + " is empty. Removing from Log and Knowledge base")
                #    del self.partitioned_log[feature]
        #TODO n-grams
        #data, features = create_feature_encoding([trace], log_name, considered_feature_types=feature_encodings.feature_types.N_GRAMS)
        #for feature in features:
        #    if feature in self.partitioned_log:
        #        self.partitioned_log[feature].remove(trace)

        # TODO for some reason log.remove(trace) throws an error, thus for the time being upon exploring, we check that trace is not in sample...
        # finally remove trace from log - this is needed since exploration works on original log right now, which should be changed :s
        # log.remove(trace)


class SequenceGuidedLogSampler:
    """
    Implements sampling based on k-grams in the activity sequence, which are correlated with deviations.
    """

    def __init__(self, log, k=3, b=10, r=10, p=3, batch_size=10, window_size=5, use_cache=True, alignment_cache={},
                 verbose=False):
        self.log = log
        self.k = k
        self.log_manager = SequenceBasedLogPreprocessor(log, k, b, r, p, verbose)
        self.partitioned_log = self.log_manager.partitioned_log
        self.batch_size = batch_size
        self.window_size = window_size
        self.available_trace_ids = set(range(len(log)))
        self.sample = Sample()
        self.sample.times["partitioning"] = self.log_manager.time
        self.sample.times["alignment"] = 0.0
        self.sample.times["sampling"] = 0.0
        self.knowledge_base = {}

        # manually define alignment cost function
        self.alignment_params = {}
        self.alignment_cache = alignment_cache

        self.use_cache = use_cache

    def explore(self, verbose=False):
        """
        Randomly samples traces to explore the search space.
        """
        # TODO: Explore with diverse sample
        sampled_indices = self.log_manager.get_random_sample(sample_size=self.batch_size,
                                                             indices=[*self.available_trace_ids])

        if verbose:
            print(f"EXPLORATION: Sampled {len(sampled_indices)} traces.")

        return sampled_indices

    def exploit(self, verbose=False):
        """
        Samples traces based on the distribution of (previously observed) positive correlations with particular k-grams.
        Returns random sample if no positive correlation is detected (less likely the later we are in sampling process).
        """

        # First pass over features - get sum of all features with positive correlation
        distribution = {}
        feature_sum = 0.0
        for k_gram in self.knowledge_base.keys():
            candidate_indices = self.knowledge_base[k_gram].trace_indices
            if self.knowledge_base[k_gram].correlation > 0.0 and len(candidate_indices) > 0:
                feature_sum += self.knowledge_base[k_gram].correlation
                distribution[k_gram] = self.knowledge_base[k_gram].correlation

        # if no correlation is known, keep exploring
        if feature_sum == 0.0:
            return self.explore(verbose)

        # convert positive correlations into probability distribution proportional to their correlation
        for correlating_k_gram in distribution.keys():
            distribution[correlating_k_gram] /= feature_sum

        # choose a feature using distribtution
        chosen_feature = random.choices([x for x in distribution.keys()], [x for x in distribution.values()], k=1)[0]

        if verbose:
            print("EXPLOITATION yields: " + str(chosen_feature) + " Prob:" + str(
                distribution[chosen_feature]) + ", Knowledge:" + str(self.knowledge_base[chosen_feature]))

        # Find traces similar to those with selected feature (activity sequence)
        # Note: This yields globally similar traces, not necessarily those with the same features

        # Select previously sampled trace with chosen feature
        reference_trace_indices = list(self.knowledge_base[chosen_feature].trace_indices)
        # reference_trace_indices = self.partitioned_log[chosen_feature]
        # candidate_trace_indices = set(self.sample_ids)
        assert(len(reference_trace_indices) != 0)

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

            if verbose:
                print(f" > Sampled {len(sampled_indices)} similar traces.")
            return sampled_indices
        # candidate_trace_indices = candidate_trace_indices.intersection(candidate_indices)
        if verbose:
            print(" > No candidate traces found for exploitation. Falling back to exploration.")
        return self.explore(verbose)

    def construct_sample(self, log_name, log, model, initial_marking, final_marking, sample_size, verbose=False):
        """
        Constructs a sample based on an exploration vs. exploitation guided sampling strategy.
        """
        # stop right away if sample size is larger than log size
        if len(log) <= sample_size:
            print("Sample size larger than log. Returning complete log")
            return log

        start_time = time.time()

        # Initializing some stuff
        for key in self.log_manager.k_gram_dict.keys():
            self.knowledge_base[key] = Coocurrence.coocurence_rule()

        self.alignment_params = init_alignment_params(model)

        # TODO decide on exploit vs explore, right now we have 80/20 split
        learning_rate = 0.8

        # Sampling process
        pbar = tqdm(list(range(sample_size)), desc=" > Sampling...", file=sys.stdout, disable=False)
        for i in pbar:
            if verbose:
                print("Sampling " + str(len(self.sample.traces)) + "/" + str(sample_size))
            sampled_trace = None

            # decide between exploration and exploitation
            decision = random.random()
            if decision > learning_rate:
                # exploration - pick a random trace
                sampled_indices = self.explore(verbose)
            else:
                # exploitation - convert positive correlations into distribution and pick trace correspondign to chosen feature
                sampled_indices = self.exploit(verbose)

            # add sample to output set, remove trace from log and partitioned log

            # Remove sampled ids from set of not-yet-sampled ids
            self.available_trace_ids.difference_update(sampled_indices)

            # Extend sample
            self.sample.traces.extend([log[i] for i in sampled_indices])

            # check trace wrt property of interest - here alignments - and update knowledge base accordingly
            self._check_for_property(sampled_indices, model, initial_marking, final_marking, verbose)

            if verbose:
                print(" > Updated knowledge base after trace analysis(only positive correlations):")
                for feature in self.knowledge_base.keys():
                    if self.knowledge_base[feature].correlation > 0.0:
                        print("      " + str(feature) + " : " + str(self.knowledge_base[feature]))
                print()

        # finally, add correlations and global fitness value to result object
        self.sample.correlations = self.knowledge_base
        fitness = -1.0

        shortest_path = alignments.apply_trace(log_implementation.Trace(), model, initial_marking,
                                               final_marking, parameters=self.alignment_params)["cost"]

        total_costs = 0.0
        upper_bound_total_costs = 0.0
        for i in range(len(self.sample.traces)):
            total_costs += self.sample.alignments[i]["cost"]
            upper_bound_total_costs += int(shortest_path) + len(self.sample.traces[i])
        self.sample.fitness = 1 - (total_costs / upper_bound_total_costs)

        # print(self.sample)

        self.sample.times["sampling"] = (time.time() - start_time) - self.sample.times["alignment"]
        return self.sample

    def _check_for_property(self, trace_ids, model, initial_marking, final_marking, verbose=False):
        """
        Analyzes the sampled traces (i.e., computes alignments and determines preceding k-grams in case of deviations)
        and updates the knowledge base accordingly.
        """

        # Calculate alignment
        #pbar = tqdm(trace_ids, desc=" > Analyzing sampled traces", file=sys.stdout, disable=not verbose)
        for trace_id in trace_ids:
            # pbar.set_description(" > Calculating alignments...")
            trace = self.log[trace_id]

            # TODO replace with proper trace classifier
            event_representation = ""
            alignment = None

            # check if alignment has been calculated already for the given trace, if so grab it, otherwise calculate it
            alignment_time = time.time()
            if self.use_cache:
                for event in trace:
                    event_representation = event_representation + " >> " + event["concept:name"]

                # Second condition checks whether an alignment has been actually precomputed for the given trace
                if event_representation in self.alignment_cache and self.alignment_cache[event_representation]:
                    alignment = self.alignment_cache[event_representation]
                else:
                    if verbose:
                        print("Trace not found in precomputed cache!")
                    alignment = alignments.apply(trace, model, initial_marking, final_marking,
                                                 parameters=self.alignment_params)
                    self.alignment_cache[event_representation] = alignment

            else:
                alignment = alignments.apply(trace, model, initial_marking, final_marking,
                                             parameters=self.alignment_params)

            alignment_time = time.time() - alignment_time
            self.sample.times["alignment"] += alignment_time
            self.sample.alignments.append(alignment)

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

                    # Update deviation statistics
                    self.sample.total_deviations += 1
                    if not deviation_found:
                        self.sample.trace_deviations += 1
                        deviation_found = True
                    self.sample.activity_deviations[LMstep[0]] += 1

                # Move on model only
                elif LMstep[0] == '>>':
                    # Hidden transition, no deviation
                    if LMstep[1] == None:
                        pass
                    # No hidden transition, deviation
                    else:
                        deviation_points.append(trace_idx)

                        # Update deviation statistics
                        self.sample.total_deviations += 1
                        if not deviation_found:
                            self.sample.trace_deviations += 1
                            deviation_found = True
                        self.sample.activity_deviations[LMstep[1]] += 1

                # Increment pointer to current position in trace
                if LMstep[0] != '>>':
                    trace_idx += 1

            # Determine indices in the trace from which to retrieve the relevant k-grams from
            deviating_indices = set()
            for d in deviation_points:
                start = max(d - self.window_size, 0)
                end = min(max(d - self.k + 1, 0), d)
                deviating_indices.update([*range(start, end)])

            k_grams = self.log_manager.get_ordered_k_grams(trace_id, self.k)

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

            # Add non-occuring k-grams to their sets
            k_grams_unrelated_to_deviations = []
            k_grams_unrelated_to_conforming = []
            for k_gram in self.log_manager.k_gram_dict.keys():
                if k_gram not in deviating_k_grams:
                    k_grams_unrelated_to_deviations.append(k_gram)
                if k_gram not in conforming_k_grams:
                    k_grams_unrelated_to_conforming.append(k_gram)

            # increase counters of all features depending on their (non)-cooccurence with conformance/deviations

            for k_gram in deviating_k_grams:
                if k_gram not in self.knowledge_base:
                    continue
                self.knowledge_base[k_gram].add_deviating()

            for k_gram in conforming_k_grams:
                if k_gram not in self.knowledge_base:
                    continue
                self.knowledge_base[k_gram].add_conforming()

            for k_gram in k_grams_unrelated_to_conforming:
                if k_gram not in self.knowledge_base:
                    continue
                self.knowledge_base[k_gram].add_unrelated_to_conforming()
                # Note: we can't add any trace indices since we don't know them for unrelated k-grams

            for k_gram in k_grams_unrelated_to_deviations:
                if k_gram not in self.knowledge_base:
                    continue
                self.knowledge_base[k_gram].add_unrelated_to_deviating()

            # update correlation coefficients
            positive_correlation = False
            change_in_correlation = 0.0
            for k_gram in self.knowledge_base.keys():
                prior = self.knowledge_base[k_gram].correlation
                self.knowledge_base[k_gram].update_correlation()
                posterior = self.knowledge_base[k_gram].correlation
                change_in_correlation += abs(posterior - prior)
                if self.knowledge_base[k_gram].correlation > 0:
                    positive_correlation = True
            self.sample.correlation_changes.append(change_in_correlation)

            if self.sample.first_positive_correlation_at < 0 and positive_correlation:
                self.sample.first_positive_correlation_at = len(self.sample.traces)


#
# class CombinedLogSampler:
#     """
#     Implements sampling based on activity sequence k-grams and event+trace features.
#     """
#
#     def __init__(self, log_name, log, k=3, p=3, batch_size=5, window_size=3, use_cache=True, alignment_cache={},
#                  verbose=False):
#         self.log_name = log_name
#         self.log = log
#         self.available_trace_ids = set(range(len(log)))
#         self.k = k
#         self.log_manager = CombinedLogPartitioning()
#         self.partitioned_log, prep_time = self.log_manager.partition(log_name, log, k, p, verbose)
#         self.knowledge_base = {}
#
#         self.batch_size = batch_size
#         self.window_size = window_size
#
#         self.use_cache = use_cache
#         self.alignment_cache = alignment_cache
#         self.alignment_params = {}
#
#         # initialize result container
#         self.sample = Sample()
#         self.sample.times["partitioning"] = prep_time
#         self.sample.times["alignment"] = 0.0
#         self.sample.times["sampling"] = 0.0
#
#     def explore(self, verbose=False):
#         """
#         Randomly samples traces to explore the search space.
#         """
#         # TODO: Explore with diverse sample
#         sampled_indices = self.log_manager.get_random_sample(sample_size=self.batch_size,
#                                                              indices=[*self.available_trace_ids])
#         if verbose:
#             print(f"EXPLORATION: Sampled {len(sampled_indices)} traces.")
#         return sampled_indices
#
#     def exploit(self, verbose=False):
#         """
#         Samples traces based on the distribution of (previously observed) positive correlations with particular features.
#         Returns random sample if no positive correlation is detected (less likely the later we are in sampling process).
#         """
#
#         # First pass over features - get sum of all features with positive correlation
#         distribution = {}
#         feature_sum = 0.0
#         for feature in self.knowledge_base.keys():
#             candidate_indices = [idx for idx in self.partitioned_log[feature] if idx in self.available_trace_ids]
#             if self.knowledge_base[feature].correlation > 0.0 and len(candidate_indices) > 0:
#                 feature_sum += self.knowledge_base[feature].correlation
#                 distribution[feature] = self.knowledge_base[feature].correlation
#
#         # if no correlation is known, keep exploring
#         if feature_sum == 0.0:
#             return self.explore()
#
#         # convert positive correlations into probability distribution proportional to their correlation
#         for correlating_feature in distribution.keys():
#             distribution[correlating_feature] /= feature_sum
#
#         # choose a feature using distribtution
#         chosen_feature = random.choices([x for x in distribution.keys()], [x for x in distribution.values()], k=1)[0]
#
#         if verbose:
#             print("EXPLOITATION yields: " + str(chosen_feature) + " Prob:" + str(
#                 distribution[chosen_feature]) + ", Knowledge:" + str(self.knowledge_base[chosen_feature]))
#
#         # randomly select new trace with chosen feature
#         candidate_indices = [x for x in self.partitioned_log[chosen_feature] if x in self.available_trace_ids]
#         sampled_indices = self.log_manager.get_random_sample(sample_size=self.batch_size,
#                                                              indices=candidate_indices)
#         return sampled_indices
#
#     def construct_sample(self, log, model, initial_marking, final_marking, sample_size,
#                          verbose=False):
#         """
#         Constructs a sample based on an exploration vs. exploitation guided sampling strategy.
#         """
#         # stop right away if sample size is larger than log size
#         if len(log) <= sample_size:
#             print("Sample size larger than log. Returning complete log")
#             return log
#
#         start_time = time.time()
#         # Initializing some stuff
#         for key in self.partitioned_log:
#             self.knowledge_base[key] = Coocurrence.coocurence_rule()
#
#         self.alignment_params = init_alignment_params(model)
#
#         # TODO decide on exploit vs explore, right now we have 80/20 split
#         learning_rate = 0.8
#
#         # Sampling process
#         while len(self.sample.traces) < sample_size:
#             if verbose:
#                 print("Sampling " + str(len(self.sample.traces)) + "/" + str(sample_size))
#             sampled_trace = None
#
#             # decide between exploration and exploitation
#             decision = random.random()
#             if decision > learning_rate:
#                 # exploration - pick a random trace
#                 sampled_indices = self.explore(verbose)
#             else:
#                 # exploitation - convert positive correlations into distribution and pick trace correspondign to chosen feature
#                 sampled_indices = self.exploit(verbose)
#
#             # add sample to output set, remove trace from log and partitioned log
#
#             # Remove sampled ids from set of not-yet-sampled ids
#             self.available_trace_ids.difference_update(sampled_indices)
#
#             # Extend sample
#             self.sample.traces.extend([log[i] for i in sampled_indices])
#
#             # Check trace wrt property of interest - here alignments - and update knowledge base accordingly
#             self._check_for_property(sampled_indices, model, initial_marking, final_marking, verbose)
#
#             if verbose:
#                 print(" > Updated knowledge base after trace analysis(only positive correlations):")
#                 for feature in self.knowledge_base.keys():
#                     if self.knowledge_base[feature].correlation > 0.0:
#                         print("      " + str(feature) + " : " + str(self.knowledge_base[feature]))
#                 print()
#
#         # finally, add correlations and global fitness value to result object
#         self.sample.correlations = self.knowledge_base
#         fitness = -1.0
#
#         shortest_path = alignments.apply_trace(log_implementation.Trace(), model, initial_marking,
#                                                final_marking, parameters=self.alignment_params)["cost"]
#         total_costs = 0.0
#         upper_bound_total_costs = 0.0
#         for i in range(len(self.sample.traces)):
#             total_costs += self.sample.alignments[i]["cost"]
#             upper_bound_total_costs += int(shortest_path) + len(self.sample.traces[i])
#         self.sample.fitness = 1 - (total_costs / upper_bound_total_costs)
#
#         # print(self.sample)
#
#         self.sample.times["sampling"] = (time.time() - start_time) - self.sample.times["alignment"]
#
#         return self.sample
#
#     def _check_for_property(self, trace_ids, model, initial_marking, final_marking, verbose=False):
#         """
#         Analyzes the sampled traces (i.e., computes alignments and determines preceding k-grams in case of deviations)
#         and updates the knowledge base accordingly.
#         """
#
#         # Calculate alignment
#         pbar = tqdm(trace_ids, desc=" > Analyzing sampled traces", file=sys.stdout, disable=not verbose)
#         for trace_id in pbar:
#             # pbar.set_description(" > Calculating alignments...")
#
#             trace = self.log[trace_id]
#
#             # TODO replace with proper trace classifier
#             event_representation = ""
#             alignment = None
#
#             # check if alignment has been calculated already for the given trace, if so grab it, otherwise calculate it
#             alignment_time = time.time()
#             if self.use_cache:
#                 for event in trace:
#                     event_representation = event_representation + " >> " + event["concept:name"]
#
#                 # Second condition checks whether an alignment has been actually precomputed for the given trace
#                 if event_representation in self.alignment_cache and self.alignment_cache[event_representation]:
#                     alignment = self.alignment_cache[event_representation]
#                 else:
#                     if verbose:
#                         print("Trace not found in precomputed cache!")
#                     alignment = alignments.apply(trace, model, initial_marking, final_marking,
#                                                  parameters=self.alignment_params)
#                     self.alignment_cache[event_representation] = alignment
#
#             else:
#                 alignment = alignments.apply(trace, model, initial_marking, final_marking,
#                                              parameters=self.alignment_params)
#             alignment_time = time.time() - alignment_time
#             self.sample.times["alignment"] += alignment_time
#             self.sample.alignments.append(alignment)
#
#             # LMstep[0] corresponds to an event in the trace and LMstep[1] corresponds to a transition in the model
#             # The following cases are possible:
#             # 1. Sync move (LMstep[0] == LMstep[1]): Both trace and model advance in the same way
#             # 2. Move on log (LMstep[1] == '>>'): A move in the log that could not be mimicked by the model: deviation
#             # 3. Move on model (LMstep[0] == '>>'):
#             #   3.1 with hidden transition (LMstep[1] == None): OK, no deviation
#             #   3.2 without hidden transition (LMstep[1] != None): not fit, deviation between log and model
#             # pbar.set_description(" > Updating k-gram-deviation-statistics...")
#             deviation_points = []
#             trace_idx = 0
#             deviation_found = False
#             for LMstep in alignment["alignment"]:
#                 # Sync move
#                 if LMstep[0] == LMstep[1]:
#                     pass
#                 # Move on log only, deviation
#                 elif LMstep[1] == '>>':
#                     deviation_points.append(trace_idx)
#
#                     # Update deviation statistics
#                     self.sample.total_deviations += 1
#                     if not deviation_found:
#                         self.sample.trace_deviations += 1
#                         deviation_found = True
#                     self.sample.activity_deviations[LMstep[0]] += 1
#
#                 # Move on model only
#                 elif LMstep[0] == '>>':
#                     # Hidden transition, no deviation
#                     if LMstep[1] == None:
#                         pass
#                     # No hidden transition, deviation
#                     else:
#                         deviation_points.append(trace_idx)
#
#                         # Update deviation statistics
#                         self.sample.total_deviations += 1
#                         if not deviation_found:
#                             self.sample.trace_deviations += 1
#                             deviation_found = True
#                         self.sample.activity_deviations[LMstep[1]] += 1
#
#                 # Increment pointer to current position in trace
#                 if LMstep[0] != '>>':
#                     trace_idx += 1
#
#             # Add k-grams/features to list of deviating or conforming sets based on their marking
#             deviating_features = []
#             conforming_features = []
#
#             # Determine indices in the trace from which to retrieve the relevant k-grams from
#             deviating_k_gram_indices = set()
#             for d in deviation_points:
#                 start = max(d - self.window_size, 0)
#                 end = min(max(d - self.k + 1, 0), d)
#                 deviating_k_gram_indices.update([*range(start, end)])
#
#             # Get trace k-grams and assign to list of deviating/conforming features
#             k_grams = self.log_manager.get_ordered_k_grams(trace_id, self.k)
#             non_conformance_in_trace = False
#             for idx, kgram in enumerate(k_grams):
#                 if idx in deviating_k_gram_indices:
#                     deviating_features.append(kgram)
#                     non_conformance_in_trace = True
#                 else:
#                     conforming_features.append(kgram)
#
#             # Determine indices of trace events related to a deviation point (window)
#             deviating_feature_indices = set()
#             for d in deviation_points:
#                 start = max(d - self.window_size, 0)
#                 end = d + 1
#                 deviating_feature_indices.update([*range(start, end)])
#
#             # Get event-level features and assign to list of deviating/conforming features
#             for idx, event in enumerate(trace):
#                 data, features = create_feature_encoding([{event}], self.log_name, include_event_level=True,
#                                                          include_trace_level=False)
#                 if idx in deviating_feature_indices:
#                     deviating_features.extend(features)
#                     non_conformance_in_trace = True
#                 else:
#                     conforming_features.extend(features)
#
#             # Add trace-level feature to set, either conforming or deviating, depending on existence of non-conformance
#             data, features = create_feature_encoding([trace], self.log_name, include_event_level=False,
#                                                      include_trace_level=True)
#             if non_conformance_in_trace:
#                 deviating_features.extend(features)
#             else:
#                 conforming_features.extend(features)
#
#             # Determine non-occuring k-grams and features
#             features_unrelated_to_deviations = []
#             features_unrelated_to_conforming = []
#             for feature in self.partitioned_log.keys():
#                 if feature not in deviating_features:
#                     features_unrelated_to_deviations.append(feature)
#                 if feature not in conforming_features:
#                     features_unrelated_to_conforming.append(feature)
#
#             # increase counters of all features depending on their (non)-cooccurence with conformance/deviations
#             for feature in deviating_features:
#                 if feature not in self.knowledge_base:
#                     continue
#                 self.knowledge_base[feature].add_deviating()
#
#             for feature in conforming_features:
#                 if feature not in self.knowledge_base:
#                     continue
#                 self.knowledge_base[feature].add_conforming()
#
#             for feature in features_unrelated_to_conforming:
#                 if feature not in self.knowledge_base:
#                     continue
#                 self.knowledge_base[feature].add_unrelated_to_conforming()
#
#             for feature in features_unrelated_to_deviations:
#                 if feature not in self.knowledge_base:
#                     continue
#                 self.knowledge_base[feature].add_unrelated_to_deviating()
#
#             # update correlation coefficients
#             positive_correlation = False
#             change_in_correlation = 0.0
#             for feature in self.knowledge_base.keys():
#                 prior = self.knowledge_base[feature].correlation
#                 self.knowledge_base[feature].update_correlation()
#                 posterior = self.knowledge_base[feature].correlation
#                 change_in_correlation += abs(posterior - prior)
#                 if self.knowledge_base[feature].correlation > 0:
#                     positive_correlation = True
#             self.sample.correlation_changes.append(change_in_correlation)
#
#             if self.sample.first_positive_correlation_at < 0 and positive_correlation:
#                 self.sample.first_positive_correlation_at = len(self.sample.traces)


class RandomLogSampler:
    def __init__(self, use_cache=True, alignment_cache={}):
        self.alignment_params = {}
        self.use_cache = use_cache
        self.alignment_cache = alignment_cache

        self.sample = Sample()
        self.sample.times["partitioning"] = 0.0
        self.sample.times["alignment"] = 0.0
        self.sample.times["sampling"] = 0.0

    def construct_sample(self, log, model, initial_marking, final_marking, sample_size, calculate_alignments=True):
        sampling_t = time.time()
        sampled_traces = random.sample(log, sample_size)
        self.sample.traces.extend(sampled_traces)
        self.sample.times["sampling"] = time.time() - sampling_t

        self.alignment_params = init_alignment_params(model)

        alignment_t = time.time()
        if calculate_alignments:
            if self.use_cache:
                pbar = tqdm(sampled_traces, desc=" > Sampling...", file=sys.stdout, disable=False)
                for trace in pbar:
                    deviation_found = False
                    event_representation = ""
                    for event in trace:
                        event_representation = event_representation + " >> " + event["concept:name"]

                    # Second condition checks whether an alignment has been actually precomputed for the given trace
                    if event_representation in self.alignment_cache and self.alignment_cache[event_representation]:
                        alignment = self.alignment_cache[event_representation]
                    else:
                        # if verbose:
                        #    print("Trace not found in precomputed cache!")
                        alignment = alignments.apply(trace, model, initial_marking, final_marking,
                                                     parameters=self.alignment_params)
                        self.alignment_cache[event_representation] = alignment
                    self.sample.alignments.append(alignment)

                    for LMstep in alignment["alignment"]:
                        # Sync move
                        if LMstep[0] == LMstep[1]:
                            pass
                        # Move on log only, deviation
                        elif LMstep[1] == '>>':
                            # Update deviation statistics
                            self.sample.total_deviations += 1
                            if not deviation_found:
                                self.sample.trace_deviations += 1
                                deviation_found = True
                            self.sample.activity_deviations[LMstep[0]] += 1

                        # Move on model only
                        elif LMstep[0] == '>>':
                            # Hidden transition, no deviation
                            if LMstep[1] == None:
                                pass
                            # No hidden transition, deviation
                            else:
                                # Update deviation statistics
                                self.sample.total_deviations += 1
                                if not deviation_found:
                                    self.sample.trace_deviations += 1
                                    deviation_found = True
                                self.sample.activity_deviations[LMstep[1]] += 1
            else:
                pbar = tqdm(sampled_traces, desc=" > Sampling...", file=sys.stdout, disable=False)
                for trace in pbar:
                    deviation_found = False
                    alignment = alignments.apply(trace, model, initial_marking, final_marking,
                                                 parameters=self.alignment_params)
                    self.sample.alignments.append(alignment)

                    for LMstep in alignment["alignment"]:
                        # Deviation
                        if (LMstep[0] == '>>' and LMstep[1] is not None) or LMstep[1] == '>>':
                            self.sample.total_deviations += 1
                            if not deviation_found:
                                self.sample.trace_deviations += 1
                                deviation_found = True

            fitness = -1.0

            shortest_path = alignments.apply_trace(log_implementation.Trace(), model, initial_marking,
                                                   final_marking, parameters=self.alignment_params)["cost"]
            total_costs = 0.0
            upper_bound_total_costs = 0.0
            for i in range(len(self.sample.traces)):
                total_costs += self.sample.alignments[i]["cost"]
                upper_bound_total_costs += int(shortest_path) + len(self.sample.traces[i])
            self.sample.fitness = 1 - (total_costs / upper_bound_total_costs)

        self.sample.times["alignment"] = time.time() - alignment_t
        return self.sample


class LongestTraceVariantLogSampler():
    def __init__(self, use_cache=True, alignment_cache={}):
        self.alignment_params = {}
        self.use_cache = use_cache
        self.alignment_cache = alignment_cache

        self.sample = Sample()
        self.sample.times["partitioning"] = 0.0
        self.sample.times["alignment"] = 0.0
        self.sample.times["sampling"] = 0.0

    def construct_sample(self, log, model, initial_marking, final_marking, sample_size, calculate_alignments=True):
        sampling_t = time.time()

        #TODO replace with get_variant_as_tuple(log)
        trace_variants = get_variants(log)
        variant_list = []
        for variant in trace_variants.values():
            variant_list.append(variant[0])

        sorted_trace_variants = sorted(variant_list, key=len)
        sorted_trace_variants.reverse()

        sampled_traces = sorted_trace_variants[:sample_size]

        self.sample.traces.extend(sampled_traces)
        self.sample.times["sampling"] = time.time() - sampling_t

        alignment_t = time.time()
        self.alignment_params = init_alignment_params(model)

        if calculate_alignments:
            if self.use_cache:
                pbar = tqdm(sampled_traces, desc=" > Sampling...", file=sys.stdout, disable=False)
                for trace in pbar:
                    deviation_found = False
                    event_representation = ""
                    for event in trace:
                        event_representation = event_representation + " >> " + event["concept:name"]

                    # Second condition checks whether an alignment has been actually precomputed for the given trace
                    if event_representation in self.alignment_cache and self.alignment_cache[event_representation]:
                        alignment = self.alignment_cache[event_representation]
                    else:
                        # if verbose:
                        #    print("Trace not found in precomputed cache!")
                        alignment = alignments.apply(trace, model, initial_marking, final_marking,
                                                     parameters=self.alignment_params)
                        self.alignment_cache[event_representation] = alignment
                    self.sample.alignments.append(alignment)

                    for LMstep in alignment["alignment"]:
                        # Sync move
                        if LMstep[0] == LMstep[1]:
                            pass
                        # Move on log only, deviation
                        elif LMstep[1] == '>>':
                            # Update deviation statistics
                            self.sample.total_deviations += 1
                            if not deviation_found:
                                self.sample.trace_deviations += 1
                                deviation_found = True
                            self.sample.activity_deviations[LMstep[0]] += 1

                        # Move on model only
                        elif LMstep[0] == '>>':
                            # Hidden transition, no deviation
                            if LMstep[1] == None:
                                pass
                            # No hidden transition, deviation
                            else:
                                # Update deviation statistics
                                self.sample.total_deviations += 1
                                if not deviation_found:
                                    self.sample.trace_deviations += 1
                                    deviation_found = True
                                self.sample.activity_deviations[LMstep[1]] += 1
            else:
                pbar = tqdm(sampled_traces, desc=" > Sampling...", file=sys.stdout, disable=False)
                for trace in pbar:
                    deviation_found = False
                    alignment = alignments.apply(trace, model, initial_marking, final_marking,
                                                 parameters=self.alignment_params)
                    self.sample.alignments.append(alignment)

                    for LMstep in alignment["alignment"]:
                        # Deviation
                        if (LMstep[0] == '>>' and LMstep[1] is not None) or LMstep[1] == '>>':
                            self.sample.total_deviations += 1
                            if not deviation_found:
                                self.sample.trace_deviations += 1
                                deviation_found = True

            fitness = -1.0

            shortest_path = alignments.apply_trace(log_implementation.Trace(), model, initial_marking,
                                                   final_marking, parameters=self.alignment_params)["cost"]
            total_costs = 0.0
            upper_bound_total_costs = 0.0
            for i in range(len(self.sample.traces)):
                total_costs += self.sample.alignments[i]["cost"]
                upper_bound_total_costs += int(shortest_path) + len(self.sample.traces[i])
            self.sample.fitness = 1 - (total_costs / upper_bound_total_costs)

        self.sample.times["alignment"] = time.time() - alignment_t
        return self.sample
