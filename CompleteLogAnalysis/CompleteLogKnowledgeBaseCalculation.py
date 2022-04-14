import pickle

import Coocurrence
import feature_encodings
from SamplingAlgorithms import Sample
from tqdm import tqdm

from LogIndexing import SequenceBasedLogPreprocessor

#TODO get rid of skipping of traces, for which no alignment has been precomputed
class CompleteFeatureLogAnalyzer:
    """
    Computes feature correlations for the complete log.
    """
    def __init__(self, window_size=3, n_gram_size=3, index_file=None):
        self.partitioned_log = {}
        self.knowledge_base = {}
        self.window_size = window_size
        self.n_gram_size = n_gram_size
        self.index_file = index_file

    def analyze_log(self, log_name, log, aligned_traces, partitioned_log, verbose=False):

        # Initializing some stuff
        self.partitioned_log = partitioned_log
        #print(partitioned_log)
        for key in self.partitioned_log:
            self.knowledge_base[key] = Coocurrence.coocurence_rule()

        # Sampling process
        pbar = tqdm(list(zip(log, aligned_traces)), desc="Analyzing complete log...", disable=not verbose)
        for trace, aligned_trace in pbar:
            # pbar.set_description(" > Calculating alignments...")
            self._check_for_property(log_name, trace, aligned_trace)

        # Print knowledge base
        for feature in self.knowledge_base.keys():
            if self.knowledge_base[feature].correlation > 0.0:
                print("      " + str(feature) + " : " + str(self.knowledge_base[feature]))
        return self.knowledge_base

    def _check_for_property(self, log_name, trace, aligned_trace):
        deviation_contexts = []
        if aligned_trace is None:
            return

        idx = 0
        deviation_contexts = []

        deviation_found = False
        deviating_n_grams = []
        for LMstep in aligned_trace["alignment"]:
            # Sync move
            if LMstep[0] == LMstep[1]:
                deviation_contexts.append(False)

            # Move on log only, deviation
            elif LMstep[1] == '>>':
                deviation_contexts.append(True)
                for i in range(max(0, len(deviation_contexts) - self.window_size), len(deviation_contexts)):
                    # print(i)
                    deviation_contexts[i] = True

                deviating_n_grams.extend(self.get_deviation_context_n_grams(trace, len(deviation_contexts) - 1))

            # Move on model only
            elif LMstep[0] == '>>':
                # Hidden transition, no deviation
                if LMstep[1] == None:
                    pass
                # No hidden transition, deviation
                else:
                    for i in range(max(0, len(deviation_contexts) - self.window_size), len(deviation_contexts)):
                        deviation_contexts[i] = True

                    deviating_n_grams.extend(self.get_deviation_context_n_grams(trace, len(deviation_contexts) - 1))

            else:
                print("Should not happen.")

        # add event-level features to list of deviating or conforming sets based on their marking
        deviating_features = []
        conforming_features = []
        non_conformance_in_trace = False
        for idx, is_deviating in enumerate(deviation_contexts):
            current_event = {trace[idx]}
            data, features = feature_encodings.create_feature_encoding_from_index([current_event],
                                                                                  feature_encodings.feature_types.EVENT_LEVEL,
                                                                                  self.index_file)
            if is_deviating:
                deviating_features.extend(features)
                non_conformance_in_trace = True
            else:
                conforming_features.extend(features)

        # add n-gram features
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

        # add trace-level feature to set, either conforming or deviating, depending on existence of non-conformance
        data, features = feature_encodings.create_feature_encoding_from_index([trace],
                                                 feature_encodings.feature_types.TRACE_LEVEL,
                                                 self.index_file)
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


class CompleteSequenceLogAnalyzer:
    """
    Computes k-gram correlations for the complete log.
    """

    def __init__(self, log, k=3, b=10, r=10, window_size=5, verbose=False):
        self.log = log
        self.k = k
        self.log_manager = SequenceBasedLogPreprocessor(log, k, b, r, verbose)
        self.window_size = window_size
        self.sample_ids = set(range(len(log)))
        self.sample = []
        self.knowledge_base = {}

    def analyze_log(self, log_name, aligned_traces, verbose=False):

        # Initializing knowledge base
        for key in self.log_manager.k_gram_dict.keys():
            self.knowledge_base[key] = Coocurrence.coocurence_rule()

        # Sampling process
        pbar = tqdm(list(enumerate(aligned_traces)), desc="Analyzing complete log...", disable=not verbose)
        for trace_id, aligned_trace in pbar:
            # pbar.set_description(" > Calculating alignments...")
            self._check_for_property(log_name, trace_id, aligned_trace)

        # Print knowledge base
        for feature in self.knowledge_base.keys():
            if self.knowledge_base[feature].correlation > 0.0:
                print("      " + str(feature) + " : " + str(self.knowledge_base[feature]))
        return self.knowledge_base

    def _check_for_property(self, log_name, trace_id, aligned_trace):
        if aligned_trace is None:
            return

        deviation_points = []
        trace_idx = 0
        for LMstep in aligned_trace["alignment"]:
            # Mark all events that appear in a deviation context
            # LMstep[0] corresponds to an event in the trace and LMstep[1] corresponds to a transition in the model
            # The following cases are possible:
            # 1. Sync move (LMstep[0] == LMstep[1]): Both trace and model advance in the same way
            # 2. Move on log (LMstep[1] == '>>'): A move in the log that could not be mimicked by the model: deviation
            # 3. Move on model (LMstep[0] == '>>'):
            #   3.1 with hidden transition (LMstep[1] == None): OK, no deviation
            #   3.2 without hidden transition (LMstep[1] != None): not fit, deviation between log and model

            # Deviation, mark current trace idx
            if (LMstep[0] == '>>' and LMstep[1] is not None) or LMstep[1] == '>>':
                deviation_points.append(trace_idx)

            # Increment pointer to current position in trace
            if LMstep[0] != '>>':
                trace_idx += 1

        # Determine indices in the trace from which to retrieve the relevant k-grams from (context window)
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

        # TODO remove safety checks for unknown features here and move them somewhere earlier in a single pass
        for k_gram in deviating_k_grams:
            if k_gram not in self.knowledge_base:
                print("ADDING UNKNOWN k-gram: " + k_gram)
                self.knowledge_base[k_gram] = Coocurrence.coocurence_rule()
            self.knowledge_base[k_gram].add_deviating()
            self.knowledge_base[k_gram].add_trace_index(trace_id)

        for k_gram in conforming_k_grams:
            if k_gram not in self.knowledge_base:
                print("  > ADDING UNKNOWN k-gram: " + k_gram)
                self.knowledge_base[k_gram] = Coocurrence.coocurence_rule()
            self.knowledge_base[k_gram].add_conforming()
            self.knowledge_base[k_gram].add_trace_index(trace_id)

        for k_gram in k_grams_unrelated_to_conforming:
            if k_gram not in self.knowledge_base:
                print("  > ADDING UNKNOWN k-gram: " + k_gram)
                self.knowledge_base[k_gram] = Coocurrence.coocurence_rule()
            self.knowledge_base[k_gram].add_unrelated_to_conforming()
            # Note: we can't add any trace indices since we don't know them for unrelated k-grams

        for k_gram in k_grams_unrelated_to_deviations:
            if k_gram not in self.knowledge_base:
                print("  > ADDING UNKNOWN k-gram: " + k_gram)
                self.knowledge_base[k_gram] = Coocurrence.coocurence_rule()
            self.knowledge_base[k_gram].add_unrelated_to_deviating()

        # update correlation coefficients
        for k_gram in self.knowledge_base.keys():
            self.knowledge_base[k_gram].update_correlation()