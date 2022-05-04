import os.path
import pickle
import time
from itertools import combinations

import numpy
import numpy as np
import pm4py
from numpy import mean, std
from pm4py.objects.log.importer.xes import importer as xes_importer
from pm4py.objects.petri_net.exporter import exporter as pnml_exporter
from pm4py.algo.conformance.alignments.petri_net import algorithm as alignments

import LogIndexing
from SamplingAlgorithms import FeatureGuidedLogSampler, SequenceGuidedLogSampler, RandomLogSampler, \
    LongestTraceVariantLogSampler

# EVALUATION PARAMETER #
log_names = ["Sepsis_Cases_-_Event_Log.xes", "BPI_Challenge_2012.xes", "BPI_Challenge_2018.xes"]
approaches = ["Random", "Longest", "Feature", "Sequence"]
samples_sizes = [100, 200, 300, 400, 500]
repetitions = 10
cached_alignments = True


def main():
    for log_name in log_names:
        log, model, initial_marking, final_marking = load_inputs(log_name, modelpath="models")

        eval_partitions(log, log_name)
        eval_quality(log, log_name, model, initial_marking, final_marking)
        eval_runtime(log, log_name, model, initial_marking, final_marking)


def eval_partitions(log, log_name):
    """
    For a given log, for each of the guided sampling procedures, constructs the log index, and writes partition
    statistics to partitions_<log_name>.csv
    """
    t = open(os.path.join("results", "partitions_" + log_name + ".csv"), "w")
    t.write("approach;total_size;partition;partition_size\n")

    for approach in approaches:
        print("Partition Evaluation: " + str(log_name) + " : " + str(approach))
        if approach == "Random":
            pass
        if approach == "Longest":
            pass
        if approach == "Feature":
            sampler = FeatureGuidedLogSampler(log, index_file="index_files/" + log_name + ".index")
            partitioning = sampler.partitioned_log
            for partition in partitioning.keys():
                t.write(
                    ";".join((str(approach), str(len(partitioning)), str(partition),
                              str(len(partitioning[partition])) + "\n")))
        if approach == "Sequence":
            sampler = SequenceGuidedLogSampler(log, batch_size=1, index_file="index_files/" + log_name + ".index")
            partitioning = sampler.partitioned_log
            for partition in partitioning.keys():
                t.write(
                    ";".join((str(approach), str(len(partitioning)), str(partition),
                              str(len(partitioning[partition])) + "\n")))
    t.close()


def eval_quality(log, log_name, model, initial_marking, final_marking):
    """
    For a given log and model, for each sampling procedure, repeatedly executes the complete sampling chain (for guided
    approaches this includes partitioning + sampling), writing different stats to files.
    The files are:
        fitness_<log_name>.csv
            for each repetition, sample fitness, number of deviating traces,
            number of deviating activities, runtimes sampling + partitioning
        knowledge_base_convergence_<log_name>.csv
            for each repetition, for each drawn trace, absolute change in knowledge based induced by this trace,
            the distance in correlations to the knowledge base correlations evaluated over the complete log
        knowledge_base_correlation_<log_name>.csv
            for each repetition, each considered feature in the knowledge with its correlation to deviations
        activities_<log_name>.csv
            for each repetition, statistics about the number of activities found to be deviating
    """
    if cached_alignments:
        print("Quality Evaluation: Loading precomputed alignments...")
        aligned_traces = pickle.load(open(os.path.join("alignment_cache", log_name + ".align"), "rb"))
        trace_keys = []
        for idx, trace in enumerate(log):
            event_representation = ""
            for event in trace:
                event_representation = event_representation + " >> " + event["concept:name"]
            trace_keys.append(event_representation)
        assert (len(trace_keys) == len(log))
        alignment_cache = dict(zip(trace_keys, aligned_traces))
    else:
        alignment_cache = {}

    fitness_results = open(os.path.join("results", "fitness_" + log_name + ".csv"), "w")
    fitness_results.write(
        "approach;sample_size;repetition;trace_variants;deviating_traces;total_deviations;num_deviating_activities"
        ";fitness;time_partitioning;time_sampling;time_alignments\n")

    knowledge_base_results = open(os.path.join("results", "knowledge_base_convergence_" + log_name + ".csv"), "w")
    knowledge_base_results.write(
        "approach;sample_size;repetition;dist_to_baseline;first_positive_at;trace_idx;cor_change\n")

    correlation_results = open(os.path.join("results", "knowledge_base_correlations_" + log_name + ".csv"), "w")
    correlation_results.write("approach;sample_size;repetition;feature;informative;correlation\n")

    activity_results = open(os.path.join("results", "activities_" + log_name + ".csv"), "w")
    activity_results.write("approach;sample_size;avg_dev_activities;stddev;avg_pw_similarity;stddev\n")

    for approach in approaches:
        for sample_size in samples_sizes:
            deviating_activities = []
            for i in range(repetitions):
                print("Quality Evaluation: " + str(log_name) + " : " + str(approach) + " : "
                      + str(sample_size) + " : " + str(i))

                sample = None
                if approach == "Random":
                    sampler = RandomLogSampler(use_cache=True)
                    sampler.alignment_cache = alignment_cache
                    sample = sampler.construct_sample(log, model, initial_marking, final_marking, sample_size)

                if approach == "Longest":
                    sampler = LongestTraceVariantLogSampler(use_cache=True)
                    sampler.alignment_cache = alignment_cache
                    sample = sampler.construct_sample(log, model, initial_marking, final_marking, sample_size)

                if approach == "Feature":
                    sampler = FeatureGuidedLogSampler(log, use_cache=True,
                                                      index_file="index_files/" + log_name + ".index")
                    sampler.alignment_cache = alignment_cache
                    sample = sampler.construct_sample(log, model, initial_marking, final_marking, sample_size)

                if approach == "Sequence":
                    sampler = SequenceGuidedLogSampler(log, batch_size=1, use_cache=True,
                                                       index_file="index_files/" + log_name + ".index")
                    sampler.alignment_cache = alignment_cache
                    sample = sampler.construct_sample(log, model, initial_marking, final_marking, sample_size)

                # get number of trace variants / constructed alignments
                trace_variants = {}
                for trace in sample.traces:
                    event_representation = ""
                    for event in trace:
                        event_representation = event_representation + " >> " + event["concept:name"]

                    if event_representation not in trace_variants:
                        trace_variants[event_representation] = ""

                print(" > " + str(sample.times["partitioning"]) + ", " + str(sample.times["alignment"]) + ", "
                      + str(sample.times["sampling"]))
                fitness_results.write(
                    ";".join((str(approach), str(sample_size), str(i), str(len(trace_variants)),
                              str(sample.trace_deviations), str(sample.total_deviations),
                              str(len(sample.activity_deviations.keys())), str(sample.fitness),
                              str(sample.times["partitioning"]), str(sample.times["sampling"]),
                              str(sample.times["alignment"]) + "\n")))

                # next, get dist of knowledge base to ground truth
                if approach != "Random" and approach != "Longest":
                    sample_knowledge_base = sample.correlations
                    ground_truth_correlations = None
                    if approach == "Feature":
                        ground_truth_correlations = pickle.load(
                            open(os.path.join("knowledge_base_cache", log_name + "_feature.knowledge_base"), "rb"))

                    if approach == "Sequence":
                        ground_truth_correlations = pickle.load(
                            open(os.path.join("knowledge_base_cache", log_name + "_sequence.knowledge_base"), "rb"))

                    a = np.array([x.correlation for x in ground_truth_correlations.values()])
                    b = np.array([x.correlation for x in sample_knowledge_base.values()])
                    dist = numpy.linalg.norm(a - b)

                    for idx, change in enumerate(sample.correlation_changes):
                        knowledge_base_results.write(";".join((str(approach), str(sample_size), str(i),
                                                               str(dist),
                                                               str(sample.first_positive_correlation_at), str(idx),
                                                               str(change) + "\n")))

                correlations = {}
                for correlation in sample.correlations.keys():
                    correlations[correlation] = float(sample.correlations[correlation].correlation)
                sorted_correlations = sorted(correlations.items(), key=lambda item: item[1])
                sorted_correlations.reverse()
                sorted_names = [x[0] for x in sorted_correlations]
                for name in sorted_names:
                    if correlations[name] > 0:
                        correlation_results.write(
                            ";".join((str(approach), str(sample_size), str(i), str(name), str(True),
                                      str(correlations[name]) + "\n")))
                    else:
                        correlation_results.write(
                            ";".join((str(approach), str(sample_size), str(i), str(name), str(False),
                                      str(correlations[name]) + "\n")))


            # Compute deviating activity stats
            pw_similarities = []
            activity_sizes = list(map(lambda s: len(s), deviating_activities))
            for x, y in combinations(deviating_activities, 2):
                pw_similarities.append(jaccard_sim(x, y))
            activity_results.write(
                ";".join((str(approach),
                          str(sample_size),
                          str(mean(activity_sizes)),
                          str(std(activity_sizes)),
                          str(mean(pw_similarities)),
                          str(std(pw_similarities)))
                         ) + "\n")

    fitness_results.close()
    knowledge_base_results.close()
    correlation_results.close()
    # deviation_dist.close()
    activity_results.close()


def jaccard_sim(s, t):
    """
    Computes the Jaccard similarity of two sets s and t
    """
    if len(s) == 0 and len(t) == 0:
        return 1
    return float(len(s.intersection(t)) / float(len(s.union(t))))


# evaluates total runtime of alignment calculations on increasing sample sizes until either one run exceeds timeout or
# complete log has been considered
def eval_runtime(log, log_name, model, initial_marking, final_marking, timeout=10800):
    """
    For a given log and model, samples n traces, constructing alignments for newly sampled trace variants, until
    n trace have been sampled, or a timeout is reached. Results are written to alignment_runtime_<log_name>.csv
    """
    runtime_results = open(os.path.join("results", "alignment_runtime_" + log_name + ".csv"), "w")
    runtime_results.write(
        "sample_size;time\n")

    timeout_reached = False
    mean_runtime = 0
    for sample_size in samples_sizes:
        for i in range(10):
            print("Runtime Evaluation: " + str(log_name) + " : " + str(
                sample_size) + " : " + str(i))

            if not timeout_reached:
                sample_t = RandomLogSampler(use_cache=True, alignment_cache={}) \
                    .construct_sample(log, model, initial_marking, final_marking, sample_size)

                print(" > " + str(str(sample_t.times["alignment"])))
                mean_runtime = sample_t.times["alignment"] / sample_size
                if sample_t.times["alignment"] > timeout:
                    print(" > TIMEOUT REACHED")
                    timeout_reached = True
                    runtime_results.write(";".join((str(sample_size), ">" + str(timeout) + "\n",)))
                    continue
                else:
                    runtime_results.write(";".join((str(sample_size), str(sample_t.times["alignment"]) + "\n",
                                                    )))

            else:
                runtime_results.write(";".join((str(sample_size), ">" + str(timeout) + "\n",
                                                )))

    for i in range(10):
        if (not timeout_reached) and mean_runtime * len(log) < timeout:
            print(f" > FULL LOG ANALYSIS (Expected time={mean_runtime * len(log)})")
            start_t = time.time()

            alignments.apply(log, model, initial_marking, final_marking, parameters=construct_alignment_param(model))
            total_t = time.time() - start_t
            print(" > " + str(str(total_t)))

            runtime_results.write(";".join("complete;" + str(total_t) + "\n",
                                           ))
        else:
            print(" > TIMEOUT REACHED")
            runtime_results.write(";".join("complete;>" + str(timeout) + "\n",
                                           ))
    runtime_results.close()


def load_inputs(log_name, modelpath=None):
    """
    loads a .xes-log, and discovers a model from it, if no model is present already
    """
    log = xes_importer.apply(os.path.join("logs", log_name))
    model = None
    if modelpath is None or not os.path.exists(os.path.join(str(modelpath), str(log_name) + ".pnml")):
        print("Model Discovery ")
        model, initial_marking, final_marking = pm4py.discover_petri_net_inductive(log, noise_threshold=0.2)
        pnml_exporter.apply(model, initial_marking, os.path.join(str(modelpath), str(log_name) + ".pnml"),
                            final_marking=final_marking)
    else:
        print("Loading Model")
        model, initial_marking, final_marking = pm4py.read_pnml(os.path.join(str(modelpath), str(log_name) + ".pnml"))
    print("Done")
    return log, model, initial_marking, final_marking


def construct_alignment_param(model):
    """
    constructs the cost function used for alignment construction
    """
    model_cost_function = dict()
    sync_cost_function = dict()
    for t in model.transitions:
        if t.label is not None:
            model_cost_function[t] = 1
            sync_cost_function[t] = 0
        else:
            model_cost_function[t] = 0
    trace_cost_function = ConstantList(1)
    return {alignments.Parameters.PARAM_MODEL_COST_FUNCTION: model_cost_function,
            alignments.Parameters.PARAM_SYNC_COST_FUNCTION: sync_cost_function,
            alignments.Parameters.PARAM_TRACE_COST_FUNCTION: trace_cost_function}


class ConstantList:
    """
    Implements a list that returns the same value for each index (e.g., cost)
    """

    def __init__(self, value):
        self.value = value

    def __getitem__(self, item):
        return self.value


if __name__ == '__main__':
    main()
