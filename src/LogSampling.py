import argparse
import os.path
import pickle
import time
import pm4py
from pm4py.objects.log.importer.xes import importer as xes_importer
from pm4py.objects.petri_net.exporter import exporter as pnml_exporter

import LogIndexing
from SamplingAlgorithms import FeatureGuidedLogSampler, SequenceGuidedLogSampler, \
    RandomLogSampler, LongestTraceVariantLogSampler


def construct_sample(log_name, algorithm, sample_size, index_file, verbose_output):
    log, model, initial_marking, final_marking = load_inputs(log_name, modelpath="models")

    cached_alignments = False

    if cached_alignments:
        print("Quality Evaluation: Loading precomputed alignments...")
        aligned_traces = pickle.load(open(os.path.join("alignment_cache", log_name + ".align"), "rb"))
        trace_keys = []
        for trace in log:
            event_representation = ""
            for event in trace:
                event_representation = event_representation + " >> " + event["concept:name"]
            trace_keys.append(event_representation)

        assert (len(trace_keys) == len(log))
        alignment_cache = dict(zip(trace_keys, aligned_traces))
    else:
        alignment_cache = {}

    t_start = time.time()
    sample = None
    if algorithm == "feature":
        partitioned_log, preprocessing_time = LogIndexing.FeatureBasedPartitioning().partition(log_name, log, index_file) #verbose=verbose_output)
        sampling_controller = FeatureGuidedLogSampler(preprocessing_time=preprocessing_time)
        sample = sampling_controller.construct_sample(log_name, log, model, initial_marking, final_marking, partitioned_log, int(sample_size), verbose=verbose_output)
    elif algorithm == "behavioural":
        sampling_controller = SequenceGuidedLogSampler(log, batch_size=1, verbose=verbose_output)
        sample = sampling_controller.construct_sample(log_name, log, model, initial_marking, final_marking, int(sample_size), verbose=verbose_output)

    #only used for debugging and evaluation purposes
    elif algorithm == "Random":
        sampling_controller = RandomLogSampler()
        sample = sampling_controller.construct_sample(log, model, initial_marking, final_marking, int(sample_size))

    elif algorithm == "Longest":
        sampling_controller = LongestTraceVariantLogSampler()
        sample = sampling_controller.construct_sample(log, model, initial_marking, final_marking, int(sample_size))

    print(f"Sampling done. Total time elapsed: {(time.time()-t_start):.3f}")
    print(f"Times: {sample.times}")
    return sample


def load_inputs(log_name, modelpath=None):
    log = log = xes_importer.apply(os.path.join("logs", log_name))
    model = None
    if modelpath is None or not os.path.exists(os.path.join(str(modelpath), str(log_name) + ".pnml")):
        print("Model Discovery ")
        model, initial_marking, final_marking = pm4py.discover_petri_net_inductive(log, noise_threshold=0.2)
        pnml_exporter.apply(model, initial_marking, os.path.join(str(modelpath), str(log_name) + ".pnml"),
                            final_marking=final_marking)
    else:
        print("Loading Model")
        model, initial_marking, final_marking = pm4py.read_pnml(os.path.join(str(modelpath), str(log_name)+".pnml"))
    print("Done")
    return log, model, initial_marking, final_marking


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("log_file", help="the name of the xes-log (located in logs) from which to sample")
    parser.add_argument("algorithm", help="the sampling strategy to use [feature, behavioural]")
    parser.add_argument("sample_size", help="the size of the final sample")
    parser.add_argument("--index_file", help="the name of the index file containing the features considered during "
                                             "indexing. If none is supplied, all features are considered")
    parser.add_argument("--verbose", help="increase verbosity of output", action="store_true")
    args = parser.parse_args()

    construct_sample(args.log_file, args.algorithm, args.sample_size, args.index_file, args.verbose)