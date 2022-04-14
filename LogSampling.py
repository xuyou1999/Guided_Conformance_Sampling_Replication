import argparse
import os.path
import pickle
import time
import pm4py
from pm4py.objects.log.importer.xes import importer as xes_importer

import LogIndexing
from SamplingAlgorithms import FeatureGuidedLogSampler, SequenceGuidedLogSampler, \
    RandomLogSampler, LongestTraceVariantLogSampler


def construct_sample(log_name, model_name, algorithm, sample_size, index_file, alignment_file, verbose_output):
    log, model, initial_marking, final_marking = load_inputs(log_name, model_name)

    #TODO turn this into optional parameter cached_alignments, that, if set, uses the specified file to construct the alignment index
    #TODO do we even want this in the final published code?

    use_cache = False

    if alignment_file is not None:
        print("Quality Evaluation: Loading precomputed alignments...")
        aligned_traces = pickle.load(open(alignment_file, "rb"))
        trace_keys = []
        for trace in log:
            event_representation = ""
            for event in trace:
                event_representation = event_representation + " >> " + event["concept:name"]
            trace_keys.append(event_representation)

        assert (len(trace_keys) == len(log))
        use_cache = True
        alignment_cache = dict(zip(trace_keys, aligned_traces))
    else:
        alignment_cache = {}

    t_start = time.time()

    sample = None
    if algorithm == "feature":
        partitioned_log, preprocessing_time = LogIndexing.FeatureBasedPartitioning().partition(log_name, log,
                                                                                               index_file=index_file,
                                                                                               verbose=verbose_output)
        sampling_controller = FeatureGuidedLogSampler(use_cache=use_cache, alignment_cache=alignment_cache, preprocessing_time=preprocessing_time, index_file=index_file, verbose=verbose_output)
        sample = sampling_controller.construct_sample(log_name, log, model, initial_marking, final_marking,
                                                      partitioned_log, int(sample_size))
    elif algorithm == "behavioural":
        sampling_controller = SequenceGuidedLogSampler(log, batch_size=1, index_file=index_file, use_cache=use_cache, alignment_cache=alignment_cache, verbose=verbose_output)
        sample = sampling_controller.construct_sample(log_name, log, model, initial_marking, final_marking,
                                                      int(sample_size))

    # only used for debugging and evaluation purposes
    elif algorithm == "Random":
        sampling_controller = RandomLogSampler(use_cache=use_cache, alignment_cache=alignment_cache,
                                               verbose=verbose_output)
        sample = sampling_controller.construct_sample(log, model, initial_marking, final_marking, int(sample_size))

    elif algorithm == "Longest":
        sampling_controller = LongestTraceVariantLogSampler(use_cache=use_cache, alignment_cache=alignment_cache,
                                                            verbose=verbose_output)
        sample = sampling_controller.construct_sample(log, model, initial_marking, final_marking, int(sample_size))
    else:
        print(f"Algorithm {algorithm} is not supported.")

    if sample is not None:
        print(f"Sampling done. Total time elapsed: {(time.time() - t_start):.3f}")
        print(f"Times: {sample.times}")

    return sample


def load_inputs(log_name, model_name):
    log = xes_importer.apply(str(log_name))
    model = None
    print("Loading Model")
    model, initial_marking, final_marking = pm4py.read_pnml(os.path.join(str(model_name)))
    print("Done")
    return log, model, initial_marking, final_marking


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("log_file", help="the name of the xes-log from which to sample")
    parser.add_argument("model_file", help="the name of the .pnml-file used for conformance checking")
    parser.add_argument("algorithm", help="the sampling strategy to use [feature, behavioural]")
    parser.add_argument("sample_size", help="the size of the final sample")
    parser.add_argument("-index_file", help="the name of the index file containing the features considered during "
                                           "indexing. If none is supplied, all features are considered")
    parser.add_argument("-alignment_file", help="the name of the cache file containing the alignments precomputed "
                                            "on the given event log and model."
                                            "If none is supplied, the alignments are computed from scratch.")
    parser.add_argument("--verbose", help="increase verbosity of output", action="store_true")
    args = parser.parse_args()

    construct_sample(args.log_file, args.model_file, args.algorithm, args.sample_size, args.index_file, args.alignment_file, args.verbose)
