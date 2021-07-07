import os.path
import pickle
import time
import pm4py
from pm4py.objects.log.importer.xes import importer as xes_importer
from pm4py.objects.petri_net.exporter import exporter as pnml_exporter

import LogPartitioning
from LogSampling import FeatureGuidedLogSampler, SequenceGuidedLogSampler, \
    RandomLogSampler, LongestTraceVariantLogSampler  # , #CompleteFeatureLogSampler

#inputs
# BPI_Challenge_2012.xes
# Road_Traffic_Fines_Management_Process.xes
# Sepsis_Cases_-_Event_Log.xes
# BPI_Challenge_2018.xes


def main():
    log_name = "BPI_Challenge_2018.xes"
    log, model, initial_marking, final_marking = load_inputs(log_name, modelpath="models")
    flag = "Feature"
    cached_alignments = True

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
    if flag == "Feature":
        partitioned_log, preprocessing_time = LogPartitioning.FeatureBasedPartitioning().partition(log_name, log, verbose=True)
        sampling_controller = FeatureGuidedLogSampler(preprocessing_time=preprocessing_time)
        sample = sampling_controller.construct_sample(log_name, log, model, initial_marking, final_marking, partitioned_log, 10, verbose=False)
    elif flag == "Sequence":
        sampling_controller = SequenceGuidedLogSampler(log, batch_size=1, verbose=True)
        sample = sampling_controller.construct_sample(log_name, log, model, initial_marking, final_marking, 10, verbose=True)

    elif flag == "Random":
        sampling_controller = RandomLogSampler()
        sample = sampling_controller.construct_sample(log, model, initial_marking, final_marking, 10)

    elif flag == "Longest":
        sampling_controller = LongestTraceVariantLogSampler()
        sample = sampling_controller.construct_sample(log, model, initial_marking, final_marking, 10)

    print(f"Sampling done. Total time elapsed: {(time.time()-t_start):.3f}")
    print(f"Times: {sample.times}")
    #print(sample.correlation_changes)
    #print(sample.first_positive_correlation_at)
    #print(f"Deviations: {sample.activity_deviations}")

def load_inputs(log_name, modelpath=None):
    # TODO if no model is present, discover and save in /models, otherwise load the discovered model
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
    main()
