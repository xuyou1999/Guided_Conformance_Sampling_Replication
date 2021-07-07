import os.path
import time
from pm4py.objects.log.importer.xes import importer as xes_importer
from pm4py.objects.petri_net.exporter import exporter as pnml_exporter

from CompleteLogAnalysis.CompleteLogKnowledgeBaseCalculation import CompleteFeatureLogAnalyzer, CompleteSequenceLogAnalyzer
from LogSampling import ConstantList
import LogPartitioning
import pickle

# import pm4py
# from pm4py.algo.conformance.alignments.petri_net import algorithm as alignments

"""
Performs analysis on complete logs instead of samples using multi-threaded computations.
"""


# inputs
# BPI_Challenge_2012.xes
# Road_Traffic_Fines_Management_Process.xes
# Sepsis_Cases_-_Event_Log.xes

def main(log_name, log, aligned_traces):
    partitioned_log, t = LogPartitioning.FeatureBasedPartitioning().partition(log_name, log)
    feature_analyzer = CompleteFeatureLogAnalyzer()
    feature_knowledge_base = feature_analyzer.analyze_log(log_name, log, aligned_traces, partitioned_log, verbose=True)

    pickle.dump(feature_knowledge_base, open(log_name + "_feature.knowledge_base", "wb"))
    print()

    sequence_analyzer = CompleteSequenceLogAnalyzer(log, verbose=True)
    sequence_knowledge_base = sequence_analyzer.analyze_log(log_name, aligned_traces, verbose=True)

    pickle.dump(sequence_knowledge_base, open(log_name + "_sequence.knowledge_base", "wb"))
    print()

    # combined_knowledge_base = dict(feature_knowledge_base)
    # combined_knowledge_base.update(sequence_knowledge_base)
    # for feature in combined_knowledge_base.keys():
    #     if combined_knowledge_base[feature].correlation > 0.0:
    #         print("      " + str(feature) + " : " + str(combined_knowledge_base[feature]))
    # pickle.dump(combined_knowledge_base, open(log_name + "_combined.knowledge_base", "wb"))
    # print()


def load_inputs(log_name, modelpath=None):
    log = log = xes_importer.apply(os.path.join("../logs", log_name))
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


if __name__ == '__main__':
    use_cache = True
    t_start = time.time()
    import pm4py

    log_names = ["BPI_Challenge_2018.xes"]
    #log_names = ["Sepsis_Cases_-_Event_Log.xes", "BPI_Challenge_2012.xes", "BPI_Challenge_2018.xes"]
    for log_name in log_names:
        print(log_name)
        log = pm4py.read_xes(os.path.join("logs", log_name))
        model, im, fm = pm4py.discover_petri_net_inductive(log, noise_threshold=0.2)
        # net, im, fm = pm4py.discover_petri_net_alpha(log)

        from pm4py.algo.conformance.alignments.petri_net import algorithm as alignments

        alignment_params = {}
        model_cost_function = dict()
        sync_cost_function = dict()
        for t in model.transitions:
            if t.label is not None:
                model_cost_function[t] = 1
                sync_cost_function[t] = 0
            else:
                model_cost_function[t] = 0

        # Will always return 1, for every index
        trace_cost_function = ConstantList(1)

        alignment_params = {alignments.Parameters.PARAM_MODEL_COST_FUNCTION: model_cost_function,
                            alignments.Parameters.PARAM_SYNC_COST_FUNCTION: sync_cost_function,
                            alignments.Parameters.PARAM_TRACE_COST_FUNCTION: trace_cost_function,
                            alignments.Parameters.CORES: 4}

        if use_cache:
            aligned_traces = pickle.load(open(os.path.join("alignment_cache", log_name + ".align"), "rb"))
            main(log_name, log, aligned_traces)
            print(f"Sampling done. Total time elapsed: {(time.time() - t_start):.3f}")

        else:
            aligned_traces = alignments.apply_multiprocessing(log[:], model, im, fm,
                                                              parameters=alignment_params)
                                                                # parameters={alignments.Parameters.CORES: 4})
            main(log_name, log, aligned_traces)
            print(f"Sampling done. Total time elapsed: {(time.time() - t_start):.3f}")
        print()