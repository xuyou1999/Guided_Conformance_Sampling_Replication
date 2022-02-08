import os
import pickle
import pm4py
import time
from pm4py.objects.log import obj as log_implementation
from pm4py.algo.conformance.alignments.petri_net import algorithm as alignments
from collections import defaultdict
from tqdm import tqdm


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


if __name__ == '__main__':
    logs = ["Sepsis_Cases_-_Event_Log.xes", "BPI_Challenge_2012.xes", "Road_Traffic_Fines_Management_Process.xes", "BPI_Challenge_2018.xes"]

    # Prepare output file
    complete_log_results = open(os.path.join("..", "results", "ground_truth", "complete_log_stats" + ".csv"), "w")
    complete_log_results.write("log_name;n_traces;n_trace_variants;deviating_traces;total_deviations;"
                               "num_deviating_activities;fitness\n")

    for log_name in logs:
        print(log_name)
        log = pm4py.read_xes(os.path.join("..", "logs", log_name))
        net, im, fm = pm4py.discover_petri_net_inductive(log, noise_threshold=0.2)

        # Load precomputed alignments
        trace_alignments = pickle.load(open(os.path.join("..", "alignment_cache", log_name + ".align"), "rb"))

        # Calculate stats
        n_deviating_traces = 0
        n_total_deviations = 0
        activity_deviations = defaultdict(int)

        # Fitness computation init
        alignment_params = init_alignment_params(net)
        shortest_path = alignments.apply_trace(log_implementation.Trace(), net, im,
                                               fm, parameters=alignment_params)["cost"]
        total_costs = 0.0
        upper_bound_total_costs = 0.0

        pbar = tqdm(zip(log, trace_alignments), total=len(log), desc="Computing alignment stats")
        for trace, alignment in pbar:
            if not alignment:
                pbar.update()
                continue

            #time.sleep(0.001)
            # Update fitness computation
            total_costs += alignment["cost"]
            upper_bound_total_costs += int(shortest_path) + len(trace)

            # Update deviation stats
            deviation_found = False
            for LMstep in alignment["alignment"]:
                # Sync move
                if LMstep[0] == LMstep[1]:
                    pass

                # Move on log only, deviation
                elif LMstep[1] == '>>':
                    n_total_deviations += 1
                    if not deviation_found:
                        n_deviating_traces += 1
                        deviation_found = True
                    activity_deviations[LMstep[0]] += 1

                # Move on model only
                elif LMstep[0] == '>>':
                    # Hidden transition, no deviation
                    if LMstep[1] == None:
                        pass
                    # No hidden transition, deviation
                    else:
                        n_total_deviations += 1
                        if not deviation_found:
                            n_deviating_traces += 1
                            deviation_found = True
                        activity_deviations[LMstep[1]] += 1
            pbar.update()

        # Total log fitness
        fitness = 1 - (total_costs / upper_bound_total_costs)

        # Number of trace variants
        variants_idxs, one_tr_per_var = alignments.__get_variants_structure(log, alignment_params)
        n_trace_variants = len(one_tr_per_var)

        # Write results
        complete_log_results.write(";".join((str(log_name),
                                             str(len(log)),
                                             str(n_trace_variants),
                                             str(n_deviating_traces),
                                             str(n_total_deviations),
                                             str(len(activity_deviations.keys())),
                                             str(str(fitness))
                             ))  + "\n")

        # Write deviating activity distribution results
        deviating_activities_results = open(os.path.join("..", "results", "ground_truth", "activity_distribution_" + log_name + ".csv"), "w")
        deviating_activities_results.write("deviating_activity;count\n")
        for activity in activity_deviations.keys():
            deviating_activities_results.write(str(activity) + ";" + str(activity_deviations[activity]) + "\n")


