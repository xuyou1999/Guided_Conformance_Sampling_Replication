import os.path
import statistics
import time

import pm4py

from pm4py.algo.conformance.alignments.petri_net import algorithm as alignments


class ConstantList:
    """
    Implements a list that returns the same value for each index (e.g., cost)
    """
    def __init__(self, value):
        self.value = value

    def __getitem__(self, item):
        return self.value



"""
Repeats alignments calculates alignments for a complete log n times, reporting runtimes
"""
def calculate_alignments_n_times(log, net, im, fm, n):
    time_list = []
    for i in range (n):
        print("Repetition "+str(i))
        t_start = time.time()
        alignments = calculate_alignments(log, net, im, fm)
        t_total = time.time() - t_start
        print(">"+str(t_total))
        time_list.append(t_total)
    print(time_list)
    print(f"Mean: {statistics.mean(time_list)}, Stddev: {statistics.stdev(time_list)}")


"""
Calculates alignments for a complete log
"""
def calculate_alignments(log, net, im, fm):
        model_cost_function = dict()
        sync_cost_function = dict()
        for t in net.transitions:
            if t.label is not None:
                model_cost_function[t] = 1
                sync_cost_function[t] = 0
            else:
                model_cost_function[t] = 0

        # Will always return 1, for every index
        trace_cost_function = ConstantList(1)

        alignment_params = {}
        alignment_params[alignments.Parameters.PARAM_MODEL_COST_FUNCTION] = model_cost_function
        alignment_params[alignments.Parameters.PARAM_SYNC_COST_FUNCTION] = sync_cost_function
        alignment_params[alignments.Parameters.PARAM_TRACE_COST_FUNCTION] = trace_cost_function
        aligned_traces = alignments.apply(log, net, im, fm, parameters=alignment_params)


if __name__ == '__main__':
    logs = ["Sepsis_Cases_-_Event_Log.xes", "BPI_Challenge_2012.xes", "Road_Traffic_Fines_Management_Process.xes"]
    #logs = ["BPI_Challenge_2012.xes"]
    for log_name in logs:
        print(log_name)
        log = pm4py.read_xes(os.path.join("logs", log_name))
        net, im, fm = pm4py.discover_petri_net_inductive(log, noise_threshold=0.2)
        calculate_alignments_n_times(log, net, im, fm, 5)
