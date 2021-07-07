class ConstantList:
    """
    Implements a list that returns the same value for each index (e.g., cost)
    """

    def __init__(self, value):
        self.value = value

    def __getitem__(self, item):
        return self.value

if __name__ == '__main__':
    import os.path
    import sys
    import pickle
    import pm4py
    import pm4pycvxopt

    #lower = int(sys.argv[1])
    #upper = int(sys.argv[2])
    #print(f"Lower: {lower} Upper: {upper}")

    logs = ["BPI_Challenge_2012.xes"]#, "Road_Traffic_Fines_Management_Process.xes", "Sepsis_Cases_-_Event_Log.xes"]
    logs = ["Road_Traffic_Fines_Management_Process.xes"]
    for log_name in logs:
        log = pm4py.read_xes(os.path.join("../logs", log_name))
        net, im, fm = pm4py.discover_petri_net_inductive(log, noise_threshold=0.2)

        from pm4py.algo.conformance.alignments.petri_net import algorithm as alignments

        alignment_params = {}
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
        alignment_params[alignments.Parameters.CORES] = 4

        # Only compute alignments for the ~95% shortest traces
        short_trace_indices = []
        for idx, trace in enumerate(log):
            if len(trace) <= 100:
                short_trace_indices.append(idx)

        aligned_traces = alignments.apply_multiprocessing([log[x] for x in short_trace_indices[:]], net, im, fm, parameters=alignment_params)

        pickle.dump(aligned_traces, open(log_name + ".align", "wb"))
        pickle.dump(short_trace_indices, open(log_name + ".indices", "wb"))