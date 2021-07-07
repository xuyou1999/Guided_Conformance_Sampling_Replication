import time
from enum import Enum

import numpy
from pm4py.objects.log.util import get_log_representation


class feature_types(Enum):
    ALL = 1
    EVENT_LEVEL = 2
    TRACE_LEVEL = 3
    N_GRAMS = 4

#TODO cache the traces and their features for constant lookup of features during sampling, duh
def create_feature_encoding(log, logname, considered_feature_types):
    if logname == "BPI_Challenge_2012.xes":
        data, feature_names = create_feature_encoding_bpi12(log, considered_feature_types)
        return data, feature_names
    elif logname == "Road_Traffic_Fines_Management_Process.xes":
        data, feature_names = create_feature_encoding_road_traffic_fines(log, considered_feature_types)
        return data, feature_names
    elif logname == "Sepsis_Cases_-_Event_Log.xes":
        data, feature_names = create_feature_encoding_sepsis_cases(log, considered_feature_types)
        return data, feature_names
    elif logname == "BPI_Challenge_2018.xes":
        data, feature_names = create_feature_encoding_bpi18(log, considered_feature_types)
        return data, feature_names
    return None, None


def create_feature_encoding_bpi12(log, feature_types_to_include):
    if feature_types_to_include == feature_types.ALL:
        discretize_equi_width(log, "AMOUNT_REQ", 100, trace_level=True)
        data, feature_names = get_log_representation.get_representation(log,
                                                                        str_ev_attr=["org:resource", "concept:name"],
                                                                        str_tr_attr=["AMOUNT_REQ"],
                                                                        num_ev_attr=[],
                                                                        num_tr_attr=[],
                                                                        # TODO N GRAMS
                                                                        str_evsucc_attr=[])
        data, feature_names = add_n_grams(log, data, feature_names, 3)
        return data, feature_names
    elif feature_types_to_include == feature_types.EVENT_LEVEL:
        data, feature_names = get_log_representation.get_representation(log,
                                                                        str_ev_attr=[
                                                                            "org:resource",
                                                                            "concept:name"],
                                                                        str_tr_attr=[],
                                                                        num_ev_attr=[],
                                                                        num_tr_attr=[],
                                                                        str_evsucc_attr=[])
        return data, feature_names
    elif feature_types_to_include == feature_types.TRACE_LEVEL:
        data, feature_names = get_log_representation.get_representation(log,
                                                                        str_ev_attr=[],
                                                                        str_tr_attr=[
                                                                            "AMOUNT_REQ"],
                                                                        num_ev_attr=[],
                                                                        num_tr_attr=[],
                                                                        str_evsucc_attr=[])
        return data, feature_names
    elif feature_types_to_include == feature_types.N_GRAMS:
        data, feature_names = get_log_representation.get_representation(log,
                                                                        str_ev_attr=[],
                                                                        str_tr_attr=[],
                                                                        num_ev_attr=[],
                                                                        num_tr_attr=[],
                                                                        str_evsucc_attr=[])
        data, feature_names = add_n_grams(log, data, feature_names, 3)
        return data, feature_names


def create_feature_encoding_road_traffic_fines(log, feature_types_to_include):
    if feature_types_to_include == feature_types.ALL:
        discretize_equi_width(log, "amount", 5, trace_level=False)
        discretize_equi_width(log, "expense", 5, trace_level=False)

        data, feature_names = get_log_representation.get_representation(log,
                                                                        str_ev_attr=["org:resource",
                                                                                     "concept:name",
                                                                                     "amount",
                                                                                     "dismissal",
                                                                                     "vehicleClass",
                                                                                     "article",
                                                                                     "points",
                                                                                     "expense"],
                                                                        str_tr_attr=[],
                                                                        num_ev_attr=[],
                                                                        num_tr_attr=[],
                                                                        str_evsucc_attr=[])
        data, feature_names = add_n_grams(log, data, feature_names, 3)
        return data, feature_names
    elif feature_types_to_include == feature_types.EVENT_LEVEL:
        data, feature_names = get_log_representation.get_representation(log,
                                                                        str_ev_attr=["org:resource",
                                                                                     "concept:name",
                                                                                     "amount", "dismissal",
                                                                                     "vehicleClass",
                                                                                     "article", "points",
                                                                                     "expense"],
                                                                        str_tr_attr=[],
                                                                        num_ev_attr=[],
                                                                        num_tr_attr=[],
                                                                        str_evsucc_attr=[])
        return data, feature_names
    elif feature_types_to_include == feature_types.TRACE_LEVEL:
        data, feature_names = get_log_representation.get_representation(log,
                                                                        str_ev_attr=[],
                                                                        str_tr_attr=[],
                                                                        num_ev_attr=[],
                                                                        num_tr_attr=[],
                                                                        str_evsucc_attr=[])
        return data, feature_names
    elif feature_types_to_include == feature_types.N_GRAMS:
        data, feature_names = get_log_representation.get_representation(log,
                                                                        str_ev_attr=[],
                                                                        str_tr_attr=[],
                                                                        num_ev_attr=[],
                                                                        num_tr_attr=[],
                                                                        str_evsucc_attr=[])
        data, feature_names = add_n_grams(log, data, feature_names, 3)
        return data, feature_names


def create_feature_encoding_sepsis_cases(log, feature_types_to_include):
    if feature_types_to_include == feature_types.ALL:
        discretize_equi_width(log, "Age", 5, trace_level=False)
        discretize_equi_width(log, "Leucocytes", 1, trace_level=False)
        discretize_equi_width(log, "CRP", 1, trace_level=False)
        discretize_equi_width(log, "LacticAcid", 1, trace_level=False)

        data, feature_names = get_log_representation.get_representation(log,
                                                                        str_ev_attr=[
                                                                            "concept:name",
                                                                            "Age", "Diagnose", "Leucocytes", "CRP",
                                                                            "LacticAcid", "org:group",
                                                                            "DiagnosticArtAstrup",
                                                                            "DiagnosticBlood", "DiagnosticECG",
                                                                            "DiagnosticIC", "DiagnosticLacticAcid",
                                                                            "DiagnosticLiquor", "DiagnosticOther",
                                                                            "DiagnosticSputum",
                                                                            "DiagnosticUrinaryCulture",
                                                                            "DiagnosticUrinarySediment",
                                                                            "DiagnosticXthorax", "DisfuncOrg",
                                                                            "Hypotensie",
                                                                            "Hypoxie", "InfectionSuspected", "Infusion",
                                                                            "Oligurie", "SIRSCritHeartRate",
                                                                            "SIRSCritLeucos", "SIRSCritTachypnea",
                                                                            "SIRSCritTemperature",
                                                                            "SIRSCriteria2OrMore"],
                                                                        str_tr_attr=[],
                                                                        num_ev_attr=[],
                                                                        num_tr_attr=[],
                                                                        str_evsucc_attr=[])
        data, feature_names = add_n_grams(log, data, feature_names, 3)
        return data, feature_names

    elif feature_types_to_include == feature_types.EVENT_LEVEL:
        data, feature_names = get_log_representation.get_representation(log,
                                                                        str_ev_attr=[
                                                                            "concept:name",
                                                                            "Age", "Diagnose", "Leucocytes", "CRP",
                                                                            "LacticAcid", "org:group",
                                                                            "DiagnosticArtAstrup",
                                                                            "DiagnosticBlood", "DiagnosticECG",
                                                                            "DiagnosticIC", "DiagnosticLacticAcid",
                                                                            "DiagnosticLiquor", "DiagnosticOther",
                                                                            "DiagnosticSputum",
                                                                            "DiagnosticUrinaryCulture",
                                                                            "DiagnosticUrinarySediment",
                                                                            "DiagnosticXthorax", "DisfuncOrg",
                                                                            "Hypotensie",
                                                                            "Hypoxie", "InfectionSuspected",
                                                                            "Infusion",
                                                                            "Oligurie", "SIRSCritHeartRate",
                                                                            "SIRSCritLeucos", "SIRSCritTachypnea",
                                                                            "SIRSCritTemperature",
                                                                            "SIRSCriteria2OrMore"],
                                                                        str_tr_attr=[],
                                                                        num_ev_attr=[],
                                                                        num_tr_attr=[],
                                                                        str_evsucc_attr=[])
        return data, feature_names
    elif feature_types_to_include == feature_types.TRACE_LEVEL:
        data, feature_names = get_log_representation.get_representation(log,
                                                                        str_ev_attr=[],
                                                                        str_tr_attr=[],
                                                                        num_ev_attr=[],
                                                                        num_tr_attr=[],
                                                                        str_evsucc_attr=[])
        return data, feature_names
    elif feature_types_to_include == feature_types.N_GRAMS:
        data, feature_names = get_log_representation.get_representation(log,
                                                                        str_ev_attr=[],
                                                                        str_tr_attr=[],
                                                                        num_ev_attr=[],
                                                                        num_tr_attr=[],
                                                                        str_evsucc_attr=[])
        data, feature_names = add_n_grams(log, data, feature_names, 3)
        return data, feature_names


def create_feature_encoding_bpi18(log, feature_types_to_include):
    #t = time.time()
    if feature_types_to_include == feature_types.ALL:
        discretize_equi_width(log, "amount_applied0", 1000, trace_level=True)
        discretize_equi_width(log, "area", 1.0, trace_level=True)
        discretize_equi_width(log, "cross_compliance", 1.0, trace_level=True)
        discretize_equi_width(log, "payment_actual0", 1000, trace_level=True)
        discretize_equi_width(log, "penalty_amount0", 100, trace_level=True)
        discretize_equi_width(log, "risk_factor", 1.0, trace_level=True)
        discretize_equi_width(log, "number_parcels", 5, trace_level=True)

        data, feature_names = get_log_representation.get_representation(log,
                                                                        str_ev_attr=["org:resource", "concept:name",
                                                                                     "success", "doctype",
                                                                                     "subprocess"],
                                                                        str_tr_attr=["basic payment", "greening",
                                                                                     "penalty_ABP", "penalty_AGP",
                                                                                     "penalty_AJLP", "penalty_AUVP",
                                                                                     "penalty_AVBP", "penalty_AVGP",
                                                                                     "penalty_AVJLP", "penalty_AVUVP",
                                                                                     "penalty_B16", "penalty_B2",
                                                                                     "penalty_B3", "penalty_B4",
                                                                                     "penalty_B5", "penalty_B5f",
                                                                                     "penalty_B6", "penalty_BGK",
                                                                                     "penalty_BGKV", "penalty_BGP",
                                                                                     "penalty_C16", "penalty_C4",
                                                                                     "penalty_C9", "penalty_CC",
                                                                                     "penalty_GP1", "penalty_JLP1",
                                                                                     "penalty_JLP2", "penalty_JLP3",
                                                                                     "penalty_JLP5", "penalty_JLP6",
                                                                                     "penalty_JLP7", "penalty_V5",
                                                                                     "redistribution", "rejected",
                                                                                     "selected_manually",
                                                                                     "selected_random", "selected_risk",
                                                                                     "small farmer", "young farmer",
                                                                                     "amount_applied0", "area",
                                                                                     "cross_compliance",
                                                                                     "payment_actual0",
                                                                                     "penalty_amount0",
                                                                                     "risk_factor", "number_parcels",
                                                                                     "department", "year"],
                                                                        num_ev_attr=[],
                                                                        num_tr_attr=[],
                                                                        str_evsucc_attr=[])
        data, feature_names = add_n_grams(log, data, feature_names, 3)
        #print(f"Preprocessing time: {time.time()-t}")
        return data, feature_names
    elif feature_types_to_include == feature_types.EVENT_LEVEL:
        data, feature_names = get_log_representation.get_representation(log,
                                                                        str_ev_attr=["org:resource", "concept:name",
                                                                                     "success", "doctype",
                                                                                     "subprocess"],
                                                                        str_tr_attr=[],
                                                                        num_ev_attr=[],
                                                                        num_tr_attr=[],
                                                                        str_evsucc_attr=[])
        return data, feature_names
    if feature_types_to_include == feature_types.TRACE_LEVEL:
        data, feature_names = get_log_representation.get_representation(log,
                                                                        str_ev_attr=[],
                                                                        str_tr_attr=["basic payment", "greening",
                                                                                     "penalty_ABP", "penalty_AGP",
                                                                                     "penalty_AJLP", "penalty_AUVP",
                                                                                     "penalty_AVBP", "penalty_AVGP",
                                                                                     "penalty_AVJLP",
                                                                                     "penalty_AVUVP",
                                                                                     "penalty_B16", "penalty_B2",
                                                                                     "penalty_B3", "penalty_B4",
                                                                                     "penalty_B5", "penalty_B5f",
                                                                                     "penalty_B6", "penalty_BGK",
                                                                                     "penalty_BGKV", "penalty_BGP",
                                                                                     "penalty_C16", "penalty_C4",
                                                                                     "penalty_C9", "penalty_CC",
                                                                                     "penalty_GP1", "penalty_JLP1",
                                                                                     "penalty_JLP2", "penalty_JLP3",
                                                                                     "penalty_JLP5", "penalty_JLP6",
                                                                                     "penalty_JLP7", "penalty_V5",
                                                                                     "redistribution", "rejected",
                                                                                     "selected_manually",
                                                                                     "selected_random",
                                                                                     "selected_risk",
                                                                                     "small farmer", "young farmer",
                                                                                     "amount_applied0", "area",
                                                                                     "cross_compliance",
                                                                                     "payment_actual0",
                                                                                     "penalty_amount0",
                                                                                     "risk_factor",
                                                                                     "number_parcels",
                                                                                     "department", "year"],
                                                                        num_ev_attr=[],
                                                                        num_tr_attr=[],
                                                                        str_evsucc_attr=[])
        return data, feature_names
    elif feature_types_to_include == feature_types.N_GRAMS:
        data, feature_names = get_log_representation.get_representation(log,
                                                                        str_ev_attr=[],
                                                                        str_tr_attr=[],
                                                                        num_ev_attr=[],
                                                                        num_tr_attr=[],
                                                                        str_evsucc_attr=[])
        data, feature_names = add_n_grams(log, data, feature_names, 3)
        return data, feature_names


def discretize_equi_width(log, attribute_name, width, trace_level=True):
    for trace in log:
        if trace_level:
            if attribute_name not in trace.attributes:
                print("Could not find attribute [" + attribute_name + "] in trace_level attributes")
                return
            trace.attributes[attribute_name] = int(trace.attributes[attribute_name]) - int(
                trace.attributes[attribute_name]) % width
        else:
            for idx, event in enumerate(trace):  #
                if attribute_name not in event:
                    continue
                event[attribute_name] = int(trace[idx][attribute_name]) - int(
                    trace[idx][attribute_name]) % width


def add_n_grams(log, data, feature_names, n):
    # get n_gram features
    n_grams = {}
    for trace in log:
        events = [*map(lambda x: x['concept:name'], trace)]
        last_index = max(0, len(events) - n + 1)
        # print(events)
        if len(events) < n:
            n_gram = events
            n_gram_string = "#".join(n_gram)
            n_grams[n_gram_string] = None
        for i in range(0, last_index):
            n_gram = events[i:i + n]
            n_gram_string = "#".join(n_gram)
            n_grams[n_gram_string] = None

    if feature_names is None:
        feature_names = n_grams.keys()
    else:
        feature_names.extend(n_grams.keys())

    # for each trace check if it contains property
    log_data = []
    for idx, trace in enumerate(log):
        trace_data = []
        events = [*map(lambda x: x['concept:name'], trace)]
        events_string = "#".join(events)
        # print(events_string)
        for n_gram in n_grams.keys():
            if n_gram in events_string:
                trace_data.append(1)
            else:
                trace_data.append(0)
        log_data.append(trace_data)
    log_data = numpy.array(log_data)

    if data is None:
        return log_data, feature_names
    else:
        data = numpy.concatenate((data, log_data), axis=1)
        return data, feature_names
