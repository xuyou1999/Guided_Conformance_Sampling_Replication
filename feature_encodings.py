import time
from enum import Enum
import os.path

import numpy
from pm4py.objects.log.util import get_log_representation
from pm4py.statistics.attributes.log.get import get_all_event_attributes_from_log, get_all_trace_attributes_from_log
from lxml import etree

from scipy.sparse import csr_matrix 

class feature_types(Enum):
    ALL = 1
    EVENT_LEVEL = 2
    TRACE_LEVEL = 3
    N_GRAMS = 4

# self implementation of get_log_representation.get_representation
# separating event and trace feature names
def get_representation2(log, str_tr_attr, str_ev_attr, num_tr_attr, num_ev_attr, str_evsucc_attr=None,
                       feature_names=None):

    dictionary = {}
    count = 0
    feature_names_partition = [0]           # [0, start str_ev, start num_tr, start num_ev, start str_evsucc, total count]
    if feature_names is None:
        feature_names = []
        for trace_attribute in str_tr_attr:
            values = get_log_representation.get_all_string_trace_attribute_values(log, trace_attribute)
            for value in values:
                dictionary[value] = count
                feature_names.append(value)
                count = count + 1
        feature_names_partition.append(count)
        for event_attribute in str_ev_attr:
            values = get_log_representation.get_all_string_event_attribute_values(log, event_attribute)
            for value in values:
                dictionary[value] = count
                feature_names.append(value)
                count = count + 1
        feature_names_partition.append(count)
        for trace_attribute in num_tr_attr:
            dictionary[get_log_representation.get_numeric_trace_attribute_rep(trace_attribute)] = count
            feature_names.append(get_log_representation.get_numeric_trace_attribute_rep(trace_attribute))
            count = count + 1
        feature_names_partition.append(count)
        for event_attribute in num_ev_attr:
            dictionary[get_log_representation.get_numeric_event_attribute_rep(event_attribute)] = count
            feature_names.append(get_log_representation.get_numeric_event_attribute_rep(event_attribute))
            count = count + 1
        feature_names_partition.append(count)
        if str_evsucc_attr:
            for event_attribute in str_evsucc_attr:
                values = get_log_representation.get_all_string_event_succession_attribute_values(log, event_attribute)
                for value in values:
                    dictionary[value] = count
                    feature_names.append(value)
                    count = count + 1
        feature_names_partition.append(count)
    else:
        count = len(feature_names)
        for index, value in enumerate(feature_names):
            dictionary[value] = index


    # idea csr matrix 

    indptr = [0]
    indices = []
    data = []

    for trace in log:
        for trace_attribute in str_tr_attr:
            trace_attr_rep = get_log_representation.get_string_trace_attribute_rep(trace, trace_attribute)
            if trace_attr_rep in dictionary:
                data.append(1)
                indices.append(dictionary[trace_attr_rep])
        for event_attribute in str_ev_attr:
            values = get_log_representation.get_values_event_attribute_for_trace(trace, event_attribute)
            for value in values:
                if value in dictionary:
                    data.append(1)
                    indices.append(dictionary[value])    
        for trace_attribute in num_tr_attr:
            this_value = get_log_representation.get_numeric_trace_attribute_rep(trace_attribute)
            if this_value in dictionary:
                data.append(get_log_representation.get_numeric_trace_attribute_value(
                    trace, trace_attribute))
                indices.append(dictionary[this_value])
        for event_attribute in num_ev_attr:
            this_value = get_log_representation.get_numeric_event_attribute_rep(event_attribute)
            if this_value in dictionary:
                data.append(get_log_representation.get_numeric_event_attribute_value_trace(
                    trace, event_attribute))
                indices.append(dictionary[this_value])
        if str_evsucc_attr:
            for event_attribute in str_evsucc_attr:
                values = get_log_representation.get_values_event_attribute_succession_for_trace(trace, event_attribute)
                for value in values:
                    if value in dictionary:
                        data.append(1)
                        indices.append(dictionary[value])  
        indptr.append(len(indices))

    return csr_matrix((data, indices, indptr), dtype=int), feature_names, feature_names_partition


# create encoding from index file
def create_feature_encoding_from_index(log, considered_feature_types, index_name: str=None):

    if index_name is not None:
        root = etree.parse(index_name).getroot()
        
        # k
        k = int(root[0].text)
        # discretize
        discretize_list = [(attrib.find("attribute").text, int(attrib.find("bucket_width").text), eval(attrib.find("trace-level").text)) 
         for attrib in root.findall("discretize")]
        # event-level
        event_list = [attrib.text for attrib in root.find("features/event-level")]
        # trace-level
        trace_list = [attrib.text for attrib in root.find("features/trace-level")]
    else:
        # slower and be careful with event-level features (get_all_trace_attributes fails)
        k = 3
        discretize_list = []
        event_list = list(get_all_event_attributes_from_log(log))
        trace_list = list(get_all_trace_attributes_from_log(log))

    if considered_feature_types == feature_types.ALL:
        for discretize_tuple in discretize_list:
            discretize_equi_width(log, discretize_tuple[0], discretize_tuple[1], trace_level=discretize_tuple[2])
            
        data, feature_names = get_log_representation.get_representation(log,
                                                                        str_ev_attr=event_list,
                                                                        str_tr_attr=trace_list,
                                                                        num_ev_attr=[],
                                                                        num_tr_attr=[],
                                                                        str_evsucc_attr=[])
        data, feature_names = add_n_grams(log, data, feature_names, k)
        #print(feature_names)
        return data, feature_names        
    elif considered_feature_types == feature_types.EVENT_LEVEL:
        data, feature_names = get_log_representation.get_representation(log,
                                                                        str_ev_attr=event_list,
                                                                        str_tr_attr=[],
                                                                        num_ev_attr=[],
                                                                        num_tr_attr=[],
                                                                        str_evsucc_attr=[])
        return data, feature_names        
    elif considered_feature_types == feature_types.TRACE_LEVEL:
        data, feature_names = get_log_representation.get_representation(log,
                                                                        str_ev_attr=[],
                                                                        str_tr_attr=trace_list,
                                                                        num_ev_attr=[],
                                                                        num_tr_attr=[],
                                                                        str_evsucc_attr=[])
        return data, feature_names        
    elif considered_feature_types == feature_types.N_GRAMS:
        data, feature_names = get_log_representation.get_representation(log,
                                                                        str_ev_attr=[],
                                                                        str_tr_attr=[],
                                                                        num_ev_attr=[],
                                                                        num_tr_attr=[],
                                                                        str_evsucc_attr=[])
        data, feature_names = add_n_grams(log, data, feature_names, k)
        return data, feature_names


#TODO cache the traces and their features for constant lookup of features during sampling, duh
def create_feature_encoding(log, logname, considered_feature_types):
    if logname == "logs/BPI_Challenge_2012.xes":
        data, feature_names = create_feature_encoding_bpi12(log, considered_feature_types)
        return data, feature_names
    elif logname == "logs/Road_Traffic_Fines_Management_Process.xes":
        data, feature_names = create_feature_encoding_road_traffic_fines(log, considered_feature_types)
        return data, feature_names
    elif logname == "logs/Sepsis_Cases_-_Event_Log.xes":
        data, feature_names = create_feature_encoding_sepsis_cases(log, considered_feature_types)
        return data, feature_names
    elif logname == "logs/BPI_Challenge_2018.xes":
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
