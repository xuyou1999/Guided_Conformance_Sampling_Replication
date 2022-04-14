import os.path
import pm4py
from pm4py.objects.log.importer.xes import importer as xes_importer
from pm4py.objects.petri_net.exporter import exporter as pnml_exporter

log_names = ["Sepsis_Cases_-_Event_Log.xes", "BPI_Challenge_2012.xes", "BPI_Challenge_2018.xes", "Road_Traffic_Fines_Management_Process.xes"]

def main():
    for log_name in log_names:
        load_inputs(log_name, modelpath="models")

def load_inputs(log_name, modelpath=None):
    log = xes_importer.apply(os.path.join("logs", log_name))
    if modelpath is None or not os.path.exists(os.path.join(str(modelpath), str(log_name) + ".pnml")):
        print("Model Discovery for log " + log_name)
        model, initial_marking, final_marking = pm4py.discover_petri_net_inductive(log, noise_threshold=0.2)
        pnml_exporter.apply(model, initial_marking, os.path.join(str(modelpath), str(log_name) + ".pnml"),
                            final_marking=final_marking)
    print("Done")

if __name__ == '__main__':
    # TODO: add log names as arguments?
    main()