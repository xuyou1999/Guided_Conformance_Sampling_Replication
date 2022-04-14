# Relevance-guided Sampling for Conformance Checking

This repository contains scripts, as well as used evaluation data for the guided sampling procedure for conformance checking scenarios, as described in "Sampling What Matters: Relevance-guided Sampling of Event Logs" (under review). The approach learns correlations between trace properties and sampling goals (currently conformance checking) to steer the sampling to be more representative.

## Usage ##

### Sampling a log ###
The syntax for executing a guided sampling-procedure of an event log is

```
python3 Logsampling.py [--verbose] [-index_file INDEX_FILE] [log_file] [model_file] [algorithm] [sample size]
```
The parameters are
* log_file - the provided log file, in .xes format
* model_file - the model file used during sampling, in .pnml format
* (Optional) index_file - the index file containing information on considered features for the indexing phase
* algorithm - the algorithm to use for the guided sampling procedure. Can be either "feature" or "behavioural"
* sample size - the size of the final sample to be returned
* (Optional) verbose - if set, the output is more detailed


For instance, executing

```
python3 Logsampling.py --verbose -index_file example_index.xml example-log.xes example-model.pnml feature 200
```
generates a sample of '200' traces from the log file 'exampe-log.xes', the model file 'example-model.pnml', using the log features specified in 'example-log.index' using a 'feature'-based knowledge-based.



### Index Files ###

To specify, what log properties should be taken into account for the relevance-guided log sampling procedure, you can specify an index file in .xml-format, that describes the trace-level attributes, event-level attributes, k-gram lengths and potential preceeding discreditization steps, that comprise the built feature index.

The structure of the .xml-file is as follows
```xml
<index>
  <k>3</k>                                        #use activity k-grams of length 3
  <discretize>
    <attribute>attribute a</attribute>            #discretize trace-level attribute 'a' using a bucket size of 10
    <trace-level>true</trace-level>
    <bucket_width>10</bucket_width>
  </discretize>
    <discretize>
    <attribute>attribute b</attribute>             #discretize event-level attribute 'b' using a bucket size of 5
    <trace-level>false</trace-level>
    <bucket_width>5</bucket_width>
  </discretize>
  <discretize>                                    #additional attributes that need discretization
    ...
  </discretize>
  
  <features>
    <trace-level>
      <attribute>attribute a</attribute>          #incorporate trace-level attribue 'a'
      ...                                         #additional trace-level attributes to incorporate
    </trace-level>
    <event-level>
      <attribute>attribute b</attribute>          #incorporate event-level attribue 'b'
      ...                                         #additional event-level attributes to incorporate
    </event-level>
  </features>
</index>
```
If no index file is provided, the approach considers all features, as well as 3-grams, and no discretization step is conducted.
If you want to incorporate the procedure in your project, please have a look at the function ```construct_sample``` in LogSampling.py, that serves as the top-level entrypoint of the approach.

## Benchmarking and Evaluation ##
Additionally, we provide the benchmarking and evaluation script ```eval.py```. The script repeatedly constructs samples for the provided log files using different sampling methods and writes the results into .csv-files. The explicit .csv-filed used during the evaluation in the publication are located under 'results'
