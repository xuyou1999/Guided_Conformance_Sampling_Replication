# Feedback-guided Sampling for Conformance Checking

This repository contains scripts, as well as used evaluation data for the guided sampling procedure for conformance checking scenarios, as described in "Sampling What Matters: Relevence-guided Sampling of Event Logs" (under review). The approach learns correlations between trace properties and sampling goals (currently conformance checking) to steer the sampling to be more representative.

## Usage ##
The syntax for starting the sampling procedure from the command line interface is

```
python3 Logsampling.py [log file] [algorithm] [sample size] [--index file] [--verbose]
```
The parameters are
* log file - the provided log file, in .xes format
* algorithm - the algorithm to use. Can be either "feature" or "behavioural"
* sample size - the size of the final sample to be returned
* (Optional) index file - the index file containing information on considered features for the indexing phase
* (Optional) verbose - if set, the output is more detailed


For instance, executing

```
python3 Logsampling.py example.xes feature 200 example_index.xml --verbose
```
creates a conformance sample of size 200 using the feature-based guided-sampling approach on the log file named 'example.xes'. Furthermore, features are created akin to the information provided in the file example_index.xml .

The index file allows the specification of the used k-gram length, the considered features of the log, as well as what features need to be discretitized before sampling. Its syntactical structure is:
```xml
<index>
  <k>2</k>
  <discretize>
    <feature>feature_to_discretize</feature>
    <bucket_width>10</bucket_width>
  </discretize>
  <discretize>
    ...
  </discretize>
  
  <trace_level>feature 1</trace_level>
  <trace_level>feature 2</trace_level>
  
  ...
  <event_level>feature 1</event_level>
  <event_level>feature 2</event_level>
  ...
</index>
```
If no index file is provided, all features and 3-grams are considered, and no discretization step is conducted.

If you want to use the provided algorithms directly in your code, please have look at the function ```construct_sample``` in LogSampling.py, where the differnet algorithms are instantiated.

## Benchmarking and Evaluation ##
Additionally we provide the benchmarking and evaluation script ```eval.py```. The script repeatedly constructs samples for the provided log files using the different algorithms and writes the results into .csv-files. The explicit .csv-filed used during the evaluation in the publication are located under src/results
