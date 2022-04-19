# Relevance-guided Sampling for Conformance Checking

This repository contains an implementation, as well as the used experimental result data for the article ["Sampling What Matters: Relevance-guided Sampling of Event Logs"](https://ieeexplore.ieee.org/document/9576875).

The proposed sampling procedures guide the selection of traces to be included in a sample by learning which features of a trace correlate with trace properties of interest (so far we used deviating traces as interesting).

## How to use ##


### Using the command line interface ###
To generate a sample of the log from the command line, run:

```
python3 Logsampling.py <-index_file INDEX_FILE> <feature|behavioural> <sample size> <log_file> <model_file>
```
The parameters are
* ``algorithm`` - the indexing type to use. Can be either "feature" or "behavioural"
* ``sample size`` - the size of the final sample to be generated
* ``log_file`` - the provided log file, in .xes format
* ``model_file`` - the model file used during sampling, in .pnml format
* ``(Optional) index_file`` - the index file containing information on considered features for the indexing phase

### Invoking the classes from code ###
To directly invoke the provided classes in your own project, add the following lines to your project:

#### Feature-based ####


#### Behavioural-based ####

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

## Evaluation and result files ##
Additionally, we provide the benchmarking and evaluation script ```eval.py```. The script repeatedly constructs samples for the provided log files using different sampling methods and writes the results into .csv-files. The explicit .csv-filed used during the evaluation in the publication are located under 'results'

## Citing this work ##
If you use this repository, please cite the following paper:
```
@INPROCEEDINGS{9576875,
  author={Kabierski, Martin and 
    Nguyen, Hoang Lam and 
    Grunske, Lars and 
    Weidlich, Matthias},
  booktitle={2021 3rd International Conference on Process Mining (ICPM)}, 
  title={Sampling What Matters: Relevance-guided Sampling of Event Logs}, 
  year={2021},
  volume={},
  number={},
  pages={64-71},
  doi={10.1109/ICPM53251.2021.9576875}}
}
```
