# long-island-sound-gpr
Preliminary python code and parameter optimization for forthcoming paper supported by my ORISE fellowship.  General approach is a Gaussian Process Regression of temperatures within embayments of the Long Island Sound, partially inspired by [3]
The goal of this model is to have fine-grained (within embayment) temperature data in order to assess susceptibility of eelgrass habitats to warm temperatures and climate change.

__Acknowledgement of Support:__
This research was supported in part by an appointment to the U.S. Environmental Protection Agency (EPA) Research Participation Program administered by the Oak Ridge Institute for Science and Education (ORISE) through an interagency agreement between the U.S. Department of Energy (DOE) and the U.S. Environmental Protection Agency. ORISE is managed by ORAU under DOE contract number DE-SC0014664. All opinions expressed herein are the author's and do not necessarily reflect the policies and views of US EPA, DOE, or ORAU/ORISE.

## Current Status
All code in this repository should be considered preliminary, and is still in the process of being fine-tuned for presentation. Additionally, although the goal of this project is to support the mission of open science, not all of the data used to configure and run this model is publicly available, and the formats in which they are publicly available may differ from the starting formats of the data in this repo.

The ultimate goal is to have a model that can be run with input from apis or easily downloadable datasets, but at the moment this model relies on many flat files of data, local paths of which exist on a config.yaml file. Some coordinates from manually acquired data are also stored (locally) in a coords.yaml file. An env.yaml file with the exact specifications of the python environment used to run code in this repository will be forthcoming. Future versions of this repository will include scripts for obtaining enough data through apis to assemble this model, even if results and parameter decisions may then differ from what I've done here.

It is also worth noting that all the python code here, with the exception of a few modules intended for import, were written as notebooks to explore the data as it was cleaned and processed. This includes the model itself, parts of which are intended as a display to see the output of the temperature model (i.e. [Geostatistics_Prediction_Dashboard](https://github.com/blawton/long-island-sound-gpr/blob/master/Geostatistics_Prediction_Dashboard_2.1.2.py)). The jupyter notebooks have been converted to python scripts via [Jupytext](https://github.com/mwouts/jupytext) in the light format, to preserve cell divisions but ensure maximum readability. Cell outputs are referenced in markdown cells or comments when they were crucial to making a decision about model parameters or how to process data, and the project is still in the process of being revised so that outputs make sense and are fully present when code is run as a script.

## References

Code for reading [NWIS](https://waterdata.usgs.gov/nwis?) rdb text files was adapted from:

<a id="1">[1]</a> 
Martin Roberge and Hydrofunctions contributors. (2016-2022). 
[Hydrofunctions](https://hydrofunctions.readthedocs.io/en/master/introduction.html)

The method of data aggregation used on seasonal temperature data and the inverse distance weighting of CT DEEP open sound data follows the same approach used in the Eelgrass Habitat Suitability Index desinged by Jamie Vaudrey, et al:

<a id="1">[2]</a> 
Vaudrey, J. et al. (2013). 
Development and Application of a GIS-based Long Island Sound Eelgrass Habitat Suitability Index Model.
Department of Marine Sciences. 3.

Another GPR based water temperature algorithm:

<a id="1">[3]</a> 
Zhang, Y. et al. (2021). 
A Gaussian process regression-based sea surface temperature interpolation algorithm.
Vol. 39 No. 4, P. 1211-1221
