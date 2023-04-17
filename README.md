# long-island-sound-gpr
Preliminary python code and parameter optimization for forthcoming paper supported by my ORISE fellowship.  General approach is a Gaussian Process Regression of temperatures within embayments of the Long Island Sound.
The goal of this model is to have fine-grained (within embayment) temperature data in order to assess susceptibility of eelgrass habitats to warm temperatures and climate change.

__Acknowledgement of Support:__
This research was supported in part by an appointment to the U.S. Environmental Protection Agency (EPA) Research Participation Program administered by the Oak Ridge Institute for Science and Education (ORISE) through an 
interagency agreement between the U.S. Department of Energy (DOE) and the U.S. Environmental Protection Agency. ORISE is managed by ORAU under DOE contract number DE-SC0014664. 
All opinions expressed herein are the author's and do not necessarily reflect the policies and views of US EPA, DOE, or ORAU/ORISE.

## Current Status
All code in this repository should be considered preliminary, and is still in the process of being fine-tuned for presentation. Additionally, although the goal of this project is to
support the mission of open science, not all of the data used to configure and run this model is publically available, and the formats in which they are publically available may differ
from the formats from which I've processed the data in this repo.

The ultimate goal is to have a model that can be run with input from apis or easily downloadable datasets, but at the moment this model relies on many flat files of data,
local paths of which are stored in a config.yaml file in the main repository. Some coordinates from manually acquired data are also stored (locally) in a coords.yaml file.
An env.yaml file with the exact specifications of the python environment used to run code in this repository will be forthcoming.

It is also worth noting that all the python code here, with the exception of a few modules intended for import, were written as notebooks to explore the data as it 
was cleaned and processed. This includes the model itself, parts of which are intended as a display to see the output of the temperature model (i.e. "Geostatistics_Prediction_Dashboard_[version].py"). The jupyter notebooks have been converted to python scripts via [Jupytext](https://github.com/mwouts/jupytext) in the light format, to preserve cell divisions but ensure maximum readability. Cell outputs are referenced in markdown cells or comments when they were crucial to making a decision about model parameters or how to process data, and the project is still in the process of being revised so that outputs make sense and are fully present when code is run as a script.
