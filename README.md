# long-island-sound-gpr
Preliminary python code and parameter optimization for forthcoming paper supported by my ORISE fellowship.  General approach is a Gaussian Process Regression of temperatures within embayments of the Long Island Sound, partially inspired by other GPR applications to water quality parameters, such as [1] and [6]. The goal of this model is to have fine-grained (within embayment) temperature data in order to assess susceptibility of eelgrass habitats to warm temperatures and climate change.

__Acknowledgement of Support:__
This research was supported in part by an appointment to the U.S. Environmental Protection Agency (EPA) Research Participation Program administered by the Oak Ridge Institute for Science and Education (ORISE) through an interagency agreement between the U.S. Department of Energy (DOE) and the U.S. Environmental Protection Agency. ORISE is managed by ORAU under DOE contract number DE-SC0014664. All opinions expressed herein are the author's and do not necessarily reflect the policies and views of US EPA, DOE, or ORAU/ORISE.

## Disclaimer
The ultimate goal is to have a model that can be run with input from apis or easily downloadable datasets, but at the moment this model relies on many flat files of data, local paths of which exist on a config.yaml file. Some coordinates from manually acquired data are also stored (locally) in a coords.yaml file. An env.yaml file with the exact specifications of the python environment used to run code in this repository will be forthcoming. Future versions of this repository will include scripts for obtaining enough data through apis to assemble this model, even if results and parameter decisions may then differ from what I've done here.

It is also worth noting that all the python code here, with the exception of a few modules intended for import, were written as notebooks to explore the data as it was cleaned and processed. The jupyter notebooks have been converted to python scripts via [Jupytext](https://github.com/mwouts/jupytext) in the light format, to preserve cell divisions but ensure maximum readability. Cell outputs are referenced in markdown cells or comments when they were crucial to making a decision about model parameters or how to process data, and the project is still in the process of being revised so that outputs make sense and are fully present when code is run as a script.


## Table of Contents
1. Current Status
2. Production Model
    * Selection Procedure for hyperparameters and kernel
    * Procedure for cleaning input dataset
3. Preliminary Results/Heatmaps

### 1. Current Status (6/15/2023)

The model for predicting a daily heatmap in the long island sound based on continuous and discrete data from that year can now be considered in production, as the hyperparameters, kernel, and processing of data used as an input have all been chosen. Initial results demonstrate when the model accurately captures the distribution of data at sampling stations and produces a realistic heatmap vs. when it does not. In particular, we are concerned most with the area where eelgrass is almost exclusively restricted to: the Eastern Sound. This window was defined for the purposes of this project by:

1. lat_min = 40.970592192000
2. lat_max= 41.545060950000
3. lon_min= -72.592354875000
4. lon_max = -71.811481513000
   
The threshold for an accurate temperature map in this window appears to be the number of continuos monitoring stations in the sound, which makes sense as without the sample size provided by data coming in at least on a daily basis, a gaussian process is not a well-chosen model.

### 2. Production Model

#### Using Cross Validation to Train Hyperparameters

The kernel and amount of noise to use in this model, within the limitations of scikit learn's kernel implementation, were determined by 5 fold (grouped) cross validation, which was performed on the overall sound dataset, because even though the focus of this project is the Eastern Sound area defined above (where eelgrass is actually present), there is not enough data in this area for a gaussian process to be reliably trained in each year.

Cross Validation training and testing sets were grouped by sampling station, because the true test of a gaussian process run over both spatial and time variables is its ability to synthesize an entire time series in each unknown location as opposed to performing time series interpolation on partially missing time series at a location with data that partially exists. From a mathematical perspective, however, in the case of a seperable kernel such as the RBF kernel, either the spatial or time variables will become simple scale factors on the covariances in the other set of variables and the idea of modeling a changing covariance structure over time is impossible without either a clever combination of kernels that allow the capture of interaction effects.

Predictors were also standardized before the training was run to improve convergence of the LFBGS algorithm used to train the Gaussian Process throughout this entire project. The standardization was run within each fold as part of an sklearn pipeline to prevent data leakage from the test set to the train set.

Cross validation was performed seperately 2019, 2020, and 2021 data, and candidates for kernel combinations in Round I were the three combinations of kernels that performed best on a much simpler interpolation task: interpolation of summer averages for sampling stations in 2019, 2020, and 2021 without any variable for time. The results for these pre-trials have not been organized but relied on the same model imlemented in the [Space](https://github.com/blawton/long-island-sound-gpr/tree/master/Space) folder of this model, as opposed to the [Space_and_Time](https://github.com/blawton/long-island-sound-gpr/tree/master/Space_and_time) folder.

For the actual hyperparameter optimization, two rounds of cross validation were conducted:

**Round I:**
Used to determine the exact kernel combination with the candidates (chosen as outlined above):
1. 1\**2 * RBF(length_scale=[1, 1, 1, 1]) + 1**2 * RationalQuadratic(alpha=1, length_scale=1)
2. 1\**2 * Matern(length_scale=[1, 1, 1, 1], nu=1.5) + 1**2 * RationalQuadratic(alpha=1, length_scale=1)
3. 1\**2 * Matern(length_scale=[1, 1, 1, 1], nu=0.5) + 1**2 * RBF(length_scale=[1, 1, 1, 1])

Note: the parameters inside the kernel functions are just sklearns notation for the initial pre-training length scales, which are not incredibly important because a) data was normalzed and b) they are ultimately trained. The results of this training are in the [Results](https://github.com/blawton/long-island-sound-gpr/tree/master/Results) folder as "Optimization_with_Pre_Processing_results_[year].csv". The kernel combination that was settled on in the case of this model was the sum of an anisotropic radial basis function kernel with a rational quadratic kernel. These are two of the most popular kernels as explained in [2] below

**Round II:**
Used to determine alpha parameter, thus avoiding implementing a whitekernel with variable noise as part of the training process. The results of this training are in the [Results](https://github.com/blawton/long-island-sound-gpr/tree/master/Results) folder as "Parameter_Optimization_time_results_[year]_II.csv" and in this case, results led to the selection of an alpha parameter of .25. Because the output of the GPs were demeaned but not normalized, this corresponds to .25 degrees C. Diminishing returns were seen as alpha increased, suggesting .25 to be a near optimal level of noise.

**Round 0:**
The one other csv in the results folder, [Parameter_Optimization_time_results_2019.csv](https://github.com/blawton/long-island-sound-gpr/tree/master/Results/Parameter_Optimization_time_results_2019.csv), is from an early round of training done only on 2019 data in order to validate that a combination of kernels performs better than a Radial Basis Function alone.

#### Procedure for cleaning input dataset


## References

<a id="1">[1]</a> 
Durham, C. et al (2019).
Process-Based Statistical Models Predict Dynamic Estuarine Salinity. 
Lagoon Environments Around the World - A Scientific Perspective

Ideas for combinations of kernels:

<a id="1">[2]</a> 
Duvenaud, David. 
[The Kernel Cookbook: Advice on Covariance functions](https://www.cs.toronto.edu/~duvenaud/cookbook/)

Code for reading [NWIS](https://waterdata.usgs.gov/nwis?) rdb text files was adapted from:

<a id="1">[3]</a> 
Martin Roberge and Hydrofunctions contributors. (2016-2022). 
[Hydrofunctions](https://hydrofunctions.readthedocs.io/en/master/introduction.html)

<a id="1">[4]</a> 
Flaxman, Seth. (2015).
[Machine Learning in Space and Time](https://www.ml.cmu.edu/research/Flaxman_Thesis_2015.pdf) \[Unpublished Doctoral Thesis\].
Carnegie Mellon University.

The method of data aggregation used on seasonal temperature data and the inverse distance weighting of CT DEEP open sound data follows the same approach used in the Eelgrass Habitat Suitability Index desinged by Jamie Vaudrey, et al:

<a id="1">[5]</a> 
Vaudrey, J. et al. (2013). 
Development and Application of a GIS-based Long Island Sound Eelgrass Habitat Suitability Index Model.
Department of Marine Sciences. 3.

Another GPR based water temperature algorithm:

<a id="1">[6]</a> 
Zhang, Y. et al. (2021). 
A Gaussian process regression-based sea surface temperature interpolation algorithm.
Vol. 39 No. 4, P. 1211-1221
