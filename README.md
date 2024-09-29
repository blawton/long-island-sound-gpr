# long-island-sound-gpr
Preliminary python code and parameter optimization for forthcoming paper supported by my ORISE fellowship.  General approach is a Gaussian Process Regression of temperatures within embayments of the Long Island Sound, partially inspired by other GPR applications to water quality parameters, such as [1] and [6]. The goal of this model is to have fine-grained (within embayment) temperature data in order to assess susceptibility of eelgrass habitats to warm temperatures and climate change.

## Acknowledgement of Support
This research was supported in part by an appointment to the U.S. Environmental Protection Agency (EPA) Research Participation Program administered by the Oak Ridge Institute for Science and Education (ORISE) through an interagency agreement between the U.S. Department of Energy (DOE) and the U.S. Environmental Protection Agency. ORISE is managed by ORAU under DOE contract number DE-SC0014664. All opinions expressed herein are the author's and do not necessarily reflect the policies and views of US EPA, DOE, or ORAU/ORISE.

## Disclaimer
The ultimate goal is to have a model that can be run with input from apis or easily downloadable datasets, but at the moment this model relies on many flat files of data, local paths of which exist on a config.yaml file. Some coordinates from manually acquired data are also stored (locally) in a coords.yaml file. An env.yaml file with the exact specifications of the python environment used to run code in this repository will be forthcoming. Future versions of this repository will include scripts for obtaining enough data through apis to assemble this model, even if results and parameter decisions may then differ from what I've done here.

It is also worth noting that all the python code here, with the exception of a few modules intended for import, were written as notebooks to explore the data as it was cleaned and processed. The jupyter notebooks have been converted to python scripts via [Jupytext](https://github.com/mwouts/jupytext) in the light format, to preserve cell divisions but ensure maximum readability. Cell outputs are referenced in markdown cells or comments when they were crucial to making a decision about model parameters or how to process data, and the project is still in the process of being revised so that outputs make sense and are fully present when code is run as a script.


## Table of Contents
1. Introduction
2. Current Status
3. Production Model
    * Selection Procedure for hyperparameters and kernel
    * Procedure for cleaning input dataset
4. Preliminary Results/Heatmaps
5. Proposed Further Direction

## 1. Introduction: 

The following figure summarizes the spatial region and data locations with which this model is concerned. In the left map (a), we see the temperature sampling locations, a mix of continuous and discrete stations, and in the right map (b) we see the distribution of eelgrass beds in the region of interest and the sampling stations where shoot counts and biomass are currently measured in the Long Island Sound.

![Figure 1](https://github.com/blawton/long-island-sound-gpr/blob/master/Figures_for_paper/Figure%201.jpg)

## 2. Current Status (6/15/2023)

The model for predicting a daily heatmap in the long island sound based on continuous and discrete data from that year can now be considered in production, as the hyperparameters, kernel, and processing of data used as an input have all been chosen. Initial results demonstrate when the model accurately captures the distribution of data at sampling stations and produces a realistic heatmap vs. when it does not. In particular, we are concerned most with the area where eelgrass is almost exclusively restricted to: the Eastern Sound. This window was defined for the purposes of this project by:

1. lat_min = 40.970592192000
2. lat_max= 41.545060950000
3. lon_min= -72.592354875000
4. lon_max = -71.811481513000
   
The threshold for an accurate temperature map in this window appears to be the number of continuos monitoring stations in the sound, which makes sense as without the sample size provided by data coming in at least on a daily basis, a gaussian process is not a well-chosen model.

## 3. Production Model

### Using Cross Validation to Train Hyperparameters

The kernel and amount of noise to use in this model, within the limitations of scikit learn's kernel implementation, were determined by 5 fold (grouped) cross validation, which was performed on the overall sound dataset, because even though the focus of this project is the Eastern Sound area defined above (where eelgrass is actually present), there is not enough data in this area for a gaussian process to be reliably trained in each year.

Cross Validation training and testing sets were grouped by sampling station, because the true test of a gaussian process run over both spatial and time variables is its ability to synthesize an entire time series in each unknown location as opposed to performing time series interpolation on partially missing time series at a location with data that partially exists. From a mathematical perspective, however, in the case of a seperable kernel such as the RBF kernel, either the spatial or time variables will become simple scale factors on the covariances in the other set of variables and the idea of modeling a changing covariance structure over time is impossible without either a clever combination of kernels that allow the capture of interaction effects.

Predictors were also standardized before the training was run to improve convergence of the LFBGS algorithm used to train the Gaussian Process throughout this entire project. The standardization was run within each fold as part of an sklearn pipeline to prevent data leakage from the test set to the train set.

Cross validation was performed seperately 2019, 2020, and 2021 data, and candidates for kernel combinations in Round I were the three combinations of kernels that performed best on a much simpler interpolation task: interpolation of summer averages for sampling stations in 2019, 2020, and 2021 without any variable for time. The results for these pre-trials have not been organized but relied on the same model imlemented in the [Space](https://github.com/blawton/long-island-sound-gpr/tree/master/Space) folder of this model, as opposed to the [Space_and_Time](https://github.com/blawton/long-island-sound-gpr/tree/master/Space_and_time) folder.

For the actual hyperparameter optimization, two rounds of cross validation were conducted:

### Cross Validation Round I
Used to determine the exact kernel combination with the candidates (chosen as outlined above):
1. 1\**2 * RBF(length_scale=[1, 1, 1, 1]) + 1**2 * RationalQuadratic(alpha=1, length_scale=1)
2. 1\**2 * Matern(length_scale=[1, 1, 1, 1], nu=1.5) + 1**2 * RationalQuadratic(alpha=1, length_scale=1)
3. 1\**2 * Matern(length_scale=[1, 1, 1, 1], nu=0.5) + 1**2 * RBF(length_scale=[1, 1, 1, 1])

Note: the parameters inside the kernel functions are just sklearns notation for the initial pre-training length scales, which are not incredibly important because a) data was normalzed and b) they are ultimately trained. The results of this training are in the [Results](https://github.com/blawton/long-island-sound-gpr/tree/master/Results) folder as "Optimization_with_Pre_Processing_results_[year].csv". The kernel combination that was settled on in the case of this model was the sum of an anisotropic radial basis function kernel with a rational quadratic kernel. These are two of the most popular kernels as explained in [2] below

### Cross Validation Round II
Used to determine alpha parameter, thus avoiding implementing a whitekernel with variable noise as part of the training process. The results of this training are in the [Results](https://github.com/blawton/long-island-sound-gpr/tree/master/Results) folder as "Parameter_Optimization_time_results_[year]_II.csv" and in this case, results led to the selection of an alpha parameter of .25. Because the output of the GPs were demeaned but not normalized, this corresponds to .25 degrees C. Diminishing returns were seen as alpha increased, suggesting .25 to be a near optimal level of noise.

### Cross Validation Round 0
The one other csv in the results folder, [Parameter_Optimization_time_results_2019.csv](https://github.com/blawton/long-island-sound-gpr/tree/master/Results/Parameter_Optimization_time_results_2019.csv), is from an early round of training done only on 2019 data in order to validate that a combination of kernels performs better than a Radial Basis Function alone.

### Procedure for cleaning input dataset

The input dataset, as mentioned in the disclaimer above, consists of a number of flat files obtained from water quality monitoring organizations, as well as files obtained by these data providers' apis. A considerable amount of effort went into both standardizing and processing all of this discordant data. This process can be divided into the following steps:

1. Inital Aggregation of the data within each organization
   * The scripts for this step and step 2 live in the corresponding provider's folder within the [Data](https://github.com/blawton/long-island-sound-gpr/tree/master/Data) folder of the overall repository
2. Processing and outlier removal within each organization (for continuous data)
   * The cleaning done varies by data type. In general not much was done to remove outliers of discrete data, but certain mislabellings of longitude, inconsistencies in naming, etc. were corrected in the same script used to read in the flat csv files
   * Continuous time series were run through a basic despiking algorithm to remove clear outliers as well as remove temperature data at the beginning and end of each year's time series, which generally saw extreme values.
3. Aggregation across organizations
   * This step required a systematic relabelling of all the names of various independent variables, as well as the dependent variable (temperature) to match a standardized template
   * For the production version of the model, this was handled in the script [Cross Org Aggregate_time.py](https://github.com/blawton/long-island-sound-gpr/blob/master/Space_and_time/Cross%20Org%20Aggregate_time.py)
4. Grouping data by day (within a certain time window for continuous data)
   * Also carried out in [Cross Org Aggregate_time.py](https://github.com/blawton/long-island-sound-gpr/blob/master/Space_and_time/Cross%20Org%20Aggregate_time.py)
   * Uses [aggregate_dataset.py](https://github.com/blawton/long-island-sound-gpr/blob/master/Functions/aggregate_dataset.py) from the functions folder to either avergage entire days together from the continuous data or to average within the middle two quartiles of the times at which discrete data were sampled. This ends up being between 6 and 8am based on the quartiles for continuous data calculated in [Cross Org Aggregate_time.py](https://github.com/blawton/long-island-sound-gpr/blob/master/Space_and_time/Cross%20Org%20Aggregate_time.py)
   * Discrete data doesn't have more than one measurement for day so it is simply labelled with its day of the year (all output data has day of year field as an integer)
   * The production version of the model uses the continuous data averaged together within the 6am to 8am time window, the output file for which is not stored in the repository because it represents preliminary data from certain monitoring orgs, but will ultimately be uploaded to the repo
5. Adding coastal features to data
   * This is done in the script [Add_Coastal_Features.py](https://github.com/blawton/long-island-sound-gpr/blob/master/Add_Coastal_Features.py), which takes as inputs the output of step 4 (aggregated daily morning data) and an arcgis-produced distance allocation map (raster/geoTIFF file) that measures distance into each embayment for pixels in LIS embayments.
   * The output of this step has a new parameter "embay_dist" representing distance into embayment, for a total of four independent variables:
      1. Lattitude
      2. Longitude
      3. Day
      4. embay_dist
## 4. Preliminary Results/Graphs

### Heuristic 1: Summer Averages
   The first test performed to see alignment of underlying data with results was a fairly simple test, namely a comparison of the distribtuion of summer averages at sampling stations. This initial metric was chosen because a simple comparison of distribution of model predicted temperatures at sample locations vs sampled temperatures would result in a difference in population for the two distributions in question. The model produces data on each day of the growing season (July 1st to August 31st) whereas most of the monitoring stations are discretely sampled, meaning they have only 4 datapoints for the growing season, spaced out every 2 weeks. This means that while model-predicted data has an even allocation between continuously-sampled locations and discrete stations, the actual sample data is skewed much more towards the continuous stations.

   By taking a summer average at every station, we avoid this effect, and in a sense use a model that first uses all available data to predict daily temperatures (more accurately daily morning temperatures because of the averaging in the 6am to 8am window mentioned above), then averages together these daily temperatures to a summer average that can be compared to the summer average of sampled data at any station. We should expect the actual station data to have fatter tails because some of the averages are only averages of 4 data points vs. the 60 or so data points modelled for every station. The results for each year can be found for both the Eastern Sound window and the overall Long Island Sound by looking in the corresponding folder in [June_Graphs](https://github.com/blawton/long-island-sound-gpr/tree/master/Graphs/June_Graphs).

   The graphs show alignment in distribution when looking at the overall LIS in 2019, 2020, and 2021.

![download](https://github.com/blawton/long-island-sound-gpr/assets/46683509/9a188e04-0a1f-489e-a236-0c3e81fa4350)

![download](https://github.com/blawton/long-island-sound-gpr/assets/46683509/12f5b21d-0625-4db1-9c2d-e12b02731aba)

![download](https://github.com/blawton/long-island-sound-gpr/assets/46683509/564b03ba-c027-492f-b7ec-c8f1f9008704)

However, we can see that as a result of limits in dataset size, this relationship falls apart for the eastern sound specfically in all years with the possible exception of 2021:

![download](https://github.com/blawton/long-island-sound-gpr/assets/46683509/d5a037a8-c533-4793-a05e-c8280e10d8a6)

![download](https://github.com/blawton/long-island-sound-gpr/assets/46683509/5bdfdcb3-c57d-4223-94ac-fabb0046db15)

![download](https://github.com/blawton/long-island-sound-gpr/assets/46683509/095dea99-6c04-4355-b4f8-116912e913b5)

### Conclusion 1: 
A summer average of temperature is a poor way to measure the susceptability of eelgrass to temperature stress for the following reasons:
   a. For discretely measured data, variance of an actual predicted summer average will be high
   b. If instead continuous data is used/modeled, a lower variance in summer average means that the data loses all meaning in predicting temperature extremes in regions where the eelgrass is most susceptible like the Eastern Sound (as shown by the last 3 figures above)
   c. Eelgrass does not respond to average, heat stress occurs on as short a timeframe as XX days (see paper)

### Heuristic 2: Infering time series at locations with a discrete Source of Truth

Instead of evaluating the spatiotemporal Gaussian Process model here by using distributions of averages (which is besides a dubious approach as the test output is trained on the test data), we can look at the ability of the GPR to model a time series at a location where no temperature data exists and then compare the result to nearby continuous time series that were actually measured at nearby locations. The results for a few example stations can be seen below:

![continuous_time_series_modeling](https://github.com/blawton/long-island-sound-gpr/blob/master/Figures_for_paper/fig12.jpg)

The above graphs of inferred temperature at White Point (WP), Niantic River (NR), and Jordan Cove (JC) eelgrass stations, shown on the map in the Introduction, demonstrate several advantages of the Gaussian Process approach:
1. Its ability to create a realistic time series for a location with no existing data
2. The stability of the predicted time series when compared to locations in the Niantic River, which are further from the open sound and thus warmer
3. The confidence interval, which gives a verisimilitude of the modeled time series, and shows when we can conclude with 95% certainty that the temperature at the less-protected White Point station and Jordan Cove stations are colder than that of the stations on the Niantic River

(White Point itself is an interesting use case of the model given its proximity to the Millstone Power Plant, which emits effluent at a consistently monitored temperature. Using this model for temperature nearest to actual eelgrass beds, the susceptibility of the eelgrass to the effluent from Millstone could be inferred).

### Conclusion 2: 

A Gaussian process model provides an option for obtaining realistic predictions that are spatial near existing measurements, yet may differ by a statistically significant margin from those existing measurements, as inferred using the covariance structure learned from training data.

### Heuristic 3:

Because it is in our interest to calculate the number of days above various temperature thresholds (described in detail in working paper), we are also interested in modeling continuous time series at locations where we have discrete test data. 

![continuous_time_series_modeling](https://github.com/blawton/long-island-sound-gpr/blob/master/Figures_for_paper/fig6.png)

### Conclusion 3: 
Time series where there is existing data have much tighter confidence intervals given proximity to existing data, and thus our modeled time series are a good candidate for a new more comprehensive metric for the temperature threat to eelgrass: days over a Heat Stress Threshold (see working paper for exact terminology)

## 5. Proposed Further Direction
The results above lead to the following 3 propostitions for advancement in existing temperature modeling for the Long Island Sound:
1. Using a probabilistic model, in this case a GPR in order to infer fine grained (both spatially and temporily) temperature data for embayments in the long island sound - this process is likely overkill for the open sound, where temperature is dicated more by hydrological phenomena that can be modelled directly with other methods such as [NYHOPS](https://hudson.dl.stevens-tech.edu/maritimeforecast/maincontrol.shtml).

Results from this iteration of output 1 for selected dates:

![continuous_time_series_modeling](https://github.com/blawton/long-island-sound-gpr/blob/master/Figures_for_paper/fig7a.png)

3. Changing the temperature component of the Eelgrass Habitat Suitability Index (EHSI)[5] to a threshold of days over a given threshold, experimental data on eelgrass phenotypes nearest to the LIS suggests should be around 25 degrees Celcius (see Appendix A). Our model does not yet have enough continuous data to esimate the extrema of days over 25 degrees celcius (continuous monitoring stations are being added by the USGS every year), so we have created the below heatmap for days above 23 degrees, which is not an acutely stressful temperature for eelgrass, but has been associated with die-offs as a summer average.

Results of a heatmap of days over 25 degrees C:


   
5. Incorporating more inputs into the probabilistic model for temperature prediction for a more accurate prediction. Good starting points would be:
   a. Depth, for which medium grained spatial data currently exists, and super-fine-grained spatial data is currently being gathered using acoustic scattering data as part of the Long Island Sound Mapping and Research Collaborative (LISMaRC), for which ongoing Phase II progress can be seen [here](https://lismap.uconn.edu/phase-ii-data-fadd-draft-2/)
   b. Hydrological inputs, for example the NYHOPS model linked above
   c. Meteorological inputs like the ones used in [6]


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
