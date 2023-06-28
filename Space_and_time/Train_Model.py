import pandas as pd
import numpy as np
from itertools import product
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF
from sklearn.gaussian_process.kernels import RationalQuadratic
from sklearn.gaussian_process.kernels import WhiteKernel
from sklearn.gaussian_process.kernels import Matern
import matplotlib.pyplot as plt
import statsmodels.api as sm
from scipy.interpolate import NearestNDInterpolator
import matplotlib as mpl
import os

from osgeo import gdal, gdalconst
import numpy as np
import matplotlib.pyplot as plt

# +
# Display params

from IPython.display import display, HTML
display(HTML("<style>.container { width:100% !important; }</style>"))

pd.options.display.max_rows=150
pd.options.display.max_columns=150
# -

# __Readme__

# This notebook implements v2.1 of the Gaussian Process Regression to predict Long Island Sound Coastal Temperature. It does not rely on a training and test set in order to evaluate accuracy, but instead builds a model for each year and evaluates it on a grid of test points in order to create a heatmap.

# __v_2.1.2__

# In this version:
#
# 1. Window of interest adjusted to include New York Embayments as per Cayla's suggestion
# 2. Output is a geotiff for making a composite
# 3. Leverage masked input to designate embayments, and use a default of embay_dist=0 for the rest of the sound
# 4. Implement v2.1.2 of the Model (heatmaps made after this change are listed as "_v3"
# 5. Added in an option for fixed kernel implementation that averages the hyperparamaters of the desired modes from the 2019 and 2020 models
# 6. Changed "test_mean" to "mean" in model_output (HUGE bug)

# __v_3__

# 1. "manipulating Coastal Distance" now produces optional embayment only map (stored in path6)
# 2. Removed fixed kernel implementation
# 3. Removed test section
# 4. Rational Quadratic Kernel Used Instead of RBF

#Graphing param
plt.rcParams["figure.figsize"]=(30, 25)

# +
#Global Params:

#Defining Eastern Sound Window (useful for visualizing cropped distance map)
lon_min=-72.592354875000
lon_max=-71.811481513000

lat_min=40.970592192000
lat_max=41.545060950000

#Whether or note to restrict to the Eastern Sound window
es=True

#Variables

station_var=["Station ID"]
#(order matters here because of how get_heatmap_inputs works)
indep_var=["Longitude", "Latitude", "embay_dist", "Day"]
dep_var= ["Temperature (C)"]
# -

#Working dir
if(os.path.basename(os.getcwd())[0:18]!="Data Visualization"):
    os.chdir("..")
assert(os.path.basename(os.getcwd())[0:18]=="Data Visualization")

# +
#Loading paths config.yml
import yaml

with open("config.yml", "r") as file:
    config = yaml.load(file, Loader=yaml.FullLoader)
# -

# All TIFFs loaded below are cropped to eastern sound, this cropping is done in "Cropping to Eastern Sound" script/nb

# +
import os
#Paths for Inputs
paths={}

# Coastal features

#GeoTiff for part of LIS not in a vaudrey embayment that was measured (open sound)
paths[1] = config["Prediction_Dashboard_path1"]

#Geotiff of embay distance clipped to be outside state lines (to remove land)
paths[2] = config["Prediction_Dashboard_path2"]

#Embayment distance clipped to be only in a vaudrey embayment that had measurements and an embay_dist value
paths[3] = config["Prediction_Dashboard_path3"]

#Final embayment distance (output of first paragraph of notebook from v 2.1.2)
paths[4]  = config["Prediction_Dashboard_path4"]

#Final embayment distance (output of first paragraph of notebook) BUT within measured embays
#This is the preferred dataset for dashboard 3 but doesn't open w/ gdal properly
paths[6] = config["Prediction_Dashboard_path6"]

#Like paths[6] but within entire sound
paths[8] = config["Prediction_Dashboard_path8"]

#Ouput Path for tiffs
paths[7] = config["Prediction_Dashboard_output"]

#CSV

#Training Data
paths[5] =  "Data/Space_and_time_agg/agg_daily_morning_coastal_features_6_21_2023.csv"

for path in paths.values():
    assert os.path.exists(path), path
# -

# # Reading csv and building model

#This is the source of the model's predictions
df=pd.read_csv(paths[5], index_col=0)
df.head(50)

#Dropping nas
print(len(df))
df.dropna(subset=["Station ID", "Longitude", "Latitude", "Temperature (C)", "Organization"], inplace=True)
print(len(df))

# #Optional Restriction of TRAINING and later testing params to Eastern Sound
if es==True:
    df=df.loc[(df["Longitude"]>lon_min) & (df["Longitude"]<lon_max)].copy()
    print(len(df))


#Building Model (adapted to dashboard from "Geostatistics_Workbook_version_2.1.2.ipynb")
def build_model(predictors, year, kernel, alpha):
    
    #Selecting data from proper year
    period=df.loc[df["Year"]==year, station_var+indep_var+dep_var]
    data=period.values

    #Partitioning X and y
    
    X_train, y_train = data[:, 1:predictors+1], data[:, -1]
    
    #Ensuring proper dtypes
    X_train=np.array(X_train, dtype=np.float64)
    y_train=np.array(y_train, dtype=np.float64)

    #Demeaning y
    y_mean=np.mean(y_train)
    y_train=y_train-y_mean

    #Normalizing predictors (X) for training sets
    X_train=X_train - np.mean(X_train, axis=0)
    X_train=X_train / np.std(X_train, axis=0)
    
    #Constructing Process
    gaussian_process = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=15, alpha=alpha)


    #Training Process
    gaussian_process.fit(X_train, y_train)

    return(gaussian_process, y_mean)

# # Parameters and Training Process

# +
#Params
predictors=4
alpha=.25
stride=20
lsb=(1e-5, 1e5)
kernel = 1 * RBF([1]*predictors, length_scale_bounds=lsb) + 1 * RationalQuadratic(length_scale_bounds=lsb)
years = [2019, 2020, 2021]
thresholds = [24, 24.5, 25]
temp_limits = (19, 27)
day_limits = (0, 50)
#continuous monitoring "orgs"
cont_orgs=["STS_Tier_II", "EPA_FISM", "USGS_Cont"]

#Days for showing model differentiation
display_days={"July": 196, "August": 227}

#No dicts for storage of models because this script saves model hyperparams to a csv below

errors= {}
models = {}
hmaps= {}
ymeans = {}
# -

# Running actual model to get hyperparams
for year in years:
    #Getting models and ymeans outputted
    models[year], ymeans[year] = build_model(predictors, year, kernel, alpha)
    print(models[year].kernel_)

# Saving Kernels
if es:
    string_kernels=[str(models[year].kernel_) for year in years]
    kernels = pd.DataFrame({"Year":years, "Kernel":string_kernels})
    kernels.to_csv("Results/ES_kernels_final.csv")
else:
    string_kernels=[str(models[year].kernel_) for year in years]
    kernels = pd.DataFrame({"Year":years, "Kernel":string_kernels})
    kernels.to_csv("Results/LISS_kernels_final.csv")


