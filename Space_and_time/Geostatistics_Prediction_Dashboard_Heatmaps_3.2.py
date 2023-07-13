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
lon_min=-72.59
lon_max=-71.81
lat_min=40.97
lat_max=41.54

#This should always be set to TRUE for this notebook
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

# # Mainpulating Coastal Distance to Use in Regression

#Getting open sound as formatted in ArcGIS Pro
open_sound = gdal.Open(paths[1])
band=open_sound.GetRasterBand(1)
array_os=band.ReadAsArray()

#Getting all sound from state lines (solely to remove Island from Mystic and other land patches)
all_sound = gdal.Open(paths[2])
band=all_sound.GetRasterBand(1)
array_all=band.ReadAsArray()

#embayment distance including only MEASURED Vaudrey embayments
embay_dist = gdal.Open(paths[3])

gt=embay_dist.GetGeoTransform()
gt

proj=embay_dist.GetProjection()
proj

band=embay_dist.GetRasterBand(1)
array_embay_dist=band.ReadAsArray()

pixelwidth=gt[1]
pixelheight=-gt[5]
pixelheight

xOrigin = gt[0]
yOrigin = gt[3]

cols = embay_dist.RasterXSize
rows = embay_dist.RasterYSize
cols

array_os

array_embay_dist

#Starting visualization
plt.imshow(array_embay_dist)

#Getting rid of mystic (state boundaries need to be double checked with os to avoid making fisher's island embayments)
array=np.where(array_all!=0, array_embay_dist, np.nan)
array=np.where(np.isnan(array), array_os, array)

#Setting negatives to nan for visualization
array=np.where(array<0, np.nan, array)

#Final Array only for embayments
embay_only=np.where(array_embay_dist>0, array, np.nan)
plt.imshow(embay_only)
plt.axis("off")

# +
# #Outputting Final Embayment Distance to avoid repeating above
# driver = gdal.GetDriverByName("GTiff")
# driver.Register()
# outds = driver.Create(paths[6], xsize=cols, ysize=rows, bands=1, eType=gdal.GDT_Int16)
# outds.SetGeoTransform(gt)
# outds.SetProjection(proj)
# outband = outds.GetRasterBand(1)
# outband.WriteArray(embay_only)
# outband.SetNoDataValue(np.nan)
# outband.FlushCache()
# outband=None
# outds=None
# -

# # Reading in Data and Building in Model

# ## Reading coastal features array

#Using the entire sound (not just eastern sound)
array=embay_only

gt=gdal.Open(paths[6]).GetGeoTransform()
gt

array

#Setting negatives to nan
array=np.where(array<0, np.nan, array)
array

#Setting zeroes to nan
array=np.where(array==0, np.nan, array)
array

#Restricting to parts of the sound with lower error
array=np.where(array>.1, np.nan, array)

array.shape

plt.rcParams["figure.figsize"]=(20, 15)
plt.axis("off")
plt.imshow(array)
plt.title("Fig 5: Embayments in Eastern Sound within Range of Model", fontsize=24)
plt.savefig("Figures_for_paper/fig5.png")

# ## Reading csv and building model

#This is the source of the model's predictions
df=pd.read_csv(paths[5], index_col=0)
df.head(50)

#Checking Dominion Stations to Make Sure they are in Vaudrey Embayment (C and NB)
df.loc[df["Organization"]=="Dominion"]

#Dropping nas
print(len(df))
df.dropna(subset=["Station ID", "Longitude", "Latitude", "Temperature (C)", "Organization"], inplace=True)
print(len(df))

#Optional Restriction of TRAINING params to Eastern Sound
df=df.loc[(df["Longitude"]>lon_min) & (df["Longitude"]<lon_max)].copy()
print(len(df))

pd.unique(df["Organization"])

#Getting list of continuous organizations for alphas (move to YAML)
cont_orgs=["STS_Tier_II", "EPA_FISM", "USGS_Cont"]


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
    x_mean = np.mean(X_train, axis=0)
    x_std = np.std(X_train, axis=0)
    X_train=X_train - x_mean
    X_train=X_train / x_std
    
    #Constructing Process
    gaussian_process = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=15, alpha=alpha)


    #Training Process
    gaussian_process.fit(X_train, y_train)

    return(gaussian_process, y_mean)


# # Setting Up Output Heatmap

# +
#Its crucial for the variables in this function to match the order of station_var+indep_var+dep_var 
#(year var added for time func.)
def get_heatmap_inputs(stride, coastal_map, gt, day):
    
    #Reading in geotransform
    pixelwidth=gt[1]
    pixelheight=-gt[5]
    
    xOrigin = gt[0]
    yOrigin = gt[3]
    
    #New array of coastal features
    resampled=coastal_map[0::stride, 0::stride]
    
    #Calculating longitudes of Resampled Array
    
    lons=np.arange(resampled.shape[1])*(pixelwidth)*(stride) + xOrigin
    #Centering
    lons +=pixelwidth/2
    
#     print("Lons")
#     print(min(lons))
#     print(max(lons))
    #Calculating lattitudes of Resampled Array
    
    lats= -np.arange(resampled.shape[0])*(pixelheight)*(stride) + yOrigin
    #Centering
    lats +=pixelheight/2
#     print("Lats")
#     print(min(lats))
#     print(max(lats))
    
    #Making latitude and longitude into a grid
    lonv, latv = np.meshgrid(lons, lats)
    
    #Flattening predictors to run through gaussian_process.predict (adding year)
    X_pred = np.column_stack((lonv.flatten(), latv.flatten(), resampled.flatten(), np.repeat(day, len(resampled.flatten()))))
    
    return(X_pred, resampled, resampled.shape)


# -

# # Model Output

# Initial guesses for model length scales are important because they prevent the model from going into an unstable mode where the middle 2nd length scale parameter blows up. Edit: this still happens by chance even with an initial guess of .1

#Coloring scales
hmin=20
hmax=26
sdmin=0
sdmax=1.2


def model_output(year, stride, process, ymean, day):
    
    #Building Heatmap
    X_pred, resampled, grid_shape = get_heatmap_inputs(stride, array, gt, day)
    
    #Creating y based on X
    X_pred=pd.DataFrame(X_pred)
    y_pred=pd.DataFrame(index=X_pred.index)
    
    #Filtering to prediction region where embay dist is non-nan
    X_pred_filt= X_pred.dropna(axis=0)
    
    #Standardizing X (make sure this matches build_model above)
    period=df.loc[df["Year"]==year, station_var+indep_var+dep_var]
    data=period.values
    X_train = data[:, 1:predictors+1]
    
    #Ensuring proper dtypes
    X_train=np.array(X_train, dtype=np.float64)    
    
    #Using training data to normalize predictors to match trained model
    X_pred_filt = X_pred_filt - np.mean(X_train, axis=0)
    X_pred_filt = X_pred_filt / np.std(X_train, axis=0)
            
    #Running process
    y_pred_filt, MSE = process.predict(X_pred_filt.values, return_std=True)
    y_pred_filt+=ymean
    
    #Adding nas to y dataset
    y_pred["Temp (C)"]=np.nan
    y_pred.loc[X_pred_filt.index, "Temp (C)"]=y_pred_filt
    
    #Creating standard error dataset
    std=pd.DataFrame(index=X_pred.index)
    std["Standard Deviation"]=np.nan
    std.loc[X_pred_filt.index, "Standard Deviation"]=MSE
    
    #Converting Output to Grid
    y_grid = y_pred.values.reshape(grid_shape)
    #y_grid = np.where(resampled!=0, y_grid, np.nan)
    std_grid= std.values.reshape(grid_shape)
    
    return(std_grid, y_grid)


# ## Parameters and Training Process

#Graphing param
plt.rcParams["figure.figsize"]=(30, 25)

# +
#Params
predictors=4
alpha=.25
stride=5
lsb=(1e-5, 1e5)
kernel = 1 * RBF([1]*predictors, length_scale_bounds=lsb) + 1 * RationalQuadratic(length_scale_bounds=lsb)
years = [2019, 2020, 2021]

#Days for showing model differentiation
display_days={"July 15th": 196, "August 15th": 227}

#Dicts for storage of models
kernels_liss = {2019: 1.06**2 * RBF(length_scale=[0.0334, 0.135, 2.29e+03, 0.185], length_scale_bounds="fixed") + 2.85**2 * RationalQuadratic(alpha=0.266, length_scale=1.85, alpha_bounds="fixed", length_scale_bounds="fixed"),
2020:0.939**2 * RBF(length_scale=[0.0683, 0.106, 7.87, 0.14], length_scale_bounds="fixed") + 5.53**2 * RationalQuadratic(alpha=0.126, length_scale=4.06, alpha_bounds="fixed", length_scale_bounds="fixed"),
2021:1.09**2 * RBF(length_scale=[0.213, 0.159, 4.78, 0.122], length_scale_bounds="fixed") + 1.51**2 * RationalQuadratic(alpha=0.412, length_scale=1.26, alpha_bounds="fixed", length_scale_bounds="fixed")
}

kernels_es = {2019: 0.645**2 * RBF(length_scale=[3.17, 3.04e+03, 6.03, 0.0479]) + 7.4**2 * RationalQuadratic(alpha=0.0217, length_scale=1.29),
2020: 1.3**2 * RBF(length_scale=[4.42, 1e+05, 22.9, 0.054]) + 3.45**2 * RationalQuadratic(alpha=0.0324, length_scale=0.451),
2021: 1.46**2 * RBF(length_scale=[5.52, 4.97, 3.84, 0.052]) + 4.03**2 * RationalQuadratic(alpha=0.0231, length_scale=0.538)    
}

#Dicts for storage of models and model properties
kernels = {}
models = {}
ymeans = {}
# -

#Loading Kernels
#Eastern Sound model
if es:
    for year in years:
        #Getting models and ymeans outputted
        models[year], ymeans[year] = build_model(predictors, year, kernels_liss[year], alpha)
        #print(models[year].kernel_)
else:
#Entire LISS model
    for year in years:
        #Getting models and ymeans outputted
        models[year], ymeans[year] = build_model(predictors, year, kernels_liss[year], alpha)
        print(models[year].kernel_)


# ## Heatmaps

#Graphing Params
plt.rcParams["figure.figsize"]=(30, 15)

# +
#Graphing selected days above

#Making Figures
hfig, hax =plt.subplots(len(years), len(display_days), gridspec_kw={'wspace':0, 'hspace':0}, squeeze=True)
efig, eax =plt.subplots(len(years), len(display_days), gridspec_kw={'wspace':0, 'hspace':0}, squeeze=True)

#Plotting
for i, year in enumerate(years):
    for j, day in enumerate(display_days.values()):     
                error, hmap = model_output(year, stride, models[year], ymeans[year], day)
                
                #Visualization Params
                bottom=.15
                top=.6
                height=hmap.shape[0]
                width=hmap.shape[1]

                #hmap plot
                him = hax[i, j].imshow(hmap[int(bottom*height):int(top*height), :], vmin=20, vmax=26)
                hax[i, j].set_title(list(display_days.keys())[j] + " " + str(year), fontsize=18)
                
                #error plot
                eim = eax[i, j].imshow(error[int(bottom*height):int(top*height), :], vmin=0, vmax=1.2)
                eax[i, j].set_title(list(display_days.keys())[j] + " " + str(year), fontsize=18)
                
#Axes off
[ax.set_axis_off() for ax in hax.ravel()]
[ax.set_axis_off() for ax in eax.ravel()]

#hmap cbar
cax = hfig.add_axes([0.1,0.05,0.8,0.03])
cbar = hfig.colorbar(him, cax=cax, orientation='horizontal')
cbar.set_label('Deg C', fontsize=24)
cbar.ax.tick_params(labelsize=24)

#error cbar
cax = efig.add_axes([0.1,0.05,0.8,0.03])
cbar = efig.colorbar(eim, cax=cax, orientation='horizontal')
cbar.set_label('Standard Deviation in Deg C', fontsize=24)
hfig.suptitle("Figure 7: Gaussian Process-Predicted Temperature in Eastern Sound Embayments", fontsize=32)
cbar.ax.tick_params(labelsize=24)

hfig.savefig("Graphs/June_Heatmaps/ES_Heatmaps.png", bbox_inches='tight')
hfig.savefig("Figures_for_paper/fig7a.png", bbox_inches='tight')

efig.savefig("Graphs/June_Heatmaps/ES_Dependent_Errors.png", bbox_inches='tight')
efig.savefig("Figures_for_paper/fig7b.png")

plt.show()
# -


# ## Summer Averages

#Params for graphs of thresholds and summer averages
thresholds = [20, 25]
stride =5
days=range(182, 244)
avg_hmaps= {}
thresh_hmaps={}

# +
#Summer Averages

plt.rcParams["figure.figsize"]=(30, 20)

#Tallying
for i, year in enumerate(years):
    total = np.zeros(1)
    fig, ax = plt.subplots()
    for day in days:
        error, hmap = model_output(year, stride, models[year], ymeans[year], day)
        if len(total)>1:
            total += hmap
        else:
            total = hmap
    total = total/len(days)
    avg_hmaps[year]=total
    avg = plt.imshow(total)
    plt.title("Summer Average Temperature, " + str(year), fontsize=24)
    cbar = fig.colorbar(avg, location="right", shrink=.75)
    cbar.set_label('Deg C', rotation=270)
    cbar.ax.tick_params(labelsize=24)
    plt.axis("off")
    plt.savefig("Graphs/June_Heatmaps/ES_Summer_Average_" + str(year) + ".png")
    plt.show()
    
# -


# ## Days Over Thresholds

# +
#Days over Chosen Threshold(s)
plt.rcParams["figure.figsize"]=(60, 60)


#Making Figures
hfig, hax =plt.subplots(len(years), len(thresholds), gridspec_kw={'wspace':0, 'hspace':0}, squeeze=True)

#Plotting
for i, year in enumerate(years):
    for j, thresh in enumerate(thresholds):
        total = np.zeros(1)
        for day in  days:
            error, hmap = model_output(year, stride, models[year], ymeans[year], day)

            #Establishing threshold as the cut
            over = np.where(hmap>thresh, 1, 0)

            #Readding nans for better visualization
            over=np.where(np.isnan(hmap), np.nan, over)

            if len(total)>1:
                total += over
            else:
                total = over
        him = hax[i, j].imshow(total)
        hax[i, j].set_title("Days over Threshold of " + str(thresh) + " degrees (C), " + str(year), fontsize=24)
        
        thresh_hmaps[str(year)+ "_" + str(thresh)] = total
        
cax = hfig.add_axes([0.1,0.05,0.8,0.03])
cbar = hfig.colorbar(him, cax=cax, orientation='horizontal')
cbar.set_label('Count of Days')
cbar.ax.tick_params(labelsize=24)
[ax.set_axis_off() for ax in hax.ravel()]
plt.savefig("Graphs/June_Heatmaps/ES_Days_over_Thresholds.png")
plt.show()
# -


thresh_hmaps

# ## Outputting TIFFs

# ### Summer Averages

new_gt=(gt[0], gt[1]*stride, gt[2], gt[3], gt[4], gt[5]*stride)
new_gt

years = [2019, 2020, 2021]
for year in years:
    #Downloading geotiff of model
    driver = gdal.GetDriverByName("GTiff")
    driver.Register()
    name = paths[7] + "Final_Temperature_Model_Summer_Average_" + str(year) + ".tif"
    outds = driver.Create(name, xsize=avg_hmaps[year].shape[1], ysize=avg_hmaps[year].shape[0], bands=1, eType=gdal.GDT_Float32)
    outds.SetGeoTransform(new_gt)
    outds.SetProjection(proj)
    outband = outds.GetRasterBand(1)
    outband.WriteArray(avg_hmaps[year])
    outband.SetNoDataValue(np.nan)
    outband.FlushCache()
    outband=None
    outds=None

#Testing saved geotiff
name = paths[7] + "Final_Temperature_Model_Summer_Average_2021.tif"
test=gdal.Open(name)
band=test.GetRasterBand(1)
array=band.ReadAsArray()
plt.imshow(array)

gt=test.GetGeoTransform()
gt

# # Updating EHSI

# ## Old Temperature Component

plt.rcParams["figure.figsize"]=(20, 15)

# +
#Creating average CTDEEP stations for years 2019-2021
new_means=pd.read_csv("Data/CT_DEEP_means_4_5_2023.csv", index_col=0)

#Restricting to years of interest
new_means=new_means.loc[new_means["Year"].isin(years)]
counts=new_means.groupby("station name").count().reset_index()

#ensuring stations have 1 measurement/year
stations=counts.loc[counts["Year"]==3, "station name"]

new_means=new_means.loc[new_means["station name"].isin(stations)]
new_means=new_means.groupby(["station name"])["temperature", "LatitudeMeasure", "LongitudeMeasure"].mean()
new_means.to_csv("Data/ct_deep_temp_2019_2021.csv")

# +
import numpy as np
import pandas as pd
import os
import json
from json import dumps
import matplotlib.pyplot as plt

#Additional imports for distance calculations
from sklearn.neighbors import DistanceMetric
from math import radians

#defining distance metric of choice
dist = DistanceMetric.get_metric('haversine')

#Reading in CT DEEP averages with fielpath for this program to be run elsewhere
assert(os.path.exists('C:\\Users\\blawton\\Data Visualization and Analytics Challenge\\Data\\'))

ct_means=pd.read_csv('C:\\Users\\blawton\\Data Visualization and Analytics Challenge\\Data\\ct_deep_temp_2019_2021.csv', index_col=0)
                                 
#Creating col for coords in radians and setting index
ct_means["coords"]=ct_means.apply(lambda x: [radians(x["LatitudeMeasure"]), radians(x["LongitudeMeasure"])], axis=1)
ct_means

#Defining function of interest
def ct_deep_idw(row, power):
    coord=row.loc["coords"]
    coord=[radians(deg) for deg in coord]    
    distances=ct_means["coords"].apply(lambda x: (dist.pairwise([x, coord]))[0,1])
    #print(distances)
    
    #Accounting for divide by zero by taking mean of first zero value
    if min(distances)>0:
        weights=1/distances
        weights=np.power(weights, power)
        weights=weights/weights.sum()
    else:
        return("Error: Divde by Zero")
    
    weighted=np.dot(weights, ct_means["temperature"])
    return(weighted)



# -

#Make sure this stride matches the above stride
stride=5


#Make sure that the inverse distance weighter is loaded, this now requires a year variable"Year" and a day variable "Day" in each row
def interpolate_means(data):
    output=data.copy(deep=True)
    output["coords"]=output.apply(lambda x: [x["Latitude"], x["Longitude"]], axis=1)
    output["interpolated"]=output.apply(ct_deep_idw, axis=1, args=[1])
    
    #Unstacking is unneccesary because STS stations have diff coords each year
    #output=output.unstack(level=-1)
    
    #Unnecessary to convert because datatype of year should be numeric for input
    #output.columns=[int(year) for year in output.columns]
    
    return(output)


## Getting IDW CTDEEP TEMP for 2009-2011
X_pred, resampled, grid_shape = get_heatmap_inputs(stride, array, gt, 0)
avg_temp=pd.DataFrame(X_pred, columns = indep_var)
working=avg_temp.loc[~avg_temp["embay_dist"].isna()]
print(working.shape)
interped =interpolate_means(working)
avg_temp["interpolated"]=np.nan
avg_temp.loc[~avg_temp["embay_dist"].isna()]=interped
print(avg_temp["embay_dist"])

#Aggregating CTDEEP idw temperatures
avg_temp=avg_temp["interpolated"].values.reshape(grid_shape)

display =plt.imshow(avg_temp)
plt.axis("off")
cbar = plt.colorbar(display, location="right", shrink=.75)
cbar.set_label('Temperature (C)', rotation=270, fontsize=18, labelpad=30)
cbar.ax.tick_params(labelsize=18)
plt.title("Figure 5: Inverse Distance Weighted CT DEEP temperature data in Eastern LIS", fontsize=24)
plt.savefig("Figures_for_paper/fig4.png")

bins=[0, 21, 21.4, 21.9, 22.3, 22.8, 23.2, 23.7, 24.1, 24.6, 25, 100]

#Reclassifying
reclass1=np.digitize(avg_temp, bins, right=False)
reclass1=10-(reclass1-1)
reclass1=np.where(~np.isnan(avg_temp), reclass1, np.nan)
display = plt.imshow(reclass1, vmin=0, vmax=10)
cbar = plt.colorbar(display, location="right", shrink=.75)
cbar.set_label('Temperature Score (Open Sound Model)', rotation=270, fontsize=18, labelpad=30)
cbar.ax.tick_params(labelsize=18)
plt.axis("off")
plt.tight_layout()

# ## New Temperature Component

years=[2019, 2020, 2021]

#Averaging summer average heatmaps from 2019 through 2021
avg_temp2= np.zeros(avg_hmaps[years[0]].shape)
for key in avg_hmaps.keys():
    print(key)
    hmap=avg_hmaps[key]
    avg_temp2+=avg_hmaps[key]
avg_temp2=avg_temp2/len(avg_hmaps.keys())
display = plt.imshow(avg_temp2)
cbar = plt.colorbar(display, location="right", shrink=.75)
cbar.set_label('Temperature', rotation=270, fontsize=18, labelpad=30)
cbar.ax.tick_params(labelsize=18)
plt.title("GP-Predicted Temperature in LIS Embayments (2019 through 2021)", fontsize=24)
plt.axis("off")
plt.tight_layout()

bins=[0, 21, 21.4, 21.9, 22.3, 22.8, 23.2, 23.7, 24.1, 24.6, 25, 100]

#Reclassifying new temperature heatmap
reclass2=np.digitize(avg_temp2, bins, right=False)
reclass2=10-(reclass2-1)
reclass2=np.where(~np.isnan(avg_temp2), reclass2, np.nan)
display=plt.imshow(reclass2)
cbar = plt.colorbar(display, location="right", shrink=.75)
cbar.set_label('Temperature Score', rotation=270, fontsize=18, labelpad=30)
cbar.ax.tick_params(labelsize=18)
plt.axis("off")
plt.title("Gaussian Processs ")
plt.tight_layout()

# ## EHSI Difference

# ### Raw Difference

plt.rcParams["figure.figsize"]=(30, 20)

display=plt.imshow(avg_temp-avg_temp2, cmap="seismic")
cbar = plt.colorbar(display, location="bottom", shrink=.75)
cbar.set_label('Increase in Temp with Embayment Data + GP (Degrees (C))', fontsize=18, labelpad=30)
cbar.ax.tick_params(labelsize=18)
plt.axis("off")
plt.title("Figure 8: GP Temperature Model and CTDEEP IDW comparison", fontsize=24)
plt.savefig(paths[7] + "EHSI_open_sound_vs_embayment_bias_graph_raw.png", )
plt.show()

# ### Reclassified Difference

plt.rcParams["figure.figsize"]=(30, 20)

display=plt.imshow(reclass1-reclass2, cmap="seismic", vmin=-6, vmax=6)
cbar = plt.colorbar(display, location="bottom", shrink=.75)
cbar.set_label('New Model Increase in EHSI Temp. Component', fontsize=18, labelpad=30)
cbar.ax.tick_params(labelsize=18)
plt.axis("off")
plt.title("Figure 8: GP Temperature Model and CTDEEP IDW comparison", fontsize=24)
plt.savefig(paths[7] + "EHSI_open_sound_vs_embayment_bias_graph.png", )
plt.savefig("Figures_for_paper/fig8.png", bbox_inches='tight')
plt.show()


