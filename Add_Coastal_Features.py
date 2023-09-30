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
import os
plt.rcParams["figure.figsize"]=(20, 20)
#Imported specifically to redefine max_iter:
from sklearn.utils.optimize import _check_optimize_result
from scipy import optimize

from osgeo import gdal
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import NearestNDInterpolator

# +
# Display params

from IPython.display import display, HTML
display(HTML("<style>.container { width:100% !important; }</style>"))

pd.options.display.max_rows=150
pd.options.display.max_columns=150
# -

# Originally, this file was in the space version of Geostatistics Workbook, but now it is a standalone file that can be used for the space only or the space and time data

#Working dir
if(os.path.basename(os.getcwd())[0:18]!="Data Visualization"):
    os.chdir("..")
assert(os.path.basename(os.getcwd())[0:18]=="Data Visualization")

# +
#Loading paths config.yml
import yaml

with open("config.yml", "r") as file:
    config = yaml.load(file, Loader=yaml.FullLoader)

# +
#Paths for loading data
import os

paths={}
outputs={}

#TIFFs

#Uncropped Coastal Distance for merging with stations
paths[1]=config["Workbook_path1"]

#CSVs

#Aggregate Data of All Organizations for Testing
#If space
#paths[2]="Data/Space_agg/agg_summer_means_4_21_2023_II.csv"
#If space and time:
paths[2]= "Data/Dominion Energy/Millstone_Eelgrass_Mapping.csv"

#Ouput for Aggregate Data above but with embay_dist attached
#If space
#outputs[1]="Data/Space_agg/agg_summer_means_coastal_features_4_21_2021.csv"
#If space and time
outputs[1]="Data/Dominion Energy/Millstone_Eelgrass_Mapping_coastal_features.csv"

for path in paths.values():
    assert(os.path.exists(path))
# -

cont_orgs=["STS_Tier_II", "EPA_FISM", "USGS_Cont"]

# # Preparing and Testing Data (only run when data is updated)

# ## Importing coastal features

#Using the updated file without points removed
embay_dist = gdal.Open(paths[1])

gt=embay_dist.GetGeoTransform()
gt

proj=embay_dist.GetProjection()
proj

band=embay_dist.GetRasterBand(1)
array=band.ReadAsArray()

pixelwidth=gt[1]
pixelheight=-gt[5]
pixelheight

xOrigin = gt[0]
yOrigin = gt[3]

cols = embay_dist.RasterXSize
rows = embay_dist.RasterYSize
cols

# ## Appending coastal distance to data

df=pd.read_csv(paths[2])
df

pd.unique(df["Organization"])

#Checking on longitude ranges
max(df["Longitude"])-min(df["Longitude"])

df["xPixel"]=((df["Longitude"] - xOrigin) / pixelwidth).astype(int)
df["yPixel"]=(-(df["Latitude"] - yOrigin) / pixelheight).astype(int)

df["embay_dist"]=df.apply(lambda x: array[x["yPixel"], x["xPixel"]], axis=1)

#Getting range of distances in df to ensure its similar to masked
max(df["embay_dist"])-min(df["embay_dist"])

# +
# #Getting locations where dist is null or 0
# print(pd.unique(df.loc[df["embay_dist"].isna(), "Station ID"]))

# pd.unique(df.loc[df["embay_dist"]==0, "Station ID"])
# -

# ## Interpolating Stations caught on Coastal Boundary

nonzero_ind=np.argwhere(array!=0)
nonzero_ind.shape
nonzero_ind

nonzero_values = [array[nonzero_ind[i, 0], nonzero_ind[i, 1]] for i in np.arange(nonzero_ind.shape[0])]
nonzero_values

interp =  NearestNDInterpolator(nonzero_ind, nonzero_values)

zeroes = df.loc[df["embay_dist"]==0]
df.loc[df["embay_dist"]==0, "embay_dist"]=interp(zeroes["yPixel"], zeroes["xPixel"])
df.loc[df["embay_dist"]==0]

#Getting range of distances in df to ensure its similar to masked
max(df["embay_dist"])-min(df["embay_dist"])

working=df.copy()

#Displaying figure along with all locations
plt.figure()
plt.imshow(array)
plt.scatter(working["xPixel"], working["yPixel"], s=1, c="Red")
plt.show()

# +
# #Stations in non-existent embayment (on final model)
# gol=df.loc[df["Station ID"].str.contains("GOL")]
# print(gol)
# df.drop(gol.index, axis=0, inplace=True)
# gol=df.loc[df["Station ID"].str.contains("GOL")]
# print(gol)

# +
# #Fixing East Beach and Barleyfield Cove (These actually should have embay_dist=0)
# df.loc[df["Station ID"]=="East Beach", "embay_dist"] = 0
# df.loc[df["Station ID"]=="Barleyfield Cove", "embay_dist"] = 0
# -

# ## Checking on stations 2019-2021 and outputting array

display_array = np.where(array>0, np.nan, .5)
display_array[0,0]=1

#Reloading Ouput of stations
working=pd.read_csv("Data/Space_and_time_agg/agg_daily_morning_coastal_features_6_21_2023.csv")
working=working.loc[working["Year"].isin(range(2019, 2022))]

# +
#Figure 1a: Non-continuous monitoring stations in LIS
plt.figure()
#plt.imshow(display_array)

dis=working.loc[~working["Organization"].isin(cont_orgs)]

#Saving data to make Figure 1
dis.to_csv("Figures_for_paper/fig1_discrete.csv")

#plt.scatter(dis["xPixel"], dis["yPixel"], cmap="gray", vmin=0, vmax=1, marker="x", c="red", s=200)
#plt.title("Fig 1a: Sampling Locations", size= "xx-large")
#plt.axis("off")
#plt.savefig("Figures_for_paper/fig1a.png")
#plt.show()

# +
#Figure 1b: only continuous stations in the LIS
plt.figure()
#plt.imshow(display_array)
cont=working.loc[working["Organization"].isin(cont_orgs)]

#Saving data to make Figure 1
cont.to_csv("Figures_for_paper/fig1_cont.csv")

#plt.scatter(cont["xPixel"], cont["yPixel"], cmap="gray", vmin=0, vmax=1, marker="x", s=200, c="red")
#plt.title("Fig 1b: Continuous Sampling Locations", size= "xx-large")
#plt.axis("off")
#plt.savefig("Figures_for_paper/fig1b.png")
#plt.show()
# -

print(array.shape)

df.to_csv(outputs[1])


