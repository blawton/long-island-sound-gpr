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

# +
#Global Params:

#Defining Eastern Sound Window (useful for visualizing cropped distance map)
lon_min=-72.592354875000
lon_max=-71.811481513000

lat_min=40.970592192000
lat_max=41.545060950000

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

#GeoTiff with 2019 Inverse Distance Weighted CTDEEP Temperature
paths[1] = config["CTDEEP_idw_input1"]

for path in paths.values():
    assert os.path.exists(path), path
# -

# # Mainpulating Coastal Distance to Use in Regression

#Getting open sound as formatted in ArcGIS Pro
idw = gdal.Open(paths[1])
band=idw.GetRasterBand(1)
array=band.ReadAsArray()

gt=idw.GetGeoTransform()
gt

proj=idw.GetProjection()
proj

pixelwidth=gt[1]
pixelheight=-gt[5]
pixelheight

xOrigin = gt[0]
yOrigin = gt[3]

cols = idw.RasterXSize
rows = idw.RasterYSize
cols

plt.imshow(array)


