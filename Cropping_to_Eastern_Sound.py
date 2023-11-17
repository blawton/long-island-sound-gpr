import pandas as pd
import numpy as np
from itertools import product
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF
from sklearn.gaussian_process.kernels import RationalQuadratic
from sklearn.gaussian_process.kernels import WhiteKernel
import matplotlib.pyplot as plt
import statsmodels.api as sm
from scipy.interpolate import NearestNDInterpolator
import matplotlib as mpl
from osgeo import gdal, gdalconst

# +
#Global Params:

#Defining Eastern Sound Window
lon_min=-72.592354875000
lon_max=-71.811481513000

lat_min=40.970592192000
lat_max=41.545060950000

# +
#Loading paths config.yml
import yaml

with open("config.yml", "r") as file:
    config = yaml.load(file, Loader=yaml.FullLoader)

# +
import os
#Paths
inputs={}
outputs={}

#Input and output for final embay_distance with open sound removed in arcgis (so that all of open sound gets embay_dist=0)
#Used in ("Geostatistics_Prediction_Dashboard...")
inputs[1] = config["Cropping_input1"]
outputs[1] = config["Cropping_output1"]

for path in inputs.values():
    assert(os.path.exists(path))
    
# for path in outputs.values():
#     assert(os.path.exists(path))

# +
# Cropping input coastal features (use this on both embayment only and sound boundary masked data)
uncropped= inputs[1]
output= outputs[1]
window = (lon_min ,lat_max ,lon_max, lat_min)

gdal.Translate(output, uncropped, projWin = window)
# -


