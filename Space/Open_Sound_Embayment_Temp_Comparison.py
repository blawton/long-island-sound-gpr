import numpy as np
import pandas as pd
import os
import json
from json import dumps
from functools import partial
import matplotlib.pyplot as plt
import requests
from io import StringIO
from IPython import display
import time
from sklearn.neighbors import DistanceMetric
from math import radians

# This notebook needs a proper readme

# +
# Display params

from IPython.display import display, HTML
display(HTML("<style>.container { width:100% !important; }</style>"))

pd.options.display.max_rows=150
pd.options.display.max_columns=150
# -

#Working dir
if(os.path.basename(os.getcwd())[0:18]!="Data Visualization"):
    os.chdir("..")
assert(os.path.basename(os.getcwd())[0:18]=="Data Visualization")

#Global Params
dist = DistanceMetric.get_metric('haversine')

# +
#Loading paths config.yml
import yaml

with open("config.yml", "r") as file:
    config = yaml.load(file, Loader=yaml.FullLoader)

# +
#Loading coords

with open("coords.yml", "r") as file:
    yaml_coords = yaml.load(file, Loader=yaml.FullLoader)

# +
import os
#Paths
paths={}

#STS Discrete Data
paths[1] = config["Comparison_path1"]

#STS Continuous Data
paths[2] = config["Comparison_path2"]

#Pre-processed STS Continuous Data (for bootstrapping)
paths[3]= config["Comparison_path3"]

#Beebe Cove (pre-process this in Data/hobo...)
paths[4]= config["Comparison_path4"]

#Mumford Cove (pre-process this in Data/hobo...)
paths[5]=config["Comparison_path5"]

#All other inputs and outputs come from or go to folders within project
#Can be left hardcoded for now

#Dirs
dirs={}

#Fishers Island All Years but 2022
dirs[1]= config["Comparison_dir1"]

#Fishers Island 2022
dirs[2]= config["Comparison_dir2"]

#Dominion (pre-process this in Data folder)
dirs[3]= config["Comparison_dir3"]

#Strings
strings={}

#Beebe Sensor
strings[1]= config["Comparison_string1"]

#Mumford Sensor
strings[2]= config["Comparison_string2"]

for path in paths.values():
    assert(os.path.exists(path))
    
for path in dirs.values():
    assert(os.path.isdir(path))
# -

# # STS Data

# ## Preparing Discrete STS Data

# +
sts=pd.read_csv(paths[1], index_col=0)
print(len(sts))

sts_discrete=sts[["MonitoringLocationIdentifier", "Embayment","State","Field Latitude (dec. deg.)","Field Longitude (dec. deg.)","Sample Date","Time on Station, 24-Hour Time (hh:mm)","Surface Sample Depth (m)", "Bottom Temperature (°C)", "Subw_Embay", "FID_EelgrassMeadows"]]
sts_discrete=sts_discrete.dropna(subset=["Bottom Temperature (°C)"])
sts_discrete.head()
# -

sts_discrete["Sample Date"]=pd.to_datetime(sts_discrete["Sample Date"])
sts_discrete=sts_discrete.loc[(sts_discrete["Sample Date"].dt.month==7) | (sts_discrete["Sample Date"].dt.month==8)].copy(deep=True)
sts_discrete["Year"]=pd.to_datetime(sts_discrete["Sample Date"]).dt.year
sts_discrete

#Ensuring Numerical Format of Coords
sts_discrete["Field Latitude (dec. deg.)"]=pd.to_numeric(sts_discrete["Field Latitude (dec. deg.)"], errors="coerce")
sts_discrete["Field Longitude (dec. deg.)"]=pd.to_numeric(sts_discrete["Field Longitude (dec. deg.)"], errors="coerce")
sts_discrete.dropna(subset=["Field Latitude (dec. deg.)", "Field Longitude (dec. deg.)"], inplace=True)
sts_discrete.reset_index(inplace=True, drop=True)

discrete_counts=sts_discrete.groupby(["MonitoringLocationIdentifier", "Year"]).count()
discrete_counts.head()

# ## Preparing Continuous STS Data

sts_cont=pd.read_csv(paths[2], index_col=0)

#Renaming Columns
sts_cont.rename(columns={"temp":"Temperature (C)"}, inplace=True)
sts_cont.head()

cont_mean=aggregate_dataset(sts_cont)
cont_mean.head()

# +
# # Handling Dates 
# sts_cont["Date"]=pd.to_datetime(sts_cont["Date"])
# sts_cont["Month"]=sts_cont["Date"].dt.month
# sts_cont["Year"]=sts_cont["Date"].dt.year

# #Getting only months of July and August
# sts_cont=sts_cont.loc[(sts_cont["Month"]==7) | (sts_cont["Month"]==8)].copy(deep=True)
# sts_cont.head()

# +
# #Getting means by month
# cont_mean=sts_cont.groupby(["Station ID", "Year", "Month"]).mean()
# cont_mean.reset_index(level=-1, inplace=True)

# #Dropping missing according to Continuous Time Series
# cont_mean.loc['NWH-O-1B-L', 2019.0]=np.nan
# cont_mean.loc['EAB-O-2B-L', 2020.0]=np.nan
# cont_mean.dropna(subset="Temperature (C)", inplace=True)

# #Getting means by year
# cont_mean.reset_index(inplace=True)
# cont_mean=cont_mean.groupby(["Station ID", "Year"]).mean()

# +
# #Resetting index
# cont_mean.reset_index(inplace=True)
# cont_mean.head()
# -

#Tagging with org
cont_mean["Organization"]="STS_Tier_II"

# # Estimating Continuous vs Discrete Error (Optional)

# This section is optional and mostly useful for calibrating the amount of white noise (alpha) for discrete vs continuous data in geostatistics workbook

# ## Toy Data (skip unless interested in confidence intervals)

# +
#Getting Stations by year with no duplicates
temp_st=sts_discrete.drop_duplicates(["MonitoringLocationIdentifier", "Year"]).copy(deep=True)
temp_st["Field Latitude (dec. deg.)"]=temp_st["Field Latitude (dec. deg.)"].apply(radians)
temp_st["Field Longitude (dec. deg.)"]=temp_st["Field Longitude (dec. deg.)"].apply(radians)

temp_st["coords"]=temp_st.apply(lambda x: [x["Field Latitude (dec. deg.)"], x["Field Longitude (dec. deg.)"]], axis=1)
# -

temp_st.reset_index(inplace=True, drop=True)
temp_st

# +
# Getting dist of sampling times with bins of continuous sampling times (for bootstrapping to calculate variance of averages)

fig, ax = plt.subplots()

times=pd.to_datetime(sts_discrete["Time on Station, 24-Hour Time (hh:mm)"])
plt.hist(times, bins=20)
plt.xticks(rotation=90)
plt.show()

# +
#Making bins so that the center of intervals correspond to the continuous sampling intervals
from datetime import datetime
import datetime as dt

#timestamp creation
arb=datetime.strptime("00:07:30", "%H:%M:%S")
bins = [(arb + dt.timedelta(minutes=int(15*n))).time() for n in np.arange(96)]
# -

#Redoing above hist with custom bins and numpy function
no_nans=sts_discrete["Time on Station, 24-Hour Time (hh:mm)"].dropna()
times=[(datetime.strptime(str(x), "%H:%M")).time() for x in no_nans]
counts, bin_edges=np.histogram(times, bins=bins, density=False)
counts


#Visualization to test
plt.bar(x=np.arange(len(counts)), height=counts)


#Making sampler for sampling time distribution, time must be index of df, if sample is missing it runs again
def temp_sampler(df, timeset, bins, sample_size, station, year):
    
    #Sampling with bins
    no_nans=timeset.dropna()
    times=[(datetime.strptime(str(x), "%H:%M")).time() for x in no_nans]
    
    counts, bin_edges = np.histogram(times, bins=bins, density=False)
    #print(len(counts))
    
    #using midpoints to identify bins
    timebins=pd.Series(bins[0:-1])
    arb=datetime.strptime("00:00:00", "%H:%M:%S")
    timebins = [(arb + dt.timedelta(minutes=int(15*n))) for n in np.arange(95)]
    timebins=[datetime.strftime(edge, "%H:%M:%S") for edge in timebins]
    #print(len(timebins))
    #print(timebins)
    
    #dropping nas and narrowing to embayment
    temp_measured=df.dropna(subset="Temperature (C)")
    working=temp_measured.loc[(temp_measured["Station ID"]==station) & (temp_measured["Year"]==year)].copy(deep=True)
    
    #4 samples of random time
    sample=pd.DataFrame()
    
    for i in np.arange(sample_size):
        time="foo"
        
        #Repeating sampling until a non-missing sample is found
        while list(working.index).count(time)==0:
            time=pd.DataFrame(timebins).sample(n=1, weights=counts, axis=0).iloc[0,0]
            #print(time)

        #using sampled time to draw random sample
        temp=working.loc[time]
            
        sample=pd.concat([sample, temp.sample(n=1, axis=0)])
        
    return(sample["Temperature (C)"])


#continuous data in right format for testing
datapoints=pd.read_csv("Data/STS_Continuous_Data/STS_Continuous_Data_Pre_Processing.csv", index_col=0)
datapoints["Time"]=pd.to_datetime(datapoints["Date Time (GMT-04:00)"]).dt.time.astype(str)
datapoints["Year"]=pd.to_datetime(datapoints["Date Time (GMT-04:00)"]).dt.year
datapoints.set_index("Time", inplace=True)
datapoints

#Testing temp_sampler with data below
temp_sampler(datapoints, sts_discrete["Time on Station, 24-Hour Time (hh:mm)"], bins, 4, "LNE-I-1B-L", 2020)

# +
# Getting dist of sampling dates (for bootstrapping to calculate variance of averages)
from datetime import datetime
fig, ax = plt.subplots()

sts_discrete["Day of year"]=pd.to_datetime(sts_discrete["Sample Date"]).dt.dayofyear
plt.hist(sts_discrete["Day of year"], bins =5)
plt.xticks(rotation=90)
plt.show()
# -

#Checking summary stats to confirm that uniform distribution is justified
sts_discrete["Day of year"].describe()


# ## Method 1 (Preferred): Summer average of discrete data for nearest station to each continuous station

# Done this way because in general there are more discrete than continuous stations and there is a discrete station some reasonable distance away from each continuous station, but NOT vice-versa

def nearest(row):
    coords=[row["Latitude"], row["Longitude"]]
    coord_rads=[radians(_) for _ in coords]

    if (int(row["Year"]) in temp_st["Year"].values):
        distances=temp_st.loc[temp_st["Year"]==row["Year"], "coords"]
        #print(distances)
        distances=distances.apply(lambda x: (dist.pairwise([x, coord_rads]))[0,1])
        distances.sort_values(inplace=True)
        nearest = temp_st.loc[distances.index[0], "MonitoringLocationIdentifier"]
        count=discrete_counts.loc[nearest, row["Year"]]["Bottom Temperature (°C)"]
        return(nearest, distances.iloc[0], count)
    else:
        return([np.nan]*3)


nearest(cont_mean.iloc[0, :])

cont_mean[["nearest", "distance", "datapoints"]]=cont_mean.apply(nearest, axis=1, result_type="expand")
cont_mean.head()


def get_summer_mean(row):
    working=sts_discrete.loc[(sts_discrete["Year"]==row["Year"]) & (sts_discrete["MonitoringLocationIdentifier"]==row["nearest"])]
    mean_discrete=working["Bottom Temperature (°C)"].mean()
    return(mean_discrete)


get_summer_mean(cont_mean.iloc[0])

cont_mean["mean_discrete"]=cont_mean.apply(get_summer_mean, axis=1)
cont_mean.head()

#Getting error column
cont_mean["error"]=cont_mean["mean_discrete"]-cont_mean["Temperature (C)"]
cont_mean["error"]=cont_mean["error"].abs()

#RMSE
cont_mean["squared_error"]=np.square(cont_mean["error"])
np.sqrt(cont_mean["squared_error"].sum()/len(cont_mean))

# +
#Plotting all data
fig, ax =plt.subplots()
plt.scatter(cont_mean["distance"], cont_mean["error"])
plt.xlabel("Haversine Distance")
plt.ylabel("Absolute Error")

for idx, row in cont_mean.iterrows():
    ax.annotate(row['Station ID'], (row['distance'], row['error']) )
    
plt.show()

# +
#Plotting data where the haversine distance is less than threshold
mdist=.0005
working=cont_mean.loc[cont_mean["distance"]<mdist]

fig, ax =plt.subplots()
plt.scatter(working["distance"], working["error"])
plt.xlabel("Haversine Distance")
plt.ylabel("Absolute Error")

for idx, row in working.iterrows():
    ax.annotate(row['Station ID'], (row['distance'], row['error']) )
    
plt.show()
# -

# ## Method 2: Randomly Sampling Continous data to estimate variance

# We use pre-processed data for bootstrapping because it retains time variable

datapoints=pd.read_csv(paths[3])
datapoints["Time"]=pd.to_datetime(datapoints["Date Time (GMT-04:00)"]).dt.time.astype(str)
datapoints.set_index("Time", inplace=True)
datapoints["Year"]=pd.to_datetime(datapoints["Date Time (GMT-04:00)"]).dt.year
datapoints.head()


def create_samples(sample_size, n_samples, station, year):
    sample_means=[]
    for i in np.arange(n_samples):
        temps=temp_sampler(datapoints, sts_discrete["Time on Station, 24-Hour Time (hh:mm)"], bins, sample_size, station=station, year=year)
        mean=temps.mean()
        sample_means.append(mean)
    return(sample_means)


ebay = create_samples(4, 750, "EAB-I-1B-L", 2019)

from scipy.stats import shapiro

stat, p_value = shapiro(ebay)
print(stat)
print(p_value)

pd.Series(ebay).describe()

plt.hist(ebay, bins=20)

np.std(ebay)

#Figuring out all available datapoints
keys=datapoints.groupby(["Station ID", "Year"])["Temperature (C)"].count().unstack(level=-1)
keys.head()

# +
#Compiling all results together (only interested in standard error, which is what std_dev should be labelled as)
years=keys.columns.values
std_devs=pd.DataFrame(index=pd.unique(datapoints["Station ID"]), columns=years)
shapiro_stats=pd.DataFrame(index=pd.unique(datapoints["Station ID"]), columns=years)
p_values=pd.DataFrame(index=pd.unique(datapoints["Station ID"]), columns=years)

for year in years:
    for station in [station for station in keys.index if keys.loc[station, year]>0]:
        samples=create_samples(4, 100, station, year)
        std_devs.loc[station, year]=np.std(samples)
        #print(station)
# -

#Actual Standard Deviations from Toy Data
std_devs

# Its pretty clear from the output above that this model overpredicts the standard error because it doesnt assume spaced out sampling, which is what the discrete sampling actually uses.

# # Comparing summer means to IDW of CTDEEP

# This Section has now been updated to use 2021 data (as of April 5, 2023)

agg_summer_means=pd.read_csv("Data/Space_agg/agg_summer_means_4_21_2023_II.csv", index_col=0)
agg_summer_means

from Data import inverse_distance_weighter as idw


#Make sure that the inverse distance weighter is loaded
def interpolate_means(summer_means):
    output=summer_means.copy(deep=True)
    output["coords"]=output.apply(lambda x: [x["Latitude"], x["Longitude"]], axis=1)
    output["interpolated"]=output.apply(idw.ct_deep_idw, axis=1, args=[1])
    output["Bias"]=output["interpolated"]-output["Temperature (C)"]
    output=output.set_index(["Station ID", "Latitude",  "Longitude", "Year"])[["Temperature (C)", "Bias", "Organization"]]
    
    #Unstacking is unneccesary because STS stations have diff coords each year
    #output=output.unstack(level=-1)
    
    #Unnecessary to convert because datatype of year should be numeric for input
    #output.columns=[int(year) for year in output.columns]
    
    output.reset_index(level=[-1, -2, -3], inplace=True)
    return(output)


agg_mean_error=interpolate_means(agg_summer_means.loc[agg_summer_means["Year"]>2010])
agg_mean_error

#First grouped by organization
agg_mean_error.groupby(["Organization", "Year"])["Bias"].describe()["mean"].loc[:, 2019:]

#Now all organizations together
agg_mean_error.groupby(["Year"])["Bias"].describe()["mean"].loc[2019:]

# # Graphing all Errors in 2d

from matplotlib import colors
for year in np.arange(2015, 2023):
    working=agg_mean_error.loc[agg_mean_error["Year"]==year]
    #print(working.tail())
    try:
        divnorm=colors.TwoSlopeNorm(vmin=-3, vcenter=0., vmax=3)
        fig, ax =plt.subplots()
        plt.scatter(x = working["Longitude"], y = working["Latitude"], cmap="coolwarm_r", norm=divnorm, c=working["Bias"])
        ax.set_title(year)
        
        #for idx, row in working.iterrows():
            #ax.annotate(idx, (row['Longitude'], row['Latitude']) )
            
        plt.show()
    except:
        print("No data for " + str(year))

#Composite of all years
divnorm=colors.TwoSlopeNorm(vmin=-3, vcenter=0., vmax=3)
fig, ax =plt.subplots()
plt.scatter(x = agg_mean_error["Longitude"], y = agg_mean_error["Latitude"], cmap="coolwarm", norm=divnorm, c=agg_mean_error["Bias"], alpha=.25)
ax.set_title("All Year Composite")
#for idx, row in data.iterrows():
    #ax.annotate(idx, (row['Longitude'], row['Latitude']))
plt.show()

# +
#agg_mean_error.to_csv("Data/interpolation_errors_4_5_2023.csv")
