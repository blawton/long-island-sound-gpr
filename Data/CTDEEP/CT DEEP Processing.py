import pandas as pd
from json import dumps
import os
import json
import numpy as np

pd.options.display.max_rows=150
pd.options.display.max_columns=150

# +
#Loading paths config.yml
import yaml

with open("../../config.yml", "r") as file:
    config = yaml.load(file, Loader=yaml.FullLoader)

# +
#Data from CT DEEP Reading (also reference points for missing coords)

paths={}
paths[1] = config["CT_DEEP_Processing_path1"]

#Path for CT DEEP Station Info w/ Coordinates (SHOULD BE IN WQP FORMAT and Station Names should have left 0s and no right Bs)
paths[2] = config["CT_DEEP_Processing_path2"]

#Reference Points to Get Missing Coordinates for Stations
paths[3] = config["CT_DEEP_Processing_path3"]

for i in np.arange(1, 4):
    assert(os.path.exists(paths[i]))

# +
#Params

#Depth Cutoff for averaging (Niantic Used as Starting Point)
#Both are inclusive
depth_min=2
depth_max=3
# -

# # Reading in and formatting data

data=pd.read_csv(paths[1])

stations=pd.read_csv(paths[2])

ref_points=pd.read_csv(paths[3])

#Dropping Temp nas (only variable of interest for this repo)
print(len(data))
data.dropna(subset=["temperature"], inplace=True)
print(len(data))

data["temperature"]=pd.to_numeric(data["temperature"])

data.head()

pd.unique(data["station name"])

data["station name"]=data["station name"].astype(str)
pd.unique(data["station name"])

#Dropping the Bs from stations (they're in the same location)
data["station name"]=data["station name"].str.rstrip("B")
pd.unique(data["station name"])

stations.head()

#Formatting stations from station table
stations["MonitoringLocationName"]=stations["MonitoringLocationName"].astype(str)
stations["station name"]=stations["MonitoringLocationName"].str.lstrip("0")
pd.unique(stations["station name"])

#Checking for duplicate names
print(len(pd.unique(stations["station name"])))
print(len(pd.unique(stations["MonitoringLocationName"])))
stations.loc[stations.duplicated(subset=["station name"])]

# # Averaging CTDEEP data by month and by station

#Pre-processing for filtering
data["depth"]=pd.to_numeric(data["depth"])

#filtering to depth range in parameters
filtered=data.loc[(data["depth"]>=depth_min) & (data["depth"]<=depth_max)].copy(deep=True)
len(filtered)

#pre-processing date
filtered.dropna(subset=["Date"], axis=0)
filtered["Date"]=pd.to_datetime(filtered["Date"])
filtered["Month"]=filtered["Date"].dt.month
filtered.head()

#filtering to july and august (summer months)
filtered=filtered.loc[(filtered["Month"]==7) | (filtered["Month"]==8)].copy(deep=True)
len(filtered)

#Adding in year for grouping
filtered["Year"]=filtered["Date"].dt.year

#Averaging temp by month, year, and station like in Niantic Study
temp_means=pd.DataFrame(filtered.groupby(["station name", "Year", "Month"])["temperature"].mean())
temp_means

temp_means.reset_index(inplace=True)
temp_means

#Averaging months where both exist, otherwise dropping datapoint
keep=temp_means.duplicated(subset=["station name", "Year"], keep=False)
len(temp_means.loc[keep])

#Investigating unaveraged stations (Ensure that 2019 onwards are mostly here)
temp_means.loc[~keep]

#cont.
filtered_means=temp_means.loc[keep].copy(deep=True)
double_means=filtered_means.groupby(["station name", "Year"]).mean()
double_means

#Reformatting
double_means.drop("Month", axis=1, inplace=True)
double_means.reset_index(inplace=True)
double_means

# # Working with coords

#Dropping S2 (2010 and 2011 station)
double_means=double_means.loc[double_means["station name"]!="S2"].copy(deep=True)
double_means

#Merging means with station locations
merged=double_means.merge(stations[["station name", "LatitudeMeasure","LongitudeMeasure", "Subw_Embay", "Embay_Pt"]], how="left", on="station name")


#Stripping left zeroes for matching with ref
merged["Station ID"]=merged["station name"].str.lstrip("0")
merged.head()

#Getting unmerged
unmerged=merged.loc[(merged["LatitudeMeasure"].isna()) | (merged["LongitudeMeasure"].isna())]
unmerged_names=pd.unique(unmerged["Station ID"])
unmerged_names

#Reference Points to get coords for unmerged stations
ref_points.head()

#Filling in unmerged
for index, row in unmerged.iterrows():
    ID=row.loc["Station ID"]
    if len(ref_points.loc[(ref_points["Station Name"]==ID) & (~ref_points["Latitude"].isna())])>1:
        raise "Error"
    else:
        working = ref_points.loc[(ref_points["Station Name"]==ID) & (~ref_points["Latitude"].isna())].copy()
        unmerged.loc[index, "LatitudeMeasure"]= working["Latitude"].values[0]
        unmerged.loc[index, "LongitudeMeasure"]= working["Longitude"].values[0]

#Filling in merged w/ unmerged
merged.loc[(merged["LatitudeMeasure"].isna()) | (merged["LongitudeMeasure"].isna())]=unmerged

#Testing to ensure all unmerged station coords were filled in (output should be empty)
merged.loc[(merged["LatitudeMeasure"].isna()) | (merged["LongitudeMeasure"].isna())]

#Dropping Station ID
merged.drop("Station ID", axis=1, inplace=True)

# +
#Saving all data to file
#merged.to_csv("CT_DEEP_means_4_5_2023.csv")
# -

# # Making geojsons for inverse-distance-weighting (optional)

# This section is optional and instead of using arcgis to inverse distance weight geojsons, the table outputted above can be put into arcgis directly, or the idw functions in this repo can be used

import geopandas as gpd
import pandas as pd

#Make sure to run in an environment with fiona
merged=pd.read_csv("CT_DEEP_means.csv", index_col=0)
merged

from shapely.geometry import Point

#Adding geometry column to merged
merged["geometry"]=[Point(merged.loc[i, "LatitudeMeasure"], merged.loc[i, "LongitudeMeasure"]) for i in merged.index]
merged

years = [2018, 2019, 2020, 2021]

# +
# for year in years:
#     working=merged.loc[merged["Year"]==year].copy(deep=True)
#     print(year)
#     print(len(working))
#     working=gpd.GeoDataFrame(working, crs="EPSG:4326")
#     name= insert_output_filename + "_" + str(year) + ".json"
#     working.to_file(name, driver="GeoJSON")
# -


