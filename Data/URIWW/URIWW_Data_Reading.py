import pandas as pd
from json import dumps
import os
import json
import numpy as np

# +
#Loading paths config.yml
import yaml

with open("../../config.yml", "r") as file:
    config = yaml.load(file, Loader=yaml.FullLoader)

# +
#Source for raw data nad monitoring stations
path1=config["URIWW_Data_Reading_path1"]
path2=config["URIWW_Data_Reading_path2"]

assert(os.path.isdir(path1))
assert(os.path.isdir(path2))
# -

#Reading in data (and making sure that stations exist for 2021)
agg_data=pd.read_csv(path1 + "1988-2021.csv")
agg_data.rename(columns={"WW ID": "Station_Name"}, inplace=True)
agg_data["Date of Sample"]=pd.to_datetime(agg_data["Date of Sample"])
print(agg_data.loc[agg_data["Date of Sample"].dt.year==2021])
agg_data.columns

#Stations
stations=pd.read_csv(path2+ "URIWW_Stations.csv", index_col=0)
stations.rename(columns={"WW_Station":"Station_Name"}, inplace=True)
stations

len(pd.unique(stations["Station_Name"]))

#Merging Station Data and Underlying Data
agg_data=agg_data.merge(stations, how="left", on="Station_Name", suffixes=["", "_s"])
agg_data.head()

# +
#Limiting Data to Eastern Sound Window (to prevent errors in future notebooks)
lon_min=-72.592354875000
lon_max=-71.811481513000

lat_min=40.970592192000
lat_max=41.545060950000
agg_data["LON_DD"]=agg_data["LON_DD"].astype('float')
agg_data["LON_DD"]=agg_data["LON_DD"].astype('float')

print(len(agg_data))
agg_data=agg_data.loc[(lon_min<=agg_data["LON_DD"]) & (agg_data["LON_DD"]<=lon_max)]
agg_data=agg_data.loc[(lat_min<=agg_data["LAT_DD"]) & (agg_data["LAT_DD"]<=lat_max)]
print(len(agg_data))
# -

#Unmatched Stations
print(len(pd.unique(agg_data.loc[agg_data["LAT_DD"].isna(), "Station_Name"])))

# +
#Dropping Stations with no GPS info
agg_data=agg_data.loc[~agg_data["LAT_DD"].isna()]

#Ensuring no null stations
agg_data.loc[agg_data["Station_Name"].isna()]
# -

agg_data

#Reindexing based on newly available parameters
print(len(agg_data))
embaydf=agg_data.loc[agg_data["Embay_Pt"]==1].copy()
print(len(embaydf))
embaydf["OrganizationIdentifier"]="URI"
embaydf.rename(columns={"Station_Name":"MonitoringLocationIdentifier"}, inplace=True)
embaydf["MonitoringLocationIdentifier"]=embaydf["MonitoringLocationIdentifier"].apply(lambda x: "URI-" + x)
#embaydf.set_index(["Subw_Embay", "OrganizationIdentifier", "MonitoringLocationIdentifier"], inplace=True)
embaydf.tail()

pd.unique(embaydf["Parameter"])

#Ensuring all temp is in C (output should be empty)
embaydf.loc[(embaydf["Parameter"]=="Temperature - 00011"), "Unit"]="C"
embaydf.loc[(embaydf["Parameter"]=="Temperature - 00011") & (embaydf["Unit"]!="C")]

print(len(embaydf))
embaydf=embaydf.loc[(embaydf["Parameter"]=="Temperature - 00011") & (embaydf["Unit"]=="C")]
print(len(embaydf))

print(pd.unique(embaydf.columns))
embaydf.sort_values("Date of Sample", inplace=True)
print(embaydf["Date of Sample"])
print(pd.unique(embaydf.columns))

embaydf.to_csv("URIWW_6_20_2023.csv")


