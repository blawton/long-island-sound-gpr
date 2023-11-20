import pandas as pd
#import shapefile
from json import dumps
import os
import json
import numpy as np
import matplotlib.pyplot as plt
import datetime

pd.set_option('display.max_columns', 50)
pd.set_option('display.max_rows', 100)

# In this notebook, we leave in non-embayment data for now because it will be checked later and this dropping reduces sample size significantly. Also as of v2 merging should be done by year and Station (because locations change).

# +
#Loading paths config.yml
import yaml

with open("../../config.yml", "r") as file:
    config = yaml.load(file, Loader=yaml.FullLoader)

# +
#Paths
paths={}

#STS Data (Original Set)
paths[1]=config["STS_Continuous_Data_Reading_path1"]

#STS data used to fill in 2021 gaps + add 2022 data
paths[2]=config["STS_Continuous_Data_Reading_path2"]

#Map of excel files from STS (format should be year:filename)
tables={2019: config["STS_Continuous_Data_Reading_tables_2019"],
        2020: config["STS_Continuous_Data_Reading_tables_2020"],
        2021: config["STS_Continuous_Data_Reading_tables_2021"],
        2022: config["STS_Continuous_Data_Reading_tables_2022"]
       }

station_sheets={2019: config["STS_Continuous_Data_Reading_stations_2019"],
                2020: config["STS_Continuous_Data_Reading_stations_2020"],
                2021: config["STS_Continuous_Data_Reading_stations_2021"],
                2022: config["STS_Continuous_Data_Reading_stations_2022"]}

#Paths for STS stations
station_paths={}

station_paths[2019] = "/".join([paths[1],  station_sheets[2019]])
station_paths[2020] = "/".join([paths[1],  station_sheets[2020]])
station_paths[2021] = "/".join([paths[2],  station_sheets[2021]])
station_paths[2022] = "/".join([paths[2],  station_sheets[2022]])

#Output file
output_file="STS_Continuous_Data_Pre_Processing_11_18_2023.csv"

for path in paths.values():
    assert(os.path.isdir(path))
    
for path in station_paths.values():
    assert(os.path.exists(path))
# -

#Global variables
dep_var="Temperature (C)"

# # Reading in and Outputting Data

# +
#Reading in stations (WHICH MUST BE FIRST TAB OF EACH EXCEL)
station={}

#nrows restricted where neccesary to only import rows corresp. to stations
station[2019] = pd.read_excel(station_paths[2019], nrows =16)
station[2020] = pd.read_excel(station_paths[2020])
station[2021] = pd.read_excel(station_paths[2021])
station[2022] = pd.read_excel(station_paths[2022])

#Giving a year to each station df for merging
for year in station.keys():
    station[year]["Year"]=year
# -

#Reading in excel files by aggregating all tabs (STATIONS TAB MUST END WITH "Stations")
dfs={}
agg_data=pd.DataFrame()
for i, path in enumerate(paths.values()):
    contents = {k:v for (k, v) in tables.items() if v in os.listdir(path)}
    # print(contents)
    for year, excel in contents.items():
        for name, df in pd.read_excel("/".join([path, excel]), sheet_name=None).items():
            if not name.endswith("Stations"):
                # print(name)
                # print(df.head())
                
                #Giving all rows in a given tab the corresp. station ID
                df["Station ID"]=name
                
                #Fixing format for 2019
                if i==1:
                    df.rename(columns={"Date Time (GMT -04:00)":"Date Time (GMT-04:00)", "Temperature (°C)": "Temperature (C)"}, inplace=True)
                
                agg_data=pd.concat([agg_data, df], axis=0)

#Dropping na for temp (only dataset we care about)
print(len(agg_data))
agg_data.dropna(subset="Temperature (C)", inplace=True)
print(len(agg_data))

#Giving agg_data a year for merging (output should be empty if all entries have year)
agg_data["Year"]=pd.to_datetime(agg_data["Date Time (GMT-04:00)"]).dt.year
print(agg_data.loc[agg_data["Year"].isna()])

#Concatenating Station Lists
stations=pd.concat(station.values(), axis=0)

#Fixing "HNC-" prefix consistency issue
hnc=stations.loc[stations["Station ID"].str.contains("HNC-"), "Station ID"]
no_pref=pd.unique(hnc.str[4:])
agg_data.replace({k:"HNC-"+ k for k in no_pref}, inplace=True)

#Testing replacement (both outputs should be empty)
print(pd.unique(agg_data.loc[~agg_data["Station ID"].isin(stations["Station ID"]), "Station ID"]))
print(stations.loc[stations["Station ID"].isin(no_pref)])

#Dropping Stations with No Coords
print(len(stations))
stations.dropna(subset=["Longitude", "Latitude"], axis=0, inplace=True)
print(len(stations))

#Fixing all Longitude Issues
stations["Longitude"]= -stations["Longitude"].abs()
# stations

#Merging
output=agg_data.merge(stations, how='left', on=["Station ID", "Year"])
# print(len(output))
# output=output.loc[~output["Embayment"].isna()]
# print(len(output))

#Testing for stations without coords (both outputs should be empty)
print(output.loc[output["Longitude"].isna()])
print(output.loc[output["Latitude"].isna()])

#Getting amount of temperature data in each year
for year in np.arange(2019, 2023):
    valid=agg_data.loc[~agg_data[dep_var].isna()].copy(deep=True)
    print(year)
    working=valid.loc[pd.to_datetime(valid["Date Time (GMT-04:00)"]).dt.year==year]
    print(len(pd.unique(working["Station ID"])), "Stations")
    working=agg_data.loc[agg_data["Date Time (GMT-04:00)"].dt.year==year]
    print(len(working), "Datapoints")

output.to_csv(output_file)


