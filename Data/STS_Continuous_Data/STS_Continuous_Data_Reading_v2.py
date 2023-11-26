import pandas as pd
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
#params

#STS Cont Data (Original Set)
main_path=config["STS_Continuous_Data_Reading_path"]

#Map of excel files from STS (format should be year:filename)
station_sheets={2019: config["STS_Continuous_Data_Reading_stations_2019"],
                2020: config["STS_Continuous_Data_Reading_stations_2020"],
                2021: config["STS_Continuous_Data_Reading_stations_2021"],
                2022: config["STS_Continuous_Data_Reading_stations_2022"]}

#Paths for STS stations
station_paths={year: "/".join([main_path,  station_sheets[year]]) for year in station_sheets.keys()}

#Output file
output_file="STS_Continuous_Data_Pre_Processing_11_18_2023.csv"

assert(os.path.isdir(main_path))
    
for path in station_paths.values():
    assert(os.path.exists(path))

#Dependent variable
dep_var="Temperature (C)"
# -

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

# +
#Reading in excel files by aggregating all tabs (STATIONS TAB MUST END WITH "Stations")
dfs={}
agg_data=pd.DataFrame()
# print(contents)

for year, excel in station_sheets.items():
    for name, df in pd.read_excel("/".join([main_path, excel]), sheet_name=None).items():
        print(name)
        if not name.endswith("Stations"):
            # print(name)
            # print(df.head())
            
            #Giving all rows in a given tab the corresp. station ID
            df["Station ID"]=name
            
            #Fixing column formats
            df.rename(columns={"Date Time (GMT -04:00)":"Date Time (GMT-04:00)", "Temperature (°C)": "Temperature (C)"}, inplace=True)
            
            agg_data=pd.concat([agg_data, df], axis=0)
# -

#Giving agg_data a year for merging (output should be empty if all entries have year)
agg_data["Year"]=pd.to_datetime(agg_data["Date Time (GMT-04:00)"]).dt.year
print("No Times:")
print(agg_data.loc[agg_data["Year"].isna()])

#Dropping na for temp (only dataset we care about)
print(len(agg_data))
agg_data.dropna(subset=[dep_var], inplace=True)
print(len(agg_data))

#Concatenating Station Lists
stations=pd.concat(station.values(), axis=0)

#Fixing "HNC-" prefix consistency issue
hnc=stations.loc[stations["Station ID"].str.contains("HNC-"), "Station ID"]
no_pref=pd.unique(hnc.str[4:])
agg_data.replace({k:"HNC-"+ k for k in no_pref}, inplace=True)

#Testing replacement (both outputs should be empty)
print("Unmatched Stations:", len(pd.unique(agg_data.loc[~agg_data["Station ID"].isin(stations["Station ID"]), "Station ID"])))
print("Stations Missing Prefix:", len(stations.loc[stations["Station ID"].isin(no_pref)]))

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
print("No Lon:", len(output.loc[output["Longitude"].isna()]))
print( "No Lat:", output.loc[output["Latitude"].isna()]))

#Getting amount of temperature data in each year
for year in np.arange(2019, 2023):
    valid=output.loc[output["Year"]==year]
    print(year)
    print(len(pd.unique(valid["Station ID"])), "Stations")
    print(len(valid), "Datapoints")
    print("Duplicates:", valid.duplicated(["Station ID", "Date Time (GMT-04:00)"]).sum())
    valid=valid.loc[(valid["Longitude"]>=-72.59) & (valid["Longitude"]<=-71.81)]
    valid=valid.loc[(valid["Latitude"]>=40.97) & (valid["Latitude"]<=41.54)]
    print(len(pd.unique(valid["Station ID"])), "Stations in the Eastern Sound")

output.to_csv(output_file)
