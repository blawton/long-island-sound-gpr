import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import os

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

#Move to YAML
cont_orgs=["STS_Tier_II", "EPA_FISM", "USGS_Cont"]

#Function for creating the summer avg requires standardized labeling: "Station ID", "Date", "Temperature (C)"
from Functions import aggregate_dataset as agg

# # Daily Summer

# For now, we'll use the interpolated continuous data to predict with complete time series, but in the future, raw data and gp interpolation of partial time series will also be considered 

# +
#Desired output names (order matters)
#These should match the input of the function aggregate_dataset
output_names = ("Date",
"Station ID",
"Latitude",
"Longitude",
"Temperature (C)")

#List of organizations to use
organization_names=["EPA_FISM",
                    "STS_Tier_II",
                    "STS_Tier_I",
                    "USGS_Discrete",
                    "URI",
                    "Dominion",
                    "USGS_Cont"]

#Input names for above outputs as dictionary of ordered pairs
#Dep. variable(s) at end for flexibility
#EPA_FISM is used to refer to hobo logger data
#This keys in this list also serve as a list of organizations and should match the paths below
input_names={"EPA_FISM":("Date", "Station ID", "Latitude", "Longitude",  "temp"), 
             "STS_Tier_II": ("Date", "Station ID", "Latitude", "Longitude", "temp"),
             "STS_Tier_I": ("Sample Date", "MonitoringLocationIdentifier", "Latitude", "Longitude", "Bottom Temperature (°C)"),
             "USGS_Discrete": ("ActivityStartDate", "MonitoringLocationIdentifier", "LatitudeMeasure", "LongitudeMeasure", "ResultMeasureValue"),
             "URI": ("Date of Sample", "MonitoringLocationIdentifier", "LAT_DD", "LON_DD", "Concentration"),
             "Dominion": ("Date", "Station", "Latitude", "Longitude", "Bot. Temp."),
             "USGS_Cont": ("datetime", "site_no", "Latitude", "Longitude", "Temperature"),   
         }
assert(list(input_names.keys())==organization_names)

#All these paths are in the project repo and thus don't need to be stored on config.yaml
organization_paths={'EPA_FISM':"Data/hobo_data_all_years/hobo_data_agg.csv",
                    'STS_Tier_II':"Data/STS Continuous Data/Interpolated_STS_Continuous_Data_4_12_2023.csv",
                    'STS_Tier_I':"Data/STS Discrete Data/STS_Discrete_4_13_2023.csv",
                    'USGS_Discrete':"Data/WQP/WQP_merged_no_STS_4_21_2023.csv",
                    'URI': "Data/URIWW/URIWW_4_19_2023.csv",
                    'Dominion': "Data/Dominion Energy/C_and_NB_data_with_coords.csv",
                    'USGS_Cont': "Data/Mystic River/agg_fully_processed_temp_and_bottom_temp.csv"
                   }
for path in organization_paths.values(): assert(os.path.exists(path))
assert(list(organization_paths.keys())==organization_names)
# +
#Loop of reading, renaming, dropping nas, and running data through aggregate_dataset function
aggregated={}

for org in organization_names:
    #Reading
    working = pd.read_csv(organization_paths[org])
    
    #Renaming
    working.rename(columns=dict(zip(input_names[org], output_names)), inplace=True)
    
    #Testing to make sure all columns are present
    for col in output_names:
        assert(list(working.columns).count(col)==1), org +  " columns count of " + col + " != 1"

    
    #Dropping nas
    print(len(working))
    working.dropna(subset=list(output_names), inplace=True)
    print(len(working))
    
    aggregated[org]=agg.daily(working)
    
    aggregated[org]["Organization"]=org
        
# -


#URI
aggregated["URI"].groupby(["Year"]).count()

#USGS_Discrete
aggregated["USGS_Discrete"].groupby(["Station ID", "Year"]).count()

#USGS_Cont
aggregated["USGS_Cont"].groupby(["Year"]).count()

# +
agg_summer_means=pd.concat(list(aggregated.values()), axis=0)

# #Checking to make sure aggregate_datasets function works
# assert(np.all(agg_summer_means["Month"].values==7.5))
# -

agg_summer_means

# +
#agg_summer_means.to_csv("Data/Space_and_time_agg/agg_summer_means_daily.csv")
# -

# # Daily Summer Morning

# This section uses uninterpolated continuous data to implement gp interpolation of partial time series. It only aggregates data since 2019 to keep code simple and it only averages continuous data between the 33rd and 66th percentile of time of day for discrete data

# +
#Desired output names (order matters)
#These should match the input of the function aggregate_dataset
output_names = ("Date",
"Time",
"Station ID",
"Latitude",
"Longitude",
"Temperature (C)")

#List of organizations to use
organization_names=["EPA_FISM",
                    "STS_Tier_II",
                    "STS_Tier_I",
                    "USGS_Discrete",
                    "URI",
                    "Dominion",
                    "USGS_Cont"]

#Input names for above outputs as dictionary of ordered pairs
#Dep. variable(s) at end for flexibility
#EPA_FISM is used to refer to hobo logger data
#This keys in this list also serve as a list of organizations and should match the paths below
input_names={"EPA_FISM":("Date", "Na", "Station ID", "Latitude", "Longitude",  "temp"), 
             "STS_Tier_II": ("Date Time (GMT-04:00)", "Null", "Station ID", "Latitude", "Longitude", "temp"),
             "STS_Tier_I": ("Sample Date", "Time on Station, 24-Hour Time (hh:mm)", "MonitoringLocationIdentifier", "Latitude", "Longitude", "Bottom Temperature (°C)"),
             "USGS_Discrete": ("ActivityStartDate", "ActivityStartTime/Time", "MonitoringLocationIdentifier", "LatitudeMeasure", "LongitudeMeasure", "ResultMeasureValue"),
             "URI": ("Date of Sample", "Time", "MonitoringLocationIdentifier", "LAT_DD", "LON_DD", "Concentration"),
             "Dominion": ("Date", "Time", "Station", "Latitude", "Longitude", "Bot. Temp."),
             "USGS_Cont": ("datetime", "Null", "site_no", "Latitude", "Longitude", "Temperature"),   
         }
assert(list(input_names.keys())==organization_names)

#All these paths are in the project repo and thus don't need to be stored on config.yaml
organization_paths={'EPA_FISM':"Data/hobo_data_all_years/hobo_data_agg.csv",
                    'STS_Tier_II':"Data/STS Continuous Data/STS_Continuous_Data_Pre_Processing_4_12_2023.csv",
                    'STS_Tier_I':"Data/STS Discrete Data/STS_Discrete_4_13_2023.csv",
                    'USGS_Discrete':"Data/WQP/WQP_merged_no_STS_4_21_2023.csv",
                    'URI': "Data/URIWW/URIWW_6_20_2023.csv",
                    'Dominion': "Data/Dominion Energy/C_and_NB_data_with_coords.csv",
                    'USGS_Cont': "Data/Mystic River/agg_fully_processed_temp_and_bottom_temp.csv"
                   }
for path in organization_paths.values(): assert(os.path.exists(path))
assert(list(organization_paths.keys())==organization_names)
# +
#Loop of reading, renaming, dropping nas
unaggregated={}

for org in organization_names:
    #Reading
    working = pd.read_csv(organization_paths[org])
    
    #Renaming
    working.rename(columns=dict(zip(input_names[org], output_names)), inplace=True)
    
    #Testing to make sure all columns are present
    for col in (set(output_names) - set(["Time"])):
        assert(list(working.columns).count(col)==1), org +  " columns count of " + col + " != 1"

    
    #Dropping nas
    print(len(working))
    working.dropna(subset=[name for name in output_names if name!="Time"], inplace=True)
    print(len(working))
    
    #Restricting to years since 2019
    working["Date"]=pd.to_datetime(working["Date"])
    working["Year"]=working["Date"].dt.year
    working=working.loc[working["Year"]>=2019].copy()
    
    #Saving dataset
    unaggregated[org]=working
    
    #Plotting (continuous distributions should be uniform)
    plt.hist(working["Date"].dt.hour)
    plt.show()


# +
#Getting discrete station time quartiles
discrete_orgs=[key for key in unaggregated.keys() if cont_orgs.count(key)==0]
print(discrete_orgs)
discrete_data=[unaggregated[key] for key in unaggregated.keys() if cont_orgs.count(key)==0]

#Making sure time was properly renamed, formatting it as date, and plotting
for i, df in enumerate(discrete_data):
    assert(list(df.columns).count("Time")==1), "No Time for " + discrete_orgs[i]
    
    #Converting dominion time
    if discrete_orgs[i]=="Dominion":
        df["Hour"]=df["Time"]/100
    else:
        df["Hour"]=pd.to_datetime(df["Time"]).dt.hour
    plt.hist(df["Hour"])
    plt.show()

#Getting quartiles of aggregate Data
discrete_data=pd.concat(discrete_data, axis=0)

bottom=discrete_data["Hour"].quantile(.25)
top=discrete_data["Hour"].quantile(.75)
print(bottom)
print(top)
# -

#Restricting continuous data to min and max of discrete sampling times
for org in cont_orgs:
    working=unaggregated[org]
    working=working.loc[(working["Date"].dt.hour>=bottom) & (working["Date"].dt.hour<=top)].copy()
    unaggregated[org]=working
    
    plt.hist(unaggregated[org]["Date"].dt.hour)
    plt.show()

#Aggregating and tagging
aggregated={}
for org in unaggregated.keys():
    aggregated[org]=agg.daily(unaggregated[org])
    aggregated[org]["Organization"]=org

#URI
aggregated["URI"].groupby(["Year"]).count()

#USGS_Discrete
aggregated["USGS_Discrete"].groupby(["Station ID", "Year"]).count()

#USGS_Cont
aggregated["USGS_Cont"].groupby(["Year"]).count()

# +
agg_summer_means=pd.concat(list(aggregated.values()), axis=0)

# #Checking to make sure aggregate_datasets function works
# assert(np.all(agg_summer_means["Month"].values==7.5))
# -

agg_summer_means

agg_summer_means.to_csv("Data/Space_and_time_agg/agg_summer_means_daily_morning_6_21.csv")
