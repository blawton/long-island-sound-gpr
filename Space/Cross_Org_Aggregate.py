import pandas as pd
import numpy as np
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

# # Getting aggregates of all data 

#Function for creating the summer avg requires standardized labeling: "Station ID", "Date", "Temperature (C)"
from Functions import aggregate_dataset as agg

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
    
    aggregated[org]=agg.summer(working)
    
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

#Checking to make sure aggregate_datasets function works
assert(np.all(agg_summer_means["Month"].values==7.5))

# +
#agg_summer_means.to_csv("Data/Space_agg/agg_summer_means_4_21_2023_II.csv")
