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
#Source for raw data
path1=config["URIWW_Data_Reading_path1"]

#Source for monitoring stations
path2=config["URIWW_Data_Reading_path2"]

assert(os.path.isdir(path1))
assert(os.path.isdir(path2))

# +
#Reading in data
agg_data=pd.DataFrame()
years=list(np.arange(1989, 2021))
years.remove(1996)
years.remove(1997)
years.remove(1998)

for year in years:
    print(year)
    working=pd.read_csv(path1+"URIWW"+str(year)+".csv")
    agg_data=pd.concat([agg_data, working], axis=0)
agg_data
# -

#Stations
stations=pd.read_csv(path2+ "URIWW_Stations.csv", index_col=0)
stations.rename(columns={"WW_Station": "WW ID"}, inplace=True)
stations

#Merging Station Data and Underlying Data
agg_data=agg_data.merge(stations, how="left", on="WW ID", suffixes=["", "_s"])
agg_data.head()

#Unmatched Stations
pd.unique(agg_data.loc[agg_data["Embay_Pt"].isna(), "WW ID"])

#Null Stations
agg_data.loc[agg_data["WW ID"].isna()]

# +
#Reindexing based on newly available parameters

embaydf=agg_data.loc[agg_data["Embay_Pt"]==1].copy(deep=True)
embaydf["OrganizationIdentifier"]="URI"
embaydf.rename(columns={"WW ID":"MonitoringLocationIdentifier"}, inplace=True)
embaydf["MonitoringLocationIdentifier"]=embaydf["MonitoringLocationIdentifier"].apply(lambda x: "URI-" + x)
#embaydf.set_index(["Subw_Embay", "OrganizationIdentifier", "MonitoringLocationIdentifier"], inplace=True)
embaydf.head()
# -

print(len(embaydf))

# +
#embaydf.to_csv("URIWW.csv")
