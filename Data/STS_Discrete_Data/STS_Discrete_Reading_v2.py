import pandas as pd
from json import dumps
import os
import json
import numpy as np

pd.set_option('display.max_columns', 50)
pd.set_option('display.max_rows', 100)

# +
#Loading paths config.yml
import yaml

with open("../../config.yml", "r") as file:
    config = yaml.load(file, Loader=yaml.FullLoader)

# +
#Path Params

#Path for Data
path1=config["STS_Discrete_Reading_path1"]
assert(os.path.isdir(path1))

#Path for Stations
path2 = config["STS_Discrete_Reading_path2"]
assert(os.path.exists(path2))

#Tables in format year:filename
tables={2018:config["STS_Discrete_Reading_tables_2018"],
        2019:config["STS_Discrete_Reading_tables_2019"],
        2020:config["STS_Discrete_Reading_tables_2020"],
        2021:config["STS_Discrete_Reading_tables_2021"]}

for table in tables.values():
    assert(os.path.exists(path1 + "/" + table))

# +
dfs={}
agg_data=pd.DataFrame()

for year, table in tables.items():
    dfs[year]=pd.read_csv(path1+ "/" + table)
    # print(dfs[year].head)
    agg_data=pd.concat([agg_data, dfs[year]], axis=0)
# -

#Renaming columns
agg_data.reset_index(inplace=True, drop=True)
agg_data.rename(columns={"Station ID":"Station_ID"}, inplace=True)

#Dropping null station names
print(len(agg_data))
agg_data.dropna(subset="Station_ID", inplace=True)
print(len(agg_data))

# +
# print(agg_data.loc[~agg_data["Notes.1"].isna(), "Notes.1"])
# print(agg_data.loc[~agg_data["Notes"].isna(), "Notes"])
# -

agg_data.drop(["Unnamed: " + str(n) for n in np.arange(44, 95)], axis=1, inplace=True)

#Testing to ensure Notes columns are mutually exclusive
print("Double Notes:", len(agg_data.loc[(~agg_data["Notes.1"].isna()) & (~agg_data["Notes"].isna())]))

#Combining Notes into one column
agg_data.loc[~agg_data["Notes.1"].isna(), "Notes"]=agg_data.loc[~agg_data["Notes.1"].isna(), "Notes.1"]
agg_data.drop("Notes.1", axis=1, inplace=True)

#Reading in stations
stations=pd.read_csv(path2)
# pd.unique(stations["Station_ID"])

#Unmatched Stations
mismatches = agg_data.loc[~agg_data["Station_ID"].isin(stations["Station_ID"])]
print("Unmatched Stations:", pd.unique(mismatches["Station_ID"]))

#Seeing if removing 3 letter prefix fixes all mismatches
no_prefix=mismatches["Station_ID"].str[4:]
no_prefix_mismatches= pd.unique(no_prefix.loc[~no_prefix.isin(stations["Station_ID"])])
# print(no_prefix_mismatches)

# 1. All the unmatched stations seem to be a result of three letter prefixes before station name (similar to STS_continuous), which we fix below
#     * Except for Flushing Bay, which is output of cell above
#     * Flushing Bay should be excluded from model in any case (dropped in next cell)

# +
#Defaulting data to station nomenclature
agg_data.loc[~agg_data["Station_ID"].isin(stations["Station_ID"]), "Station_ID"]=no_prefix
print(len(agg_data))

#Removing Flushing Bay data
agg_data=agg_data.loc[~agg_data["Station_ID"].isin(no_prefix_mismatches)].copy()
print(len(agg_data))
# -

#Merging Station Data and Underlying Data
agg_data=agg_data.merge(stations, how="left", on="Station_ID", suffixes=["", "_s"])
# agg_data.head()

#Testing for unmerged data (this output should be empty)
print("Still unmatched:", len(agg_data.loc[agg_data["Embay_Pt"].isna()]))

# Like all other discrete data sources, we drop data outside of a vaudrey embayment

# +
#Dropping Embayment Data
print(len(agg_data))
embaydf=agg_data.loc[agg_data["Embay_Pt"]==1].copy(deep=True)
print(len(embaydf))

#Adding Organization Identifier
embaydf["OrganizationIdentifier"]="CFE-STS"

#Renaming to match WQX format
embaydf.rename(columns={"Station_ID":"MonitoringLocationIdentifier"}, inplace=True)
embaydf["MonitoringLocationIdentifier"]=embaydf["MonitoringLocationIdentifier"].apply(lambda x: "CFE-STS-" + x)
#embaydf.set_index(["Subw_Embay", "OrganizationIdentifier", "MonitoringLocationIdentifier"], inplace=True)
# embaydf.head()

# +
# embays=pd.unique(agg_data["Subw_Embay"])
# embays
# -

embaydf.to_csv("STS_Discrete_11_18_2023.csv")
