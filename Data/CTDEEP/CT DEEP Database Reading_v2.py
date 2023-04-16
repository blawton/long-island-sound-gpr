import pandas as pd
#import shapefile
from json import dumps
import os
import json
import numpy as np

pd.options.display.max_rows=150
pd.options.display.max_columns=150

# Note that cruise names only go up to cruises HYAUG22 and WQAUG22 although part of the first ''SEP22 cruise was on the last day of august. Including this September Cruise and then filtering for august would give stations uneven seasonal distribution so it is intentionally excluded.
#
# The capitalization issue resulting in "WQJAN19" not merging with an associated date was fixed.

# +
#Loading paths config.yml
import yaml

with open("../../config.yml", "r") as file:
    config = yaml.load(file, Loader=yaml.FullLoader)

# +
paths={}
#Old version of database
paths[1]=config["CTDEEP_Reading_path1"]
#New additions to database
paths[2]=config["CTDEEP_Reading_path2"]
#Field Data Sheet
paths[3]=config["CTDEEP_Reading_path3"]
#Times Cruises were on Station through Aug 2022
paths[4]=config["CTDEEP_Reading_path4"]

for i in np.arange(1, 5):
    assert os.path.exists(paths[i])

# +
#Reading in data
data=pd.read_csv(paths[1])
data_add=pd.read_csv(paths[2])
field=pd.read_csv(paths[3])
cruise_times=pd.read_csv(paths[4])

data.loc[data["cruise name"]=="WQJUL21"].head()
# -

print(len(data))

#Refotmatting data addendum
data_add.rename(columns={"Cruise":"cruise name"}, inplace=True)
pd.unique(data_add["station name"])

# # Getting dates for data

cruise_times

#Testing to ensure all station names are included in cruise times (this output should be empty)
mismatch1 = list(set(pd.unique(data["station name"])).difference(set(pd.unique(cruise_times["Station Name"]))))
mismatch1

#Testing to ensure all addendum station names are included in cruise times (should also be empty)
mismatch2 = list(set(pd.unique(data_add["station name"])).difference(set(pd.unique(cruise_times["Station Name"]))))
mismatch2

#Getting overlap between cruises in two datasets (should be empty)
cruises1=pd.unique(data["cruise name"])
cruises2=pd.unique(data_add["cruise name"])
list(set(cruises1).intersection(set(cruises2)))

#Getting non-overlap between stations in two datasets
difference = list(set(pd.unique(data["station name"])).symmetric_difference(set(pd.unique(data_add["station name"]))))
difference

#Getting non-overlap between stations in data and cruise times
difference1 = list(set(pd.unique(data["station name"])).symmetric_difference(set(pd.unique(cruise_times["Station Name"]))))
difference1

#Getting non-overlap between stations in data and cruise times
difference2 = list(set(pd.unique(data_add["station name"])).symmetric_difference(set(pd.unique(cruise_times["Station Name"]))))
difference2

#For consistent dates/times, we take all from the cruise times sheet
data_add.drop("Date", axis=1, inplace=True)
data_add

# # Merging and Outputting

#Test to ensure that the relevant columns from data and data_add have the same names
print(data.columns)
print(data_add.columns)

#Merging old and new
all_data=pd.concat([data, data_add], axis=0)

#Capitalizing Station names to avoid errors
all_data["station name"]=all_data["station name"].str.upper()
print(pd.unique(all_data["station name"]))
cruise_times["Station Name"]=cruise_times["Station Name"].str.upper()
print(pd.unique(cruise_times["Station Name"]))

# +
#Adding dates and times (REMEMBER TO MERGE BY CRUISE AND STATION - THIS WAS DONE WRONG BEFORE)

#Length should be the same before and after merge
print(len(all_data))
dated=all_data.merge(cruise_times, how="left", left_on=["cruise name", "station name"], right_on=["Cruise", "Station Name"])
print(len(dated))

dated.head()
# -

#Testing to make sure everything after 2010 but before fall '22 is present (or Trawl)
undated=pd.unique(dated.loc[dated["Cruise"].isna(), "cruise name"])
print([cruise for cruise in undated if (cruise[-2]=="1" or cruise[-2]=="2")])

#Filtering undated (unmerged) entries
print(len(dated))
dated.dropna(subset=["Cruise"], inplace=True)
print(len(dated))

#Removing duplicate columns
dated.drop(["Cruise", "Station Name"], axis=1, inplace=True)

# +
#dated.to_csv("CT_DEEP_Post_2010_1.12.2023.csv")
# -


