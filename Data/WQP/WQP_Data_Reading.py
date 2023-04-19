import pandas as pd
from json import dumps
import os
import json
import numpy as np
import matplotlib.pyplot as plt

pd.options.display.max_rows=150
pd.options.display.max_columns=150

# +
#Loading paths config.yml
import yaml

with open("../../config.yml", "r") as file:
    config = yaml.load(file, Loader=yaml.FullLoader)

# +
#Naming convention for WQP Data
filename = "resultphyschem"

#Range of Years
years = np.arange(2010, 2023)

#Data Source for raw data
path1 = config["WQP_Data_Reading_path1"]

#Data Source for stations
path2 = config["WQP_Data_Reading_path2"]

assert(os.path.isdir(path1))
assert(os.path.isdir(path2))
# -

#Getting filepaths for each year
paths = [path1 + "resultphyschem" + str(year) + ".csv" for year in years]

wqpdf=pd.DataFrame()
for path in paths:
    file= path.rsplit("/")[-1]
    working=pd.read_csv(path)
    wqpdf=pd.concat([wqpdf, working])


# # Merging Non-STS Data

#Reading in stations
wqpstations=pd.read_csv(path2 + "WQPortal_Stations.csv")

#Extracting CTDEEP Stations for use elsewhere
wqpstations.loc[wqpstations["OrganizationIdentifier"]=="CT_DEP01_WQX"].to_csv("CTDEEP_Stations.csv")

#Merging Station Data and Underlying Data
wqpdf=wqpdf.merge(wqpstations, how="left", on="MonitoringLocationIdentifier", suffixes=["", "_s"])
wqpdf.head()

#Ensuring completeness of the merge (should be empty df)
wqpdf.loc[wqpdf["Embay_Pt"].isna()]

# # Restricting to Embayment Data

#Reindexing based on newly available parameters
embaydf=wqpdf.loc[wqpdf["Embay_Pt"]==1].copy(deep=True)
embaydf.head()

#Outputting non-STS Data
print(len(embaydf))
embdaydf=embaydf.loc[embaydf["OrganizationIdentifier"]!="CFE-STS"].copy()
print(len(embaydf))

#Ensuring all temp is in C (output should be empty)
embaydf.loc[(embaydf["CharacteristicName"]=="Temperature, water") & (embaydf["ResultMeasure/MeasureUnitCode"]!="deg C")]

print(len(embaydf))
embaydf=embaydf.loc[(embaydf["CharacteristicName"]=="Temperature, water") & (embaydf["ResultMeasure/MeasureUnitCode"]=="deg C")]
print(len(embaydf))

embaydf.to_csv("WQP_merged_no_STS_4_19_2023.csv")