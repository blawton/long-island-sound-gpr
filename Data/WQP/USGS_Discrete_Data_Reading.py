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
#params

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

#Dependent Variable
depvar="Temperature, water"
depvar_unit="deg C"
# -

#Getting filepaths for each year
paths = ["/".join([path1, "resultphyschem"]) + str(year) + ".csv" for year in years]

wqpdf=pd.DataFrame()
for path in paths:
    file= path.rsplit("/")[-1]
    working=pd.read_csv(path)
    wqpdf=pd.concat([wqpdf, working])


# # Merging Non-STS Data

#Reading in stations
wqpstations=pd.read_csv("/".join([path2, "WQPortal_Stations.csv"]))

#Extracting CTDEEP Stations for use elsewhere
wqpstations.loc[wqpstations["OrganizationIdentifier"]=="CT_DEP01_WQX"].to_csv("../CTDEEP/CTDEEP_Stations.csv")

#Merging Station Data and Underlying Data
wqpdf=wqpdf.merge(wqpstations, how="left", on="MonitoringLocationIdentifier", suffixes=["", "_s"])
wqpdf.head()

#Ensuring completeness of the merge (should be empty df)
print("Unmatched Stations:", len(wqpdf.loc[wqpdf["Embay_Pt"].isna()]))

# # Restricting to Embayment Data

#Reindexing based on newly available parameters
embaydf=wqpdf.loc[wqpdf["Embay_Pt"]==1].copy(deep=True)
# embaydf.head()

#Outputting non-STS Data
print(len(embaydf))
embaydf=embaydf.loc[embaydf["OrganizationIdentifier"]!="CFE-STS"].copy()
print(len(embaydf))

print(pd.unique(embaydf["OrganizationIdentifier"]))

#Ensuring all temp is in C (output should be empty)
print("Non metric units:", len(embaydf.loc[(embaydf["CharacteristicName"]==depvar) & (embaydf["ResultMeasure/MeasureUnitCode"]!=depvar_unit)]))

#Restricting to depvar
print(len(embaydf))
embaydf=embaydf.loc[(embaydf["CharacteristicName"]==depvar) & (embaydf["ResultMeasure/MeasureUnitCode"]==depvar_unit)]
print(len(embaydf))

embaydf.to_csv("USGS_Discrete_11_18_2023.csv")
