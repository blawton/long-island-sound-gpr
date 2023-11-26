import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import scipy.fft as fft
from statsmodels.tsa.seasonal import seasonal_decompose
from scipy.ndimage import median_filter

# +
#Loading paths config.yml
import yaml

with open("../../config.yml", "r") as file:
    config = yaml.load(file, Loader=yaml.FullLoader)

# +
#params

input_file="STS_Continuous_Data_Pre_Processing_11_18_2023.csv"
pre_output_file="Processed_STS_Continuous_Data_11_18_2023.csv"
output_file="Interpolated_STS_Continuous_Data_11_18_2023.csv"

#Despiking params

#Window length of measurements to consider
window=10

#Threshold for difference of measurement above median of the window around a point
thresh=1

#params
years=[2019, 2020, 2021, 2022]
# -

#Global Variables
dep_var="Temperature (C)"

# v2 of this notebook is significantly streamlined, with only the actual processing tasks included

# # STS

#Reading data
sts = pd.read_csv(input_file, index_col=0)
sts

#seeing if there is indeed data to interpolate
sts.loc[pd.to_numeric(sts["Temperature (C)"], errors="coerce").isna()]

help(sts.reindex)


# ## Despiking

#Despiking function for non-endpoints
def despike(df, threshold, window_size):
    filtered=median_filter(df.values, size=window_size)
    despiked=np.where(np.abs(filtered-df.values)>=threshold, filtered, df.values)
    return(pd.DataFrame(despiked, index=df.index))


#Making a dictionary of data from each site
sts_dict={}
for station in pd.unique(sts["Station ID"]):
    working=sts.loc[sts["Station ID"]==station].copy(deep=True)
    working.rename(columns={"Date Time (GMT-04:00)": "Date"}, inplace=True)
    working["Date"]=pd.to_datetime(working["Date"], format='mixed')
    working["Year"]=working["Date"].dt.year
    
    #saving
    sts_dict[station]=working

sts_dict["EAB-I-1B-L"].head()

#Graphing
for index, df in sts_dict.items():
    for year in pd.unique(df["Year"]):
        working=df.loc[df["Year"]==year]
        plt.plot(working["Date"], working[dep_var], c="tab:blue")
    plt.title(index)
    plt.xticks(rotation=90)
    plt.show()

# +
#Eliminating left and right endpoints
despiked={}

#Left (shift is +1 for .isna condition)
for index, df in sts_dict.items():
    
    #Ensuring proper order and integer index
    working=df.sort_values("Date")
    working.reset_index(inplace=True, drop=True)
    
    #Setting left endpoints as np.nan to avoid confusion
    working.loc[(~working[dep_var].isna()) & (working.shift(periods=1)[dep_var].isna())]=np.nan
    
    despiked[index]=working

#Right (shift is -1 for .isna condition)
for index, df in despiked.items():
    
    #Ensuring proper order and integer index
    working=df.sort_values("Date")
    working.reset_index(inplace=True, drop=True)
    
    #Setting left endpoints as np.nan to avoid confusion
    working.loc[(~working[dep_var].isna()) & (working.shift(periods=-1)[dep_var].isna())]=np.nan
    
    despiked[index]=working
# -

#Testing endpoint removal
for index, df in despiked.items():
    for year in pd.unique(df["Year"]):
        working=df.loc[df["Year"]==year]
        plt.plot(working["Date"], working[dep_var], c="tab:blue")
    plt.title(index)
    plt.xticks(rotation=90)
    plt.show()

#Despiking non-endpoints
for index, df in despiked.items():
    
    #Despiking should be done within each year (otherwise median window will be confused)
    agg=pd.DataFrame()
    for year in pd.unique(df["Year"]):
        working=df.loc[df["Year"]==year].copy(deep=True)
        
        #Ensuring proper order
        working.sort_values("Date", inplace=True)
        
        #Setting index as needed for function
        working.set_index(["Station ID", "Date", "Year"], inplace=True, drop=True)
        
        #Despiking
        working[dep_var]=despike(working[dep_var], thresh, window)
        
        #Concatenating
        agg=pd.concat([agg, working], axis=0)
    
    agg.reset_index(inplace=True)
    despiked[index]=agg

#Testing despiking non-endpoints
for index, df in despiked.items():
    for year in pd.unique(df["Year"]):
        working=df.loc[df["Year"]==year]
        plt.plot(working["Date"], working[dep_var], c="tab:blue")
    plt.title(index)
    plt.xticks(rotation=90)
    plt.show()

pre_output=pd.DataFrame()
for index, df in despiked.items():
    df["Station ID"]=index
    pre_output=pd.concat([pre_output, df])
pre_output.to_csv(pre_output_file)

# ## Simple Approach to Interpolation

# In linear interpolation, we first group by date because it makes no sense to run linear interpolation over larger gaps based on diel fluctuations

#Getting pre-interp length (of hourly data)
sum=0
for index, df in despiked.items():
    working=df.copy()
    working["Date"]=pd.to_datetime(working["Date"], errors='coerce').dt.round("H")
    sum+=len(pd.unique(working["Date"]))
print("Pre-interp datapoints:", sum)

# +
#Switching to time by hour

for_interp={}
for index, df in despiked.items():
    agg=pd.DataFrame()
    
    for year in pd.unique(df["Year"]):
        #Selecting year's data
        working=df.loc[df["Year"]==year].copy()
    
        #Rounding to nearest hour
        working["Date"]=pd.to_datetime(working["Date"], errors='coerce').dt.round("H")
    
        #Dropping nas
        working.dropna(subset=["Date"], inplace=True)
    
        #Averaging datapoints
        working=working.groupby("Date")[[dep_var, "Year", "Latitude", "Longitude"]].mean()
        working.reset_index(inplace=True)
    
        #Reindexing to hourly datapoint
        new_index=pd.date_range(working["Date"].min().floor('H'), working["Date"].max().ceil('H'), freq='H')
        working.set_index("Date", inplace=True)
        # pre = len(working)
        working=working.reindex(new_index, fill_value=np.nan)
        working.reset_index(inplace=True, names="Date")
        post=len(working)
        # print(post-pre)
        
        #Aggregating
        agg=pd.concat([agg, working])
    
    print(index, "NA percentage:", len(agg.loc[agg["Temperature (C)"].isna()])/len(agg))
        
    #saving
    for_interp[index]=agg
# -

lin_interpolated={}
#interpolation with rolling averages
for index, df in for_interp.items():

    #this part should be done within each year
    agg=pd.DataFrame()
    for year in pd.unique(df["Date"].dt.year):

        #Interpolating by each hour of the day (inefficient but accurate)
        for time in pd.unique(df["Date"].dt.time):
            working=df.loc[(df["Date"].dt.year==year) & (df["Date"].dt.time==time)].copy(deep=True)
            
            #Interpolating
            working[dep_var]=working[dep_var].interpolate(method='linear')
            working["Latitude"]=working["Latitude"].interpolate(method='linear')
            working["Longitude"]=working["Longitude"].interpolate(method='linear')
            
            #Concatenating
            agg=pd.concat([agg, working], axis=0)

    #Re-sorting and outputting
    lin_interpolated[index]=agg.sort_values("Date")

#Testing interpolation
for index, df in lin_interpolated.items():
    for year in pd.unique(df["Date"].dt.year):
        working=df.loc[df["Date"].dt.year==year]
        plt.plot(working["Date"], working[dep_var], c="tab:blue")
    plt.title(index)
    plt.xticks(rotation=90)
    plt.show()

#Checking new na percentages
for index, df in lin_interpolated.items():
    print(index, "NA percentage:", len(df.loc[df["Temperature (C)"].isna()])/len(df))

# +
#Aggregating and Saving in Data Folder
output=pd.DataFrame()
for index, df in lin_interpolated.items():
    df["Station ID"]=index
    output=pd.concat([output, df])

#Refilling year var of interpolated data
output["Year"]=output["Date"].dt.year

print("Post-interp datapoints:", len(output))
print("Missing datapoints:", len(output.loc[output["Latitude"].isna()]))
output.to_csv(output_file)
# -


