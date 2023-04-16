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

input_file=config["STS_Continuous_Processing_input"]
output_file=config["STS_Continuous_Processing_output"]

assert(os.path.exists(input_file))
assert(os.path.exists(output_file))
# -

# v2 of this notebook is significantly streamlined, with only the actual processing tasks included

# # STS

#Reading data
sts = pd.read_csv(input_file, index_col=0)
sts

# ## Despiking

# +
#Despiking Parameters

#Window length of measurements to consider
window=5

#Threshold for difference of measurement above median of the window around a point
#Above Threshold, measurement will be replaced with median of the window
thresh=.5


# -

#Despiking function for non-endpoints
def despike(df, threshold, window_size):
    filtered=median_filter(df.values, size=window_size)
    despiked=np.where(np.abs(filtered-df.values)>=threshold, filtered, df.values)
    return(pd.DataFrame(despiked, index=df.index))


#Making a dictionary of data from each site
sts_dict={}
for station in pd.unique(sts["Station ID"]):
    working=sts.loc[sts["Station ID"]==station].copy(deep=True)
    working.rename(columns={"Date Time (GMT-04:00)": "Date", "Temperature (C)":"temp"}, inplace=True)
    working["Date"]=pd.to_datetime(working["Date"])
    working["Year"]=working["Date"].dt.year
    
    #saving
    sts_dict[station]=working

#Graphing
for index, df in sts_dict.items():
    for year in pd.unique(df["Year"]):
        working=df.loc[df["Year"]==year]
        plt.plot(working["Date"], working["temp"], c="tab:blue")
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
    working.loc[(~working["temp"].isna()) & (working.shift(periods=1)["temp"].isna())]=np.nan
    
    despiked[index]=working

#Right (shift is -1 for .isna condition)
for index, df in despiked.items():
    
    #Ensuring proper order and integer index
    working=df.sort_values("Date")
    working.reset_index(inplace=True, drop=True)
    
    #Setting left endpoints as np.nan to avoid confusion
    working.loc[(~working["temp"].isna()) & (working.shift(periods=-1)["temp"].isna())]=np.nan
    
    despiked[index]=working
# -

#Testing endpoint removal
for index, df in despiked.items():
    for year in pd.unique(df["Year"]):
        working=df.loc[df["Year"]==year]
        plt.plot(working["Date"], working["temp"], c="tab:blue")
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
        working["temp"]=despike(working["temp"], thresh, window)
        
        #Concatenating
        agg=pd.concat([agg, working], axis=0)
    
    agg.reset_index(inplace=True)
    despiked[index]=agg

#Testing despiking
for index, df in despiked.items():
    for year in pd.unique(df["Year"]):
        working=df.loc[df["Year"]==year]
        plt.plot(working["Date"], working["temp"], c="tab:blue")
    plt.title(index)
    plt.xticks(rotation=90)
    plt.show()

# ## Simple Approach to Interpolation

# In linear interpolation, we first group by date because it makes no sense to run linear interpolation over larger gaps based on diel fluctuations

#Grouping by date (no time)
for_interp={}
for index, df in despiked.items():
    working=df.copy(deep=True)
    working["Date"]=working["Date"].dt.date
    working=working.groupby("Date").mean()
    working.reset_index(inplace=True)
    
    #saving
    for_interp[index]=working

lin_interpolated={}
#interpolation with rolling averages
for index, df in for_interp.items():

    #this part should be done within each year
    agg=pd.DataFrame()
    for year in pd.unique(df["Year"]):
        working=df.loc[df["Year"]==year].copy(deep=True)
        
        #Interpolating
        working["temp"]=working["temp"].interpolate(method='linear')
        
        #Concatenating
        agg=pd.concat([agg, working], axis=0)
        
    lin_interpolated[index]=agg

#Testing interpolation
for index, df in lin_interpolated.items():
    for year in pd.unique(df["Year"]):
        working=df.loc[df["Year"]==year]
        plt.plot(working["Date"], working["temp"], c="tab:blue")
    plt.title(index)
    plt.xticks(rotation=90)
    plt.show()

#Ensuring that (most) data has a datapoint for each day in August and July
missing_data=[]
for index, df in lin_interpolated.items():
    for year in pd.unique(df["Year"]):
        working=df.loc[df["Year"]==year].copy(deep=True)
        working["Month"]=pd.to_datetime(working["Date"]).dt.month
        if len(working.loc[(working["temp"].isna()) & ((working["Month"]==7) | (working["Month"]==8))])>0:
            missing_data.append((year, index))
print(missing_data)

#Aggregating and Saving in Data Folder
output=pd.DataFrame()
for index, df in lin_interpolated.items():
    df["Station ID"]=index
    output=pd.concat([output, df])
output.to_csv(output_file)
