import numpy as np
import pandas as pd

#Function for creating the summer avg requires standardized labeling: "Station ID", "Date", "Temperature (C)"
def summer(df):
    
    #Date handling
    df["Date"]=pd.to_datetime(df["Date"])
    df["Month"]=df["Date"].dt.month
    df["Year"]=df["Date"].dt.year
    df.dropna(subset="Date", inplace=True)

    #Limiting to July and August
    working=df.loc[(df["Month"]==7) | (df["Month"]==8)]
    working=pd.DataFrame(working.groupby(["Station ID",
                                          "Year", "Month"])["Temperature (C)"].mean())
    
    #Only getting years with both months of data
    working.reset_index(inplace=True)
    keep=working.duplicated(subset=["Station ID", "Year"], keep=False)
    working=working.loc[keep].copy(deep=True)
    
    #Meaning July and August
    means=working.groupby(["Station ID", "Year"]).mean()
    means.reset_index(inplace=True)
    
    #Re-adding Lat and Lon from input df
    #Important to merge on both station ID and year because coords can change
    means=means.merge(df[["Station ID", "Year",
                          "Latitude","Longitude"]].drop_duplicates(["Station ID", "Year"]),
                      how="left", on=["Station ID", "Year"])
    
    return(means)

#Similar function but grouped by day in order to perform spatial and time interpolation
def daily(df):
    
    #Date handling
    df["Date"]=pd.to_datetime(df["Date"])
    df["Month"]=df["Date"].dt.month
    df["Year"]=df["Date"].dt.year
    df["Day"]=df["Date"].dt.dayofyear
    df.dropna(subset="Date", inplace=True)

    #Limiting to July and August (for now)
    working=df.loc[(df["Day"]>=152) & (df["Day"]<=274)]
    means=pd.DataFrame(working.groupby(["Station ID",
                                          "Year", "Day"])["Temperature (C)"].mean())
    means.reset_index(inplace=True)

#     #Only getting years with both months of data
#     working.reset_index(inplace=True)
#     keep=working.duplicated(subset=["Station ID", "Year"], keep=False)
#     working=working.loc[keep].copy(deep=True)

#     #Meaning July and August
#     means=working.groupby(["Station ID", "Year"]).mean()
#     means.reset_index(inplace=True)

    #Re-adding Lat and Lon from input df
    #Important to merge on both station ID and year because coords can change
    means=means.merge(df[["Station ID", "Year",
                          "Latitude","Longitude"]].drop_duplicates(["Station ID", "Year"]),
                      how="left", on=["Station ID", "Year"])
    
    return(means)
