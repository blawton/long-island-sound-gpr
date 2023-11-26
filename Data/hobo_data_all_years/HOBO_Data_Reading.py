import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import ttest_ind
import scipy.stats as stats
import math
import os
from scipy.ndimage import median_filter

# The rest of the HOBO Logger Data is Aggregated with this Data in the notebook "Open Sound Embayment Temp Comparison- [version].ipynb" (this should be changed)

# +
#Loading paths config.yml
import yaml

with open("../../config.yml", "r") as file:
    config = yaml.load(file, Loader=yaml.FullLoader)

# +
#Loading coords

with open("../../coords.yml", "r") as file:
    yaml_coords = yaml.load(file, Loader=yaml.FullLoader)

# +
#params

#Fishers Island Data
paths={}
paths[1]=config["Fishers_Island_Reading_path"] + "/"

#Beebe Cove
paths[2]= config["Comparison_path4"]

#Mumford Cove
paths[3]=config["Comparison_path5"]

#Dictionary of sites mapped to their numbers
site_dict=dict(zip(np.arange(1,5), ["West Harbor", "Barleyfield Cove", "Hay Harbor", "East Beach"]))

#Sites and time window (excluding 2022)
years = np.arange(2019, 2023)
site_dict=dict(zip(np.arange(1,5), ["West Harbor", "Barleyfield Cove", "Hay Harbor", "East Beach"]))

strings={}
#Beebe Sensor
strings[1]= config["Comparison_string1"]

#Mumford Sensor
strings[2]= config["Comparison_string2"]

#Testing paths
for path in paths.values():
    assert(os.path.exists(path))

dep_var="temp"

#Despiking params
thresh=3
window=24

#output
output_file="HOBO_Data_Agg_11_18_2023.csv"
# -

# # Fishers Island

#Accomodates the different formats in each year of source data
site_data={}
for i in np.arange(1, 5):
    site_data[i]=pd.DataFrame()
for year in years:
    print(year)
    print(os.listdir(paths[1]+str(year)))
    for i in np.arange(1, 5):
        try:
            #We translate names to site numbers programmatically to avoid human error
            if (year < 2020):
                working =pd.read_csv(paths[1] + str(year) + '/Fishers_Island_'+ str(i) + '.csv')
            elif (year >= 2020):
                working = pd.read_csv(paths[1] + str(year) + '/' + site_dict[i].replace(" ", "").lower() + str(year) + ".csv")
            working['Year']=year
            
            #Reading in data using a different format depending on year
            if (year < 2018):
                working.drop(working.index[0], axis=0, inplace=True)
                working.drop(working.columns[0], axis=1, inplace=True)
                working.rename(columns={working.columns[0]:"date", working.columns[1]:"temp"}, inplace=True)
                try:
                    working["date"]=pd.to_datetime(working["date"],  format='%d/%m/%Y')
                except:
                    pass
                try:
                    working["date"]=pd.to_datetime(working["date"])   
                except:
                    pass
                
                working["date"]=working["date"].apply(lambda dt: dt.replace(year=year))

            elif (year <2020):
                working.drop(0, axis=0, inplace=True)
                working.drop(working.columns[0], axis=1, inplace=True)
                working.rename(columns={working.columns[0]:"date", working.columns[1]:"temp"}, inplace=True)
                working["date"]=pd.to_datetime(working["date"])            
                
                working["date"]==working["date"].apply(lambda dt: dt.replace(year=year))
                
            else:
                working.rename(columns={working.columns[1]:"date", working.columns[2]:"time", working.columns[3]:"temp"}, inplace=True)
                working.drop(working.index[:3], axis=0, inplace=True)
                try: 
                    working["date"]=pd.to_datetime(working["date"] + " " + working["time"])
                except:
                    print("date read error for site " +str(i))
                
                #Dropping original time column to make format uniform
                working.drop("time", axis=1, inplace=True)
                
            site_data[i]=pd.concat([site_data[i], working], ignore_index=True)
            
        except:
            print("No data for site " + str(i))

#Getting rid of nas
for site in np.arange(1, 5):
    site_data[site]["temp"]=pd.to_numeric(site_data[site]["temp"], errors='coerce')
    site_data[site].dropna(subset="temp", inplace=True)

# Converting all temps to celcius
for site in np.arange(1,5):
    working = site_data[site]
    working.loc[working["Year"]<2020,"temp"]=(working.loc[working["Year"]<2020, "temp"] - 32)*(5/9)

#Ensure that times are not neccesary because there are hourly records for each day
for site in np.arange(1, 5):
    working = site_data[site]
    counts=working.groupby("date")["temp"].count()
    print("Entries with a time problem:", counts.loc[(counts!=24) & (~counts.index.astype(str).str.contains(":"))])

#Making keys FISM specific
keys=list(site_data.keys())
print(keys)
for i in keys:
    site_data["FISM_" + str(i)]=site_data[i]
    site_data.pop(i)
print(site_data.keys())

# # Mumford and Beebe Cove

# +
#reading in mumford and beebe in from files
beebe=pd.read_csv(paths[2], header=1, index_col=0)
mum=pd.read_csv(paths[3], header=1, index_col=0)

beebe.rename(columns={strings[1]: "temp", "Date Time, GMT-04:00":"Date"}, inplace=True)
mum.rename(columns={strings[2]: "temp", "Date Time, GMT-04:00":"Date"}, inplace=True)

beebe.Date=pd.to_datetime(beebe.Date)
mum.Date=pd.to_datetime(mum.Date)

#No more converting to Fahrenheight
#beebe.temp=beebe.temp*(9/5) + 32
#mum.temp=mum.temp*(9/5) + 32

beebe.drop(beebe.columns[2:], axis=1, inplace=True)
mum.drop(mum.columns[2:], axis=1, inplace=True)
beebe.dropna(subset=["temp"], inplace=True)
mum.dropna(subset=["temp"], inplace=True)

beebe.head()
# -

#Adding mumford and beebe (keep in mind these are unprocessed)
hobo_data={}
hobo_data["Mumford Cove"]=mum
hobo_data["Beebe Cove"]=beebe

# # Adding Coords and Aggregating

#Reading in GPS coords from yaml (compared to original output as check)
coords={}
for key, value in yaml_coords.items():
    coords[key.replace("_", " ")]= tuple(value)
coords

# +
#Adding coords and aggregating data
agg=pd.DataFrame()

for key, df in hobo_data.items():
    assert(list(df.columns).count("temp")==1)
    assert(list(df.columns).count("Date")==1)
    df["Latitude"]= coords[key][0]
    df["Longitude"]= coords[key][1]
    df["Station ID"]= key
    df["Year"]=df["Date"].dt.year
    agg=pd.concat([agg, df])

# +
#Adding coords to FISM data and aggregating
fism_agg=pd.DataFrame()

for site_no, site_name in site_dict.items():
    df = site_data["FISM_" + str(site_no)]
    df["Latitude"]=coords[site_name][0]
    df["Longitude"]=coords[site_name][1]
    df["Station ID"]=site_name
    fism_agg=pd.concat([fism_agg, df])

fism_agg.rename(columns={"date":"Date"}, inplace=True)
fism_agg=fism_agg[["Date", "temp", "Year", "Latitude", "Longitude", "Station ID"]]
# -

agg=pd.concat([fism_agg, agg])
agg.head()


# # Cleaning

#Despiking function
def despike(df, threshold, window_size):
    filtered=median_filter(df.values, size=window_size)
    despiked=np.where(np.abs(filtered-df.values)>=threshold, filtered, df.values)
    return(pd.DataFrame(despiked, index=df.index))


# +
output=pd.DataFrame()

for station in pd.unique(agg["Station ID"]):
    # print(station)
    df=agg.loc[agg["Station ID"]==station]
    for year in pd.unique(df["Year"]):
        # print(year)
        working=df.loc[df["Year"]==year].copy()
        working.sort_values(by=["Date"], inplace=True)

        #Despiking
        working[dep_var]=despike(working[dep_var], thresh, window)

        #No interpolation of data here
        # print(working[dep_var].count()/len(working))
        # working[dep_var]=working[dep_var].interpolate(method='linear', limit_area='inside')
        # print(working[dep_var].count()/len(working), "(post correction)")
        
        output=pd.concat([output, working])

# -

# # Outputting

#Summary
print(output.groupby(["Station ID", "Year"]).count())

# +
#Outputting HOBO Logger Data to Data Folder

output.to_csv(output_file)
# -

# # Graphing to check work

for station in np.unique(agg["Station ID"]):

    #Pre-despiking
    working=agg.loc[agg["Station ID"]==station]
    plt.plot(working["Date"], working["temp"])
    plt.title(station + " pre")
    plt.show()

    #post-despiking
    working=output.loc[output["Station ID"]==station]
    plt.plot(working["Date"], working["temp"])
    plt.title(station + " post")
    plt.show()
