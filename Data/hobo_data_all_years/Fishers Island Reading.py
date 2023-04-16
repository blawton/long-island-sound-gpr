import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import ttest_ind
import scipy.stats as stats
import math
import os

# The rest of the HOBO Logger Data is Aggregated with this Data in the notebook "Open Sound Embayment Temp Comparison- [version].ipynb" (this should be changed)

# +
#Loading paths config.yml
import yaml

with open("../../config.yml", "r") as file:
    config = yaml.load(file, Loader=yaml.FullLoader)

# +
#Directory where data is sourced
path=config["Fishers_Island_Reading_path"]

#Directory where aggregated data is to be stored
dest=config["Fishers_Island_Reading_dest"]

#Dictionary of sites mapped to their numbers
site_dict=dict(zip(np.arange(1,5), ["West Harbor", "Barleyfield Cove", "Hay Harbor", "East Beach"]))

#Testing paths
assert(os.path.isdir(path))
assert(os.path.isdir(dest))
# -

#Sites and time window (excluding 2022)
years = np.arange(2015, 2022)
site_dict=dict(zip(np.arange(1,5), ["West Harbor", "Barleyfield Cove", "Hay Harbor", "East Beach"]))

# # Preparing Data

# +
#Lowering names of new directories
yearpath = path + "2020/"
for file in os.listdir(path + "2020"):
    #print(file)
    #print(yearpath + file.lower())
    os.rename(yearpath + file, yearpath + file.lower())
    
yearpath = path + "2021/"
for file in os.listdir(path + "2021"):
    #print(file)
    #print(yearpath + file.lower())
    os.rename(yearpath + file, yearpath + file.lower())
# -

# # Loading Data

# +
#Accomodates the different formats in each year of source data
site_data={}
for i in np.arange(1, 5):
    site_data[i]=pd.DataFrame()
    for year in years:
        print (year)
        try:
            #We translate names to site numbers programmatically to avoid human error
            if (year < 2020):
                working =pd.read_csv(path + str(year) + '/Fishers_Island_'+ str(i) + '.csv')
            elif (year >= 2020):
                working = pd.read_csv(path + str(year) + '/' + site_dict[i].replace(" ", "").lower() + str(year) + ".csv")
            working['Year']=year
            
            if (year < 2018):
                working.drop(working.index[0], axis=0, inplace=True)
                working.drop(working.columns[0], axis=1, inplace=True)
                working.rename(columns={working.columns[0]:"date", working.columns[1]:"temp"}, inplace=True)
                try:
                    working["date"]=pd.to_datetime(working["date"],  format='%d/%m/%Y')
                except:
                    pass
                try:
                    working["date"]=pd.to_datetime(working["date"], infer_datetime_format=True)   
                except:
                    pass
                
                working["date"]=working["date"].apply(lambda dt: dt.replace(year=year))

            elif (year <2020):
                working.drop(0, axis=0, inplace=True)
                working.drop(working.columns[0], axis=1, inplace=True)
                working.rename(columns={working.columns[0]:"date", working.columns[1]:"temp"}, inplace=True)
                working["date"]=pd.to_datetime(working["date"], infer_datetime_format=True)            
                
                working["date"]==working["date"].apply(lambda dt: dt.replace(year=year))
                
            else:
                working.rename(columns={working.columns[1]:"date", working.columns[2]:"time", working.columns[3]:"temp"}, inplace=True)
                working.drop("time", axis=1, inplace=True)
                working.drop(working.index[:3], axis=0, inplace=True)
                try: 
                    working['date']=pd.to_datetime(working["date"], format='%-d/%m/%Y') 
                except:
                    working["date"]=pd.to_datetime(working["date"],  infer_datetime_format=True)
                
                #working["date"]=working["date"].apply(lambda dt: dt.replace(year=year))
                
            site_data[i]=pd.concat([site_data[i], working], ignore_index=True)
        except:
            print("No data for site " + str(i))

site_data[3].head()
# -

print(site_data[4].loc[site_data[4]["Year"]==2020])

# +
#Getting rid of nas

for site in np.arange(1, 5):
    site_data[site]["temp"]=pd.to_numeric(site_data[site]["temp"], errors='coerce')
    site_data[site].dropna(subset="temp", inplace=True)
# -

# Converting all temps to celcius
for site in np.arange(1,5):
    working = site_data[site]
    working.loc[working["Year"]<2020,"temp"]=(working.loc[working["Year"]<2020, "temp"] - 32)*(5/9)

#Ensure that times are not neccesary because there are hourly records for each day
for site in np.arange(1, 5):
    working = site_data[site]
    counts=working.groupby("date")["temp"].count()
    print(counts.loc[(counts!=24) & (~counts.index.astype(str).str.contains(":"))])

#Outputting data
for site in np.arange(1, 5):
    site_data[site].to_csv(dest + site_dict[site] +".csv")

# # Means

# +
all_data=pd.DataFrame()
for site in np.arange(1,5):
    working=site_data[site][["temp", "Year"]].copy(deep=True)
    working["Site"]=site
    all_data=pd.concat([all_data, working])
    
all_data.head()
# -

all_data.groupby(["Year", "Site"]).mean().unstack(level=1)

means = all_data.groupby(["Year", "Site"]).mean().unstack(level=1)
means.to_csv("../../Charts/means_within_year.csv")

# # Variance

all_data.groupby(["Year", "Site"]).var().unstack(level=1)

variances = all_data.groupby(["Year", "Site"]).var().unstack(level=1)
variances.to_csv("../../Charts/variance_within_year.csv")

# # Histograms and T-Tests:

# +
#Setting figure size

plt.rcParams["figure.figsize"]=(21, 14)

# +
#Getting date distributions

seasons = dict(zip(np.arange(1,13), ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"]))
print(seasons.values())
for site in np.arange(1, 5):
    fig, ax = plt.subplots()
    site_data[site]["date"].dt.month.hist(bins=20)
    ax.set_xticks(list(seasons.keys()))
    ax.set_xticklabels(list(seasons.values()))
    plt.show()
# -

# # Regular Trends

for site in np.arange(1, 5):
    fig, ax =plt.subplots()
    plot_data=site_data[site]
    plt.scatter(plot_data["date"], plot_data["temp"])
    plot_data = plot_data.groupby("Year").mean().reset_index()
    plot_data["Year"]=pd.to_datetime(plot_data["Year"], format="%Y")
    plt.plot(plot_data["Year"], plot_data["temp"], color="tab:orange")
    ax.set_title(site_dict[site]+ " Temp Trend Over Time", size=18)
    ax.set_xlabel("Date", size=16)
    ax.set_ylabel("Water Temp (Degrees Fahrenheight)", size=16)
    plt.savefig("Graphs/"+site_dict[site] + "_trend.png")
    plt.show()




