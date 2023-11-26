import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import regex as re
from scipy.ndimage import median_filter

# +
pd.set_option('display.max_rows', 200)
pd.set_option('display.max_columns', 500)

from IPython.core.display import display, HTML
display(HTML("<style>.container { width:100% ! important; }<style>"))
plt.rcParams["figure.figsize"]=(21, 10)

# +
#params

#Loading station dict
station_file="USGS_Cont_ES_Stations_11_18_2023.csv"
input_file="USGS_Cont_ES_Pre_Processing_11_18_2023.csv"
pre_output_file="USGS_Cont_ES_Processed_11_18_2023.csv"
output_file="USGS_Cont_ES_Interpolated_11_18_2023.csv"

#Regex for variables

#dep_var
dep_var="Temperature.*$(?<!([Tt][Oo][Pp]|[Bb][Oo][Tt][Tt][Oo][Mm]))"
alt_dep_var="Temperature.*$(?<=[Bb][Oo][Tt][Tt][Oo][Mm])"
new_dep_var="Temperature"

#Despiking params
thresh=2
window=20


# -

#Despiking function
def despike(df, threshold, window_size):
    filtered=median_filter(df.values, size=window_size)
    despiked=np.where(np.abs(filtered-df.values)>=threshold, filtered, df.values)
    return(pd.DataFrame(despiked, index=df.index))


# +
#Reading
input=pd.read_csv(input_file, index_col=0)
print(len(input))

stations=pd.read_csv(station_file, index_col=0)
station_dict={row.iloc[0]:row.iloc[1] for _, row in stations.iterrows()}
print(station_dict)

# +
#Quick Pre-Processing

#Dropping qualification code
input.drop([column for column in input.columns if column[-3:]=="_cd"], axis=1, inplace=True)

#Ensuring numeric
for col in [col for col in input.columns.difference(["datetime", "site_no"])]:
    input[col]=pd.to_numeric(input[col], errors="coerce")

#Ensuring all stations have coordinates
print(len(input))
input.dropna(subset=["Latitude"], inplace=True)
print(len(input))

# +
print(input.columns)

#Fixing typos (overlap should be 0)
print(len(input.loc[(~input["Temperature_Bottom"].isna()) & (~input["Temperature_Botttom"].isna())]))
input.loc[~input["Temperature_Botttom"].isna(), "Temperature_Bottom"]= input.loc[~input["Temperature_Botttom"].isna(), "Temperature_Botttom"]
input.drop(["Temperature_Botttom"], axis=1, inplace=True)
print(input.columns)

# +
#Seperating by variables
dep_vars=[col for col in input.columns if re.match(dep_var, col) is not None]
alt_dep_vars=[col for col in input.columns if re.match(alt_dep_var, col) is not None]

var_priority=dep_vars+alt_dep_vars
print(var_priority)

# +
#Subbing in alt vars when main vars are na

#Only temp data
input_data=input[dep_vars+alt_dep_vars]
print(len(input_data.loc[input_data["Temperature"].isna()]))
input_data.fillna(method="bfill", axis=1, inplace=True)
print(len(input_data.loc[input_data["Temperature"].isna()]))

#Subbing temp data back in
input["Temperature"]=input_data["Temperature"]
input.drop(input.columns[6:], axis=1, inplace=True)
input.head()
# -

#Dealing with dates
input["datetime"]=pd.to_datetime(input["datetime"])
input["Year"]=input["datetime"].dt.year

#Splitting into sites
dfs={}
agg_l=0
for site_no, site in station_dict.items():
    length=len(input.loc[(input["site_no"]==site_no) & ~(input["Temperature"].isna())])
    if length>0:
        dfs[site]=input.loc[input["site_no"]==site_no]
        agg_l+=len(dfs[site])
#Should roughly match above
print(agg_l)

# # Processing

# +
#Despiking and pre-output
processed={}
pre_output=pd.DataFrame()

for site, df in dfs.items():
    
    processed[site]=pd.DataFrame()
    for year in pd.unique(df["Year"]):
        working=df.loc[df["Year"]==year].copy()
        working.sort_values(by=["datetime"], inplace=True)

        #Despiking
        working["Temperature"]=despike(working["Temperature"], thresh, window)

        #Saving
        processed[site]=pd.concat([processed[site], working])
        pre_output=pd.concat([pre_output, working])
        
print("Datapoints:", pre_output[new_dep_var].count())
pre_output.dropna(subset=[new_dep_var], inplace=True)
pre_output.to_csv(pre_output_file)

# +
#Linear interpolation

output=pd.DataFrame()
for site, df in processed.items():
    for year in pd.unique(df["Year"]):

        #Selecting data
        working=df.loc[df["Year"]==year].copy()
        working.sort_values(by=["datetime"], inplace=True)

        #Running linear interpolation
        print(len(working.loc[working[new_dep_var].isna()])/len(working))
        working[new_dep_var]=working[new_dep_var].interpolate(method='linear', limit_area='inside')
        print(len(working.loc[working[new_dep_var].isna()])/len(working), "(post correction)")
        output=pd.concat([output, working])

print("Datapoints:", output[new_dep_var].count())
output.to_csv(output_file)
# -

# Spaces match the .tsv file, so the data was correctly parsed, the percentage of nas is small wherever there is data

output.head()

# # Yearly means by site (ignoring provisional/non-provisional)

counts=pd.DataFrame()
for site, df in dfs.items():
    counts=pd.concat([counts, df.groupby(["site_no", "Year"]).count()])
counts

# # Graphs

for site_no, station_name in station_dict.items():
    working=output.loc[output["site_no"]==site_no].copy(deep=True)
    print(len(working.loc[working["Temperature"].isna()]))
    sc=[col for col in working.columns if "Specific" in col]
    x=working["datetime"]
    y=working.drop(["site_no", "Year", "datetime"]+sc, axis=1)
    plt.plot(x, y, label=y.columns)
    plt.title(station_name)
    plt.xticks(rotation=90)
    plt.legend(loc="upper right", bbox_to_anchor=(1.25, 1))
    plt.show()



