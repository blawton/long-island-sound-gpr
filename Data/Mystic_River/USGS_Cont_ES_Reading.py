import numpy as np
import pandas as pd
import requests
from io import StringIO
from IPython import display
import time
import os
import requests
import regex as re

# +
pd.set_option('display.max_rows', 200)
pd.set_option('display.max_columns', 500)
pd.set_option('display.max_colwidth', None)

from IPython.core.display import display, HTML
display(HTML("<style>.container { width:100% ! important; }<style>"))

# +
#Loading paths config.yml
import yaml

with open("../../config.yml", "r") as file:
    config = yaml.load(file, Loader=yaml.FullLoader)
# -
#Local Data Source
output_file = "USGS_Cont_ES_Pre_Processing_11_18_2023.csv"
stations_output = "USGS_Cont_ES_Stations_11_18_2023.csv"
#This url gets temperature for the entire Eastern Sound Area
url="https://nwis.waterdata.usgs.gov/usa/nwis/uv/?referred_module=qw&nw_longitude_va=-72.59&nw_latitude_va=41.54&se_longitude_va=-71.81&se_latitude_va=40.97&coordinate_format=decimal_degrees&index_pmcode_00010=1&group_key=NONE&sitefile_output_format=html_table&column_name=agency_cd&column_name=site_no&column_name=station_nm&range_selection=date_range&begin_date=2019-01-01&end_date=2023-11-19&format=rdb&date_format=YYYY-MM-DD&rdb_compression=value&list_of_search_criteria=lat_long_bounding_box%2Crealtime_parameter_selection"


# +
#Reading from url

f=requests.get(url)
text=f.text


# +
#Getting stations
def get_stations(all_text):
    i=0
    prev=0
    station_end_line=False
    lines=[]
    station_dict={}
    
    while not station_end_line:
        while all_text[i]!="\n":
            i+=1
        line=all_text[prev:i]
        i+=1
        prev=i
        if re.match("\# *USGS", line) is not None:
            lines.append(line)
        elif len(lines)!=0:
            station_end_line=True
            
    for line in lines:
        station_dict[re.search("USGS (\d+) (\w+)", line).group(1)]= re.search("USGS (\d+) ([\w ]+)", line).group(2)
    return(station_dict)
    
#Testing
get_stations(text[0:5000])
# -

#Getting Stations
station_dict=get_stations(text)


# # Parsing Downloaded File of Water Data

# Data from https://waterdata.usgs.gov/nwis/uv/?referred_module=qw

#rdb reader from github further modified for LARGE usgs file types (adapted from https://github.com/mroberge/hydrofunctions)
def read_rdb(text):
    """Read strings that are in rdb format.

    Args:
        text (str):
            A long string containing the contents of a rdb file. A common way
            to obtain these would be from the .text property of a requests
            response, as in the example usage below.

    Returns:
        header (multi-line string):
            Every commented line at the top of the rdb file is marked with a
            '#' symbol. Each of these lines is stored in this output.
        outputDF (pandas.DataFrame):
            A dataframe containing the information in the rdb file. `site_no`
            and `parameter_cd` are interpreted as a string, but every other number
            is interpreted as a float or int; missing values as an np.nan;
            strings for everything else.
        columns (list of strings):
            The column names, taken from the rdb header row.
        dtypes (list of strings):
            The second header row from the rdb file. These mostly tell the
            column width, and typically record everything as string data ('s')
            type. The exception to this are dates, which are listed with a 'd'.
    """
    outputs=[]
    parameter_lists = []
    try:
        parameters= []
        datalines = []
        count = 0
        for line in text.splitlines():
            if (line[0] != '#') and (count == 0):
                columns = line.split()
                count +=1
            elif (line[0] != '#') and (count != 0):
                datalines.append(line.split('\t'))
                count += 1
            elif (line[0] == '#') and (count != 0):
                outputs.append(pd.DataFrame(datalines, columns=columns))
                if len(parameters)>0:
                    parameter_lists.append(pd.DataFrame(parameters[1:], columns=parameters[0]))
                    parameters = []
                datalines = []
                count = 0 
            elif (line[0] == '#') and (count == 0):
                if (len(line.split('     '))>3):
                    parameters.append(line.split('     ')[1:])
                    
        outputs.append(pd.DataFrame(datalines, columns=columns))
        if len(parameters)>0:
            parameter_lists.append(pd.DataFrame(parameters[1:], columns=parameters[0]))
    except:
        print(datalines[-1])
        print(columns)
        print(parameters)
        print(
            "There appears to be an error processing the file that the USGS returned."
        )
        raise
    
    return parameter_lists, outputs


#Tesing parser on first 5000 lines
test_pms, test_dfs = read_rdb(text[0:5000])
# print(test_pms)
# print(test_dfs)

# # Reading and Processing

pms, dfs = read_rdb(text)
dfs

dfs[2].head()

pms[2]

print(len(dfs))

#Quick Processing
for df in dfs:
    df.fillna(value= np.nan, inplace=True)
    df.drop(0, axis=0, inplace=True)
for pm in pms:
    pm[pm.columns[1]] = pm[pm.columns[1]].str.lstrip(" ")
    pm["Param"]=pm[pm.columns[0]] + "_" + pm[pm.columns[1]]
    pm["Param"]=pm["Param"].str.strip(" ")
print(pms[0])
dfs[0].head()

# +
#Replacing param with beginning of description
for i, df in enumerate(dfs):
    if not len(pms[i].loc[pms[i]["Description"].str.split(", ", expand=True)[0].duplicated()>0]):
        for _ , row in pms[i].iterrows():
            df.columns=[column.replace(str(row.loc["Param"]), row.loc['Description'].split(', ')[0]) for column in df.columns]
    else:
        for _ , row in pms[i].iterrows():
            df.columns=[column.replace(str(row.loc["Param"]), row.loc['Description'].split(', ')[0] + "_" + row.loc['Description'].split(', ')[-1]) for column in df.columns]
        
print(dfs[2].columns)


# -

# # Coords

#rdb reader from hydrofunctions for usgs file types (adapted from https://github.com/mroberge/hydrofunctions)
def read_rdb_2(text):
    """Read strings that are in rdb format.

    Args:
        text (str):
            A long string containing the contents of a rdb file. A common way
            to obtain these would be from the .text property of a requests
            response, as in the example usage below.

    Returns:
        header (multi-line string):
            Every commented line at the top of the rdb file is marked with a
            '#' symbol. Each of these lines is stored in this output.
        outputDF (pandas.DataFrame):
            A dataframe containing the information in the rdb file. `site_no`
            and `parameter_cd` are interpreted as a string, but every other number
            is interpreted as a float or int; missing values as an np.nan;
            strings for everything else.
        columns (list of strings):
            The column names, taken from the rdb header row.
        dtypes (list of strings):
            The second header row from the rdb file. These mostly tell the
            column width, and typically record everything as string data ('s')
            type. The exception to this are dates, which are listed with a 'd'.
    """
    try:
        headerlines = []
        datalines = []
        count = 0
        for line in text.splitlines():
            if line[0] == "#":
                headerlines.append(line)
            elif count == 0:
                columns = line.split()
                count += 1
            elif count == 1:
                dtypes = line.split()
                count += 1
            else:
                datalines.append(line)
        data = "\n".join(datalines)
        header = "\n".join(headerlines)

        outputDF = pd.read_csv(
            StringIO(data),
            sep="\t",
            comment="#",
            header=None,
            names=columns,
            dtype={"site_no": str, "parameter_cd": str},
            # When converted like this, poorly-formed dates will be saved as strings
            parse_dates=True,
            
            # If dates are converted like this, then poorly formed dates will stop the process
            # converters={"peak_dt": pd.to_datetime},
        )
        
        # Another approach would be to convert date columns later, and catch errors
        # try:
        #   outputDF.peak_dt = pd.to_datetime(outputDF.peak_dt)
        # except ValueError as err:
        #   print(f"Unable to parse date. reason: '{str(err)}'.")

    except:
        print(
            "There appears to be an error processing the file that the USGS "
            "returned. This sometimes occurs if you entered the wrong site "
            "number. We were expecting an RDB file, but we received the "
            f"following instead:\n{text}"
        )
        raise

    return header, outputDF, columns, dtypes


#Testing function and getting templates for df
testdf = read_rdb_2(requests.get('https://waterservices.usgs.gov/nwis/site/?format=rdb&sites=01196530&siteStatus=all').text)[1]
print(testdf)

# +
#Getting coordinates for all stations in station_dict
usgs_df = pd.DataFrame()

for station in station_dict.keys():
    url = 'https://waterservices.usgs.gov/nwis/site/?format=rdb&sites=' + str(station) + '&siteStatus=all'
    station_text= requests.get(url).text
    df = read_rdb_2(station_text)[1]
    usgs_df=pd.concat([usgs_df, df])
    time.sleep(.1)

usgs_df.head()
# -

print(len(usgs_df))

#Merging
for i, df in enumerate(dfs):
    dfs[i]=dfs[i].merge(usgs_df[["site_no", "dec_lat_va", "dec_long_va", "station_nm"]], how="left", on="site_no")
    dfs[i].rename(columns={"dec_lat_va": "Latitude", "dec_long_va":"Longitude"}, inplace=True)

dfs[2].head()

# # Outputting

# +
#Aggregating
output=pd.DataFrame()
for df in dfs:
    output = pd.concat([output, df])

#Summing by year and station
output["Year"]=pd.to_datetime(output["datetime"]).dt.year
# -

#Keeping only columns with temperature 
temp_cols= [col for col in output.columns if (re.match("Temperature", col) is not None) and (col[-3:]!="_cd")]
# print(temp_cols)
summary = output.groupby(["Year", "station_nm"])[temp_cols].count()
print(summary.head())

#Dropping entries if they have no temperature
print(len(output))
output.dropna(subset=temp_cols, how="all", inplace=True)
print(len(output))

stations = pd.DataFrame.from_dict(station_dict, columns=["station_nm"], orient='index')
stations.reset_index(names=["site_no"], inplace=True)
stations

output.to_csv(output_file)
stations.to_csv(stations_output)


