import numpy as np
import pandas as pd
import requests
from io import StringIO
from IPython import display
import time
import os

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

# +
#Local Data Source
path= config["Mystic_River_Reading_path"]

assert(os.path.exists(path))


# -

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


# # Reading and Processing

with open(path) as f:
    text=f.read()
    pms, dfs = read_rdb(text)
dfs

dfs[2]

pms[2]

#Renaming pms to fix dissolved oxygen issue
for pm in pms:
    do_cols=pm.loc[pm["Description"].str.contains("Dissolved oxygen")].copy(deep=True)
    do_cols["Description"]=do_cols["Description"].apply(lambda x: x.split(", ")[0]+ "_" + x.split(", ")[-2] + ", " + x.split(", ")[-1])
    pm.loc[pm["Description"].str.contains("Dissolved oxygen")]=do_cols

for df in dfs:
    df.fillna(value= np.nan, inplace=True)
    df.drop(0, axis=0, inplace=True)
for pm in pms:
    print(pm.columns)
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

site_dict={412047071580800: "MYSTIC HARBOR AT SAFE HARBOR MARINA",
412117071580800: "MYSTIC RIVER AT ROUTE 1 BRIDGE",
412141071580200: "MYSTIC RIVER AT MYSTIC SEAPORT",
412240071574700: "MYSTIC RIVER US RTE I95 BRIDGE"}


# # Coords

#rdb reader from hydrofunctions for usgs file types (adapted from https://github.com/mroberge/hydrofunctions)
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
testdf = read_rdb(requests.get('https://waterservices.usgs.gov/nwis/site/?format=rdb&sites=01196530&siteStatus=all').text)[1]
print(testdf)

# +
#Getting coordinates for all stations in site_dict
usgs_df = pd.DataFrame()

for station in site_dict.keys():
    url = 'https://waterservices.usgs.gov/nwis/site/?format=rdb&sites=' + str(station) + '&siteStatus=all'
    text= requests.get(url).text
    df = read_rdb(text)[1]
    usgs_df=pd.concat([usgs_df, df])
    time.sleep(.1)

usgs_df.head()
# -

#Merging
for i, df in enumerate(dfs):
    dfs[i]=dfs[i].merge(usgs_df[["site_no", "dec_lat_va", "dec_long_va"]], how="left", on="site_no")
    dfs[i].rename(columns={"dec_lat_va": "Latitude", "dec_long_va":"Longitude"}, inplace=True)

# # Outputting

for df in dfs:
    site_no=int(df.iloc[0, 1])
    site=site_dict[site_no]
    print(site)
    df.to_csv(site + ".csv")

dfs[1]
