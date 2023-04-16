import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from statsmodels.tsa.stattools import adfuller

# +
pd.set_option('display.max_rows', 200)
pd.set_option('display.max_columns', 500)

from IPython.core.display import display, HTML
display(HTML("<style>.container { width:100% ! important; }<style>"))
plt.rcParams["figure.figsize"]=(21, 10)
# -

#All Mystic River Locations
site_dict={412047071580800: "MYSTIC HARBOR AT SAFE HARBOR MARINA",
412117071580800: "MYSTIC RIVER AT ROUTE 1 BRIDGE",
412141071580200: "MYSTIC RIVER AT MYSTIC SEAPORT",
412240071574700: "MYSTIC RIVER US RTE I95 BRIDGE"}

dfs={}
for site in site_dict.values():
    dfs[site]=pd.read_csv(site +".csv")

# # Processing

test=dfs[site_dict[412141071580200]]
test.loc[(test["datetime"]=="2022-08-31 08:48")]

# Because spaces match the .tsv file, the data was correctly parsed

#Quick pre-processing
for site in dfs.keys():
    df=dfs[site]
    
    #Dealing with dates
    df["datetime"]=pd.to_datetime(df["datetime"])
    df["Year"]=df["datetime"].dt.year

    #Dropping qualification code
    df.drop([column for column in df.columns if column[-3:]=="_cd"], axis=1, inplace=True)
    df.replace(site_dict, inplace=True)
    
    #Dropping unnamed col
    df.drop("Unnamed: 0", axis=1, inplace=True)
    
    #Ensuring numeric
    for col in [col for col in df.columns.difference(["datetime", "site_no"])]:
        df[col]=pd.to_numeric(df[col], errors="coerce")

#Aggregating
agg=pd.DataFrame()
for df in dfs.values():
    agg=pd.concat([agg, df], axis=0)
agg

#Outputting
agg.to_csv("agg_fully_processed.csv")

# # Yearly means by site (ignoring provisional/non-provisional)

counts=pd.DataFrame()
for site, df in dfs.items():
    counts=pd.concat([counts, df.groupby(["site_no", "Year"]).count()])
counts

# +
means=pd.DataFrame()

for site, df in dfs.items():
    means=pd.concat([means, df.groupby(["site_no", "Year"]).mean()])

means
# -

# # Graphs

dfs["MYSTIC HARBOR AT SAFE HARBOR MARINA"]

for site, df in dfs.items():
    working=df.copy(deep=True)
    sc=[col for col in working.columns if "Specific" in col]
    x=working["datetime"]
    y=working.drop(["site_no", "Year", "datetime"]+sc, axis=1)
    plt.plot(x, y, label=y.columns)
    plt.title(site)
    plt.xticks(rotation=90)
    plt.legend(loc="upper right", bbox_to_anchor=(1.25, 1))
    plt.show()

