#Comparing summer means from moderately updated file
import pandas as pd

df1=pd.read_csv("agg_summer_means.csv")
df2=pd.read_csv("agg_summer_means_4_19_2023.csv")

df=df1.merge(df2, how="outer", on=["Station ID", "Year"])

df.head()

#Making sure no stations were lost
assert(len(df.loc[(~df["Temperature (C)_x"].isna()) &(df["Temperature (C)_y"].isna())])
       ==0)

#Below two stations were lost because Month_x=7
df.loc[(~df["Temperature (C)_x"].isna()) &(df["Temperature (C)_y"].isna())]

both=df.loc[(~df["Temperature (C)_x"].isna()) &(~df["Temperature (C)_y"].isna())]


both.loc[both["Temperature (C)_x"]!=both["Temperature (C)_y"]]


both_org_matches=both.loc[both["Organization_x"]==both["Organization_y"]]

both_org_matches.loc[both_org_matches["Temperature (C)_x"]!=both_org_matches["Temperature (C)_y"]]

# Only org mismatches and missing data cause a mismatch of data, will check to see if the agg_summer_means output using updated inputs (agg_summer_means_4_21_2023) has the same issues

# # Fixing Org Mismatches

df1=pd.read_csv("agg_summer_means.csv")
df2=pd.read_csv("agg_summer_means_4_21_2023_I.csv")

df=df1.merge(df2, how="outer", on=["Station ID", "Year"])

df.head()

both=df.loc[(~df["Temperature (C)_x"].isna()) &(~df["Temperature (C)_y"].isna())]


print(len(both.loc[both["Temperature (C)_x"]!=both["Temperature (C)_y"]]))
both.loc[both["Temperature (C)_x"]!=both["Temperature (C)_y"]]

# Fixed, but the above discrepancies remain, last row is likely a result of changes in URIWW_Data_Reading.ipynb, but won't affect model (from 2014)

# # Comparing aggregation with new inputs

df1=pd.read_csv("agg_summer_means.csv")
df2=pd.read_csv("agg_summer_means_4_21_2023_II.csv")

df=df1.merge(df2, how="outer", on=["Station ID", "Year"])

df.head()

both=df.loc[(~df["Temperature (C)_x"].isna()) &(~df["Temperature (C)_y"].isna())]


print(len(both.loc[both["Temperature (C)_x"]!=both["Temperature (C)_y"]]))
diff =both.loc[both["Temperature (C)_x"]!=both["Temperature (C)_y"]]

#Only really care about 2019 and after
diff=diff.loc[diff["Year"]>2018]
diff["Difference"]=diff["Temperature (C)_x"]-diff["Temperature (C)_y"]
diff["Difference"].hist()

# Very acceptable level of error, should not change model, any change in the model will be a result of adding in stations that were previously not considered or losing a couple STS stations


