import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

sts = pd.read_csv("sTS Discrete Data/STS_Discrete_4_13_2023.csv")
uri = pd.read_csv("URIWW/URIWW_6_20_2023.csv")
print(sts.columns)
print(pd.unique(uri["Parameter"]))
print(sts.loc[~sts["Surface Turbidity (NTU)"].isna(), "Surface Turbidity (NTU)"])




#Dropping nas and ensuring numeric dtype of turbidity data
sts = sts.loc[~sts["Surface Turbidity (NTU)"].isna()]
sts.loc[sts["Surface Turbidity (NTU)"]=="< 1", "Surface Turbidity (NTU)"]=.5
sts["Surface Turbidity (NTU)"]=pd.to_numeric(sts["Surface Turbidity (NTU)"])

#Grouping by station
sts=sts.groupby("MonitoringLocationIdentifier").mean()



# +
# #Binning data
# print(df["Surface Turbidity (NTU)"].max())
# bins=np.quantile(df["Surface Turbidity (NTU)"], np.linspace(0, 1, 10))
# print(bins)
# df["Binned Turbidity"]=np.digitize(df["Surface Turbidity (NTU)"], bins)
# -

plt.scatter(sts["Field Longitude (dec. deg.)"], np.log(sts["Surface Turbidity (NTU)"]))
plt.vlines([-72.59], color="red", ymin=plt.gca().get_ylim()[0], ymax=plt.gca().get_ylim()[1])
plt.title("Log-scaled Turbity at Various Longitudes in LIS")
plt.ylabel("ln(NTU)")
plt.xlabel("Longitude")

help(plt.gca())


