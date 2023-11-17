import pandas as pd
import numpy as np

gt1=pd.read_csv("Millstone_Eelgrass_Mapping.csv")
gt2=pd.read_csv("Millstone_Repro_Counts.csv")
gt3=pd.read_csv("Millstone_Veg_Counts.csv")

gt1.head()
gt1.rename(columns={"Lat":"Latitude", "Long":"Longitude"}, inplace=True)
gt1["Date"]=pd.to_datetime(gt1["Date"])
gt1.head()

gt2.head()

gt3.head()
gt3.rename(columns={"DATE":"Date", "STA": "Station ID"}, inplace=True)
gt3["Date"]=pd.to_datetime(gt3["Date"])
gt3["Organization"]="Dominion Energy"
gt3.head()

gt3.to_csv("Millstone_Shoot_Counts.csv")


