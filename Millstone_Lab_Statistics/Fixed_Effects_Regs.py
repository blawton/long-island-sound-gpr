import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# # Pre-2012 vs post 2012

scounts.plot

scounts=pd.read_csv("../Data/Dominion Energy/Millstone_Shoot_Counts.csv", index_col=0)
scounts["Year"]=pd.to_datetime(scounts["Date"]).dt.year
scounts["Month"]=pd.to_datetime(scounts["Date"]).dt.month
scounts=scounts.groupby(["Month", "Year", "Station ID"]).mean()
scounts.reset_index(inplace=True)
plot_list=[scounts.loc[scounts["Year"]<=2012, "Number of Vegetative Shoots"], scounts.loc[scounts["Year"]>2012, "Number of Vegetative Shoots"]]
plt.boxplot(plot_list, labels=["Pre-2012", "Post-2012"])
plt.xlabel("Month")
plt.ylabel("Mean of Vegetative Shoot Count")
plt.show()

# # Preparing Data

scounts=pd.read_csv("../Data/Dominion Energy/Millstone_Shoot_Counts.csv", index_col=0)
scounts["Year"]=pd.to_datetime(scounts["Date"]).dt.year
scounts["Month"]=pd.to_datetime(scounts["Date"]).dt.month
scounts=scounts.groupby(["Month", "Year", "Station ID"]).mean()
scounts.reset_index(inplace=True)
plot_list=[scounts.loc[scounts["Month"]==month, "Number of Vegetative Shoots"] for month in pd.unique(scounts["Month"])]
plt.boxplot(plot_list, labels=pd.unique(scounts["Month"]))
plt.xlabel("Month")
plt.ylabel("Mean of Vegetative Shoot Count")
plt.show()

# The former cell suggests that a regression on the change in shoot count from July to September could be instructive

# +
rcounts=pd.read_csv("../Data/Dominion Energy/Millstone_Repro_Counts.csv")
print(pd.unique(rcounts["Station"]))
rcounts.replace({"Jordan Cove":"JC", "Niantic River":"NR", "White Point":"WP"}, inplace=True)

rcounts=rcounts.groupby(["Year", "Month", "Station"]).mean().reset_index()
months=pd.unique(rcounts["Month"])
plot_list=[rcounts.loc[rcounts["Month"]==month, "ReproCount"] for month in months]
plt.boxplot(plot_list, labels=months)
plt.show()
# -


rcounts

# +
# Making 8 combinations of temperature to test: (whole summer average vs. July/Aug avg) x (bottom/top) x (C/NB)
avg_periods={"JA": [7, 8], "Summer": [6, 7, 8]}
inputs=pd.read_csv("../Data/Dominion Energy/C_and_NB_data.csv")
inputs.dropna(subset="Station", inplace=True)
indep_vars=pd.DataFrame()
print(pd.unique(inputs["Station"]))

for sta in pd.unique(inputs["Station"]):
    working=inputs.loc[inputs["Station"]==sta].copy()
    working=working[["Date", "Surf. Temp.", "Bot. Temp."]]
    working["Month"]=pd.to_datetime(working["Date"]).dt.month
    working["Year"]=pd.to_datetime(working["Date"]).dt.year
    for name, period in avg_periods.items():
        to_mean=working.loc[working["Month"].isin(period)]
        meaned=to_mean.groupby("Year").mean()
        indep_vars["_".join([name, "surf", sta])]=meaned["Surf. Temp."]
        indep_vars["_".join([name, "bot", sta])]=meaned["Bot. Temp."]
    
indep_vars
# -

# # Temperature Analysis Trend







# # Running C and NB Regressions

from linearmodels import PanelOLS
import statsmodels.api as sm

#Lag of dep variables from independent variables
year_lag=0

# +
# Making dependent variables
dep_vars=scounts.copy()
dep_vars.rename(columns={"Number of Vegetative Shoots":"ShootCount"}, inplace=True)
dep_vars=dep_vars.set_index(["Station ID", "Year", "Month"])["ShootCount"].unstack(level=-1)
dep_vars.columns=months
dep_vars["Summer_Avg"]=dep_vars.mean(axis=1)
dep_vars["Sep_Minus_July"]=dep_vars[9]-dep_vars[7]
dep_vars=dep_vars[["Summer_Avg", "Sep_Minus_July"]]


dep_vars2 = rcounts.copy()
dep_vars2.rename(columns={"Station":"Station ID"}, inplace=True)
dep_vars2=dep_vars2.set_index(["Station ID", "Year", "Month"])["ReproCount"].unstack(level=-1)
dep_vars2.columns=months
dep_vars2["Summer_Avg"]=dep_vars2.mean(axis=1)
dep_vars2["Sep_Minus_July"]=dep_vars2[9]-dep_vars2[7]

dep_vars = dep_vars.merge(dep_vars2[["Summer_Avg", "Sep_Minus_July"]], how="outer", on=["Station ID", "Year"], suffixes=["_reprocount", "_shootcount"])
# -

var

# +
dep_cols=dep_vars.columns
indep_cols=indep_vars.columns


var=dep_vars.reset_index()
to_merge=indep_vars.reset_index()
to_merge["Year"]+=year_lag

var=var.merge(to_merge, on=["Year"])
var.set_index(["Station ID", "Year"], inplace=True)
print(len(var))
var.dropna(how="any", axis=0, inplace=True)
print(len(var))
agg=pd.DataFrame()

for dep in dep_cols:
    for indep in indep_cols:
        X=sm.add_constant(var[indep])
        mod=PanelOLS(var[dep], var[indep], entity_effects=True)
        results=mod.fit()
        tabs=results.summary.as_html()
        dfs=pd.read_html(tabs)
        #print(df[0].iloc[0, 3])
        res=dfs[1]
        res["r_squared"]=res.iloc[0, 3]
        res.columns=["independent_var"] + list(res.iloc[0, 1:])
        res.drop(0, axis=0, inplace=True)
        res["dependent_var"]=dep
        res.set_index(["independent_var", "dependent_var"], inplace=True)
        agg=pd.concat([agg, res], axis=0)
        
agg.sort_values(by=["P-value"], inplace=True)
agg


# +
#agg.to_csv("../Results/fixed_effects_results/NB_C_temperature_regressions.csv")

# +
# Plotting variables

plt.scatter(var["Summer_bot_C"], var["Summer_Avg_shootcount"])
plt.title("July August Temp at C vs. Summer Mean Shootcount")
plt.xlabel("Station C Summer mean temp (deg C)")
plt.ylabel("Mean Shootcount")
plt.show()
# -

# # Running Regressions (Intake)

from linearmodels import PanelOLS
import statsmodels.api as sm

# +
#Reading in intake temp
indep=pd.read_csv("../Data/Dominion Energy/Intake_temps.csv")
indep["Year"]=pd.to_numeric(indep["Year"], errors="coerce")
indep.set_index("Year", inplace=True)
indep["JA_Intake"]=indep[["Jul", "Aug"]].mean(axis=1)
indep["Summer_Intake"]=indep[["Jun", "Jul", "Aug"]].mean(axis=1)
indep=indep.loc[list(range(1976, 2022)), ["Summer_Intake", "JA_Intake"]]

indep_var2=indep
indep.head()

# +
# Making dependent variables
dep_vars=scounts.copy()
dep_vars.rename(columns={"Number of Vegetative Shoots":"ShootCount"}, inplace=True)
dep_vars=dep_vars.set_index(["Station ID", "Year", "Month"])["ShootCount"].unstack(level=-1)
dep_vars.columns=months
dep_vars["Summer_Avg"]=dep_vars.mean(axis=1)
dep_vars["Sep_Minus_July"]=dep_vars[9]-dep_vars[7]
dep_vars=dep_vars[["Summer_Avg", "Sep_Minus_July"]]


dep_vars2 = rcounts.copy()
dep_vars2.rename(columns={"Station":"Station ID"}, inplace=True)
dep_vars2=dep_vars2.set_index(["Station ID", "Year", "Month"])["ReproCount"].unstack(level=-1)
dep_vars2.columns=months
dep_vars2["Summer_Avg"]=dep_vars2.mean(axis=1)
dep_vars2["Sep_Minus_July"]=dep_vars2[9]-dep_vars2[7]

dep_vars = dep_vars.merge(dep_vars2[["Summer_Avg", "Sep_Minus_July"]], how="outer", on=["Station ID", "Year"], suffixes=["_reprocount", "_shootcount"])
dep_vars

# +
dep_cols=dep_vars.columns
indep_cols=indep_var2.columns


var=dep_vars.reset_index()
to_merge=indep_var2.reset_index()

var=var.merge(indep_var2, on=["Year"])
var.set_index(["Station ID", "Year"], inplace=True)
print(len(var))
var.dropna(how="any", axis=0, inplace=True)
print(len(var))
agg=pd.DataFrame()

for dep in dep_cols:
    for indep in indep_cols:
        X=sm.add_constant(var[indep])
        mod=PanelOLS(var[dep], var[indep], entity_effects=True)
        results=mod.fit()
        tabs=results.summary.as_html()
        dfs=pd.read_html(tabs)
        #print(df[0].iloc[0, 3])
        res=dfs[1]
        res["r_squared"]=res.iloc[0, 3]
        res.columns=["independent_var"] + list(res.iloc[0, 1:])
        res.drop(0, axis=0, inplace=True)
        res["dependent_var"]=dep
        res.set_index(["independent_var", "dependent_var"], inplace=True)
        agg=pd.concat([agg, res], axis=0)

agg.sort_values(by="P-value", inplace=True)
agg


# +
data=var.reset_index()

plt.scatter(data["Summer_Intake"], data["Summer_Avg_shootcount"])
plt.title("Summer Intake Temps (1985 Onwards)")
plt.show()

plt.scatter(data.loc[data["Year"]>=2011, "Summer_Intake"], data.loc[data["Year"]>=2011, "Summer_Avg_shootcount"])
plt.title("Summer Intake Temps (2011 Onwards)")
plt.show()

# +
#Running only above 20 degrees
dep_cols=dep_vars.columns
indep_cols=indep_var2.columns

var=dep_vars.reset_index()
to_merge=indep_var2.reset_index()

#Year restriction and dropping nas
print(len(var))
var.dropna(how="any", axis=0, inplace=True)
var=var.loc[var["Year"]>=2011]
print(len(var))

var=var.merge(indep_var2, on=["Year"])
var.set_index(["Station ID", "Year"], inplace=True)


#df for results
agg=pd.DataFrame()

for dep in dep_cols:
    for indep in indep_cols:
        X=sm.add_constant(var[indep])
        mod=PanelOLS(var[dep], var[indep], entity_effects=True)
        results=mod.fit()
        tabs=results.summary.as_html()
        dfs=pd.read_html(tabs)
        #print(df[0].iloc[0, 3])
        res=dfs[1]
        res["r_squared"]=res.iloc[0, 3]
        res.columns=["independent_var"] + list(res.iloc[0, 1:])
        res.drop(0, axis=0, inplace=True)
        res["dependent_var"]=dep
        res.set_index(["independent_var", "dependent_var"], inplace=True)
        agg=pd.concat([agg, res], axis=0)
        
agg.sort_values(by="P-value", inplace=True)
agg.head()
# -

# # Intake (Lagged Regression)

lag=1

# +
#Reading in intake temp
indep=pd.read_csv("../Data/Dominion Energy/Intake_temps.csv")
indep["Year"]=pd.to_numeric(indep["Year"], errors="coerce")
indep.set_index("Year", inplace=True)
indep=indep.iloc[:, :12].copy()
indep.columns=range(1, 13)
indep=pd.DataFrame(indep.stack())
indep.reset_index(level=-1, inplace=True)
indep.columns=["Month", "Temp"]

#Setting indep var for intake regressions
indep_var2=indep
indep.head()

# +
# Making dependent variables
dep_vars=scounts.copy()
dep_vars.rename(columns={"Number of Vegetative Shoots":"ShootCount"}, inplace=True)
dep_vars=dep_vars.set_index(["Station ID", "Year", "Month"])["ShootCount"].unstack(level=-1)
dep_vars.columns=months
dep_vars=pd.DataFrame(dep_vars.stack())
dep_vars.reset_index(level=2, inplace=True)
dep_vars.columns=["Month", "ShootCount"]
dep_vars["Month"]-=lag
dep_vars=dep_vars.reset_index().set_index(["Station ID", "Month", "Year"])

dep_vars2 = rcounts.copy()
dep_vars2.rename(columns={"Station":"Station ID"}, inplace=True)
dep_vars2=dep_vars2.set_index(["Station ID", "Year", "Month"])["ReproCount"].unstack(level=-1)
dep_vars2.columns=months
dep_vars2=pd.DataFrame(dep_vars2.stack())
dep_vars2.reset_index(level=2, inplace=True)
dep_vars2.columns=["Month", "ReproCount"]
dep_vars2["Month"]-=lag
dep_vars2=dep_vars2.reset_index().set_index(["Station ID", "Month", "Year"])

dep_vars = dep_vars.merge(dep_vars2["ReproCount"], left_index=True, right_index=True)
dep_vars

# +
dep_cols=["ShootCount", "ReproCount"]
indep_cols=["Temp"]

var=dep_vars.reset_index()
to_merge=indep_var2.reset_index()
to_merge["Year"]+=year_lag

var=var.merge(indep_var2, on=["Year", "Month"])
var.set_index(["Station ID", "Year"], inplace=True)
print(len(var))
var.dropna(how="any", axis=0, inplace=True)
print(len(var))
agg=pd.DataFrame()

for dep in dep_cols:
    for indep in indep_cols:
        X=sm.add_constant(var[indep])
        mod=PanelOLS(var[dep], var[[indep, "Month"]], entity_effects=True)
        results=mod.fit()
        tabs=results.summary.as_html()
        dfs=pd.read_html(tabs)
        #print(df[0].iloc[0, 3])
        res=dfs[1]
        res["r_squared"]=res.iloc[0, 3]
        res.columns=["independent_var"] + list(res.iloc[0, 1:])
        res.drop(0, axis=0, inplace=True)
        res["dependent_var"]=dep
        res.set_index(["independent_var", "dependent_var"], inplace=True)
        agg=pd.concat([agg, res], axis=0)
agg.head()

# +
#Running on data over 20 degrees fahrenheight
dep_cols=["ShootCount", "ReproCount"]
indep_cols=["Temp"]

var=dep_vars.reset_index()
to_merge=indep_var2.reset_index()
to_merge["Year"]+=year_lag

var=var.merge(indep_var2, on=["Year", "Month"])
var.set_index(["Station ID", "Year"], inplace=True)

#Temp restriction and dropping nas
print(len(var))
var=var.loc[var["Temp"]>=20]
var.dropna(how="any", axis=0, inplace=True)
print(len(var))

#df for all results
agg=pd.DataFrame()

for dep in dep_cols:
    for indep in indep_cols:
        X=sm.add_constant(var[indep])
        mod=PanelOLS(var[dep], var[[indep, "Month"]], entity_effects=True)
        results=mod.fit()
        tabs=results.summary.as_html()
        dfs=pd.read_html(tabs)
        #print(df[0].iloc[0, 3])
        res=dfs[1]
        res["r_squared"]=res.iloc[0, 3]
        res.columns=["independent_var"] + list(res.iloc[0, 1:])
        res.drop(0, axis=0, inplace=True)
        res["dependent_var"]=dep
        res.set_index(["independent_var", "dependent_var"], inplace=True)
        agg=pd.concat([agg, res], axis=0)
agg.head()
# -

plt.scatter(var["Temp"], var["ShootCount"])
plt.title("1 Month Lagged ShootCount vs Intake Temperature")
plt.xlabel("Intake Temp")
plt.ylabel("Temp")
plt.show()

agg.sort_values(by=["P-value"]).to_csv("../Results/fixed_effects_results/Lagged_Intake_temperature_regressions.csv")
