import pandas as pd
import numpy as np

pd.options.display.max_columns=50

years=[2019, 2020, 2021]

pd.options.display.float_format = '{:,.3f}'.format

# # LIS JA

kernels={"1**2 * Matern(length_scale=[1, 1, 1, 1], nu=0.5) + 1**2 * RBF(length_scale=[1, 1, 1, 1])":"Matern + RBF", "1**2 * Matern(length_scale=[1, 1, 1, 1], nu=1.5) + 1**2 * RationalQuadratic(alpha=1, length_scale=1)":"Matern + Rational Quadratric", "1**2 * RBF(length_scale=[1, 1, 1, 1]) + 1**2 * RationalQuadratic(alpha=1, length_scale=1)":"RBF + Rational Quadratic"}
agg=pd.DataFrame()
for year in years:
    working = pd.read_csv("Optimization_with_Pre_Processing_results_" + str(year) + ".csv", index_col=0)
    working["Year"]=year
    agg=pd.concat([agg, working])
agg.sort_values(["param_model__kernel", "Year"], inplace=True)
agg.set_index(["param_model__kernel", "Year"], inplace=True)
agg.reset_index(inplace=True)
agg.replace(kernels, inplace=True)
agg.set_index(["param_model__kernel", "Year"], inplace=True)
agg

#Getting average r^2 values
agg.groupby("param_model__kernel")["mean_test_r2"].mean()

#Getting average rmse values
means=agg[["mean_test_neg_root_mean_squared_error", "mean_test_r2"]].reset_index()
print(means)
for i, kernel in enumerate(pd.unique(agg.reset_index()["param_model__kernel"])):
    means=pd.concat([means, pd.DataFrame([{
        "param_model__kernel":kernel, 
        "Year":"Intertemporal Mean",
        "mean_test_neg_root_mean_squared_error":np.mean(means.loc[means["param_model__kernel"]==kernel, "mean_test_neg_root_mean_squared_error"]),
        "mean_test_r2":np.mean(means.loc[means["param_model__kernel"]==kernel, "mean_test_r2"])

    }])])
means.sort_values(["param_model__kernel", "Year"], inplace=True)
means.set_index(["param_model__kernel", "Year"], inplace=True)
means

#Getting RMSE of chosen Kernel
res = pd.DataFrame(means.loc["RBF + Rational Quadratic"]["mean_test_neg_root_mean_squared_error"])
res.rename(columns={"mean_test_neg_root_mean_squared_error":"RMSE Gaussian Process"})

res2=pd.read_csv("CT_DEEP_Interpolation/RMSE_LIS_JA.csv", index_col=0)
res2.columns=["RMSE (IDW)"]
res2["RMSE (GP)"]=-res.values
res2["Percent Change"]=(res2.iloc[:, 1]-res2.iloc[:,0])/res2.iloc[:, 0]*100
res2.columns=pd.MultiIndex.from_product([["LIS DOY 182-244"], res2.columns])
res2_LIS_JA=res2.copy()
res2_LIS_JA

# # ES JA

kernels={"1.09**2 * RBF(length_scale=[0.213, 0.159, 4.78, 0.122]) + 1.51**2 * RationalQuadratic(alpha=0.412, length_scale=1.26)": "Entire LIS", "1.46**2 * RBF(length_scale=[5.52, 4.97, 3.84, 0.052]) + 4.03**2 * RationalQuadratic(alpha=0.0231, length_scale=0.538)": "ES"}
agg=pd.DataFrame()
for year in years:
    working = pd.read_csv("Optimization_with_Pre_Processing_results_es_ja_" + str(year) + ".csv", index_col=0)
    working["Year"]=year
    agg=pd.concat([agg, working])
agg.sort_values(["param_model__kernel", "Year"], inplace=True)
agg.set_index(["param_model__kernel", "Year"], inplace=True)
agg.reset_index(inplace=True)
agg.replace(kernels, inplace=True)
agg.set_index(["param_model__kernel", "Year"], inplace=True)
agg

#Getting average r^2 values
agg.groupby("param_model__kernel")["mean_test_r2"].mean()

#Getting average rmse values
means=agg[["mean_test_neg_root_mean_squared_error", "mean_test_r2"]].reset_index()
print(means)
for i, kernel in enumerate(pd.unique(agg.reset_index()["param_model__kernel"])):
    means=pd.concat([means, pd.DataFrame([{
        "param_model__kernel":kernel, 
        "Year":"Intertemporal Mean",
        "mean_test_neg_root_mean_squared_error":np.mean(means.loc[means["param_model__kernel"]==kernel, "mean_test_neg_root_mean_squared_error"]),
        "mean_test_r2":np.mean(means.loc[means["param_model__kernel"]==kernel, "mean_test_r2"])

    }])])
means.sort_values(["param_model__kernel", "Year"], inplace=True)
means.set_index(["param_model__kernel", "Year"], inplace=True)
means

#Formatting columns
means.columns=["Mean of Neg. RMSE", "Mean of R^2"]
means

#Getting RMSE of chosen Kernel
res = means.reset_index(level=0, drop=True)
res

res2=pd.read_csv("CT_DEEP_Interpolation/RMSE_ES_JA.csv", index_col=0)
res2.columns=["RMSE (IDW)"]
res2["RMSE (GP)"]=-res.iloc[:,0].values
res2["Percent Change"]=(res2.iloc[:, 1]-res2.iloc[:,0])/res2.iloc[:, 0]*100
res2.columns=pd.MultiIndex.from_product([["Eastern Sound DOY 182-244"], res2.columns])
res2_ES_JA=res2.copy()
res2_ES_JA

# # LIS full date range

kernels={"1.09**2 * RBF(length_scale=[0.213, 0.159, 4.78, 0.122]) + 1.51**2 * RationalQuadratic(alpha=0.412, length_scale=1.26)": "Entire LIS", "1.46**2 * RBF(length_scale=[5.52, 4.97, 3.84, 0.052]) + 4.03**2 * RationalQuadratic(alpha=0.0231, length_scale=0.538)": "ES"}
agg=pd.DataFrame()
for year in years:
    working = pd.read_csv("Optimization_with_Pre_Processing_results_lis_entirely_" + str(year) + ".csv", index_col=0)
    working["Year"]=year
    agg=pd.concat([agg, working])
agg.sort_values(["param_model__kernel", "Year"], inplace=True)
agg.set_index(["param_model__kernel", "Year"], inplace=True)
agg.reset_index(inplace=True)
agg.replace(kernels, inplace=True)
agg.set_index(["param_model__kernel", "Year"], inplace=True)
agg

#Getting average r^2 values
agg.groupby(level=0)["mean_test_r2"].mean()

#Getting average rmse values
means=agg[["mean_test_neg_root_mean_squared_error", "mean_test_r2"]].reset_index()
print(means)
for i, kernel in enumerate(pd.unique(agg.reset_index()["param_model__kernel"])):
    means=pd.concat([means, pd.DataFrame([{
        "param_model__kernel":kernel, 
        "Year":"Intertemporal Mean",
        "mean_test_neg_root_mean_squared_error":np.mean(means.loc[means["param_model__kernel"]==kernel, "mean_test_neg_root_mean_squared_error"]),
        "mean_test_r2":np.mean(means.loc[means["param_model__kernel"]==kernel, "mean_test_r2"])

    }])])
means.sort_values(["param_model__kernel", "Year"], inplace=True)
means.set_index(["param_model__kernel", "Year"], inplace=True)
means

#Formatting columns
means.columns=["Mean of Neg. RMSE", "Mean of R^2"]
means

#Getting RMSE of chosen Kernel
res = means.reset_index(level=0, drop=True)
res

res2=pd.read_csv("CT_DEEP_Interpolation/RMSE_LIS_152_274.csv", index_col=0)
res2.columns=["RMSE (IDW)"]
res2["RMSE (GP)"]=-res.iloc[:,0].values
res2["Percent Change"]=(res2.iloc[:, 1]-res2.iloc[:,0])/res2.iloc[:, 0]*100
res2.columns=pd.MultiIndex.from_product([["LIS DOY 152-274"], res2.columns])
res2_LIS_152_274=res2.copy()
res2_LIS_152_274

# # ES Full date range

agg=pd.DataFrame()
for year in years:
    working = pd.read_csv("Optimization_with_Pre_Processing_results_es_vs_lis_" + str(year) + ".csv", index_col=0)
    working["Year"]=year
    agg=pd.concat([agg, working])
agg.sort_values(["param_model__kernel", "Year"], inplace=True)
agg.set_index(["param_model__kernel", "Year"], inplace=True)
agg.reset_index(inplace=True)
kernels=dict(zip(pd.unique(agg["param_model__kernel"]), ["Entire LIS", "ES"]))
agg.replace(kernels, inplace=True)
agg=agg.loc[agg["param_model__kernel"]=="ES"]
agg.set_index(["param_model__kernel", "Year"], inplace=True)
agg

#Getting average r^2 values
agg.groupby("param_model__kernel")["mean_test_r2"].mean()

#Getting average rmse values
means=agg[["mean_test_neg_root_mean_squared_error", "mean_test_r2"]].reset_index()
print(means)
for i, kernel in enumerate(pd.unique(agg.reset_index()["param_model__kernel"])):
    means=pd.concat([means, pd.DataFrame([{
        "param_model__kernel":kernel, 
        "Year":"Intertemporal Mean",
        "mean_test_neg_root_mean_squared_error":np.mean(means.loc[means["param_model__kernel"]==kernel, "mean_test_neg_root_mean_squared_error"]),
        "mean_test_r2":np.mean(means.loc[means["param_model__kernel"]==kernel, "mean_test_r2"])

    }])])
means.sort_values(["param_model__kernel", "Year"], inplace=True)
means.set_index(["param_model__kernel", "Year"], inplace=True)
means

#Getting RMSE of chosen Kernel
res = means.reset_index(level=0, drop=True)

res2=pd.read_csv("CT_DEEP_Interpolation/RMSE_ES_152_274.csv", index_col=0)
res2.columns=["RMSE (IDW)"]
res2["RMSE (GP)"]=-res.iloc[:, 0].values
res2["Percent Change"]=(res2.iloc[:, 1]-res2.iloc[:,0])/res2.iloc[:, 0]*100
res2.columns=pd.MultiIndex.from_product([["Eastern Sound DOY 152-274"], res2.columns])
res2_ES_152_274=res2.copy()
res2_ES_152_274

# # All Results together

means=pd.concat([res2_LIS_JA, res2_ES_JA, res2_LIS_152_274, res2_ES_152_274], axis=1)
means




