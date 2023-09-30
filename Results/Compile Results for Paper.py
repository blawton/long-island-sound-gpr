import pandas as pd
import numpy as np

pd.options.display.max_columns=50

years=[2019, 2020, 2021]

pd.options.display.float_format = '{:,.3f}'.format

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

#Formatting columns
means.columns=["Mean of Neg. RMSE", "Mean of R^2"]
means

#Getting RMSE of chosen Kernel
res = pd.DataFrame(-means.loc["1**2 * RBF(length_scale=[1, 1, 1, 1]) + 1**2 * RationalQuadratic(alpha=1, length_scale=1)"]["mean_test_neg_root_mean_squared_error"])
res.rename(columns={"mean_test_neg_root_mean_squared_error":"RMSE Gaussian Process"})

res2=pd.read_csv("CT_DEEP_interpolation_RMSE.csv", index_col=0)
res2.columns=["RMSE Approach I (IDW)"]
res2["RMSE Approach II (Gaussian Process)"]=res.values

res2


