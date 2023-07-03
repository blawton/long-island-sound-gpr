import pandas as pd
import numpy as np

pd.options.display.max_columns=50

years=[2019, 2020, 2021]

agg=pd.DataFrame()
for year in years:
    working = pd.read_csv("Optimization_with_Pre_Processing_results_" + str(year) + ".csv", index_col=0)
    working["Year"]=year
    agg=pd.concat([agg, working])
agg.sort_values(["param_model__kernel", "Year"], inplace=True)
agg.set_index(["param_model__kernel", "Year"], inplace=True)

agg

#Getting average r^2 values
agg.groupby("param_model__kernel")["mean_test_r2"].mean()

#Getting average rmse values
means=agg[["mean_test_neg_root_mean_squared_error", "mean_test_r2"]].reset_index()
print(means)
for kernel in pd.unique(agg.reset_index()["param_model__kernel"]):
    means=pd.concat([means, pd.DataFrame([{
        "param_model__kernel":kernel, 
        "Year":"Intertemporal Mean",
        "mean_test_neg_root_mean_squared_error":np.mean(means.loc[means["param_model__kernel"]==kernel, "mean_test_neg_root_mean_squared_error"]),
        "mean_test_r2":np.mean(means.loc[means["param_model__kernel"]==kernel, "mean_test_r2"])

    }])])
means.sort_values(["param_model__kernel", "Year"], inplace=True)
means.set_index(["param_model__kernel", "Year"], inplace=True)
means

means


