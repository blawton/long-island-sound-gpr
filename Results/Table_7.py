import pandas as pd
import numpy as np

#Params
years=[2019, 2020, 2021]
roots=["Optimization_with_Pre_Processing_results_lis_ja", "Optimization_with_Pre_Processing_results_lis_ja_knn"]
names={"Optimization_with_Pre_Processing_results_lis_ja": "GP",
       "Optimization_with_Pre_Processing_results_lis_ja_knn": "KNN"}

agg=pd.DataFrame()
for year in years:
    for root in roots:
        data = pd.read_csv(root+ "_" + str(year) + ".csv", index_col=0)
        data.index=pd.MultiIndex.from_product([[names[root]], [year]])
        agg=pd.concat([agg, data])
agg=agg[["mean_test_r2", "mean_test_neg_root_mean_squared_error"]]
agg = agg.unstack(level=0)
agg=pd.concat([agg, pd.DataFrame(data=[agg.mean(axis=0)], index=["Mean Across Years"])])

agg.columns=pd.MultiIndex.from_product([["R-squared", "RMSE"], ["GP w/ Embayment Data", "IDW kNN w/ Embayment Data"]])

agg["RMSE"]=-agg["RMSE"]

agg

agg.to_csv("../Figures_for_Paper/tab7.csv")


