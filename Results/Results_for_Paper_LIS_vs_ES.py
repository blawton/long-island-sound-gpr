import pandas as pd
import numpy as np

years=[2019, 2020, 2021]
agg=pd.DataFrame()
for year in years:
    working = pd.read_csv("Optimization_with_Pre_Processing_results_es_vs_lis_" + str(year) + ".csv", index_col=0)
    working.index=pd.MultiIndex.from_product([[year], ["LIS Overall", "Eastern Sound Only"]])
    working.sort_index(inplace=True)
    agg=pd.concat([agg, working])
cols=["mean_test_r2", "mean_test_neg_root_mean_squared_error"]
agg=agg[cols]
agg.rename(columns=dict(zip(cols, ["Mean of CV R-Squared", "Mean of CV RMSE"])), inplace=True)
agg




