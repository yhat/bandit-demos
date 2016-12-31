from bandit import *
import pandas as pd
import statsmodels.formula.api as sm

df = pd.DataFrame({"A": [10,20,30,40,50], "B": [20, 30, 10, 40, 50], "C": [32, 234, 23, 23, 42523]})
result = sm.ols(formula="A ~ B + C", data=df).fit()
metadata = {'R2': result.rsquared, 'AIC': result.aic}

bandit.Metadata(metadata)
print(metadata)

df.to_csv('/job/output-files/dataframe.csv')
