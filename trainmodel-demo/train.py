from bandit import Bandit
import pandas as pd
import numpy as np
import time
import statsmodels.formula.api as sm

df = pd.DataFrame({ \
    "A": np.random.normal(100,10,5).tolist(), \
    "B": np.random.normal(50,5,5).tolist(), \
    "C": np.random.normal(1000,100,5).tolist() \
})
result = sm.ols(formula="A ~ B + C", data=df).fit()

metadata = {'R2': result.rsquared, 'AIC': result.aic}

bandit = Bandit()

bandit.metadata.R2 = result.rsquared
bandit.metadata.AIC = result.aic

df.to_csv('/job/output-files/dataframe.csv')

# email = Email()
#
# email.body(result.summary())
# email.attachment('/job/output-files/dataframe.csv')

for x in range(10):
    for y in range(10):
        for tag in ["a", "b", "c", "d", "e", "f", "g"]:
            bandit.report(tag, np.log((10/(y+1)*10)) + np.random.rand())
        time.sleep(0.1)
