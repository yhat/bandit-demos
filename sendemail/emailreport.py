from bandit import Bandit
import pandas as pd
import numpy as np
import time
from time import gmtime, strftime
import statsmodels.formula.api as sm

df = pd.DataFrame({ \
    "A": np.random.normal(100,10,5).tolist(), \
    "B": np.random.normal(50,5,5).tolist(), \
    "C": np.random.normal(1000,100,5).tolist() \
})
result = sm.ols(formula="A ~ B + C", data=df).fit()

metadata = {'R2': result.rsquared, 'AIC': result.aic}

bandit = Bandit('colin', 'c4548110-cc4b-11e6-a5c5-0242ac110003','http://54.201.192.120/')
bandit.metadata.R2 = result.rsquared
bandit.metadata.AIC = result.aic
bandit.metadata['value1'] = 2

for x in range(10):
    for y in range(10):
        bandit.report('a', x, y)

for x in range(10):
    for y in range(10):
        for tag in ["a", "b", "c", "d", "e", "f", "g"]:
            bandit.report(tag, y, np.random.rand())
        time.sleep(0.1)

print(strftime("%Y-%m-%d %H:%M:%S", gmtime()))
print(metadata)

df.to_csv('/job/output-files/dataframe.csv')

# bandit.email.body("HI")
# print(bandit.email)

# email.body(strftime("%Y-%m-%d %H:%M:%S", gmtime()))
# email.attachment('/job/output-files/dataframe.csv')
