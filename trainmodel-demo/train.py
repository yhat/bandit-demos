from bandit import Bandit
import pandas as pd
import numpy as np
import time
import statsmodels.formula.api as sm

df = pd.DataFrame({ \
    "A": np.random.normal(100,10,50).tolist(), \
    "B": np.random.normal(50,5,50).tolist(), \
    "C": np.random.exponential(5,50).tolist() \
})
result = sm.ols(formula="A ~ B + C", data=df).fit()

metadata = {'R2': result.rsquared, 'AIC': result.aic}

# bandit = Bandit()
bandit = Bandit('colin', 'c4548110-cc4b-11e6-a5c5-0242ac110003','http://54.201.192.120/')
# 
# for x in range(10):
#     for y in range(10):
#         # for tag in ["a", "b", "c", "d", "e", "f", "g"]:
#         # bandit.report('tag', float(np.log((10/(y+1)*10)) + np.random.rand()))
#     time.sleep(0.1)


bandit.metadata.R2 = result.rsquared
bandit.metadata.AIC = result.aic

# df.to_csv('/job/output-files/dataframe.csv')

# email = Email()
#
# email.body(result.summary())
# email.attachment('/job/output-files/dataframe.csv')

# bandit.get_job_results()
# bandit.get_job_results('bandit-demos', 'Tensorflow')
