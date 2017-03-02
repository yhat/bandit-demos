from bandit import *
import pandas as pd
import numpy as np
import time
from time import gmtime, strftime
import statsmodels.formula.api as sm

bandit = Bandit()

df = pd.DataFrame({ \
    "A": np.random.normal(100,10,5).tolist(), \
    "B": np.random.normal(50,5,5).tolist(), \
    "C": np.random.normal(1000,100,5).tolist() \
})

result = sm.ols(formula="A ~ B + C", data=df).fit()

bandit.metadata.R2 = result.rsquared
bandit.metadata.AIC = result.aic
bandit.metadata['value1'] = 2

print('The Time is: ', strftime("%Y-%m-%d %H:%M:%S", time.localtime()))

# print(metadata)
#
# df.to_csv('/job/output-files/dataframe.csv')

# body = 'This is an email body'
# email = job.Email(["colin@yhathq.com"])
# email.subject("HI")
# email.body(body)
#
# email.body(strftime("%Y-%m-%d %H:%M:%S", gmtime()))
#
# print(email)
# there are no attachments
# bandit.email.attachment('/job/output-files/dataframe.csv')
