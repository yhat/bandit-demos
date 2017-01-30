from bandit import *
import datetime
import pandas as pd
import numpy as np
import time
import statsmodels.formula.api as sm
import matplotlib
matplotlib.use('Agg')
import seaborn as sns

dir_path = os.path.dirname(os.path.realpath(__file__))

df = pd.DataFrame({ \
    "A": np.random.normal(100,10,50).tolist(), \
    "B": np.random.normal(50,5,50).tolist(), \
    "C": np.random.exponential(5,50).tolist() \
})
result = sm.ols(formula="A ~ B + C", data=df).fit()

metadata = {'R2': result.rsquared, 'AIC': result.aic}

with open(output_dir + 'model_stats.txt', "w") as text_file:
    model_summary = str(result.summary())
    text_file.write(model_summary)

# bandit = Bandit()
bandit = Bandit('colin', 'c4548110-cc4b-11e6-a5c5-0242ac110003','http://54.201.192.120/')
#
for x in range(10):
    for y in range(10):
        bandit.report('tag', float(np.log((10/(y+1)*10)) + np.random.rand()))
        time.sleep(0.1)


bandit.metadata.R2 = result.rsquared
bandit.metadata.AIC = result.aic

chart = sns.distplot(df.A)
chart.figure.savefig(output_dir + 'dist.png')

# save
df.head().to_csv(output_dir + 'datasample.csv')

today = datetime.date.today().strftime('%Y_%m_%d')

email = Email()
email.subject = '%s model results' % today

body = '''
Below is the result of the successful nightly model training script

Model Stats: %s
- Model:
- Adj. R2:
''' % result.model.formula, result.rsquared_adj

email.body = body
email.add_attachment(output_dir + 'datasample.csv')
email.add_attachment(output_dir + 'model_stats.txt')
email.add_attachment(output_dir + 'dist.png')
email.send('colin@yhathq.com')
