from sklearn.linear_model import LogisticRegression
import pandas as pd
import numpy as np
import os

from bandit import Bandit

bandit = Bandit('colin', 'c4548110-cc4b-11e6-a5c5-0242ac110003','http://54.201.192.120/')

dir_path = os.path.dirname(os.path.realpath(__file__))
df = pd.read_csv(os.path.join(dir_path, "LoanStats3a.csv"), skiprows=1)
df_head = df.head()

def is_poor_coverage(row):
    pct_null = float(row.isnull().sum()) / row.count()
    return pct_null < 0.8

df_head[df_head.apply(is_poor_coverage, axis=1)]
df = df[df.apply(is_poor_coverage, axis=1)]

df['year_issued'] = df.issue_d.apply(lambda x: int(x.split("-")[0]))
df_term = df[df.year_issued < 2012]

bad_indicators = [
    "Late (16-30 days)",
    "Late (31-120 days)",
    "Default",
    "Charged Off"
]

df_term['is_rent'] = df_term.home_ownership=="RENT"
df_term['is_bad'] = df_term.loan_status.apply(lambda x: x in bad_indicators)
features = ['last_fico_range_low', 'last_fico_range_high', 'is_rent']
glm = LogisticRegression()
glm.fit(df_term[features], df_term.is_bad)
score = glm.score(df_term[features], df_term.is_bad)

bandit.metadata.fit = score
bandit.metadata.rows = df_term.is_bad.shape[0]

glm.predict_log_proba(df_term[features].head())

# from ggplot import *

def calculate_score(log_odds):
    # 300 baseline + (40 points equals double risk) * odds
    return 300 + (40 / np.log(2)) * (-log_odds)

probs = glm.predict_proba(df_term[features])[:,1]
log_probs = glm.predict_log_proba(df_term[features])[:,1]
scores = calculate_score(log_probs)
# qplot(scores)
# qplot(probs, scores)


from yhat import Yhat, YhatModel

class LoanModel(YhatModel):
    REQUREMENTS = ['numpy==1.11.2', 'scikit-learn==0.18.1', 'pandas==0.19.1']
    def execute(self, data):
        data['is_rent'] = data['home_ownership']=="RENT"
        data = {k: [v] for k,v in data.items()}
        data = pd.DataFrame(data)
        data = data[features]
        prob = glm.predict_proba(data)[0][1]
        if prob > 0.3:
            decline_code = "Credit score too low"
        else:
            decline_code = ""
        odds = glm.predict_log_proba(data)[0][1]
        score = calculate_score(odds)

        output = {
            "prob_default": [prob],
            "decline_code": [decline_code],
            "score": [score]
        }

        return output

df_term[features].head()

test = {
    "last_fico_range_low": 705,
    "last_fico_range_high": 732,
    "home_ownership": "MORTGAGE"
}

LoanModel().execute(test)

yh = Yhat("colin", "d325fc5bcb83fc197ee01edb58b4b396", "https://sandbox.c.yhat.com/")
yh.deploy("LendingClub", LoanModel, globals(), True)
