from bandit import Bandit
import pandas as pd
import numpy as np
import time

bandit = Bandit('colin', 'c4548110-cc4b-11e6-a5c5-0242ac110003','http://54.201.192.120/')

t=0
while (t<15):
    for tag in ["a", "b", "c", "d", "e", "f", "g"]:
        bandit.report(tag, t, np.random.rand())
    time.sleep(0.1)
    print t
    t=t+1

print(strftime("%Y-%m-%d %H:%M:%S", gmtime()))
print(metadata)
