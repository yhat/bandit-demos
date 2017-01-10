from bandit import Bandit
import pandas as pd
import numpy as np
import time

bandit = Bandit()

t=0
while (t<15):
    for tag in ["a", "b", "c", "d", "e", "f", "g"]:
        bandit.report(tag, t, np.random.rand())
    time.sleep(0.1)
    print t
    t=t+1
