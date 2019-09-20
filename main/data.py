# -*- coding: utf-8 -*-
"""
Created on Thu Sep  5 20:55:28 2019

@author: Otto Alexander
"""

import pandas as pd

amzn = pd.read_pickle("../data/prices/AMZN.pkl")
fb = pd.read_pickle("../data/prices/FB.pkl")
goog = pd.read_pickle("../data/prices/GOOG.pkl")
googl = pd.read_pickle("../data/prices/GOOGL.pkl")
ibm = pd.read_pickle("../data/prices/IBM.pkl")
intc = pd.read_pickle("../data/prices/INTC.pkl")
msft = pd.read_pickle("../data/prices/MSFT.pkl")
t = pd.read_pickle("../data/prices/T.pkl")
X = pd.concat([amzn["Close"], fb["Close"]], axis=1)
symbols = ["AMZN Close", "FB Close"]
X.columns = symbols
y = pd.read_pickle("../data/prices/AAPL.pkl")