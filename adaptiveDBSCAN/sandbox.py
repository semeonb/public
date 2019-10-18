#!/usr/bin/env python3
from datetime import datetime
import pandas as pd
import stDBSCAN


df = pd.DataFrame([[31.930203, 35.333588, datetime(2019, 1, 1, 12, 0, 0)],
                   [31.930203, 35.333588, datetime(2019, 1, 1, 12, 0, 1)],
                   [31.930203, 35.333588, datetime(2019, 1, 1, 12, 0, 2)],
                   [31.930203, 35.333588, datetime(2019, 1, 1, 12, 0, 10)],
                   [31.930203, 35.333188, datetime(2019, 5, 1, 12, 2, 0)],
                   [31.930203, 35.333588, datetime(2019, 1, 1, 12, 2, 5)],
                   [31.930203, 35.533588, datetime(2019, 1, 1, 12, 2, 8)],
                   [31.930203, 35.633588, datetime(2019, 1, 1, 12, 5, 8)]],
                  columns=['lat', 'lon', 'ts'])

st = stDBSCAN.Clusters(df=df, metersThresh=100, tempThresh=6000)
a = st.getClusters()

print(a)
