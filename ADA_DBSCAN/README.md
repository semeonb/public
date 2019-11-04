# Adaptive DBSCAN

An adaptation of DBSCAN algorithm to geospatial and temporal data.
Receives pandas Dataframe with latitiude, longtitude and timestamp data.

Parameters:

metersThresh - epsilon in meters for DBSCAN algorithm <br />
tempThresh - maximum possible time in seconds between 2 points in the same cluster <br />
minNeigh - minimal number of points in a single cluster [default: 2] <br />
latCol - latitude column name in pandas dataframe <br />
lonCom - Longtitude column name in pandas dataframe <br />
