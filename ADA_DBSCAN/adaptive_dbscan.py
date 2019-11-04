#!/usr/bin/env python3

"""
This is python implementation of ST_DBSCAN algorithm
https://www.sciencedirect.com/science/article/pii/S0169023X06000218

input paramrters:
coordinates - list of shapely coordinates (lon, lat)
time_series - list of unix timestamps corresponding to each of those coordinates
"""

from functools import reduce
import pyproj
from shapely.geometry import linestring
from shapely import wkt

import pandas as pd

geodesic = pyproj.Geod(ellps='WGS84')


def calcDist(loc1, loc2):
    _, _, distance = geodesic.inv(lats1=loc1.y, lons1=loc1.x, lats2=loc2.y, lons2=loc1.x)
    return distance


def linestringToList(lstr):
    return list(lstr.coords)


def pdApply(funct, df, column):

    def _obj(row):
        return funct(row[column])

    return df.apply(_obj, axis=1)


class cluster(object):

    def __init__(self, coordinates, time_series, min_pts):

        self.coordinates = coordinates
        self.time_series = time_series
        self.min_pts = min_pts
        self.st_pairs = zip(coordinates, time_series)

    def _avg_speed(self, x, y):
        return calcDist(loc1=x[0], loc2=y[0])/(y[1] - x[1])

    def _speed_series(self):
        return reduce(function=self._avg_speed, sequence=self.st_pairs)


df = pd.read_csv('/Users/semeonbalagula/work/public/ADA_DBSCAN/data.csv')
df['pathLineString'] = df['pathLineString'].apply(wkt.loads)
df['tsFrom'] = pd.to_datetime(df['tsFrom'])
df['tsTo'] = pd.to_datetime(df['tsTo'])

# df['coords_list'] = df.apply list(df.loc[0]['pathLineString'].coords)

df['coords_list'] = pdApply(funct=linestringToList, df=df, column='pathLineString')

print(df)
