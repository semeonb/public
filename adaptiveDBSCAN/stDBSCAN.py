import pandas as pd

import scipy.spatial as spatial
from sklearn.cluster import DBSCAN
from math import sin, cos
import pyproj

import utils


class Clusters(object):

    def __init__(self, df, metersThresh, tempThresh, minNeigh=2, spatialClusterCol='s_cluster',
                 temporalClusterCol='t_cluster', latCol='lat', lonCol='lon', timeCol='ts'):
        self.metersThresh = metersThresh  # Maximal radius around the core point
        self.tempThresh = tempThresh  # maximal number of seconds, points might be one from another
        self.minNeigh = minNeigh  # Minimal neighbours for a point to have to form a cluster
        self.spatialClusterCol = spatialClusterCol  # Name of the spatial cluster column to give
        self.temporalClusterCol = temporalClusterCol  # Name of the temporal cluster column to give
        self.latCol = latCol  # Latitude column name
        self.lonCol = lonCol  # Longitude column name
        self.timeCol = timeCol  # time column in the dataframe
        self.df = df  # dataframe to build the clustering on
        self.kdtree = spatial.cKDTree(df[[latCol, lonCol]])  # Build kd-tree based on dataframe
        self.noise = -1
        self.geodesic = pyproj.Geod(ellps='WGS84')

    def splitFrameToPeriods(self):
        periodId = 0
        indexList = []
        neighborsList = []
        dfSorted = self.df.sort_values(by=self.timeCol).copy()
        dfSorted['delta'] = (self.df[self.timeCol] - self.df[self.timeCol].shift()).fillna(0)
        for index, values in dfSorted.iterrows():
            if values['delta'].total_seconds() <= self.tempThresh:
                neighborsList.append(index)
            else:
                indexList.append([periodId, neighborsList])
                neighborsList = []
                neighborsList.append(index)
                periodId += 1
        indexList.append([periodId, neighborsList])
        return [[i[0], self.df.loc[i[1]]] for i in indexList]

    def cluster(self, df):
        cnt = df.shape[0]
        center = [sum(df[self.latCol])/cnt, sum(df[self.lonCol])/cnt]

        def _calcDist(row):
            fwd_azimuth, back_azimuth, distance = \
                self.geodesic.inv(lats1=center[0], lons1=center[1], lats2=row[self.latCol],
                                  lons2=row[self.lonCol])
            return (cos(fwd_azimuth) * distance, sin(fwd_azimuth) * distance)

        df['xy'] = df.apply(_calcDist, axis=1)
        df[['x', 'y']] = pd.DataFrame(df['xy'].tolist(), index=df.index)
        dbscan = DBSCAN(eps=self.metersThresh, min_samples=self.minNeigh)

        dfClusters = pd.DataFrame(dbscan.fit_predict(df[['x', 'y']].values),
                                  columns=[self.spatialClusterCol], index=df.index.values)
        return dfClusters

    def calculate(self, data):
        ix = data[0]
        df = data[1].copy()
        df[self.temporalClusterCol] = ix
        if df.shape[0] >= self.minNeigh:
            return pd.merge(left=df, right=self.cluster(df.copy()), left_index=True,
                            right_index=True)
        else:
            df[self.spatialClusterCol] = self.noise
            return df

    def getClusters(self):
        results = []
        dfList = self.splitFrameToPeriods()
        results.append(utils.run_in_procs(func=self.calculate, func_args=dfList,
                                          processes=len(dfList)))
        return pd.concat(results[0])
