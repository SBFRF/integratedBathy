#! /usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt
import datetime as dt
import scipy
from scipy import interpolate
import pickle

def interpolate_masked_lidar(Xlidar,Ylidar,Zlidar,XX,YY,method='linear'):
    points=np.vstack((Xlidar.flat[:],Ylidar.flat[:])).T
    values=Zlidar.flat[:]
    good_values=values[~values.mask]
    good_points=points[~values.mask]
    Z_interp=scipy.interpolate.griddata(good_points,good_values,(XX,YY),method=method,fill_value=np.nan)
    return good_values,good_points,Z_interp

def combine_and_interpolate_masked_lidar(X,Y,Z,XX,YY,method='linear'):
    points,values=[],[]
    good_values,good_points=[],[]
    for ii in range(len(X)):
        points.append(np.vstack((X[ii].flat[:],Y[ii].flat[:])).T)
        values.append(Z[ii].flat[:])
        good_values.append(values[ii][~values[ii].mask])
        good_points.append(points[ii][~values[ii].mask])
    #
    all_good_points=np.vstack(good_points)
    all_good_values=np.concatenate(good_values)
    Z_interp=scipy.interpolate.griddata(all_good_points,all_good_values,(XX,YY),method=method,fill_value=np.nan)
    return all_good_points,all_good_values,Z_interp


def extend_alongshore(X,Y,Z):
    Z_nans=np.isnan(Z)
    Y_nonnan=np.where(~Z_nans,Y,np.inf)
    Y_start_index=np.argmin(Y_nonnan,axis=0)
    Y_nonnan=np.where(~Z_nans,Y,-np.inf)
    Y_end_index=np.argmax(Y_nonnan,axis=0)
    Z_tmp=Z.copy()
    for ii in range(Y.shape[1]): #fix this loop
        Z_tmp[:Y_start_index[ii],ii]=Z[Y_start_index[ii],ii]
        Z_tmp[Y_end_index[ii]:,ii]=Z[Y_end_index[ii],ii]
    return Y_start_index,Y_end_index,Z_tmp
