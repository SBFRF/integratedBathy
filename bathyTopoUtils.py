#! /usr/bin/env python

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import datetime as dt
import scipy
from scipy import interpolate
import pickle

from numpy.random import default_rng
rng = default_rng()

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

def combine_and_interpolate_masked_lidar_with_crosscheck(X,Y,Z,XX,YY,check_frac=None,method='linear'):
    points,values=[],[]
    good_values,good_points=[],[]
    check_values,check_points=[],[]
    for ii in range(len(X)):
        points.append(np.vstack((X[ii].flat[:],Y[ii].flat[:])).T)
        values.append(Z[ii].flat[:])
        good_values.append(values[ii][~values[ii].mask])
        good_points.append(points[ii][~values[ii].mask])
    #
    all_good_points=np.vstack(good_points)
    all_good_values=np.concatenate(good_values)
    extracted_values,extracted_points,Z_check=None,None,None
    if check_frac is not None and 0. < check_frac and check_frac < 1.:
        extracted_points,extracted_values,new_points,new_values,inds=extract_values_and_points(all_good_points,
                                                                                               all_good_values,
                                                                                               check_frac)
        all_good_values=new_values
        all_good_points=new_points
    Z_interp=scipy.interpolate.griddata(all_good_points,all_good_values,(XX,YY),method=method,fill_value=np.nan)
    if extracted_values is not None:
        Z_check=scipy.interpolate.griddata(all_good_points,all_good_values,extracted_points,method=method,fill_value=np.nan)
    return all_good_points,all_good_values,Z_interp,extracted_points,extracted_values,Z_check

def extract_values_and_points(points,values,fraction):
    inds = rng.choice(values.size,size=int(fraction*values.size),replace=False)
    extracted_values=values[inds]
    extracted_points=points[inds]
    new_values=np.delete(values,inds)
    new_points=np.delete(points,inds,axis=0)
    return extracted_points,extracted_values,new_points,new_values,inds

def plot_bathy2d(XX,YY,ZZ,error_fraction,error_estimate,datelabel,plotdir=os.path.curdir):
    assert os.path.isdir(plotdir), '{0} not a valid directory'.format(plotdir)
    
    mpl.rcParams['axes.linewidth']=2
    mpl.rcParams['axes.labelweight']='bold'
    mpl.rcParams['axes.titlesize']=16
    mpl.rcParams['axes.titleweight']='bold'
    mpl.rcParams['axes.labelsize']='large'
    mpl.rcParams['font.size']=16
    mpl.rcParams['xtick.labelsize']='large'
    mpl.rcParams['ytick.labelsize']='large'

    fig=plt.figure(figsize=(12,14))
    ax = fig.add_subplot(111)

    im=ax.pcolor(XX,YY,ZZ,shading='auto',cmap=plt.get_cmap('gist_earth'))
    title='Gridded Topo and Bathymetry for {0}\n'.format(datelabel)
    if error_fraction is not None and error_estimate is not None:
        title+='{0}% cross-validation error {1:7.5f}'.format(100.*float(error_fraction),float(error_estimate))
    ax.set_title(title)
                                                                                                         
    ax.set_xlabel('X FRF [m]')
    ax.set_ylabel('Y FRF [m]')

    fig.colorbar(im)

    plotname='griddedTopoBathy_{0}.png'.format(datelabel)
    plt.savefig(os.path.join(plotdir,plotname))

    return fig

