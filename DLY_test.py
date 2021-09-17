import datetime as DT
import os, sys
import warnings
import netCDF4 as nc
import numpy as np
from sblib import geoprocess as gp
from sblib import sblib as sb
from getdatatestbed.getDataFRF import getObs
import makenc
from matplotlib import pyplot as plt
from scipy.interpolate import griddata
from sblib import gridTools as gT
import pandas as pd
from bsplineFunctions import bspline_pertgrid
import matplotlib as m
import pyproj
import datetime as DT
import wrappers
from scaleCinterp_python.DEM_generator import DEM_generator
import MakeUpdatedBathyDEM as mBATHY

"""
# how much does my new stuff look like what allison had?
bathyDict = {}
nc_url = '/home/david/BathyTroubleshooting/ncFiles/2015/CMTB-integratedBathyProduct_survey_201510.nc'
bathy = nc.Dataset(nc_url)
bathy_times = nc.num2date(bathy.variables['time'][:], bathy.variables['time'].units, bathy.variables['time'].calendar)
surveyNumber = bathy.variables['surveyNumber'][:]


elevation = bathy.variables['elevation'][:]
xFRF = bathy.variables['xFRF'][:]
yFRF = bathy.variables['yFRF'][:]

# why is this not masked?!!!!!
updateTimes = bathy.variables['updateTime'][:]
fig_loc = '/home/david/BathyTroubleshooting'

for tt in range(0, np.shape(elevation)[0]):

    dSTR = DT.datetime.strftime(bathy_times[tt], '%Y-%m-%dT%H:%M:%SZ')
    # look at my two surveys...
    fig_name = 'testBathy_' + dSTR[0:10] + '.png'
    plt.figure()
    plt.pcolor(xFRF, yFRF, elevation[tt, :, :], cmap=plt.cm.jet, vmin=-13, vmax=5)
    cbar = plt.colorbar()
    cbar.set_label('(m)')
    axes = plt.gca()
    axes.set_xlim([-50, 700])
    axes.set_ylim([-50, 1050])
    plt.xlabel('xFRF (m)')
    plt.ylabel('yFRF (m)')
    plt.savefig(os.path.join(fig_loc, fig_name))
    plt.close()


    # alongshore transect plots
    x_loc_check1 = int(100)
    x_loc_check2 = int(200)
    x_loc_check3 = int(350)
    x_check1 = np.where(xFRF == x_loc_check1)[0][0]
    x_check2 = np.where(xFRF == x_loc_check2)[0][0]
    x_check3 = np.where(xFRF == x_loc_check3)[0][0]

    # plot X and Y transects from newZdiff to see if it looks correct?
    fig_name = 'testBathy_' + dSTR[0:10] + '_Ytrans' + '.png'
    fig = plt.figure(figsize=(8, 9))
    ax1 = plt.subplot2grid((3, 1), (0, 0), colspan=1)
    ax1.plot(yFRF, elevation[tt, :, x_check1], 'r')
    ax1.set_xlabel('Alongshore - $y$ ($m$)', fontsize=16)
    ax1.set_ylabel('Elevation ($m$)', fontsize=16)
    ax1.set_title('$X=%s$' % (str(x_loc_check1)), fontsize=16)
    for tick in ax1.xaxis.get_major_ticks():
        tick.label.set_fontsize(14)
    for tick in ax1.yaxis.get_major_ticks():
        tick.label.set_fontsize(14)
    ax1.tick_params(labelsize=14)
    ax1.text(0.10, 0.95, '(a)', horizontalalignment='left', verticalalignment='top',
             transform=ax1.transAxes, fontsize=16)
    ax1.set_xlim([-50, 1050])

    ax2 = plt.subplot2grid((3, 1), (1, 0), colspan=1)
    ax2.plot(yFRF, elevation[tt, :, x_check2], 'r')
    ax2.set_xlabel('Alongshore - $y$ ($m$)', fontsize=16)
    ax2.set_ylabel('Elevation ($m$)', fontsize=16)
    ax2.set_title('$X=%s$' % (str(x_loc_check2)), fontsize=16)
    for tick in ax2.xaxis.get_major_ticks():
        tick.label.set_fontsize(14)
    for tick in ax2.yaxis.get_major_ticks():
        tick.label.set_fontsize(14)
    ax2.tick_params(labelsize=14)
    ax2.text(0.10, 0.95, '(b)', horizontalalignment='left', verticalalignment='top',
             transform=ax2.transAxes, fontsize=16)
    ax2.set_xlim([-50, 1050])

    ax3 = plt.subplot2grid((3, 1), (2, 0), colspan=1)
    ax3.plot(yFRF, elevation[tt, :, x_check3], 'r')
    ax3.set_xlabel('Alongshore - $y$ ($m$)', fontsize=16)
    ax3.set_ylabel('Elevation ($m$)', fontsize=16)
    ax3.set_title('$X=%s$' % (str(x_loc_check3)), fontsize=16)
    for tick in ax3.xaxis.get_major_ticks():
        tick.label.set_fontsize(14)
    for tick in ax3.yaxis.get_major_ticks():
        tick.label.set_fontsize(14)
    ax3.tick_params(labelsize=14)
    ax3.text(0.10, 0.95, '(c)', horizontalalignment='left', verticalalignment='top',
             transform=ax3.transAxes, fontsize=16)
    ax3.set_xlim([-50, 1050])

    fig.subplots_adjust(wspace=0.4, hspace=0.1)
    fig.tight_layout(pad=1, h_pad=2.5, w_pad=1, rect=[0.0, 0.0, 1.0, 0.925])
    fig.savefig(os.path.join(fig_loc, fig_name), dpi=300)
    plt.close()

    # cross-shore transect plots
    y_loc_check1 = int(250)
    y_loc_check2 = int(500)
    y_loc_check3 = int(750)
    y_check1 = np.where(yFRF == y_loc_check1)[0][0]
    y_check2 = np.where(yFRF == y_loc_check2)[0][0]
    y_check3 = np.where(yFRF == y_loc_check3)[0][0]
    # plot a transect going in the cross-shore just to check it
    fig_name = 'testBathy_' + dSTR[0:10] + '_Xtrans' + '.png'

    fig = plt.figure(figsize=(8, 9))
    ax1 = plt.subplot2grid((3, 1), (0, 0), colspan=1)
    ax1.plot(xFRF, elevation[tt, y_check1, :], 'b')
    ax1.set_xlabel('Cross-shore - $x$ ($m$)', fontsize=16)
    ax1.set_ylabel('Elevation ($m$)', fontsize=16)
    ax1.set_title('$Y=%s$' % (str(y_loc_check1)), fontsize=16)
    for tick in ax1.xaxis.get_major_ticks():
        tick.label.set_fontsize(14)
    for tick in ax1.yaxis.get_major_ticks():
        tick.label.set_fontsize(14)
    ax1.tick_params(labelsize=14)
    ax1.text(0.10, 0.95, '(a)', horizontalalignment='left', verticalalignment='top',
             transform=ax1.transAxes, fontsize=16)

    ax2 = plt.subplot2grid((3, 1), (1, 0), colspan=1)
    ax2.plot(xFRF, elevation[tt, y_check2, :], 'b')
    ax2.set_xlabel('Cross-shore - $x$ ($m$)', fontsize=16)
    ax2.set_ylabel('Elevation ($m$)', fontsize=16)
    ax2.set_title('$Y=%s$' % (str(y_loc_check2)), fontsize=16)
    for tick in ax2.xaxis.get_major_ticks():
        tick.label.set_fontsize(14)
    for tick in ax2.yaxis.get_major_ticks():
        tick.label.set_fontsize(14)
    ax2.tick_params(labelsize=14)
    ax2.text(0.10, 0.95, '(b)', horizontalalignment='left', verticalalignment='top',
             transform=ax2.transAxes, fontsize=16)

    ax3 = plt.subplot2grid((3, 1), (2, 0), colspan=1)
    ax3.plot(xFRF, elevation[tt, y_check3, :], 'b')
    ax3.set_xlabel('Cross-shore - $x$ ($m$)', fontsize=16)
    ax3.set_ylabel('Elevation ($m$)', fontsize=16)
    ax3.set_title('$Y=%s$' % (str(y_loc_check3)), fontsize=16)
    for tick in ax3.xaxis.get_major_ticks():
        tick.label.set_fontsize(14)
    for tick in ax3.yaxis.get_major_ticks():
        tick.label.set_fontsize(14)
    ax3.tick_params(labelsize=14)
    ax3.text(0.10, 0.95, '(c)', horizontalalignment='left', verticalalignment='top',
             transform=ax3.transAxes, fontsize=16)

    fig.subplots_adjust(wspace=0.4, hspace=0.1)
    fig.tight_layout(pad=1, h_pad=2.5, w_pad=1, rect=[0.0, 0.0, 1.0, 0.925])
    fig.savefig(os.path.join(fig_loc, fig_name), dpi=300)
    plt.close()


t = 1
"""


"""
# check the integrated bathymetry from those days allison showed
dSTR_s = '2016-10-22T00:00:00Z'
dSTR_sr = '2016-10-19T00:00:00Z'
d_s = DT.datetime.strptime(dSTR_s, '%Y-%m-%dT%H:%M:%SZ')
d_sr = DT.datetime.strptime(dSTR_sr, '%Y-%m-%dT%H:%M:%SZ')
backgroundDict = {}
nc_url = 'http://134.164.129.55/thredds/dodsC/cmtb/integratedBathyProduct/survey/survey.ncml'
old_bathy = nc.Dataset(nc_url)
ob_times = nc.num2date(old_bathy.variables['time'][:], old_bathy.variables['time'].units, old_bathy.variables['time'].calendar)

# find newest time prior to this
t_mask = (ob_times <= d_s)  # boolean true/false of time
t_idx = np.where(t_mask)[0][-1]  # I want the MOST RECENT ONE - i.e., the last one
backgroundDict['elevation'] = old_bathy.variables['elevation'][t_idx, :]
backgroundDict['xFRF'] = old_bathy.variables['xFRF'][:]
backgroundDict['yFRF'] = old_bathy.variables['yFRF'][:]
backgroundDict['time'] = ob_times[t_idx]

newDict = mBATHY.getSurveyData(d_sr, d_s)

# plot this this and save the results to a test file.
fig_loc = '/home/david/BathyTroubleshooting'


# zoomed in pcolor plot on AOI
fig_name = 'cBathyDEM_' + dSTR_s[0:10] + '.png'
plt.figure()
plt.pcolor(backgroundDict['xFRF'], backgroundDict['yFRF'], backgroundDict['elevation'], cmap=plt.cm.jet, vmin=-13, vmax=5)
cbar = plt.colorbar()
cbar.set_label('(m)')
axes = plt.gca()
axes.set_xlim([-50, 700])
axes.set_ylim([-50, 1050])
plt.xlabel('xFRF (m)')
plt.ylabel('yFRF (m)')
plt.savefig(os.path.join(fig_loc, fig_name))
plt.close()


#alongshore transect plots
x_loc_check1 = int(100)
x_loc_check2 = int(200)
x_loc_check3 = int(350)
x_check1 = np.where(backgroundDict['xFRF'] == x_loc_check1)[0][0]
x_check2 = np.where(backgroundDict['xFRF'] == x_loc_check2)[0][0]
x_check3 = np.where(backgroundDict['xFRF'] == x_loc_check3)[0][0]

# plot X and Y transects from newZdiff to see if it looks correct?
fig_name = 'cBathyDEM_' + dSTR_s[0:10] + '_Ytrans' + '.png'
fig = plt.figure(figsize=(8, 9))
ax1 = plt.subplot2grid((3, 1), (0, 0), colspan=1)
ax1.plot(backgroundDict['yFRF'], backgroundDict['elevation'][:, x_check1], 'r')
ax1.set_xlabel('Alongshore - $y$ ($m$)', fontsize=16)
ax1.set_ylabel('Elevation ($m$)', fontsize=16)
ax1.set_title('$X=%s$' % (str(x_loc_check1)), fontsize=16)
for tick in ax1.xaxis.get_major_ticks():
    tick.label.set_fontsize(14)
for tick in ax1.yaxis.get_major_ticks():
    tick.label.set_fontsize(14)
ax1.tick_params(labelsize=14)
ax1.text(0.10, 0.95, '(a)', horizontalalignment='left', verticalalignment='top',
         transform=ax1.transAxes, fontsize=16)
ax1.set_xlim([-50, 1050])

ax2 = plt.subplot2grid((3, 1), (1, 0), colspan=1)
ax2.plot(backgroundDict['yFRF'], backgroundDict['elevation'][:, x_check2], 'r')
ax2.set_xlabel('Alongshore - $y$ ($m$)', fontsize=16)
ax2.set_ylabel('Elevation ($m$)', fontsize=16)
ax2.set_title('$X=%s$' % (str(x_loc_check2)), fontsize=16)
for tick in ax2.xaxis.get_major_ticks():
    tick.label.set_fontsize(14)
for tick in ax2.yaxis.get_major_ticks():
    tick.label.set_fontsize(14)
ax2.tick_params(labelsize=14)
ax2.text(0.10, 0.95, '(b)', horizontalalignment='left', verticalalignment='top',
         transform=ax2.transAxes, fontsize=16)
ax2.set_xlim([-50, 1050])

ax3 = plt.subplot2grid((3, 1), (2, 0), colspan=1)
ax3.plot(backgroundDict['yFRF'], backgroundDict['elevation'][:, x_check3], 'r')
ax3.set_xlabel('Alongshore - $y$ ($m$)', fontsize=16)
ax3.set_ylabel('Elevation ($m$)', fontsize=16)
ax3.set_title('$X=%s$' % (str(x_loc_check3)), fontsize=16)
for tick in ax3.xaxis.get_major_ticks():
    tick.label.set_fontsize(14)
for tick in ax3.yaxis.get_major_ticks():
    tick.label.set_fontsize(14)
ax3.tick_params(labelsize=14)
ax3.text(0.10, 0.95, '(c)', horizontalalignment='left', verticalalignment='top',
         transform=ax3.transAxes, fontsize=16)
ax3.set_xlim([-50, 1050])

fig.subplots_adjust(wspace=0.4, hspace=0.1)
fig.tight_layout(pad=1, h_pad=2.5, w_pad=1, rect=[0.0, 0.0, 1.0, 0.925])
fig.savefig(os.path.join(fig_loc, fig_name), dpi=300)
plt.close()


# whats the deal with the related survey - make the grid just from the survey and look at it....


t = 1
"""


"""
test = sb.FRFcoord(861, 200)
print test['Lat']
print test['Lon']
t = 1
"""

# makeUpdatedBathyDEM test script

"""
# wrote this to a netCDF file, so I don't need to do this anymore!!!
# Hard Code Resolution to 5 m (dx & dy) - want whole number FRF coords

# start with background DEM
# **NOTE: will have to edit this to look for more recent bathy first!!!!!**
# hard code my corners....
LLHC = (-50, -200)
URHC = (1305, 5000)  # 1305 is the location of the 11-m awac rounded UP to the nearest 5 m
dx = 5
dy = 5
coord_system = 'FRF'
temp = gT.makeBackgroundBathyCorners(LLHC, URHC, dx, dy, coord_system=coord_system)
Zi = temp['bottomElevation']
xFRFi = temp['xFRF']
yFRFi = temp['yFRF']

# first plot - original bathy
fig_name = 'OrigBathy.png'
fig_loc = 'C:\Users\dyoung8\Desktop\David Stuff\Projects\CSHORE\Bathy Interpolation\Test Figures'
plt.contourf(xFRFi, yFRFi, Zi)
plt.axis('equal')
plt.xlabel('xFRF')
plt.ylabel('yFRF')
plt.colorbar()
plt.savefig(os.path.join(fig_loc, fig_name))
plt.close()

# save this as an nc file on my PC?
Zi_vec = Zi.reshape((1, Zi.shape[0] * Zi.shape[1]))[0]
xFRFi_vec = xFRFi.reshape((1, xFRFi.shape[0] * xFRFi.shape[1]))[0]
yFRFi_vec = yFRFi.reshape((1, yFRFi.shape[0] * yFRFi.shape[1]))[0]

# convert FRF coords to lat/lon
test = gp.FRF2ncsp(xFRFi_vec, yFRFi_vec)
temp = gp.ncsp2LatLon(test['StateplaneE'], test['StateplaneN'])
lat_vec = temp['lat']
lon_vec = temp['lon']

lat = lat_vec.reshape(xFRFi.shape[0], xFRFi.shape[1])
lon = lon_vec.reshape(xFRFi.shape[0], xFRFi.shape[1])

nc_dict = {}
nc_dict['elevation'] = Zi
nc_dict['xFRF'] = xFRFi[0, :]
nc_dict['yFRF'] = yFRFi[:, 1]
nc_dict['latitude'] = lat
nc_dict['longitude'] = lon

nc_loc = 'C:\Users\dyoung8\Desktop\David Stuff\Projects\CSHORE\Bathy Interpolation\TestNCfiles'
nc_name = 'backgroundDEMt0.nc'

global_yaml = 'C:\Users\dyoung8\PycharmProjects\makebathyinterp\yamls\BATHY\FRFt0_global.yml'
var_yaml = 'C:\Users\dyoung8\PycharmProjects\makebathyinterp\yamls\BATHY\FRFt0_var.yml'

makenc.makenc_t0BATHY(os.path.join(nc_loc, nc_name), nc_dict, globalYaml=global_yaml, varYaml=var_yaml)
"""


# list of inputs!!!!!
x_smooth = 40  # scale c interp x-direction smoothing
y_smooth = 100  # scale c interp y-direction smoothing
# splinebctype - this is the type of spline you want to force
# options are....
# 2 - second derivative goes to zero at boundary
# 1 - first derivative goes to zero at boundary
# 0 - value is zero at boundary
# 10 - force value and derivative(first?!?) to zero at boundary
splinebctype = 10
lc = 4  # spline smoothing constraint value
dxm = 1  # coarsening of the grid for spline (e.g., 2 means calculate with a dx that is 2x input dx)
# can be tuple if you want to do dx and dy seperately (dxm, dym), otherwise dxm is used for both
dxi = 1  # fining of the grid for spline (e.g., 0.1 means return spline on a grid that is 10x input dx)
# as with dxm, can be a tuple if you want separate values for dxi and dyi
targetvar = 0.8 # this is the target variance used in the spline function.
wbysmooth = 300  # y-edge smoothing scale
wbxsmooth = 100  # x-edge smoothing scale
# It is used in conjunction with the MSE from splineCinterp to compute the spline weights (wb)
dSTR_s = '2008-01-01T00:00:00Z'
dSTR_e = '2018-06-01T00:00:00Z'

dir_loc = '/home/david/BathyTroubleshooting/BackgroundFiles'

# this is where I am going to save the monthy nc files

scalecDict = {}
scalecDict['x_smooth'] = x_smooth
scalecDict['y_smooth'] = y_smooth
splineDict = {}
splineDict['splinebctype'] = splinebctype
splineDict['lc'] = lc
splineDict['dxm'] = dxm
splineDict['dxi'] = dxi
splineDict['targetvar'] = targetvar
splineDict['wbysmooth'] = wbysmooth
splineDict['wbxsmooth'] = wbxsmooth

# wrappers.makeBathyCBATHY(dSTR_s, dSTR_e, dir_loc, scalecDict=scalecDict, splineDict=splineDict)
# wrappers.makeBathySurvey(dSTR_s, dSTR_e, dir_loc, scalecDict=scalecDict, splineDict=splineDict, plot=1)

# ncml_url = 'https://chlthredds.erdc.dren.mil/thredds/dodsC/frf/projects/bathyduck/data/cbathy_old/cbathy.ncml'
# temp = mBATHY.getGridded(ncml_url=ncml_url,  scalecDict=scalecDict, splineDict=splineDict, plot=1)

# mBATHY.makeUpdatedBATHY_grid(dSTR_s, dSTR_e, dir_loc, ncml_url, scalecDict=scalecDict, splineDict=splineDict, plot=1)
# mBATHY.makeUpdatedBATHY_transects(dSTR_s, dSTR_e, dir_loc, scalecDict=scalecDict, splineDict=splineDict, plot=1)


gT.makeTimeMeanBackgroundBathy(dir_loc, dSTR_s=dSTR_s, dSTR_e=dSTR_e, scalecDict=scalecDict, splineDict=None, plot=None)


t = 1


"""
# make plots for spike
nc_b_loc = 'C:\Users\dyoung8\Desktop\David Stuff\Projects\CSHORE\Bathy Interpolation\TestNCfiles'
nc_b_name = 'backgroundDEMt0_TimeMean.nc'
cs_array_url = 'http://134.164.129.55/thredds/dodsC/FRF/oceanography/waves/8m-array/2017/FRF-ocean_waves_8m-array_201707.nc'
fig_loc = 'C:\Users\dyoung8\Desktop\David Stuff\Projects\CSHORE\Bathy Interpolation\Test Figures\TimeMeanFigs'

# pull the original background DEM
old_bathy = nc.Dataset(os.path.join(nc_b_loc, nc_b_name))
Z = old_bathy.variables['elevation'][:]
xFRF_vec = old_bathy.variables['xFRF'][:]
yFRF_vec = old_bathy.variables['yFRF'][:]

# plot the bathymetry before and after....
# where is the cross shore array?
test = nc.Dataset(cs_array_url)
Lat = test['latitude'][:]
Lon = test['longitude'][:]
# convert to FRF
temp = gp.FRFcoord(Lon, Lat)
CSarray_X = temp['xFRF']
CSarray_Y = temp['yFRF']

rows, cols = np.shape(Z)

# original data
y_ind = int(np.where(yFRF_vec == 900)[0][0])
fig_name = 'backgroundDEM_TM_CStrans_900' + '.png'
plt.figure()
plt.plot(xFRF_vec, Z[y_ind, :])
plt.plot([CSarray_X, CSarray_X], [np.min(Z[y_ind, :]), np.max(Z[y_ind, :])], 'r', label='8m-array')
plt.xlabel('xFRF (m)')
plt.ylabel('yFRF (m)')
plt.legend()
plt.savefig(os.path.join(fig_loc, fig_name))
plt.close()
"""

# force the survey to start at the first of the month and end at the last of the month!!!!
# dSTR_s = dSTR_s[0:7] + '-01T00:00:00Z'

"""
# check that thing for Pat
url = 'http://chlthredds.erdc.dren.mil/thredds/dodsC/frf/oceanography/waves/CS02-SBE26/CS02-SBE26.ncml'
test = nc.Dataset(url)

nc_vars = [var for var in test.variables]
time = test.variables['time'][:]

def roundtime(dt=None, roundto=60):
    seconds = (dt - dt.min).seconds
    # // is a floor division, not a comment on following line:
    rounding = (seconds + roundto / 2) // roundto * roundto
    return dt + DT.timedelta(0, rounding - seconds, -dt.microsecond)

alt_time = nc.num2date(test.variables['time'][:], test.variables['time'].units, test.variables['time'].calendar)
for num in range(0, len(alt_time)):
    alt_time[num] = roundtime(alt_time[num], roundto=1 * 60)
"""



"""
# fix FRF coord test
xFRF = 0
yFRF = 0

temp = gp.FRF2ncsp(xFRF, yFRF)
E = temp['StateplaneE']
N = temp['StateplaneN']

temp = gp.ncsp2LatLon(E, N)
lat = temp['lat']
lon = temp['lon']

temp = gp.LatLon2utm(lat, lon)

utmE = temp['utmE']
utmN = temp['utmN']
zl = temp['zl']
zn = temp['zn']

# FRF
p1 = -9000
p2 = 500
test1 = gp.FRFcoord(p1, p2)

# SP
p1 = test1['StateplaneE']
p2 = test1['StateplaneN']
test2 = gp.FRFcoord(p1, p2)

# LL
p1 = test1['Lat']
p2 = test1['Lon']
test3 = gp.FRFcoord(p2, p1)

# UTM
p1 = test1['utmE']
p2 = test1['utmN']
test4 = gp.FRFcoord(p1, p2)
"""

# ok, this is going to be my test script for splining a perturbation tile back into the background DEM

"""
# get a grid
x0 = 0
y0 = 0
x1 = 2000
y1 = 1500
coord_system = 'frf'

dx = 10
dy = 20
LLHC = (x0, y0)
URHC = (x1, y1)
grid = gT.makeBackgroundBathyCorners(LLHC, URHC, dx, dy, coord_system)


# set bottom elevation to zero and replace it with a perturbation (do like a parabola with some noise in it...)
Zi = 0*grid['bottomElevation']
wb = 0*grid['bottomElevation']
splinebctype = None
lc = 2
dxm = [4, 1]
dxi = 1

x = grid['xFRF']
y = grid['yFRF']
rows = np.shape(Zi)[0]
cols = np.shape(Zi)[1]

# make my fake perturbation
x_cen = 0.5*(x1 - x0)
y_cen = 0.5*(y1 - y0)
sig_x = 400
sig_y = 500
sig_n = 0.5
A = 2.0

for ii in range(0, rows):
    for jj in range(0, cols):
        # make my gaussian
        val = np.square(x[ii, jj] - x_cen)*(1/float(2*np.square(sig_x))) + np.square(y[ii, jj] - y_cen)*(1/float(2*np.square(sig_y)))
        # add some noise
        noise = np.random.normal(0, sig_n)
        Zi[ii, jj] = 1 - (A*np.exp(-1*val) + noise)
        wb[ii, jj] = np.random.normal(0.95, 0.1)

wb[wb >= 1] = 0.999
wb[0,:] = 0
wb[-1, :] = 0


Zi_vec = Zi.reshape((1, Zi.shape[0] * Zi.shape[1]))[0]
wb_vec = wb.reshape((1, wb.shape[0] * wb.shape[1]))[0]
x_vec = x.reshape((1, x.shape[0] * x.shape[1]))[0]
y_vec = y.reshape((1, y.shape[0] * y.shape[1]))[0]

columns = ['x', 'y', 'Zi', 'wb']
df = pd.DataFrame(index=range(0, np.size(Zi_vec)), columns=columns)

sloc = 'C:\Users\RDCHLDLY\Desktop\David Stuff\Projects\CSHORE\Bathy Interpolation\Test Figures'
sname = 'test_data.csv'

df['x'] = x_vec
df['y'] = y_vec
df['Zi'] = Zi_vec
df['wb'] = wb_vec

df.to_csv(os.path.join(sloc, sname), index=False)
"""

"""
# saved one to csv so I could compare to matlab, and also so I don't have to make a new one every time.

floc = 'C:\Users\RDCHLDLY\Desktop\David Stuff\Projects\CSHORE\Bathy Interpolation\Test Figures'
fname = 'test_data.csv'

df = pd.read_csv(os.path.join(floc, fname))
df = pd.DataFrame(df)

x_vec = np.asarray(df['x'])
y_vec = np.asarray(df['y'])
Zi_vec = np.asarray(df['Zi'])
wb_vec = np.asarray(df['wb'])

dim1 = 75
dim2 = 200

x = x_vec.reshape(dim1, dim2)
y = y_vec.reshape(dim1, dim2)
Zi = Zi_vec.reshape(dim1, dim2)
wb = wb_vec.reshape(dim1, dim2)


splinebctype = None
lc = 2
dxm = [2, 1]
dxi = 1



#plot her up....
fig_name = 'FakeZi.png'
fig_loc = 'C:\Users\RDCHLDLY\Desktop\David Stuff\Projects\CSHORE\Bathy Interpolation\Test Figures'
plt.contourf(x, y, Zi)
plt.axis('equal')
plt.xlabel('xFRF')
plt.ylabel('yFRF')
plt.colorbar()
plt.savefig(os.path.join(fig_loc, fig_name))
plt.close()


#ok, go time
newZi = bspline_pertgrid(Zi, wb, splinebctype=10, lc=2, dxm=2, dxi=1)

Ny, Nx = np.shape(x)


#plot her up....
fig_name = 'FakeZiNew.png'
fig_loc = 'C:\Users\RDCHLDLY\Desktop\David Stuff\Projects\CSHORE\Bathy Interpolation\Test Figures'
plt.contourf(x, y, newZi)
plt.axis('equal')
plt.xlabel('xFRF')
plt.ylabel('yFRF')
plt.colorbar()
plt.savefig(os.path.join(fig_loc, fig_name))
plt.close()

#plot her up....
fig_name = 'FakeZiNew_Transects.png'
fig_loc = 'C:\Users\RDCHLDLY\Desktop\David Stuff\Projects\CSHORE\Bathy Interpolation\Test Figures'
plt.plot(x[0, :], newZi[int(np.fix(Ny/2)), :], 'r')
plt.plot(x[0, :], Zi[int(np.fix(Ny/2)), :], 'b')
plt.xlabel('xFRF')
plt.ylabel('Elevation')
plt.savefig(os.path.join(fig_loc, fig_name))
plt.close()

"""

"""
# ok, what we are going to do here is make a .csv file out of the .nc file of the transect
# that Meg used in the practice code she gave us....

floc = 'C:\Users\dyoung8\Desktop\David Stuff\Projects\CSHORE\Bathy Interpolation\MegsPracticeFiles'
fname = 'FRF_20120214_1069_FRF_NAVD88_LARC_GPS_UTC_v20160402.csv'

d_s = DT.datetime.strptime('2012-02-17T12:00:00Z', '%Y-%m-%dT%H:%M:%SZ')
d_e = DT.datetime.strptime('2012-02-18T12:00:00Z', '%Y-%m-%dT%H:%M:%SZ')
frf_Data = getObs(d_s, d_e)
bathy_data = frf_Data.getBathyTransectFromNC()

elev = bathy_data['elevation']
xFRF = bathy_data['xFRF']
yFRF = bathy_data['yFRF']
lat = bathy_data['lat']
lon = bathy_data['lon']
prof_num = bathy_data['profileNumber']
survey_num = bathy_data['surveyNumber']

columns = ['notUsed1', 'notUsed2', 'notUsed3', 'lat', 'lon', 'notUsed4', 'notUsed5', 'xFRF', 'yFRF', 'bottomElevation']
df = pd.DataFrame(index=range(0, np.size(elev)), columns=columns)

df['lat'] = lat
df['lon'] = lon
df['xFRF'] = xFRF
df['yFRF'] = yFRF
df['bottomElevation'] = elev

df.to_csv(os.path.join(floc, fname), index=False)

t = 1
"""

"""
# this is just to test the makeBackgroundBathy script I wrote
x0 = 901951.6805
y0 = 274093.156
temp = gp.ncsp2LatLon(x0, y0)
origin = (temp['lat'], temp['lon'])

geo_ang = 51.8
dx = 10
dy = 20
ni = 30
nj = 10
coord_system = 'latlon'

test = gT.makeBackgroundBathyAzimuth(origin, geo_ang, dx, dy, ni, nj, coord_system)

# check to see if this actually worked...
fig_name = 'newGrid_Azimuth_LL.png'
fig_loc = 'C:\Users\RDCHLDLY\Desktop\David Stuff\Projects\CSHORE\Bathy Interpolation\Test Figures'
plt.contourf(test['longitude'], test['latitude'], test['bottomElevation'])
plt.axis('equal')
plt.xlabel('lon')
plt.ylabel('lat')
plt.savefig(os.path.join(fig_loc, fig_name))
plt.close()
"""

"""
x0 = 0
y0 = 0
x1 = 2000
y1 = 1500

dx = 10
dy = 20
coord_system = 'latlon'

if coord_system == 'utm':
    temp = gp.FRF2ncsp(x0, y0)
    temp2 = gp.ncsp2utm(temp['StateplaneE'], temp['StateplaneN'])
    LLHC = (temp2['utmE'], temp2['utmN'])

    temp = gp.FRF2ncsp(x1, y1)
    temp2 = gp.ncsp2utm(temp['StateplaneE'], temp['StateplaneN'])
    URHC = (temp2['utmE'], temp2['utmN'])

    fig_name = 'newGrid_Corners_UTM.png'
    label_x = 'utmE'
    label_y = 'utmN'
    x_var = 'utmEasting'
    y_var = 'utmNorthing'

elif coord_system == 'latlon':
    temp = gp.FRF2ncsp(x0, y0)
    temp2 = gp.ncsp2LatLon(temp['StateplaneE'], temp['StateplaneN'])
    LLHC = (temp2['lat'], temp2['lon'])

    temp = gp.FRF2ncsp(x1, y1)
    temp2 = gp.ncsp2LatLon(temp['StateplaneE'], temp['StateplaneN'])
    URHC = (temp2['lat'], temp2['lon'])

    fig_name = 'newGrid_Corners_LL.png'
    label_x = 'longitude'
    label_y = 'latitude'
    x_var = 'longitude'
    y_var = 'latitude'


elif coord_system == 'stateplane':
    temp = gp.FRF2ncsp(x0, y0)
    LLHC = (temp['StateplaneE'], temp['StateplaneN'])

    temp = gp.FRF2ncsp(x1, y1)
    URHC = (temp['StateplaneE'], temp['StateplaneN'])

    fig_name = 'newGrid_Corners_SP.png'
    label_x = 'easting'
    label_y = 'northing'
    x_var = 'easting'
    y_var = 'northing'


elif coord_system == 'FRF':
    LLHC = (x0, y0)
    URHC = (x1, y1)
    fig_name = 'newGrid_Corners_FRF.png'
    label_x = 'xFRF'
    label_y = 'yFRF'
    x_var = 'xFRF'
    y_var = 'yFRF'


test = gT.makeBackgroundBathyCorners(LLHC, URHC, dx, dy, coord_system)

# check to see if this actually worked...
fig_name = fig_name
fig_loc = 'C:\Users\RDCHLDLY\Desktop\David Stuff\Projects\CSHORE\Bathy Interpolation\Test Figures'
plt.contourf(test[x_var], test[y_var], test['bottomElevation'])
plt.axis('equal')
plt.xlabel(label_x)
plt.ylabel(label_y)
plt.savefig(os.path.join(fig_loc, fig_name))
plt.close()
"""

