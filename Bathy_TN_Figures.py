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
from gridTools import gridTools as gT
import pandas as pd
from bsplineFunctions import bspline_pertgrid
import matplotlib as m
import pyproj
import datetime as DT
from scaleCinterp_python.DEM_generator import DEM_generator
import MakeUpdatedBathyDEM as mBATHY


# all we are going to do with this is make sexy Figures of these bathymetries for the TN

# Figure 3 - regional DEM vs. time averaged DEM

# load the nc files....
# original background
nc_b_loc = 'C:\Users\dyoung8\Desktop\David Stuff\Projects\CSHORE\Bathy Interpolation\TestNCfiles_WA'
nc_b_name = 'backgroundDEMt0.nc'

nc_ta_loc = 'C:\Users\dyoung8\Desktop\David Stuff\Projects\CSHORE\Bathy Interpolation\TestNCfiles_WA'
nc_ta_name = 'backgroundDEMt0_TimeMean.nc'

# CS-array url - I just use this to get the position, not for any data
cs_array_url = 'http://134.164.129.55/thredds/dodsC/FRF/oceanography/waves/8m-array/2017/FRF-ocean_waves_8m-array_201707.nc'

# location of these figures
fig_loc = 'C:\Users\dyoung8\Desktop\David Stuff\Projects\CSHORE\Bathy Interpolation\TechNote\Figures'


# where is the cross shore array?
test = nc.Dataset(cs_array_url)
Lat = test['latitude'][:]
Lon = test['longitude'][:]
# convert to FRF
temp = gp.FRFcoord(Lon, Lat)
CSarray_X = temp['xFRF']
CSarray_Y = temp['yFRF']


# get sum data!!!!!
old_bathy = nc.Dataset(os.path.join(nc_b_loc, nc_b_name))
Zi = old_bathy.variables['elevation'][:]
xFRFi_vec = old_bathy.variables['xFRF'][:]
yFRFi_vec = old_bathy.variables['yFRF'][:]
xFRFi, yFRFi = np.meshgrid(xFRFi_vec, yFRFi_vec)

new_bathy = nc.Dataset(os.path.join(nc_ta_loc, nc_ta_name))
Zn = new_bathy.variables['elevation'][:]
xFRFn_vec = new_bathy.variables['xFRF'][:]
yFRFn_vec = new_bathy.variables['yFRF'][:]
xFRFn, yFRFn = np.meshgrid(xFRFn_vec, yFRFn_vec)


# make secksy plots
fig_name = 'Figure3_RegionalVsTimeMean_V2' + '.png'

fig = plt.figure(figsize=(10, 6))

ax1 = plt.subplot2grid((1, 2), (0, 0), colspan=1)
pc1 = ax1.pcolor(xFRFi, yFRFi, Zi[:, :], cmap=plt.cm.jet, vmin=-10, vmax=3)
ax1.plot(CSarray_X, CSarray_Y, 'rX', label='8m-array')
ax1.set_xlabel('Cross-shore - $x$ ($m$)', fontsize=16)
ax1.set_ylabel('Alongshore - $y$ ($m$)', fontsize=16)
for tick in ax1.xaxis.get_major_ticks():
    tick.label.set_fontsize(14)
for tick in ax1.yaxis.get_major_ticks():
    tick.label.set_fontsize(14)
ax1.tick_params(labelsize=14)
ax1.legend()
ax1.text(-0.15, 1.05, '(a)', horizontalalignment='left', verticalalignment='top', transform=ax1.transAxes, fontsize=16)


ax2 = plt.subplot2grid((1, 2), (0, 1), colspan=1)
pc2 = ax2.pcolor(xFRFn, yFRFn, Zn[:, :], cmap=plt.cm.jet, vmin=-10, vmax=3)
ax2.plot(CSarray_X, CSarray_Y, 'rX', label='8m-array')
ax2.set_xlabel('Cross-shore - $x$ ($m$)', fontsize=16)
ax2.set_ylabel('Alongshore - $y$ ($m$)', fontsize=16)
for tick in ax2.xaxis.get_major_ticks():
    tick.label.set_fontsize(14)
for tick in ax2.yaxis.get_major_ticks():
    tick.label.set_fontsize(14)
ax2.tick_params(labelsize=14)
ax2.legend()
ax2.text(-0.15, 1.05, '(b)', horizontalalignment='left', verticalalignment='top', transform=ax2.transAxes, fontsize=16)

cbar1 = fig.colorbar(pc1, ax=ax1)
cbar2 = fig.colorbar(pc2, ax=ax2)
cbar1.set_label('Elevation ($m$)')
cbar2.set_label('Elevation ($m$)')

fig.subplots_adjust(wspace=0.4, hspace=0.1)
fig.tight_layout(pad=1, h_pad=2.5, w_pad=1, rect=[0.0, 0.0, 1.0, 0.925])

fig.savefig(os.path.join(fig_loc, fig_name), dpi=300)
plt.close()



