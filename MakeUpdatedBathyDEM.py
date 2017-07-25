
import netCDF4 as nc
import numpy as np
import sys, os
import makenc
from gridTools import gridTools as gT
from getdatatestbed.getDataFRF import getObs, getDataTestBed
import datetime as DT
from scalecInterp_python.DEM_generator import DEM_generator
from bsplineFunctions import bspline_pertgrid
import matplotlib.pyplot as plt

# Hard Code Resolution to 5 m (dx & dy) - want whole number FRF coords
resolution = 5

# start with background DEM
# **NOTE: will have to edit this to look for more recent bathy first!!!!!**
# hard code my corners....
LLHC = (-50, -200)
URHC = (1305, 2000)  # 1305 is the location of the 11-m awac rounded UP to the nearest 5 m
dx = resolution
dy = resolution
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



# build my new bathymetry from the FRF gridded product
filelist = ['http://134.164.129.55/thredds/dodsC/FRF/survey/gridded/FRF_20160726_1121_FRF_NAVD88_LARC_GPS_UTC_v20170320_grid_latlon.nc']


# check bounds of this nc file first
dataX, dataY, dataZ = [], [], []
ncfile = nc.Dataset(filelist[0])
dataX = ncfile['xFRF'][:]
dataY = ncfile['yFRF'][:]
dataZ = ncfile['elevation'][:]
if isinstance(dataZ, np.ma.MaskedArray):
    maskZ = np.ma.getmask(dataZ)
    mask_ind = np.where(maskZ == False)
    dataX = dataX[mask_ind]
    dataY = dataY[mask_ind]
else:
    pass


x0, y0 = max(dataX), max(dataY)  # north east corner of grid
x1, y1 = min(dataX), min(dataY)     # south west corner of grid
dict = {'x0': x0,    #gp.FRFcoord(x0, y0)['Lon'],  # -75.47218285,
        'y0': y0,    #gp.FRFcoord(x0, y0)['Lat'],  #  36.17560399,
        'x1': x1,    #gp.FRFcoord(x1, y1)['Lon'],  # -75.75004989,
        'y1': y1,    #gp.FRFcoord(x1, y1)['Lat'],  #  36.19666112,
        'lambdaX': 5,  # grid spacing in x  -  Here is where CMS would hand array of variable grid spacing
        'lambdaY': 5,  # grid spacing in y
        'msmoothx': 100,  # smoothing length scale in x
        'msmoothy': 200,  # smoothing length scale in y
        'msmootht': 1,   # smoothing length scale in Time
        'filterName': 'hanning',
        'nmseitol': 0.75,
        'grid_coord_check': 'FRF',
        'grid_filename': '',  # should be none if creating background Grid!  becomes best guess grid
        'data_coord_check': 'FRF',
        'filelist': filelist
        }

out = DEM_generator(dict)

Zn = out['Zi']
Zn = Zn
MSEn = out['MSEi']
MSRn = out['MSRi']
NMSEn = out['NMSEi']
xFRFn_vec = out['x_out']
yFRFn_vec = out['y_out']

xFRFn, yFRFn = np.meshgrid(xFRFn_vec, yFRFn_vec)



# find out where the overlap is
xFRFi_vec = xFRFi[0,:]
yFRFi_vec = yFRFi[:,0]

x1 = np.where(xFRFi_vec == min(xFRFn_vec))[0][0]
x2 = np.where(xFRFi_vec == max(xFRFn_vec))[0][0]
y1 = np.where(yFRFi_vec == min(yFRFn_vec))[0][0]
y2 = np.where(yFRFi_vec == max(yFRFn_vec))[0][0]

Zi_s = Zi[y1:y2+1, x1:x2+1]
"""
#check to see if this is correct?
xFRFi_s = xFRFi[y1:y2+1, x1:x2+1]
yFRFi_s = yFRFi[y1:y2+1, x1:x2+1]
"""

# get the difference!!!!
Zdiff = Zn - Zi_s

# spline time?
splinebctype = 10
lc = 2
dxm = 2
dxi = 1
targetvar = 0.45
wb = 1 - np.divide(MSEn, targetvar + MSEn)
newZdiff = bspline_pertgrid(Zdiff, wb, splinebctype=splinebctype, lc=lc, dxm=dxm, dxi=dxi)
newZn = Zi_s + newZdiff

# get my new pretty splined grid and see what happens if I dont spline
newZi = Zi.copy()
Zi_ns = Zi.copy()
newZi[y1:y2+1, x1:x2+1] = newZn
Zi_ns[y1:y2+1, x1:x2+1] = Zn


# with splining - the winner?
fig_name = 'UpdatedBathy.png'
plt.contourf(xFRFi, yFRFi, newZi)
plt.axis('equal')
plt.xlabel('xFRF')
plt.ylabel('yFRF')
plt.colorbar()
plt.savefig(os.path.join(fig_loc, fig_name))
plt.close()

# ok, lets plot these three up to see how they look.

# second plot - if I didnt spline
fig_name = 'UpdatedBathy_ns.png'
plt.contourf(xFRFi, yFRFi, Zi_ns)
plt.axis('equal')
plt.xlabel('xFRF')
plt.ylabel('yFRF')
plt.colorbar()
plt.savefig(os.path.join(fig_loc, fig_name))
plt.close()







