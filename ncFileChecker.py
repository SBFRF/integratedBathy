import netCDF4 as nc
import numpy as np
from matplotlib import pyplot as plt
import os


nc_loc = 'http://134.164.129.55/thredds/dodsC/cmtb/projects/DavidTest/2017/CMTB-integratedBathyProduct_survey_201709.nc'
bathy = nc.Dataset(nc_loc)

tempDict = {}
tempDict['time'] = nc.num2date(bathy.variables['time'][:], bathy.variables['time'].units, bathy.variables['time'].calendar)
tempDict['elevation'] = bathy.variables['elevation'][:]
tempDict['xFRF'] = bathy.variables['xFRF'][:]
tempDict['yFRF'] = bathy.variables['yFRF'][:]

# zoomed in pcolor plot on AOI
fig_loc = 'C:/Users/dyoung8/Desktop/David Stuff/Projects/CSHORE/Bathy Interpolation/Test Figures/'
fig_name = 'ncFileCheckFig' + '.png'
plt.figure()
plt.pcolor(tempDict['xFRF'], tempDict['yFRF'], tempDict['elevation'][-1, :, :], cmap=plt.cm.jet, vmin=-13, vmax=5)
cbar = plt.colorbar()
cbar.set_label('(m)')
axes = plt.gca()
plt.xlabel('xFRF (m)')
plt.ylabel('yFRF (m)')
plt.savefig(os.path.join(fig_loc, fig_name))
plt.close()

t = 1