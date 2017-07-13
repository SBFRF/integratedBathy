# ok, this is going to import the data from the geoTIFF file I got from Mike!!!
# it will include fill values for the missing data!!!

import gdal
import os, sys
import geoprocess as gp
import numpy as np
from makenc import makenc_intBATHY
sys.path.append('/home/number/repos')
import sblib as sb
import matplotlib.pyplot as plt


gdal.UseExceptions()
floc = '/home/number/BackgroundDEM/'
fname = 'NCDEM_UTM_update_10m_wt9_v20170703.tif'
ds = gdal.Open(os.path.join(floc, fname))
band = ds.GetRasterBand(1)
bottomElevation = band.ReadAsArray()



nrows, ncols = bottomElevation.shape

# I'm making the assumption that the image isn't rotated/skewed/etc.
# This is not the correct method in general, but let's ignore that for now
# If dxdy or dydx aren't 0, then this will be incorrect
x0, dx, dxdy, y0, dydx, dy = ds.GetGeoTransform()
x1 = x0 + dx * ncols
y1 = y0 + dy * nrows

"""
fig_name = 'test_geoTIFF.png'
plt.imshow(bottomElevation, cmap='gist_earth', extent=[x0, x1, y1, y0])
plt.savefig(os.path.join(floc, fig_name))
plt.close()
"""

# get vector of x's and y's
x_vec = np.linspace(x0, x1, ncols)
y_vec = np.linspace(y0, y1, nrows)

# create the grid that matches up with the elevation
utmE, utmN = np.meshgrid(x_vec, y_vec)

"""
fig_name2 = 'test_geoTIFF_mesh.png'
plt.contourf(utmE, utmN, bottomElevation)
plt.axis('equal')
plt.savefig(os.path.join(floc, fig_name2))
plt.close()
"""
# ok, looks like it actually worked....

#now lets trim them up...
utmE_sl = utmE[np.where(abs(utmN[:,0] - 3980000) == min(abs(utmN[:,0] - 3980000))), :]
utmE_sl = utmE_sl[0][0]
bElev_sl = bottomElevation[np.where(abs(utmN[:,0] - 3980000) == min(abs(utmN[:,0] - 3980000))), :]
bElev_sl = bElev_sl[0][0]
utmE_sl_sl = utmE_sl[np.where(bElev_sl != -9999)]
min_E = min(utmE_sl_sl)
max_E = max(utmE_sl_sl)

utmN_sl = utmN[:, int(np.where(abs(utmE[0,:] - 450000) == min(abs(utmE[0,:] - 450000)))[0][0])]
bElev_sl = bottomElevation[:, int(np.where(abs(utmE[0,:] - 450000) == min(abs(utmE[0,:] - 450000)))[0][0])]
utmN_sl_sl = utmN_sl[np.where(bElev_sl != -9999)]
min_N = min(utmN_sl_sl)
max_N = max(utmN_sl_sl)

ind_nmin = int(max(np.where(utmN[:,0] >= min_N)[0]))
ind_nmax = int(min(np.where(utmN[:,0] <= max_N)[0]))
ind_emin = int(min(np.where(utmE[0,:] >= min_E)[0]))
ind_emax = int(max(np.where(utmE[0,:] <= max_E)[0]))

utmE_N = utmE[ind_nmax:ind_nmin, ind_emin:ind_emax]
utmN_N = utmN[ind_nmax:ind_nmin, ind_emin:ind_emax]
bottomElevation_N = bottomElevation[ind_nmax:ind_nmin, ind_emin:ind_emax]

"""
fig_name3 = 'test_geoTIFF_mesh_trimmed.png'
plt.contourf(utmE_N, utmN_N, bottomElevation_N)
plt.axis('equal')
plt.savefig(os.path.join(floc, fig_name3))
plt.close()
"""
# ok, I think that worked also....
del utmE
del utmN
del bottomElevation

utmE = utmE_N
utmN = utmN_N
bottomElevation = bottomElevation_N

del utmE_N
del utmN_N
del bottomElevation_N

# next step is to convert them to 1d arrays, pass them to the lat lon converter and state plane converter,
# then convert that stuff back to the 2d array like a boss

"""
# just a bit of test data to be sure I do it properly
test = np.array([[0, 1, 2], [3, 4, 5]])
test1 = test.reshape((1, test.shape[0]*test.shape[1]))
test2 = test1.reshape((test.shape[0], test.shape[1]))
"""

utmE_vec = utmE.reshape((1, utmE.shape[0]*utmE.shape[1]))[0]
utmN_vec = utmN.reshape((1, utmN.shape[0]*utmN.shape[1]))[0]

ll_dict = gp.utm2LatLon(utmE_vec, utmN_vec, 18, 'S')

lat = ll_dict['lat'].reshape((utmE.shape[0], utmE.shape[1]))
lon = ll_dict['lon'].reshape((utmE.shape[0], utmE.shape[1]))

# make the dictionary I'm going to pass to makenc

# show time?
dir = os.getcwd()
nc_name = 'backgroundDEM'
ofname = os.path.join(dir, nc_name + '.nc')
globalYaml =  os.path.join(dir, 'FRFRegional_global.yml')
varYaml =  os.path.join(dir, 'FRFRegional_var.yml')

dataDict = {}
dataDict['bottomElevation'] = bottomElevation
dataDict['utmEasting'] = utmE
dataDict['utmNorthing'] = utmN
dataDict['latitude'] = lat
dataDict['longitude'] = lon

makenc_intBATHY(ofname, dataDict, globalYaml, varYaml)

