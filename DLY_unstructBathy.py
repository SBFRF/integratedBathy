from prepdata import inputOutput
from sblib import gridTools as gT
from sblib import geoprocess as gp
from sblib import sblib
import makenc
import os
import pickle
import netCDF4 as nc
import numpy as np
from matplotlib import pyplot as plt
import matplotlib.tri as tri
import datetime as DT
from plotting import nonoperationalPlots as noP

# ok, for right now this is just David's test script to create a new "base grid" from unstructured grid data
# we will be moving the functionality into seperate locations as appropriate later

# step 1 - we need to make a new regional background DEM that covers the ENTIRE domain of the simulation!!!!
inputDict = {}
inputDict['gridTEL'] = '/home/david/PycharmProjects/cmtb/grids/CMS/CMS-Flow-FRF.tel'
# so, load the .tel file - NOTE!!! as of 4/25/2017 this couldnt be done in makebathy!
# PrepData is not in the makebathy repo.  Is it even a submoduel?

# now lets doctor up the .tel file
cmsfio = inputOutput.cmsfIO()
cmsfio.read_CMSF_telDict(inputDict['gridTEL'])
ugridDict = {}
ugridDict['coord_system'] = 'ncsp'
ugridDict['x'], ugridDict['y'] = gT.convertGridNodes2ncsp(cmsfio.tel_dict['origin_easting'], cmsfio.tel_dict['origin_northing'], cmsfio.tel_dict['azimuth'], cmsfio.tel_dict['xPos'], cmsfio.tel_dict['yPos'])
ugridDict['units'] = 'm'

# convert all of these to FRF to be handed to makeBackgroundBathy?
temp1 = gp.ncsp2FRF(ugridDict['x'], ugridDict['y'])
ugridDict['xFRF'] = temp1['xFRF']
ugridDict['yFRF'] = temp1['yFRF']

"""
# this actually takes almost no time at all.  its all the regional DEM that takes forever...
# ok, we need to break this up into smaller pieces because it is incredibly slow.
# so, save ugridDict as a pickle
ploc = '/home/david/BathyTroubleshooting/BackgroundFiles'
pname = 'ugridDict' + '.p'
psname = os.path.join(ploc, pname)
pickle.dump(ugridDict, open(psname, 'wb'))

ploc = '/home/david/BathyTroubleshooting/BackgroundFiles'
pname = 'ugridDict' + '.p'
psname = os.path.join(ploc, pname)
# now lets load this pickle file back in!
ugridDict = pickle.load(open(psname, 'rb'))
"""

"""
# ok, ok, now I have a nice point cloud.  determine full extents?
dx = 5
dy = 5
buffer = 200
minX = sblib.baseRound(min(ugridDict['xFRF']) - buffer, dx)
minY = sblib.baseRound(min(ugridDict['yFRF']) - buffer, dy)
maxX = sblib.baseRound(max(ugridDict['xFRF']) + buffer, dx)
maxY = sblib.baseRound(max(ugridDict['yFRF']) + buffer, dy)
LLHC = (minX, minY)
URHC = (maxX, maxY)
rDEM = gT.makeBackgroundBathyCorners(LLHC, URHC, dx, dy, coord_system='FRF')

# you probably want to go ahead and save this, then reload it, because it takes forever!!
Zi = rDEM['bottomElevation']
xFRFi = rDEM['xFRF']
yFRFi = rDEM['yFRF']

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

nc_loc = '/home/david/BathyTroubleshooting/BackgroundFiles'
nc_name = 'backgroundDEMt0_tel.nc'

global_yaml = '/home/david/PycharmProjects/makebathyinterp/yamls/BATHY/FRFt0_global.yml'
var_yaml = '/home/david/PycharmProjects/makebathyinterp/yamls/BATHY/FRFt0_var.yml'

makenc.makenc_t0BATHY(os.path.join(nc_loc, nc_name), nc_dict, globalYaml=global_yaml, varYaml=var_yaml)
t = 1

"""
# now reload it!
nc_loc = '/home/david/BathyTroubleshooting/BackgroundFiles'
nc_name = 'backgroundDEMt0tel_TimeMean.nc'
rDEM = nc.Dataset(os.path.join(nc_loc, nc_name))
zDEM = rDEM['elevation'][:]
xFRFdem = rDEM['xFRF'][:]
yFRFdem = rDEM['yFRF'][:]

# okay this has nans in it somehow?  we need those to be gone?
xFRFdem2 = np.array(xFRFdem)
yFRFdem2 = np.array(yFRFdem)
t = 1

# make a new "time-mean" bathy out of this one?


"""
# to make sure we are covering everything
# plot the rDEM with the triangulated grid from the .tel file overlaid on top?
ploc = '/home/david/BathyTroubleshooting/BackgroundFiles/TestFigs'
pname = 'rDEMcheck.png'

# perform triangulation
triang = tri.Triangulation(ugridDict['xFRF'], ugridDict['yFRF'])

# generate the plot.
plt.figure()
plt.gca().set_aspect('equal')
plt.contourf(xFRFdem, yFRFdem, zDEM, cmap='coolwarm')
plt.triplot(triang, 'b-', lw=0.1)

# set some other labels
plt.ylabel('yFRF', fontsize=12)
plt.xlabel('xFRF', fontsize=12)

# save time
plt.savefig(os.path.join(ploc, pname), dpi=300)
# load it up in SMS like Spicer did to make sure I have't screwed up the conversion?
# it looks, offset just a bit?  I need to think about this a little more.
t = 1
"""

# now we need to make a dictionary that has the same keys as this "bathy" dictionary to hand to interpIntegrated...
# so we can reinterpolate the whole .tel file grid
bathy = {}
bathy['xFRF'] = xFRFdem
bathy['yFRF'] = yFRFdem
bathy['elevation'] = zDEM
bathy['time'] = DT.datetime.strptime('1980-01-01T00:00:00Z', '%Y-%m-%dT%H:%M:%SZ')   # just make something up for this, since it is the regional DEM
newzDict = gT.interpIntegratedBathy4UnstructGrid(ugridDict=ugridDict, bathy=bathy)

# now i need to replace my old depth values with the new ones
depth = cmsfio.tel_dict['depth'].copy()
newDepth = -1*newzDict['z']
# if node is inactive, we do not need to replace!
newDepth[depth < -900] = np.nan  # so set those to nan!

# what does this look like?  did I break it?
xT = ugridDict['xFRF'][~np.isnan(newDepth)]
yT = ugridDict['yFRF'][~np.isnan(newDepth)]
newDepthT = newDepth[~np.isnan(newDepth)]

"""
# okay, what we are looking for here is any discontinuous stuff in the region where the model is actually going
# to be run.  don't really care if the portion of the grid on the sound is messed up
figloc = '/home/david/BathyTroubleshooting/NewTel'
figname = 'newTelGrid'
ofname = os.path.join(figloc, figname + '.png')
pDict = {}
pDict['x'] = xT
pDict['y'] = yT
pDict['xLabel'] = 'xFRF ($m$)'
pDict['yLabel'] = 'yFRF ($m$)'
pDict['cbarLabel'] = 'Depth ($m$)'
pDict['z'] = newDepthT
pDict['cbarColor'] = 'RdYlBu'
noP.plotUnstructBathy(ofname=ofname, pDict=pDict)
"""

# integrate this back into the .tel file?
depthN = depth.copy()
depthN[~np.isnan(newDepth)] = newDepth[~np.isnan(newDepth)]
ncDict = cmsfio.tel_dict.copy()
del ncDict['depth']
# copy over the new depth
ncDict['depth'] = depthN
# include the xFRF, yFRF positions of each cell
ncDict['xFRF'] = ugridDict['xFRF']
ncDict['yFRF'] = ugridDict['yFRF']

ncLoc = '/home/david/BathyTroubleshooting/BackgroundFiles'
ncName = 'CMSFtel0.nc'
ncgYaml = '/home/david/PycharmProjects/makebathyinterp/yamls/BATHY/CMSFtel0_global.yml'
ncvYaml = '/home/david/PycharmProjects/makebathyinterp/yamls/BATHY/CMSFtel0_var.yml'

makenc.makenc_CMSFtel(ofname=os.path.join(ncLoc, ncName), dataDict=ncDict, globalYaml=ncgYaml, varYaml=ncvYaml)

t = 1
