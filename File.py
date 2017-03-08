
# import PyNGL as ngl
import netCDF4 as nc
import numpy as np
import sys
sys.path.append('/home/spike/CMTB/gridsSTWAVE')
sys.path.append('C:/Users/spike/Documents/repos/CMTB/gridsSTWAVE')
from gridTools import gridTools
import sblib as sb
import makenc
from inputOutput import stwaveIO
import datetime as DT

# todaysNC = nc.Dataset('http://bones/thredds/dodsC/FRF/survey/gridded/FRF_20170127_1123_FRF_NAVD88_LARC_GPS_UTC_v20170203_grid_latlon.nc')
# rawDEM = '/home/spike/repos/makeBathyInterp/Data/FRF_NCDEM_Nad83_geographic_MSL_meters.xyz'
# gridNodesX = np.linspace(0, 10000, 155)
# gridNodesY = np.linspace(-100, 1000, 72)
version_prefix = 'HP'
# yesterdaysGrid = nc.Dataset('http://crunchy:8080/thredds/dodsC/CMTB/%s_STWAVE_data/Local_Field/Local_Field.ncml' %version_prefix)['bathymetry'][-1]


yesterdaysSim = 'C:/Users/spike/Documents/repos/CMTB/gridsSTWAVE/Minigrid_5m.sim'
yesterdaysDEP = 'C:/Users/spike/Documents/repos/CMTB/gridsSTWAVE/Minigrid_5m.dep'
# yesterdaysDEP = 'C:\Users\spike\Documents\repos\CMTB\gridsSTWAVE\Minigrid_5m.dep'

# x, y, z = gridTools.openGridXYZ(rawDEM)
# # grid nodes for Background
# xNode = np.unique(x)
# yNode = np.unique(y)

iStatePlane, jStatePlane, spatial = gridTools.CreateGridNodesInStatePlane(yesterdaysSim)
# icoords = sb.FRFcoord(iStatePlane[0], iStatePlane[1]) # converting i coordinates to Lat and Lon
# jcoords = sb.FRFcoord(jStatePlane[0], jStatePlane[1]) # convertin J coordinates to Lat and Lon
outIfrf, outJfrf = gridTools.convertGridNodesFromStatePlane(iStatePlane, jStatePlane)
# checking the looping method vs the array processing in FRF coord
# assert (outIgeographic[:,0] == icoords['Lon']).all()
# assert (outIgeographic[:,1] == icoords['Lat']).all()
# assert (outJgeographic[:,0] == jcoords['Lon']).all()
# assert (outJgeographic[:,1] == jcoords['Lat']).all()
# # now checking in FRF
# assert (outIfrf[:,0] == icoords['FRF_X']).all()
# assert (outIfrf[:,1] == icoords['FRF_Y']).all()
# assert (outJfrf[:,0] == jcoords['FRF_X']).all()
# assert (outJfrf[:,1] == jcoords['FRF_Y']).all()
#outIfrf[:,1]
xFRF, yFRF = np.meshgrid(outIfrf[:, 0], outJfrf[:,1])

lat, lon = np.empty_like(xFRF), np.empty_like(xFRF)
easting, northing = np.empty_like(xFRF), np.empty_like(xFRF)
for yy in range(0, xFRF.shape[1]):
    for xx in range(0, xFRF.shape[0]):
        convert = sb.FRFcoord(xFRF[xx,yy], yFRF[xx, yy])
        lat[xx, yy] = convert['Lat']
        lon[xx, yy] = convert['Lon']
        easting[xx, yy] = convert['StateplaneE']
        northing[xx, yy] = convert['StateplaneN']
STIO = stwaveIO('')
STIO.depfname_nest = [yesterdaysDEP]
DEPpacket = STIO.DEPload(nested=True)
elevation = - DEPpacket['bathy'] # flip depths to elevations (sign convention)
yesterdaysBathyPacket = {'latitude': lat,
                      'longitude': lon,
                      'easting': easting,
                      'northing': northing,
                      'xFRF': xFRF[0,:],
                      'yFRF': yFRF[:,0],
                      'elevation': elevation,
                      'azimuth': spatial['azi'],
                      'time': nc.date2num(DT.datetime(2007,01,01), 'seconds since 1970-01-01')}
ofname = 'todaysBathyOriginal.nc'
globalYaml = 'C:/Users/spike/Documents/repos/makebathyinterp/yamls/TodaysBathySTWAVEGlobal.yml'
varYaml = 'C:/Users/spike/Documents/repos/makebathyinterp/yamls/TodaysBathy_var.yml'
makenc.makenc_todaysBathyCMTB(yesterdaysBathyPacket, ofname, globalYaml, varYaml)

##
## Now incorporate new data into bathy gridding process
# function inputs
yesterdaysBathyNC = nc.Dataset('todaysBathyOriginal.nc')
oldBathy = yesterdaysBathyNC['elevation'][0]
xFRF = yesterdaysBathyNC['xFRF']
yFRF = yesterdaysBathyNC['yFRF']

todaysNC = nc.Dataset('http://bones/thredds/dodsC/FRF/survey/gridded/FRF_20170127_1123_FRF_NAVD88_LARC_GPS_UTC_v20170203_grid_latlon.nc')
todaysBathy = todaysNC['bathymetry'][0]

# 1. eliminiate grid points in the 'old' bathy based on location
#

# 2. put 'old' data into xraw, yraw, zraw 1D arrays

# 3. put New data into xnew, ynew, znew

# 4. append new to raw values

# 5. create grid nodes of output

# 6. run natural neighbor interpolation scheme
# # set up gridding procedure
# ngl.nnsetp
# # run natural neighbor algorithm
# ngl.natgrid

# 7. create new dictionary for New grid Creation
yesterdaysBathyPacket = {'latitude': lat,
                      'longitude': lon,
                      'easting': easting,
                      'northing': northing,
                      'xFRF': xFRF[0,:],
                      'yFRF': yFRF[:,0],
                      'elevation': elevation,
                      'azimuth': spatial['azi'],
                      'time': time}
# make NetCDF file using Dictonary.
ofname = 'todaysBathyNew.nc'
globalYaml = 'C:/Users/spike/Documents/repos/makebathyinterp/yamls/TodaysBathySTWAVEGlobal.yml'
varYaml = 'C:/Users/spike/Documents/repos/makebathyinterp/yamls/TodaysBathy_var.yml'
makenc.makenc_todaysBathyCMTB(yesterdaysBathyPacket, ofname, globalYaml, varYaml)

