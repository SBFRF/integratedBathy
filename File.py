
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


"""

"""
# todaysNC = nc.Dataset('http://bones/thredds/dodsC/FRF/survey/gridded/FRF_20170127_1123_FRF_NAVD88_LARC_GPS_UTC_v20170203_grid_latlon.nc')
# rawDEM = '/home/spike/repos/makeBathyInterp/Data/FRF_NCDEM_Nad83_geographic_MSL_meters.xyz'
# gridNodesX = np.linspace(0, 10000, 155)
# gridNodesY = np.linspace(-100, 1000, 72)
version_prefix = 'HP'
# yesterdaysGrid = nc.Dataset('http://crunchy:8080/thredds/dodsC/CMTB/%s_STWAVE_data/Local_Field/Local_Field.ncml' %version_prefix)['bathymetry'][-1]


yesterdaysSim = 'C:/Users/spike/Documents/repos/CMTB/gridsSTWAVE/Minigrid_5m.sim'
yesterdaysDEP = 'C:/Users/spike/Documents/repos/CMTB/gridsSTWAVE/Minigrid_5m.dep'
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
outIfrf[:,1]
xFRF, yFRF = np.meshgrid(outIfrf[:, 0], outJfrf[:,1])

lat, lon = np.empty_like(xFRF), np.empty_like(xFRF)
easting, northing = np.empty_like(xFRF), np.empty_like(xFRF)
for yy in xFRF.shape[0]:
    for xx in xFRF.shape[1]:
        convert = sb.FRFcoord(xFRF[xx,yy], yFRF[xx, yy])
        lat[xx, yy] = convert['Lat']
        lon[xx, yy] = convert['Lon']
        easting[xx, yy] = convert['StateplaneE']
        northing[xx, yy] = convert['StateplaneN']
stio = stwaveIO(yesterdaysDEP)
DEPpacket = stio.DEPload(yesterdaysDEP)
todaysBathyPacket = {'latitude': lat,
                      'longitude': lon,
                      'easting': easting,
                      'northing': northing,
                      'xFRF': xFRF,
                      'yFRF': yFRF,
                      'elevation': 1}

makenc.makenc_todaysBathyCMTB(todaysBathyPacket, ofname, globalYaml, varYaml)
#
# todaysGrid = todaysNC['elevation'][0]
#
#
# # set up gridding procedure
# ngl.nnsetp
# # run natural neighbor algorithm
# ngl.natgrid