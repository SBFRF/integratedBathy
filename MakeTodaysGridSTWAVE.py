
import netCDF4 as nc
import numpy as np
import sys
import makenc
from gridTools import gridTools


# Select which resolution for STWAVE
resolution = '5'
dataLocation = '/home/number/thredds_data/grids/STWAVE_%sm' % resolution  # this is where data comes from and goes to, eventually thredds address
##################################################################
# MAKE FIRST BATHY FROM SIM FILE
##################################################################
yesterdaysSim = '/home/number/CMTB/gridsSTWAVE/Minigrid_%sm.sim' % resolution
yesterdaysDEP = '/home/number/CMTB/gridsSTWAVE/Minigrid_%sm.dep' % resolution
# get the data packet from DEP
yesterdaysBathyPacket = gridTools.GetOriginalGridFromSTWAVE(yesterdaysSim, yesterdaysDEP)
# make netCDF file for original bathy
yesterdaysListfname = ['%s/todaysBathyOriginal_STWAVE_%sm.nc' %(dataLocation, resolution)]
globalYaml = 'yamls/TodaysBathySTWAVEGlobal.yml'
varYaml = 'yamls/TodaysBathy_var.yml'
makenc.makenc_todaysBathyCMTB(yesterdaysBathyPacket, yesterdaysListfname[0], globalYaml, varYaml)

##################################################################
# # # ## now loop through history on thredds #####################
##################################################################
methods = ['matplotlib'] # plant, metpy, and matplotlib also valid
for interpType in methods:
    newncfile = nc.Dataset('http://bones/thredds/dodsC/FRF/survey/gridded/gridded.ncml')
    for idxNew, time in enumerate(newncfile['time'][:]):
        date = nc.num2date(time, newncfile['time'].units)  # date for the new survey
        # remove fill values from newNCfile
        newXfrf = newncfile['xFRF'][:]
        newYfrf = newncfile['yFRF'][:]
        newZfrf = newncfile['elevation'][idxNew]
        if type(newZfrf) == np.ma.MaskedArray:
            # now remove masked data based on Z from the x and y coords
            newXfrf = newXfrf[~np.all(newZfrf.mask, axis=0)]
            newYfrf = newYfrf[~np.all(newZfrf.mask, axis=1)]
            newZfrf = newZfrf[~np.all(newZfrf.mask, axis=1), :]  #
            newZfrf = newZfrf[:, ~np.all(newZfrf.mask, axis=0)]

        backgroundGridnc = nc.Dataset(yesterdaysListfname[-1])  # updating to the background grid
        idxOld = 0   #len(yesterdaysListfname)-1  # counts upwards with each one made

        # make data packet
        dataPacket = {'time': date,
                      'newZfrf': newZfrf,
                      'newYfrf': newYfrf,
                      'newXfrf': newXfrf,
                      'oldBathy': backgroundGridnc['elevation'][idxOld],
                      'modelGridX': backgroundGridnc['xFRF'][:],
                      'modelGridY': backgroundGridnc['yFRF'][:],
                      'latitude': backgroundGridnc['latitude'][:],
                      'longitude': backgroundGridnc['longitude'][:],
                      'easting': backgroundGridnc['easting'][:],
                      'northing': backgroundGridnc['northing'][:],
                      'azimuth': backgroundGridnc['azimuth']}

        print ' Working on %s interpolation now for %s' % (interpType, date)
        ofnameNC = dataLocation + '/todaysBathyNewFromGrids_%s_STWAVE_%sm.nc' % (date.strftime('%Y-%m-%d'), resolution)
        gridTools.MakeTodaysBathy(ofnameNC=ofnameNC, dataPacket=dataPacket, plotFlag=True, interpType= interpType)
        yesterdaysListfname.append(ofnameNC)