
import netCDF4 as nc
import numpy as np
import sys
sys.path.append('/home/spike/CMTB')
# sys.path.append('/home/spike/CMTB/gridsSTWAVE')
import makenc
from gridTools import gridTools

dataLocation = '/home/spike/repos/makeBathyInterp/Data'  # this is where data comes from and goes to, eventually thredds address
##################################################################
# MAKE FIRST BATHY FROM SIM FILE
##################################################################
yesterdaysSim = '/home/spike/CMTB/gridsSTWAVE/Minigrid_5m.sim'
yesterdaysDEP = '/home/spike/CMTB/gridsSTWAVE/Minigrid_5m.dep'
# get the data packet from DEP
yesterdaysBathyPacket = gridTools.GetOriginalGridFromSTWAVE(yesterdaysSim, yesterdaysDEP)
# make netCDF file for original bathy
yesterdaysListfname = ['%s/todaysBathyOriginal_STWAVE_5m.nc' %dataLocation]
globalYaml = '/home/spike/repos/makeBathyInterp/yamls/TodaysBathySTWAVEGlobal.yml'
varYaml = '/home/spike/repos/makeBathyInterp/yamls/TodaysBathy_var.yml'
makenc.makenc_todaysBathyCMTB(yesterdaysBathyPacket, yesterdaysListfname[0], globalYaml, varYaml)

##################################################################
# # # ## now loop through history on thredds #####################
##################################################################
methods = ['ngl', 'ngl'] # plant, metpy, and matplotlib also valid
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
        ofnameNC = dataLocation + '/' + interpType + '/todaysBathyNewFromGrids_%s.nc' % (date.strftime('%Y-%m-%d'))
        gridTools.MakeTodaysBathy(ofnameNC=ofnameNC, dataPacket=dataPacket, plotFlag=True, interpType= interpType)
        yesterdaysListfname.append(ofnameNC)