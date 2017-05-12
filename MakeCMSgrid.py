from inputOutput import cmsIO
from gridTools import gridTools
import makenc
import netCDF4 as nc
import numpy as np
# base dep file
cio = cmsIO()
depFname = 'exampleCMSfiles/CMS-Wave-FRF.dep'
z, dx, dy = cio.ReadCMS_dep(depFname)
simFname = 'exampleCMSfiles/CMS-Wave-FRF.sim'
x0, y0, azi = cio.ReadCMS_sim(simFname)
bathyPacket = gridTools.makeCMSgridNodes(x0, y0, azi, dx, dy, z, plot=False)
# make bathy packet into netCDF file
dataLocation = 'Data'  # this is where data comes from and goes to, eventually thredds address
yesterdaysListfname = ['%s/todaysBathyOriginal_CMSwave_v1.nc' %dataLocation]
globalYaml = 'yamls/TodaysBathyCMSGlobal.yml'  # yaml for Todaysbathy files
varYaml = 'yamls/TodaysBathy_var.yml'
makenc.makenc_todaysBathyCMTB(bathyPacket, yesterdaysListfname[0], globalYaml, varYaml)


methods = ['ngl'] # plant, metpy, and matplotlib also valid
for interpType in methods:
    newncfile = nc.Dataset('http://bones/thredds/dodsC/FRF/survey/gridded/gridded.ncml')
    for idxNew, time in enumerate(newncfile['time'][:]):
        date = nc.num2date(time, newncfile['time'].units)  # date for the new survey
        # remove fill values from newNCfile
        newXfrf = newncfile['xFRF'][:]  # coordinates of FRF grid product
        newYfrf = newncfile['yFRF'][:]  # y coordinate of FRF grid product
        newZfrf = newncfile['elevation'][idxNew] # elevations of FRF grid product
        if type(newZfrf) == np.ma.MaskedArray:
            # now remove masked data based on Z from the x and y coords
            newXfrf = newXfrf[~np.all(newZfrf.mask, axis=0)]
            newYfrf = newYfrf[~np.all(newZfrf.mask, axis=1)]
            newZfrf = newZfrf[~np.all(newZfrf.mask, axis=1), :]  #
            newZfrf = newZfrf[:, ~np.all(newZfrf.mask, axis=0)]
        # update new bathy from last measured bathy to get better evolution of 'updated' bathymetry
        backgroundGridnc = nc.Dataset(yesterdaysListfname[-1])  # updating to the background grid
        idxOld = 0   #len(yesterdaysListfname)-1  # counts upwards with each one made

        # make data packet
        dataPacket = {'time': date,
                      'newZfrf': newZfrf,
                      'newYfrf': newYfrf,
                      'newXfrf': newXfrf,
                      'oldBathy': backgroundGridnc['elevation'][idxOld],  # send only 2D array to function
                      'modelGridX': backgroundGridnc['xFRF'][:],
                      'modelGridY': backgroundGridnc['yFRF'][:],
                      'latitude': backgroundGridnc['latitude'][:],
                      'longitude': backgroundGridnc['longitude'][:],
                      'easting': backgroundGridnc['easting'][:],
                      'northing': backgroundGridnc['northing'][:],
                      'azimuth': backgroundGridnc['azimuth']}

        print ' Working on %s interpolation now for %s' % (interpType, date)
        ofnameNC = dataLocation + '/' + interpType + '/todaysBathyNewFromGrids_%s_CMS.nc' % (date.strftime('%Y-%m-%d'))
        gridTools.MakeTodaysBathy(ofnameNC=ofnameNC, dataPacket=dataPacket, plotFlag=True, interpType= interpType)
        yesterdaysListfname.append(ofnameNC)