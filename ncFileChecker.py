import netCDF4 as nc
import numpy as np


nc_loc = 'http://134.164.129.55/thredds/dodsC/FRF/geomorphology/elevationTransects/survey/FRF_20090804_1040_FRF_NAVD88_CRAB_GPS_UTC_v20160513.nc'
bathy = nc.Dataset(nc_loc)

times = nc.num2date(bathy.variables['time'][:], bathy.variables['time'].units, bathy.variables['time'].calendar)
all_surveys = bathy.variables['surveyNumber'][:]
all_profNum = bathy.variables['profileNumber'][:]

t = 1
np.unique(all_profNum)