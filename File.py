
import PyNGL as ngl
import netCDF4 as nc
import numpy as np
sys.append(/home/spike/CMTB)
from STplotLib import DEPload
"""

"""
todaysNC = nc.Dataset('http://bones/thredds/dodsC/FRF/survey/gridded/FRF_20170127_1123_FRF_NAVD88_LARC_GPS_UTC_v20170203_grid_latlon.nc')
rawDEM = '/home/spike/repos/makeBathyInterp/Data/FRF_NCDEM_Nad83_geographic_MSL_meters.xyz'
gridNodesX = np.linspace(0, 10000, 155)
gridNodesY = np.linspace(-100, 1000, 72)
version_prefix = 'HP'
yesterdaysGrid = nc.Dataset('http://crunchy:8080/thredds/dodsC/CMTB/%s_STWAVE_data/Local_Field/Local_Field.ncml' %version_prefix)['bathymetry'][-1]
def openGridXYZ(demFile):
    """
    this function opens the DEM background file and parse's to return a grid
    :param demFile:
    :return: grid
    """
    bufsize = 65536
    # open file, read lines
    f = open(demFile)
    lines = f.readlines()
    f.close()
    x, y, z, = [], [], []
    # parse x y z of the lines
    for line in lines:
        split = line.split()
        x.append(split[0])
        y.append(split[1])
        z.append(split[2])
    # faster to convert after than during the read lines
    x = np.array(x)
    y = np.array(y)
    z = np.array(z)
    return x, y, z



x, y, z = openGridXYZ(rawDEM)
# grid nodes for Background
xNode = np.unique(x)
yNode = np.unique(y)

DEPload()



todaysGrid = todaysNC['elevation'][0]


# set up gridding procedure
ngl.nnsetp
# run natural neighbor algorithm
ngl.natgrid