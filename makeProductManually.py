import wrappers
import os, glob
import matplotlib
matplotlib.use('Agg')
""" This came from DLY_test, separated now to make cleaner will process/re process gridded data of interest
 will use the last data on thredds previous to start date as background."""
def runBathyProductManually(version, *args):
    """This will run the bathy product manually from the command line (for speed) with a version

    :param version: acceptable versoins are cBKF, cBKF-T, and survey
    :arg 0 this is start date
    :return: a bunch of gridded products
    """
    # list of inputs!!!!!
    x_smooth = 20   # scale c interp x-direction smoothing
    y_smooth = 100  # scale c interp y-direction smoothing
    # splinebctype - this is the type of spline you want to force
    # 2 - second derivative goes to zero at boundary
    # 1 - first derivative goes to zero at boundary
    # 0 - value is zero at boundary
    # 10 - force value and derivative(first?!?) to zero at boundary
    splinebctype = 10
    lc = [3, 8]  # spline smoothing constraint value [x, y] directions
    dxm = [1, 3]  # coarsening of the grid for spline (e.g., 2 means calculate with a dx that is 2x input dx)
    # can be tuple if you want to do dx and dy seperately (dxm, dym), otherwise dxm is used for both
    dxi = 1  # fining of the grid for spline (e.g., 0.1 means return spline on a grid that is 10x input dx)
    # as with dxm, can be a tuple if you want separate values for dxi and dyi
    targetvar = 0.25 # this is the target variance used in the spline function.
    wbysmooth = 325  # y-edge smoothing scale
    wbxsmooth = 100  # x-edge smoothing scale
    # It is used in conjunction with the MSE from splineCinterp to compute the spline weights (wb)
    if len(args[0]) == 2:
        dSTR_s = args[0][0]
        dSTR_e = args[0][1]
    else:
        dSTR_s = '2016-07-17T00:00:00Z'
        dSTR_e = '2016-09-01T00:00:00Z'
    cBathyYbounds = [0, 1250]
    cBathyXbounds = [0, 500]
    scalecDict = {}
    scalecDict['x_smooth'] = x_smooth
    scalecDict['y_smooth'] = y_smooth
    splineDict = {}
    splineDict['splinebctype'] = splinebctype
    splineDict['lc'] = lc
    splineDict['dxm'] = dxm
    splineDict['dxi'] = dxi
    splineDict['targetvar'] = targetvar
    splineDict['wbysmooth'] = wbysmooth
    splineDict['wbxsmooth'] = wbxsmooth
    # first make survey
    if version == 'survey':
        wrappers.makeBathySurvey(dSTR_s, dSTR_e, dir_loc='/home/number/thredds_data/integratedBathyProduct/survey', scalecDict=scalecDict, splineDict=splineDict,
                         ncStep='monthly', plot=True)
    # #then make all cBKF
    elif version == 'cBKF':
        wrappers.makeBathyCBATHY(dSTR_s, dSTR_e, dir_loc='/home/number/thredds_data/integratedBathyProduct/cBKF_long', scalecDict=scalecDict, splineDict=splineDict,
                             plot=True, xbounds=cBathyXbounds, ybounds=cBathyYbounds, ncStep='daily')
    # # then make cBKF-T
    elif version == 'cBKF-T':
        wrappers.makeBathyCBATHY(dSTR_s, dSTR_e, dir_loc='/home/number/thredds_data/integratedBathyProduct/cBKF-T_long', scalecDict=scalecDict, splineDict=splineDict,
                             plot=True, xbounds=cBathyXbounds, ybounds=cBathyYbounds, ncStep='daily', waveHeightThreshold=1.2)

if __name__ == "__main__":
    import getopt, sys
    opts, args = getopt.getopt(sys.argv[1:], "h", ["help"])
    version = args[0]
    argsIn = args[1:]
    runBathyProductManually(version,argsIn)
