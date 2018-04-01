import wrappers
import os, glob
""" This came from DLY_test, separated now to make cleaner will process/re process gridded data of interest
 will use the last data on thredds previous to start date as background."""
def runBathyProductManually(version):
    """This will run the bathy product manually from the command line (for speed) with a version

    :param version: acceptable versoins are cBKF, cBKF-T, and survey

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
    wbysmooth = 300  # y-edge smoothing scale
    wbxsmooth = 100  # x-edge smoothing scale
    # It is used in conjunction with the MSE from splineCinterp to compute the spline weights (wb)
    dSTR_s = '2015-09-12T00:00:00Z'
    dSTR_e = '2016-10-01T00:00:00Z'

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
        wrappers.makeBathyCBATHY(dSTR_s, dSTR_e, dir_loc='/home/number/thredds_data/integratedBathyProduct/cBKF', scalecDict=scalecDict, splineDict=splineDict,
                             plot=True, xbounds=[0,500], ybounds=[0,1000], ncStep='daily')
    # # then make cBKF-T
    elif version == 'cBKF-T':
        wrappers.makeBathyCBATHY(dSTR_s, dSTR_e, dir_loc='/home/number/thredds_data/integratedBathyProduct/cBKF-T', scalecDict=scalecDict, splineDict=splineDict,
                             plot=True, xbounds=[0,500], ybounds=[0,1000], ncStep='daily', waveHeightThreshold=1.2)

if __name__ == "__main__":
    import getopt, sys
    opts, args = getopt.getopt(sys.argv[1:], "h", ["help"])
    version = args[0]
    runBathyProductManually(version)
