import wrappers
import os, glob
""" This came from DLY_test, separated now to make cleaner """
# list of inputs!!!!!
x_smooth = 20   # scale c interp x-direction smoothing
y_smooth = 100  # scale c interp y-direction smoothing
# splinebctype - this is the type of spline you want to force

# 2 - second derivative goes to zero at boundary
# 1 - first derivative goes to zero at boundary
# 0 - value is zero at boundary
# 10 - force value and derivative(first?!?) to zero at boundary
splinebctype = 10
lc = [3, 12]  # spline smoothing constraint value [x, y] directions
dxm = 1  # coarsening of the grid for spline (e.g., 2 means calculate with a dx that is 2x input dx)
# can be tuple if you want to do dx and dy seperately (dxm, dym), otherwise dxm is used for both
dxi = 1  # fining of the grid for spline (e.g., 0.1 means return spline on a grid that is 10x input dx)
# as with dxm, can be a tuple if you want separate values for dxi and dyi
targetvar = 0.25 # this is the target variance used in the spline function.
wbysmooth = 300  # y-edge smoothing scale
wbxsmooth = 100  # x-edge smoothing scale
# It is used in conjunction with the MSE from splineCinterp to compute the spline weights (wb)
dSTR_s = '2014-01-01T00:00:00Z'
dSTR_e = '2018-01-01T00:00:00Z'

dir_loc = '/home/number/thredds_data/integratedBathyProduct/survey'

# this is where I am going to save the monthy nc files

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
wrappers.makeBathySurvey(dSTR_s, dSTR_e, dir_loc, scalecDict=scalecDict, splineDict=splineDict)

# wrappers.makeBathyCBATHY(dSTR_s, dSTR_e, dir_loc, scalecDict=scalecDict, splineDict=splineDict)
for xx in [20]:
    scalecDict['x_smooth'] = xx
    for ll in [(3,8), (3,12)]:
        splineDict['lc'] = ll
        print ll, xx
        wrappers.makeBathySurvey(dSTR_s, dSTR_e, dir_loc, scalecDict=scalecDict, splineDict=splineDict)
        flist = glob.glob(os.path.join(dir_loc,'2015','CMTB*201510.nc'))
        os.rename(flist[0], flist[0].split('.')[0]+'_{}_{}.nc'.format(xx, ll))