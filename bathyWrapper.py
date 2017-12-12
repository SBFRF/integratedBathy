import netCDF4 as nc
import MakeUpdatedBathyDEM as mBATHY
import datetime as dt



def bathyWrapper():
    # ok, this is just my first crack at the wrapper for the makeUpdatedBathy script that spicer asked for
    # im going to have to get him to show me how to get it to run automatically

    # all this is going to do is wake up, check the most recent survey date and the most recent integrated bathy date
    # if the most recent survey is newer than the latest integrated bathy, then we run the script, otherwise, do nothing

    # generic locations
    survey_ncml = 'http://134.164.129.55/thredds/dodsC/FRF/geomorphology/elevationTransects/survey/surveyTransects.ncml'
    bathy_ncml = 'http://134.164.129.55/thredds/dodsC/cmtb/integratedBathyProduct/survey/survey.ncml'
    dir_loc = '/home/number/thredds_data/integratedBathyProduct/survey'

    # scale c and spline stuff
    # list of inputs!!!!!
    x_smooth = 40  # scale c interp x-direction smoothing
    y_smooth = 100  # scale c interp y-direction smoothing
    # splinebctype - this is the type of spline you want to force
    # options are....
    # 2 - second derivative goes to zero at boundary
    # 1 - first derivative goes to zero at boundary
    # 0 - value is zero at boundary
    # 10 - force value and derivative(first?!?) to zero at boundary
    splinebctype = 10
    off = 10 # offset for the edge splining!!!!!
    lc = 4  # spline smoothing constraint value
    dxm = 2  # coarsening of the grid for spline (e.g., 2 means calculate with a dx that is 2x input dx)
    # can be tuple if you want to do dx and dy seperately (dxm, dym), otherwise dxm is used for both
    dxi = 1  # fining of the grid for spline (e.g., 0.1 means return spline on a grid that is 10x input dx)
    # as with dxm, can be a tuple if you want seperate values for dxi and dyi
    targetvar = 0.8 # this is the target variance used in the spline function.

    # store that noise in a dictionary
    scalecDict = {}
    scalecDict['x_smooth'] = x_smooth
    scalecDict['y_smooth'] = y_smooth
    splineDict = {}
    splineDict['splinebctype'] = splinebctype
    splineDict['lc'] = lc
    splineDict['dxm'] = dxm
    splineDict['off'] = off
    splineDict['dxi'] = dxi
    splineDict['targetvar'] = targetvar

    # get survey info
    survey = nc.Dataset(survey_ncml)
    # pull down all the times....
    times = nc.num2date(survey.variables['time'][:], survey.variables['time'].units, survey.variables['time'].calendar)
    # get the last one
    survey_time = times[-1]

    # get bathy info
    bathy = nc.Dataset(bathy_ncml)
    # pull down all the times....
    times = nc.num2date(bathy.variables['time'][:], bathy.variables['time'].units, bathy.variables['time'].calendar)
    # get the last one
    bathy_time = times[-1]


    if survey_time > bathy_time:

        # do some stuff
        # tack on a day to survey time so that I'm sure to pull that specific survey?
        survey_time_N = survey_time + dt.timedelta(days=1)


        # convert these times to datestrings
        dSTR_s = bathy_time.strftime('%Y-%m-%dT%H%M%SZ')
        dSTR_e = survey_time_N.strftime('%Y-%m-%dT%H%M%SZ')

        mBATHY.makeUpdatedBATHY_transects(dSTR_s, dSTR_e, dir_loc, scalecDict=scalecDict, splineDict=splineDict, plot=None)
    else:
        print 'no new bathys'

if __name__ == "__main__":


    print '___________________\n________________\n___________________\n________________\n___________________\n________________\n'
    print 'USACE FRF Coastal Model Test Bed : Integrated Bathymetry Product'

    bathyWrapper()


