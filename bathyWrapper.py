import matplotlib
matplotlib.use('Agg')
import netCDF4 as nc
import MakeUpdatedBathyDEM as mBATHY
import datetime as dt
import wrappers
import datetime as DT


def bathyWrapper():
    """ This script is going to wake up, check the most recent survey date and the most recent integrated bathy date
    if the most recent survey is newer than the latest integrated bathy, then we run the script, otherwise, do nothing,
    function runs with no input

    Returns:
     A netCDF files for bathymetry

    """
    # Server Locations  generic locations
    survey_ncml = 'http://134.164.129.55/thredds/dodsC/FRF/geomorphology/elevationTransects/survey/surveyTransects.ncml'
    bathy_ncml = 'http://134.164.129.55/thredds/dodsC/cmtb/integratedBathyProduct/survey/survey.ncml'
    dir_loc = u'/home/number/thredds_data/integratedBathyProduct/survey'

    # set input parameters for scale C interp in to dictionaries
    scalecDict = {'x_smooth': 20,         # scale c interp x-direction smoothing
                  'y_smooth': 100}        # scale c interp y-direction smoothing
    # splinebctype - this is the type of spline you want to force options are....
        # 2 - second derivative goes to zero at boundary
        # 1 - first derivative goes to zero at boundary
        # 0 - value is zero at boundary
        # 10 - force value and derivative (first?!?) to zero at boundary
    splineDict = {'splinebctype': 10,
                  'lc': [2,8],       # spline smoothing constraint value
                  'dxm': [1,3],      # coarsening of the grid for spline (e.g., 2 means calculate with a dx that is 2x input dx)  can be tuple if you want to do dx and dy seperately (dxm, dym), otherwise dxm is used for both
                  'off': 10,         # offset for the edge splining!!!!!
                  'dxi': 1,          # fining of the grid for spline (e.g., 0.1 means return spline on a grid that is 10x input dict as with dxm, can be a tuple if you want seperate values for dxi and dyi
                  'targetvar': 0.3,  # this is the target variance used in the spline function.
                  'wbysmooth': 300,  # y-edge smoothing scale
                  'wbxsmooth': 100}  # x-edge smoothing scale

    # get survey info
    survey = nc.Dataset(survey_ncml)
    # pull down all the times....
    survey_times = nc.num2date(survey['time'][:], survey['time'].units, survey['time'].calendar)

    # get integrated bathymetry info
    bathy = nc.Dataset(bathy_ncml)
    # pull down all the times....
    bathy_times = nc.num2date(bathy.variables['time'][:], bathy.variables['time'].units, bathy.variables['time'].calendar)

    print(" Last survey Time: {}  and Last Itegrated bathy time: {} ".format(survey_times[-1], bathy_times[-1]))
    if survey_times[-1] >= bathy_times[-1]:
        # go get survey and make the survey
        # tack on a day to survey time so that I'm sure to pull that specific survey
        survey_time_N = survey_times[-1] + dt.timedelta(days=1)

        # convert these times to datestrings
        dSTR_s = bathy_times[-1].strftime('%Y-%m-%dT%H%M%SZ')
        dSTR_e = survey_time_N.strftime('%Y-%m-%dT%H%M%SZ')
        print('start: {}   end:  {}'.format(dSTR_s, dSTR_e))
        wrappers.makeBathySurvey(dSTR_s, dSTR_e, dir_loc, scalecDict=scalecDict, splineDict=splineDict)

if __name__ == "__main__":
    print('Making CMTB: Integrated Bathymetry Product')
    bathyWrapper()  # run the above script


