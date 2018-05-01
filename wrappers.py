import os
import numpy as np
from getdatatestbed.getDataFRF import getObs
import datetime as DT
import MakeUpdatedBathyDEM as mBATHY
from collections import OrderedDict
import netCDF4 as nc
from sblib import geoprocess as gp
import makenc
from matplotlib import pyplot as plt
from sblib import sblib as sb

def makeBathyCBATHY(dSTR_s, dSTR_e, dir_loc, scalecDict=None, splineDict=None, ncStep='daily', plot=None, **kwargs):
    """
    :param dSTR_s: string that determines the start date of the times of the surveys you want to use to update the DEM
                    format is  dSTR_s = '2013-01-04T00:00:00Z'
                    no matter what you put here, it will always round it down to the beginning of the month
    :param dSTR_e: string that determines the end date of the times of the surveys you want to use to update the DEM
                    format is dSTR_e = '2014-12-22T23:59:59Z'
                    no matter what you put here, it will always round it up to the end of the month
    :param dir_loc: place where you want to save the .nc files that get written
                    the function will make the year directories inside of this location on its own.
    :param scalecDict: keys are:
                        x_smooth - x direction smoothing length for scalecInterp
                        y_smooth - y direction smoothing length for scalecInterp

                        if not specified it will default to:
                        x_smooth = 20
                        y_smooth = 200
    :param splineDict: keys are:
                        splinebctype
                            options are....
                            2 - second derivative goes to zero at boundary
                            1 - first derivative goes to zero at boundary
                            0 - value is zero at boundary
                            10 - force value and derivative(first?!?) to zero at boundary
                        lc - spline smoothing constraint value (integer <= 1)
                        dxm -  coarsening of the grid for spline (e.g., 2 means calculate with a dx that is 2x input dx)
                                can be tuple if you want to do dx and dy separately (dxm, dym), otherwise dxm is used for both
                        dxi - fining of the grid for spline (e.g., 0.1 means return spline on a grid that is 10x input dx)
                                as with dxm, can be a tuple if you want separate values for dxi and dyi
                        targetvar - this is the target variance used in the spline function.
                        wbysmooth - y-edge smoothing length scale
                        wbxsmooth - x-edge smoothing length scale

                        if not specified it will default to:
                        splinebctype = 10
                        lc = 4
                        dxm = 2
                        dxi = 1
                        targetvar = 0.45
                        wbysmooth = 300
                        wbxsmooth = 100
    :param ncStep: do you want to make monthly or daily nc files?
                    options are 'monthly' or 'daily'.  'monthly' is NOT recommended for CBATHY, it will create very large files
    :param **kwargs:
        xbounds a tupel or list that bounds the cbathy querey  (frf coords)
        ybounds a tupel or list that bounds the cbaty data querey (frf coords)
        waveHeightThreshold: runs the thresholded wave height kalman filter for cBathy

    :return:
            writes ncfiles for the new DEM's to the location dir_loc
    """
    # we are starting from the most up-to-date integrated bathymetry product, so this is the ncml from that!
    # this is the URL by which the scripts starts
    nc_b_url = 'http://134.164.129.55/thredds/dodsC/cmtb/integratedBathyProduct/survey/survey.ncml'
    # this is the ncml for the bathymetries that is checked to find where to pick up from
    nc_url = 'http://134.164.129.55/thredds/dodsC/cmtb/integratedBathyProduct/cBKF/cBKF.ncml'
    if 'xbounds' in kwargs and np.array(kwargs['xbounds']).size == 2:
        xminC, xmaxC = kwargs['xbounds'][0], kwargs['xbounds'][1]
    else:
        print('Assumed cBathy xBounds from Holman 2013')
        xminC, xmaxC = 0, 500  # from
    if 'ybounds' in kwargs and np.array(kwargs['ybounds']).size == 2:
        yminC, ymaxC = kwargs['ybounds'][0], kwargs['ybounds'][1]
    else:
        print('Assumed cBahy yBounds from Holamn 2013')
        yminC, ymaxC = 0, 1000
    # Yaml files for my .nc files!!!!!
    global_yaml = 'yamls/IntegratedcBathy_Global.yml'
    var_yaml = 'yamls/IntegratedBathy_grid_var.yml'

    ################################################
    # Setup done, now check what file size to make #
    ################################################
    if 'monthly' in ncStep:
        # force the survey to start at the first of the month and end at the last of the month!!!!
        dSTR_s = dSTR_s[0:7] + '-01T00:00:00Z'
        if dSTR_e[5:7] == '12':
            dSTR_e = str(int(dSTR_e[0:4]) + 1) + '-01' + '-01T00:00:00Z'
        else:
            dSTR_e = dSTR_e[0:5] + str(int(dSTR_e[5:7]) + 1).zfill(2) + '-01T00:00:00Z'
        d_e = DT.datetime.strptime(dSTR_e, '%Y-%m-%dT%H:%M:%SZ')
        d_s = DT.datetime.strptime(dSTR_s, '%Y-%m-%dT%H:%M:%SZ')
        dList = OrderedDict(
            ((d_s + DT.timedelta(_)).strftime(r"%b-%y"), None) for _ in xrange((d_e - d_s).days + 31)).keys()
        dList = [DT.datetime.strptime(date, r"%b-%y") for date in dList]

    elif 'daily' in ncStep:
        # force the survey to start at the beginning of the day and end at the beginning of the next day
        dSTR_s = dSTR_s[0:11] + '00:00:00Z'
        dSTR_e = dSTR_e[0:11] + '00:00:00Z'
        d_e = DT.datetime.strptime(dSTR_e, '%Y-%m-%dT%H:%M:%SZ')
        d_e = d_e + DT.timedelta(days=1)
        d_s = DT.datetime.strptime(dSTR_s, '%Y-%m-%dT%H:%M:%SZ')
        dList = OrderedDict(((d_s + DT.timedelta(_)), None) for _ in xrange((d_e - d_s).days + 1)).keys()

    else:
        raise NotImplementedError('ncStep input not recognized.  Acceptable inputs include "daily" or "monthly".')
    #######################################
    # Begin Loop Through time             #
    #######################################
    nsteps = np.size(dList) - 1
    for tt in range(0, nsteps):  # break into 'daily' or 'monthly'
        # pull out the dates I need for this step!
        d1 = dList[tt]
        d2 = dList[tt + 1]

        # get the data that I need
        go = getObs(d1, d2)
        cBathy = go.getBathyGridcBathy(xbounds=[xminC, xmaxC], ybounds=[yminC, ymaxC])
        if cBathy == None:
            continue # go to next step, don't make anything if there's no data
        print('\n     Doing Grid at {}\n'.format(cBathy['time'][0].date()))
        # prep the data for interpolation
        # put new data into dictionary
        if 'waveHeightThreshold' in kwargs:  # first try thresholded cBathy
            # set variables
            waveHsThreshold = kwargs['waveHeightThreshold']
            # go get wave data
            try:
                rawspec = go.getWaveSpec('waverider-26m')
            except: # when there's no data at 26 go to 17
                rawspec = go.getWaveSpec('waverider-17m')

            from sblib import kalman_filter
            newDict = kalman_filter.cBathy_ThresholdedLogic(cBathy, rawspec, waveHsThreshold)
            if newDict == None:
                print('kalman Filter Returned NONE')
                continue
            nc_url = 'http://134.164.129.55/thredds/dodsC/cmtb/integratedBathyProduct/cBKF-T/cBKF-T.ncml'
            ncFnameBase = 'CMTB-integratedBathyProduct_cBKF-T_'
            global_yaml = 'yamls/IntegratedcBKF-T_Global.yml'
            fig_loc = 'figures/cbathy_thresh'
        else:
            newDict = {}
            newDict['elevation'] = -cBathy['depthKF'] # flip sign to negative down
            newDict['xFRF'] = cBathy['xm']
            newDict['yFRF'] = cBathy['ym']
            newDict['surveyMeanTime'] = cBathy['epochtime'][-1] # this is the last log of update time
            newDict['epochtime'] = cBathy['epochtime']
            ncFnameBase = 'CMTB-integratedBathyProduct_cBKF_'
            fig_loc = 'figures/cbathy'
        # put old/background data into dictionary, decide where to get background data from
        backgroundDict = {}
        try:
            # look for the .nc file that I just wrote!!! (if more than 1st of loop)
            old_bathy = nc.Dataset(os.path.join(prev_nc_loc, prev_nc_name))
            ob_times = nc.num2date(old_bathy.variables['time'][-1], old_bathy.variables['time'].units,
                                   old_bathy.variables['time'].calendar)
            # find newest time prior to this
            t_mask = (ob_times <= d1)  # boolean true/false of time
            t_idx = np.where(t_mask)[0][-1]  # I want the MOST RECENT ONE - i.e., the last one
            # backgroundDict['elevation'] = old_bathy.variables['elevation'][t_idx, :]
            # backgroundDict['xFRF'] = old_bathy.variables['xFRF'][:]
            # backgroundDict['yFRF'] = old_bathy.variables['yFRF'][:]
            # backgroundDict['updateTime'] = old_bathy.variables['updateTime'][-1]
        except (IOError, UnboundLocalError):  # IO for when files aren't on server, unbound handles first time through loop
            try:
                # look for the most up to date bathy in the ncml file....
                old_bathy = nc.Dataset(nc_url)
                ob_times = nc.num2date(old_bathy.variables['time'][:], old_bathy.variables['time'].units,
                                       old_bathy.variables['time'].calendar)
                # find newest time prior to this
                t_mask = (ob_times <= d1)  # boolean true/false of time
                t_idx = np.where(t_mask)[0][-1]  # I want the MOST RECENT ONE - i.e., the last one
                # backgroundDict['elevation'] = old_bathy.variables['elevation'][t_idx, :]
                # backgroundDict['xFRF'] = old_bathy.variables['xFRF'][:]
                # backgroundDict['yFRF'] = old_bathy.variables['yFRF'][:]
                # backgroundDict['updateTime'] = old_bathy.variables['updateTime'][-1]
            except (IOError, IndexError): #index errors for first one
                # there is no established data set, current, build from scratch
                #  start with the nc_b_url data (this case integrated bathy)
                old_bathy = nc.Dataset(nc_b_url)
                ob_times = nc.num2date(old_bathy.variables['time'][:], old_bathy.variables['time'].units, old_bathy.variables['time'].calendar)
                # find newest time prior to this
                t_mask = (ob_times <= d_s)  # boolean true/false of time
                t_idx = np.where(t_mask)[0][-1]  # I want the MOST RECENT ONE - i.e., the last one
        backgroundDict['elevation'] = old_bathy.variables['elevation'][t_idx, :]
        backgroundDict['xFRF'] = old_bathy.variables['xFRF'][:]
        backgroundDict['yFRF'] = old_bathy.variables['yFRF'][:]
        backgroundDict['updateTime'] = old_bathy.variables['updateTime'][t_idx]

        # go time!  this is scaleC + spline!
        updatedBathy = mBATHY.makeUpdatedBATHY(backgroundDict, newDict, scalecDict=scalecDict, splineDict=splineDict)
        ##########################################
        # Data preparation for output file       #
        ##########################################
        # get position stuff that will be constant for all surveys!!!
        xFRFi, yFRFi = np.meshgrid(updatedBathy['xFRF'], updatedBathy['yFRF'])
        xFRFi_vecN = xFRFi.reshape((1, xFRFi.shape[0] * xFRFi.shape[1]))[0]
        yFRFi_vecN = yFRFi.reshape((1, yFRFi.shape[0] * yFRFi.shape[1]))[0]
        # convert FRF coords to lat/lon
        test = gp.FRF2ncsp(xFRFi_vecN, yFRFi_vecN)
        # go through stateplane to avoid FRFcoords trying to guess the input coordinate systems
        temp = gp.ncsp2LatLon(test['StateplaneE'], test['StateplaneN'])
        lat_vec = temp['lat']
        lon_vec = temp['lon']
        E_vec = test['StateplaneE']
        N_vec = test['StateplaneN']

        lat = lat_vec.reshape(xFRFi.shape[0], xFRFi.shape[1])
        lon = lon_vec.reshape(xFRFi.shape[0], xFRFi.shape[1])
        E = E_vec.reshape(xFRFi.shape[0], xFRFi.shape[1])
        N = N_vec.reshape(xFRFi.shape[0], xFRFi.shape[1])

        xFRF = xFRFi[0, :]
        yFRF = yFRFi[:, 1]
        latitude = lat
        longitude = lon
        easting = E
        northing = N

        # write the nc_file for this month, like a boss, with greatness
        nc_dict = {}
        nc_dict['elevation'] = updatedBathy['elevation']
        nc_dict['updateTime'] = updatedBathy['updateTime']
        nc_dict['xFRF'] = xFRF
        nc_dict['yFRF'] = yFRF
        nc_dict['latitude'] = latitude
        nc_dict['longitude'] = longitude
        nc_dict['northing'] = northing
        nc_dict['easting'] = easting
        nc_dict['y_smooth'] = updatedBathy['smoothAL']
        # also want cbathy time....
        nc_dict['time'] = newDict['epochtime']

        # create file name string
        if 'monthly' in ncStep:
            nc_name = ncFnameBase + d1.strftime('%Y%m') + '.nc'
        elif 'daily' in ncStep:
            nc_name = ncFnameBase + d1.strftime('%Y%m%d') + '.nc'

        # check to see if year directory exists
        if not os.path.isdir(os.path.join(dir_loc, d1.strftime('%Y'))):
            # if not, make year directory
            os.makedirs(os.path.join(dir_loc, d1.strftime('%Y')))

        # create file save path (complete with year)
        nc_loc = os.path.join(dir_loc, d1.strftime('%Y'))
        # save em like a boss?
        makenc.makenc_tiBATHY(os.path.join(nc_loc, nc_name), nc_dict, globalYaml=global_yaml, varYaml=var_yaml)

        # save this location for next time through the loop
        prev_nc_name = nc_name
        prev_nc_loc = nc_loc

        # plot QA QC plots to ensure we're doing what we think we are
        if plot is not None:
           import plotFuncs
           plotFuncs.bathyQAQCplots(fig_loc, cBathy['time'][0], updatedBathy=updatedBathy)


def makeBathySurvey(dSTR_s, dSTR_e, dir_loc, scalecDict=None, splineDict=None, ncStep='monthly', plot=None):
    """
    :param dSTR_s: string that determines the start date of the times of the surveys you want to use to update the DEM
                    format is  dSTR_s = '2013-01-04T00:00:00Z'
                    no matter what you put here, it will always round it down to the beginning of the month
    :param dSTR_e: string that determines the end date of the times of the surveys you want to use to update the DEM
                    format is dSTR_e = '2014-12-22T23:59:59Z'
                    no matter what you put here, it will always round it up to the end of the month
    :param dir_loc: place where you want to save the .nc files that get written
                    the function will make the year directories inside of this location on its own.
    :param scalecDict: keys are:
                        x_smooth - x direction smoothing length for scalecInterp
                        y_smooth - y direction smoothing length for scalecInterp

                        if not specified it will default to:
                        x_smooth = 100
                        y_smooth = 200
    :param splineDict: keys are:
                        splinebctype
                            options are....
                            2 - second derivative goes to zero at boundary
                            1 - first derivative goes to zero at boundary
                            0 - value is zero at boundary
                            10 - force value and derivative(first?!?) to zero at boundary
                        lc - spline smoothing constraint value (integer <= 1)
                        dxm -  coarsening of the grid for spline (e.g., 2 means calculate with a dx that is 2x input dx)
                                can be tuple if you want to do dx and dy separately (dxm, dym), otherwise dxm is used for both
                        dxi - fining of the grid for spline (e.g., 0.1 means return spline on a grid that is 10x input dx)
                                as with dxm, can be a tuple if you want separate values for dxi and dyi
                        targetvar - this is the target variance used in the spline function.
                        wbysmooth - y-edge smoothing length scale
                        wbxsmooth - x-edge smoothing length scale

                        if not specified it will default to:
                        splinebctype = 10
                        lc = 4
                        dxm = 2
                        dxi = 1
                        targetvar = 0.45
                        wbysmooth = 300
                        wbxsmooth = 100
    :param ncStep: do you want to make monthly or daily nc files?
                    options are 'monthly' or 'daily'.  'monthly' is NOT recommended for CBATHY!!!!!
    :return:
            writes ncfiles for the new DEM's to the location dir_loc
    """
    # this is the standard background bathymetry that we started from.
    nc_b_url = 'http://134.164.129.55/thredds/dodsC/cmtb/grids/TimeMeanBackgroundDEM/backgroundDEMt0_TimeMean.nc'
    # nc_b_url = 'BackgroundFiles/backgroundDEMt0_TimeMean.nc'
    # this is the ncml for the bathymetries that is checked to find where to pick up from
    nc_url = 'http://134.164.129.55/thredds/dodsC/cmtb/integratedBathyProduct/survey/survey.ncml'
    fig_loc = 'figures/suvey'  # will make if doesn't exist
    # Yaml files for my .nc files!!!!!
    global_yaml = 'yamls/IntegratedBathy_Global.yml'
    var_yaml = 'yamls/BATHY/FRFti_var.yml'

    # check the ncStep input
    d_s = None
    ################################################
    # Setup done, now check what file size to make #
    ################################################
    if 'monthly' in ncStep:
        # force the survey to start at the first of the month and end at the last of the month!!!!
        dSTR_s = dSTR_s[0:7] + '-01T00:00:00Z'
        if dSTR_e[5:7] == '12':
            dSTR_e = str(int(dSTR_e[0:4]) + 1) + '-01' + '-01T00:00:00Z'
        else:
            dSTR_e = dSTR_e[0:5] + str(int(dSTR_e[5:7]) + 1).zfill(2) + '-01T00:00:00Z'
        d_e = DT.datetime.strptime(dSTR_e, '%Y-%m-%dT%H:%M:%SZ')
        d_s = DT.datetime.strptime(dSTR_s, '%Y-%m-%dT%H:%M:%SZ')
        dList = OrderedDict(((d_s + DT.timedelta(_)).strftime(r"%b-%y"), None) for _ in xrange((d_e - d_s).days)).keys()
        dList = [DT.datetime.strptime(date, r"%b-%y") for date in dList]


    elif 'daily' in ncStep:
        # force the survey to start at the beginning of the day and end at the beginning of the next day
        dSTR_s = dSTR_s[0:11] + '00:00:00Z'
        dSTR_e = dSTR_e[0:11] + '00:00:00Z'
        d_e = DT.datetime.strptime(dSTR_e, '%Y-%m-%dT%H:%M:%SZ')
        d_e = d_e + DT.timedelta(days=1)
        d_s = DT.datetime.strptime(dSTR_s, '%Y-%m-%dT%H:%M:%SZ')
        dList = OrderedDict(((d_s + DT.timedelta(_)), None) for _ in xrange((d_e - d_s).days + 1)).keys()

    else:
        assert d_s is not None, 'ncStep input not recognized.  Acceptable inputs include daily or monthly.'

    # loop time
    nsteps = np.size(dList) - 1
    for tt in range(0, nsteps):
        # pull out the dates I need for this step!
        d1 = dList[tt]
        d2 = dList[tt + 1]
        print('\n     Doing Grid at {}\n'.format(d1.date()))

        # prep the data
        # get the survey data that I need
        newDict = mBATHY.getSurveyData(d1, d2)

        if newDict['elevation'] is None:
            continue
        else:
            # create my list of times for each survey?
            surveys = np.unique(newDict['surveyNumber'])
            surveyTime = np.zeros(np.shape(surveys))
            for ss in range(0, len(surveys)):
                # get the times of each survey
                idt = (newDict['surveyNumber'] == surveys[ss])
                # convert the times to a single time for each survey
                stimesS = newDict['surveyTime'][idt]
                if (max(stimesS) - min(stimesS)) > DT.timedelta(days=3):
                    surveyTime[ss] = -1000
                else:
                    # pull out the mean time
                    stimeMS = min(stimesS) + (max(stimesS) - min(stimesS)) / 2
                    # round it to nearest 12 hours.
                    stimeM = sb.roundtime(stimeMS, roundTo=1 * 6 * 3600) + DT.timedelta(hours=0.25)
                    # check to make sure the mean time is in the month we just pulled
                    if (stimeM >= d1) & (stimeM < d2):
                        timeunits = 'seconds since 1970-01-01 00:00:00'
                        surveyTime[ss] = nc.date2num(stimeM, timeunits)
                    else:
                        surveyTime[ss] = -1000

            # throw out weird surveys
            indKeep = np.where(surveyTime >= 0)
            surveyTime = surveyTime[indKeep]
            surveys = surveys[indKeep]

            # also remove that data from newDict
            tempSurvNum = newDict['surveyNumber']
            tempElev = newDict['elevation']
            tempxFRF = newDict['xFRF']
            tempyFRF = newDict['yFRF']
            tempProfNum = newDict['profileNumber']
            tempSurvTime = newDict['surveyTime']
            indKeepData = np.where(np.in1d(tempSurvNum, surveys))
            newDict['surveyNumber'] = tempSurvNum[indKeepData]
            newDict['elevation'] = tempElev[indKeepData]
            newDict['xFRF'] = tempxFRF[indKeepData]
            newDict['yFRF'] = tempyFRF[indKeepData]
            newDict['profileNumber'] = tempProfNum[indKeepData]
            newDict['surveyTime'] = tempSurvTime[indKeepData]
            newDict['surveyMeanTime'] = surveyTime


            # background data
            backgroundDict = {}
            try:  # look for the .nc file that I just wrote!!!
                old_bathy = nc.Dataset(os.path.join(prev_nc_loc, prev_nc_name))
                ob_times = nc.num2date(old_bathy.variables['time'][:], old_bathy.variables['time'].units,
                                       old_bathy.variables['time'].calendar)
                # find newest time prior to this
                t_mask = (ob_times <= d_s)  # boolean true/false of time
                t_idx = np.where(t_mask)[0][-1]  # I want the MOST RECENT ONE - i.e., the last one
                backgroundDict['elevation'] = old_bathy.variables['elevation'][t_idx, :]
                backgroundDict['xFRF'] = old_bathy.variables['xFRF'][:]
                backgroundDict['yFRF'] = old_bathy.variables['yFRF'][:]
                backgroundDict['updateTime'] = old_bathy.variables['updateTime'][t_idx, :]
            except:
                try: # if i didn't just write any, look to the thredds server to find where i last left off
                    # look for the most up to date bathy in the ncml file....
                    old_bathy = nc.Dataset(nc_url)
                    ob_times = nc.num2date(old_bathy.variables['time'][:], old_bathy.variables['time'].units,
                                           old_bathy.variables['time'].calendar)
                    # find newest time prior to this
                    t_mask = (ob_times <= d_s)  # boolean true/false of time
                    t_idx = np.where(t_mask)[0][-1]  # I want the MOST RECENT ONE - i.e., the last one
                    backgroundDict['elevation'] = old_bathy.variables['elevation'][t_idx, :]
                    backgroundDict['xFRF'] = old_bathy.variables['xFRF'][:]
                    backgroundDict['yFRF'] = old_bathy.variables['yFRF'][:]
                    backgroundDict['updateTime'] = old_bathy.variables['updateTime'][t_idx, :]
                except:
                    # pull the time mean background bathymetry if you don't have any updated ones
                    old_bathy = nc.Dataset(nc_b_url)
                    backgroundDict['elevation'] = old_bathy.variables['elevation'][:]
                    backgroundDict['xFRF'] = old_bathy.variables['xFRF'][:]
                    backgroundDict['yFRF'] = old_bathy.variables['yFRF'][:]
                    backgroundDict['tmBackTog'] = True
                    # it wont have an update time in this case, because it came from the time-mean background.
                    # so what should go here instead?
                    tempUpTime = np.zeros(np.shape(backgroundDict['elevation']))
                    tempUpTime[:] = np.nan
                    backgroundDict['updateTime'] = np.ma.array(tempUpTime, mask=np.ones(np.shape(backgroundDict['elevation'])), fill_value=-999)

                    # how we looking?
                    # fig_loc = 'C:\Users\dyoung8\Desktop\David Stuff\Projects\CSHORE\Bathy Interpolation\TestFigures_CBATHY'
                    # # background I made from the immediately preceding survey?
                    # tempD = d1.strftime(r"%Y-%m-%d")
                    # fig_name = 'backgroundDEM_' + tempD[0:4] + tempD[5:7] + '.png'
                    # plt.figure()
                    # plt.pcolor(backgroundDict['xFRF'], backgroundDict['yFRF'], backgroundDict['elevation'], cmap=plt.cm.jet, vmin=-13, vmax=5)
                    # cbar = plt.colorbar()
                    # cbar.set_label('(m)')
                    # axes = plt.gca()
                    # axes.set_xlim([-50, 550])
                    # axes.set_ylim([-50, 1050])
                    # plt.xlabel('xFRF (m)')
                    # plt.ylabel('yFRF (m)')
                    # plt.savefig(os.path.join(fig_loc, fig_name))
                    # plt.close()
                    #
                    # t = 1
                    # go time!  this is scaleC + spline!
            out = mBATHY.makeUpdatedBATHY(backgroundDict, newDict, scalecDict=scalecDict, splineDict=splineDict)
            elevation = out['elevation']
            smoothAL = out['smoothAL']
            updateTime = out['updateTime']

            # check to see if any of the data has nan's.
            # If so that means there were not enough profile lines to create a survey.  Check for and remove them
            idN = np.isnan(smoothAL)
            if np.sum(idN) > 0:
                # drop all NaN stuff
                elevation = elevation[~idN, :, :]
                smoothAL = smoothAL[~idN]
                surveyTime = surveyTime[~idN]
                surveys = surveys[~idN]
                updateTime = updateTime[~idN, :, :]

            # is there anything left?
            test = np.shape(elevation)
            if test[0] < 1:
                # you got rid of everything in this file!
                continue
            else:
                # write this to the appropriate nc_file!
                # get position stuff that will be constant for all surveys!!!
                xFRFi, yFRFi = np.meshgrid(out['xFRF'], out['yFRF'])
                xFRFi_vecN = xFRFi.reshape((1, xFRFi.shape[0] * xFRFi.shape[1]))[0]
                yFRFi_vecN = yFRFi.reshape((1, yFRFi.shape[0] * yFRFi.shape[1]))[0]
                # convert FRF coords to lat/lon
                test = gp.FRF2ncsp(xFRFi_vecN, yFRFi_vecN)
                # go through stateplane to avoid FRFcoords trying to guess the input coordinate systems
                temp = gp.ncsp2LatLon(test['StateplaneE'], test['StateplaneN'])
                lat_vec = temp['lat']
                lon_vec = temp['lon']
                E_vec = test['StateplaneE']
                N_vec = test['StateplaneN']

                lat = lat_vec.reshape(xFRFi.shape[0], xFRFi.shape[1])
                lon = lon_vec.reshape(xFRFi.shape[0], xFRFi.shape[1])
                E = E_vec.reshape(xFRFi.shape[0], xFRFi.shape[1])
                N = N_vec.reshape(xFRFi.shape[0], xFRFi.shape[1])

                xFRF = xFRFi[0, :]
                yFRF = yFRFi[:, 1]
                latitude = lat
                longitude = lon
                easting = E
                northing = N
                # write the nc_file for this month, like a boss, with greatness
                nc_dict = {}
                nc_dict['elevation'] = elevation
                nc_dict['xFRF'] = xFRF
                nc_dict['yFRF'] = yFRF
                nc_dict['latitude'] = latitude
                nc_dict['longitude'] = longitude
                nc_dict['northing'] = northing
                nc_dict['easting'] = easting
                nc_dict['time'] = np.array(surveyTime)
                nc_dict['surveyNumber'] = surveys
                nc_dict['y_smooth'] = smoothAL
                nc_dict['updateTime'] = updateTime

                # what does the name need to be?
                tempD = d1.strftime(r"%Y-%m-%d")
                if 'monthly' in ncStep:
                    nc_name = 'CMTB-integratedBathyProduct_survey_' + tempD[0:4] + tempD[5:7] + '.nc'
                elif 'daily' in ncStep:
                    nc_name = 'CMTB-integratedBathyProduct_grid_' + tempD[0:4] + tempD[5:7] + tempD[8:10] + '.nc'

                # check to see if year directory exists
                if os.path.isdir(os.path.join(dir_loc, tempD[0:4])):
                    pass
                else:
                    # if not, make year directory
                    os.makedirs(os.path.join(dir_loc, tempD[0:4]))

                # where am I saving these nc's as I make them
                nc_loc = os.path.join(dir_loc, tempD[0:4])
                # save em?
                makenc.makenc_tiBATHY(os.path.join(nc_loc, nc_name), nc_dict, globalYaml=global_yaml, varYaml=var_yaml)

                # save this location for next time through the loop
                prev_nc_name = nc_name
                prev_nc_loc = nc_loc

                # plot some stuff ( fig_loc, d1,
                if plot is not None:
                   import plotFuncs
                   plotFuncs.bathyQAQCplots(fig_loc, d1, updatedBathy=out)

