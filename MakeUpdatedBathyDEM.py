import os
import netCDF4 as nc
import numpy as np
from sblib import geoprocess as gp
from sblib import sblib as sb
import makenc
from matplotlib import pyplot as plt
from bsplineFunctions import bspline_pertgrid
import datetime as DT
from scalecInterp_python.DEM_generator import DEM_generator



def makeUpdatedBATHY(dSTR_s, dSTR_e, dir_loc, scalecDict=None, splineDict=None, plot=None):

    #hard coded variables
    filelist = ['http://134.164.129.55/thredds/dodsC/FRF/geomorphology/elevationTransects/survey/surveyTransects.ncml']
    # this is just the location of the ncml for the transects!!!!!
    nc_url = 'http://134.164.129.62:8080/thredds/dodsC/CMTB/grids/UpdatedBackgroundDEM/UpdatedBackgroundDEM.ncml'
    # this is just the location of the ncml for the already created UpdatedDEM
    nc_b_loc = 'C:\Users\dyoung8\Desktop\David Stuff\Projects\CSHORE\Bathy Interpolation\TestNCfiles'
    nc_b_name = 'backgroundDEMt0.nc'
    # these together are the location of the standard background bathymetry that we started from.


    #check scalecDict and splineDict
    if scalecDict is None:
        x_smooth = 100  # scale c interp x-direction smoothing
        y_smooth = 200  # scale c interp y-direction smoothing
    else:
        x_smooth = scalecDict['x_smooth']  # scale c interp x-direction smoothing
        y_smooth = scalecDict['y_smooth']  # scale c interp y-direction smoothing

    if splineDict is None:
        splinebctype = 10
        lc = 4
        dxm = 2
        dxi = 1
        targetvar = 0.45
    else:
        splinebctype = splineDict['splinebctype']
        lc = splineDict['lc']
        dxm = splineDict['dxm']
        dxi = splineDict['dxi']
        targetvar = splineDict['targetvar']


    # force the survey to start at the first of the month and end at the last of the month!!!!
    dSTR_s = dSTR_s[0:7] + '-01T00:00:00Z'
    if dSTR_e[5:7] == '12':
        dSTR_e = str(int(dSTR_e[0:4]) + 1) + '-01' + '-01T00:00:00Z'
    else:
        dSTR_e = dSTR_e[0:5] + str(int(dSTR_e[5:7]) + 1).zfill(2) + '-01T00:00:00Z'

    d_s = DT.datetime.strptime(dSTR_s, '%Y-%m-%dT%H:%M:%SZ')

    # how many months, years between my start and end times?
    year_end = dSTR_e[0:4]
    month_end = dSTR_e[5:7]
    year_start = dSTR_s[0:4]
    month_start = dSTR_s[5:7]
    # how many years between them?
    num_yrs = int(year_end) - int(year_start)

    # show time....
    for ii in range(0, num_yrs):

        # make year directories!!!
        yrs_dir = str(int(year_start) + int(ii))
        # check to see if year directory exists
        if os.path.isdir(os.path.join(dir_loc, yrs_dir)):
            pass
        else:
            # if not, make year directory
            os.makedirs(os.path.join(dir_loc, yrs_dir))

        # where am I saving these nc's as I make them
        nc_loc = os.path.join(dir_loc, yrs_dir)

        # make a list of the months in this year
        if (yrs_dir == year_start) and (yrs_dir == year_end):
            # just one year?
            num_months = int(month_end) - int(month_start)
            months = [str(int(month_start) + int(jj)).zfill(2) for jj in range(0, num_months + 1)]
        elif (yrs_dir == year_start):
            # this is the start year
            num_months = int('12') - int(month_start)
            months = [str(int(month_start) + int(jj)).zfill(2) for jj in range(0, num_months + 1)]
        elif (yrs_dir == year_end):
            # this is the end year
            num_months = int(month_end) - int('01')
            months = [str(int('01') + int(jj)).zfill(2) for jj in range(0, num_months + 1)]
        else:
            # I need all months
            num_months = int('12') - int('01')
            months = [str(int('01') + int(jj)).zfill(2) for jj in range(0, num_months + 1)]

        # ok, now to make my nc files, I just need to go through and find all surveys that fall in these months
        bathy = nc.Dataset(filelist[0])
        # pull down all the times....
        times = nc.num2date(bathy.variables['time'][:], bathy.variables['time'].units, bathy.variables['time'].calendar)
        all_surveys = bathy.variables['surveyNumber'][:]

        for jj in range(0, len(months)):
            # pull out the beginning and end time associated with this month
            d1STR = yrs_dir + '-' + months[jj] + '-01T00:00:00Z'
            d1 = DT.datetime.strptime(d1STR, '%Y-%m-%dT%H:%M:%SZ')
            if int(months[jj]) == 12:
                d2STR = str(int(yrs_dir) + 1) + '-' + '01' + '-01T00:00:00Z'
                d2 = DT.datetime.strptime(d2STR, '%Y-%m-%dT%H:%M:%SZ')
            else:
                d2STR = yrs_dir + '-' + str(int(months[jj]) + 1) + '-01T00:00:00Z'
                d2 = DT.datetime.strptime(d2STR, '%Y-%m-%dT%H:%M:%SZ')

            # find some stuff here...
            mask = (times >= d1) & (times < d2)  # boolean true/false of time
            idx = np.where(mask)[0]

            # what surveys are in this range?
            surveys = np.unique(bathy.variables['surveyNumber'][idx])

            # if there are no surveys here, then skip the rest of this loop...
            if len(surveys) < 1:
                print('No surveys found for ' + yrs_dir + '-' + months[jj])
                continue
            else:
                pass

            # otherwise..., check to see the times of all the surveys...?
            for tt in range(0, len(surveys)):
                ids = (all_surveys == surveys[tt])
                surv_times = times[ids]
                # pull out the mean time
                surv_timeM = surv_times[0] + (surv_times[-1] - surv_times[0]) / 2
                # round it to nearest 12 hours.
                surv_timeM = sb.roundtime(surv_timeM, roundTo=1 * 12 * 3600)

                # if the rounded time IS in the month, great
                if (surv_timeM >= d1) and (surv_timeM < d2):
                    pass
                else:
                    # if not set it to a fill value
                    surveys[tt] == -1000
            # drop all the surveys that we decided are not going to go into this monthly file!
            surveys = surveys[surveys >= 0]

            # SEARCH FOR MOST RECENT BATHY HERE!!!
            try:
                # look for the .nc file that I just wrote!!!
                old_bathy = nc.Dataset(os.path.join(prev_nc_loc, prev_nc_name))
                ob_times = nc.num2date(old_bathy.variables['time'][:], old_bathy.variables['time'].units,
                                       old_bathy.variables['time'].calendar)

                # find newest time prior to this
                t_mask = (ob_times <= d1)  # boolean true/false of time
                t_idx = np.where(t_mask)[0][-1]  # I want the MOST RECENT ONE - i.e., the last one

                Zi = old_bathy.variables['elevation'][t_idx, :]
                xFRFi_vec = old_bathy.variables['xFRF'][:]
                yFRFi_vec = old_bathy.variables['yFRF'][:]
            except:
                try:
                    # look for the most up to date bathy in the ncml file....
                    old_bathy = nc.Dataset(nc_url)
                    ob_times = nc.num2date(old_bathy.variables['time'][:], old_bathy.variables['time'].units,
                                           old_bathy.variables['time'].calendar)
                    # find newest time prior to this
                    t_mask = (ob_times <= d_s)  # boolean true/false of time
                    t_idx = np.where(t_mask)[0][-1]  # I want the MOST RECENT ONE - i.e., the last one

                    Zi = old_bathy.variables['elevation'][t_idx, :]
                    xFRFi_vec = old_bathy.variables['xFRF'][:]
                    yFRFi_vec = old_bathy.variables['yFRF'][:]

                except:
                    # load the background bathy from netDCF file if you can't get the ncml
                    old_bathy = nc.Dataset(os.path.join(nc_b_loc, nc_b_name))
                    Zi = old_bathy.variables['elevation'][:]
                    xFRFi_vec = old_bathy.variables['xFRF'][:]
                    yFRFi_vec = old_bathy.variables['yFRF'][:]

            # read out the dx and dy of the background grid!!!
            # assume this is constant grid spacing!!!!!
            dx = abs(xFRFi_vec[1] - xFRFi_vec[0])
            dy = abs(yFRFi_vec[1] - yFRFi_vec[0])

            xFRFi, yFRFi = np.meshgrid(xFRFi_vec, yFRFi_vec)
            rows, cols = np.shape(xFRFi)

            # pre-allocate my netCDF dictionary variables here....
            elevation = np.zeros((len(surveys), rows, cols))
            xFRF = np.zeros(cols)
            yFRF = np.zeros(rows)
            latitude = np.zeros((rows, cols))
            longitude = np.zeros((rows, cols))
            surveyNumber = np.zeros(len(surveys))
            surveyTime = np.zeros(len(surveys))

            # ok, now that I have the list of the surveys I am going to keep.....
            for tt in range(0, len(surveys)):

                # get the times of each survey
                ids = (all_surveys == surveys[tt])
                surv_times = times[ids]
                s_mask = (times <= max(surv_times)) & (times >= min(surv_times))  # boolean true/false of time
                s_idx = np.where(s_mask)[0]

                # pull out this NC stuf!!!!!!!!
                dataX, dataY, dataZ = [], [], []
                dataX = bathy['xFRF'][s_idx]
                dataY = bathy['yFRF'][s_idx]
                dataZ = bathy['elevation'][s_idx]
                profNum = bathy['profileNumber'][s_idx]
                survNum = bathy['surveyNumber'][s_idx]
                stimes = nc.num2date(bathy.variables['time'][s_idx], bathy.variables['time'].units, bathy.variables['time'].calendar)
                # pull out the mean time
                stimeM = min(stimes) + (max(stimes) - min(stimes)) / 2
                # round it to nearest 12 hours.
                stimeM = sb.roundtime(stimeM, roundTo=1 * 12 * 3600)

                assert len(np.unique(survNum)) == 1, 'MakeUpdatedBathyDEM error: You have pulled down more than one survey number!'
                assert isinstance(dataZ, np.ndarray), 'MakeUpdatedBathyDEM error: Script only handles np.ndarrays for the transect data at this time!'


                # build my new bathymetry from the FRF transect files

                # divide my survey up into the survey lines!!!
                profNum_list = np.unique(profNum)
                prof_minX = np.zeros(np.shape(profNum_list))
                prof_maxX = np.zeros(np.shape(profNum_list))
                for ss in range(0, len(profNum_list)):
                    # pull out all x-values corresponding to this profNum
                    Xprof = dataX[np.where(profNum == profNum_list[ss])]
                    prof_minX[ss] = min(Xprof)
                    prof_maxX[ss] = max(Xprof)

                # this rounds all these numbers down to the nearest dx
                prof_minX = prof_minX - (prof_minX % dx)
                prof_maxX = prof_maxX - (prof_maxX % dx)
                # note: this only does what you want if the numbers are all POSITIVE!!!!

                # check my y-bounds
                prof_maxY = max(dataY) - (max(dataY) % dy)
                minY = min(dataY)
                # it does check for negative y-values!!!
                if minY > 0:
                    prof_minY = minY - (minY % dy)
                else:
                    prof_minY = minY - (minY % dy) + dy

                # ok, I am going to force the DEM generator function to always go to the grid specified by these bounds!!
                # if you want to hand it a best guess grid, i.e., 'grid_filename' make SURE it has these bounds!!!!!!
                # or just don't hand it grid_filename....
                x0, y0 = np.median(prof_maxX), prof_maxY
                x1, y1 = np.median(prof_minX), prof_minY
                # currently using the median of the min and max X extends of each profile,
                # and just the min and max of the y-extents of all the profiles.

                # check to see if this is past the bounds of your background DEM.
                # if so, truncate so that it does not exceed.
                if x0 > max(xFRFi_vec):
                    x0 = max(xFRFi_vec)
                else:
                    pass
                if x1 < min(xFRFi_vec):
                    x1 = min(xFRFi_vec)
                else:
                    pass
                if y0 > max(yFRFi_vec):
                    y0 = max(yFRFi_vec)
                else:
                    pass
                if y1 < min(yFRFi_vec):
                    y1 = min(yFRFi_vec)
                else:
                    pass

                dict = {'x0': x0,  # gp.FRFcoord(x0, y0)['Lon'],  # -75.47218285,
                        'y0': y0,  # gp.FRFcoord(x0, y0)['Lat'],  #  36.17560399,
                        'x1': x1,  # gp.FRFcoord(x1, y1)['Lon'],  # -75.75004989,
                        'y1': y1,  # gp.FRFcoord(x1, y1)['Lat'],  #  36.19666112,
                        'lambdaX': dx,
                        # grid spacing in x  -  Here is where CMS would hand array of variable grid spacing
                        'lambdaY': dy,  # grid spacing in y
                        'msmoothx': x_smooth,  # smoothing length scale in x
                        'msmoothy': y_smooth,  # smoothing length scale in y
                        'msmootht': 1,  # smoothing length scale in Time
                        'filterName': 'hanning',
                        'nmseitol': 0.75,
                        'grid_coord_check': 'FRF',
                        'grid_filename': '',  # should be none if creating background Grid!  becomes best guess grid
                        'data_coord_check': 'FRF',
                        'xFRF_s': dataX,
                        'yFRF_s': dataY,
                        'Z_s': dataZ,
                        }

                out = DEM_generator(dict)

                plt.figure()
                plt.plot(dataX, dataY, 'r*')

                # check out what I get from scaleCInterp?
                """
                plt.figure()
                plt.subplot(221)
                plt.title('Zi')
                plt.pcolor(out['x_out'], out['y_out'], out['Zi'] )
                plt.colorbar()
                plt.subplot(222)
                plt.title('MSEi')
                plt.pcolor(out['x_out'], out['y_out'], out['MSEi'])
                plt.colorbar()
                plt.subplot(223)
                plt.title('NMSEi')
                plt.pcolor(out['x_out'], out['y_out'], out['NMSEi'])
                plt.colorbar()
                plt.subplot(224)
                plt.title('MSRi')
                plt.pcolor(out['x_out'], out['y_out'], out['MSRi'])
                plt.colorbar()
                plt.tight_layout()
                """

                # read some stuff from this dict like a boss
                Zn = out['Zi']
                MSEn = out['MSEi']
                MSRn = out['MSRi']
                NMSEn = out['NMSEi']
                xFRFn_vec = out['x_out']
                yFRFn_vec = out['y_out']

                # make my the mesh for the new subgrid
                xFRFn, yFRFn = np.meshgrid(xFRFn_vec, yFRFn_vec)

                x1 = np.where(xFRFi_vec == min(xFRFn_vec))[0][0]
                x2 = np.where(xFRFi_vec == max(xFRFn_vec))[0][0]
                y1 = np.where(yFRFi_vec == min(yFRFn_vec))[0][0]
                y2 = np.where(yFRFi_vec == max(yFRFn_vec))[0][0]

                Zi_s = Zi[y1:y2 + 1, x1:x2 + 1]

                """
                #check to see if this is correct?
                xFRFi_s = xFRFi[y1:y2+1, x1:x2+1]
                yFRFi_s = yFRFi[y1:y2+1, x1:x2+1]
                # these variables should be the same as xFRFn_vec and yFRFn_vec
                """

                # get the difference!!!!
                Zdiff = Zn - Zi_s

                # spline time?
                wb = 1 - np.divide(MSEn, targetvar + MSEn)
                newZdiff = bspline_pertgrid(Zdiff, wb, splinebctype=splinebctype, lc=lc, dxm=dxm, dxi=dxi)
                newZn = Zi_s + newZdiff

                # get my new pretty splined grid and see what happens if I dont spline
                newZi = Zi.copy()
                Zi_ns = Zi.copy()
                newZi[y1:y2 + 1, x1:x2 + 1] = newZn
                Zi_ns[y1:y2 + 1, x1:x2 + 1] = Zn

                """
                # with splining - the winner?
                fig_loc = 'C:\Users\dyoung8\Desktop\David Stuff\Projects\CSHORE\Bathy Interpolation\Test Figures'
                fig_name = 'UpdatedBathy.png'
                plt.contourf(xFRFi, yFRFi, newZi)
                plt.axis('equal')
                plt.xlabel('xFRF')
                plt.ylabel('yFRF')
                plt.colorbar()
                plt.savefig(os.path.join(fig_loc, fig_name))
                plt.close()

                # second plot - if I didnt spline
                fig_name = 'UpdatedBathy_ns.png'
                plt.contourf(xFRFi, yFRFi, Zi_ns)
                plt.axis('equal')
                plt.xlabel('xFRF')
                plt.ylabel('yFRF')
                plt.colorbar()
                plt.savefig(os.path.join(fig_loc, fig_name))
                plt.close()
                """

                # update Zi for next iteration
                del Zi
                Zi = newZi

                elevation[tt, :, :] = newZi
                surveyNumber[tt] = np.unique(survNum)[0]
                timeunits = 'seconds since 1970-01-01 00:00:00'
                surveyTime[tt] = nc.date2num(stimeM, timeunits)
                # timeM is the mean time between the first and last time of the survey rounded to the nearest 12 hours
                # this is going to be the date and time of the survey to the closest noon.
                # remember it needs to be in seconds since 1970

            # get position stuff that will be constant for all surveys!!!
            xFRFi_vecN = xFRFi.reshape((1, xFRFi.shape[0] * xFRFi.shape[1]))[0]
            yFRFi_vecN = yFRFi.reshape((1, yFRFi.shape[0] * yFRFi.shape[1]))[0]
            # convert FRF coords to lat/lon
            test = gp.FRF2ncsp(xFRFi_vecN, yFRFi_vecN)
            # go through stateplane to avoid FRFcoords trying to guess the input coordinate systems
            temp = gp.ncsp2LatLon(test['StateplaneE'], test['StateplaneN'])
            lat_vec = temp['lat']
            lon_vec = temp['lon']

            lat = lat_vec.reshape(xFRFi.shape[0], xFRFi.shape[1])
            lon = lon_vec.reshape(xFRFi.shape[0], xFRFi.shape[1])

            xFRF = xFRFi[0, :]
            yFRF = yFRFi[:, 1]
            latitude = lat
            longitude = lon

            # write the nc_file for this month, like a boss, with greatness
            nc_dict = {}
            nc_dict['elevation'] = elevation
            nc_dict['xFRF'] = xFRF
            nc_dict['yFRF'] = yFRF
            nc_dict['latitude'] = latitude
            nc_dict['longitude'] = longitude
            # also want survey number and survey time....
            nc_dict['surveyNumber'] = surveyNumber
            nc_dict['time'] = surveyTime

            nc_name = 'backgroundDEM_' + months[jj] + '.nc'

            # save this location for next time through the loop
            prev_nc_name = nc_name
            prev_nc_loc = nc_loc

            global_yaml = 'C:\Users\dyoung8\PycharmProjects\makebathyinterp\yamls\BATHY\FRFti_global.yml'
            var_yaml = 'C:\Users\dyoung8\PycharmProjects\makebathyinterp\yamls\BATHY\FRFti_var.yml'

            makenc.makenc_tiBATHY(os.path.join(nc_loc, nc_name), nc_dict, globalYaml=global_yaml, varYaml=var_yaml)

            # make my QA/QC plot

            # where is the cross shore array?
            test = nc.Dataset(
                'http://134.164.129.55/thredds/dodsC/FRF/oceanography/waves/8m-array/2017/FRF-ocean_waves_8m-array_201707.nc')
            Lat = test['latitude'][:]
            Lon = test['longitude'][:]
            # convert to FRF
            temp = gp.FRFcoord(Lon, Lat)
            CSarray_X = temp['xFRF']
            CSarray_Y = temp['yFRF']

            fig_loc = 'C:\Users\dyoung8\Desktop\David Stuff\Projects\CSHORE\Bathy Interpolation\Test Figures\QAQCfigs'
            fig_name = 'backgroundDEM_' + yrs_dir + '-' + months[jj] + '.png'

            plt.figure()
            plt.contourf(xFRF, yFRF, elevation[-1, :, :])
            cbar = plt.colorbar()
            cbar.set_label('(m)')
            plt.scatter(dataX, dataY, marker='o', c='k', s=1, alpha=0.25, label='Transects')
            plt.plot(CSarray_X, CSarray_Y, 'rX', label='CS-array')
            plt.xlabel('xFRF (m)')
            plt.ylabel('yFRF (m)')
            plt.legend()
            plt.savefig(os.path.join(fig_loc, fig_name))
            plt.close()


def subgridBounds(surveyDict, gridDict, xMax=1000):

    dataX = surveyDict['dataX']
    dataY = surveyDict['dataX']
    profNum = surveyDict['profNum']

    dx = gridDict['dx']
    dy = gridDict['dy']
    xFRFi_vec = gridDict['xFRFi_vec']
    yFRFi_vec = gridDict['yFRFi_vec']

    # divide my survey up into the survey lines!!!
    profNum_list = np.unique(profNum)
    prof_minX = np.zeros(np.shape(profNum_list))
    prof_maxX = np.zeros(np.shape(profNum_list))
    prof_meanY = np.zeros(np.shape(profNum_list))
    for ss in range(0, len(profNum_list)):
        # pull out all x-values corresponding to this profNum
        Xprof = dataX[np.where(profNum == profNum_list[ss])]
        Yprof = dataY[np.where(profNum == profNum_list[ss])]
        prof_meanY[ss] = np.mean(Yprof)
        prof_minX[ss] = min(Xprof)
        prof_maxX[ss] = max(Xprof)

    # this rounds all these numbers down to the nearest dx
    prof_minX = prof_minX - (prof_minX % dx)
    prof_maxX = prof_maxX - (prof_maxX % dx)
    # note: this only does what you want if the numbers are all POSITIVE!!!!

    # check my y-bounds
    prof_maxY = max(dataY) - (max(dataY) % dy)
    minY = min(dataY)
    # it does check for negative y-values!!!
    if minY > 0:
        prof_minY = minY - (minY % dy)
    else:
        prof_minY = minY - (minY % dy) + dy

    #do not allow any prof_maxX to exceed xMax
    prof_maxX[prof_maxX > xMax] = xMax

    # ok, I am going to force the DEM generator function to always go to the grid specified by these bounds!!
    # if you want to hand it a best guess grid, i.e., 'grid_filename' make SURE it has these bounds!!!!!!
    # or just don't hand it grid_filename....
    x0, y0 = np.median(prof_maxX), prof_maxY
    x1, y1 = np.median(prof_minX), prof_minY
    # currently using the median of the min and max X extends of each profile,
    # and just the min and max of the y-extents of all the profiles.

    # check to see if this is past the bounds of your background DEM.
    # if so, truncate so that it does not exceed.
    if x0 > max(xFRFi_vec):
        x0 = max(xFRFi_vec)
    else:
        pass
    if x1 < min(xFRFi_vec):
        x1 = min(xFRFi_vec)
    else:
        pass
    if y0 > max(yFRFi_vec):
        y0 = max(yFRFi_vec)
    else:
        pass
    if y1 < min(yFRFi_vec):
        y1 = min(yFRFi_vec)
    else:
        pass







