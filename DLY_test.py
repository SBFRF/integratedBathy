
import os, sys
import netCDF4 as nc
import numpy as np
from sblib import geoprocess as gp
from sblib import sblib as sb
import makenc
from matplotlib import pyplot as plt
from bsplineFunctions import bspline_pertgrid
import datetime as DT
from scaleCinterp_python.DEM_generator import DEM_generator
import MakeUpdatedBathyDEM as mBATHY

"""
test = sb.FRFcoord(861, 200)
print test['Lat']
print test['Lon']
t = 1
"""

# makeUpdatedBathyDEM test script

"""
# wrote this to a netCDF file, so I don't need to do this anymore!!!
# Hard Code Resolution to 5 m (dx & dy) - want whole number FRF coords

# start with background DEM
# **NOTE: will have to edit this to look for more recent bathy first!!!!!**
# hard code my corners....
LLHC = (-50, -200)
URHC = (1305, 5000)  # 1305 is the location of the 11-m awac rounded UP to the nearest 5 m
dx = 5
dy = 5
coord_system = 'FRF'
temp = gT.makeBackgroundBathyCorners(LLHC, URHC, dx, dy, coord_system=coord_system)
Zi = temp['bottomElevation']
xFRFi = temp['xFRF']
yFRFi = temp['yFRF']

# first plot - original bathy
fig_name = 'OrigBathy.png'
fig_loc = 'C:\Users\dyoung8\Desktop\David Stuff\Projects\CSHORE\Bathy Interpolation\Test Figures'
plt.contourf(xFRFi, yFRFi, Zi)
plt.axis('equal')
plt.xlabel('xFRF')
plt.ylabel('yFRF')
plt.colorbar()
plt.savefig(os.path.join(fig_loc, fig_name))
plt.close()

# save this as an nc file on my PC?
Zi_vec = Zi.reshape((1, Zi.shape[0] * Zi.shape[1]))[0]
xFRFi_vec = xFRFi.reshape((1, xFRFi.shape[0] * xFRFi.shape[1]))[0]
yFRFi_vec = yFRFi.reshape((1, yFRFi.shape[0] * yFRFi.shape[1]))[0]

# convert FRF coords to lat/lon
test = gp.FRF2ncsp(xFRFi_vec, yFRFi_vec)
temp = gp.ncsp2LatLon(test['StateplaneE'], test['StateplaneN'])
lat_vec = temp['lat']
lon_vec = temp['lon']

lat = lat_vec.reshape(xFRFi.shape[0], xFRFi.shape[1])
lon = lon_vec.reshape(xFRFi.shape[0], xFRFi.shape[1])

nc_dict = {}
nc_dict['elevation'] = Zi
nc_dict['xFRF'] = xFRFi[0, :]
nc_dict['yFRF'] = yFRFi[:, 1]
nc_dict['latitude'] = lat
nc_dict['longitude'] = lon

nc_loc = 'C:\Users\dyoung8\Desktop\David Stuff\Projects\CSHORE\Bathy Interpolation\TestNCfiles'
nc_name = 'backgroundDEMt0.nc'

global_yaml = 'C:\Users\dyoung8\PycharmProjects\makebathyinterp\yamls\BATHY\FRFt0_global.yml'
var_yaml = 'C:\Users\dyoung8\PycharmProjects\makebathyinterp\yamls\BATHY\FRFt0_var.yml'

makenc.makenc_t0BATHY(os.path.join(nc_loc, nc_name), nc_dict, globalYaml=global_yaml, varYaml=var_yaml)
"""


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
off = 20 # offset for the edge splining!!!!!
lc = 4  # spline smoothing constraint value
dxm = 2  # coarsening of the grid for spline (e.g., 2 means calculate with a dx that is 2x input dx)
# can be tuple if you want to do dx and dy seperately (dxm, dym), otherwise dxm is used for both
dxi = 1  # fining of the grid for spline (e.g., 0.1 means return spline on a grid that is 10x input dx)
# as with dxm, can be a tuple if you want seperate values for dxi and dyi
targetvar = 0.8 # this is the target variance used in the spline function.
wbysmooth = 300
wbxsmooth = 100
# It is used in conjunction with the MSE from splineCinterp to compute the spline weights (wb)
dSTR_s = '2015-10-01T00:00:00Z'
dSTR_e = '2015-11-01T00:00:00Z'

# dir_loc = 'C:\Users\dyoung8\Desktop\David Stuff\Projects\CSHORE\Bathy Interpolation\TestNCfiles_gridded'
dir_loc = '/home/number/Desktop/DavidTestNC'
# dir_loc = 'C:\Users\dyoung8\Desktop\David Stuff\Projects\CSHORE\Bathy Interpolation\TestNCfiles_WA'

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
splineDict['wbxsmooth'] = wbxsmooth
splineDict['wbysmooth'] = wbysmooth


ncml_url = 'https://chlthredds.erdc.dren.mil/thredds/dodsC/frf/projects/bathyduck/data/cbathy_old/cbathy.ncml'
# temp = mBATHY.getGridded(ncml_url=ncml_url, d1=dSTR_s, d2=dSTR_e)

mBATHY.makeUpdatedBATHY_grid(dSTR_s, dSTR_e, dir_loc, ncml_url, scalecDict=scalecDict, splineDict=splineDict, plot=1)
# mBATHY.makeUpdatedBATHY_transects(dSTR_s, dSTR_e, dir_loc, scalecDict=scalecDict, splineDict=splineDict, plot=1)

t = 1



# force the survey to start at the first of the month and end at the last of the month!!!!
dSTR_s = dSTR_s[0:7] + '-01T00:00:00Z'
if dSTR_e[5:7] == '12':
    dSTR_e = str(int(dSTR_e[0:4]) + 1) + '-01' + '-01T00:00:00Z'
else:
    dSTR_e = dSTR_e[0:5] + str(int(dSTR_e[5:7]) + 1).zfill(2) + '-01T00:00:00Z'

d_s = DT.datetime.strptime(dSTR_s, '%Y-%m-%dT%H:%M:%SZ')
d_e = DT.datetime.strptime(dSTR_e, '%Y-%m-%dT%H:%M:%SZ')
d_e = d_e - DT.timedelta(seconds=1)


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
        months = [str(int(month_start) + int(jj)).zfill(2) for jj in range(0, num_months+1)]
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
    filelist = ['http://134.164.129.55/thredds/dodsC/FRF/geomorphology/elevationTransects/survey/surveyTransects.ncml']
    bathy = nc.Dataset(filelist[0])
    # pull down all the times....
    times = nc.num2date(bathy.variables['time'][:], bathy.variables['time'].units, bathy.variables['time'].calendar)
    all_surveys = bathy.variables['surveyNumber'][:]

    for jj in range(0, len(months)):
        # pull out the beginning and end time associated with this month
        d1STR = yrs_dir + '-' + months[jj] + '-01T00:00:00Z'
        d1 = DT.datetime.strptime(d1STR, '%Y-%m-%dT%H:%M:%SZ')
        if int(months[jj]) == 12:
            d2STR = str(int(yrs_dir)+1) + '-' + '01' + '-01T00:00:00Z'
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
        nc_url = 'http://134.164.129.62:8080/thredds/dodsC/CMTB/grids/UpdatedBackgroundDEM/UpdatedBackgroundDEM.ncml'

        nc_b_loc = 'C:\Users\dyoung8\Desktop\David Stuff\Projects\CSHORE\Bathy Interpolation\TestNCfiles'
        nc_b_name = 'backgroundDEMt0.nc'

        # load the background bathy from netDCF file now
        try:
            # look for the .nc file that I just wrote!!!
            old_bathy = nc.Dataset(os.path.join(prev_nc_loc, prev_nc_name))
            ob_times = nc.num2date(old_bathy.variables['time'][:], old_bathy.variables['time'].units, old_bathy.variables['time'].calendar)

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
                ob_times = nc.num2date(old_bathy.variables['time'][:], old_bathy.variables['time'].units, old_bathy.variables['time'].calendar)
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
            prof_minX = prof_minX - (prof_minX%dx)
            prof_maxX = prof_maxX - (prof_maxX%dx)
            # note: this only does what you want if the numbers are all POSITIVE!!!!

            # check my y-bounds
            prof_maxY = max(dataY) - (max(dataY)%dy)
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


            dict = {'x0': x0,    #gp.FRFcoord(x0, y0)['Lon'],  # -75.47218285,
                    'y0': y0,    #gp.FRFcoord(x0, y0)['Lat'],  #  36.17560399,
                    'x1': x1,    #gp.FRFcoord(x1, y1)['Lon'],  # -75.75004989,
                    'y1': y1,    #gp.FRFcoord(x1, y1)['Lat'],  #  36.19666112,
                    'lambdaX': dx,  # grid spacing in x  -  Here is where CMS would hand array of variable grid spacing
                    'lambdaY': dy,  # grid spacing in y
                    'msmoothx': x_smooth,  # smoothing length scale in x
                    'msmoothy': y_smooth,  # smoothing length scale in y
                    'msmootht': 1,   # smoothing length scale in Time
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
            prof_minX[ss] = min(Xprof)
            prof_maxX[ss] = max(Xprof)

            # this rounds all these numbers down to the nearest dx
            prof_minX = prof_minX - (prof_minX%dx)
            prof_maxX = prof_maxX - (prof_maxX%dx)
            # note: this only does what you want if the numbers are all POSITIVE!!!!

            # check my y-bounds
            prof_maxY = max(dataY) - (max(dataY)%dy)
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

            #make my the mesh for the new subgrid
            xFRFn, yFRFn = np.meshgrid(xFRFn_vec, yFRFn_vec)

            x1 = np.where(xFRFi_vec == min(xFRFn_vec))[0][0]
            x2 = np.where(xFRFi_vec == max(xFRFn_vec))[0][0]
            y1 = np.where(yFRFi_vec == min(yFRFn_vec))[0][0]
            y2 = np.where(yFRFi_vec == max(yFRFn_vec))[0][0]

            Zi_s = Zi[y1:y2+1, x1:x2+1]

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
            newZi[y1:y2+1, x1:x2+1] = newZn
            Zi_ns[y1:y2+1, x1:x2+1] = Zn

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
        test = nc.Dataset('http://134.164.129.55/thredds/dodsC/FRF/oceanography/waves/8m-array/2017/FRF-ocean_waves_8m-array_201707.nc')
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

# gT.makeTimeMeanBackgroundBathy_temp(dir_loc, dSTR_s=dSTR_s, dSTR_e=dSTR_e, scalecDict=scalecDict, splineDict=None, plot=1)


t = 1
"""
# make plots for spike
nc_b_loc = 'C:\Users\dyoung8\Desktop\David Stuff\Projects\CSHORE\Bathy Interpolation\TestNCfiles'
nc_b_name = 'backgroundDEMt0_TimeMean.nc'
cs_array_url = 'http://134.164.129.55/thredds/dodsC/FRF/oceanography/waves/8m-array/2017/FRF-ocean_waves_8m-array_201707.nc'
fig_loc = 'C:\Users\dyoung8\Desktop\David Stuff\Projects\CSHORE\Bathy Interpolation\Test Figures\TimeMeanFigs'

# pull the original background DEM
old_bathy = nc.Dataset(os.path.join(nc_b_loc, nc_b_name))
Z = old_bathy.variables['elevation'][:]
xFRF_vec = old_bathy.variables['xFRF'][:]
yFRF_vec = old_bathy.variables['yFRF'][:]

# plot the bathymetry before and after....
# where is the cross shore array?
test = nc.Dataset(cs_array_url)
Lat = test['latitude'][:]
Lon = test['longitude'][:]
# convert to FRF
temp = gp.FRFcoord(Lon, Lat)
CSarray_X = temp['xFRF']
CSarray_Y = temp['yFRF']

rows, cols = np.shape(Z)

# original data
y_ind = int(np.where(yFRF_vec == 900)[0][0])
fig_name = 'backgroundDEM_TM_CStrans_900' + '.png'
plt.figure()
plt.plot(xFRF_vec, Z[y_ind, :])
plt.plot([CSarray_X, CSarray_X], [np.min(Z[y_ind, :]), np.max(Z[y_ind, :])], 'r', label='8m-array')
plt.xlabel('xFRF (m)')
plt.ylabel('yFRF (m)')
plt.legend()
plt.savefig(os.path.join(fig_loc, fig_name))
plt.close()
"""

# force the survey to start at the first of the month and end at the last of the month!!!!
dSTR_s = dSTR_s[0:7] + '-01T00:00:00Z'








"""
# check that thing for Pat
url = 'http://chlthredds.erdc.dren.mil/thredds/dodsC/frf/oceanography/waves/CS02-SBE26/CS02-SBE26.ncml'
test = nc.Dataset(url)

nc_vars = [var for var in test.variables]
time = test.variables['time'][:]

def roundtime(dt=None, roundto=60):
    seconds = (dt - dt.min).seconds
    # // is a floor division, not a comment on following line:
    rounding = (seconds + roundto / 2) // roundto * roundto
    return dt + DT.timedelta(0, rounding - seconds, -dt.microsecond)

alt_time = nc.num2date(test.variables['time'][:], test.variables['time'].units, test.variables['time'].calendar)
for num in range(0, len(alt_time)):
    alt_time[num] = roundtime(alt_time[num], roundto=1 * 60)
"""



"""
# fix FRF coord test
xFRF = 0
yFRF = 0

temp = gp.FRF2ncsp(xFRF, yFRF)
E = temp['StateplaneE']
N = temp['StateplaneN']

temp = gp.ncsp2LatLon(E, N)
lat = temp['lat']
lon = temp['lon']

temp = gp.LatLon2utm(lat, lon)

utmE = temp['utmE']
utmN = temp['utmN']
zl = temp['zl']
zn = temp['zn']

# FRF
p1 = -9000
p2 = 500
test1 = gp.FRFcoord(p1, p2)

# SP
p1 = test1['StateplaneE']
p2 = test1['StateplaneN']
test2 = gp.FRFcoord(p1, p2)

# LL
p1 = test1['Lat']
p2 = test1['Lon']
test3 = gp.FRFcoord(p2, p1)

# UTM
p1 = test1['utmE']
p2 = test1['utmN']
test4 = gp.FRFcoord(p1, p2)
"""








# ok, this is going to be my test script for splining a perturbation tile back into the background DEM

"""
# get a grid
x0 = 0
y0 = 0
x1 = 2000
y1 = 1500
coord_system = 'frf'

dx = 10
dy = 20
LLHC = (x0, y0)
URHC = (x1, y1)
grid = gT.makeBackgroundBathyCorners(LLHC, URHC, dx, dy, coord_system)


# set bottom elevation to zero and replace it with a perturbation (do like a parabola with some noise in it...)
Zi = 0*grid['bottomElevation']
wb = 0*grid['bottomElevation']
splinebctype = None
lc = 2
dxm = [4, 1]
dxi = 1

x = grid['xFRF']
y = grid['yFRF']
rows = np.shape(Zi)[0]
cols = np.shape(Zi)[1]

# make my fake perturbation
x_cen = 0.5*(x1 - x0)
y_cen = 0.5*(y1 - y0)
sig_x = 400
sig_y = 500
sig_n = 0.5
A = 2.0

for ii in range(0, rows):
    for jj in range(0, cols):
        # make my gaussian
        val = np.square(x[ii, jj] - x_cen)*(1/float(2*np.square(sig_x))) + np.square(y[ii, jj] - y_cen)*(1/float(2*np.square(sig_y)))
        # add some noise
        noise = np.random.normal(0, sig_n)
        Zi[ii, jj] = 1 - (A*np.exp(-1*val) + noise)
        wb[ii, jj] = np.random.normal(0.95, 0.1)

wb[wb >= 1] = 0.999
wb[0,:] = 0
wb[-1, :] = 0


Zi_vec = Zi.reshape((1, Zi.shape[0] * Zi.shape[1]))[0]
wb_vec = wb.reshape((1, wb.shape[0] * wb.shape[1]))[0]
x_vec = x.reshape((1, x.shape[0] * x.shape[1]))[0]
y_vec = y.reshape((1, y.shape[0] * y.shape[1]))[0]

columns = ['x', 'y', 'Zi', 'wb']
df = pd.DataFrame(index=range(0, np.size(Zi_vec)), columns=columns)

sloc = 'C:\Users\RDCHLDLY\Desktop\David Stuff\Projects\CSHORE\Bathy Interpolation\Test Figures'
sname = 'test_data.csv'

df['x'] = x_vec
df['y'] = y_vec
df['Zi'] = Zi_vec
df['wb'] = wb_vec

df.to_csv(os.path.join(sloc, sname), index=False)
"""

"""
# saved one to csv so I could compare to matlab, and also so I don't have to make a new one every time.

floc = 'C:\Users\RDCHLDLY\Desktop\David Stuff\Projects\CSHORE\Bathy Interpolation\Test Figures'
fname = 'test_data.csv'

df = pd.read_csv(os.path.join(floc, fname))
df = pd.DataFrame(df)

x_vec = np.asarray(df['x'])
y_vec = np.asarray(df['y'])
Zi_vec = np.asarray(df['Zi'])
wb_vec = np.asarray(df['wb'])

dim1 = 75
dim2 = 200

x = x_vec.reshape(dim1, dim2)
y = y_vec.reshape(dim1, dim2)
Zi = Zi_vec.reshape(dim1, dim2)
wb = wb_vec.reshape(dim1, dim2)


splinebctype = None
lc = 2
dxm = [2, 1]
dxi = 1



#plot her up....
fig_name = 'FakeZi.png'
fig_loc = 'C:\Users\RDCHLDLY\Desktop\David Stuff\Projects\CSHORE\Bathy Interpolation\Test Figures'
plt.contourf(x, y, Zi)
plt.axis('equal')
plt.xlabel('xFRF')
plt.ylabel('yFRF')
plt.colorbar()
plt.savefig(os.path.join(fig_loc, fig_name))
plt.close()


#ok, go time
newZi = bspline_pertgrid(Zi, wb, splinebctype=10, lc=2, dxm=2, dxi=1)

Ny, Nx = np.shape(x)


#plot her up....
fig_name = 'FakeZiNew.png'
fig_loc = 'C:\Users\RDCHLDLY\Desktop\David Stuff\Projects\CSHORE\Bathy Interpolation\Test Figures'
plt.contourf(x, y, newZi)
plt.axis('equal')
plt.xlabel('xFRF')
plt.ylabel('yFRF')
plt.colorbar()
plt.savefig(os.path.join(fig_loc, fig_name))
plt.close()

#plot her up....
fig_name = 'FakeZiNew_Transects.png'
fig_loc = 'C:\Users\RDCHLDLY\Desktop\David Stuff\Projects\CSHORE\Bathy Interpolation\Test Figures'
plt.plot(x[0, :], newZi[int(np.fix(Ny/2)), :], 'r')
plt.plot(x[0, :], Zi[int(np.fix(Ny/2)), :], 'b')
plt.xlabel('xFRF')
plt.ylabel('Elevation')
plt.savefig(os.path.join(fig_loc, fig_name))
plt.close()

"""
















"""
# ok, what we are going to do here is make a .csv file out of the .nc file of the transect
# that Meg used in the practice code she gave us....

floc = 'C:\Users\RDCHLDLY\Desktop\David Stuff\Projects\CSHORE\Bathy Interpolation\MegsPracticeFiles'
fname = 'FRF_20160726_1121_FRF_NAVD88_LARC_GPS_UTC_v20170320.csv'

d_s = DT.datetime.strptime('2016-07-27T12:00:00Z', '%Y-%m-%dT%H:%M:%SZ')
d_e = DT.datetime.strptime('2016-07-28T12:00:00Z', '%Y-%m-%dT%H:%M:%SZ')
frf_Data = getObs(d_s, d_e)
bathy_data = frf_Data.getBathyTransectFromNC()

elev = bathy_data['elevation']
xFRF = bathy_data['xFRF']
yFRF = bathy_data['yFRF']
lat = bathy_data['lat']
lon = bathy_data['lon']
prof_num = bathy_data['profileNumber']
survey_num = bathy_data['surveyNumber']

columns = ['notUsed1', 'notUsed2', 'notUsed3', 'lat', 'lon', 'notUsed4', 'notUsed5', 'xFRF', 'yFRF', 'bottomElevation']
df = pd.DataFrame(index=range(0, np.size(elev)), columns=columns)

df['lat'] = lat
df['lon'] = lon
df['xFRF'] = xFRF
df['yFRF'] = yFRF
df['bottomElevation'] = elev

df.to_csv(os.path.join(floc, fname), index=False)
"""

"""
# this is just to test the makeBackgroundBathy script I wrote
x0 = 901951.6805
y0 = 274093.156
temp = gp.ncsp2LatLon(x0, y0)
origin = (temp['lat'], temp['lon'])

geo_ang = 51.8
dx = 10
dy = 20
ni = 30
nj = 10
coord_system = 'latlon'

test = gT.makeBackgroundBathyAzimuth(origin, geo_ang, dx, dy, ni, nj, coord_system)

# check to see if this actually worked...
fig_name = 'newGrid_Azimuth_LL.png'
fig_loc = 'C:\Users\RDCHLDLY\Desktop\David Stuff\Projects\CSHORE\Bathy Interpolation\Test Figures'
plt.contourf(test['longitude'], test['latitude'], test['bottomElevation'])
plt.axis('equal')
plt.xlabel('lon')
plt.ylabel('lat')
plt.savefig(os.path.join(fig_loc, fig_name))
plt.close()
"""

"""
x0 = 0
y0 = 0
x1 = 2000
y1 = 1500

dx = 10
dy = 20
coord_system = 'latlon'

if coord_system == 'utm':
    temp = gp.FRF2ncsp(x0, y0)
    temp2 = gp.ncsp2utm(temp['StateplaneE'], temp['StateplaneN'])
    LLHC = (temp2['utmE'], temp2['utmN'])

    temp = gp.FRF2ncsp(x1, y1)
    temp2 = gp.ncsp2utm(temp['StateplaneE'], temp['StateplaneN'])
    URHC = (temp2['utmE'], temp2['utmN'])

    fig_name = 'newGrid_Corners_UTM.png'
    label_x = 'utmE'
    label_y = 'utmN'
    x_var = 'utmEasting'
    y_var = 'utmNorthing'

elif coord_system == 'latlon':
    temp = gp.FRF2ncsp(x0, y0)
    temp2 = gp.ncsp2LatLon(temp['StateplaneE'], temp['StateplaneN'])
    LLHC = (temp2['lat'], temp2['lon'])

    temp = gp.FRF2ncsp(x1, y1)
    temp2 = gp.ncsp2LatLon(temp['StateplaneE'], temp['StateplaneN'])
    URHC = (temp2['lat'], temp2['lon'])

    fig_name = 'newGrid_Corners_LL.png'
    label_x = 'longitude'
    label_y = 'latitude'
    x_var = 'longitude'
    y_var = 'latitude'


elif coord_system == 'stateplane':
    temp = gp.FRF2ncsp(x0, y0)
    LLHC = (temp['StateplaneE'], temp['StateplaneN'])

    temp = gp.FRF2ncsp(x1, y1)
    URHC = (temp['StateplaneE'], temp['StateplaneN'])

    fig_name = 'newGrid_Corners_SP.png'
    label_x = 'easting'
    label_y = 'northing'
    x_var = 'easting'
    y_var = 'northing'


elif coord_system == 'FRF':
    LLHC = (x0, y0)
    URHC = (x1, y1)
    fig_name = 'newGrid_Corners_FRF.png'
    label_x = 'xFRF'
    label_y = 'yFRF'
    x_var = 'xFRF'
    y_var = 'yFRF'


test = gT.makeBackgroundBathyCorners(LLHC, URHC, dx, dy, coord_system)

# check to see if this actually worked...
fig_name = fig_name
fig_loc = 'C:\Users\RDCHLDLY\Desktop\David Stuff\Projects\CSHORE\Bathy Interpolation\Test Figures'
plt.contourf(test[x_var], test[y_var], test['bottomElevation'])
plt.axis('equal')
plt.xlabel(label_x)
plt.ylabel(label_y)
plt.savefig(os.path.join(fig_loc, fig_name))
plt.close()
"""

