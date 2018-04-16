# ok, this is basically going to be the exact same thing as the scaleCweightsTest function in matlab.
# Then we are going to go through and figure out what is poopy
# David Young, ace surfer, King of the Universe
# 12/4/2017

import numpy as np
from scaleCinterp_python.DEM_generator import DEM_generator, makeWBflow, makeWBflow2D
import os
import pandas as pd
import MakeUpdatedBathyDEM as mBATHY
import netCDF4 as nc
import scipy.io as spio
from matplotlib import pyplot as plt

# name and location of the survey .csv file!!!
floc = 'C:\Users\dyoung8\Desktop\MegTest\MegsPracticeFiles'
fname = 'FRF_20120214_1069_FRF_NAVD88_LARC_GPS_UTC_v20160402'

# read in my .csv file like a boss and turn it into survey stuff
df = pd.read_csv(os.path.join(floc, fname + '.csv'))
del df['notUsed1']
del df['notUsed2']
del df['notUsed3']
del df['notUsed4']
del df['lat']
del df['lon']
dataX = np.array(df['xFRF'])
dataY = np.array(df['yFRF'])
dataZ = np.array(df['bottomElevation'])
profNum = np.array(df['profNum'])

# random variables I will need to be able to run this!!
dx = 5
dy = 5
x_smooth = 40
y_smooth = 100
wbysmooth = 300
wbxsmooth = 100
targetvar = 0.45

# load the background bathy from netDCF file so I don't have to go through THREDDS
nc_b_loc = 'C:\Users\dyoung8\Desktop\David Stuff\Projects\CSHORE\Bathy Interpolation\TestNCfiles_WA'
nc_b_name = 'backgroundDEMt0_TimeMean.nc'
old_bathy = nc.Dataset(os.path.join(nc_b_loc, nc_b_name))
Zi = old_bathy.variables['elevation'][:]
xFRFi_vec = old_bathy.variables['xFRF'][:]
yFRFi_vec = old_bathy.variables['yFRF'][:]

# what are my subgrid bounds?
surveyDict = {}
surveyDict['dataX'] = dataX
surveyDict['dataY'] = dataY
surveyDict['profNum'] = profNum

gridDict = {}
gridDict['dx'] = dx
gridDict['dy'] = dy
gridDict['xFRFi_vec'] = xFRFi_vec
gridDict['yFRFi_vec'] = yFRFi_vec

maxSpace = 249
surveyFilter = True
temp = mBATHY.subgridBounds2(surveyDict, gridDict, maxSpace=maxSpace, surveyFilter=surveyFilter)

x0 = temp['x0']
x1 = temp['x1']
y0 = temp['y0']
y1 = temp['y1']

if surveyFilter is True:
    xS0 = temp['xS0']
    xS1 = temp['xS1']
    yS0 = temp['yS0']
    yS1 = temp['yS1']
    # throw out all points in the survey that are outside of these bounds!!!!
    test1 = np.where(dataX <= xS0, 1, 0)
    test2 = np.where(dataX >= xS1, 1, 0)
    test3 = np.where(dataY <= yS0, 1, 0)
    test4 = np.where(dataY >= yS1, 1, 0)
    test_sum = test1 + test2 + test3 + test4
    dataXn = dataX[test_sum >= 4]
    dataYn = dataY[test_sum >= 4]
    dataZn = dataZ[test_sum >= 4]
    del dataX
    del dataY
    del dataZ
    dataX = dataXn
    dataY = dataYn
    dataZ = dataZn
    del dataXn
    del dataYn
    del dataZn
else:
    pass


max_spacing = temp['max_spacing']
del temp

# if the max spacing is too high, bump up the smoothing!!
y_smooth_u = y_smooth  # reset y_smooth if I changed it during last step
if max_spacing is None:
    pass
elif 2 * max_spacing > y_smooth:
    y_smooth_u = int(dy * round(float(2 * max_spacing) / dy))
else:
    pass


# check my filtered input survey data against the corresponding .mat files
"""
cloc = 'C:\Users\dyoung8\Desktop\MegTest\CompareVars'
cname = 'filteredSurvey' + '.mat'
mat = spio.loadmat(os.path.join(cloc, cname), squeeze_me=True)
dataX_mat = mat['xFRF_s']
dataY_mat = mat['yFRF_s']
dataZ_mat = mat['Z_s']
# looks good so far
"""

# also plot the output like a boss
cloc = 'C:\Users\dyoung8\Desktop\MegTest\CompareVars'
fig_name = 'scaleCoutputBackgroundInput_Python' + '.png'
plt.pcolor(xFRFi_vec, yFRFi_vec, Zi, cmap=plt.cm.jet, vmin=-0, vmax=0.15)
cbar = plt.colorbar()
cbar.set_label('Elevation ($m$)', fontsize=16)
plt.xlabel('Cross-shore - $x$ ($m$)', fontsize=16)
plt.ylabel('Alongshore - $y$ ($m$)', fontsize=16)
plt.legend(prop={'size': 14})
plt.tick_params(axis='both', which='major', labelsize=12)
plt.tick_params(axis='both', which='minor', labelsize=10)
ax1 = plt.gca()
ax1.spines['right'].set_visible(False)
ax1.spines['top'].set_visible(False)
plt.axis('tight')
plt.tight_layout()
plt.savefig(os.path.join(cloc, fig_name))
plt.close()


dict = {'x0': x0,  # gp.FRFcoord(x0, y0)['Lon'],  # -75.47218285,
        'y0': y0,  # gp.FRFcoord(x0, y0)['Lat'],  #  36.17560399,
        'x1': x1,  # gp.FRFcoord(x1, y1)['Lon'],  # -75.75004989,
        'y1': y1,  # gp.FRFcoord(x1, y1)['Lat'],  #  36.19666112,
        'lambdaX': dx,
        # grid spacing in x  -  Here is where CMS would hand array of variable grid spacing
        'lambdaY': dy,  # grid spacing in y
        'msmoothx': x_smooth,  # smoothing length scale in x
        'msmoothy': y_smooth_u,  # smoothing length scale in y
        'msmootht': 1,  # smoothing length scale in Time
        'filterName': 'hanning',
        # 'nmseitol': 0.75, # why did Spicer use 0.75?  Meg uses 0.25
        'nmseitol': 0.25,
        'grid_coord_check': 'FRF',
        'grid_filename': '',  # should be none if creating background Grid!  becomes best guess grid
        'data_coord_check': 'FRF',
        'xFRF_s': dataX,
        'yFRF_s': dataY,
        'Z_s': dataZ,
        'xFRFi_vec': xFRFi_vec,  # x-positions from the full background bathy
        'yFRFi_vec': yFRFi_vec,  # y-positions from the full background bathy
        'Zi': Zi,  # full background bathymetry elevations
        }

out = DEM_generator(dict)

# get my outputs from DEM generator
Zn = out['Zi']
MSEn = out['MSEi']
MSRn = out['MSRi']
NMSEn = out['NMSEi']
xFRFn_vec = out['x_out']
yFRFn_vec = out['y_out']
id_filt = out['id_filt']

# make a grid out of my x and y vectors
xFRFn, yFRFn = np.meshgrid(xFRFn_vec, yFRFn_vec)

# get the elevations from the original bathymetry.
x1 = np.where(xFRFi_vec == min(xFRFn[0, :]))[0][0]
x2 = np.where(xFRFi_vec == max(xFRFn[0, :]))[0][0]
y1 = np.where(yFRFi_vec == min(yFRFn[:, 1]))[0][0]
y2 = np.where(yFRFi_vec == max(yFRFn[:, 1]))[0][0]
Zi_s = Zi[y1:y2 + 1, x1:x2 + 1]


# take a look at my transects
cloc = 'C:\Users\dyoung8\Desktop\MegTest\CompareVars'  # location of these figures

x_loc_check1 = int(100)
x_loc_check2 = int(200)
x_loc_check3 = int(350)
x_check1 = np.where(xFRFn_vec == x_loc_check1)[0][0]
x_check2 = np.where(xFRFn_vec == x_loc_check2)[0][0]
x_check3 = np.where(xFRFn_vec == x_loc_check3)[0][0]

# convert the mse to weights for the plots?
MSEn = np.power(MSEn, 2)
wb = 1 - np.divide(MSEn, targetvar + MSEn)

# that manual weight editing that Nathanial and Meg do...
Nysmooth = int(Lysmooth / dy)
if Nysmooth < 1:
    Nysmooth = 1
else:
    pass

# this is Meg and Nathanial's spline only in the alongshore!
# wb_spline = makeWBflow(yFRFn, Nysmooth, dy)

# this is our 2D spline
wb_dict = {'x_grid': xFRFn,
           'y_grid': yFRFn,
           'ax': wbxsmooth/float(max(xFRFn_vec)),
           'ay': wbysmooth/float(max(yFRFn_vec)),
           }
wb_spline = makeWBflow2D(wb_dict)


wb = np.multiply(wb, wb_spline)

# also plot the output like a boss
fig_name = 'wb_Python' + '.png'
plt.pcolor(xFRFn, yFRFn, wb, cmap=plt.cm.jet, vmin=0, vmax=1)
cbar = plt.colorbar()
cbar.set_label('Weights', fontsize=16)
plt.scatter(dataX, dataY, marker='o', c='k', s=2, alpha=0.35, label='SurveyPoints')
plt.scatter(dataX[~id_filt], dataY[~id_filt], marker='o', c='r', s=2, alpha=0.5, label='BadSurveyPoints')
plt.xlabel('Cross-shore - $x$ ($m$)', fontsize=16)
plt.ylabel('Alongshore - $y$ ($m$)', fontsize=16)
plt.legend(prop={'size': 14})
plt.tick_params(axis='both', which='major', labelsize=12)
plt.tick_params(axis='both', which='minor', labelsize=10)
ax1 = plt.gca()
ax1.spines['right'].set_visible(False)
ax1.spines['top'].set_visible(False)
plt.axis('tight')
plt.tight_layout()
plt.savefig(os.path.join(cloc, fig_name))
plt.close()

# plot X and Y transects and weights
fig_name = 'scaleCtransects_Python' + '.png'
fig = plt.figure(figsize=(8, 9))
ax1 = plt.subplot2grid((3, 1), (0, 0), colspan=1)
ax1.plot(yFRFn[:, x_check1], Zn[:, x_check1], 'r', label='Scale-C')
ax1.plot(yFRFn[:, x_check1], Zi_s[:, x_check1], 'k--', label='Background')
ax4 = ax1.twinx()
ax4.plot(yFRFn[:, x_check1], wb[:, x_check1], 'g--', label='Weights')
ax4.tick_params('y', colors='g')
ax4.set_ylabel('Weights', fontsize=16)
ax4.yaxis.label.set_color('green')
ax1.set_xlabel('Alongshore - $y$ ($m$)', fontsize=16)
ax1.set_ylabel('Elevation ($m$)', fontsize=16)
ax1.set_title('$X=%s$' % (str(x_loc_check1)), fontsize=16)
for tick in ax1.xaxis.get_major_ticks():
    tick.label.set_fontsize(14)
for tick in ax1.yaxis.get_major_ticks():
    tick.label.set_fontsize(14)
ax1.tick_params(labelsize=14)
ax1.legend()
ax1.text(0.10, 0.95, '(a)', horizontalalignment='left', verticalalignment='top', transform=ax1.transAxes,
         fontsize=16)

ax2 = plt.subplot2grid((3, 1), (1, 0), colspan=1)
ax2.plot(yFRFn[:, x_check2], Zn[:, x_check2], 'r', label='Scale-C')
ax2.plot(yFRFn[:, x_check2], Zi_s[:, x_check2], 'k--', label='Background')
ax5 = ax2.twinx()
ax5.plot(yFRFn[:, x_check2], wb[:, x_check2], 'g--', label='Weights')
ax5.tick_params('y', colors='g')
ax5.set_ylabel('Weights', fontsize=16)
ax5.yaxis.label.set_color('green')
ax2.set_xlabel('Alongshore - $y$ ($m$)', fontsize=16)
ax2.set_ylabel('Elevation ($m$)', fontsize=16)
ax2.set_title('$X=%s$' % (str(x_loc_check2)), fontsize=16)
for tick in ax2.xaxis.get_major_ticks():
    tick.label.set_fontsize(14)
for tick in ax2.yaxis.get_major_ticks():
    tick.label.set_fontsize(14)
ax2.tick_params(labelsize=14)
ax2.legend()
ax2.text(0.10, 0.95, '(b)', horizontalalignment='left', verticalalignment='top', transform=ax2.transAxes,
         fontsize=16)

ax3 = plt.subplot2grid((3, 1), (2, 0), colspan=1)
ax3.plot(yFRFn[:, x_check3], Zn[:, x_check3], 'r', label='Scale-C')
ax3.plot(yFRFn[:, x_check3], Zi_s[:, x_check3], 'k--', label='Background')
ax6 = ax3.twinx()
ax6.plot(yFRFn[:, x_check3], wb[:, x_check3], 'g--', label='Weights')
ax6.set_ylabel('Weights', fontsize=16)
ax6.tick_params('y', colors='g')
ax6.yaxis.label.set_color('green')
ax3.set_xlabel('Alongshore - $y$ ($m$)', fontsize=16)
ax3.set_ylabel('Elevation ($m$)', fontsize=16)
ax3.set_title('$X=%s$' % (str(x_loc_check3)), fontsize=16)
for tick in ax3.xaxis.get_major_ticks():
    tick.label.set_fontsize(14)
for tick in ax3.yaxis.get_major_ticks():
    tick.label.set_fontsize(14)
ax3.tick_params(labelsize=14)
ax3.legend()
ax3.text(0.10, 0.95, '(c)', horizontalalignment='left', verticalalignment='top', transform=ax3.transAxes,
         fontsize=16)

fig.subplots_adjust(wspace=0.4, hspace=0.1)
fig.tight_layout(pad=1, h_pad=2.5, w_pad=1, rect=[0.0, 0.0, 1.0, 0.925])
fig.savefig(os.path.join(cloc, fig_name), dpi=300)
plt.close()

t = 1
