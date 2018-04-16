from getdatatestbed import getDataFRF
import datetime as DT
import netCDF4 as nc
import pandas
import numpy as np
from matplotlib import pyplot as plt
#############################################
TotalStart = DT.datetime(2015,11,01)
TotalEnd = DT.datetime(2016,1,1)
go= getDataFRF.getObs(TotalStart, TotalEnd)
transectNC= nc.Dataset('http://bones/thredds/dodsC/FRF/geomorphology/elevationTransects/survey/surveyTransects.ncml')

# indentify survey transect indicies
idx = (transectNC['time'][:] > (TotalStart-DT.datetime(1970,01,01)).total_seconds()) & (transectNC['time'][:] < (TotalEnd-DT.datetime(1970,01,01)).total_seconds())
surveyNums = np.unique(transectNC['surveyNumber'][idx])
df = pandas.Series(nc.num2date(transectNC['time'][idx], 'seconds since 1970-01-01'))
surveyDate = df.map(lambda t: t.date()).unique()
for date in surveyDate:
    gdTB = getDataFRF.getDataTestBed(DT.datetime.combine(date, DT.datetime.min.time()), DT.datetime.combine(date, DT.datetime.min.time())+DT.timedelta(1))
    bathy = gdTB.getBathyIntegratedTransect()
    go = getDataFRF.getObs(DT.datetime.combine(date, DT.datetime.min.time()), DT.datetime.combine(date, DT.datetime.min.time())+DT.timedelta(3))
    trans = go.getBathyTransectFromNC()

    profile = trans['elevation'][trans['profileNumber'][:] == 951]
    xs = trans['xFRF'][trans['profileNumber'][:] == 951]
    idxXshore_smo = np.argmin(np.abs(bathy['yFRF'][:] - 945))

    fname='/home/number/Public/paperFigs/XshoreSmooth_{}'.format(date.strftime('%Y%m%d'))
    fname2='/home/number/Public/paperFigs/YshoreSmooth_{}'.format(date.strftime('%Y%m%d'))
    fname3='/home/number/Public/paperFigs/XshoreSmooth_{}_all'.format(date.strftime('%Y%m%d'))

# make the xshore plot
    plt.figure()
    plt.title('xShore Smoothing Comparison')
    plt.plot(xs, profile, '.', label='measured')
    plt.plot(bathy['xFRF'], bathy['elevation'][idxXshore_smo,:],'.', label='smoothed-20m')
    plt.xlim([150,300])
    plt.ylim([-5, -1])
    plt.legend()
    plt.savefig(fname)
    plt.close()
# make the alongshore plot
    plt.figure()
    for val in [100,200,300,400,500,600,700]:
        idxYshore_smo = np.argmin(np.abs(bathy['xFRF'][:] - val))
        plt.plot(bathy['yFRF'], bathy['elevation'][:, idxYshore_smo], '.', label='x={}'.format(val))
    plt.xlim([500, 1200])
    plt.legend()
    plt.savefig(fname2)
    plt.close()

    # make the xshore plot
    plt.figure()
    plt.title('xShore Smoothing Comparison')
    plt.plot(xs, profile, '.', label='measured')
    plt.plot(bathy['xFRF'], bathy['elevation'][idxXshore_smo,:],'.', label='smoothed-20m')
    plt.legend()
    plt.savefig(fname3)
    plt.close()