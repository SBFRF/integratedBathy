import numpy as np
import netCDF4 as nc
import matplotlib.pyplot as plt
from sblib import sblib as sb
#  This script is designed to check work from make product manually with loop over which the smoothing scales and
# the lc values were changed and
#
base = "/home/number/Public/2015/CMTB-integratedBathyProduct_survey_201510"
for lc in [(3,8)]:#, (3,12)]: # [(1,4), (2,4) , (2,6), (3,6):
    # smo100 = nc.Dataset(base+'_100_{}.nc'.format(lc))
    # smo40 = nc.Dataset(base+'_40_{}.nc'.format(lc))
    smo20 = nc.Dataset(base + '.nc') #.format(lc))
    for t in range(len(smo20['time'][:])):
        # smo10 = nc.Dataset(base+'_10_{}.nc'.format(lc))
        # Survey = nc.Dataset('http://bones/thredds/dodsC/FRF/geomorphology/DEMs/surveyDEM/FRF_20170127_1130_FRF_NAVD88_LARC_GPS_UTC_v20170320_grid_latlon.nc')
        NowDate = nc.num2date(smo20['time'][t],'seconds since 1970-01-01')
        Tras = nc.Dataset('http://bones/thredds/dodsC/FRF/geomorphology/elevationTransects/survey/surveyTransects.ncml')
        idx = np.in1d(sb.baseRound(Tras['time'][:], 86400), sb.baseRound(smo20['time'][t],86400))

        profile = Tras['elevation'][idx][Tras['profileNumber'][idx] == 951]
        xs  = Tras['xFRF'][idx][Tras['profileNumber'][idx] == 951]
        # idxXshore = np.argmin(np.abs(Survey['yFRF'][:] - 945))
        idxXshore_smo = np.argmin(np.abs(smo20['yFRF'][:] - 945))

        fname = '/home/number/Public/Xshore_{}_{}_{}.png'.format(lc[0],lc[1], NowDate.strftime('%Y-%m-%d'))
        fname1 = '/home/number/Public/all_Xshore_{}_{}_{}.png'.format(lc[0],lc[1], NowDate.strftime('%Y-%m-%d'))
        fname2 = '/home/number/Public/Yshore_{}_{}_{}.png'.format(lc[0],lc[1], NowDate.strftime('%Y-%m-%d'))
        fname3 = '/home/number/Public/pcolor_{}_{}_{}.png'.format(lc[0],lc[1], NowDate.strftime('%Y-%m-%d'))

        plt.figure(figsize=(12,12))
        plt.suptitle('lc values x={} y={}'.format(lc[0], lc[1]))
        plt.title('xShore Smoothing Comparison')
        plt.plot(xs, profile, '.', label='measured')
        plt.plot(smo20['xFRF'], smo20['elevation'][t,idxXshore_smo,:],'.', label='smoothed-20m')
        # plt.plot(smo10['xFRF'], smo10['elevation'][0,idxXshore_smo,:],'.', label='smoothed-10m')
        # plt.plot(smo40['xFRF'], smo40['elevation'][0,idxXshore_smo,:],'.', label='smoothed-40m')
        # plt.plot(smo100['xFRF'], smo100['elevation'][0,idxXshore_smo,:],'.', label='smoothed-100m')
        plt.ylim([-4,-1])
        plt.xlim([150,300])
        plt.legend()
        plt.savefig(fname)
        plt.close()

        plt.figure(figsize=(12,12))
        plt.suptitle('lc values x={} y={}'.format(lc[0], lc[1]))
        plt.title('xShore Smoothing Comparison')
        plt.plot(xs, profile, '.', label='measured')
        plt.plot(smo20['xFRF'], smo20['elevation'][t,idxXshore_smo,:],'.', label='smoothed-20m')
        # plt.ylim([-4,-1])
        plt.legend()
        plt.savefig(fname1)
        plt.close()

        plt.figure(figsize=(12,12))
        plt.suptitle('lc values x={} y={}'.format(lc[0], lc[1]))
        for val in [100,200,300,400,500,600,700]:
            idxYshore_smo = np.argmin(np.abs(smo20['xFRF'][:] - val))
            # plt.plot(smo10['yFRF'], smo10['elevation'][0, :, idxYshore_smo], '.', label='smoothed_10_%s' %val)
            plt.plot(smo20['yFRF'], smo20['elevation'][t, :, idxYshore_smo], '.', label='smoothed_20_%s' %val)
            # plt.plot(smo40['yFRF'], smo40['elevation'][0, :, idxYshore_smo], '.', label='smoothed_40_%s' %val)
            # plt.plot(smo100['yFRF'], smo100['elevation'][0, :, idxYshore_smo], '.', label='smoothed_100_%s' %val)
        plt.xlim([400,1200])
        plt.legend()
        plt.savefig(fname2)
        plt.close()

        plt.figure(figsize=(12,12))
        plt.suptitle('lc values x={} y={}'.format(lc[0], lc[1]))
        ax1= plt.subplot(131)
        plt.title('40 m smoothing in x shore ')
        # a = ax1.pcolormesh(smo40['xFRF'][:], smo40['yFRF'][:], smo40['elevation'][0], vmin=-5)
        a= ax2.pcolormesh(smo20['xFRF'][:], smo20['yFRF'][:], smo20['elevation'][0], vmin=-5)
        plt.colorbar(a)
        ax1.set_ylim([400,1600])
        ax1.set_xlim([0,600])
        ax2 = plt.subplot(132)
        plt.title('20 m smoothing in xshore')
        a= ax2.pcolormesh(smo20['xFRF'][:], smo20['yFRF'][:], smo20['elevation'][t], vmin=-5)
        plt.colorbar(a)
        ax2.set_ylim([400,1600])
        ax2.set_xlim([0,600])
        ax3 = plt.subplot(133)
        plt.title('10 m smoothing in xshore')
        a=ax3.pcolormesh(smo20['xFRF'][:], smo20['yFRF'], smo20['elevation'][-1], vmin=-5)
        plt.colorbar(a)
        ax3.set_ylim([400,1600])
        ax3.set_xlim([0,600])
        plt.savefig(fname3)
        plt.close()