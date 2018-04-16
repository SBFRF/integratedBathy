import netCDF4 as nc
import matplotlib.pyplot as plt
import numpy as np

url = 'http://bones/thredds/dodsC/cmtb/waveModels/STWAVE/CB/Local-Field/Local-Field.ncml'
ncfile = nc.Dataset(url)

timelist = np.arange(0, len(ncfile['time'][:]),  len(ncfile['time'][:])/30)
for t in timelist:
    fname2 = 'Archive/figures/CheckPreviousBathyIntegration/AlongShore_smoothing_{}.png'.format(nc.num2date(ncfile['time'][t],'seconds since 1970-01-01').strftime('%Y%m%dT%H%M%SZ'))
    plt.figure(figsize=(12,12))
    plt.suptitle('AlongShore Smoothing Check')
    for val in [100,200,300,400,500,600,700]:
        idxYshore_smo = np.argmin(np.abs(ncfile['xFRF'][:] - val))
        plt.plot(ncfile['yFRF'], ncfile['bathymetry'][t, :, idxYshore_smo], '.', label='x={}'.format(val))

    plt.xlim([-10,1200])
    plt.legend()
    plt.savefig(fname2)
    plt.close()