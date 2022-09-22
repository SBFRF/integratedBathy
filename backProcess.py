import matplotlib
matplotlib.use('Agg')
from bathyTopoWrapper import  generateDailyGriddedTopo
import datetime as DT
import numpy as np

start = DT.datetime(2018, 3, 13)
end = DT.datetime(2019, 3, 15)

# note Script works backwards in time

datelist = np.arange(start, end, DT.timedelta(days=1))
outPath = "/thredds_data/integratedBathyProduct/integratedBathyTopo"
for date in datelist[::-1]:
    datein = np.datetime_as_string(date).split('T')[0]
    print(f' Working on Date {datein}')
    gridded_bathy = generateDailyGriddedTopo(datein, outPath, verbose=1,
                                         datacache=None, cross_check_fraction=0.05,
                                         plotdir='./plots', server='chl')#, slack='cmtb')