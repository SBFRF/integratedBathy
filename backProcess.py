from bathyTopoWrapper import  generateDailyGriddedTopo
import datetime as DT
import numpy as np

start = DT.datetime.today()
end = DT.datetime(2021, 7, 1) #DT.datetime(2021, 6, 1)
# note Script works backwards in time

datelist = np.arange(end, start, DT.timedelta(days=1))
outPath = "/thredds_data/integratedBathyProduct/integratedBathyTopo"
for date in datelist[::-1]:
    datein = np.datetime_as_string(date).split('T')[0]
    print(f' Working on Date {datein}')
    gridded_bathy = generateDailyGriddedTopo(datein, outPath, verbose=1,
                                         datacache=None, cross_check_fraction=0.05,
                                         plotdir='./plots', server='FRF', slack='cmtb')