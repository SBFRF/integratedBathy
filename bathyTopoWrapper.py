import os
import argparse
import numpy as np
import netCDF4 as nc
import datetime as dt
import scipy
from scipy import interpolate
import pickle
import sys
import bathyTopoUtils as dut
from getdatatestbed import getDataFRF
import testbedutils
import testbedutils.py2netCDF as py2netCDF
import testbedutils.geoprocess as geoprocess

# hard coded yaml directory
yaml_dir=os.path.join(os.path.abspath(os.path.dirname(__file__)),'yamls')  #eventually move to os.pardir


def generateDailyGriddedTopo(dSTR_s, dir_loc, method_flag=0, xFRF_lim=(0,1100.), yFRF_lim=(0,1400.), dxFRF=(5.,5.),
                             verbose=0, datacache=None, cross_check_fraction=None):
    """
    :param dSTR_s: string that determines the start date of the times of the surveys you want to use to update the DEM
                    format is  dSTR_s = '2013-01-04'
                    no matter what you put here, it will always round it down to the beginning of the month
    :param dir_loc: place where you want to save the .nc files that get written
                    the function will make the year directories inside of this location on its own.
    :param method_flag: how interpolation is handled
                    0 - linear interpolation
    :param xFRF_lim: acrossshore limits of the output grid in FRF coordinates
    :param yFRF_lim: alongshore limits of the output grid in FRF coordinates
    :param dxFRF_lim: output grid resolution in FRF coordinates

    :return: writes out the .nc files for the gridded bathymetry and topography

    """
    ## setup grid
    xx=np.arange(xFRF_lim[0],xFRF_lim[1],dxFRF[0])
    yy=np.arange(yFRF_lim[0],yFRF_lim[1],dxFRF[1])
    nx,ny=xx.size,yy.size
    XX,YY=np.meshgrid(xx,yy)
    assert XX.shape==(ny,nx)
    assert YY.shape==(ny,nx)
    
    ## setup start and end dates
    # force to start at 00:00:00Z
    dSTR_s0 = dSTR_s[0:10]+'T00:00:00Z'
    d2 = dt.datetime.strptime(dSTR_s0, '%Y-%m-%dT%H:%M:%SZ')
    d1 = d2 - dt.timedelta(days=1)
    if verbose > 0:
        print("Trying to run from {0} to {1}".format(d1.date(),d2.date()))
        
    ## configure the method 
    interp_method = 'linear'
    assert method_flag == 0

    ## configure input and output files
    global_yaml = os.path.join(yaml_dir,'IntegratedBathyTopo_Global.yml')
    var_yaml = os.path.join(yaml_dir,'IntegratedBathyTopo_grid_var.yml')

    # output netcdf
    outfile=os.path.join(dir_loc,'IntegratedBathyTopo-{0}.nc'.format(d1.date()))

    # holds the lidar data sets
    Xlidar,Ylidar,Zlidar=[],[],[]
    
    ## for grabbing from the thredd server
    go = getDataFRF.getObs(d1, d2)
    #gtb = getDataFRF.getDataTestBed(d1,d2)

    ## dune
    if datacache is not None:
        dune_file=os.path.join(datacache,'topo_dune_{0}.p'.format(d1.date()))
        if os.path.isfile(dune_file):
            print('NOTE: reading topo_dune file from cache: {0}'.format(dune_file))
            topo_dune=pickle.load(open(dune_file,"rb"))
        else:
            topo_dune = go.getLidarDEM(lidarLoc='dune')
            try:
                pickle.dump(topo_dune, open( dune_file, "wb" ) )
            except:
                print('unable to pickle dune topo to {0}'.format(dune_file))
    else:
        topo_dune = go.getLidarDEM(lidarLoc='dune')

    if topo_dune is not None:
        if verbose > 0:
            print('nx,ny for topo lidar = ({0},{1})'.format(topo_dune['xFRF'].shape[0],topo_dune['yFRF'].shape[0]))
            print('xFRF range for pier lidar = ({0},{1})'.format(topo_dune['xFRF'].min(),topo_dune['xFRF'].max()))
            print('yFRF range for pier lidar = ({0},{1})'.format(topo_dune['yFRF'].min(),topo_dune['yFRF'].max()))

            print('nt,nx,ny for topo lidar = ({0},{1},{2})'.format(topo_dune['elevation'].shape[0],
                                                                   topo_dune['elevation'].shape[1],topo_dune['elevation'].shape[2]))
            #


        Xdune,Ydune=np.meshgrid(topo_dune['xFRF'],topo_dune['yFRF'])
        points_dune=np.vstack((Xdune.flat[:],Ydune.flat[:])).T

        nt_dune=topo_dune['elevation'].shape[0]

        # add to the list
        Xlidar.append(np.tile(Xdune,(nt_dune,1,1)))
        Ylidar.append(np.tile(Ydune,(nt_dune,1,1)))
        Zlidar.append(topo_dune['elevation'])
    else:
        print('WARNING no dune lidar found!')
    ## pier
    if datacache is not None:
        pier_file=os.path.join(datacache,'topo_pier_{0}.p'.format(d1.date()))
        if os.path.isfile(pier_file):
            print('NOTE: reading topo_dune file from cache: {0}'.format(pier_file))
            topo_pier=pickle.load(open(pier_file,"rb"))
        else:
            topo_pier = go.getLidarDEM(lidarLoc='pier')
            try:
                pickle.dump(topo_pier, open( pier_file, "wb" ) )
            except:
                print('unable to pickle dune topo to {0}'.format(pier_file))
    else:
        topo_pier = go.getLidarDEM(lidarLoc='pier')
    
    if topo_pier is not None:
        Xpier,Ypier=np.meshgrid(topo_pier['xFRF'],topo_pier['yFRF'])
        points_pier=np.vstack((Xpier.flat[:],Ypier.flat[:])).T

        if verbose > 0:
            print('nx,ny for pier lidar = ({0},{1})'.format(topo_pier['xFRF'].shape[0],topo_pier['yFRF'].shape[0]))
            print('xFRF range for pier lidar = ({0},{1})'.format(topo_pier['xFRF'].min(),topo_pier['xFRF'].max()))
            print('yFRF range for pier lidar = ({0},{1})'.format(topo_pier['yFRF'].min(),topo_pier['yFRF'].max()))
        #
        nt_pier=topo_pier['elevation'].shape[0]

        # add to the list
        Xlidar.append(np.tile(Xpier,(nt_pier,1,1)))
        Ylidar.append(np.tile(Ypier,(nt_pier,1,1)))
        Zlidar.append(topo_pier['elevation'])
    else:
        print('WARNING no pier lidar found!')

    assert len(Xlidar)==len(Ylidar)
    assert len(Xlidar)==len(Zlidar)

    if len(Xlidar) > 0:
        ## interpolate topography points
        all_points,all_values,Z_all=dut.combine_and_interpolate_masked_lidar(Xlidar,Ylidar,Zlidar,
                                                                             XX,YY,method=interp_method)

    else:
        print('WARNING no lidar data found!')
        all_points=np.empty(shape=(0,2))
        all_values=np.empty(shape=(0,))
    ## load bathymetry transects from survey
    if datacache is not None:
        bathy_file=os.path.join(datacache,'bathy_data_{0}.p'.format(d1.date()))
        if os.path.isfile(bathy_file):
            print('NOTE: reading topo_dune file from cache: {0}'.format(pier_file))
            bathy_data=pickle.load(open(bathy_file,"rb"))
        else:
            bathy_data = go.getBathyTransectFromNC()
            try:
                pickle.dump(bathy_data, open( bathy_file, "wb" ) )
            except:
                print('unable to pickle dune topo to {0}'.format(bathy_file))
    else:
        bathy_data = go.getBathyTransectFromNC()

    if bathy_data is not None:
        bathy_points=np.vstack((bathy_data['xFRF'],bathy_data['yFRF'])).T
        bathy_values=bathy_data['elevation']
    else:
        print('WARNING bathy points failed to download!')
        bathy_points=np.empty(shape=(0,2))
        bathy_values=np.empty(shape=(0,))
    ## combine bathy transects and topo
    points=np.vstack((all_points,bathy_points))
    values=np.concatenate((all_values,bathy_values))

    if verbose > 0:
        print('Total number of data points is {0}'.format(points.shape[0]))

    Z_interp=scipy.interpolate.griddata(points,values,(XX,YY),method=interp_method,fill_value=np.nan)

    ## extend in the alongshore to the grid boundaries
    Y_start_index,Y_end_index,Z_gridded=dut.extend_alongshore(XX,YY,Z_interp)

    ## Write out the gridded product
    # get position stuff that will be constant for all surveys!!!
    xFRFi_vecN = XX.reshape((1, XX.shape[0] * XX.shape[1]))[0]
    yFRFi_vecN = YY.reshape((1, YY.shape[0] * YY.shape[1]))[0]
    # convert FRF coords to lat/lon
    test = geoprocess.FRF2ncsp(xFRFi_vecN, yFRFi_vecN)
    # go through stateplane to avoid FRFcoords trying to guess the input coordinate systems
    temp = geoprocess.ncsp2LatLon(test['StateplaneE'], test['StateplaneN'])
    lat_vec = temp['lat']
    lon_vec = temp['lon']
    E_vec = test['StateplaneE']
    N_vec = test['StateplaneN']

    latitude = lat_vec.reshape(XX.shape[0], XX.shape[1])
    longitude= lon_vec.reshape(XX.shape[0], XX.shape[1])
    easting  = E_vec.reshape(XX.shape[0], XX.shape[1])
    northing = N_vec.reshape(XX.shape[0], XX.shape[1])

    xFRF = XX[0, :]
    yFRF = YY[:, 1]

    nc_dict={}
    nc_dict['elevation'] = Z_gridded[np.newaxis,:,:]
    nc_dict['xFRF'] = xFRF
    nc_dict['yFRF'] = yFRF
    nc_dict['latitude'] = latitude
    nc_dict['longitude'] = longitude
    nc_dict['northing'] = northing
    nc_dict['easting'] = easting
    nc_dict['updateTime'] = np.ones_like(nc_dict['elevation'])*d1.timestamp()
    nc_dict['time'] = d1.timestamp()
    nc_dict['survey_time'] = np.nanmean(bathy_data['epochtime'])

    py2netCDF.makenc_generic(outfile,global_yaml,var_yaml,nc_dict)
    
    return nc_dict

if __name__=="__main__":
    def check_datestring(s):
        try:
            return dt.datetime.strptime(s, "%Y-%m-%d")
        except ValueError:
            msg = "Not a valid date: '{0}'.".format(s)
            raise argparse.ArgumentTypeError(msg)
    def check_path(s):
        if os.path.isdir(s):
            return s
        msg = "Not a valid directory path: '{0}'.".format(s)
        raise argparse.ArgumentTypeError(msg)

    parser = argparse.ArgumentParser(description="Input for daily bathy-topo product")

    parser.add_argument('day',
                        help="Day for computing bathy-topo product - format YYYY-MM-DD",
                        nargs='?',
                        default=dt.datetime.today().strftime("%Y-%m-%d"),
                        type=check_datestring)
    parser.add_argument('-O','--odir',
                        help="path for writing output bathy netcdf files",
                        default="/thredds_data/integratedBathyProduct/integratedBathyTopo",
                        type=check_path)

    args = parser.parse_args()

    gridded_bathy = generateDailyGriddedTopo(args.day.strftime("%Y-%m-%d"), args.odir, verbose=1,
                                             datacache=None)
