"""
Created on 2/19/2016
This script is desinged to create netCDF files using the netCDF4 module from python as
part of the Coastal Model Test Bed (CMTB)

@author: Spicer Bak
@contact: Spicer.Bak@usace.army.mil
"""

import numpy as np
import netCDF4 as nc
import csv
import datetime as DT
import yaml
import time as ttime
try:
    import sblib as sb
except ImportError:
    import sys
    sys.path.append('c:\users\u4hncasb\documents\code_repositories\sblib')
    sys.path.append('/home/spike/repos/sblib')
    sys.path.append('/home/number/repos/sblib')
    import sblib as sb


def readflags(flagfname, header=1):
    """
    This function reads the flag file from the data in to the STWAVE CMTB runs
    :param flagfname: the relative/absolute location of the flags file
    :return: flags of data dtype=dictionary
    """
    times = []
    waveflag = []
    windflag = []
    WLflag = []
    curflag = []
    allflags = []
    with open(flagfname, 'rb') as f:
        reader = csv.reader(f)  # opening file
        for row in reader:  # iteratin

            # g over the open file
            if len(row) > 1 and row[0] != 'Date':
                waveflag.append(int(row[2]))  # appending wave data flag
                windflag.append(int(row[3]))  # appending Wind data flag
                WLflag.append(int(row[4]))  # appending Water Level Flag data
                curflag.append(int(row[5]))  # appending ocean Currents flag data
                times.append(DT.datetime.strptime(row[0]+row[1], '%Y-%m-%d%H%M'))
                allflags.append([int(row[2]), int(row[3]), int(row[4]), int(row[5])])
    # creating array of flags
    allflags = np.array(allflags)

    # putting data into a dictionary
    flags = {'time': times,
             'windflag': windflag,
             'waveflag': waveflag,
             'WLflag': WLflag,
             'curflag': curflag,
             'allflags': allflags
             }
    return flags

def checkflags(flags, ):
    """
    This function is here to ensure that the flags are of equal length as the time
    :param flags:
    :return:
    """

def import_template_file(yaml_location):
    """
    This function loads a yaml file and returns the attributes in dictionary
    written by: ASA
    :param yaml_location: yaml file location
    :return:
    """
    # load the template
    f = open(yaml_location)
    # use safe_load instead load
    vars_dict = yaml.safe_load(f)
    f.close()
    return vars_dict

def init_nc_file(nc_filename, attributes):
    """
    Create the netCDF file and write the Global Attributes
    written by ASA
    """

    ncfile = nc.Dataset(nc_filename, 'w', clobber=True)

    # Write some Global Attributes
    for key, value in attributes.iteritems():
        # Skip and empty fields or this will bomb
        #print 'key %s; value %s' %( key, value)
        if value is not None:
            setattr(ncfile, key, value)
        #if key == 'geospatial_lat_min':
        #    lat = float(value)
        #if key == 'geospatial_lon_min':
        #    lon = float(value)

    dt_today = ttime.strftime("%Y-%m-%d")
    ncfile.date_created = dt_today
    ncfile.date_issued = dt_today

    # ID is a unique identifier for the file
    # ncfile.id = os.path.split(nc_filename)[1].split('.nc')[0]

    # ncfile.qcstage = '3'
    # ncfile.qcstage_possible_values = '0, 1, 2, 3'
    # ncfile.qcstage_value_meanings = 'None, Processed_R/T, Post-Processed, Final'

    #return ncfile, lat, lon
    return ncfile

def write_data_to_nc(ncfile, template_vars, data_dict, write_vars='_variables'):
    '''
    This function actually writes the variables and the variable attributes to
    the netCDF file
    ncfile is an open fid

    written by: ASA
    in the yaml, the "[variable]:" needs to be in the data dictionary,
     the output netcdf variable will take the name "name:"
    '''

    # Keep track of any errors found
    num_errors = 0
    error_str = ''

    # write some more global attributes if present
    if '_attributes' in template_vars:
        for var in template_vars['_attributes']:
            if var in data_dict:
                setattr(ncfile, var, data_dict[var])

    # List all possible variable attributes in the template
    possible_var_attr = ['standard_name', 'long_name', 'coordinates', 'flag_values', 'flag_meanings',
                         'positive', 'valid_min', 'valid_max', 'calendar', 'description', 'cf_role', 'missing_value']

    # Write variables to file
    accept_vars = template_vars['_variables']

    for var in accept_vars:
        if var in data_dict:
            try:
                if "fill_value" in template_vars[var]:
                    new_var = ncfile.createVariable(template_vars[var]["name"],
                                                    template_vars[var]["data_type"],
                                                    template_vars[var]["dim"],
                                                    fill_value=template_vars[var]["fill_value"])
                else:
                    new_var = ncfile.createVariable(template_vars[var]["name"],
                                                    template_vars[var]["data_type"],
                                                    template_vars[var]["dim"])

                new_var.units = template_vars[var]["units"]

                # Write the attributes
                for attr in possible_var_attr:
                    if attr in template_vars[var]:
                        if template_vars[var][attr] == 'NaN':
                            setattr(new_var, attr, np.nan)
                        else:
                            setattr(new_var, attr, template_vars[var][attr])
                # Write the short_name attribute as the variable name
                if 'short_name' in template_vars[var]:
                    new_var.short_name = template_vars[var]["short_name"]
                else:
                    new_var.short_name = template_vars[var]["name"]
                # _____________________________________________________________________________________
                # Write the data (1D, 2D, or 3D)
                #______________________________________________________________________________________
                if var == "station_name":
                    station_id = data_dict[var]
                    data = np.empty((1,), 'S'+repr(len(station_id)))
                    data[0] = station_id
                    new_var[:] = nc.stringtochar(data)
                elif len(template_vars[var]["dim"]) == 0:
                    try:
                        new_var[:] = data_dict[var]
                    except Exception, e:
                        new_var = data_dict[var]

                elif len(template_vars[var]["dim"]) == 1:
                    # catch some possible errors for frequency and direction arrays
                    if template_vars[var]["data_type"] == 'str':
                        for i, c in enumerate(template_vars[var]["data_type"]):
                            new_var[i] = data_dict[var][i]
                    else:
                        try:
                            new_var[:] = data_dict[var]
                        except IndexError:
                            try:
                                new_var[:] = data_dict[var][0][0]
                            except Exception, e:
                                raise e

                elif len(template_vars[var]["dim"]) == 2:
                    # create an empty 2d data set of the correct sizes
                    try:
                        # handles row vs col data, rather than transposing the array just figure out which it is
                        length = data_dict[var][0].shape[1]
                        if data_dict[var][0].shape[0] > length:
                            length = data_dict[var][0].shape[0]

                        x = np.empty([data_dict[var].shape[0], length], dtype=np.float64)
                        for i in range(data_dict[var].shape[0]):
                            # squeeze the 3d array in to 2d as dimension is not needed
                            x[i] = np.squeeze(data_dict[var][i])
                        new_var[:, :] = x
                    except Exception, e:
                        # if the tuple fails must be right...right?
                        new_var[:] = data_dict[var]

                elif len(template_vars[var]["dim"]) == 3:
                    # create an empty 3d data set of the correct sizes
                    # this portion was modified by Spicer Bak
                    assert data_dict[var].shape == new_var.shape, 'The data must have the Same Dimensions  (missing time?)'
                    x = np.empty([data_dict[var].shape[0], data_dict[var].shape[1], data_dict[var].shape[2]], np.float64)
                    for i in range(data_dict[var].shape[0]):
                        x[i] = data_dict[var][i]
                    new_var[:, :, :] = x[:, :, :]

            except Exception, e:
                num_errors += 1
                error_str += 'ERROR WRITING VARIABLE: ' + var + ' - ' + str(e) + '\n'
                print error_str

    return num_errors, error_str

def makenc_field(data_lib, globalyaml_fname, flagfname, ofname, griddata, var_yaml_fname):
    """
    This is a function that takes wave nest dictionary and Tp_nest dictionnary and creates the high resolution
    near shore field data from the Coastal Model Test Bed


    :param data_lib:  data lib is a library of data with keys the same name as associated variables to be written in the
                    netCDF file to be created
    :param globalyaml_fname:
    :param flagfname:
    :param ofname: the file name to be created
    :param griddata:
    :param var_yaml_fname:
    :return:
    """

    # import global atts
    globalatts = import_template_file(globalyaml_fname)
    # import variable data and meta
    var_atts = import_template_file(var_yaml_fname)
    # import flag data
    flags = readflags(flagfname)['allflags']
    data_lib['flags'] = flags
    globalatts['grid_dx'] = griddata['dx']
    globalatts['grid_dy'] = griddata['dy']
    globalatts['n_cell_y'] = griddata['NJ']
    globalatts['n_cell_x'] = griddata['NI']
    # making bathymetry the length of time so it can be concatnated
    if data_lib['waveHsField'].shape[1] != data_lib['bathymetry'].shape[1]:
        data_lib['waveHsField']=data_lib['waveHsField'][:,:data_lib['bathymetry'].shape[1],:]
    data_lib['bathymetry'] = np.full_like(data_lib['waveHsField'], data_lib['bathymetry'], dtype=np.float32 )
    if 'bathymetryDate' in data_lib:
        data_lib['bathymetryDate'] = np.full_like(data_lib['time'], data_lib['bathymetryDate'], dtype=np.float32 )


    #data_lib['bathymetry'] =
    fid = init_nc_file(ofname, globalatts)  # initialize and write inital globals

    #### create dimensions
    tdim = fid.createDimension('time', np.shape(data_lib['waveHsField'])[0])
    xdim = fid.createDimension('X_shore', np.shape(data_lib['waveHsField'])[1])
    ydim = fid.createDimension('Y_shore', np.shape(data_lib['waveHsField'])[2])
    inputtypes = fid.createDimension('in_type', np.shape(flags)[1]) # there are 4 input dtaa types for flags
    statnamelen = fid.createDimension('station_name_length', len(data_lib['station_name']))
    #if 'bathymetryDate' in data_lib:
    #    bathyDate_length = fid.createDimension('bathyDate_length', np.shape(data_lib['bathymetry'])[0])

    # bathydate = fid.createDimension('bathyDate_length', np.size(data_lib['bathymetryDate']))

    # write data to the nc file
    write_data_to_nc(fid, var_atts, data_lib)
    # close file
    fid.close()

def makenc_FRFTransect(bathyDict, ofname, globalYaml, varYaml):
    """
    This function makes netCDF files from csv Transect data library created with sblib.load_FRF_transect

    :
    :return:
    """
    globalAtts = import_template_file(globalYaml)  # loading global meta data attributes from  yaml
    varAtts = import_template_file(varYaml)  # loading variables to write and associated meta data

    # initializing output ncfile
    fid =init_nc_file(ofname, globalAtts)

    # creating dimensions of data
    tdim = fid.createDimension('time', np.shape(bathyDict['time'])[0])

    # write data to the ncfile
    write_data_to_nc(fid, varAtts, bathyDict)
    # close file
    fid.close()

def makenc_FRFGrid(gridDict, ofname, globalYaml, varYaml):
    """
    This is a function that makes netCDF files from the FRF Natural neighbor tool created by
    Spicer Bak using the pyngl library. the transect dictionary is created using the natural
    neighbor tool in FRF_natneighbor.py

    :param tranDict:
    :param ofname:
    :param globalYaml:
    :param varYaml:
    :return: netCDF file with gridded data in it
    """
    globalAtts = import_template_file(globalYaml)
    varAtts = import_template_file(varYaml)

    # create netcdf file
    fid = init_nc_file(ofname, globalAtts)

    # creating dimensions of data
    xShore = fid.createDimension('xShore', np.shape(gridDict['zgrid'])[0])
    yShore = fid.createDimension('yShore', np.shape(gridDict['zgrid'])[1])
    time = fid.createDimension('time', np.size(gridDict['time']))

    # creating lat/lon and state plane coords
    #xgrid, ygrid = np.meshgrid(gridDict['xgrid'], gridDict['ygrid'])
    xx, yy = np.meshgrid(gridDict['xgrid'], gridDict['ygrid'])
    latGrid = np.zeros(np.shape(yy))
    lonGrid = np.zeros(np.shape(xx))
    statePlN = np.zeros(np.shape(yy))
    statePlE = np.zeros(np.shape(xx))
    for iy in range(0, np.size(gridDict['zgrid'], axis=1)):
        for ix in range(0, np.size(gridDict['zgrid'], axis=0)):
            coords = sb.FRFcoord(xx[iy, ix], yy[iy, ix])#, grid[iy, ix]))
            statePlE[iy, ix] = coords['StateplaneE']
            statePlN[iy, ix] = coords['StateplaneN']
            latGrid[iy, ix] = coords['Lat']
            lonGrid[iy, ix] = coords['Lon']
            assert xx[iy, ix] == coords['FRF_X']
            assert yy[iy, ix] == coords['FRF_Y']

    # put these data into the dictionary that matches the yaml
    gridDict['Latitude'] = latGrid[:, 0]
    gridDict['Longitude'] = lonGrid[0, :]
    gridDict['Easting'] = statePlE[:, 0]
    gridDict['Northing'] = statePlN[0, :]
    gridDict['FRF_Xshore'] = gridDict.pop('xgrid')
    gridDict['FRF_Yshore'] = gridDict.pop('ygrid')
    # addding 3rd dimension for time
    a=gridDict.pop('zgrid').T
    gridDict['Elevation'] = np.full([1, a.shape[0], a.shape[1]], fill_value=[a], dtype=np.float32)
    # write data to file
    write_data_to_nc(fid, varAtts, gridDict)
    # close file
    fid.close()

def makenc_Station(stat_data, globalyaml_fname, flagfname, ofname, griddata, stat_yaml_fname):
    """

    This function will make netCDF files from the station output data from the
    Coastal Model Test Bed of STWAVE for the STATion files

    :param stat_data:
    :param globalyaml_fname:
    :param flagfname:
    :param ofname:
    :param griddata:
    :param stat_yaml_fname:

    :return: a nc file with station data in it
    """
     # import global yaml data
    globalatts = import_template_file(globalyaml_fname)
    # import variable data and meta
    stat_var_atts = import_template_file(stat_yaml_fname)
    # import flag data
    flags = readflags(flagfname)['allflags']
    stat_data['flags'] = flags # this is a library of flags
    globalatts['grid_dx'] = griddata['dx']
    globalatts['grid_dy'] = griddata['dy']
    globalatts['n_cell_y'] = griddata['NJ']
    globalatts['n_cell_x'] = griddata['NI']
    fid = init_nc_file(ofname, globalatts)  # initialize and write inital globals

    #### create dimensions
    tdim = fid.createDimension('time', np.shape(stat_data['time'])[0])  # None = size of the dimension, what does this gain me if i know it
    inputtypes = fid.createDimension('input_types_length', np.shape(flags)[1]) # there are 4 input dtaa types for flags
    statnamelen = fid.createDimension('station_name_length', len(stat_data['station_name']))
    northing = fid.createDimension('Northing', 1L)
    easting = fid.createDimension('Easting', 1L )
    Lon = fid.createDimension('Lon', np.size(stat_data['Lon']))    
    Lat = fid.createDimension('Lat', np.size(stat_data['Lat']))
    dirbin = fid.createDimension('waveDirectionBins', np.size(stat_data['waveDirectionBins']))
    frqbin = fid.createDimension('waveFrequency', np.size(stat_data['waveFrequency']))
    
    #
    # convert to Lat/lon here

    # write data to the nc file
    write_data_to_nc(fid, stat_var_atts, stat_data)
    # close file
    fid.close()

def convert_FRFgrid(gridFname, ofname, globalYaml, varYaml, plotFlag=False):
    """
    This function will convert the FRF gridded text product into a NetCDF file

    :param gridFname: input FRF gridded product
    :param ofname:  output netcdf filename
    :param globalYaml: a yaml file containing global meta data
    :param varYaml:  a yaml file containing variable meta data
    :param plotFlag: true or false for creation of QA plots
    :return: None
    """
    # Defining rigid parameters

    # defining the bounds of the FRF gridded product
    gridYmax = 1100  # maximum FRF Y distance for netCDF file
    gridYmin = -100  # minimum FRF Y distance for netCDF file
    gridXmax = 950  # maximum FRF X distance for netCDF file
    gridXmin = 50  # minimum FRF xdistance for netCDF file
    fill_value= '-999.0'
    # main body
    # load Grid from file
    xyz = sb.importFRFgrid(gridFname)

    # make dictionary in right form
    dx = np.median(np.diff(xyz['x']))
    dy = np.max(np.diff(xyz['y']))
    xgrid = np.unique(xyz['x'])
    ygrid = np.unique(xyz['y'])

    # putting the loaded grid into a 2D array
    zgrid = np.zeros((len(xgrid), len(ygrid)))
    rc = 0
    for i in range(np.size(ygrid, axis=0 )):
        for j in range(np.size(xgrid, axis=0)):
            zgrid[j, i] = xyz['z'][rc]
            rc += 1
    if plotFlag == True:
        from matplotlib import pyplot as plt
        plt.pcolor(xgrid, ygrid, zgrid.T)
        plt.colorbar()
        plt.title('FRF GRID %s' % ofname[:-3].split('/')[-1])
        plt.savefig(ofname[:-4] + '_RawGridTxt.png')
        plt.close()
    # aking labels in FRF coords for
    ncXcoord = np.linspace(gridXmin, gridXmax, num=(gridXmax - gridXmin) / dx + 1, endpoint=True)
    ncYcoord = np.linspace(gridYmin, gridYmax, num=(gridYmax - gridYmin) / dy + 1, endpoint=True)
    frame = np.full((np.shape(ncXcoord)[0], np.shape(ncYcoord)[0]), fill_value=fill_value)

    # find the overlap locations between grids
    xOverlap = np.intersect1d(xgrid, ncXcoord)
    yOverlap = np.intersect1d(ygrid, ncYcoord)
    assert len(yOverlap) >= 3, 'The overlap between grid nodes and netCDF grid nodes is short'
    lastX = np.argwhere(ncXcoord == xOverlap[-1])[0][0]
    firstX = np.argwhere(ncXcoord == xOverlap[0])[0][0]
    lastY = np.argwhere(ncYcoord == yOverlap[-1])[0][0]
    firstY = np.argwhere(ncYcoord == yOverlap[0])[0][0]

    # fill the frame grid with the loaded data
    frame[firstX:lastX+1, firstY:lastY+1] = zgrid

    # run data check
    assert set(xOverlap).issubset(ncXcoord), 'The FRF X values in your function do not fit into the netCDF format, please rectify'
    assert set(yOverlap).issubset(ncYcoord), 'The FRF Y values in your function do not fit into the netCDF format, please rectify'

    # putting the data into a dictioary to make a netCDF file
    fields = gridFname.split('_')
    for fld in fields:
        if len(fld) == 8:
            dte = fld  # finding the date in the file name
            break
    gridDict = {'zgrid': frame,
                'xgrid': ncXcoord,
                'ygrid': ncYcoord,
                'time': nc.date2num(DT.datetime(int(dte[:4]), int(dte[4:6]),
                                                       int(dte[6:])), 'seconds since 1970-01-01')}

    # making the netCDF file from the gridded data
    makenc_FRFGrid(gridDict, ofname, globalYaml, varYaml)

def  makenc_todaysBathyCMTB(gridDict, ofname, globalYaml, varYaml):
    """

    :param gridDict:
    :param ofname:
    :param globalYaml:
    :param varYaml:
    :return:
    """
    globalAtts = import_template_file(globalYaml)
    varAtts = import_template_file(varYaml)

    # create netcdf file
    fid = init_nc_file(ofname, globalAtts)

    # creating dimensions of data
    xFRF = fid.createDimension('xFRF', gridDict['xFRF'].shape[0])
    yFRF = fid.createDimension('yFRF', gridDict['yFRF'].shape[0])
    time = fid.createDimension('time', np.size(gridDict['time']))
    # write data to file
    write_data_to_nc(fid, varAtts, gridDict)
    # close file
    fid.close()

def makenc_CSHORErun(ofname, dataDict, globalYaml, varYaml):
    """
       This is a function that makes netCDF files from CSHORE model runs created by
       David Young using all the stuff Spicer Bak used. You have to build dataDict from the different dictionaries
       output by cshore_io.load_CSHORE_results().  YOU DONT HAVE TO HAND IT LAT LON THOUGH!!!

       :param dataDict:
                keys:
                ['time']
                ['X_shore']
                ['aveE'] - depth averaged eastward current!
                ['stdE'] - standard deviation of eastward current
                ['aveN'] - same as above but northward current
                ['stdN'] - same as above but northward
                ['waveHs']
                ['waveMeanDirection']
                ['waterLevel']
                ['stdWaterLevel']
                ['setup']
                ['runup2perc']
                ['runupMean']
                ['qbx'] - cross-shore bed load sediment transport rate
                ['qsx'] - cross-shore suspended sediment transport rate
                ['qby'] - same as above but alongshore
                ['qsx'] - same as above but alongshore
                ['probabilitySuspension'] - probability that sediment will be suspended at particular node
                ['probabilityMovement'] - probability that sediment will move
                ['suspendedSedVolume']
                ['bottomElevation']
                ['surveyNumber']
                ['profileNumber']
                ['bathymetryDate']
                ['yFRF']
       :param ofname:
       :param globalYaml:
       :param varYaml:

       :return: netCDF file with CSHORE model results in it
       """
    globalAtts = import_template_file(globalYaml)
    varAtts = import_template_file(varYaml)

    # create netcdf file
    fid = init_nc_file(ofname, globalAtts)

    # note: you have to hand this the yFRF coordinates of the BC gage if you want to get lat/lon..
    lx = np.size(dataDict['X_shore'], axis=0)
    lat = np.zeros(lx)
    lon = np.zeros(lx)
    for ii in range(0, lx):
        coords = sb.FRFcoord(dataDict['X_shore'][ii], dataDict['yFRF'])
        lat[ii] = coords['Lat']
        lon[ii] = coords['Lon']
    dataDict['latitude'] = lat
    dataDict['longitude'] = lon

    # ok, we are HARD CODING the dimensions to ALWAYS be at the 8m ARRAY (xFRF = 914.44 rounded DOWN to 914)
    # we will just fill in the missing values with nans as required
    array8m_loc = 914

    # creating dimensions of data
    new_s = np.shape(range(0, array8m_loc + 1))[0]
    new_t = np.shape(dataDict['waveHs'])[0]
    xShore = fid.createDimension('X_shore', new_s)
    time = fid.createDimension('time', new_t)

    # check to see if the grid I am importing is smaller than my netCDF grid
    if np.shape(range(0, array8m_loc + 1))[0] == np.shape(dataDict['X_shore']):
        # the model grid is the same as the netCDF grid, so do nothing
        pass
    else:
        dataDict_n = {'X_shore': np.flipud(np.array(range(0, array8m_loc + 1)) + 0.0),
                      'time': dataDict['time'],
                      'aveE': np.full((new_t, new_s), fill_value=np.nan),
                      'stdE': np.full((new_t, new_s), fill_value=np.nan),
                      'aveN': np.full((new_t, new_s), fill_value=np.nan),
                      'stdN': np.full((new_t, new_s), fill_value=np.nan),
                      'waveHs': np.full((new_t, new_s), fill_value=np.nan),
                      'waveMeanDirection': np.full((new_t, new_s), fill_value=np.nan),
                      'waterLevel': np.full((new_t, new_s), fill_value=np.nan),
                      'stdWaterLevel': np.full((new_t, new_s), fill_value=np.nan),
                      'setup': np.full((new_t, new_s), fill_value=np.nan),
                      'runup2perc': dataDict['runup2perc'],
                      'runupMean': dataDict['runupMean'],
                      'qbx': np.full((new_t, new_s), fill_value=np.nan),
                      'qsx': np.full((new_t, new_s), fill_value=np.nan),
                      'qby': np.full((new_t, new_s), fill_value=np.nan),
                      'qsy': np.full((new_t, new_s), fill_value=np.nan),
                      'probabilitySuspension': np.full((new_t, new_s), fill_value=np.nan),
                      'probabilityMovement': np.full((new_t, new_s), fill_value=np.nan),
                      'suspendedSedVolume': np.full((new_t, new_s), fill_value=np.nan),
                      'bottomElevation': np.full((new_t, new_s), fill_value=np.nan),
                      'latitude': np.full((new_s), fill_value=np.nan),
                      'longitude': np.full((new_s), fill_value=np.nan),
                      'surveyNumber': dataDict['surveyNumber'],
                      'profileNumber': dataDict['profileNumber'],
                      'bathymetryDate': dataDict['bathymetryDate'],
                      'yFRF': dataDict['yFRF'], }

        if 'FIXED' in ofname:
            dataDict_n['bottomElevation'] = np.full((new_s), fill_value=np.nan)
        elif 'MOBILE' in ofname:
            dataDict_n['bottomElevation'] = np.full((new_t, new_s), fill_value=np.nan)
        else:
            print 'You need to modify makenc_CSHORErun in makenc.py to accept your new version name!'

        # find index of first point on dataDict grid
        min_x = min(dataDict['X_shore'])
        ind_minx = int(np.argwhere(dataDict_n['X_shore'] == min_x))
        max_x = max(dataDict['X_shore'])
        ind_maxx = int(np.argwhere(dataDict_n['X_shore'] == max_x))

        for ii in range(0, int(new_t)):
            dataDict_n['aveE'][ii][ind_maxx:ind_minx + 1] = dataDict['aveE'][ii]
            dataDict_n['stdE'][ii][ind_maxx:ind_minx + 1] = dataDict['stdE'][ii]
            dataDict_n['aveN'][ii][ind_maxx:ind_minx + 1] = dataDict['aveN'][ii]
            dataDict_n['stdN'][ii][ind_maxx:ind_minx + 1] = dataDict['stdN'][ii]
            dataDict_n['waveHs'][ii][ind_maxx:ind_minx + 1] = dataDict['waveHs'][ii]
            dataDict_n['waveMeanDirection'][ii][ind_maxx:ind_minx + 1] = dataDict['waveMeanDirection'][ii]
            dataDict_n['waterLevel'][ii][ind_maxx:ind_minx + 1] = dataDict['waterLevel'][ii]
            dataDict_n['stdWaterLevel'][ii][ind_maxx:ind_minx + 1] = dataDict['stdWaterLevel'][ii]
            dataDict_n['setup'][ii][ind_maxx:ind_minx + 1] = dataDict['setup'][ii]
            dataDict_n['qbx'][ii][ind_maxx:ind_minx + 1] = dataDict['qbx'][ii]
            dataDict_n['qsx'][ii][ind_maxx:ind_minx + 1] = dataDict['qsx'][ii]
            dataDict_n['qby'][ii][ind_maxx:ind_minx + 1] = dataDict['qby'][ii]
            dataDict_n['qsy'][ii][ind_maxx:ind_minx + 1] = dataDict['qsy'][ii]
            dataDict_n['probabilitySuspension'][ii][ind_maxx:ind_minx + 1] = dataDict['probabilitySuspension'][ii]
            dataDict_n['probabilityMovement'][ii][ind_maxx:ind_minx + 1] = dataDict['probabilityMovement'][ii]
            dataDict_n['suspendedSedVolume'][ii][ind_maxx:ind_minx + 1] = dataDict['suspendedSedVolume'][ii]
            dataDict_n['latitude'][ind_maxx:ind_minx + 1] = dataDict['latitude'][ii]
            dataDict_n['longitude'][ind_maxx:ind_minx + 1] = dataDict['longitude'][ii]

        if 'FIXED' in ofname:
            dataDict_n['bottomElevation'][ind_maxx:ind_minx + 1] = dataDict['bottomElevation']
        elif 'MOBILE' in ofname:
            for ii in range(0, int(new_t)):
                dataDict_n['bottomElevation'][ii][ind_maxx:ind_minx + 1] = dataDict['bottomElevation'][ii]
        else:
            print 'You need to modify makenc_CSHORErun in makenc.py to accept your new version name!'

        # check to see if I screwed up!
        assert set(dataDict.keys()) == set(dataDict_n.keys()), 'You are missing dictionary keys in the new dictionary!'
        # replace the dictionary with the new dictionary
        del dataDict
        dataDict = dataDict_n
        del dataDict_n

    # write data to file
    write_data_to_nc(fid, varAtts, dataDict)
    # close file
    fid.close()

def makenc_initBATHY(ofname, dataDict, globalYaml, varYaml):
    """
    :param ofname: this is the name of the ncfile you are building
    :param dataDict: keys must include...
        latitude - decimal degrees
        longitude - decimal degrees
        bottomElevation - in m NAVD88 I think
        utmNorthing - this is utm in meters (not feet)
        utmEasting - this is utm in meters (not feet)

        note: each of these must be 2d arrays of the SAME SHAPE!!!!

    :param globalYaml:
    :param varYaml:
    :return: writes out the ncfile
    """

    globalAtts = import_template_file(globalYaml)
    varAtts = import_template_file(varYaml)

    # create netcdf file
    fid = init_nc_file(ofname, globalAtts)

    # creating dimensions of data
    ni = fid.createDimension('ni', dataDict['utmEasting'].shape[1])
    nj = fid.createDimension('nj', dataDict['utmEasting'].shape[0])

    # write data to file
    write_data_to_nc(fid, varAtts, dataDict)
    # close file
    fid.close()

def makenc_t0BATHY(ofname, dataDict, globalYaml, varYaml):
    """
    # this is the script that builds the t0 netCDF file from the initial Bathy DEM (intBathy)

    :param ofname: this is the name of the ncfile you are building
    :param dataDict: keys must include...
        latitude - decimal degrees
        longitude - decimal degrees
        bottomElevation - in m NAVD88 I think
        note: each of these must be 2d arrays of the SAME SHAPE!!!!

        xFRF - in m
        yFRF - in m
        note: these are 1D arrays that contain the coordinates in each dimension

    :param globalYaml:
    :param varYaml:
    :return: writes out the ncfile
    """

    globalAtts = import_template_file(globalYaml)
    varAtts = import_template_file(varYaml)

    # create netcdf file
    fid = init_nc_file(ofname, globalAtts)

    # creating dimensions of data
    xFRF = fid.createDimension('xFRF', dataDict['xFRF'].shape[0])
    yFRF = fid.createDimension('yFRF', dataDict['yFRF'].shape[0])

    # write data to file
    write_data_to_nc(fid, varAtts, dataDict)
    # close file
    fid.close()

def makenc_tiBATHY(ofname, dataDict, globalYaml, varYaml):
    """
    # this is the script that builds the monthly ti netCDF file by incorporating the new survey data into the most recent Bathy DEM

    :param ofname: this is the name of the ncfile you are building
    :param dataDict: keys must include...
        latitude - decimal degrees
        longitude - decimal degrees
        note: these must be 2d arrays of the SAME SHAPE!!!!

        xFRF - in m
        yFRF - in m
        note: these are 1D arrays that contain the coordinates in each dimension

        elevation - in m
        note: this is an ns X yFRF X xFRF array

        surveyNumber - this is a 1D array of length ns (number of surveys in the month)
        surveyTime - this is a 1D array of length ns
        y_smooth - this is a 1D array of length ns (the cross-shore smoothing scale used)

        updateTime - this is the most recent update to this cell at this time-step for every point in the grid.
                    this will ALWAYS be a 3D masked array.  values that are masked have NEVER been updated
                    (i.e., still working off the time mean background bathymetry.)

    :param globalYaml:
    :param varYaml:
    :return: writes out the ncfile
    """

    globalAtts = import_template_file(globalYaml)
    varAtts = import_template_file(varYaml)

    # create netcdf file
    fid = init_nc_file(ofname, globalAtts)

    # creating dimensions of data
    time = fid.createDimension('time', dataDict['time'].shape[0])
    xFRF = fid.createDimension('xFRF', dataDict['xFRF'].shape[0])
    yFRF = fid.createDimension('yFRF', dataDict['yFRF'].shape[0])

    # Note: you have to pass this a non-masked array that already has fill values where you want them to be.
    tempUpdateTime = dataDict['updateTime']
    tempUpdateTime[np.ma.getmask(tempUpdateTime)] = -999
    del dataDict['updateTime']
    dataDict['updateTime'] = np.array(tempUpdateTime)
    # remove the mask?

    # write data to file
    write_data_to_nc(fid, varAtts, dataDict)
    # close file
    fid.close()