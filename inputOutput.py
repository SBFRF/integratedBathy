import numpy as np
import glob
import datetime as DT
import warnings

from docutils.nodes import line


class stwaveIO():

    def __init__(self, fpath):
        """
        must be in directory of the output files to be read in for processing
        :rtype: object
        """
        dpath = fpath[fpath.rfind('/')+1:]  # datestring of folder name
        # if len(fpath) > 0 and fpath.split('.')[-1] == '.dep':  # specific instance when dep is denoted
        #     self.depfnamenes = fpath
        if len(fpath) > 0:

            self.statfname = glob.glob(fpath + '/%s.station.out' % dpath)
            self.statfname_nest = glob.glob(fpath + '/%snested.station.out' % dpath)
            self.Tpfname = glob.glob(fpath + '/%s.Tp.out' % dpath)  # peak period out
            self.Tpfname_nest = glob.glob(fpath + '/%snested.Tp.out' % dpath)
           # self.logfname = glob.glob(fpath + '/%s.log.out' % dpath)  # log file out
            #self.logfname_nest = glob.glob(fpath + '/%snested.log.out' % dpath)
            self.wavefname = glob.glob(fpath + '/%s.wave.out' % dpath)  # wave output file out name
            self.wavefname_nest = glob.glob(fpath + '/%snested.wave.out' % dpath)
            self.obsefname = glob.glob(fpath + '/%s.obse.out' % dpath)
            self.obsefname_nest = glob.glob(fpath + '/%snested.obse.out' % dpath)
            self.depfname = glob.glob(fpath + '/%s.dep' % dpath)
            self.depfname_nest = glob.glob(fpath + '/%snested.dep' % dpath)
            self.inpfname = glob.glob(fpath + '/%s.inp' % dpath)
            self.inpfname_nest = glob.glob(fpath + '/%snested.inp' %dpath)
            if len(self.obsefname) < 1:
                self.obsefname = glob.glob(fpath + '/*.obse.out')
            if len(self.statfname) > 1:
                print 'Parent Depth file: %s' % self.depfname
                print 'Nested Depth File: %s' % self.depfname_nest
                print 'Parent Station File: %s' % self.statfname
                print 'Nested Station File:%s' % (self.statfname_nest)
                print 'Parent Peak Period out: %s' % self.Tpfname
                print "Nested Peak Period out: %s" % self.Tpfname_nest
                #
                # print '\nParent Log file: %s' % self.logfname
                # print 'Nested Log file: %s\n' % self.logfname_nest

                print 'Parent Wave output file: %s' % self.wavefname
                print 'Nested Wave output file: %s' % self.wavefname_nest

                print 'Parent Observation output file: %s' % self.obsefname
                print 'Nested Observation output file: %s' % self.obsefname_nest

                print 'Parent Inp file: %s' % self.inpfname
                print 'Nested Inp file: %s' % self.inpfname_nest

            if len(self.statfname_nest) > 1:
                self.nest = True  # there is nested data

        else:
            self.nest = None  # there is NOT nested data


    def parseSTWAVEsim(self,stwaveSimFile):
        """
        This function will parse out the STWAVE sim file, currently only
        will output grid parameters:
            ni, nj, dx, dy, azi, x0, y0
        :param stwaveSimFile:
        :return:
        """
        import numpy as np
        f = open(stwaveSimFile, 'r')
        stLines = f.readlines()
        outDict = {}
        f.close()
        assert stLines[0] == '# STWAVE_SIM_FILE\n', 'please check file type'
        for line in stLines:
            # if the line is the beggining of a section
            # if line[0] == '&':
            #     if line == '&spatial_grid_parms':
            if line.split('=')[0].strip() == 'x0':
                outDict['x0'] = np.float(line.split('=')[-1].strip().split(',')[0])
            elif line.split('=')[0].strip() == 'y0':
                outDict['y0'] = np.float(line.split('=')[-1].strip().split(',')[0])
            elif line.split('=')[0].strip() == 'azimuth':
                outDict['azi'] = np.float(line.split('=')[-1].strip().split(',')[0])
            elif line.split('=')[0].strip() == 'dx':
                outDict['dx'] = np.float(line.split('=')[-1].strip().split(',')[0])
            elif line.split('=')[0].strip() == 'dy':
                outDict['dy'] = np.float(line.split('=')[-1].strip().split(',')[0])
            elif line.split('=')[0].strip() == 'n_cell_i':
                outDict['ni'] = int(line.split('=')[-1].strip().split(',')[0])
            elif line.split('=')[0].strip() == 'n_cell_j':
                outDict['nj'] = int(line.split('=')[-1].strip().split(',')[0])

        return outDict

    def DEPload(self, nested=0, Headerlines=15):
        """
        this function loads the depth (bathy) files that were input for the STWAVE model
        :param nested:
        :param Headerlines:
        :return:
        """
        if nested == 0:  # if the sim file is not nested
            f = open(self.depfname[0], 'r')
            depfile = f.readlines()
        elif nested == 1:  # if the file is nested
            f = open(self.depfname_nest[0], 'r')
            depfile = f.readlines()
        else:

            raise TypeError, 'self.Nested option must be 1 or 0'
        f.close()
        # finding the comma at the end of the line
        eol = self.findeol(depfile)
        # find the number of header lines before data
        while depfile[Headerlines].split()[0] != 'IDD':
            Headerlines += 1
        assert depfile[Headerlines].split()[
                   0] == 'IDD', 'The headerlines are wrong for this file, info will not be parsed correctly'

        # begin parsing header data data
        for ii in range(0, Headerlines):
            if depfile[ii].split()[0] == 'NumRecs':
                numrecs = int(depfile[ii].split()[-1][:-1])
            elif depfile[ii].split()[0] == 'NI':
                NI = int(depfile[ii].split()[-1][:-1])
            elif depfile[ii].split()[0] == 'NJ':
                NJ = int(depfile[ii].split()[-1][:-1])
            elif depfile[ii].split()[0] == 'DX':
                dx = float(depfile[ii].split()[-1][:-1])
            elif depfile[ii].split()[0] == 'DY':
                dy = float(depfile[ii].split()[-1][:-1])
            elif depfile[ii].split()[0] == 'Azimuth':
                azi = float(depfile[ii].split()[-1][:-1])
            elif depfile[ii].split()[0] == 'GridName':
                gridname = depfile[ii].split()[-1][:-1]

        # initializing the array to keep the bathy data
        dep_data = np.zeros((numrecs, NI, NJ))

        timedata = []
        for ii in range(Headerlines, len(depfile) - Headerlines,
                        NI * NJ + 1):  # interval numer of cells plus extra line for date
            timestring = self.depfname_nest[0][self.depfname_nest[0].rfind('/') + 1:self.depfname_nest[0].rfind('.')]
            if nested == 1:
                timestring = timestring[:-6]
            try:  # try a short string for time string
                timedata.append(DT.datetime(int(timestring[0:4]), int(timestring[5:7]), int(timestring[8:10])))
            except ValueError:
                try:
                    timedata.append(DT.datetime(int(timestring[0:4]), int(timestring[5:7]), int(timestring[8:10]),
                                            int(timestring[11:13]), int(timestring[14:16])))
                except ValueError:
                    timedata.append('NaN')
            # try:
            #     # timestring = str(int(depfile[ii].split()[1]))#[3:eol + 1]))
            #     timestring = self.depfname_nest[0][self.depfname_nest[0].rfind('/') + 1:-4]
            # except ValueError:
            #     print 'nested dep file still has no IDD label'
            #     if
            #         if nested == 1:
            #             timestring = ''.join(self.depfname_nest[0].split('/')[1].split('-'))+'0000'
            #         elif nested == 0:
            #             timestring = ''.join(self.depfname[0].split('/')[1].split('-'))+'0000'

            for pp in range(1, NI * NJ + 1):  # every cross shore column plust ext
                # spatial data sets are read from [0,max(NJ)] to [max(NI),max(NJ)] then to [1,max(NJ-1)] to [max(NI),max(NJ)-1]
                # At the FRF pier that's the south east corner to the south west corner, then progressing north
                dep_data[(ii - Headerlines) / (NI * NJ + 1), -((pp - 1) - (((pp - 1) / NI) * (NI))), -pp / NI] = float(
                    depfile[ii + pp][:-2])
        # packaging data
        dep_packet = {'bathy': dep_data,
                      'time': np.array(timedata),
                      'dx': dx,
                      'dy': dy,
                      'NI': NI,
                      'NJ': NJ,
                      'gridFname': gridname,
                      'meta': 'depth file from STWAVE simulation, positive values point down, units meters, grid origin is in .sim file'
                      }
        # applying specific decision based dictionary parameters
        try:
            nestedgrid = self.find_nested_from_inp(nested=nested)  # finding the old
            dep_packet['nested_grid_name'] = nestedgrid
        except AttributeError:
            dep_packet['nested_grid_name'] = 'Nofilename.dep'
        try:
            dep_packet['azimuth'] = azi
        except UnboundLocalError, NameError:
            azi = -999
            warnings.warn('Dep file does not have azimuth written in the meta data')
            dep_packet['azimuth'] = azi
        return dep_packet

    def find_nested_from_inp(self, nested):
        """
        This function loads the .inp file used for the input for the inerpolation
        scheme, it reads in the parent and the nested file that was interpolated into
        the 'background' data set.
        nested = 0 is regional grid
        nested = 1 is the nested grid (uses the word nested before the extension to identify)
        """
        #flist = glob.glob(path+'/*.inp')
        if nested == 0: # and len(flist) == 2:
            fname = self.inpfname[0]
        elif nested == 1: # and len(flist) ==2:
            fname = self.inpfname_nest[0]
        else:
            print "<<ERROR>> nested value must be 1/0"
        # now the proper file name for the inp file is identified
        fid=open(fname,'r')
        inps = fid.readlines()  # data are in here
        fid.close()  # closing file
        for line in range(0,len(inps)):
            try:
                a=inps[line].index('scatter_filename =')
                start = inps[line].index('"') # beggingin of file name
                end = inps[line].index('.txt"')  # end of filename
                rfname=inps[line][start+1:end+4] #20is the length of scatter fileanme
                break
            except ValueError:
                a=0
        return rfname

    def makencml(self, fname):
        '''
        This function makes a ncml file named fname
        :input:
        fname: string

        :output:
        ncml, placed in appropriate folder on thredds
        '''
        f = open(fname, 'w')
        f.write('<netcdf xmlns="http://www.unidata.ucar.edu/namespaces/netcdf/ncml-2.2">\n')
        f.write('<aggregation dimName="time" type="joinExisting" recheckEvery="5 min" >\n')
        f.write('    <scan location="./" suffix=".nc" />\n')
        f.write('  </aggregation>\n')
        f.write('</netcdf>\n')
        f.close()

    def waveload(self, nested=0, headerlines=25):
        """
        this function returns data from the .wave.out file output from STWAVE.  The mean period (Tm),
        mean direction (Dm), and significant wave height are output

        :param nested:
        :param headerlines: the number of headerlines to begin file
        :return:
        """
        if nested == 0:
            f = open(self.wavefname[0], 'r')
            wavefile = f.readlines()
            f.close()
        elif nested == 1:
            f = open(self.wavefname_nest[0], 'r')
            wavefile = f.readlines()
            f.close()
        else:
            print "nested option must be 1 or 0"
            raise
        eol = self.findeol(wavefile)
        # datatype = int(wavefile[3][13:-2])
        numrecs = int(wavefile[4][wavefile[4].index('=') + 1:eol])
        # numflds = int(wavefile[5][12:eol])
        NI = int(wavefile[6][wavefile[6].index('=') + 1:eol])  # number of cells in x direction
        NJ = int(wavefile[7][wavefile[7].index('=') + 1:eol])  # number of cells in J direction
        dx = float(wavefile[8][wavefile[8].index('=') + 1:eol])  # X-dir cell size
        dy = float(wavefile[9][wavefile[9].index('=') + 1:eol])  # y-dir cell size
        azi = float(wavefile[10][wavefile[10].index('=') + 1:eol])  # grid azimuth
        gridname = str(wavefile[11][wavefile[11].index('=') + 1:eol])
        Hs_name = str(wavefile[15][wavefile[15].index('=') + 1:eol - 1])
        Tm_name = str(wavefile[16][wavefile[16].index('=') + 1:eol - 1])
        Dm_name = str(wavefile[17][wavefile[17].index('=') + 1:eol - 1])
        Hs_units = str(wavefile[18][wavefile[18].index('=') + 1:eol - 1])
        Tm_units = str(wavefile[19][wavefile[19].index('=') + 1:eol - 1])
        Dm_units = str(wavefile[20][wavefile[20].index('=') + 1:eol - 1])

        # initalizing vars
        Tm_data = np.zeros((numrecs, NI, NJ))
        Hs_data = np.zeros((numrecs, NI, NJ))
        dm_data = np.zeros((numrecs, NI, NJ))
        timedata = []
        # looping through to each IDD number
        for ii in range(headerlines, len(wavefile) - headerlines,
                        NI * NJ + 1):  # interval numer of cells plus extra line for date
            timestring = str(int(wavefile[ii][3:-2]))
            timedata.append(DT.datetime(int(timestring[0:4]), int(timestring[4:6]),
                                        int(timestring[6:8]), int(timestring[8:10]), int(timestring[10:12])))
            for pp in range(1, NI * NJ + 1):
                linepp = wavefile[ii + pp].split()
                dm_data[(ii - headerlines) / (NI * NJ + 1), -((pp - 1) - (((pp - 1) / NI) * (NI))), -pp / NI] = \
                    float(linepp[2])  # [52:])  # used to be -25:-1
                Tm_data[(ii - headerlines) / (NI * NJ + 1), -((pp - 1) - (((pp - 1) / NI) * (NI))), -pp / NI] = \
                    float(linepp[1])  # [27:49])  # used to be (27:52) 25:-25
                Hs_data[(ii - headerlines) / (NI * NJ + 1), -((pp - 1) - (((pp - 1) / NI) * (NI))), -pp / NI] = \
                    float(linepp[0])  # [0:26])  # used to be 1:25
                # print pp, ii
        wavepacket = {'time': np.array(timedata),
                      'dx': dx,
                      'dy': dy,
                      'NI': NI,
                      'NJ': NJ,
                      'azimuth': azi,
                      'gridname': gridname,
                      'Tm_units': Tm_units,
                      'Hs_units': Hs_units,
                      'Dm_units': Dm_units,
                      'Tm_field': Tm_data,
                      'Hs_field': Hs_data,
                      'Dm_field': dm_data,
                      'meta': 'directions in degrees STWangle convention, height is in meters, period is in s'
                      }
        return wavepacket

    def statload(self, nested=0, headerlines=37):
        """
        This function loads the Station output file. and parses it into useful data

        :param headerlines: the lines to skip, set to default (37) with 10 outputs
                            where the data actually starts
        :param nested:  nested==0: uses non-nested or parent station file
                        nested==1: uses nested simulation station file

        """

        # assert len(self.statfname[0])!=0,'The Station file Wasn''t Found for this model\n Check Directory path'
        if nested == 0:
            f = open(self.statfname[0], 'r')
            statf = f.readlines()
        elif nested == 1:
            f = open(self.statfname_nest[0], 'r')
            statf = f.readlines()
        else:
            print 'Nested option must be 1 or 0'
            raise
        f.close()

        eol = self.findeol(statf)
        # set up parameters
        numrecs = int(statf[4][statf[4].index('=') + 1:eol])  # number of records
        numflds = int(statf[5][statf[5].index('=') + 1:eol])  # number of collection locations
        numStats = int(statf[6][statf[6].index('=') + 1:eol])  # number or Stations in Record, based off NI
        # below fields should be a count of numflds
        snapid = np.zeros((numrecs, numStats))
        Easting = np.zeros((numrecs, numStats))
        Northing = np.zeros((numrecs, numStats))
        Hs = np.zeros((numrecs, numStats))
        Tm = np.zeros((numrecs, numStats))
        Dm = np.zeros((numrecs, numStats))
        Tp = np.zeros((numrecs, numStats))
        Umag = np.zeros((numrecs, numStats))
        Udir = np.zeros((numrecs, numStats))
        eta = np.zeros((numrecs, numStats))  # water elevation
        time = []
        for ii in range(headerlines, np.size(statf), numStats):  # running top to bottom in the
            timestring = statf[ii].split()[0]
            time.append(
                DT.datetime(int(timestring[0:4]), int(timestring[4:6]), int(timestring[6:8]), int(timestring[8:10]),
                            int(timestring[10:12])))

            for zz in range(0, numStats):
                # put each gague in columns (zz) time going down
                snapid[int((ii - headerlines) / numStats), zz] = (int(statf[ii + zz].split()[0]))
                Easting[int((ii - headerlines) / numStats), zz] = (statf[ii + zz].split()[1])
                Northing[int((ii - headerlines) / numStats), zz] = (statf[ii + zz].split()[2])
                Hs[int((ii - headerlines) / numStats), zz] = (statf[ii + zz].split()[3])
                Tm[int((ii - headerlines) / numStats), zz] = (statf[ii + zz].split()[4])
                Dm[int((ii - headerlines) / numStats), zz] = (statf[ii + zz].split()[5])
                Tp[int((ii - headerlines) / numStats), zz] = (statf[ii + zz].split()[6])
                Umag[int((ii - headerlines) / numStats), zz] = (statf[ii + zz].split()[7])
                Udir[int((ii - headerlines) / numStats), zz] = (statf[ii + zz].split()[8])
                eta[int((ii - headerlines) / numStats), zz] = (statf[ii + zz].split()[9])

        meta = 'Check The STWAVE RUN code for gauge organization '

        stationpacket = {'time': np.array(time),
                         'snapid': snapid,
                         'Easting': Easting,
                         'Northing': Northing,
                         'Hs': Hs,
                         'Tm': Tm,
                         'WaveDm': Dm,
                         'Tp': Tp,
                         'Umag': Umag,
                         'Udir': Udir,
                         'WL': eta,
                         'meta': meta
                         }
        return stationpacket

    def write_frfGridFileText(self, FRFgrid, ofname, LatLon=False, plot=True):
        """
        This function will write the cBathy grid (output from get data get_cBathyFromNc) to
        a text file grid of the same standard and format to that of the standard grid
        :param FRFgrid: this is the output from go.get_cBathyFromNc
        :return:
        cBathy_grid['ym']
        cBathy_grid['xm']
        cbathy_grid['depth']  - These are expectd to have values under water of negative a standard that is set
        by the FRF grid latlon.txt file conventions

        """
        datestring = ofname[ofname.rfind('/') + 1:-4]
        version_prefix = ofname.split('_')[0]
        ym = FRFgrid['ym']
        xm = FRFgrid['xm']
        depth = FRFgrid['depth']
        if plot == True:
            plt.contourf(ym, xm, depth[0, :, :].T)
            plt.colorbar()
            plt.ylim(0, 800)
            plt.xlim((np.max(xm), np.min(xm)))
            plt.savefig('%s_STWAVE_data/' % version_prefix + datestring + '/figures/QA' +
                        datestring + 'preFRFcoord.png')
            plt.close()
        if LatLon == True:
            LatGrid, LonGrid = np.meshgrid(xm, ym)
            grid = depth[0, :, :]  # depths here are negitive values

            f = open(ofname, 'w')
            for iy in range(0, np.size(grid, axis=0)):
                for ix in range(0, np.size(grid, axis=1)):
                    pack = sb.FRFcoord(xm[ix], ym[iy])
                    # print 'x coord:%d  Y coord: %d ' %(cBathy_grid['xm'][ix], cBathy_grid['ym'][iy])
                    # print 'Lon: %f  Lat: %f' %(pack['Lon'], pack['Lat'])
                    LatGrid[iy, ix] = pack['Lat']
                    LonGrid[iy, ix] = -pack['Lon']

                    # for ix in range(0, np.size(grid, axis=0)):
                    #     for iy in range(0, np.size(grid, axis=1)):
                    # these indicies were flipped for fitting purposes - the last one had backwards shape
                    f.write("%f, %f, %f\n" % (
                    -pack['Lon'], pack['Lat'], grid[iy, ix].T))  # (yy[ix, iy], xx[ix, iy], grid[ix, iy]))
            f.close()

            if plot == True:
                xcoord = LatGrid[0, :]
                ycoord = LonGrid[:, 0]

                plt.contourf(ycoord, xcoord, grid.T)
                plt.colorbar()
                # plt.ylim((np.min(ycoord), np.max(ycoord)))
                # plt.xlim((np.min(xcoord), np.max(xcoord)))
                plt.savefig('%s_STWAVE_data/' % version_prefix + datestring + '/figures/QA' + datestring +
                            'postFRFgridWrite.png')
                plt.close()

        else:
            xcoord = xm
            ycoord = ym

            grid = depth.T  # made neg to fit into work flow
            # transpose of the mesh grids produced below
            yy, xx = np.meshgrid(xcoord, ycoord)  # putting x and y coords into grid
            f = open(ofname, 'w')
            for iy in range(0, np.size(grid, axis=0)):
                for ix in range(0, np.size(grid, axis=1)):
                    # these indicies were flipped for fitting purposes - the last one had backwards shape
                    f.write("%f, %f, %f\n" % (xx[ix, iy], yy[ix, iy], grid[iy, ix]))
            f.close()
        # chopping off the prefix of the grid name to conform with parent code
        outname = datestring + '.txt'
        return outname

    def write_inp(self, pathbase, nestfile, basefile, nested=0, dataout='MTBdata/', dtrun=1, hirez_interp=False):
        """
        This function writes the .inp file for use with interpolation fortran executable
            :param pathbase: this is where the inp file is to be put
            :param nestfile: the name of the updated bathy to be integrated to the background grid
                this must be a grid file, with the number of records placed at the top
            :param basefile: the name of the background grid for the updated bathy to be
            :param nested: whether the word nested should be appended to the files: inp sim dep
            :param dataout: this is the output location of the inp files.
            :param hirez_interp: this should be 1 if the grid resolution is higher than that of the background grid dataset
            update this function takes 'yesterday's bathy and uses that as a 'background' to
            interpolate to
        """
        smooth = 100  # number of grid cells to do linear smoothing over smooths over background portion of grid
        # only used in high Rez

        # file inputs
        if nested == 1:
            # simfile = pathbase + 'nested.sim'  # this was old version
            otptfile = pathbase + 'nested.dep'
        elif nested == 0:
            # simfile = pathbase + '.sim'
            otptfile = pathbase + '.dep'
        simfile = basefile[:-4] + '.sim'  # new version with static sim for interp

        sct_tpt = '-1.0'
        wrtsct = ".true."
        newidd = pathbase[0:4] + pathbase[5:7] + pathbase[8:10]
        udwt = ".true."
        ct = "0.d0"
        write_scatter_xy = '.true'
        if hirez_interp == True:
            only_scatter = '.true.'
        else:
            only_scatter = '.false.'

        # write file
        outputdir = dataout + pathbase
        try:
            dtNow = DT.datetime.strptime(pathbase, '%Y-%m-%dT%H%M%SZ')
        except ValueError:
            dtNow = DT.datetime.strptime(pathbase, '%Y-%m-%d')
        if np.size(dtrun) >= 1:  # if you want to use yesreday's bathy
            yester_str = whatisyesterday(now=dtNow, days=dtrun / 24)

        if nested == 1:
            ofile = str(pathbase) + 'nested.inp'  # name of output inp file
            # if np.size(dtrun) >= 1:
            # basefile = '../%s/%snested.dep' %(yester_str, yester_str) # yesterday's bathy ... hold static
        elif nested == 0:
            ofile = str(pathbase) + '.inp'  # name of ouput inp
            # if np.size(dtrun) == 1:
            # basefile = '../%s/%s.dep' %(yester_str, yester_str) # yesterday's bathy ... hold static
        # open and write the file
        f = open(outputdir + '/' + ofile, 'w')
        f.write('# Interpolation Input Options\n&interp_input\n')
        f.write(
            '   stw_sim_filename = "%s",\n   stw_depth_filename = "%s",\n   scatter_filename = "%s",\n   stw_updated_filename = "%s",\n' % (
                simfile, basefile, nestfile, otptfile))
        if hirez_interp == True:
            f.write(
                '   scatter_topo_posneg = %s,\n   new_idd_str = "%s",\n   write_scatter_xyz = %s,\n   update_wet = %s,\n' \
                '   dry_cutoff = %s,\n   write_scatter_xy = %s\n   only_scatter = %s\n/\n\n' % (
                    sct_tpt, newidd, wrtsct, udwt, ct, write_scatter_xy, only_scatter))
        else:
            f.write(
                '   scatter_topo_posneg = %s,\n   new_idd_str = "%s",\n   write_scatter_xyz = %s,\n   update_wet = %s,\n   dry_cutoff = %s\n/\n\n' % (
                    sct_tpt, newidd, wrtsct, udwt, ct))
        f.close()
        if hirez_interp == True:
            inpt3 = str(pathbase) + 'step3.inp'  # new file name for input 3
            grd = str(pathbase) + 'nested.grd'  # adcirc grid file name
            f = open(outputdir + '/' + inpt3, 'w')
            f.write('scatter_transform.elev\ndel.off\n%s' % grd)
            f.close()

            inpt4 = str(pathbase) + 'step4.inp'
            opt4 = str(pathbase) + 'adc2stw_weights.out'
            f = open(outputdir + '/' + inpt4, 'w')
            f.write('%s\n%s\n%s\n' % (grd, simfile, opt4))
            f.close()

            inpt5 = str(pathbase) + 'step5.inp'
            f = open(outputdir + '/' + inpt5, 'w')
            f.write("2\n%s\n%s\n%s\n-99999.d0\n'%s'\n100\n%d" % (opt4, grd, otptfile, basefile, smooth))
            f.close()

    def write_spec(self, date_str, path, STwavepacket, windpacket=0):
        """
        This function writes the spec file with the given wave spectra
        output packet from the prep_spec function
        Returns = [numrecs]
        windpacket=0 => constant 0 wind
        """

        SP_zone = 3200  # set for state plane FIPS North Carolina
        coord_sys = '"STATEPLANE"'  # if files aren't loading into STwave, this works with LOCAL
        numrecs = np.size(STwavepacket['STdWED'], axis=0)
        azimuth = 200  # Just metadata # changed from 198.2
        # %elevation adjustment in meters relative to bathymetry datum
        DADD = np.zeros(np.size(STwavepacket['STdWED'], axis=0))
        numpoints = 1  # % number of spatial points providing data (how many buoys)
        # coordinats of thos points [915887.133, 283454.273]
        Xcorr = [915887.13, ]  #
        Ycorr = [283454.27, ]  # %(XCOOR, YCOOR) = point location
        # wind
        if windpacket == 0:
            Umag = np.zeros(numrecs)  # wind magnitude
            Udir = np.zeros(numrecs)  # wind Direction relative to STwave Coords
        elif windpacket != 0:
            Umag = windpacket['windspeed']
            Udir = windpacket['winddir']
        if STwavepacket['STdWED'].ndim != 3:
            STwavepacket['STdWED'] = np.expand_dims(STwavepacket['STdWED'], axis=0)
        assert STwavepacket['STdWED'].shape[
                   1] == 62, 'this check, is in place to make sure the shape is properly setup [t, freq, dir'

        # _______________________________________________________________________
        # writing output file
        if date_str.split('.')[-1] == 'out':
            outputdir = path
            ospecname = date_str  # this is for the instance when want to write out an output file
            refTime = STwavepacket['snapbase'][0]
            refUnits = 'mm'
        else:
            outputdir = path + date_str
            ospecname = str(date_str) + '.eng'  # outputdir+infname[-18:-2]+'eng'

        print 'SPEC output file location/name : ', '/' + outputdir + '/' + ospecname
        # open the output file
        f = open(outputdir + '/' + ospecname, 'w')
        f.write('# STWAVE_SPECTRAL_DATASET\n#\n&datadims\n  datatype = 0,\n  numrecs = %d,\n' % numrecs)
        f.write('  numfreq = %d,\n  numangle = %d,\n' % (
            np.size(STwavepacket['wavefreqbin']), np.size(STwavepacket['wavedirbin'])))
        f.write("  numpoints = %d,\n  azimuth = %f,\n  coord_sys = %s,\n  spzone = %d," % (
            numpoints, azimuth, coord_sys, SP_zone))
        if date_str.split('.')[-1] == 'out':
            f.write('\n  RecUnits = "%s",\n  RefTime = "%s",' % (refUnits, refTime))
        f.write('\n/')
        f.write('\n#Frequencies\n')
        for pp in range(0, np.size(STwavepacket['wavefreqbin'])):
            f.write('%.4f ' % STwavepacket['wavefreqbin'][pp])
            # if pp % sqrt(np.size(STwavepacket['wavefreqbin'])) == 7:
            #     f.write('\n')
        f.write('\n#\n')
        # ________________________________________
        # writing Spectral Energy data
        for ii in range(0, numrecs):  # loop through snaps
            for zz in range(0, numpoints):  # looping through locations
                # Writing header for each snap
                f.write('   %s %.1f %.1f %f %f %.2f %.2f\n' % (
                    STwavepacket['snapbase'][ii], Umag[ii], Udir[ii], STwavepacket['peakf'][ii], DADD[ii], Xcorr[zz],
                    Ycorr[zz]))
                for aa in range(0, STwavepacket['STdWED'].shape[1]):
                    f.write('    ')
                    for bb in range(0, STwavepacket['STdWED'].shape[2]):
                        f.write('%e ' % STwavepacket['STdWED'][ii, aa, bb])
                    f.write('\n')
        f.close()

    def write_sim(self, datestring, path, snapbase, nested=0, windpacket=0, WLpacket=0, curpacket=0, statloc=0, full=0):
        """
        this writes the input .sim file for STWAVE model for the USACE Coastal Model Test Bed

        _______Input variables
        filebase: this is a timestamp        (string)
        numrecs: number of wave/wind records (integer) (removed)
        dxdy: cell resolution in meters      (integer)
        snapbase: a timestamp for the timesteps (string)
        windpacket: a packet containing wind data (dictionary)
        WLpacket: a packet containing WL data matching timestamps (dictionary)
        curpacket: a packet containing current data w. matching timesteps (dictionary)
        statloc: an array of locations for wave monitoring (2 by x array)
        nested: 1 for a nested sim 0 for a outer grid
        """
        if full == 1:
            cellPartitionBreak = 52  # 20 is the minimum
        elif full == 0:
            cellPartitionBreak = 20  # this will give most processors
        numrecs = len(snapbase)
        outputdir = path + datestring  # the directory of oput data
        version_prefix = outputdir.split('_')[0]
        # set standard parameters
        if full == 1:
            iplane = 1
        elif full == 0:
            iplane = 0  # 0=half plane 1=full plane
        if windpacket == 0:
            iprp = 1  # 0=prop and source 1=propagation only
        else:
            iprp = 0
        icur = 0  # 0=nocurrents 1= current for each snap 2=same current all snap
        ibreak = 0  # 0=no break record 1=record break cell 2=record dissipation
        irs = 0  # 0=no rad stress 1=calc rad stress
        if nested == 1:
            nnest = 0  # num opt cell output spectre
        elif nested == 0:
            nnest = 3  # using 3 output locations
        if np.size(statloc) > 1:
            nstations = np.size(statloc, axis=0)
        else:
            nstations = 0  # num opt interp output points
        if nested == 1:
            ibnd = 2  # linear interp along nested boundary
        elif nested == 0:
            ibnd = 0  # 0=single point || interp input spectra used with mult input spectra
        ifric = 3  # 0=no bottom fric (1,2,3,4 other fric models)
        idep_opt = 0  # 0=use *.dep file for depth 1=use constant slope
        isurge = 0  # 0=spatially constant surge 1= spatially variable WL correction
        iwind = 0  # 0=spatially constant wind 1= spatially variable wind
        i_bc1 = 2  # input from SPEC (*.eng) file
        i_bc2 = i_bc4 = 3  # 1-D transformed spectra (lateral boundaries)
        i_bc3 = 0  # zero spectrum, land
        idd_spec_type = 2  # 1 = non= spaced intervals endter info under snap_idds
        numsteps = len(snapbase)  # number of snap IDD's to process

        # Setting parameters based on input grid
        # read grid parameters
        if nested == 0:
            ifdep = '"' + datestring + '.dep"'  # input bathy file
        elif nested == 1:
            ifdep = '"' + datestring + 'nested.dep"'
        # this grab dxdy from live background dep file
        dxdyDict = self.getGridDxDyfromDep(ifdep.split('"')[1])

        n_cell_i = dxdyDict['NI']
        n_cell_j = dxdyDict['NJ']
        dx = dxdyDict['dx']
        dy = dxdyDict['dy']

        # setting wave monitoring output locations
        if nested == 0 and dx == 50 and n_cell_i == 342:
            inestpts = [[308, 426], [308, 443], [308, 460]]
        elif nested == False and dx == 50 and n_cell_i == 94:
            inestpts = [[57, 426], [57, 443], [57, 460]]
        elif nested == 0 and dx == 10:
            inestpts = [[1536, 2124], [1536, 2213], [1536, 2302]]

        # defining partioning based on grids (above)
        if full == 0:
            n_grd_part_i = 1  # grid partitions for parallel processing
            n_grd_part_j = np.floor(n_cell_j / cellPartitionBreak)  # partitions in the j direction
        elif full == 1:
            n_grd_part_i = np.floor(n_cell_i / cellPartitionBreak)
            n_grd_part_j = np.floor(n_cell_j / cellPartitionBreak)  # 12

        nproc = n_grd_part_i * n_grd_part_j

        # parallel options
        n_init_iters = 20  # num of init its to perform per snap
        init_iters_stop_value = 0.05  # init its stop convergence criteria
        init_iters_stop_percent = 100.0  # init its stop % crit
        n_final_iters = 20  # max num of final its to perform per snap
        final_iters_stop_value = 0.05  # final its stop converge crit
        final_iters_stop_percent = 99.8  # final its stop % crit
        # input output options
        DEFAULT_INPUT_IO_TYPE = 1  # 1= ascii 2= bianary (XMDF)
        DEFAULT_OUTPUT_IO_TYPE = 1  # 0=nooutput 1=ascii 2=xmdf 3 ascii/xmdf

        # Set Spatial Grid Parameters Section
        coord_sys = "'STATEPLANE'"  # stateplane/local/UTM
        spzone = 3200  # state plane FIPS code
        if nested == 0:
            if n_cell_i == 342:
                x0 = 910161.18  # X-coord origin [m]
                y0 = 300875.53  # Y-coord origin [m]
            elif n_cell_i == 94:
                x0 = 898385.56  # origin from the 17m grid
                y0 = 297033.90
        elif nested == 1:
            x0 = 902361.0  # nested origin
            y0 = 275970.0  # nested origin
        azimuth = 200  # azimuth (rotation) of grid in degrees # changed from 198.2

        # print 'DEP file is fixed to the reigonal base %s' %ifdep
        ifsurge = glob.glob('*.surge')  # input WL file
        if nested == 0:
            ifspec = "'" + datestring + ".eng'"  # input spectral file
        elif nested == 1:
            ifspec = "'" + datestring + 'nested.eng"'
        ifwind = glob.glob('*.wind')  # input wind file
        iffric = []  # glob.glob('*.fric')     # input bottom fric file
        ifcurr = glob.glob('*.curr')  # input water currents file
        io_type_dep = 1  # 1=ASCII 2=XMDF (binary)
        io_type_surge = 1  # 1=ASCII 2=XMDF (binary)
        io_type_wind = 1  # 1=ASCII 2=XMDF (binary)
        io_type_spec = 1  # 1=ASCII 2=XMDF (binary)
        io_type_fric = 1  # 1=ASCII 2=XMDF (binary)
        io_type_curr = 1  # 1=ASCII 2=XMDF (binary)

        # Set model Output Files Section

        if nested == 0:  # parent file
            ofNEST = '"' + str(datestring) + 'nested.eng"'  # name of nested output file
            ofWAVE = '"' + str(datestring) + '.wave.out"'  # wave field output
            ofOBSE = '"' + str(datestring) + '.obse.out"'  # output spec file
            ofBREAK = '"' + str(datestring) + '.break.out"'  # breaker indicies
            ofRADS = '"' + str(datestring) + '.rads.out"'  # radiation stress file
            ofSELH = '"' + str(datestring) + '.selh.out"'  # selected wave out put
            ofSTATION = '"' + str(datestring) + '.station.out"'  # output station file
            ofLOGS = '"' + str(datestring) + '.log.out"'  # name of run time log files
            ofTP = '"' + str(datestring) + '.Tp.out"'  # name of output peake wave period file
            ofXMDF_SPATIAL = '"' + str(datestring) + '.spatial.out.h5"'  # name of XMDF spatial output file

        elif nested == 1:
            ofNEST = '"' + str(datestring) + '.nest.out"'  # name of nested output for the nested sim
            ofWAVE = '"' + str(datestring) + 'nested.wave.out"'  # wave field output
            ofOBSE = '"' + str(datestring) + 'nested.obse.out"'  # output spec file
            ofBREAK = '"' + str(datestring) + 'nested.break.out"'  # breaker indicies
            ofRADS = '"' + str(datestring) + 'nested.rads.out"'  # radiation stress file
            ofSELH = '"' + str(datestring) + 'nested.selh.out"'  # selected wave out put
            ofSTATION = '"' + str(datestring) + 'nested.station.out"'  # output station file
            ofLOGS = '"' + str(datestring) + 'nested.log.out"'  # name of run time log files
            ofTP = '"' + str(datestring) + 'nested.Tp.out"'  # name of output peake wave period file
            ofXMDF_SPATIAL = '"' + str(datestring) + 'nested.spatial.out.h5"'  # name of XMDF spatial output file
        io_type_tp = 1  # 0=nooutput 1=ascii 2=xmdf 3 ascii/xmdf
        io_type_nest = 1  # 0=nooutput 1=ascii 2=xmdf 3 ascii/xmdf
        io_type_selh = 1  # 0=nooutput 1=ascii 2=xmdf 3 ascii/xmdf
        io_type_rads = 1  # 0=nooutput 1=ascii 2=xmdf 3 ascii/xmdf
        io_type_break = 1  # 0=nooutput 1=ascii 2=xmdf 3 ascii/xmdf
        io_type_obse = 1  # 0=nooutput 1=ascii 2=xmdf 3 ascii/xmdf
        io_type_wave = 1  # 0=nooutput 1=ascii 2=xmdf 3 ascii/xmdf
        io_type_station = 1  # 0=nooutput 1=ascii 2=xmdf 3 ascii/xmdf
        if io_type_tp == 0:
            print ' TP file are not output'
        if io_type_wave == 0:
            print 'WAVE output files are not output'
        # set Snap IDDs
        if len(snapbase) == 1:
            i_time_inc = 0  # this is a single record run
        elif int(snapbase[1][-4:]) - int(snapbase[0][-4:]) == 100:
            i_time_inc = 60  # hourly increments
        elif int(snapbase[1][-4:]) - int(snapbase[0][-4:]) == 30 or int(snapbase[1][-4:]) - int(snapbase[0][-4:]) == 70:
            i_time_inc = 30  # fix this                 # time increment between snaps
        else:
            print '%%%%%%%%%%\nERROR:  YOU HAVE FUNKY TIMESTAMPS!!!\n%%%%%%%%%%'
        i_time_inc_units = 'mm'  # units of increment
        iyear_start = snapbase[0][0:4]  # defining the year of the sim start
        imon_start = snapbase[0][4:6]  # start mon
        iday_start = snapbase[0][6:8]  # start day
        ihr_start = snapbase[0][8:10]
        imin_start = snapbase[0][10:]
        isec_start = 0
        iyear_end = snapbase[-1][0:4]
        imon_end = snapbase[-1][4:6]
        iday_end = snapbase[-1][6:8]
        ihr_end = snapbase[-1][8:10]
        imin_end = snapbase[-1][10:]
        isec_end = 0
        # _________________________________________________________________
        # Setting I,J obse file locations
        # note these are pulled from the .log file the nstations
        print 'the OBSE LOCATIONS FOR THE SPECTRA OBSERVATIONS are manually input based on the .log output locations'
        if nested == 0 and n_cell_i == 342:  # and (dx == 5 or dx == 10):
            if version_prefix == 'FP':
                selectpts = [[248, 427],  # 17m Waverider
                             [298, 438],  # AWAC11m   [11m]
                             [306, 439],  # AWAC8m    [8m]
                             [312, 439],  # AWAC5m    [6m]
                             [316, 439],  # AWAC4.5m  [4.5m]
                             [318, 439],  # adop3m    [3m]
                             [320, 439],  # paros04   [2m]
                             [321, 439],  # paros03   [1m]
                             [322, 439],  # paros02   [0m]
                             [1, 367]]  # WwaveRIder [26m] - source # moved
            elif version_prefix in ['HP', 'CB', 'CBHP']:
                selectpts = [[248, 427],  # 17m Waverider
                             [298, 438],  # AWAC11m   [11m]
                             [306, 439],  # AWAC8m    [8m]
                             [312, 439],  # AWAC5m    [6m]
                             [316, 439],  # AWAC4.5m  [4.5m]
                             [318, 439],  # adop3m    [3m]
                             [320, 439],  # paros04   [2m]
                             [321, 439],  # paros03   [1m]
                             [322, 439],  # paros02   [0m]
                             [2, 367]]  # WwaveRIder [26m] - source
                            # moved to node 2 due to first cell not being able to output in Half plane
                if version_prefix == 'CB' or version_prefix == 'CBHP':
                    # nearshore locations for Steve and britts wave data in 50 m grid space
                    selectpts.extend(([325, 442],
                                      [325, 442],
                                      [324, 442],
                                      [323, 442],
                                      [325, 443],
                                      [325, 443],
                                      [324, 443],
                                      [323, 443]))
        elif nested == 0 and n_cell_i == 94:  # short parent simulation (starting at 17m array)
            selectpts = [[2, 428],  # 17m Waverider
                         [51, 439],  # AWAC11m   [11m]
                         [58, 439],  # AWAC8m    [8m]
                         [65, 439],  # AWAC5m    [6m]
                         [69, 440],  # AWAC4.5m  [4.5m]
                         [71, 440],  # adop3m    [3m]
                         [73, 440],  # paros04   [2m]
                         [74, 440],  # paros03   [1m]
                         [74, 440],  # paros02   [0m]
                         [-99999, -99999]]  # WwaveRIder [26m] - source
        elif nested == 1 and dx == 5:
            # standard runs, 5 m grid spacing
            selectpts = [[-99999, -99999],  # 17m Waverider
                         [-99999, -99999],  # AWAC11m   [11m]
                         [17, 132],  # AWAC8m    [8m]
                         [79, 132],  # AWAC5m    [6m]
                         [120, 132],  # AWAC4.5m  [4.5m]
                         [140, 132],  # adop3m    [3m]
                         [160, 132],  # paros04   [2m]
                         [170, 132],  # paros03   [1m]
                         [175, 132],  # paros02   [0m]
                         [-99999, -99999]]  # WwaveRIder [26m] - source
        elif nested == 1 and dx == 10:  # version_prefix == 'CB' or version_prefix == 'CBHP':
            # bathy duck expt 10m grid spacing
            selectpts = [[-99999, -99999],  # 17m Waverider
                         [-99999, -99999],  # AWAC11m   [11m]
                         [9, 66],  # AWAC8m    [8m]
                         [40, 66],  # AWAC5m    [6m]
                         [60, 66],  # AWAC4.5m  [4.5]
                         [70, 66],  # adop3m    [3m]
                         [80, 66],  # paros04   [2m]
                         [85, 66],  # paros03   [1m]
                         [88, 66],  # paros02   [0m]
                         [-99999, -99999]]  # WwaveRIder [26m] - source
            selectpts.extend(([88, 84],
                              [86, 84],
                              [81, 84],
                              [76, 84],
                              [88, 91],
                              [85, 92],
                              [80, 91],
                              [75, 91]))
        nselct = len(selectpts)  # num opt cell outputs (obse)

        # ________________________________________________________________
        # Setting optional inputs
        # Set Wind
        if windpacket == None:
            umag = np.zeros(numrecs)
            udir = np.zeros(numrecs)  # make sure this is direction of i direction (grid normal)
        elif windpacket != 0:  #

            if np.size(windpacket['avgspd']) == numrecs:
                umag = windpacket['avgspd']
                udir = windpacket['avgdir']
                iprp = 0
            else:
                print ' CHECK WIND RECORDS, not same as wave records count\n The wind has been set to zero'
                umag = np.zeros(numrecs)
                udir = np.zeros(numrecs)  # make sure this is direction of i direction (grid normal)

        # Set WL
        if WLpacket == None:
            WL = np.zeros(numrecs)  # WL data
        elif WLpacket != None:
            if np.size(WLpacket['avgWL']) == numrecs:
                WL = WLpacket['avgWL']
            else:
                print '\n.\n.\nCheck WATER LEVELS!!!!\n. The Record Count is off\n.WL has set to zero'
                WL = np.zeros(numrecs)

        # Set Currents
        if curpacket != None:
            icur = 1
        # __________________

        ### begin Writing file construction
        if nested == 0:
            osimfile = datestring + ".sim"
        elif nested == 1:
            osimfile = datestring + "nested.sim"
        print 'Output SIM file location/name: ', '/' + outputdir + '/' + osimfile
        # open and write header to simfile
        f = open(osimfile, 'w')  # outputdir + '/' + osimfile, 'w')
        f.write('# STWAVE_SIM_FILE\n#STWAVE Model Parameter Input File\n')
        f.write('# file automatically generated as part of FRF Coastal Model test bed\n')
        f.write('#\n######################################################################\n#\n')
        # writing standard parms
        f.write('#\n# Standard Input Section\n#\n')
        f.write('&std_parms\n')
        f.write('  iplane = %d,\n  iprp = %d,\n  icur = %d,\n  ibreak = %d,\n' % (iplane, iprp, icur, ibreak))
        f.write('  irs = %d,\n  nselct = %d,\n  nnest = %d,\n  nstations = %d,\n' % (irs, nselct, nnest, nstations))
        f.write('  ibnd = %d,\n  ifric = %d,\n  idep_opt = %d,\n  isurge = %d,\n' % (ibnd, ifric, idep_opt, isurge))
        f.write('  iwind = %d,\n  i_bc1 = %d,\n  i_bc2 = %d,\n  i_bc3 = %d,\n  i_bc4 = %d\n/\n' % (
            iwind, i_bc1, i_bc2, i_bc3, i_bc4))
        # writing runtime parameters
        f.write('#\n# Runtime Parameters Section\n#\n')
        f.write('&run_parms\n  idd_spec_type = %d,\n  numsteps = %d,\n' % (idd_spec_type, numsteps))
        f.write('  n_grd_part_i = %d,\n  n_grd_part_j = %d,\n  n_init_iters = %d,\n' % (
            n_grd_part_i, n_grd_part_j, n_init_iters))
        f.write('  init_iters_stop_value = %f,\n  init_iters_stop_percent = %.1f,\n  n_final_iters = %d,\n' % (
            init_iters_stop_value, init_iters_stop_percent, n_final_iters))
        f.write('  final_iters_stop_value = %f,\n  final_iters_stop_percent = %.1f,\n' % (
            final_iters_stop_value, final_iters_stop_percent))
        f.write('  DEFAULT_INPUT_IO_TYPE = %d,\n  DEFAULT_OUTPUT_IO_TYPE = %d\n/\n' % (
            DEFAULT_INPUT_IO_TYPE, DEFAULT_OUTPUT_IO_TYPE))
        # writing Spatial Grid Parameters Section
        f.write('#\n# Spatial Grid Parameters Section\n#\n&spatial_grid_parms\n')
        f.write('  coord_sys = %s,\n  spzone = %d,\n  x0 = %.8f,\n  y0 = %.8f,\n' % (coord_sys, spzone, x0, y0))
        f.write('  azimuth = %f,\n  dx = %f,\n  dy = %f,\n  n_cell_i = %d,\n  n_cell_j = %d\n/\n' % (
            azimuth, dx, dy, n_cell_i, n_cell_j))
        # writing Input files section
        f.write('#\n# Input Files Section\n#\n&input_files\n')
        if len(ifdep) > 0:
            f.write('  DEP = "%s",\n' % str(ifdep)[1:-1])
        else:
            print 'ERROR: There is no DEP input file written to sim file!!!!!!!!!!!!!'
        if len(ifsurge) > 0:
            f.write('  SURGE = "%s",\n' % str(ifsurge)[1:-1])
        if len(ifspec) > 0:
            f.write('  SPEC = "%s",\n' % str(ifspec)[1:-1])
        else:
            print 'ERROR: There is no SPEC input file written to sim file!!!'
        if len(ifwind) > 0:
            f.write('  WIND = "%s",\n' % str(ifwind)[1:-1])
        if len(iffric) > 1:
            f.write('  FRIC = "%s",\n' % str(iffric)[1:-1])
        if len(ifcurr) > 0:
            f.write('  CURR = %s,\n' % str(ifcurr)[1:-1])
        # writing IO types for applicable files
        if len(ifdep) > 0:
            f.write('  io_type_dep = %d,\n' % io_type_dep)
        if len(ifsurge) > 0:
            f.write('  io_type_surge = %d,\n' % io_type_surge)
        if len(ifwind) > 0:
            f.write('  io_type_wind = %d\n' % io_type_wind)
        if len(ifspec) > 0:
            f.write('  io_type_spec = %d,\n' % io_type_spec)
        if len(iffric) > 0:
            f.write('  io_type_fric = %d,\n' % io_type_fric)
        if len(ifcurr) > 0:
            f.write('  CURR = %s,\n' % io_type_curr)

        # writing output files section
        f.write('\n/\n#\n# Output Files Section\n#\n&output_files\n')
        f.write('  WAVE = %s,\n' % ofWAVE)
        if nselct > 0:
            f.write('  OBSE = %s,\n' % ofOBSE)
        if ibreak == 2:
            f.write('  BREAK = %s,\n' % ofBREAK)
        elif ibreak == 3:
            f.write('  BREAK = %s,\n' % ofBREAK)
        if irs == 1:
            f.write('  RADS = %s,\n' % ofRADS)
        if nselct > 0:
            f.write('  SELH = %s,\n' % ofSELH)
        if nstations > 0:
            f.write('  STATION = %s,\n' % ofSTATION)
        if nested == 0 or nested == 1:
            f.write('  NEST = %s,\n' % ofNEST)
        f.write('  LOGS = %s,\n' % ofLOGS)
        f.write('  TP = %s,\n' % ofTP)
        f.write('  XMDF_SPATIAL = %s,\n' % ofXMDF_SPATIAL)
        f.write('  io_type_tp = %d,\n' % io_type_tp)
        f.write('  io_type_nest = %d,\n' % io_type_nest)
        f.write('  io_type_selh = %d,\n' % io_type_selh)
        f.write('  io_type_rads = %d,\n' % io_type_rads)
        f.write('  io_type_break = %d,\n' % io_type_break)
        f.write('  io_type_obse = %d,\n' % io_type_obse)
        f.write('  io_type_wave = %d,\n' % io_type_wave)
        f.write('  io_type_station = %d\n/\n' % io_type_station)
        # Writing the Time parameters section
        f.write('#\n# Time Parameters Section\n#\n&time_parms\n')
        f.write("  i_time_inc = %d,\n  i_time_inc_units = '%s',\n" % (i_time_inc, i_time_inc_units))
        if iyear_start > 1:
            f.write(
                '  iyear_start = %s,\n  imon_start = %s,\n  iday_start = %s,\n' % (iyear_start, imon_start, iday_start))
            f.write('  ihr_start = %s,\n  imin_start = %s,\n  isec_start = %s,\n' % (ihr_start, imin_start, isec_start))
            f.write('  iyear_end = %s,\n  imon_end = %s,\n  iday_end = %s,\n' % (iyear_end, imon_end, iday_end))
            f.write('  ihr_end = %s,\n  imin_end = %s,\n  isec_end = %s\n/\n' % (ihr_end, imin_end, isec_end))
        # writing constant friction
        if ifric == 3:
            f.write('#\n# Constant Bottom Friction Value\n#\n&const_fric\n')
            f.write('cf_const = .073\n/\n')
            # writing snap IDDs
        if idd_spec_type == -2:
            f.write('#\n# Snap IDDs\n#\n@snap_idds\n')  # section header
            for snap in range(0, numrecs):
                if snap == numrecs - 1:
                    f.write('  idds(%d) = %s\n/\n' % (snap + 1, snapbase[snap] + '00'))
                else:
                    f.write('  idds(%d) = %s,\n' % (
                        snap + 1, snapbase[snap] + '00'))  # here's where the record id would go in
        if nselct > 0:
            f.write('#\n# Select Point Data\n#\n@select_pts\n')
            for slpt in range(0, len(selectpts)):
                if slpt == (nselct - 1):
                    f.write('  iout(%d) = %d, jout(%d) = %d\n/\n' % (
                        slpt + 1, selectpts[slpt][0], slpt + 1, selectpts[slpt][1]))
                else:
                    f.write('  iout(%d) = %d, jout(%d) = %d,\n' % (
                        slpt + 1, selectpts[slpt][0], slpt + 1, selectpts[slpt][1]))
        # writing neted point data
        if nnest > 0:
            f.write('#\n# Nest Point Data\n#\n@nest_pts\n')
            for npt in range(0, nnest):
                if npt == (nnest - 1):
                    f.write('  inest(%d) = %d, jnest(%d) = %d\n/\n' % (
                        npt + 1, inestpts[npt][0], npt + 1, inestpts[npt][1]))
                else:
                    f.write(
                        '  inest(%d) = %d, jnest(%d) = %d,\n' % (npt + 1, inestpts[npt][0], npt + 1, inestpts[npt][1]))

        # writing station location data
        if nstations > 0:
            f.write('#\n# Station Location Data\n#\n@station_locations\n')  # section header
            for stations in range(0, nstations):
                if stations == (nstations - 1):
                    f.write('  stat_xcoor(%d) = %f,  stat_ycoor(%d) = %f\n/\n' % (
                        stations + 1, statloc[stations, 0], stations + 1, statloc[stations, 1]))
                else:
                    f.write('  stat_xcoor(%d) = %f,  stat_ycoor(%d) = %f,\n' % (
                        stations + 1, statloc[stations, 0], stations + 1, statloc[stations, 1]))
        # writing Winds
        f.write('#\n# Spatially Constant Winds\n#\n@const_wind\n')
        for snap in range(0, numrecs):
            if snap == numrecs - 1:
                f.write('  umag_const_in(%d) = %f, udir_const_in(%d) = %f\n/\n' % (
                    snap + 1, umag[snap], snap + 1, udir[snap]))
            else:
                f.write('  umag_const_in(%d) = %f, udir_const_in(%d) = %f,\n' % (
                    snap + 1, umag[snap], snap + 1, udir[snap]))

        # Writing Water Levels
        f.write('#\n# Spatially Constant Water Level Adjustment\n#\n@const_surge\n')
        for snap in range(0, numrecs):
            if snap == numrecs - 1:
                f.write('  dadd_const_in(%d) = %.2f\n/\n' % (snap + 1, WL[snap]))
            else:
                f.write('  dadd_const_in(%d) = %.2f,\n' % (snap + 1, WL[snap]))
        # clos the file
        f.close()
        return nproc

    def write_flags(self, date_str, path, wavepacket, windpacket=0, WLpacket=0, curpacket=0, gridFlag=False):
        """
        This function is designed to dump data on the interpolated data to a file
        """
        filebase = path + date_str + '/' + 'Flags' + date_str + '.out.txt'
        time = wavepacket['time']

        if curpacket == None:
            curpacket = {'flag': np.full(len(wavepacket['flag']), 5, dtype=np.int)}
        if WLpacket == None:
            WLpacket = {'flag': np.full(len(wavepacket['flag']), 5, dtype=np.int)}
        if windpacket == None:
            windpacket = {'flag': np.full(len(wavepacket['flag']), 5, dtype=np.int)}
        if gridFlag == True:
            gflag = np.ones_like(time) * 4
        else:
            gflag = np.zeros_like(time)
        assert (len(wavepacket['flag']) == len(windpacket['flag']) == len(WLpacket['flag']) == len(
            curpacket['flag'])), 'The flags are of different lengths'
        f = open(filebase, 'w')
        f.write(
            'STWAVE DATA FLAG OUPUT FILE\nData written to output file to keep track of data used for each run\ntime record in UTC (0000-2400)\n')
        f.write(
            'Flag Values:\n0: Good Data\n1:Linearly Interpolated Data\n2: Place holder for future flag\n3: Place holder for future flag\n4: marked when 17m Grid used\n5: No Data/Place holder for no data\n')
        f.write('Date, time, Wave Flag, Wind Flag, WL Flag, Current Flag, Grid Flag\n')
        for cc in range(0, len(wavepacket['flag'])):
            f.write('%4d-%02d-%02d,%02d%02d,%d,%d,%d,%d,%d\n' % (
                time[cc].year, time[cc].month, time[cc].day, time[cc].hour, time[cc].minute, wavepacket['flag'][cc],
                windpacket['flag'][cc], WLpacket['flag'][cc], curpacket['flag'][cc], gflag[cc]))

        f.close()

    def findeol(self, fname, search=','):
        """
        finding the comma at the end of the STWAVE output files
        """
        for ii in range(-1, -len(fname[5]), -1):
            if fname[5][ii] == search:
                val = ii

        return val

    def obseload(self, nested=0, Headerlines=3):
        """
        this file will parse out the obse file from stwave. the obse file is
        the spectra associated with the output observation stations
        """
        if nested == 0 or nested == False:
            f = open(self.obsefname[0], 'r')
            obfile = f.readlines()
        elif nested == 1 or nested == True:
            f = open(self.obsefname_nest[0], 'r')
            obfile = f.readlines()
        else:
            print 'Nested option must be 1 or 0'
            raise
        f.close()
        eol = self.findeol(obfile)

        # begin parsing header data

        # first find freq's, has to start after header data dimensions
        while obfile[Headerlines].split()[0] != '#Frequencies':

            if obfile[Headerlines].split('=')[0].split()[0] in ['NumRecs', 'numrecs']:
                numrecs =  int(obfile[Headerlines].split('=')[1][:eol])
            elif obfile[Headerlines].split('=')[0].split()[0] in ['numfreq', 'Numfreq']:
                numfreq =  int(obfile[Headerlines].split('=')[1][:eol])
            elif obfile[Headerlines].split('=')[0].split()[0] in ['numangle', 'Numangle']:
                numang = int(obfile[Headerlines].split('=')[1][:eol])
            elif obfile[Headerlines].split('=')[0].split()[0] in ['numpoints', 'NumPoints']:
                numpts = int(obfile[Headerlines].split('=')[1][:eol])
            elif obfile[Headerlines].split('=')[0].split()[0] in ['azimuth', 'Azimuth']:
                azi = float(obfile[Headerlines].split('=')[1][:eol])
            elif obfile[Headerlines].split('=')[0].split()[0] in ['RefTime']:
                try:  # makin sure value is integer
                    reftime = int(obfile[Headerlines].split('=')[1][:eol].split('"')[0])
                except ValueError:  #for the instance when the ref time is in quotes
                    reftime = int(obfile[Headerlines].split('=')[1][:eol].split('"')[1])

            Headerlines += 1
        freqline = Headerlines+1

        # headerlines starts after frequencies find start of data lines
        while obfile[Headerlines].split()[0] != str(reftime):
            Headerlines += 1

        # parse out freqs
        freqs = []
        for ii in range(freqline, Headerlines):
            for freq in obfile[ii].split():
                if freq == '#':
                    break
                else:
                    freqs.append(float(freq))

        # spetra starts
        if len(obfile[-1].split()) == 72 or len(obfile[-1].split()) == 35:
            # the file is of the format that doesn't wrap the directions, has all 72/ 35 on one line
            snapStart = range(Headerlines, len(obfile) - Headerlines, numfreq + 1)  # the lines on which the snapstring resides
        else:  # file wrapps directions with 3 per line then next frequency
            snapStart = range(Headerlines, len(obfile) - Headerlines, int((numfreq) * np.ceil(numang/3.)+1)) # line indicies on which snap string resides
        # initialize variables
        IDD, uMag, uDir, Fp, WL, iOut, jOut= [], [], [], [], [], [], []
        spec = np.zeros((numrecs, numpts, numfreq, numang))
        uMag = np.zeros((numrecs, numpts))
        uDir = np.zeros((numrecs, numpts))
        Fp = np.zeros((numrecs, numpts))
        WL = np.zeros((numrecs, numpts))
        #parse spec

        for line in snapStart:
            # spectral locations
            tt = int(np.floor(len(IDD)/ numpts))
            ss = int(np.remainder(len(iOut), numpts))
            # splitting the headder for the individual spectrum
            string = obfile[line].split()
            # parsing that string
            IDD.append(DT.datetime(int(string[0][:4]), int(string[0][4:6]),
                                   int(string[0][6:8]), int(string[0][8:10]),
                                   int(string[0][10:12]), int(string[0][12:14])))
            uMag[tt, ss] = float(string[1])
            uDir[tt, ss] = float(string[2])
            Fp[tt, ss] = float(string[3])
            WL[tt, ss] = float(string[4])
            try:
                iOut.append(int(string[5]))  # standard operation
                jOut.append(int(string[6]))
            except ValueError:
                iOut.append(float(string[5]))  # stateplane operation (for input spec == obse)
                jOut.append(float(string[6]))
            # grabbing the spectrum
            try:
                for ff, frqline in enumerate(np.arange(line+1, int((numfreq)*np.ceil(numang/3.)+line))):  # line+1+numfreq)): # old file arangement
                    for dd, energy in enumerate(obfile[frqline].split()):
                       fff = int(np.floor(ff/np.ceil(numang/3.)))
                       ddd = int(dd + (3 * np.remainder(ff, np.ceil(numang/3.))))
                       spec[tt, ss, fff, ddd] = float(energy)
                       # seperate out each spec for each loc
            except IndexError:  # this is the scenario that has all directions on same line, not wrapped (increments of 3)
                for ff, frqline in enumerate(np.arange(line+1, int((numfreq) + line)+1)):
                    spec[tt, ss, ff, :] = obfile[frqline].split()
        if numang == 35:
            directions = np.arange(-85, 90, 5)
        else:
            directions = np.arange(0, 360, 5 )
        print 'directions associated with spectral output '
        dataout = {'Frequencies': np.array(freqs),
                   'spec': spec,
                   'iout': np.array(iOut[0:numpts]),
                   'jout': np.array(jOut[0:numpts]),
                   'IDD': np.array(np.unique(IDD)),
                   'uMag': uMag,
                   'WL': WL,
                   'Fp': Fp,
                   'directions': directions }
        return dataout

    def TPload(self, nested=0, Headerlines=21):
        """
        This function takes the Peak period file output by
        :param Headerlines: first line to start parsing data ie 20 lines of headers, start on line 21
        :param nested:
        """
        if nested == 0:  # if the sim file is not nested
            f = open(self.Tpfname[0], 'r')
            Tpfile = f.readlines()
        elif nested == 1:  # if the file is nested
            f = open(self.Tpfname_nest[0], 'r')
            Tpfile = f.readlines()
        else:
            print 'Nested option must be 1 or 0'
            raise
        f.close()
        eol = self.findeol(Tpfile)
        # begin parsing header data data

        # datatype = int(Tpfile[3][-4:-2])
        numrecs = int(Tpfile[4][12:eol])
        numflds = int(Tpfile[5][12:eol])
        NI = int(Tpfile[6][7:eol])  # number of cells in x direction
        NJ = int(Tpfile[7][7:eol])  # number of cells in J direction
        dx = float(Tpfile[8][7:eol])  # X-dir cell size
        dy = float(Tpfile[9][7:eol])  # y-dir cell size
        azi = float(Tpfile[10][12:eol])  # grid azimuth
        gridname = str(Tpfile[11][15:eol])
        fieldname = str(Tpfile[15][18:eol - 1])
        fieldunits = str(Tpfile[16][19:eol - 1])
        snapincrement = int(Tpfile[17][12:eol])  # increment between data
        incunits = str(Tpfile[18][15:eol - 1])
        reftime = int(Tpfile[19][14:eol])  # start time of snap
        # initializing the array to keep the Tp field data and time
        Tp_data = np.zeros((numrecs, NI, NJ))
        timedata = []

        for ii in range(Headerlines, len(Tpfile) - Headerlines,
                        NI * NJ + 1):  # interval numer of cells plus extra line for date
            timestring = str(int(Tpfile[ii][3:-2]))
            timedata.append(DT.datetime(int(timestring[0:4]), int(timestring[4:6]),
                                        int(timestring[6:8]), int(timestring[8:10]), int(timestring[10:12])))
            for pp in range(1, NI * NJ + 1):  # every cross shore column plus ext
                # spatial data sets are read from [0,max(NJ)] to [max(NI),max(NJ)] then to [1,max(NJ-1)] to [max(NI),max(NJ)-1]
                # At the FRF pier that's the south east corner to the south west corner, then progressing north
                Tp_data[(ii - Headerlines) / (NI * NJ + 1), -((pp - 1) - (((pp - 1) / NI) * (NI))), -pp / NI] = float(
                    Tpfile[ii + pp][:-2])

        Tp_packet = {'Tp_field': Tp_data, 'time': np.array(timedata), 'dx': dx, 'dy': dy,
                     'azimuth': azi, 'gridname': gridname, 'fieldname': fieldname,
                     'units': fieldunits, 'numrecs': numrecs, 'NI': NI, 'NJ': NJ,
                     'meta': 'Peak period data are organized by [date,xcoord,ycoord]\n cellsize in meters, NI/NJ: number of cells'
                     }
        return Tp_packet

class cmsIO():
    """
    This class takes care of the input and output for the CMS wave/Flow model

    """

    def writeCMS_std(self, fname, gaugeLocs):
        """
        This function will write the
        """
        # beggining of Function
        iprp = -1  # [0, 1, -1] [ wave generation and propagation, propagation only, fast-mode ]
        icur = 0  # [0, 1, 2] [no current, with currnent input *.cur sequential datasets, holding currents fixed (using first)]
        ibrk = 0  # [0, 1, 2] [no .brk file, break indicies .brk, energy dissipation fluxes *.brk]
        irs = 0  # [0, 1, 2] [no .rad file, output rad stresses *.rad, output rad stresses *.rad and setup *.wav]
        kout = np.shape(gaugeLocs)[
            0]  # [0, n] [no obs and selhts.out files, n for output of spectra *.obs, and paramaters (selhts.out) at n locations]
        ibnd = 0  # [0, 1, 2] [no .nst file, nested grid with linear interp at boundry input spec, morphic interp of boundary input spec]
        iwet = 0  # [0, 1] [normal wetting/drying using waterlevl, neglect water level input]
        ibf = 1  # [0,  1, 2, 3, 4] [no fric, constant darcy-weisbach, variable darcy weisbach, constant manning, vairable manning]
        iark = 0  # [0, 1] [no, yes] forward reflection
        iarkr = 0  # [0, 1] [no, yes] backward reflection
        akap = 4.  # [diffraction intensity factor, 0 none, 4 high ]
        bf = 0.005  # constant friction coeff
        ark = 0.5  # constant reflection coeff 0= none, 1=max
        arkr = 0.3  # constant backward reflection coeff 0= none, 1=max
        iwvbk = 0  # wave breaking [0, 1, 2, 3] [Goda, Miche, battjes and janssen, chawla and kirby]
        nonln = 0  # (none, default)  1 (nonlinear wave-wave interaction)
        igrav = 0  # (none, default)  1 (infra-gravity wave enter inlets)
        irunup = 0  # (none, default)  1 (automatic, runup output relative to absolute datum)
        #                 2 (automatic, runup output relative to updated MWL)
        imud = 0  # (mud.dat, default)   1 (none)  --------  need it for users
        #     who may not want to  include mud effect as the mud.dat exists
        #       (typical max kinematic viscosity in mud.dat is 0.04 m*m/sec)
        iwnd = 1  # wind.dat will be applied if it exists; default)
        # 1 (neglect *.wind card file as indicated in the sim file or
        # no wind.dat provided or neglect wind.dat if it exists)
        #  2 (dismiss incident wave inflation under stronger wind forcing)
        # --------  specify iwnd = 0 in steering if users decide not
        # to use the wind field  input when the wind field file exists
        isolv = 1  # (GSR solver,  default)  1 (ADI)
        ixmdf = 0  # (output ascii, default) 1 (output xmdf)  2 (input & output xmdf)
        iproc = 0  # (same as 1, default)  n (n processors for isolv = 0)
        #                 --- approx. processor number=(total row)/300
        iview = 1  # 0 (half-plane, default), 1, 2 (full-plane)
        #   --- for the full plane, users can provide the additional input
        #       wave spectrum file wave.spc (same format as the *.eng) along the
        #       opposite side boundary (an imaginary origin for this wave.spc at the
        #       opposite corner; users can use SMS to rotate the CMS-Wave grid 180
        #       deg to generate this wave.spc).
        #     iview = 1 won't read/need wave.spc even provided;
        #     iview = 2 will read wave.spc if exists. You can add a 'SPEC2'
        #             card in *.sim to specify this *.spc (no need to
        #              use the default wave.spc filename).
        iroll = 0  # 0 to 4 (wave roller effect, 0 for no effect, default
        #   4 for strong effect) -- more effective for finer resultion in
        # the surf zone, say, for the cross-shore spacing < 10 m

        string = ('%d  %d  %d  %d  '
                  '%d %d %d %d %d %d '
                  '%f %f %f %f %d %d %d '
                  '%d %d %d %d %d %d %d %d\n') % (iprp, icur, ibrk, irs, kout,
                                                  ibnd, iwet, ibf, iark, iarkr, akap,
                                                  bf, ark, arkr, iwvbk, nonln, igrav,
                                                  irunup, imud,  iwnd, isolv, ixmdf,
                                                  iproc,iview, iroll)
        assert fname.split('.')[-1] == 'std', 'check fname in extension'
        f = open(fname, 'w')  # open the file
        f.write(string)  # write the header
        for station in gaugeLocs:  # loop through the gauges
            f.write('%d %d\n' % (station[0], station[1]))
        f.close()  # close the file

    def writeCMS_spec(self, ofname, wavePacket, wlPacket, windPacket=0):
        """
        This function writes the CMS spec file with the given wave spectra
        output packet from the prep_spec function
        Returns = [numrecs]
        windpacket=0 => constant 0 wind
        """
        numpoints = 1  # just incase there are multiple forcing points initialize off of
        numrecs = np.size(wavePacket['dWED'], axis=0)
        azimuth = 200  # Just metadata # changed from 198.2
        # elevation adjustment in meters relative to bathymetry datum
        DADD = wlPacket['avgWL']
        # coordinats of thos points [915887.133, 283454.273]
        Xcorr = [915887.13, ]  #
        Ycorr = [283454.27, ]  # %(XCOOR, YCOOR) = point location
        # wind
        if windPacket == 0:
            Umag = np.zeros(numrecs)  # wind magnitude
            Udir = np.zeros(numrecs)  # wind Direction relative to STwave Coords
        elif windPacket != 0:
            Umag = windPacket['windspeed']
            Udir = windPacket['winddir']
        # if its a single record,
        if wavePacket['dWED'].ndim != 3:
            wavePacket['dWED'] = np.expand_dims(wavePacket['dWED'], axis=0)
        assert wavePacket['dWED'].shape[
                   1] == 62, 'this check, is in place to make sure the shape is properly setup [t, freq, dir'

        # _______________________________________________________________________
        # writing output file

        print 'SPEC output file location/name : ', ofname
        # open the output file
        f = open(ofname, 'w')
        f.write(' %d %d\n ' % (wavePacket['wavefreqbin'].size, wavePacket['wavedirbin'].size))
        # writting frequencies
        for pp in range(0, np.size(wavePacket['wavefreqbin'])):
            f.write('%.5f ' % wavePacket['wavefreqbin'][pp])
            if pp % 10 == 9:
                f.write('\n ')  # this writes a line break every 10 freqs
        f.write('\n')
        # ________________________________________
        # writing Spectral Energy data
        for ii in range(0, numrecs):  # loop through snaps
            for zz in range(0, numpoints):  # looping through locations
                # Writing header for each snap
                f.write('%s %.1f %.1f %f %f\n' % (
                    wavePacket['snapbase'][ii], Umag[ii], Udir[ii],
                    wavePacket['peakf'][ii], DADD[ii]))
                for aa in range(0, wavePacket['dWED'].shape[1]):
                    f.write('    ')
                    for bb in range(0, wavePacket['dWED'].shape[2]):
                        f.write('%e ' % wavePacket['dWED'][ii, aa, bb])
                    f.write('\n')
        f.close()

    def writeCMS_sim(self, fname, datestring, origin):
        """
        this function creates sim files for CMS Wave
        """
        azimuth = 200.
        STPN_N = origin[0]  # easting origin
        STPN_E = origin[1]  # Northing origin
        assert STPN_E > 280000, 'Check your Easting, not in range'
        assert STPN_N > 90000, 'Check your Northing, not in range'

        headerString = 'CMS-WAVE    %.4f    %.4f       %.4f\n' %(STPN_N, STPN_E, azimuth)

        f = open(fname, 'w')
        f.write(headerString)
        f.write('DEP       ' + datestring + '.dep\n')
        f.write('OPTS      ' + datestring + '.std\n')
        f.write('CURR      ' + datestring + '.cur\n')
        f.write('SPEC      ' + datestring + '.eng\n')
        f.write('WAVE      ' + datestring + '.wav\n')
        f.write('OBSE      ' + datestring + '.obs\n')
        f.write('NEST      ' + datestring + '.nst\n')
        f.write('BREAK     ' + datestring + '.brk\n')
        f.write('SPGEN     ' + datestring + '.txt\n')
        f.write('RADS      ' + datestring + '.rad\n')
        f.write('STRUCT    ' + datestring + '.struct\n')
        f.write('FRIC      ' + datestring + '.fric\n')
        f.write('FREF      ' + datestring + '.fref\n')
        f.write('BREF      ' + datestring + '.bref\n')
        f.write('MUD       ' + datestring + '.mud\n')
        f.write('WIND      ' + datestring + '.wind\n')
        f.close()

    def writeCMS_dep(self, fname, depPacket):
        """
        This function writes out the cms Dep file

        :param fname: output file name
        :param depPacket:
        :return:
        """
    def ReadCMS_dep(self, fname):

        # fname = '/home/spike/CMTB/HP_CMS_data/CMS-Wave-FRF.dep'  # example file.
        # open file and read it
        f = open(fname)
        depFile = f.readlines()
        f.close()

        meta = depFile[0].split()
        nX = int(meta[0])   # number of cells in X
        nY = int(meta[1])   # number of cells in Y
        dx = float(meta[2])   # not sure what this value is for 'non uniform spacing'
        dy = float(meta[-1])  # key resides here for uniform/non-uniform

        if dy == 999:  # check if its uniform grid
            nonUniform = True
        # load data
        if nonUniform == True:
            # load xSpacing's
            # here is a list of starts for every new ____________?

            # first read last cell spacing, reading from bottom of file
            startjs, jj = len(depFile)-nY/len(depFile[1].split())-1, []
            for j in np.arange(startjs, startjs+np.ceil(nY/len(depFile[1].split())) + 1 ,dtype=int):
                jj.extend(depFile[j].split())
            # next read in X spacings, find start for I spacings, moving up one section
            startis, ii = len(depFile)-nY/len(depFile[1].split())-1 - nX/len(depFile[1].split())-1, []
            for i in np.arange(startis, startis+ np.ceil(nX/len(depFile[1].split())) + 1, dtype=int):
                ii.extend(depFile[i].split())
            # now parse through elevations, moving from top, to startis
            data = np.zeros((nX, nY))
            lineStart = np.arange(1, startis, np.ceil(nX / len(depFile[1].split())) + 1, dtype=int)
            for bb, line in enumerate(lineStart):
                kk = []  # must be cleared with each loop
                for k in np.arange(0, np.ceil(nX/len(depFile[1].split()))+1, dtype=int):
                    kk.extend(depFile[line+k].split())
                data[:, bb] = np.array(kk, dtype=float)

        ii = np.array(ii, dtype=float)
        jj = np.array(jj, dtype=float)
        assert len(ii) == nX, 'Xcoordinates do not match number given in meta data - CMSreadDEP'
        assert len(jj) == nY, 'Ycoordinates do not match number given in meta data - CMSreadDEP'

        return data, ii, jj

    def ReadCMS_sim(self, fname):
        """
        this function reads the sim file and parses the data out for things of interest
        :param fname:
        :return: grid origin, x, y, and azimuth
        """

        f = open(fname, 'r')
        sim = f.readlines()
        f.close()
        split = sim[0].split()
        easting = float(split[1])  # grid origin in easting [m]
        northing = float(split[2]) # grid origin in northing [m]
        azimuth = float(split[3])  # parsing grid azimuth (angle of j axis from grid)

        for line in sim:
            split = line.split()
            if split[0] == 'DEP':
                self.depFname = split[1]
            elif split[0] == 'OPTS':
                self.optsFname = split[1]
            elif split[0] == 'CURR':
                self.currFname = split[1]
            elif split[0] == 'SPEC':
                self.specFname = split[1]
            elif split[0] == 'WAVE':
                self.waveFname = split[1]
            elif split[0] == 'OBSE':
                self.obseFname = split[1]
            elif split[0] == 'NEST':
                self.nestFname = split[1]
            elif split[0] == 'BREAK':
                self.breakFname = split[1]
            elif split[0] == 'SPGEN':
                self.spgenFname = split[1]
            elif split[0] == 'RADS':
                self.radsFname = split[1]
            elif split[0] == 'STRUCT':
                self.structFname = split[1]
            elif split[0] == 'FRIC':
                self.fricFname = split[1]
            elif split[0] == 'FREF':
                self.frefFname = split[1]
            elif split[0] == 'BREF':
                self.brefFname = split[1]
            elif split[0] == 'MUD':
                self.mudFname = split[1]
            elif split[0] == 'WIND':
                self.windFname = split[1]

        return easting, northing, azimuth