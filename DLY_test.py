import datetime as DT
import os, sys
import warnings
import netCDF4 as nc
import numpy as np
sys.path.append('C:\Users\RDCHLDLY\PycharmProjects\cmtb')
from sblib import geoprocess as gp
import makenc
from matplotlib import pyplot as plt



LLHC = [0, 0]
URHC = [2000, 4000]
coord_system = 'frf'


# go on fxn....

# first check the coord_system string to see if it matches!
coord_list = ['FRF', 'LAT/LON']
import pandas as pd
import string
exclude = set(string.punctuation)
columns = ['coord', 'user']
df = pd.DataFrame(index=range(0, np.size(coord_list)), columns=columns)
df['coord'] = coord_list
df['user'] = coord_system
df['coordToken'] = df.coord.apply(lambda x: ''.join(ch for ch in str(x) if ch not in exclude).strip().upper())
df['coordToken'] = df.coordToken.apply(lambda x: ''.join(str(x).split()))
df['userToken'] = df.user.apply(lambda x: ''.join(ch for ch in str(x) if ch not in exclude).strip().upper())
df['userToken'] = df.userToken.apply(lambda x: ''.join(str(x).split()))
userToken = np.unique(np.asarray(df['userToken']))[0]
assert df['coordToken'].str.contains(userToken).any(), 'makeBackgroundBathy Error: invalid coord_system string.  Acceptable strings include %s' %coord_list

# second, check the format of the corner inputs
LLHC = np.asarray(LLHC)
URHC = np.asarray(URHC)
assert len(LLHC) == len(URHC) == 2, 'makeBackgroundBathy Error: invalid corner input.  corner inputs must be of form (x, y)'

# ok, now to start off, we need to find the center of this rectangle, and the corner to corner distance in m.
# what this will do for us is allow us to quickly check to see if the bounds of our data are outside of
# our backgroundDEM without checking a ton of points!

