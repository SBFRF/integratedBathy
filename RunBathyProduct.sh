whoami=$(whoami)
mypath='/home/'$whoami'/anaconda2/bin:'
export PATH=$mypath:$PATH
echo $PATH
cd '/home/'$whoami'/repos/makebathyinterp'
python 'bathyWrapper.py'
