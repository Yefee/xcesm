# About xcesm
Xcesm tries to provide an easy-to-use plugin for xarray to better handle CESM output in python. 

# Features
Xcesm is still in developing, right now it has the following features:
* quick plot on global map (quickmap)
* regrid pop output to linear grids (regrid, nearest interpolation)
* compute global mean (gbmean, gbmeanpop)
* diagnose AMOC, PRECP, d18O(only support for iCESM), Heat transport etc.
* truncate ocean as several main basins (ocean_region)

More feature will be added in the future.

# How to install
### via git
```
git clone https://github.com/Yefee/xcesm.git
cd xcesm
python setup.py install
```

# How to use
### regrid
```
import xarray as xr
import xcesm
ds = xr.open_dataset('/examples/data/salt.nc')
# defalut to 1x1 degree
salt_rgd = ds.SALT.utils.regrid()

print(ds.SALT.shape)
(384, 320)

print(salt_rgd.shape)
(181, 361)
```

### quick plot
```
salt_rgd.utils.quickmap()
```
![salt_distribution](https://github.com/Yefee/xcesm/blob/master/xcesm/examples/fig/salt.png)


# And more
I don't have time to write documentation recently, but it will be released in this summer!

