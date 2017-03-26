# About xcesm
Xcesm tries to provide a easy-to-use plugin for xarray to better handle CESM output in python. 

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
