import xarray as _xr
import pkg_resources as _pk
# will append when needed
locations = {'Green_land': [72, 73, 321, 323],
            'Brazil': [-25, -15, 290, 310],
            'Hulu': [30.5, 33.5, 117.5, 120.5],
            'Nino34': [-5,5,190,240]}

# http://stackoverflow.com/questions/779495/python-access-data-in-package-subdirectory
DATA_PATH = _pk.resource_filename('xcesm', 'utils/')

mask_g16 = _xr.open_dataarray(DATA_PATH + 'REGION_MASK_gx1v6.nc')
tarea_g16 = _xr.open_dataarray(DATA_PATH + 'TAREA_gx1v6.nc')

# ocean basin for pop output
def ocean_region():

    rg = mask_g16

    regions = {'Atlantic': (rg==6) | ((rg==1)&((rg.TLONG>=300) | (rg.TLONG<=20))),
               'Pacific':  (rg==2) | ((rg==1)&((rg.TLONG>=150)& (rg.TLONG<=290))),
               'Pacific_LGM': ((rg==2) & (rg.TLAT<=40) | (rg.TLAT>=55) & (rg.TLONG >=125) |
                (rg.TLONG>=140)) | ((rg==1)&((rg.TLONG>=150)& (rg.TLONG<=290))),
               'SouthernOcn': rg==1}

    return regions



