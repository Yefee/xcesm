import os
import xarray as xr

# will append when needed
locations = {'Green_land': [72, 73, 321, 323],
            'Brazil': [-25, -15, 290, 310],
            'Hulu': [30.5, 33.5, 117.5, 120.5],
            'Nino34': [-5,5,190,240]}


SETS = {'precp': ['PRECC', 'PRECL'],
        'd18op': ['PRECRC_H216Or', 'PRECSC_H216Os', 'PRECRL_H216OR', 'PRECSL_H216OS',
                  'PRECRC_H218Or', 'PRECSC_H218Os', 'PRECRL_H218OR', 'PRECSL_H218OS'],
        'moc': ['MOC']}

COMP = {'precp': 'atm',
        'd18op': 'atm',
        'moc': 'ocn'}





DATA_PATH = os.path.join(os.path.dirname(__file__), '../config/')
mask_g16 = xr.open_dataarray(DATA_PATH + 'REGION_MASK_gx1v6.nc')
mask_g35 = xr.open_dataarray(DATA_PATH + 'REGION_MASK_gx3v5.nc')
mask_g37 = xr.open_dataarray(DATA_PATH + 'REGION_MASK_gx3v7.nc')
tarea_g16 = xr.open_dataarray(DATA_PATH + 'TAREA_gx1v6.nc')
tarea_g35 = xr.open_dataarray(DATA_PATH + 'TAREA_gx3v5.nc')
tarea_g37 = xr.open_dataarray(DATA_PATH + 'TAREA_gx3v7.nc')
dz_g16 = xr.open_dataarray(DATA_PATH + 'DZ_gx1v6.nc')
dz_g35 = xr.open_dataarray(DATA_PATH + 'DZ_gx3v5.nc')

# ocean basin for pop output
def ocean_region():

    rg = mask_g16

    regions = {'Atlantic': (rg==6) | ((rg==1)&((rg.TLONG>=300) | (rg.TLONG<=20))),
               'Pacific':  (rg==2) | ((rg==1)&((rg.TLONG>=150)& (rg.TLONG<=290))),
               'Pacific_LGM': ((rg==2) & (rg.TLAT<=40) | (rg.TLAT>=55) & (rg.TLONG >=125) |
                (rg.TLONG>=140)) | ((rg==1)&((rg.TLONG>=150)& (rg.TLONG<=290))),
               'SouthernOcn': rg==1}

    return regions


def open_data(var, project_name='iTRACE', **kwargs):
    return iTRACE(var, project_name).open_data(**kwargs)


class open_iTrace:

    def __init__(self, var, project_name='iTRACE', **kwargs):
        self.ice, self.ico, self.igo, self.igom = iTRACE(var, project_name).open_data(**kwargs)

class open_iTrace_forcing:
    def __init__(self):
        self.DATA_PATH = os.environ['iTRACE_DATA']
        self.solin_jja = xr.open_mfdataset(os.path.join(self.DATA_PATH, 'forcing/*.SOLIN.*.JJA.nc'))
        self.solin_djf = xr.open_mfdataset(os.path.join(self.DATA_PATH, 'forcing/*.SOLIN.*.DJF.nc'))
        self.ghgs = xr.open_mfdataset(os.path.join(self.DATA_PATH, 'forcing/iTRACE_ghgs.nc'), decode_times=False)

# ITRACE: data path for iTRACE
class iTRACE:
    def __init__(self, var, project_name='iTRACE'):
        self.var = var
        self.project_name = project_name
        self.iTRACE_flag = False
        self.OCN_VAR = ['TEMP', 'SALT', 'VVEL', 'UVEL', 'N_HEAT']
        if self.project_name == 'iTRACE':
            self.DATA_PATH = os.environ['iTRACE_DATA']
        elif self.project_name == 'TRACE':
            self.DATA_PATH = os.environ['TRACE_DATA']
        elif self.project_name == 'LGM2CO2':
            self.DATA_PATH = os.environ['LGM2CO2_DATA']
        else:
            self.DATA_PATH = os.environ['CESM_DATA']
        
    
    def _extend(self, multilevellist):
        result = []
        ext = result.extend
        for ml in multilevellist:
            ext(ml)
        
        return result
            
    def get_path(self):
        import glob
        varlist, component = self.get_varlist()
        fl = []
        for v in varlist:
            if component == 'atm':
                path = os.path.join(self.DATA_PATH, 'atm/ANN')      
            elif component == 'ocn':
                path = os.path.join(self.DATA_PATH, 'ocn/ANN')
            else:
                pass
            temp = glob.glob(path + '/*.' + v + '.*.nc')
            fl.append(temp)
        
        fl = self._extend(fl)
        # get subsets 
        ico = [f for f in fl if 'ice_orb.' in f or 'ico.' in f]
        ice = [f for f in fl if 'itrace.03' in f or 'ice.' in f]
        igo = [f for f in fl if 'ice_ghg_orb.' in f or 'igo.' in f]
        igom = [f for f in fl if 'ice_ghg_orb_mwtr.' in f or 'igom.' in f]

        # check for iTRACE, other dataset would take effect
        if ico and ice and igo: # need to add igom later
            fl = dict(ice=ice, ico=ico, igo=igo, igom=igom)
            self.iTRACE_flag = True
            return fl
        else:
            return fl

            
    def get_varlist(self):

        # if self.var in SETS.keys():
        #     varlist = SETS[self.var]
        #     component = COMP[self.var]
        # else:
        #     pass

        if self.var == 'precp':
            varlist = ['PRECC', 'PRECL']
            component = 'atm'
        elif self.var == 'd18op':
            varlist = ['PRECRC_H216Or', 'PRECSC_H216Os', 'PRECRL_H216OR', 'PRECSL_H216OS',
                       'PRECRC_H218Or', 'PRECSC_H218Os', 'PRECRL_H218OR', 'PRECSL_H218OS']
            component = 'atm'
        elif self.var == 'flux':
            varlist = ['FLNT', 'FSNT', 'LHFLX', 'SHFLX', 'FSNS', 'FLNS', 'LANDFRAC', 'ICEFRAC']
            component = 'atm'
        elif self.var == 'MOC':
            varlist = ['MOC']
            component = 'ocn'
        elif self.var == 'ocn_heat':
            varlist = ['N_HEAT', 'SHF', 'ADVT', 'ADVT_ISOP', 'ADVT_SUBM', 'HDIFT']
            component = 'ocn'
        else:
            if len(self.var) > 1 and isinstance(self.var, list):
                raise ValueError('Var set is not supported yet.')
            else:
                varlist = self.var.split()
                component = 'atm'
                if self.var in self.OCN_VAR: # modify it later
                    component = 'ocn'
                           
        return varlist, component
        
    
    def open_data(self, **kwargs):
        
        data = self.get_path()
        if self.iTRACE_flag:
            ico = xr.open_mfdataset(data['ico'], **kwargs)
            ice = xr.open_mfdataset(data['ice'], **kwargs)
            igo = xr.open_mfdataset(data['igo'], **kwargs)
            igom = xr.open_mfdataset(data['igom'], **kwargs)
            return ice, ico, igo, igom
        else:
            if len(data) > 1:
                return xr.open_mfdataset(data, **kwargs)
            else:
                return xr.open_dataset(data[0], **kwargs)
