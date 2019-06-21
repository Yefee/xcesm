import os
import xarray as xr

# will append when needed
locations = {'Green_land': [72, 73, 321, 323],
            'Green_land_s': [72, 75, 310, 320],
            'Brazil': [-25, -15, 290, 310],
            'Hulu': [30.5, 33.5, 117.5, 120.5],
            'Sanbao': [30.5, 33.5, 108.5, 112.5],
            'DomeC': [-76.0, -74.0, 122.0, 124.0],
            'Bittoo':[29.5, 32.5, 78.5, 81.5],
            'Mawmluh':[25.2-1.5, 25.2+1.5, 91.4-1.5, 91.4+1.5],
            'Nino34': [-5,5,190,240]}


SETS = {'precp': ['PRECC', 'PRECL'],
        'd18op': ['PRECRC_H216Or', 'PRECSC_H216Os', 'PRECRL_H216OR', 'PRECSL_H216OS',
                  'PRECRC_H218Or', 'PRECSC_H218Os', 'PRECRL_H218OR', 'PRECSL_H218OS'],
        'dDp': ['PRECRC_H216Or', 'PRECSC_H216Os', 'PRECRL_H216OR', 'PRECSL_H216OS',
                  'PRECRC_HDOr', 'PRECSC_HDOs', 'PRECRL_HDOR', 'PRECSL_HDOS'],
        'd18ov': ['H216OV','H218OV'],
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
huw_g16 = xr.open_dataarray(DATA_PATH + 'HUW_gx1v6.nc')
hus_g16 = xr.open_dataarray(DATA_PATH + 'HUS_gx1v6.nc')
dxu_g16 = xr.open_dataarray(DATA_PATH + 'DXU_gx1v6.nc')
dyu_g16 = xr.open_dataarray(DATA_PATH + 'DYU_gx1v6.nc')
dxt_g16 = xr.open_dataarray(DATA_PATH + 'DXT_gx1v6.nc')
dyt_g16 = xr.open_dataarray(DATA_PATH + 'DYT_gx1v6.nc')
angle_g16 = xr.open_dataarray(DATA_PATH + 'ANGLE_gx1v6.nc')
angle_g35 = xr.open_dataarray(DATA_PATH + 'ANGLE_gx3v5.nc')
dz_g16 = xr.open_dataarray(DATA_PATH + 'DZ_gx1v6.nc')
dz_g35 = xr.open_dataarray(DATA_PATH + 'DZ_gx3v5.nc')
kmt_g16 = xr.open_dataarray(DATA_PATH + 'KMT_gx1v6.nc') - 1 # land is -1, ocean starts from 0
kmt_cube_g16 = xr.open_dataarray(DATA_PATH + 'KMT_CUBE_gx1v6.nc') # 3-D mask for kmt
kmt_cube_g35 = xr.open_dataarray(DATA_PATH + 'KMT_CUBE_gx3v5.nc') # 3-D mask for kmt

# CCSM4
hyai_t42 = xr.open_dataarray(DATA_PATH + 'hyai_t42.nc')
hyam_t42 = xr.open_dataarray(DATA_PATH + 'hyam_t42.nc')
hybi_t42 = xr.open_dataarray(DATA_PATH + 'hybi_t42.nc')
hybm_t42 = xr.open_dataarray(DATA_PATH + 'hybm_t42.nc')

# CESM1
hyai_cesm1_t42 = xr.open_dataarray(DATA_PATH + 'hyai_cesm1_t42.nc')
hyam_cesm1_t42 = xr.open_dataarray(DATA_PATH + 'hyam_cesm1_t42.nc')
hybi_cesm1_t42 = xr.open_dataarray(DATA_PATH + 'hybi_cesm1_t42.nc')
hybm_cesm1_t42 = xr.open_dataarray(DATA_PATH + 'hybm_cesm1_t42.nc')

landfrac = xr.open_dataarray(DATA_PATH + 'cam_landfrac.nc')

# oxygen isotope data
hulu = xr.open_dataarray(DATA_PATH + 'hulu_d18o.nc')
hzz1 = xr.open_dataarray(DATA_PATH + 'hzz1_d18o.nc')
hzz2 = xr.open_dataarray(DATA_PATH + 'hzz2_d18o.nc')
sanbao = xr.open_dataarray(DATA_PATH + 'sanbao_d18o.nc')
gisp2 = xr.open_dataarray(DATA_PATH + 'gisp2_d18o.nc')

# Pa/Th data
path = xr.open_dataarray(DATA_PATH + 'path_Bermuda.nc')
path_MD95 = xr.open_dataarray(DATA_PATH + 'path_MD95.nc')
path_SU81 = xr.open_dataarray(DATA_PATH + 'path_SU81.nc')
path_SU90 = xr.open_dataarray(DATA_PATH + 'path_SU90.nc')

# Dome C temp reconstruction
domec = xr.open_dataset(DATA_PATH + 'domeC_dD_temp.nc')

# sea level from melt water 
sea_level = xr.open_dataarray(DATA_PATH + 'sea_level_from_mwr.nc')


# ocean basin for pop output
def ocean_region(grid='gx1v6'):

    if grid == 'gx1v6':
        rg = mask_g16
    elif grid == 'gx3v7':
        rg = mask_g37
    elif grid == 'gx3v5':
        rg = mask_g35
    else: 
        raise ValueError('The gird is not supported.')

    regions = {'Atlantic': (rg==6) | ((rg==1)&((rg.TLONG>=300) | (rg.TLONG<=20))),
               'Pacific':  (rg==2) | ((rg==1)&((rg.TLONG>=150)& (rg.TLONG<=290))),
               'Indo_Pacific': (rg==2) | ((rg==1)&((rg.TLONG>=25) & (rg.TLONG<=290))) | (rg==3),
               'Arc_Atlantic': ((rg>5) & (rg!=7)) | ((rg==1)&( (rg.TLONG>=300) | (rg.TLONG<=20))),
               'Pacific_LGM': ((rg==2) & (rg.TLAT<=40) | (rg.TLAT>=55) & (rg.TLONG >=125) |
                (rg.TLONG>=140)) | ((rg==1)&((rg.TLONG>=150)& (rg.TLONG<=290))),
               'SouthernOcn': rg==1,
               'North_Atlantic': (((rg.TLAT<=70) & (rg.TLAT>=50) & ((rg.TLONG <=20) | (rg.TLONG >=280))) & (rg > 0))}

    return regions


def open_data(var, project_name='iTRACE', **kwargs):
    return iTRACE(var, project_name).open_data(**kwargs)


class open_iTrace:

    def __init__(self, var, project_name='iTRACE', **kwargs):
        self.ice, self.ico, self.igo, self.igom = iTRACE(var, project_name).open_data(**kwargs)
        print("Data bundle has been successfully loaded!")

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
        self.OCN_VAR = ['TEMP', 'SALT', 'VVEL', 'UVEL', 'N_HEAT', 'WVEL', 
                        'VNT', 'RHO', 'MOC', 'VISOP', 'UISOP', 'VSUBM',
                        'USUBM', 'R18O']
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
        if component == 'atm':
            igofl = [f for f in igo if '.20ka.' in f and '0999' in f]
            igofl.sort()
            igom = igofl + igom
        elif component == 'ocn':
            igofl = [f for f in igo if '.20ka.' in f and ('0099' in f or '0199' in f or '0299' in f or '0399' in f or '0499' in f or '0599' in f or '0599' in f or '0699' in f or '0799' in f or '0899' in f or '0999' in f)]
            igofl.sort()
            igom = igofl + igom
        else:
            Warning("20ka to 19ka all forcing run data has not been loaded!")
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
        elif self.var == 'dDp':
            varlist = ['PRECRC_H216Or', 'PRECSC_H216Os', 'PRECRL_H216OR', 'PRECSL_H216OS',
                       'PRECRC_HDOr', 'PRECSC_HDOs', 'PRECRL_HDOR', 'PRECSL_HDOS']
            component = 'atm'
        elif self.var == 'd18ov':
            varlist = ['H216OV','H218OV']
            component = 'atm'
        elif self.var == 'flux':
            varlist = ['FLNT', 'FSNT', 'LHFLX', 'SHFLX', 'FSNS', 'FLNS', 'LANDFRAC', 'ICEFRAC']
            component = 'atm'
        elif self.var == 'flux-toa':
            varlist = ['FLNT', 'FSNT']
            component = 'atm'
        elif self.var == 'MOC':
            varlist = ['MOC']
            component = 'ocn'
        elif self.var == 'ocn_heat':
            varlist = ['SHF', 'ADVT', 'ADVT_ISOP', 'ADVT_SUBM', 'HDIFT']
            component = 'ocn'
        elif self.var == 'uvt':
            varlist = ['UVEL', 'VVEL', 'TEMP']
            component = 'ocn'
        elif self.var == 'uivit':
            varlist = ['UISOP', 'VISOP', 'TEMP']
            component = 'ocn'
        elif self.var == 'usvst':
            varlist = ['USUBM', 'VSUBM', 'TEMP']
            component = 'ocn'
        elif self.var == 'uvt-total':
            varlist = ['USUBM', 'VSUBM', 'TEMP', 'UISOP', 'VISOP','VVEL', 'UVEL']
            component = 'ocn'
        elif self.var == 'path':
            varlist = ['PA_P', 'TH_P']
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
            ico = xr.open_mfdataset(data['ico'], **kwargs).sortby('time')
            ice = xr.open_mfdataset(data['ice'], **kwargs).sortby('time')
            igo = xr.open_mfdataset(data['igo'], **kwargs).sortby('time')
            igom = xr.open_mfdataset(data['igom'], **kwargs).sortby('time')
            return ice, ico, igo, igom
        else:
            if len(data) > 1:
                return xr.open_mfdataset(data, **kwargs).sortby('time')
            else:
                return xr.open_dataset(data[0], **kwargs).sortby('time')
