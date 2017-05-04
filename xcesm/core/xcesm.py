#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 13 22:17:01 2017

@author: Yefee
"""

from __future__ import absolute_import

import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
from . import utils as utl
from ..config import cesmconstant as cc

@xr.register_dataset_accessor('cam')
class CAMDiagnosis(object):
    def __init__(self, xarray_obj):
        self._obj = xarray_obj


    # precipation
    @property
    def precp(self):
        try:
            precc = self._obj.PRECC
            precl = self._obj.PRECL
            precp = (precc + precl) * cc.sday * cc.rhofw # convert to mm/day
            precp.name = 'precp'
        except:
            raise ValueError('object has no PRECC.')
        return precp

    # d18op
    @property
    def d18op(self):
        '''
        compute d18O precp
        '''
        try:
            p16 = self._obj.PRECRC_H216Or + self._obj.PRECSC_H216Os + \
            self._obj.PRECRL_H216OR + self._obj.PRECSL_H216OS

            p18 = self._obj.PRECRC_H218Or + self._obj.PRECSC_H218Os + \
            self._obj.PRECRL_H218OR + self._obj.PRECSL_H218OS

            p16.values[p16.values < 1e-50] = np.nan
            d18op = (p18 / p16 -1)*1000
            d18op.name = 'd18op'
        except:
            raise ValueError('object has no PRECRC_H216Or.')
        return d18op


    def _compute_heat_transport(self, dsarray, method):

        from scipy import integrate
        '''
        compute heat transport using surface(toa,sfc) flux
        '''
        lat_rad = np.deg2rad(dsarray.lat)
        coslat = np.cos(lat_rad)
        field = coslat * dsarray

        if method is "Flux_adjusted":
            field = field - field.mean("lat")
            print("The heat transport is computed by Flux adjestment.")
        elif method is "Flux":
            print("The heat transport is computed by Flux.")
        elif method is "Dynamic":
            print("The heat transport is computed by dynamic method.")
            raise ValueError("Dynamic method has not been implimented.")
        else:
            raise ValueError("Method is not supported.")


        try:
            latax = field.get_axis_num('lat')
        except:
            raise ValueError('No lat coordinate!')

        integral = integrate.cumtrapz(field, x=lat_rad, initial=0., axis=latax)


        transport = 1e-15 * 2 * np.math.pi * integral * cc.rearth **2  # unit in PW

        if isinstance(field, xr.DataArray):
            result = field.copy()
            result.values = transport
        return result

    # heat transport
#    @property
    def planet_heat_transport(self, method="Flux"):

        '''
        compute heat transport using surface(toa,sfc) flux
        '''

        OLR = self._obj.FLNT.mean('lon')
        ASR = self._obj.FSNT.mean('lon')
        Rtoa = ASR - OLR  # net downwelling radiation
#        LHF = self._obj.LHFLX.mean('lon')
#        SHF = self._obj.SHFLX.mean('lon')
#        LWsfc = self._obj.FLNS.mean('lon')
#        SWsfc = -self._obj.FSNS.mean('lon')
#
##        LHF = self._obj.LHFLXOI.mean('lon')
##        SHF = self._obj.SHFLXOI.mean('lon')
#
##        LHF = self._obj.LHFLX.where(self._obj.FSNSOI >0)
##        LHF.values = np.nan_to_num(LHF.values)
##        LHF = LHF.mean('lon')
##
###        sum('lon') / self._obj.LHFLX.shape[self._obj.LHFLX.get_axis_num('lon')]
##        SHF = self._obj.SHFLX.where(self._obj.FSNSOI >0)
##        SHF.values = np.nan_to_num(SHF.values)
##        SHF = SHF.mean('lon')
##        .sum('lon') / self._obj.SHFLX.shape[self._obj.SHFLX.get_axis_num('lon')]
#
#
##        LWsfc = self._obj.FLNSOI.mean('lon')
##        SWsfc = -self._obj.FSNSOI.mean('lon')
#        #  energy flux due to snowfall
##        SnowFlux = (self._obj.PRECSC.mean('lon') + self._obj.PRECSL.mean('lon')) * cc.rhofw * cc.latice
#        SurfaceRadiation = LWsfc + SWsfc  # net upward radiation from surface
##        SurfaceHeatFlux = SurfaceRadiation + LHF + SHF + SnowFlux  # net upward surface heat flux
#        SurfaceHeatFlux = SurfaceRadiation + LHF + SHF  # net upward surface heat flux
#        Fatmin = Rtoa + SurfaceHeatFlux  # net heat flux in to atmosphere

#        AHT = self._compute_heat_transport(Fatmin, method)
        PHT = self._compute_heat_transport(Rtoa, method)
#        OHT = PHT - AHT
#        return PHT, AHT, OHT
        return PHT








@xr.register_dataset_accessor('pop')
class POPDiagnosis(object):
    def __init__(self, xarray_obj):
        self._obj = xarray_obj


    # amoc
    @property
    def amoc(self):
        try:
            moc = self._obj.MOC.isel(transport_reg=1,moc_comp=0).copy()
            moc.values[np.abs(moc.values) < 1e-6] = np.nan
            # amoc area
            if moc.moc_z[-1] > 1e5:
                z_bound = moc.moc_z[(moc.moc_z > 2e4) & (moc.moc_z < 5e5)] #cm
            else:
                z_bound = moc.moc_z[(moc.moc_z > 2e2) & (moc.moc_z < 5e3)] #m
            lat_bound = moc.lat_aux_grid[
                        (moc.lat_aux_grid > 10) & (moc.lat_aux_grid < 60)]
            if "time" in moc.dims:
                amoc = moc.sel(moc_z=z_bound).sel(lat_aux_grid=lat_bound).groupby('time').max()
            else:
                amoc = moc.sel(moc_z=z_bound).sel(lat_aux_grid=lat_bound).max()
        except:
            raise ValueError('object has no MOC.')
        return amoc

    @property
    def ocnreg(self):
        return utl.region()

    # convert depth unit to m
    def chdep(self):
        if self._obj.z_t[-1] > 1e5:
            self._obj['z_t'] /= 1e2
        else:
            pass
        return self._obj

    # compute ocean heat transport
    def ocn_heat_transport(self, dlat=1, grid='g16'):

        from scipy import integrate
        # check time dimension
        if 'time' in self._obj.dims:
            flux = self._obj.SHF.mean('time')
        else:
            flux = self._obj.SHF

        area = dict(g16=utl.tarea_g16, g35=utl.tarea_g35, g37=utl.tarea_g37)
        flux_area = flux * area[grid] * 1e-4 # convert to m2
        #
        lat_bins = np.arange(-90,91,dlat)
        lat = np.arange(-89.5,90,dlat)

        if 'TLAT' in flux_area.coords.keys():
            flux_lat = flux_area.groupby_bins('TLAT', lat_bins, labels = lat).sum()
            latax = flux_lat.get_axis_num('TLAT_bins')
        elif 'ULAT' in flux_area.coords.keys():
            flux_lat = flux_area.groupby_bins('ULAT', lat_bins, labels = lat).sum()
            latax = flux_lat.get_axis_num('ULAT_bins')
        flux_lat.values = flux_lat - flux_lat.mean() # remove bias
        flux_lat.values = np.nan_to_num(flux_lat.values)
        integral = integrate.cumtrapz(flux_lat, x=None, initial=0., axis=latax)
        OHT = flux_lat.copy()
        OHT.values = integral *1e-15
        return OHT



@xr.register_dataarray_accessor('utils')
class Utilities(object):
    def __init__(self, xarray_obj):
        self._obj = xarray_obj

    # regrid pop variables
    def regrid(self, dlon=1, dlat=1, grid_style='T'):
        import pyresample

        dims = self._obj.dims
        shape = self._obj.shape
        temp = self._obj.values
        temp = temp.reshape(-1,shape[-2],shape[-1])   # this requires time and z_t are at the first two axises
        temp = temp.transpose(1,2,0)    #lat, lon rightmost

        if grid_style is 'T':
            lon_curv = self._obj.TLONG.values
            lat_curv = self._obj.TLAT.values
        elif grid_style is 'U':
            lon_curv = self._obj.ULONG.values
            lat_curv = self._obj.ULAT.values

        # set lon to -180 to 180
        lon_curv[lon_curv>180] = lon_curv[lon_curv>180] - 360

        # targit grid
        lon = np.arange(-180.,181,dlon)
        lat = np.arange(-90.,91,dlat)
        lon_lin, lat_lin = np.meshgrid(lon,lat)
        lon_lin = pyresample.utils.wrap_longitudes(lon_lin)
        #define two grid systems
        orig_def = pyresample.geometry.SwathDefinition(lons=lon_curv, lats=lat_curv)
        targ_def = pyresample.geometry.SwathDefinition(lons=lon_lin, lats=lat_lin)
        rgd_data = pyresample.kd_tree.resample_nearest(orig_def, temp,
        targ_def, radius_of_influence=1000000*np.sqrt(dlon**2), fill_value=np.nan)

        rgd_data = rgd_data.transpose(2,0,1) #reshape back

        if len(dims) > 3:
            rgd_data = rgd_data.reshape(shape[0],shape[1],len(lat), len(lon))
            return xr.DataArray(rgd_data, coords=[self._obj[dims[0]], self._obj[dims[1]], lat,lon],
                                dims=[dims[0], dims[1], 'lat', 'lon'])

        elif len(dims) > 2:
            rgd_data = rgd_data.reshape(shape[0],len(lat), len(lon))
            return xr.DataArray(rgd_data, coords=[self._obj[dims[0]], lat,lon],
                                dims=[dims[0], 'lat', 'lon'])

        elif len(dims) > 1:
            rgd_data = rgd_data.squeeze()
            return xr.DataArray(rgd_data, coords=[lat,lon], dims=['lat', 'lon'])

        else:
            raise ValueError('Dataarray has more than 4 dimensions.')

    def globalmean(self,method='C'):

        lonmn = self._obj.mean('lon')
        lat_rad = xr.ufuncs.deg2rad(self._obj.lat)
        lat_cos = np.cos(lat_rad)
        total = lonmn * lat_cos
        if method == 'C':
            return total.sum("lat") / lat_cos.sum() - cc.tkfrz
        elif method == 'K':
            return total.sum("lat") / lat_cos.sum()
        else:
            raise ValueError('method not supported, use K or C instead.')

    def gbmeanpop(self, grid='g16'):

        if grid == 'g16':
            return self._obj
        if self._obj.size > 1e5:
            return self._obj / utl.tarea_g16.sum()

    def gbvolmean(self, grid='g16'):

        if grid == 'g16':
            vol = utl.dz_g16 * utl.tarea_g16
            total = self._obj * vol

        elif grid == 'g35':
            vol = utl.dz_g35 * utl.tarea_g35
            total = self._obj * vol
        elif grid == 'g37':
            vol = utl.dz_g37 * utl.tarea_g37
            total = self._obj * vol
        else:
            raise ValueError('Grid is not suppported.')

        if 'time' in self._obj.dims:
            output = total.groupby('time').sum() / vol.sum()
        else:
            output = total.sum() / vol.sum()

        output.name = self._obj.name
        return output

    def zonalmean(self):
        return self._obj.mean('lon')

    def selloc(self,loc='green_land', grid_method='regular'):

        if grid_method == 'regular':
            lat = self._obj.lat
            lon = self._obj.lon
        elif grid_method == 'T':
            lat = self._obj.TLAT
            lon = self._obj.TLONG
        elif grid_method == 'U':
            lat = self._obj.ULAT
            lon = self._obj.ULONG

#        if lon.max() > 180:
#            lon = lon[lon>180] - 180
        # later shall be wrapped into utils module
        loc = utl.locations[loc]
        return self._obj.where((lat > loc[0]) & (lat < loc[1]) & (lon > loc[2])
                               & (lon < loc[3]))
#        if loc == 'green_land':
#            return self._obj.where((lat > 72) & (lat < 73) & (lon > 321)
#                & (lon < 323))
#        elif loc == 'Brazil':
#            return self._obj.where((lat > -25) & (lat < -15) & (lon > 290)
#                & (lon < 310))
#        elif loc == 'Hulu':
#            return self._obj.where((lat > 30.5) & (lat < 33.5) & (lon > 117.5)
#                & (lon < 120.5))
#        else:
#            raise ValueError('Loc is unsupported.')
    def _selbasin(self,region='Atlantic'):
        basin = utl.region()
        return self._obj.where(basin[region])

    def Atlantic(self):
        return self._selbasin()

    def Pacific(self):
        return self._selbasin(region='Pacific')

    def Pacific_LGM(self):
        return self._selbasin(region='Pacific_LGM')

    def Southern_Ocn(self):
        return self._selbasin(region='SouthernOcn')


    # compute ocean heat transport
    def ocn_heat_transport(self, dlat=1, grid='g16'):

        from scipy import integrate
        # check time dimension
        if 'time' in self._obj.dims:
            flux = self._obj.mean('time')
        else:
            flux = self._obj

        area = dict(g16=utl.tarea_g16, g35=utl.tarea_g35, g37=utl.tarea_g37)
        flux_area = flux * area[grid] * 1e-4 # convert to m2
        #
        lat_bins = np.arange(-90,91,dlat)
        lat = np.arange(-89.5,90,dlat)

        if 'TLAT' in flux_area.coords.keys():
            flux_lat = flux_area.groupby_bins('TLAT', lat_bins, labels = lat).sum()
            latax = flux_lat.get_axis_num('TLAT_bins')
        elif 'ULAT' in flux_area.coords.keys():
            flux_lat = flux_area.groupby_bins('ULAT', lat_bins, labels = lat).sum()
            latax = flux_lat.get_axis_num('ULAT_bins')
        flux_lat.values = flux_lat - flux_lat.mean() # remove bias
        flux_lat.values = np.nan_to_num(flux_lat.values)
        integral = integrate.cumtrapz(flux_lat, x=None, initial=0., axis=latax)
        OHT = flux_lat.copy()
        OHT.values = integral *1e-15
        return OHT

    def interp_lat(self, dlat=1):
        import re
        from scipy.interpolate import interp1d
        data = self._obj
        coords_name = list(data.dims)
        lat = []
        name = []
        for n in coords_name:
            name.append(n)
            if re.search('lat', n, re.IGNORECASE) is not None:
                lat.append(n)

        if len(lat) > 1:
            raise ValueError("datarray has more than one lat dim.")
        else:
            lat = lat.pop()

        latax = data.get_axis_num(lat)
        lat_out = np.arange(-89,90,dlat)
        fun = interp1d(data[lat], data.values, axis=latax, fill_value='extrapolate')
        data_out = fun(lat_out)

        # reconstruct it to dataarray
        name.pop(latax)
        coords_out = []
        dim = []
        for n in name:
            coords_out.append(data[n])
            dim.append(n)

        coords_out.insert(latax, lat_out)
        dim.insert(latax, 'lat')
        output = xr.DataArray(data_out, coords=coords_out, dims=dim)


        return output

    def quickmap(self, ax=None, central_longitude=180, cmap='BlueDarkRed18', **kwargs):

        import cartopy.crs as ccrs
        from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
        import nclcmaps

        if central_longitude == 180:
            xticks = [0, 60, 120, 180, 240, 300, 359.99]
        elif central_longitude == 0:
            xticks = [-180, -120, -60, 0, 60, 120, 180]
        else:
            central_longitude=180
            xticks = [0, 60, 120, 180, 240, 300, 359.99]
            print("didn't explicitly give center_lat, use 180 as defalut.")


        if ax is None:

            ax = plt.axes(projection=ccrs.PlateCarree(central_longitude=central_longitude))
#            ax = plt.axes(projection=ccrs.Orthographic(-80, 35))

        cmaps = nclcmaps.cmaps(cmap)
        self._obj.plot(ax=ax,cmap=cmaps,transform=ccrs.PlateCarree(), infer_intervals=True,
                       cbar_kwargs={'orientation': 'horizontal',
                                    'fraction':0.09,
                                    'aspect':15}, **kwargs)

        #set other properties
        ax.set_global()
        ax.coastlines(linewidth=0.6)
        ax.set_xticks(xticks, crs=ccrs.PlateCarree())
        ax.set_yticks([-90, -60, -30, 0, 30, 60, 90], crs=ccrs.PlateCarree())
        lon_formatter = LongitudeFormatter(zero_direction_label=True,
                                           number_format='.0f')
        lat_formatter = LatitudeFormatter()
        ax.xaxis.set_major_formatter(lon_formatter)
        ax.yaxis.set_major_formatter(lat_formatter)
        ax.set_xlabel('')
        ax.set_ylabel('')
        return ax


