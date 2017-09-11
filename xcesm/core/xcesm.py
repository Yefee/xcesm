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



    def compute_heat_transport(self, dsarray, method):

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
        return result.T

    # heat transport
#    @property
    def planet_heat_transport(self, method="Flux"):

        '''
        compute heat transport using surface(toa,sfc) flux
        '''

        OLR = self._obj.FLNT.mean('lon')
        ASR = self._obj.FSNT.mean('lon')
        Rtoa = ASR - OLR  # net downwelling radiation

        try:
            # this block use atm flux to infer ocean heat trnasport but, use ocn output shall be more accurate
            # make sea mask from landfrac and icefrac
            lnd = self._obj.LANDFRAC
            ice = self._obj.ICEFRAC
            mask = 1- (lnd + ice) * 0.5
            LHF = self._obj.LHFLX * mask
            LHF = LHF.mean('lon')
            SHF = self._obj.SHFLX * mask
            SHF = SHF.mean('lon')

            LWsfc = self._obj.FLNS * mask
            LWsfc = LWsfc.mean('lon')
            SWsfc = -self._obj.FSNS * mask
            SWsfc = SWsfc.mean('lon')

            SurfaceRadiation = LWsfc + SWsfc  # net upward radiation from surface
        #        SurfaceHeatFlux = SurfaceRadiation + LHF + SHF + SnowFlux  # net upward surface heat flux
            SurfaceHeatFlux = SurfaceRadiation + LHF + SHF  # net upward surface heat flux
            Fatmin = Rtoa + SurfaceHeatFlux  # net heat flux in to atmosphere

            AHT = self.compute_heat_transport(Fatmin, method)
            PHT = self.compute_heat_transport(Rtoa, method)
            OHT = PHT - AHT
        except:
            PHT = self.compute_heat_transport(Rtoa, method)
            AHT = None
            OHT = None


#        return PHT, AHT, OHT
        return PHT, AHT, OHT



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
                        (moc.lat_aux_grid > 40) & (moc.lat_aux_grid < 80)]
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

    @property
    def path(self):
        return self._obj.PA_P / self._obj.TH_P

    # convert depth unit to m
    def chdep(self):
        if self._obj.z_t[-1] > 1e5:
            self._obj['z_t'] /= 1e2
        else:
            pass
        return self._obj

    def _selbasin(self,region='Atlantic'):
        basin = utl.ocean_region()
        return self._obj.where(basin[region])

    def Atlantic(self):
        return self._selbasin(region='Atlantic')

    def Arc_Atlantic(self):
        return self._selbasin(region='Arc_Atlantic')

    def Pacific(self):
        return self._selbasin(region='Pacific')

    def Indo_Pacific(self):
        return self._selbasin(region='Indo_Pacific')

    def Pacific_LGM(self):
        return self._selbasin(region='Pacific_LGM')

    def Southern_Ocn(self):
        return self._selbasin(region='SouthernOcn')

    def North_Atlantic(self):
        return self._selbasin(region='North_Atlantic')

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


    def mass_streamfun(self, dlat = 0.6, dlon = 0.1, OHT=False, region='global'):
        '''
        compute mass stream function in theta coordinates.
        reference to Ferrari and Ferreira 2011.
        '''
        dz = utl.dz_g16 * 1e-2 #convert to m
        angle = utl.angle_g16
        angle['ULONG'] = self._obj.ULONG # fix Ulong lost bug

        # meridional velocity
        VVEL = (self._obj.UVEL * np.sin(angle) + self._obj.VVEL * np.cos(angle)) * 1e-2 # convert to m

        # check region
        if region=='Global':
            T = self._obj.TEMP
        elif region == 'Indo_Pacific':
            VVEL = VVEL.utils.Indo_Pacific()
            T = self._obj.TEMP.utils.Indo_Pacific()
        elif region == 'Arc_Atlantic':
            VVEL = VVEL.utils.Arc_Atlantic()
            T = self._obj.TEMP.utils.Arc_Atlantic()
        else:
            raise ValueError('region is not supported.')

        T = T.utils.regrid(dlat=dlat, dlon=dlon)
        V = VVEL.utils.regrid(grid_style='U', dlat=dlat, dlon=dlon)

        # dxdz
        latrad = np.deg2rad(V.lat)
        lonrad = np.deg2rad(V.lon)
        dlon = lonrad[1] - lonrad[0]
        dx = cc.rearth * np.cos(latrad) * dlon # unit in m

        dzdx = dz * dx
        work = V * dzdx
#        work = work.fillna(0) # fill nan as 0
        Tmin = np.floor(T.min())
        Tmax = np.round(T.max())

        y = 0   # lat index
        k = 0   # theta index
        dt = 0.5    # theta resolution
        temp_range = np.arange(Tmin,Tmax,dt)
        Psi = np.zeros([len(work.lat),len(temp_range)])
        for t in temp_range:
            y = 0
            for l in work.lat:
                work1 = work.sel(lat=l).values
                Tsel = T.sel(lat=l).values
                if t <= np.nanmax(Tsel):
                    Psi[y, k] = np.nansum(work1[(Tsel>=np.nanmin(Tsel)) & (Tsel<=t)])
                else:
                    Psi[y, k] = np.NaN
                y += 1
            k += 1

        Psi = Psi * 1e-6 # convert to Sv
        Psi[Psi==0] = np.NaN
        Psi = -xr.DataArray(Psi, coords={'lat':work.lat, 'theta': temp_range},
                                dims=['lat', 'theta'])

        # smooth the stream function to remove noise
        Psi = Psi.rolling(lat=11, center=True).mean()
        Psi = Psi.rolling(theta=3, center=True).mean()

        if OHT:
            from scipy import integrate
            Psim3 = Psi * 1e6 # to m3/s
            Psim3 = Psim3.fillna(0)
            Psim3HT = Psim3 * cc.rhosw * cc.cpsw * 1e-15 # to PW
            theta_ax = Psim3HT.get_axis_num('theta')
            integral = integrate.cumtrapz(Psim3HT, x=Psim3HT.theta, initial=0., axis=theta_ax)
            OHT = xr.DataArray(integral, coords={'lat':Psi.lat, 'theta': temp_range},
                                        dims=['lat', 'theta'])
            return Psi.T, OHT.T
        else:
            return Psi.T


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
        lon = np.arange(-180.,180.01,dlon)
        lat = np.arange(-90.,89.999,dlat)
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

    def meridionalmean(self):

        lat_rad = np.deg2rad(self._obj.lat)
        coslat = np.cos(lat_rad)
        field = coslat * self._obj

        return field.sum() / coslat.sum()

    def selloc(self,loc='Green_land', grid_method='regular'):

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
                               & (lon < loc[3]), drop=True)

    def _selbasin(self,region='Atlantic'):
        basin = utl.ocean_region()
        return self._obj.where(basin[region])

    def Atlantic(self):
        return self._selbasin(region='Atlantic')

    def Arc_Atlantic(self):
        return self._selbasin(region='Arc_Atlantic')

    def Pacific(self):
        return self._selbasin(region='Pacific')

    def Indo_Pacific(self):
        return self._selbasin(region='Indo_Pacific')

    def Pacific_LGM(self):
        return self._selbasin(region='Pacific_LGM')

    def Southern_Ocn(self):
        return self._selbasin(region='SouthernOcn')

    def North_Atlantic(self):
        return self._selbasin(region='North_Atlantic')


    # compute ocean heat transport
    def ocn_heat_transport(self, dlat=1, grid='g16', method='Flux_adjusted', lat_bd=90):

        from scipy import integrate

        flux = self._obj
        area = dict(g16=utl.tarea_g16, g35=utl.tarea_g35, g37=utl.tarea_g37)
        flux_area = flux.copy()
        flux_area.values = flux * area[grid] * 1e-4 # convert to m2

        lat_bins = np.arange(-90,91,dlat)
        lat = np.arange(-89.5,90,dlat)


        if 'TLAT' in flux_area.coords.keys():
            flux_lat = flux_area.groupby_bins('TLAT', lat_bins, labels = lat).sum('stacked_nlat_nlon')
            latax = flux_lat.get_axis_num('TLAT_bins')
        elif 'ULAT' in flux_area.coords.keys():
            flux_area = flux_area.rename({"ULAT":"TLAT"})
            flux_lat = flux_area.groupby_bins('TLAT', lat_bins, labels = lat).sum('stacked_nlat_nlon')
            latax = flux_lat.get_axis_num('TLAT_bins')

        TLAT_bins = flux_lat.TLAT_bins
        if method == "Flux_adjusted":

            flux_lat = flux_lat.where(TLAT_bins < lat_bd) # north bound
            flat_ave = flux_lat.mean('TLAT_bins')
            flux_lat.values = flux_lat - flat_ave # remove bias
            flux_lat = flux_lat.fillna(0)
            print("The ocean heat trasnport is computed by Flux adjustment.")

        elif method == "Flux":
            flux_lat = flux_lat.fillna(0)
            print("The ocean heat trasnport is computed by original flux.")
        else:
            raise ValueError("method is not suppoprted.")

        flux_lat.values = -np.flip(flux_lat.values, latax)   # integrate from north pole
        integral = integrate.cumtrapz(flux_lat, x=None, initial=0., axis=latax)
        OHT = flux_lat.copy()
        OHT["TLAT_bins"] = np.flip(flux_lat.TLAT_bins.values, 0)
        OHT.values = integral *1e-15

        return OHT


    def hybrid_to_pressure(self, stride='m', P0=100000.):
        """
        Brought from darpy:https://github.com/darothen/darpy/blob/master/darpy/analysis.py
        Convert hybrid vertical coordinates to pressure coordinates
        corresponding to model sigma levels.
        Parameters
        ----------
        data : xarray.Dataset
            The dataset to inspect for computing vertical levels
        stride : str, either 'm' or 'i'
            Indicate if the field is on the model level interfaces or
            middles for referencing the correct hybrid scale coefficients
        P0 : float, default = 1000000.
            Default reference pressure in Pa, used as a fallback.
        """

        # A, B coefficients
        a = dict(i42=utl.hyai_t42, m42=utl.hyam_t42)
        b = dict(i42=utl.hybi_t42, m42=utl.hybm_t42)

        if stride == 'm':
            a = a['m42']
            b = b['m42']
        else:
            a = a['i42']
            b = b['i42']

        P0_ref = P0
        PS = self._obj  # Surface pressure field

        pres_sigma = a*P0_ref + b*PS

        return pres_sigma

    def shuffle_dim(self, dim='lev'):

        data = self._obj
        ind_lev = data.get_axis_num('lev')
        dim = list(data.dims)
        dim.pop(ind_lev)
        dim_new = ['lev'] + dim
        data = data.transpose(*dim_new)
        return data

    def interp_to_pressure(self, coord_vals, new_coord_vals, interpolation='lin'):
        """
        browwed from darpy
        tested with NCL code.
        Interpolate all columns simultaneously by iterating over
        vertical dimension of original dataset, following methodology
        used in UV-CDAT.
        Parameters
        ----------
        data : xarray.DataArray
            The data (array) of values to be interpolated
        coord_vals : xarray.DataArray
            An array containing a 3D field to be used as an alternative vertical coordinate
        new_coord_vals : iterable
            New coordinate values to inerpolate to
        reverse_coord : logical, default=False
            Indicates that the coord *increases* from index 0 to n; should be "True" when
            interpolating pressure fields in CESM
        interpolation : str
            "log" or "lin", indicating the interpolation method
        Returns
        -------
        list of xarray.DataArrays of length equivalent to that of new_coord_vals, with the
        field interpolated to each value in new_coord_vals
        """

        # Shuffle dims so that 'lev' is first for simplicity
        data = self._obj
        data_orig_dim = list(data.dims)
        data = self.shuffle_dim(data)

        coords_out = {'lev': new_coord_vals}
        for c in data.dims:
            if c == 'lev':
                continue
            coords_out[c] = data.coords[c]


        # Find the 'lev' axis for interpolating
        orig_shape = data.shape
        axis = data.get_axis_num('lev')
        n_lev = orig_shape[axis]

        n_interp = len(new_coord_vals)  # Number of interpolant levels

        data_interp_shape = [n_interp, ] + list(orig_shape[1:])
        data_new = np.zeros(data_interp_shape)

        # Shape of array at any given level
        flat_shape = coord_vals.isel(lev=0).shape

        # Loop over the interpolant levels
        for ilev in range(n_interp):

            lev = new_coord_vals[ilev]

            P_abv = np.ones(flat_shape)
            # Array on level above, below
            A_abv, A_bel = -1.*P_abv, -1.*P_abv
            # Coordinate on level above, below
            P_abv, P_bel = -1.*P_abv, -1.*P_abv

            # Mask area where coordinate == levels
            P_eq = np.ma.masked_equal(P_abv, -1)

            # Loop from the second sigma level to the last one
            for i in range(1, n_lev):

                a = np.ma.greater_equal(coord_vals.isel(lev=i), lev)
                b = np.ma.less_equal(coord_vals.isel(lev=i - 1), lev)


                # Now, if the interpolant level is between the two
                # coordinate levels, then we can use these two levels for the
                # interpolation.
                a = (a & b)

                # Coordinate on level above, below
                P_abv = np.where(a, coord_vals[i], P_abv)
                P_bel = np.where(a, coord_vals[i - 1], P_bel)
                # Array on level above, below
                A_abv = np.where(a, data[i], A_abv)
                A_bel = np.where(a, data[i-1], A_bel)

                P_eq = np.where(coord_vals[i] == lev, data[i], P_eq)

            # If no data below, set to missing value; if there is, set to
            # (interpolating) level
            P_val = np.ma.masked_where((P_bel == -1), np.ones_like(P_bel)*lev)

            # Calculate interpolation
            if interpolation == 'log':
                tl = np.log(P_val/P_bel)/np.log(P_abv/P_bel)*(A_abv - A_bel) + A_bel
            elif interpolation == 'lin':
                tl = A_bel + (P_val-P_bel)*(A_abv - A_bel)/(P_abv - P_bel)
            else:
                raise ValueError("Don't know how to interpolate '{}'".format(interpolation))
            tl.fill_value = np.nan

            # Copy into result array, masking where values are missing
            # because of bad interpolation (out of bounds, etc.)
            tl[tl.mask] = np.nan
            data_new[ilev] = tl

        dataout = xr.DataArray(data_new, coords=coords_out, dims=data.dims)
        dataout = dataout.transpose(*data_orig_dim)
        return dataout


    def mass_streamfun(self):

        from scipy import integrate

        data = self._obj
#        lonlen = len(data.lon)
        if 'lon' in data.dims:
            data = data.fillna(0).mean('lon')
        levax = data.get_axis_num('lev')
        stream = integrate.cumtrapz(data * np.cos(np.deg2rad(data.lat)), x=data.lev * 1e2, initial=0., axis=levax)
        stream = stream * 2 * np.pi  / cc.g * cc.rearth * 1e-9
        stream = xr.DataArray(stream, coords=data.coords, dims=data.dims)

        return stream

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

        # get attributes back
        if data.name is not None:
            output = output.rename(data.name)

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


