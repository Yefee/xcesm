#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 25 20:42:44 2017

@author: Yefee
"""
from .share_constant import *

# sec in siderial day ~ sec
sday = SHR_CONST_SDAY

# earth rot ~ rad/sec
omega = SHR_CONST_OMEGA

# radius of earth ~ m
rearth = SHR_CONST_REARTH

# acceleration of gravity ~ m/s^2
g = SHR_CONST_G

# Stefan-Boltzmann constant ~ W/m^2/K^4
stebol = SHR_CONST_STEBOL

# Boltzmann's constant ~ J/K/molecule
boltz = SHR_CONST_BOLTZ

# Avogadro's number ~ molecules/kmole
avogad = SHR_CONST_AVOGAD

# Universal gas constant ~ J/K/kmole
rgas = SHR_CONST_RGAS

# molecular weight dry air ~ kg/kmole
mwdair = SHR_CONST_MWDAIR

# molecular weight water vapor
mwwv = SHR_CONST_MWWV

# Dry air gas constant ~ J/K/kg
rdair = SHR_CONST_RDAIR

# Water vapor gas constant ~ J/K/kg
rwv = SHR_CONST_RWV

# RWV/RDAIR - 1.0
zvir = SHR_CONST_ZVIR

# Von Karman constant
karman = SHR_CONST_KARMAN

# freezing T of fresh water ~ K (intentionally made == to TKTRIP)
tkfrz = SHR_CONST_TKFRZ

# triple point of fresh water ~ K
tktrip = SHR_CONST_TKTRIP

# density of dry air at STP   ~ kg/m^3
rhoair = SHR_CONST_RHODAIR

# density of fresh water ~ kg/m^3
rhofw = SHR_CONST_RHOFW

# density of sea water ~ kg/m^3
rhosw = SHR_CONST_RHOSW

# density of ice   ~ kg/m^3
rhoice = SHR_CONST_RHOICE

# specific heat of dry air ~ J/kg/K
cpdair = SHR_CONST_CPDAIR

# specific heat of fresh h2o ~ J/kg/K
cpfw = SHR_CONST_CPFW

# specific heat of sea h2o ~ J/kg/K
cpsw = SHR_CONST_CPSW

# specific heat of water vap ~ J/kg/K
cpwv = SHR_CONST_CPWV

# specific heat of fresh ice ~ J/kg/K
cpice = SHR_CONST_CPICE

# latent heat of fusion ~ J/kg
latice = SHR_CONST_LATICE

# latent heat of evaporation ~ J/kg
larvap = SHR_CONST_LATVAP

# latent heat of sublimation ~ J/kg
latsub = SHR_CONST_LATSUB

# ocn ref salinity (psu)
ocn_ref_sal = SHR_CONST_OCN_REF_SAL

# ice ref salinity (psu)
ice_ref_sal = SHR_CONST_ICE_REF_SAL
