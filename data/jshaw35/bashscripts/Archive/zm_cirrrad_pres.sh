#!/bin/bash

# description: compile and run cesm 1.2.2. default version

# enter analysis node and user cesm directory
# ssh analysis
# cd cesmruns

# name case
#case=e2000_bn_cirrradbg_rhmini0.90l1.05_1xco2_binprecip_1
case=e2000_bn_cirrradbg_rhmini1to1.2_1.5xco2seed18_binprecip_2
#case=f2000_bnorig_cirradbg_iceopt4_1
#case=def_2000_cirradbg_lp_1
#case=def_2000_cirrradbg_1
#case=def_nodustrad_d2.90_1k_0.50dustrefimodcode

#d2.65 d1.85 d0.50
#c0.8875

# create case
#$CESMROOT/releases/cesm1_2_2/scripts/create_newcase -case $case -compset 'F_2000_CAM5' -res f19_g16 -mach yellowstone # B_2000_CAM5, B_1850-2000_CAM5, B_RCP8.5_CAM5_CN # f05-g16
create_newcase -case $case -compset 'E_1850_CAM5' -res f19_g16 -mach omega # B_2000_CAM5, B_1850-2000_CAM5, B_RCP8.5_CAM5_CN

#E_1850_CAM5
#E_2000_CN_GLC_CISM1
#F_2000_CAM5

cd $case

./xmlchange -file env_build.xml -id CAM_CONFIG_OPTS -val '-phys cam5'

# set model start time and run length
#./xmlchange -file env_run.xml -id RUN_STARTDATE -val '2004-01-01'
./xmlchange -file env_run.xml -id REST_N -val '1'
./xmlchange -file env_run.xml -id STOP_N -val '100'
./xmlchange -file env_run.xml -id STOP_OPTION -val 'nyears'
./xmlchange -file env_run.xml -id RESUBMIT -val '0'
#./xmlchange -file env_build.xml -id CAM_CONFIG_OPTS -append -val '-cosp -offline_dyn -nlev 56'
#./xmlchange -file env_build.xml -id CAM_CONFIG_OPTS -append -val '-offline_dyn -nlev 56'
#./xmlchange -file env_run.xml -id CALENDAR -val 'GREGORIAN'
#./xmlchange -file env_build.xml -id CAM_NAMELIST_OPTS -append -val 'ncdata=&apos;/home/fas/long/zm56/scratch/cesm_input/wa4_cesm1_1_b02_geos5_2x_sim153f.cam2.i.2000-01-01-00000.nc&apos;'
#./xmlchange -file env_build.xml -id CAM_NAMELIST_OPTS -append -val 'ncdata=&apos;/home/fas/long/zm56/scratch/cesm_input/camchem_ic_2008-01-01_1.9x2.5_L56_c110118.nc&apos;'

# check env_build.xml and env_run.xml for the correct scratch dir
#  cesm 1.0.4: EXEROOT in env_build.xml, DOUT_S_ROOT in env_run.xml:  $CCSMUSER_SCRATCH -> /scratch/fas/long
#./xmlchange -file env_build.xml -id EXEROOT -val '/scratch/fas/long/$CCSMUSER/CESM/$CASE'            # cesm 1.0 (1/2)
#./xmlchange -file env_run.xml -id DOUT_S_ROOT -val '/scratch/fas/long/$CCSMUSER/CESM/Archives/$CASE' # cesm 1.0 (2/2)
#./xmlchange -file env_build.xml -id CCSMGROUP_SCRATCH -val '/scratch/fas/long'                        # cesm 1.2 (1/1)
./xmlchange -file env_build.xml -id CCSMGROUP_SCRATCH -val '/gpfs/scratch60/fas/long'                 # cesm 1.2 (1/1)

# specify co2 cycle
#./xmlchange -file env_build.xml -id CAM_CONFIG_OPTS -append -val '-co2_cycle'
#./xmlchange -file env_build.xml -id CAM_NAMELIST_OPTS -append -val 'co2flux_fuel_file="/home/fas/long/zm56/co2flux_fossil_RCP85_2005-2100-monthly_0.9x1.25_c20101013.nc"'

# add sourcemods:
#cp /home/fas/long/zm56/dustmod/cleansky/modal_aer_opt.F90 /home/fas/long/zm56/cesmruns/${case}/SourceMods/src.cam

#cp /home/fas/long/zm56/dustmod/cleansky/radiation.F90 /home/fas/long/zm56/cesmruns/${case}/SourceMods/src.cam
#cp /home/fas/long/zm56/dustmod/cloudrad_cleandryskyfull/radiation.F90 /home/fas/long/zm56/cesmruns/${case}/SourceMods/src.cam
cp /home/fas/long/zm56/dustmod/cirrrad_bg/radiation.F90 /home/fas/long/zm56/cesmruns/${case}/SourceMods/src.cam
#cp /home/fas/long/zm56/dustmod/cirrrad_bg2/radiation.F90 /home/fas/long/zm56/cesmruns/${case}/SourceMods/src.cam

#cp /home/fas/long/zm56/dustmod/MUGEFRAC_output/micro_mg_cam.F90 /home/fas/long/zm56/cesmruns/${case}/SourceMods/src.cam
#cp /home/fas/long/zm56/dustmod/cospfix/cospsimulator_intr.F90 /home/fas/long/zm56/cesmruns/${case}/SourceMods/src.cam

#cp /home/fas/long/zm56/dustmod/barahona_orig/* /home/fas/long/zm56/cesmruns/${case}/SourceMods/src.cam

#cp /home/fas/long/zm56/dustmod/barahona/* /home/fas/long/zm56/cesmruns/${case}/SourceMods/src.cam
#cp /home/fas/long/zm56/dustmod/barahona_seed/* /home/fas/long/zm56/cesmruns/${case}/SourceMods/src.cam
cp /home/fas/long/zm56/dustmod/barahona_seed18/* /home/fas/long/zm56/cesmruns/${case}/SourceMods/src.cam

#cp /home/fas/long/zm56/dustmod/barahona_hyb/* /home/fas/long/zm56/cesmruns/${case}/SourceMods/src.cam
#cp /home/fas/long/zm56/dustmod/barahona_zm/* /home/fas/long/zm56/cesmruns/${case}/SourceMods/src.cam

cp /home/fas/long/zm56/dustmod/icecldfrac_tune/cldwat2m_macro.F90 /home/fas/long/zm56/cesmruns/${case}/SourceMods/src.cam

cp /home/fas/long/zm56/dustmod/bin_precip/micro_mg_cam.F90 /home/fas/long/zm56/cesmruns/${case}/SourceMods/src.cam

#cp /home/fas/long/zm56/dustmod/mugeplus/* /home/fas/long/zm56/cesmruns/${case}/SourceMods/src.cam
#cp /home/fas/long/zm56/dustmod/isotherm_cam1.2/* /home/fas/long/zm56/cesmruns/${case}/SourceMods/src.cam

#cp /home/fas/long/zm56/dustmod/wbfweak/micro_mg1_0.F90 /home/fas/long/zm56/cesmruns/${case}/SourceMods/src.cam
#cp /home/fas/long/zm56/dustmod/epsiweak/micro_mg1_0.F90 /home/fas/long/zm56/cesmruns/${case}/SourceMods/src.cam

#cp /home/fas/long/zm56/dustmod/mpc/zm_0.01epsi/src.cam/micro_mg1_0.F90 /home/fas/long/zm56/cesmruns/${case}/SourceMods/src.cam
#cp /home/fas/long/zm56/dustmod/mpc/zm_1e9epsi/src.cam/micro_mg1_0.F90 /home/fas/long/zm56/cesmruns/${case}/SourceMods/src.cam
#cp /home/fas/long/zm56/dustmod/mpc/zm_1e-9epsi/src.cam/micro_mg1_0.F90 /home/fas/long/zm56/cesmruns/${case}/SourceMods/src.cam
#cp /home/fas/long/zm56/dustmod/mpc/zm_1e30epsi/src.cam/micro_mg1_0.F90 /home/fas/long/zm56/cesmruns/${case}/SourceMods/src.cam
#cp /home/fas/long/zm56/dustmod/mpc/zm_1e-30epsi/src.cam/micro_mg1_0.F90 /home/fas/long/zm56/cesmruns/${case}/SourceMods/src.cam

#cp /home/fas/long/zm56/dustmod/mpc/zm/src.cam/* /home/fas/long/zm56/cesmruns/${case}/SourceMods/src.cam
#cp /home/fas/long/zm56/dustmod/mpc/zm_100epsi/src.cam/* /home/fas/long/zm56/cesmruns/${case}/SourceMods/src.cam
#cp /home/fas/long/zm56/dustmod/mpc/zm_10000epsi/src.cam/* /home/fas/long/zm56/cesmruns/${case}/SourceMods/src.cam
#cp /home/fas/long/zm56/dustmod/mpc/zm_0.01epsi/src.cam/* /home/fas/long/zm56/cesmruns/${case}/SourceMods/src.cam
#cp /home/fas/long/zm56/dustmod/mpc/zm_0.0001epsi/src.cam/* /home/fas/long/zm56/cesmruns/${case}/SourceMods/src.cam
#cp /home/fas/long/zm56/dustmod/mpc/zm_1e9epsi/src.cam/* /home/fas/long/zm56/cesmruns/${case}/SourceMods/src.cam
#cp /home/fas/long/zm56/dustmod/mpc/zm_1e-9epsi/src.cam/* /home/fas/long/zm56/cesmruns/${case}/SourceMods/src.cam
#cp /home/fas/long/zm56/dustmod/mpc/zm_1e30epsi/src.cam/* /home/fas/long/zm56/cesmruns/${case}/SourceMods/src.cam
#cp /home/fas/long/zm56/dustmod/mpc/zm_1e-30epsi/src.cam/* /home/fas/long/zm56/cesmruns/${case}/SourceMods/src.cam

#cp /home/fas/long/zm56/dustmod/subgrid_emis/mods_zka_uvonly/cam/* /home/fas/long/zm56/cesmruns/${case}/SourceMods/src.cam
#cp /home/fas/long/zm56/dustmod/subgrid_emis/mods_z16defk14a14_vars_uvonly/cam/* /home/fas/long/zm56/cesmruns/${case}/SourceMods/src.cam
#cp /home/fas/long/zm56/dustmod/subgrid_emis/mods_z16defk14a14_amplifyreg_uvonly/cam/* /home/fas/long/zm56/cesmruns/${case}/SourceMods/src.cam
#cp /home/fas/long/zm56/dustmod/subgrid_emis/mods_z16defk14a14_amplifynafrica_uvonly/cam/* /home/fas/long/zm56/cesmruns/${case}/SourceMods/src.cam
#cp /home/fas/long/zm56/dustmod/subgrid_emis/mods_z16defk14a14_amplifynwafrica_uvonly/cam/* /home/fas/long/zm56/cesmruns/${case}/SourceMods/src.cam
#cp /home/fas/long/zm56/dustmod/subgrid_emis/mods_z16defk14a14_amplifymideast_uvonly/cam/* /home/fas/long/zm56/cesmruns/${case}/SourceMods/src.cam
#cp /home/fas/long/zm56/dustmod/subgrid_emis/mods_z16defk14a14_amplifyIrAfPak_uvonly/cam/* /home/fas/long/zm56/cesmruns/${case}/SourceMods/src.cam
#cp /home/fas/long/zm56/dustmod/subgrid_emis/mods_z16defk14a14_amplifyarabia_uvonly/cam/* /home/fas/long/zm56/cesmruns/${case}/SourceMods/src.cam
#cp /home/fas/long/zm56/dustmod/subgrid_emis/mods_z16defk14a14_amplifynafricaarabia_uvonly/cam/* /home/fas/long/zm56/cesmruns/${case}/SourceMods/src.cam
#cp /home/fas/long/zm56/dustmod/subgrid_emis/mods_z16defk14a14_amplifynotnafricamideast_uvonly/cam/* /home/fas/long/zm56/cesmruns/${case}/SourceMods/src.cam
#cp /home/fas/long/zm56/dustmod/subgrid_emis/mods_z16defk14a14_amplifynotnafricaarabia_uvonly/cam/* /home/fas/long/zm56/cesmruns/${case}/SourceMods/src.cam

#cp /home/fas/long/zm56/dustmod/subgrid_emis/mods_zka_uvonly/clm/* /home/fas/long/zm56/cesmruns/${case}/SourceMods/src.clm
#cp /home/fas/long/zm56/dustmod/subgrid_emis/mods_z16defk14a14_vars_uvonly/clm/* /home/fas/long/zm56/cesmruns/${case}/SourceMods/src.clm

#cp /home/fas/long/zm56/dustmod/subgrid_emis/mods_zka_uvonly/share/* /home/fas/long/zm56/cesmruns/${case}/SourceMods/src.share
#cp /home/fas/long/zm56/dustmod/subgrid_emis/mods_z16defk14a14_vars_uvonly/share/* /home/fas/long/zm56/cesmruns/${case}/SourceMods/src.share

# specify cam5
#./xmlchange -file env_build.xml -id CAM_CONFIG_OPTS -val '-phys cam5'

# add user_nl_cam file to define history files
# user_nl_cam not yet working:
# cp ../user_nl_cam .

# configure model
# cesm 1.0.4: ./configure -case
./cesm_setup

# change land model surface input
#echo "fsurdat = '/home/fas/long/zm56/dustmod/ck_surfdata_1.9x2.5_simyr2000_c091005.nc'" >> user_nl_clm

# change atm model soil erodibility input
#echo "soil_erod='/home/fas/long/zm56/dustmod/albani2014/dst_source2x2tuned_cam5merra_30jul13.nc'" >> user_nl_cam

# set carbon cycle
#cat <<TXT2 >> user_nl_cam
#&co2_cycle_nl
# co2_flag=.true.
# co2_readFlux_fuel=.true.
# co2flux_fuel_file='/home/fas/long/zm56/co2flux_fossil_RCP85_2005-2100-monthly_0.9x1.25_c20101013.nc'
#/
#TXT2

# add dust variables to history
cat <<TXT2 >> user_nl_cam
&aerosol_nl
 fincl1 ='DSTSFDRY','DSTSFMBL','DSTSFWET','dst_a1SF', 'dst_a3SF','dst_a1','dst_a3','dst_c1','dst_c3','bc_a1','bc_c1','ncl_a1','ncl_c1','ncl_a2','ncl_c2','ncl_a3','ncl_c3','num_a1','num_c1','num_a2','num_c2','num_a3','num_c3','pom_a1','pom_c1','so4_a1','so4_c1','so4_a2','so4_c2','so4_a3','so4_c3','soa_a1','soa_c1','soa_a2','soa_c2','dst_a1DDF','dst_c1DDF','dst_a3DDF','dst_c3DDF','dst_a1SFWET','dst_c1SFWET','dst_a3SFWET','dst_c3SFWET','CMEIOUT','MNUCCCO','MNUCCTO','MNUCCDO','MNUCCDOhet','BERGO','BERGSO','HOMOO','PRAO','PRCO','PRCIO','PRAIO','WSUBI','NIHF','NIDEP','NIIMM','NIMEY','NCAL','NCAI','FREQS','ADSNOW','FREQS','ANSNOW','AQSNOW','DSNOW','NSNOW','EFFICE','ICINC','ICIMR','ICWNC','ICIMRST','ICIMRCU','ICIMRTOT','MPDQ','MPDT','MPDW2V','MPDW2I','MPDW2P','MPDI2V','MPDI2W','MPDI2P','NDROPSNK','MPDICE','MPDLIQ','QCRESO','QIRESO','MSACWIO','PSACWSO','MELTO','VPRAO','VPRCO','RACAU'
/
rad_diag_1= 'A:Q:H2O','N:O2:O2', 'N:CO2:CO2','N:ozone:O3','N:N2O:N2O','N:CH4:CH4','N:CFC11:CFC11', 'N:CFC12:CFC12','M:mam3_mode1:/apps/hpc/Data/CESM/1.0/inputdata/atm/cam/physprops/mam3_mode1_rrtmg_c110318.nc','M:mam3_mode2:/apps/hpc/Data/CESM/1.0/inputdata/atm/cam/physprops/mam3_mode2_rrtmg_c110318.nc', 'M:mam3_mode3:/apps/hpc/Data/CESM/1.0/inputdata/atm/cam/physprops/mam3_mode3_rrtmg_c110318.nc'
/
&cldfrc_nl
 cldfrc_rhminl = 0.8875D0
/
&chem_inparm
 ext_frc_specifier              = 'SO2         -> /apps/hpc/Data/CESM/1.0/inputdata/atm/cam/chem/trop_mozart_aero/emis/ar5_mam3_so2_elev_2000_c090726.nc',
         'bc_a1       -> /apps/hpc/Data/CESM/1.0/inputdata/atm/cam/chem/trop_mozart_aero/emis/ar5_mam3_bc_elev_2000_c090726.nc',
         'num_a1      -> /apps/hpc/Data/CESM/1.0/inputdata/atm/cam/chem/trop_mozart_aero/emis/ar5_mam3_num_a1_elev_2000_c090726.nc',
         'num_a2      -> /apps/hpc/Data/CESM/1.0/inputdata/atm/cam/chem/trop_mozart_aero/emis/ar5_mam3_num_a2_elev_2000_c090726.nc',
         'pom_a1      -> /apps/hpc/Data/CESM/1.0/inputdata/atm/cam/chem/trop_mozart_aero/emis/ar5_mam3_oc_elev_2000_c090726.nc',
         'so4_a1      -> /apps/hpc/Data/CESM/1.0/inputdata/atm/cam/chem/trop_mozart_aero/emis/ar5_mam3_so4_a1_elev_2000_c090726.nc',
         'so4_a2      -> /apps/hpc/Data/CESM/1.0/inputdata/atm/cam/chem/trop_mozart_aero/emis/ar5_mam3_so4_a2_elev_2000_c090726.nc'
 srf_emis_specifier             = 'DMS       -> /apps/hpc/Data/CESM/1.0/inputdata/atm/cam/chem/trop_mozart_aero/emis/aerocom_mam3_dms_surf_2000_c090129.nc',
         'SO2       -> /apps/hpc/Data/CESM/1.0/inputdata/atm/cam/chem/trop_mozart_aero/emis/ar5_mam3_so2_surf_2000_c090726.nc',
         'SOAG      -> /apps/hpc/Data/CESM/1.0/inputdata/atm/cam/chem/trop_mozart_aero/emis/ar5_mam3_soag_1.5_surf_2000_c100217.nc',
         'bc_a1     -> /apps/hpc/Data/CESM/1.0/inputdata/atm/cam/chem/trop_mozart_aero/emis/ar5_mam3_bc_surf_2000_c090726.nc',
         'num_a1    -> /apps/hpc/Data/CESM/1.0/inputdata/atm/cam/chem/trop_mozart_aero/emis/ar5_mam3_num_a1_surf_2000_c090726.nc',
         'num_a2    -> /apps/hpc/Data/CESM/1.0/inputdata/atm/cam/chem/trop_mozart_aero/emis/ar5_mam3_num_a2_surf_2000_c090726.nc',
         'pom_a1    -> /apps/hpc/Data/CESM/1.0/inputdata/atm/cam/chem/trop_mozart_aero/emis/ar5_mam3_oc_surf_2000_c090726.nc',
         'so4_a1    -> /apps/hpc/Data/CESM/1.0/inputdata/atm/cam/chem/trop_mozart_aero/emis/ar5_mam3_so4_a1_surf_2000_c090726.nc',
         'so4_a2    -> /apps/hpc/Data/CESM/1.0/inputdata/atm/cam/chem/trop_mozart_aero/emis/ar5_mam3_so4_a2_surf_2000_c090726.nc'
 tracer_cnst_cycle_yr           = 2000
 tracer_cnst_file               = 'oxid_1.9x2.5_L26_1850-2005_c091123.nc'
 tracer_cnst_filelist           = 'oxid_1.9x2.5_L26_clim_list.c090805.txt'
/
&prescribed_ozone_nl
 prescribed_ozone_cycle_yr              = 2000
 prescribed_ozone_datapath              = '/apps/hpc/Data/CESM/1.0/inputdata/atm/cam/ozone'
 prescribed_ozone_file          = 'ozone_1.9x2.5_L26_2000clim_c091112.nc'
 prescribed_ozone_name          = 'O3'
 prescribed_ozone_type          = 'CYCLICAL'
/
&chem_surfvals_nl
 ch4vmr         = 1760.0e-9
 co2vmr         = 530.9e-6
 f11vmr         = 653.45e-12
 f12vmr         = 535.0e-12
 flbc_list              = ' '
 n2ovmr         = 316.0e-9
/
TXT2

# co2vmr         = 353.9e-6
# co2vmr         = 530.9e-6

#cat <<TXT2 >> user_nl_clm
#&clm_inparm
# fglcmask = '/home/fas/long/zm56/glcmaskdata_1.9x2.5_Gland5km.nc'
# fsurdat = '/home/fas/long/zm56/surfdata_1.9x2.5_simyr2000_glcmec10_c120927.nc'
#TXT2

#cat <<TXT2 >> user_nl_cism
# horiz_grid_file = '/home/fas/long/zm56/fracdata_1.9x2.5_gx1v6_c090206.glc.nc'
# cisminputfile = '/home/fas/long/zm56/Greenland_5km_v1.1_SacksRev_c110629.nc'
#TXT2

#./xmlchange -file env_run.xml -id DOCN_SOM_FILENAME -val 'pop_frc.gx1v6.100513.nc'
./xmlchange -file env_run.xml -id DOCN_SOM_FILENAME -val 'pop_frc.b.c40.B1850CN.f19_g16.100105.nc'

#cp CaseDocs/docn.streams.txt.som user_docn.streams.txt
#sed -i "/docn7/c\ /home/fas/long/zm56/pop_frc.b.c40.B1850CN.f19_g16.100105.nc" docn.streams.txt.som
#sed -i "/docn7/c\ file1 = '/home/fas/long/zm56/pop_frc.gx1v6.100513.nc'" docn.streams.txt.som

#&cldfrc_nl
# cldfrc_rhminl = 0.925D0
#/

# cldfrc_iceopt          =  4

#rad_diag_2= 'A:Q:H2O','N:O2:O2', 'N:CO2:CO2','N:ozone:O3','N:N2O:N2O','N:CH4:CH4','N:CFC11:CFC11', 'N:CFC12:CFC12','M:mam3_mode1:/apps/hpc/Data/CESM/1.0/inputdata/atm/cam/physprops/mam3_mode1_rrtmg_c110318.nc','M:mam3_mode2:/apps/hpc/Data/CESM/1.0/inputdata/atm/cam/physprops/mam3_mode2_rrtmg_c110318.nc', 'M:mam3_mode3:/apps/hpc/Data/CESM/1.0/inputdata/atm/cam/physprops/mam3_mode3_rrtmg_c110318.nc'
#/

#&aerosol_nl
# dust_emis_fact         = 0.50D0
#/

#&rad_cnst_nl
# mode_defs              = 'mam3_mode1:accum:=', 'A:num_a1:N:num_c1:num_mr:+',
#         'A:so4_a1:N:so4_c1:sulfate:/apps/hpc/Data/CESM/1.0/inputdata/atm/cam/physprops/sulfate_rrtmg_c080918.nc:+', 'A:pom_a1:N:pom_c1:p-organic:/apps/hpc/Data/CESM/1.0/inputdata/atm/cam/physprops/ocpho_rrtmg_c101112.nc:+',
#         'A:soa_a1:N:soa_c1:s-organic:/apps/hpc/Data/CESM/1.0/inputdata/atm/cam/physprops/ocphi_rrtmg_c100508.nc:+', 'A:bc_a1:N:bc_c1:black-c:/apps/hpc/Data/CESM/1.0/inputdata/atm/cam/physprops/bcpho_rrtmg_c100508.nc:+',
#         'A:dst_a1:N:dst_c1:dust:/home/fas/long/zm56/dustmod/cleansky/dust1_rrtmg_zender.nc:+', 'A:ncl_a1:N:ncl_c1:seasalt:/apps/hpc/Data/CESM/1.0/inputdata/atm/cam/physprops/ssam_rrtmg_c100508.nc',
#         'mam3_mode2:aitken:=', 'A:num_a2:N:num_c2:num_mr:+',
#         'A:so4_a2:N:so4_c2:sulfate:/apps/hpc/Data/CESM/1.0/inputdata/atm/cam/physprops/sulfate_rrtmg_c080918.nc:+', 'A:soa_a2:N:soa_c2:s-organic:/apps/hpc/Data/CESM/1.0/inputdata/atm/cam/physprops/ocphi_rrtmg_c100508.nc:+',
#         'A:ncl_a2:N:ncl_c2:seasalt:/apps/hpc/Data/CESM/1.0/inputdata/atm/cam/physprops/ssam_rrtmg_c100508.nc', 'mam3_mode3:coarse:=',
#         'A:num_a3:N:num_c3:num_mr:+', 'A:dst_a3:N:dst_c3:dust:/home/fas/long/zm56/dustmod/cleansky/dust1_rrtmg_zender.nc:+',
#         'A:ncl_a3:N:ncl_c3:seasalt:/apps/hpc/Data/CESM/1.0/inputdata/atm/cam/physprops/ssam_rrtmg_c100508.nc:+', 'A:so4_a3:N:so4_c3:sulfate:/apps/hpc/Data/CESM/1.0/inputdata/atm/cam/physprops/sulfate_rrtmg_c080918.nc'
# rad_climate            = 'A:Q:H2O', 'N:O2:O2', 'N:CO2:CO2',
#         'N:ozone:O3', 'N:N2O:N2O', 'N:CH4:CH4',
#         'N:CFC11:CFC11', 'N:CFC12:CFC12', 'M:mam3_mode1:/apps/hpc/Data/CESM/1.0/inputdata/atm/cam/physprops/mam3_mode1_rrtmg_c110318.nc',
#         'M:mam3_mode2:/apps/hpc/Data/CESM/1.0/inputdata/atm/cam/physprops/mam3_mode2_rrtmg_c110318.nc', 'M:mam3_mode3:/apps/hpc/Data/CESM/1.0/inputdata/atm/cam/physprops/mam3_mode3_rrtmg_c110318.nc'
#/
#
#TXT2

#/home/fas/long/zm56/dustmod/cleansky/dust1_rrtmg_zender_nodustrad.nc

#&phys_ctl_nl
# conv_water_in_rad              =  1
# deep_scheme            = 'ZM'
# do_clubb_sgs           =  .false.
# do_tms         =  .true.
# eddy_scheme            = 'diag_TKE'
#! history_aero_optics            = .true.
#! history_aerosol                =     .true.
#! history_amwg           =        .true.
#! history_budget         =      .true.
#! history_eddy           =        .false.
# macrop_scheme          = 'park'
# microp_scheme          = 'MG'
# radiation_scheme               = 'rrtmg'
# shallow_scheme         = 'UW'
# srf_flux_avg           = 0
# use_subcol_microp              = .false.
# waccmx_opt             = 'off'
#/
#&physconst_nl
# tms_orocnst            =  1.0D0
# tms_z0fac              =  0.075D0
#/
#&aerosol_nl
# fincl1 ='DSTSFDRY','DSTSFMBL','DSTSFWET','dst_a1SF', 'dst_a3SF','dst_a1','dst_a3','dst_c1','dst_c3','bc_a1','bc_c1','ncl_a1','ncl_c1','ncl_a2','ncl_c2','ncl_a3','ncl_c3','num_a1','num_c1','num_a2','num_c2','num_a3','num_c3','pom_a1','pom_c1','so4_a1','so4_c1','so4_a2','so4_c2','so4_a3','so4_c3','soa_a1','soa_c1','soa_a2','soa_c2','dst_a1DDF','dst_c1DDF','dst_a3DDF','dst_c3DDF','dst_a1SFWET','dst_c1SFWET','dst_a3SFWET','dst_c3SFWET','MET_RLX','MET_TAUX','MET_TAUY','MET_SHFX','MET_QFLX','MET_PS','MET_T','MET_U','MET_V','MET_SNOWH','MET_TS','MET_OCNFRC','MET_ICEFRC','CMEIOUT','MNUCCCO','MNUCCTO','MNUCCDO','MNUCCDOhet','BERGO','BERGSO','HOMOO','PRAO','PRCO','PRCIO','PRAIO','WSUBI','NIHF','NIDEP','NIIMM','NIMEY','NCAL','NCAI','FREQS','ADSNOW','FREQS','ANSNOW','AQSNOW','DSNOW','NSNOW','EFFICE','ICINC','ICIMR','ICWNC','ICIMRST','ICIMRCU','ICIMRTOT','MPDQ','MPDT','MPDW2V','MPDW2I','MPDW2P','MPDI2V','MPDI2W','MPDI2P','NDROPSNK','MPDICE','MPDLIQ','QCRESO','QIRESO','MSACWIO','PSACWSO','MELTO','VPRAO','VPRCO','RACAU'
#/
#&rad_cnst_nl
# mode_defs              = 'mam3_mode1:accum:=', 'A:num_a1:N:num_c1:num_mr:+',
#         'A:so4_a1:N:so4_c1:sulfate:/apps/hpc/Data/CESM/1.0/inputdata/atm/cam/physprops/sulfate_rrtmg_c080918.nc:+', 'A:pom_a1:N:pom_c1:p-organic:/apps/hpc/Data/CESM/1.0/inputdata/atm/cam/physprops/ocpho_rrtmg_c101112.nc:+',
#         'A:soa_a1:N:soa_c1:s-organic:/apps/hpc/Data/CESM/1.0/inputdata/atm/cam/physprops/ocphi_rrtmg_c100508.nc:+', 'A:bc_a1:N:bc_c1:black-c:/apps/hpc/Data/CESM/1.0/inputdata/atm/cam/physprops/bcpho_rrtmg_c100508.nc:+',
#         'A:dst_a1:N:dst_c1:dust:/home/fas/long/zm56/dustmod/cleansky/dust1_rrtmg_zender.nc:+', 'A:ncl_a1:N:ncl_c1:seasalt:/apps/hpc/Data/CESM/1.0/inputdata/atm/cam/physprops/ssam_rrtmg_c100508.nc',
#         'mam3_mode2:aitken:=', 'A:num_a2:N:num_c2:num_mr:+',
#         'A:so4_a2:N:so4_c2:sulfate:/apps/hpc/Data/CESM/1.0/inputdata/atm/cam/physprops/sulfate_rrtmg_c080918.nc:+', 'A:soa_a2:N:soa_c2:s-organic:/apps/hpc/Data/CESM/1.0/inputdata/atm/cam/physprops/ocphi_rrtmg_c100508.nc:+',
#         'A:ncl_a2:N:ncl_c2:seasalt:/apps/hpc/Data/CESM/1.0/inputdata/atm/cam/physprops/ssam_rrtmg_c100508.nc', 'mam3_mode3:coarse:=',
#         'A:num_a3:N:num_c3:num_mr:+', 'A:dst_a3:N:dst_c3:dust:/home/fas/long/zm56/dustmod/cleansky/dust1_rrtmg_zender.nc:+',
#         'A:ncl_a3:N:ncl_c3:seasalt:/apps/hpc/Data/CESM/1.0/inputdata/atm/cam/physprops/ssam_rrtmg_c100508.nc:+', 'A:so4_a3:N:so4_c3:sulfate:/apps/hpc/Data/CESM/1.0/inputdata/atm/cam/physprops/sulfate_rrtmg_c080918.nc'
# rad_climate            = 'A:Q:H2O', 'N:O2:O2', 'N:CO2:CO2',
#         'N:ozone:O3', 'N:N2O:N2O', 'N:CH4:CH4',
#         'N:CFC11:CFC11', 'N:CFC12:CFC12', 'M:mam3_mode1:/apps/hpc/Data/CESM/1.0/inputdata/atm/cam/physprops/mam3_mode1_rrtmg_c110318.nc',
#         'M:mam3_mode2:/apps/hpc/Data/CESM/1.0/inputdata/atm/cam/physprops/mam3_mode2_rrtmg_c110318.nc', 'M:mam3_mode3:/apps/hpc/Data/CESM/1.0/inputdata/atm/cam/physprops/mam3_mode3_rrtmg_c110318.nc'
#/
#TXT2

#&cam_inparm
# bnd_topo  = '/home/fas/long/zm56/scratch/cesm_input/USGS-gtopo30_1.9x2.5_phys_geos5_c100929.nc'
# met_data_file        = 'MERRA_19x2_20040101.nc'
# met_data_path        = '/home/fas/long/zm56/scratch/cesm_input/MERRA_glade'
# met_filenames_list        = '/home/fas/long/zm56/scratch/cesm_input/merra_filelist_2004-8.txt'
# met_max_rlx            = 0.083
#/

#dust1_rrtmg_zender_nodustrad.nc

#&rad_cnst_nl
# mode_defs              = 'mam3_mode1:accum:=', 'A:num_a1:N:num_c1:num_mr:+',
#         'A:so4_a1:N:so4_c1:sulfate:/apps/hpc/Data/CESM/1.0/inputdata/atm/cam/physprops/sulfate_rrtmg_c080918.nc:+', 'A:pom_a1:N:pom_c1:p-organic:/apps/hpc/Data/CESM/1.0/inputdata/atm/cam/physprops/ocpho_rrtmg_c101112.nc:+',
#         'A:soa_a1:N:soa_c1:s-organic:/apps/hpc/Data/CESM/1.0/inputdata/atm/cam/physprops/ocphi_rrtmg_c100508.nc:+', 'A:bc_a1:N:bc_c1:black-c:/apps/hpc/Data/CESM/1.0/inputdata/atm/cam/physprops/bcpho_rrtmg_c100508.nc:+',
#         'A:dst_a1:N:dst_c1:dust:/home/fas/long/zm56/dustmod/albani2014/dust1_rrtmg_zender.nc:+', 'A:ncl_a1:N:ncl_c1:seasalt:/apps/hpc/Data/CESM/1.0/inputdata/atm/cam/physprops/ssam_rrtmg_c100508.nc',
#         'mam3_mode2:aitken:=', 'A:num_a2:N:num_c2:num_mr:+',
#         'A:so4_a2:N:so4_c2:sulfate:/apps/hpc/Data/CESM/1.0/inputdata/atm/cam/physprops/sulfate_rrtmg_c080918.nc:+', 'A:soa_a2:N:soa_c2:s-organic:/apps/hpc/Data/CESM/1.0/inputdata/atm/cam/physprops/ocphi_rrtmg_c100508.nc:+',
#         'A:ncl_a2:N:ncl_c2:seasalt:/apps/hpc/Data/CESM/1.0/inputdata/atm/cam/physprops/ssam_rrtmg_c100508.nc', 'mam3_mode3:coarse:=',
#         'A:num_a3:N:num_c3:num_mr:+', 'A:dst_a3:N:dst_c3:dust:/home/fas/long/zm56/dustmod/albani2014/dust1_rrtmg_zender.nc:+',
#         'A:ncl_a3:N:ncl_c3:seasalt:/apps/hpc/Data/CESM/1.0/inputdata/atm/cam/physprops/ssam_rrtmg_c100508.nc:+', 'A:so4_a3:N:so4_c3:sulfate:/apps/hpc/Data/CESM/1.0/inputdata/atm/cam/physprops/sulfate_rrtmg_c080918.nc'
# rad_climate            = 'A:Q:H2O', 'N:O2:O2', 'N:CO2:CO2',
#         'N:ozone:O3', 'N:N2O:N2O', 'N:CH4:CH4',
#         'N:CFC11:CFC11', 'N:CFC12:CFC12', 'M:mam3_mode1:/apps/hpc/Data/CESM/1.0/inputdata/atm/cam/physprops/mam3_mode1_rrtmg_c110318.nc',
#         'M:mam3_mode2:/apps/hpc/Data/CESM/1.0/inputdata/atm/cam/physprops/mam3_mode2_rrtmg_c110318.nc', 'M:mam3_mode3:/apps/hpc/Data/CESM/1.0/inputdata/atm/cam/physprops/mam3_mode3_rrtmg_c110318.nc'
#/

#&cospsimulator_nl
# docosp = .true.
# cosp_lmodis_sim = .true.
# cosp_lmisr_sim = .true.
# cosp_llidar_sim = .true.
#/
#&cloudfrc_nl
# cldfrc_rhminl = 0.8875D0
#/

# met_max_rlx            = 0.083

#&cloudfrc_nl
# cldfrc_rhminl = 0.89D0
#/

#metdata_nl

# define walltime
#sed -i "s/#BSUB -W 8:00/#BSUB -W 10:00/" ${case}.run

#sed -i "/domain1/c\ domain1 = '/home/fas/long/zm56/pop_frc.gx1v6.100513.nc'" Buildconf/docn.input_data_list
#sed -i "/file1/c\ file1 = '/home/fas/long/zm56/pop_frc.gx1v6.100513.nc'" Buildconf/docn.input_data_list
#sed -i "/fsurdat/c\ fsurdat = '/home/fas/long/zm56/surfdata_1.9x2.5_simyr2000_glcmec10_c120927.nc'" Buildconf/clm.input_data_list
#sed -i "/fglcmask/c\ fglcmask = '/home/fas/long/zm56/glcmaskdata_1.9x2.5_Gland5km.nc'" Buildconf/clm.input_data_list
#sed -i "/horiz_grid_file/c\ horiz_grid_file = '/home/fas/long/zm56/fracdata_1.9x2.5_gx1v6_c090206.glc.nc'" Buildconf/cism.input_data_list
#sed -i "/cisminputfile/c\ cisminputfile = '/home/fas/long/zm56/Greenland_5km_v1.1_SacksRev_c110629.nc'" Buildconf/cism.input_data_list

#sed -i "/domain1/c\ domain1 = '/home/fas/long/zm56/pop_frc.gx1v6.100513.nc'" CaseDocs/docn.streams.txt.som
#sed -i "/file1/c\ file1 = '/home/fas/long/zm56/pop_frc.gx1v6.100513.nc'" CaseDocs/docn.streams.txt.som
#sed -i "/fsurdat/c\ fsurdat = '/home/fas/long/zm56/surfdata_1.9x2.5_simyr2000_glcmec10_c120927.nc'" CaseDocs/lnd_in
#sed -i "/fglcmask/c\ fglcmask = '/home/fas/long/zm56/glcmaskdata_1.9x2.5_Gland5km.nc'" CaseDocs/lnd_in
#sed -i "/horiz_grid_file/c\ horiz_grid_file = '/home/fas/long/zm56/fracdata_1.9x2.5_gx1v6_c090206.glc.nc'" CaseDocs/cism.config
#sed -i "/cisminputfile/c\ cisminputfile = '/home/fas/long/zm56/Greenland_5km_v1.1_SacksRev_c110629.nc'" CaseDocs/cism.config

# build model
# cesm 1.0.4: ./${case}.bulldog.build
./${case}.build

# change land path in Buildconf/cml.buildnml.csh
# cesm 1.0.4: Buildconf/clm.buildnml.csh
#sed -i "/fsurdat/c\ fsurdat = '/home/fas/long/zm56/dustmod/ck_surfdata_1.9x2.5_simyr2000_c091005.nc'" Buildconf/clmconf/lnd_in
#sed -i "/fsurdat/c\fsurdat = /home/fas/long/zm56/dustmod/ck_surfdata_1.9x2.5_simyr2000_c091005.nc" Buildconf/clm.input_data_list
#sed -i "/fsurdat/c\ fsurdat = '/home/fas/long/zm56/dustmod/ck_surfdata_1.9x2.5_simyr2000_c091005.nc'" CaseDocs/lnd_in

# submit job
# cesm 1.0.4: qsub ${case}.bulldog.run
#bsub -P UYLE0005 -W 02:00 -l nodes=8:ppn=16 -q small < ${case}.run
#bsub -P UYLE0005 -W 02:00 -n 64 -q regular < ${case}.run
#nodes=8:ppn=16
#bsub -P UYLE0005 -W 02:00 -q small < ${case}.run
#bsub -P UYLE0005 < ${case}.run

sed -i '9i#SBATCH --exclude=c03n[01-16]' ${case}.run

#sed -i "s/#SBATCH --time=168:00:00/#SBATCH --time=40:00:00/" ${case}.run
cp ${case}.run ${case}.run_fas
sed -i "s/#SBATCH --partition=week/#SBATCH --partition=scavenge/" ${case}.run
sed -i "/#SBATCH -A long/d" ${case}.run
#sed -i "s/#SBATCH --time=168:00:00/#SBATCH --time=40:00:00/" ${case}.run
cp ${case}.run ${case}.run_scavenge
./${case}.submit
cp ${case}.run_fas ${case}.run
sleep 1m
./${case}.submit
