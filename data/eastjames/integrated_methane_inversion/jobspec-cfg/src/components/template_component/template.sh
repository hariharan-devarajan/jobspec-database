#!/bin/bash

# Functions available in this file include:
#   - setup_template 

# Description: Setup template GCClassic run directory
# Usage:
#   setup_template
setup_template() {
    printf "\n=== CREATING TEMPLATE RUN DIRECTORY ===\n"

    cd ${GCClassicPath}/run

    # The createRunDir.sh script assumes the file ~/.geoschem/config exists
    # and contains the path to GEOS-Chem input data
	export GC_USER_REGISTERED=true
    if [[ ! -f ${HOME}/.geoschem/config ]]; then
	mkdir -p ${HOME}/.geoschem
	echo "export GC_DATA_ROOT=${DataPath}" >> ${HOME}/.geoschem/config
	source ${HOME}/.geoschem/config
    fi

    if [[ -d ${RunTemplate} ]]; then
	printf "\nERROR: ${RunTemplate} already exists. Please remove or set 'SetupTemplateRunDir: false' in config.yml.\n"
	exit 9999
    fi

    # Commands to feed to createRunDir.sh
    # Run directories are created for the global domain by default. If needed,
    # the regional domain specified in config.yml will be adjusted for below.
    if [[ "$Met" == "MERRA2" || "$Met" == "MERRA-2" || "$Met" == "merra2" ]]; then
	metNum="1"
    elif [[ "$Met" == "GEOSFP" || "$Met" == "GEOS-FP" || "$Met" == "geosfp" ]]; then
	metNum="2"
    else
	printf "\nERROR: Meteorology field ${Met} is not supported by the IMI. "
	printf "\n Options are GEOSFP or MERRA2.\n"
	exit 1
    fi	
    if [ "$Res" = "4.0x5.0" ]; then
	cmd="3\n${metNum}\n1\n2\n${RunDirs}\n${runDir}\nn\n"
    elif [ "$Res" == "2.0x2.5" ]; then
	cmd="3\n${metNum}\n2\n2\n${RunDirs}\n${runDir}\nn\n"
    elif [ "$Res" == "0.5x0.625" ]; then
	cmd="3\n${metNum}\n3\n1\n2\n${RunDirs}\n${runDir}\nn\n"
    elif [ "$Res" == "0.25x0.3125" ]; then
	cmd="3\n${metNum}\n4\n1\n2\n${RunDirs}\n${runDir}\nn\n"
    else
	printf "\nERROR: Grid resolution ${Res} is not supported by the IMI. "
	printf "\n Options are 0.25x0.3125, 0.5x0.625, 2.0x2.5, or 4.0x5.0.\n"
	exit 1
    fi

    # Create run directory
    printf ${cmd} | ./createRunDir.sh >> createRunDir.log 2>&1
    rm -f createRunDir.log
    printf "\nCreated ${RunTemplate}\n"

    cd ${RunTemplate}

    if "$isAWS"; then
	# Update GC data download to silence output from aws commands
	sed -i "s/command: 'aws s3 cp --request-payer=requester '/command: 'aws s3 cp --request-payer=requester --only-show-errors '/" download_data.yml
    fi

    # Modify geoschem_config.yml based on settings in config.yml
    sed -i -e "s:20190101:${StartDate}:g" \
           -e "s:20190201:${EndDate}:g" geoschem_config.yml
    if "$isRegional"; then
        sed -i -e "s:-180.0, 180.0:${Lons}:g" \
               -e "s:-90.0, 90.0:${Lats}:g" geoschem_config.yml
    fi
    
    # Modify config and link obs directory for TROPOMI online operator
    if "$CustomGC"; then
        sed -i -e "s|TROPOMI: false|TROPOMI: true|g" geoschem_config.yml
        ln -s ${SatObsDir} "SatelliteObservations"
    fi

    # For CH4 inversions always turn analytical inversion on
    sed -i "/analytical_inversion/{N;s/activate: false/activate: true/}" geoschem_config.yml

    # Custom HEMCO_Config.rc
    if "$CustomHEMCOConfig"; then
        echo 'WARNING: Using custom HEMCO_Config.rc file!'
        mv HEMCO_Config.rc orig.HEMCO_Config.rc
        cp ${CustomHEMCOConfigFile} HEMCO_Config.rc
        
        sed -i -e "s|/path/to/total/ch4/emissions Total_CH4_Emis 2018|${PrecomputedEmisDir}/HEMCO_sa_diagnostics.\$YYYY\$MM\$DD0000.nc EmisCH4_Total 2018-2023|g" \
               -e "s|/path/to/soil/sink SOIL_ABSORPTION 2018|${PrecomputedEmisDir}/HEMCO_sa_diagnostics.\$YYYY\$MM\$DD0000.nc EmisCH4_SoilAbsorb 2018-2023|g" \
               -e "s|GC_BCs                 :       true|GC_BCs                 :       false|g" HEMCO_Config.rc

        # Modify HEMCO_Diagn.rc to archive summed emissions and soil sink
        sed -i '23,85d' HEMCO_Diagn.rc
        sed -i -e "s|EmisCH4_Total        CH4   -1   -1   -1   2   kg/m2/s  CH4_emissions_from_all_sectors|EmisCH4_Total CH4 0 1 -1 2 kg/m2/s CH4_emissions_from_all_sectors|g" \
               -e "s|EmisCH4_Oil          CH4    0    1   -1   2   kg/m2/s  CH4_emissions_from_oil|EmisCH4_Soil CH4 0 2 -1 2 kg/m2/s CH4_soil_absorption|g" HEMCO_Diagn.rc


    fi

    # Also turn on analytical inversion option in HEMCO_Config.rc
    OLD="--> AnalyticalInv          :       false"
    NEW="--> AnalyticalInv          :       true "
    sed -i "s/$OLD/$NEW/g" HEMCO_Config.rc

    # Modify path to state vector file in HEMCO_Config.rc
    OLD=" StateVector.nc"
    NEW=" ${RunDirs}/StateVector.nc"
    sed -i -e "s@$OLD@$NEW@g" HEMCO_Config.rc

    # Modify HEMCO_Config.rc if running Kalman filter
    if "$KalmanMode"; then
        sed -i -e "s|use_emission_scale_factor: false|use_emission_scale_factor: true|g" geoschem_config.yml
        sed -i -e "s|--> Emis_ScaleFactor       :       false|--> Emis_ScaleFactor       :       true|g" \
               -e "s|gridded_posterior.nc|${RunDirs}/ScaleFactors.nc|g" HEMCO_Config.rc
    fi

    # Turn other options on/off according to settings above
    if "$UseEmisSF"; then
	OLD="use_emission_scale_factor: false"
	NEW="use_emission_scale_factor: true"
	sed -i "s/$OLD/$NEW/g" geoschem_config.yml
    fi
    if "$UseOHSF"; then
	OLD="use_OH_scale_factors: false"
	NEW="use_OH_scale_factors: true"
	sed -i "s/$OLD/$NEW/g" geoschem_config.yml
    fi

    # Modify HEMCO_Config.rc based on settings in config.yml
    # Use cropped met fields (add the region to both METDIR and the met files)
    if "$isRegional"; then
	sed -i -e "s:GEOS_${Res}:GEOS_${Res}_${RegionID}:g" HEMCO_Config.rc
	sed -i -e "s:GEOS_${Res}:GEOS_${Res}_${RegionID}:g" HEMCO_Config.rc.gmao_metfields
        sed -i -e "s:\$RES:\$RES.${RegionID}:g" HEMCO_Config.rc.gmao_metfields
    fi

    # Determine length of inversion period in days
    InvPeriodLength=$(( ( $(date -d ${EndDate} "+%s") - $(date -d ${StartDate} "+%s") ) / 86400))

    # If inversion period is < 32 days, use End diagnostic output frequency
    if (( ${InvPeriodLength} < 32 )) || $KalmanMode; then
        sed -i -e "s|DiagnFreq:                   Monthly|DiagnFreq:                   End|g" HEMCO_Config.rc
    fi

    # Modify path to BC files
    sed -i -e "s:\$ROOT/SAMPLE_BCs/v2021-07/CH4:${fullBCpath}:g" HEMCO_Config.rc

    # Modify HISTORY.rc
    sed -i -e "s:'CH4':#'CH4':g" \
           -e "s:'Metrics:#'Metrics:g" \
           -e "s:'Restart:#'Restart:g" \
           -e "s:'SpeciesConc:#'SpeciesConc:g" \
           -e "s:'StateMet:#'StateMet:g" HISTORY.rc
    
    # If turned on, save out hourly CH4 concentrations to daily files
    if "$HourlyCH4"; then
        sed -i -e 's/SpeciesConc.frequency:      00000100 000000/SpeciesConc.frequency:      00000000 010000/g' \
    	       -e 's/SpeciesConc.duration:       00000100 000000/SpeciesConc.duration:       00000001 000000/g' \
               -e 's/SpeciesConc.mode:           '\''time-averaged/SpeciesConc.mode:           '\''instantaneous/g' HISTORY.rc
    fi

    # Remove sample restart file
    rm -f Restarts/GEOSChem.Restart.20190101_0000z.nc4

    # Copy template run script
    cp ${InversionPath}/src/geoschem_run_scripts/ch4_run.template .

    # Compile GEOS-Chem and store executable in template run directory
    printf "\nCompiling GEOS-Chem...\n"
    cd build
    cmake ${InversionPath}/GCClassic >> build_geoschem.log 2>&1
    cmake . -DRUNDIR=..  >> build_geoschem.log 2>&1 
    make -j install >> build_geoschem.log 2>&1
    cd ..
    if [[ -f gcclassic ]]; then
        rm -rf build #jde
        mv build_info ../GEOSChem_build_info
        #cp build_info ../GEOSChem_build_info
    else
        printf "\nGEOS-Chem build failed! \n\nSee ${RunTemplate}/build/build_geoschem.log for details\n"
        exit 999
    fi
    printf "\nDone compiling GEOS-Chem \n\nSee ${RunDirs}/GEOSChem_build_info for details\n\n"
    
    # Navigate back to top-level directory
    cd ..

    printf "\n=== DONE CREATING TEMPLATE RUN DIRECTORY ===\n"
}
