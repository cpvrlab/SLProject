# Definition of resource files for erlebar application and distribution to iOS and Android bundles
function(copy_resources_erlebar TARGET_DIR)
	#definition of directory to copy
	#(filenames are defined relative to SL_PROJECT_DATA_ROOT so that we can easily concetenate the target fullpath-filename)
	set(SL_PROJECT_DATA_ROOT ${SL_PROJECT_ROOT}/data/)
	
	#ATTENTION: in the following one can only "select" files to copy from SL_PROJECT_DATA_ROOT but the files will be distributed in the same directory structure
		
	# Definition
	file(GLOB_RECURSE 
		TEXTURES
		RELATIVE
		${SL_PROJECT_DATA_ROOT}
		${SL_PROJECT_ROOT}/data/images/textures/CompileError.png
		${SL_PROJECT_ROOT}/data/images/textures/cursor.png
		${SL_PROJECT_ROOT}/data/images/textures/LiveVideoError.png
		${SL_PROJECT_ROOT}/data/images/textures/LogoCPVR_256L.png
		${SL_PROJECT_ROOT}/data/images/textures/logo_admin_ch.png
		${SL_PROJECT_ROOT}/data/images/textures/logo_bfh.png
		${SL_PROJECT_ROOT}/data/images/textures/earth2048_C.jpg
		${SL_PROJECT_ROOT}/data/images/textures/earthCloud1024_C.jpg
		${SL_PROJECT_ROOT}/data/images/textures/icon_back.png
		${SL_PROJECT_ROOT}/data/images/textures/back1white.png
		${SL_PROJECT_ROOT}/data/images/textures/left1white.png
	    )

	file(GLOB_RECURSE 
		FONTS
		RELATIVE
		${SL_PROJECT_DATA_ROOT}
		${SL_PROJECT_ROOT}/data/images/fonts/*.png
		${SL_PROJECT_ROOT}/data/images/fonts/*.ttf
	    )

	file(GLOB_RECURSE 
		MODELS
		RELATIVE
		${SL_PROJECT_DATA_ROOT}
		${SL_PROJECT_ROOT}/data/models/*
	    )
		
	file(GLOB_RECURSE 
		SHADERS
		RELATIVE
		${SL_PROJECT_DATA_ROOT}
	    ${SL_PROJECT_ROOT}/data/shaders/*.vert
	    ${SL_PROJECT_ROOT}/data/shaders/*.frag
	    )

	file(GLOB_RECURSE 
		CALIBRATIONS
		RELATIVE
		${SL_PROJECT_DATA_ROOT}
		${SL_PROJECT_ROOT}/data/calibrations/calib_in_params.yml
		${SL_PROJECT_ROOT}/data/calibrations/aruco_detector_params.yml
		${SL_PROJECT_ROOT}/data/calibrations/cam_calibration_main.xml
		#${SL_PROJECT_ROOT}/data/calibrations/ORBvoc.bin
		${SL_PROJECT_ROOT}/data/calibrations/voc_fbow.bin
	    )

	file(GLOB_RECURSE 
		CONFIG
		RELATIVE
		${SL_PROJECT_DATA_ROOT}
		${SL_PROJECT_ROOT}/data/config/dummyFile.txt
		${SL_PROJECT_ROOT}/data/config/StringsEnglish.json
		${SL_PROJECT_ROOT}/data/config/StringsFrench.json
		${SL_PROJECT_ROOT}/data/config/StringsGerman.json
		${SL_PROJECT_ROOT}/data/config/StringsItalian.json
		${SL_PROJECT_ROOT}/data/config/TesterConfig.json
	    )
	
	file(GLOB_RECURSE 
		ERLEBAR
		RELATIVE
		${SL_PROJECT_DATA_ROOT}
		${SL_PROJECT_ROOT}/data/erlebAR/*
	    )
		
	# Distribution
	set(RESOURCES
		${TEXTURES}
		${VIDEOS}
		${FONTS}
		${MODELS}
		${SHADERS}
		${CALIBRATIONS}
		${CONFIG}
		${ERLEBAR}
		)
		
	foreach(FILE_TO_COPY ${RESOURCES})
		get_filename_component(PATH_TO_COPY ${FILE_TO_COPY} DIRECTORY)
		file(COPY "${SL_PROJECT_DATA_ROOT}/${FILE_TO_COPY}" DESTINATION "${TARGET_DIR}/${PATH_TO_COPY}")	
	endforeach()	

endfunction(copy_resources_slprojectdemo)























file(MAKE_DIRECTORY
    ${APK_ASSETS}/images
    ${APK_ASSETS}/images/fonts
    ${APK_ASSETS}/images/textures
    ${APK_ASSETS}/videos
    ${APK_ASSETS}/models
    ${APK_ASSETS}/shaders
    ${APK_ASSETS}/calibrations
    ${APK_ASSETS}/config
	${APK_ASSETS}/voc
    )

file(COPY ${FONTS}          DESTINATION ${APK_ASSETS}/images/fonts)
file(COPY ${TEXTURES}       DESTINATION ${APK_ASSETS}/images/textures)
file(COPY ${VIDEOS}         DESTINATION ${APK_ASSETS}/videos)
file(COPY ${MODELS}         DESTINATION ${APK_ASSETS}/models)
file(COPY ${SHADERS}        DESTINATION ${APK_ASSETS}/shaders)
file(COPY ${CALIBRATIONS}   DESTINATION ${APK_ASSETS}/calibrations)
file(COPY ${CONFIG}         DESTINATION ${APK_ASSETS}/config)
file(COPY ${VOC}         	DESTINATION ${APK_ASSETS}/voc)
