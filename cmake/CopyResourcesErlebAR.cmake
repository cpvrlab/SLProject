#internal utility functions:

#copy list in data, which is relative to SRC directory to DEST directory
function(copy_data_src_dest DATA SRC DEST)
	foreach(FILE_TO_COPY ${DATA})
		get_filename_component(PATH_TO_COPY ${FILE_TO_COPY} DIRECTORY)
		file(COPY "${SRC}/${FILE_TO_COPY}" DESTINATION "${DEST}/${PATH_TO_COPY}")	
	endforeach()	
endfunction(copy_data_src_dest)

# Definition of resource files for erlebar application and distribution to iOS and Android bundles
function(copy_resources_erlebar TARGET_DIR)
	#definition of directory to copy
	#(filenames are defined relative to SOURCE_DIR so that we can easily concetenate the target fullpath-filename)
	set(SOURCE_DIR ${SL_PROJECT_ROOT}/data/)
	
	#ATTENTION: in the following one can only "select" files to copy from SOURCE_DIR but the files will be distributed in the same directory structure
		
	# Definition
	file(GLOB_RECURSE 
		TEXTURES
		RELATIVE
		${SOURCE_DIR}
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
		${SOURCE_DIR}
		${SL_PROJECT_ROOT}/data/images/fonts/*.png
		${SL_PROJECT_ROOT}/data/images/fonts/*.ttf
	    )

	file(GLOB_RECURSE 
		MODELS
		RELATIVE
		${SOURCE_DIR}
		${SL_PROJECT_ROOT}/data/models/FBX/Axes/axes_blender.fbx
	    )
		
	file(GLOB_RECURSE 
		SHADERS
		RELATIVE
		${SOURCE_DIR}
	    ${SL_PROJECT_ROOT}/data/shaders/*.vert
	    ${SL_PROJECT_ROOT}/data/shaders/*.frag
		${SL_PROJECT_ROOT}/data/shaders/*.glsl
	    )

	file(GLOB_RECURSE 
		CALIBRATIONS
		RELATIVE
		${SOURCE_DIR}
		#${SL_PROJECT_ROOT}/data/calibrations/ORBvoc.bin
		${SL_PROJECT_ROOT}/data/calibrations/voc_fbow.bin
	    )

	file(GLOB_RECURSE 
		CONFIG
		RELATIVE
		${SOURCE_DIR}
		${SL_PROJECT_ROOT}/data/config/StringsEnglish.json
		${SL_PROJECT_ROOT}/data/config/StringsFrench.json
		${SL_PROJECT_ROOT}/data/config/StringsGerman.json
		${SL_PROJECT_ROOT}/data/config/StringsItalian.json
	    )
	
	file(GLOB_RECURSE 
		ERLEBAR
		RELATIVE
		${SOURCE_DIR}
		${SL_PROJECT_ROOT}/data/erleb-AR/*
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
		
	copy_data_src_dest("${RESOURCES}" "${SOURCE_DIR}" "${TARGET_DIR}")

endfunction(copy_resources_erlebar)

#copy erlebar test resouces
function(copy_resources_erlebar_test TARGET_DIR)
	#definition of directory to copy
	#(filenames are defined relative to SOURCE_DIR so that we can easily concetenate the target fullpath-filename)
	set(SOURCE_DIR ${SL_PROJECT_ROOT}/erleb-AR/)
	
	#ATTENTION: in the following one can only "select" files to copy from SOURCE_DIR but the files will be distributed in the same directory structure
		
	# Definition
	file(GLOB_RECURSE 
		ERLEBAR_TEST
		RELATIVE
		${SOURCE_DIR}
		${SOURCE_DIR}/locations/*
		${SOURCE_DIR}/calibrations/*
	    )
		
	# Distribution
	set(RESOURCES
		${ERLEBAR_TEST}
		)
		
	copy_data_src_dest("${RESOURCES}" "${SOURCE_DIR}" "${TARGET_DIR}")

endfunction(copy_resources_erlebar_test)