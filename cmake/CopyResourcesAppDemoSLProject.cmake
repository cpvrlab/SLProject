# # Definition of resource files for SLProject-demo application and distribution to iOS and Android bundles

function(copy_resources_slprojectdemo TARGET_DIR)
	# Definition
	file(GLOB_RECURSE 
		TEXTURES
	    ${SL_PROJECT_ROOT}/data/images/textures/brick0512_C.png
	    ${SL_PROJECT_ROOT}/data/images/textures/brick*.jpg
	    ${SL_PROJECT_ROOT}/data/images/textures/CompileError.png
	    ${SL_PROJECT_ROOT}/data/images/textures/Checkerboard0512_C.png
	    ${SL_PROJECT_ROOT}/data/images/textures/Chess0256_C.bmp
	    ${SL_PROJECT_ROOT}/data/images/textures/cursor.png
	    ${SL_PROJECT_ROOT}/data/images/textures/Desert*_C.jpg
	    ${SL_PROJECT_ROOT}/data/images/textures/earth*.jpg
	    ${SL_PROJECT_ROOT}/data/images/textures/features_stones.png
	    ${SL_PROJECT_ROOT}/data/images/textures/grass0512_C.jpg
	    ${SL_PROJECT_ROOT}/data/images/textures/gray_0256_C.jpg
	    ${SL_PROJECT_ROOT}/data/images/textures/i*_0000b.png
	    ${SL_PROJECT_ROOT}/data/images/textures/LiveVideoError.png
	    ${SL_PROJECT_ROOT}/data/images/textures/LogoCPVR_256L.png
	    ${SL_PROJECT_ROOT}/data/images/textures/MuttenzerBox*.png
	    ${SL_PROJECT_ROOT}/data/images/textures/Pool*.png
	    ${SL_PROJECT_ROOT}/data/images/textures/rusty-metal_2048*.jpg
	    ${SL_PROJECT_ROOT}/data/images/textures/Testmap_0512_C.png
	    ${SL_PROJECT_ROOT}/data/images/textures/tile1_0256_C.jpg
	    ${SL_PROJECT_ROOT}/data/images/textures/tree1_1024_C.png
	    ${SL_PROJECT_ROOT}/data/images/textures/Vision*.png
	    ${SL_PROJECT_ROOT}/data/images/textures/wood*.jpg
	    ${SL_PROJECT_ROOT}/data/images/textures/Wave_radial10_256C.jpg
	    )

	file(GLOB_RECURSE VIDEOS
	    ${SL_PROJECT_ROOT}/data/videos/street3.mp4
	    )

	file(GLOB_RECURSE FONTS
	    ${SL_PROJECT_ROOT}/data/images/fonts/*.png
	    ${SL_PROJECT_ROOT}/data/images/fonts/*.ttf
	    )

	# If you add new models you must delete ${CMAKE_CURRENT_LIST_DIR}/src/main/assets
	file(GLOB_RECURSE MODELS
	    "${SL_PROJECT_ROOT}/data/models/3DS/*"
	    "${SL_PROJECT_ROOT}/data/models/DAE/*"
	    "${SL_PROJECT_ROOT}/data/models/FBX/*"		
	    #${SL_PROJECT_ROOT}/data/models/GLTF/* # exclude these models from releas apk
	    #${SL_PROJECT_ROOT}/data/models/PLY/* # exclude these models from releas apk
	    )
		
	#message(STATUS "models: ${MODELS}")

	file(GLOB_RECURSE SHADERS
	    ${SL_PROJECT_ROOT}/data/shaders/*.vert
	    ${SL_PROJECT_ROOT}/data/shaders/*.frag
	    )

	file(GLOB_RECURSE CALIBRATIONS
	    ${SL_PROJECT_ROOT}/data/calibrations/calib_in_params.yml
	    #${SL_PROJECT_ROOT}/data/calibrations/ORBvoc.bin
	    ${SL_PROJECT_ROOT}/data/calibrations/voc_fbow.bin
	    ${SL_PROJECT_ROOT}/data/calibrations/lbfmodel.yaml
	    ${SL_PROJECT_ROOT}/data/calibrations/aruco_detector_params.yml
	    )

	file(GLOB_RECURSE CONFIG
		${SL_PROJECT_ROOT}/data/config/dummyFile.txt
	    )
	
	if(FALSE)
		file(GLOB_RECURSE 
			TEST
			RELATIVE
			${SL_PROJECT_ROOT}/data/
			${SL_PROJECT_ROOT}/data/test1/*
		)
		
		file(MAKE_DIRECTORY ${TARGET_DIR}/test1)
		message(STATUS "TEST: ${TEST}")
		foreach(filetocopy ${TEST})
			message(STATUS "${SL_PROJECT_ROOT}/data/${filetocopy}")
			get_filename_component(filepath ${filetocopy} DIRECTORY)
			message(STATUS "${TARGET_DIR}/${filepath}")
			#file(MAKE_DIRECTORY ${filepath})
			
			file(COPY "${SL_PROJECT_ROOT}/data/${filetocopy}" DESTINATION "${TARGET_DIR}/${filepath}" FILE_PERMISSIONS OWNER_READ OWNER_WRITE GROUP_WRITE GROUP_READ WORLD_READ)
		
		endforeach()
	endif()

	# Distribution
	if("${SYSTEM_NAME_UPPER}" STREQUAL "ANDROID")
		message(STATUS "Copying resources for android to ${TARGET_DIR} (BundleResourcesAppDemoSLProject.cmake)")

		file(MAKE_DIRECTORY
		    ${TARGET_DIR}/fonts
		    ${TARGET_DIR}/textures
		    ${TARGET_DIR}/videos
		    ${TARGET_DIR}/models
		    ${TARGET_DIR}/shaders
		    ${TARGET_DIR}/calibrations
		    ${TARGET_DIR}/config
		    )

		file(COPY ${FONTS}          DESTINATION ${TARGET_DIR}/fonts)
		file(COPY ${TEXTURES}       DESTINATION ${TARGET_DIR}/textures)
		file(COPY ${VIDEOS}         DESTINATION ${TARGET_DIR}/videos)
		#file(COPY ${MODELS}         DESTINATION ${TARGET_DIR}/models)
		install(${MODELS}         DESTINATION ${TARGET_DIR}/models)
		file(COPY ${SHADERS}        DESTINATION ${TARGET_DIR}/shaders)
		file(COPY ${CALIBRATIONS}   DESTINATION ${TARGET_DIR}/calibrations)
		file(COPY ${CONFIG}         DESTINATION ${TARGET_DIR}/config)
		
	elseif("${SYSTEM_NAME_UPPER}" STREQUAL "IOS")
		message(STATUS "Copying resources for iOS to ${TARGET_DIR} (BundleResourcesAppDemoSLProject.cmake)")
		

		#install(${TEST}         DESTINATION ${TARGET_DIR})
		# In this case we copy the selected resources to the build directory. In this way we can use a trick
		# to add a folder reference (blue folders in Xcode) that allows us to preserve the directory structure in the bundle and on iOS device.
		if(FALSE)
			
			file(MAKE_DIRECTORY
			    ${TARGET_DIR}/fonts
			    ${TARGET_DIR}/textures
			    ${TARGET_DIR}/videos
			    ${TARGET_DIR}/models
			    ${TARGET_DIR}/shaders
			    ${TARGET_DIR}/calibrations
			    ${TARGET_DIR}/config
			    )
			
			file(COPY ${FONTS}          DESTINATION ${TARGET_DIR}/fonts)
			file(COPY ${TEXTURES}       DESTINATION ${TARGET_DIR}/textures)
			file(COPY ${VIDEOS}         DESTINATION ${TARGET_DIR}/videos)
			file(COPY ${MODELS}         DESTINATION ${TARGET_DIR}/models)
			file(COPY ${SHADERS}        DESTINATION ${TARGET_DIR}/shaders)
			file(COPY ${CALIBRATIONS}   DESTINATION ${TARGET_DIR}/calibrations)
			file(COPY ${CONFIG}         DESTINATION ${TARGET_DIR}/config)
		
		endif()
		
	endif()

endfunction(copy_resources_slprojectdemo)
