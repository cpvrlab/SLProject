# Definition of resource files for SLProject-demo application and distribution to iOS and Android bundles
function(copy_resources_slprojectdemo TARGET_DIR)
	
	#definition of directory to copy
	#(filenames are defined relative to SL_PROJECT_DATA_ROOT so that we can easily concetenate the target fullpath-filename)
	set(SL_PROJECT_DATA_ROOT ${SL_PROJECT_ROOT}/data/)
	
	# Definition
	file(GLOB_RECURSE 
			TEXTURES
			RELATIVE
			${SL_PROJECT_DATA_ROOT}
			${SL_PROJECT_ROOT}/data/images/textures/brick0512_C.png
			${SL_PROJECT_ROOT}/data/images/textures/brick*.jpg
			${SL_PROJECT_ROOT}/data/images/textures/CompileError.png
			${SL_PROJECT_ROOT}/data/images/textures/Checkerboard0512_C.png
			${SL_PROJECT_ROOT}/data/images/textures/Chess0256_C.bmp
			${SL_PROJECT_ROOT}/data/images/textures/cursor.png
			${SL_PROJECT_ROOT}/data/images/textures/Desert*_C.jpg
			${SL_PROJECT_ROOT}/data/images/textures/earth*.jpg
			${SL_PROJECT_ROOT}/data/images/textures/earth*.png
			${SL_PROJECT_ROOT}/data/images/textures/earth*.ktx2
			${SL_PROJECT_ROOT}/data/images/textures/features_stones.png
			${SL_PROJECT_ROOT}/data/images/textures/grass0512_C.jpg
			${SL_PROJECT_ROOT}/data/images/textures/gray_0256_C.jpg
			${SL_PROJECT_ROOT}/data/images/textures/i*_0000b.png
			${SL_PROJECT_ROOT}/data/images/textures/LiveVideoError.png
			${SL_PROJECT_ROOT}/data/images/textures/LogoCPVR_256L.png
			${SL_PROJECT_ROOT}/data/images/textures/MuttenzerBox*.png
			${SL_PROJECT_ROOT}/data/images/textures/Particle*.png
			${SL_PROJECT_ROOT}/data/images/textures/Pool*.png
			${SL_PROJECT_ROOT}/data/images/textures/rusty-metal_2048*.jpg
			${SL_PROJECT_ROOT}/data/images/textures/Testmap_1024_*.jpg
			${SL_PROJECT_ROOT}/data/images/textures/TexNotFound.png
			${SL_PROJECT_ROOT}/data/images/textures/tile1_0256_C.jpg
			${SL_PROJECT_ROOT}/data/images/textures/tree1_1024_C.png
			${SL_PROJECT_ROOT}/data/images/textures/Vision*.png
			${SL_PROJECT_ROOT}/data/images/textures/wood*.jpg
			${SL_PROJECT_ROOT}/data/images/textures/Wave_radial10_256C.jpg
			${SL_PROJECT_ROOT}/data/images/textures/gold-scuffed*.png
			${SL_PROJECT_ROOT}/data/images/textures/env_barce_rooftop.hdr
			${SL_PROJECT_ROOT}/data/mediapipe/*.binarypb
			${SL_PROJECT_ROOT}/data/mediapipe/*.tflite
			${SL_PROJECT_ROOT}/data/mediapipe/*.txt
	    )

	file(GLOB_RECURSE 
			VIDEOS
			RELATIVE
			${SL_PROJECT_DATA_ROOT}
			${SL_PROJECT_ROOT}/data/videos/street3.mp4
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
			"${SL_PROJECT_ROOT}/data/models/3DS/*"
			"${SL_PROJECT_ROOT}/data/models/DAE/*"
			"${SL_PROJECT_ROOT}/data/models/FBX/*"
			"${SL_PROJECT_ROOT}/data/models/GLTF/*"
			"${SL_PROJECT_ROOT}/data/models/PLY/*"
	    )

	file(GLOB_RECURSE
			ERLEB-AR
			RELATIVE
			${SL_PROJECT_DATA_ROOT}
			"${SL_PROJECT_ROOT}/data/erleb-AR/models/*"
			)
	file(GLOB_RECURSE 
			SHADERS
			RELATIVE
			${SL_PROJECT_DATA_ROOT}
	    	${SL_PROJECT_ROOT}/data/shaders/*.vert
	    	${SL_PROJECT_ROOT}/data/shaders/*.frag
			${SL_PROJECT_ROOT}/data/shaders/*.glsl
	    )

	file(GLOB_RECURSE 
			CALIBRATIONS
			RELATIVE
			${SL_PROJECT_DATA_ROOT}
			${SL_PROJECT_ROOT}/data/calibrations/calib_in_params.yml
			#${SL_PROJECT_ROOT}/data/calibrations/ORBvoc.bin
			${SL_PROJECT_ROOT}/data/calibrations/voc_fbow.bin
			${SL_PROJECT_ROOT}/data/calibrations/lbfmodel.yaml
			${SL_PROJECT_ROOT}/data/calibrations/haarcascade_frontalface_alt2.xml
			${SL_PROJECT_ROOT}/data/calibrations/aruco_detector_params.yml
	    )

	file(GLOB_RECURSE 
			CONFIG
			RELATIVE
			${SL_PROJECT_DATA_ROOT}
			${SL_PROJECT_ROOT}/data/config/dummyFile.txt
	    )
	
	# Distribution
	set(RESOURCES
			${TEXTURES}
			${VIDEOS}
			${ERLEB-AR}
			${FONTS}
			${MODELS}
			${SHADERS}
			${CALIBRATIONS}
			${CONFIG}
		)
		
	foreach(FILE_TO_COPY ${RESOURCES})
		get_filename_component(PATH_TO_COPY ${FILE_TO_COPY} DIRECTORY)
		file(COPY "${SL_PROJECT_DATA_ROOT}/${FILE_TO_COPY}" DESTINATION "${TARGET_DIR}/${PATH_TO_COPY}")
	endforeach()

endfunction(copy_resources_slprojectdemo)
