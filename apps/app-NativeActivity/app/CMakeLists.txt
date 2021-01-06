# For more information about using CMake with Android Studio, read the
# documentation: https://d.android.com/studio/projects/add-native-code.html
#find_package(OpenCV REQUIRED)

set(target native-lib)

set(headers
    	#${ANDROID_NDK}/sources/android/native_app_glue/android_native_app_glue.h
		#${CMAKE_CURRENT_SOURCE_DIR}/src/main/cpp/AppDemoNativeSensorsInterface.h
		#${SL_PROJECT_ROOT}/apps/app_demo_wai/source/Gui/AppDemoWaiGui.h
		#${SL_PROJECT_ROOT}/apps/app_demo_wai/source/Gui/AppDemoGuiInfosDialog.h
		#${SL_PROJECT_ROOT}/apps/app_demo_wai/source/Gui/GUIPreferences.h
		#${SL_PROJECT_ROOT}/apps/app_demo_wai/source/GuiPrefab/AppDemoGuiAbout.h
		#${SL_PROJECT_ROOT}/apps/app_demo_wai/source/GuiPrefab/AppDemoGuiError.h
		#${SL_PROJECT_ROOT}/apps/app_demo_wai/source/GuiPrefab/AppDemoGuiInfosFrameworks.h
		#${SL_PROJECT_ROOT}/apps/app_demo_wai/source/GuiPrefab/AppDemoGuiInfosMapNodeTransform.h
		#${SL_PROJECT_ROOT}/apps/app_demo_wai/source/GuiPrefab/AppDemoGuiInfosScene.h
		#${SL_PROJECT_ROOT}/apps/app_demo_wai/source/GuiPrefab/AppDemoGuiInfosSensors.h
		#${SL_PROJECT_ROOT}/apps/app_demo_wai/source/GuiPrefab/AppDemoGuiInfosTracking.h
		#${SL_PROJECT_ROOT}/apps/app_demo_wai/source/GuiPrefab/AppDemoGuiProperties.h
		#${SL_PROJECT_ROOT}/apps/app_demo_wai/source/GuiPrefab/AppDemoGuiSceneGraph.h
		#${SL_PROJECT_ROOT}/apps/app_demo_wai/source/GuiPrefab/AppDemoGuiSlamLoad.h
		#${SL_PROJECT_ROOT}/apps/app_demo_wai/source/GuiPrefab/AppDemoGuiStatsDebugTiming.h
		#${SL_PROJECT_ROOT}/apps/app_demo_wai/source/GuiPrefab/AppDemoGuiStatsTiming.h
		#${SL_PROJECT_ROOT}/apps/app_demo_wai/source/GuiPrefab/AppDemoGuiStatsVideo.h
		#${SL_PROJECT_ROOT}/apps/app_demo_wai/source/GuiPrefab/AppDemoGuiTrackedMapping.h
		#${SL_PROJECT_ROOT}/apps/app_demo_wai/source/GuiPrefab/AppDemoGuiTransform.h
		#${SL_PROJECT_ROOT}/apps/app_demo_wai/source/GuiPrefab/AppDemoGuiUIPrefs.h
		#${SL_PROJECT_ROOT}/apps/app_demo_wai/source/GuiPrefab/AppDemoGuiVideoControls.h
		#${SL_PROJECT_ROOT}/apps/app_demo_wai/source/GuiPrefab/AppDemoGuiVideoStorage.h
		#${SL_PROJECT_ROOT}/apps/app_demo_wai/source/GuiPrefab/AppDemoGuiSlamParam.h
    ${SL_PROJECT_ROOT}/apps/source/wai/WAIAppTest.h
		#${SL_PROJECT_ROOT}/apps/app_demo_wai/source/GLSLextractor.h
		#${SL_PROJECT_ROOT}/apps/app_demo_wai/source/GLSLHessian.h
		#${SL_PROJECT_ROOT}/apps/app_demo_wai/source/AppDirectories.h
		#${SL_PROJECT_ROOT}/apps/app_demo_wai/source/AppWAIScene.h
		#${SL_PROJECT_ROOT}/apps/app_demo_wai/source/WAICalibration.h
		#${SL_PROJECT_ROOT}/apps/app_demo_wai/source/WAIUtils.h
		#${SL_PROJECT_ROOT}/apps/source/wai/WAIMapStorage.h
		#${SL_PROJECT_ROOT}/apps/source/wai/AppWAISlamParamHelper.h
		#${SL_PROJECT_ROOT}/apps/source/wai/ImGuiWrapper.h
		#${SL_PROJECT_ROOT}/apps/source/wai/FeatureExtractorFactory.h
	${SENS_ROOT}/SENS.h
	${SENS_ROOT}/SENSCamera.h
	${SENS_ROOT}/SENSFrame.h
	${SENS_ROOT}/SENSException.h
	${SENS_ROOT}/SENSUtils.h
	${SENS_ROOT}/SENSVideoStream.h
	${SENS_ROOT}/android/SENSNdkCamera.h
	${SENS_ROOT}/android/SENSNdkCameraUtils.h
    )

set(sources
    ${CMAKE_CURRENT_SOURCE_DIR}/src/main/cpp/AppDemoNativeWAIMain.cpp
	#${CMAKE_CURRENT_SOURCE_DIR}/src/main/cpp/AppDemoNativeSensorsInterface.cpp
	#${SL_PROJECT_ROOT}/apps/app_demo_wai/source/Gui/AppDemoWaiGui.cpp
	#${SL_PROJECT_ROOT}/apps/app_demo_wai/source/Gui/AppDemoGuiInfosDialog.cpp
	#${SL_PROJECT_ROOT}/apps/app_demo_wai/source/Gui/GUIPreferences.cpp
    #${SL_PROJECT_ROOT}/apps/app_demo_wai/source/GuiPrefab/AppDemoGuiAbout.cpp
    #${SL_PROJECT_ROOT}/apps/app_demo_wai/source/GuiPrefab/AppDemoGuiError.cpp
	#${SL_PROJECT_ROOT}/apps/app_demo_wai/source/GuiPrefab/AppDemoGuiInfosFrameworks.cpp
	#${SL_PROJECT_ROOT}/apps/app_demo_wai/source/GuiPrefab/AppDemoGuiInfosMapNodeTransform.cpp
	#${SL_PROJECT_ROOT}/apps/app_demo_wai/source/GuiPrefab/AppDemoGuiInfosScene.cpp
	#${SL_PROJECT_ROOT}/apps/app_demo_wai/source/GuiPrefab/AppDemoGuiInfosSensors.cpp
	#${SL_PROJECT_ROOT}/apps/app_demo_wai/source/GuiPrefab/AppDemoGuiInfosTracking.cpp
	#${SL_PROJECT_ROOT}/apps/app_demo_wai/source/GuiPrefab/AppDemoGuiProperties.cpp
	#${SL_PROJECT_ROOT}/apps/app_demo_wai/source/GuiPrefab/AppDemoGuiSceneGraph.cpp
	#${SL_PROJECT_ROOT}/apps/app_demo_wai/source/GuiPrefab/AppDemoGuiSlamLoad.cpp
	#${SL_PROJECT_ROOT}/apps/app_demo_wai/source/GuiPrefab/AppDemoGuiStatsDebugTiming.cpp
	#${SL_PROJECT_ROOT}/apps/app_demo_wai/source/GuiPrefab/AppDemoGuiStatsTiming.cpp
	#${SL_PROJECT_ROOT}/apps/app_demo_wai/source/GuiPrefab/AppDemoGuiStatsVideo.cpp
	#${SL_PROJECT_ROOT}/apps/app_demo_wai/source/GuiPrefab/AppDemoGuiTrackedMapping.cpp
	#${SL_PROJECT_ROOT}/apps/app_demo_wai/source/GuiPrefab/AppDemoGuiTransform.cpp
	#${SL_PROJECT_ROOT}/apps/app_demo_wai/source/GuiPrefab/AppDemoGuiUIPrefs.cpp
	#${SL_PROJECT_ROOT}/apps/app_demo_wai/source/GuiPrefab/AppDemoGuiVideoControls.cpp
	#${SL_PROJECT_ROOT}/apps/app_demo_wai/source/GuiPrefab/AppDemoGuiVideoStorage.cpp
	#${SL_PROJECT_ROOT}/apps/app_demo_wai/source/GuiPrefab/AppDemoGuiSlamParam.cpp
    ${SL_PROJECT_ROOT}/apps/source/wai/WAIAppTest.cpp
		#${SL_PROJECT_ROOT}/apps/app_demo_wai/source/GLSLextractor.cpp
		#${SL_PROJECT_ROOT}/apps/app_demo_wai/source/GLSLHessian.cpp
		#${SL_PROJECT_ROOT}/apps/app_demo_wai/source/AppWAIScene.cpp
		#${SL_PROJECT_ROOT}/apps/app_demo_wai/source/WAICalibration.cpp
		#${SL_PROJECT_ROOT}/apps/source/wai/WAIMapStorage.cpp
    #${SL_PROJECT_ROOT}/apps/source/wai/ImGuiWrapper.cpp
    #${SL_PROJECT_ROOT}/apps/source/wai/FeatureExtractorFactory.cpp
	${SENS_ROOT}/SENSCamera.cpp
	${SENS_ROOT}/SENSUtils.cpp
	${SENS_ROOT}/SENSVideoStream.cpp
	${SENS_ROOT}/android/SENSNdkCamera.cpp
	${SENS_ROOT}/android/SENSNdkCameraUtils.cpp
    )

set(CMAKE_SHARED_LINKER_FLAGS
    "${CMAKE_SHARED_LINKER_FLAGS} -u ANativeActivity_onCreate")

add_library(native_app_glue STATIC
    ${ANDROID_NDK}/sources/android/native_app_glue/android_native_app_glue.c)

add_library(${target}
    SHARED
        ${sources}
        ${headers}
    )

target_include_directories(${target}
    PRIVATE
        ${ANDROID_NDK}/sources/android/native_app_glue/
        ${CMAKE_CURRENT_SOURCE_DIR}/src/main/cpp/
        ${SL_PROJECT_ROOT}/apps/source/wai/
		${SENS_ROOT}
    PUBLIC
        #${SL_PROJECT_ROOT}/apps/app_demo_wai/source
        #${SL_PROJECT_ROOT}/apps/app_demo_wai/source/Gui
        #${SL_PROJECT_ROOT}/apps/app_demo_wai/source/GuiPrefab
        #${SL_PROJECT_ROOT}/lib-WAI/source
        ${OpenCV_INCLUDE_DIR}
        ${g2o_INCLUDE_DIR}
        ${SL_PROJECT_ROOT}/externals/eigen
        ${SL_PROJECT_ROOT}/externals/fbow
        ${DEFAULT_INCLUDE_DIRECTORIES}
    )

target_link_libraries(${target}
    PRIVATE
        android
        native_app_glue
        EGL
        GLESv3
        log
        camera2ndk
        mediandk
        #lib-WAI
        lib-fbow
        lib-DBoW2
        lib-SLProject

    PUBLIC
        libc++_shared.so
        z

    INTERFACE
    )

#
# Copy APK contents
#
file(GLOB_RECURSE TEXTURES
    ${SL_PROJECT_ROOT}/data/images/textures/CompileError.png
    ${SL_PROJECT_ROOT}/data/images/textures/cursor.png
    ${SL_PROJECT_ROOT}/data/images/textures/LiveVideoError.png
    ${SL_PROJECT_ROOT}/data/images/textures/LogoCPVR_256L.png
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
    ${SL_PROJECT_ROOT}/data/models/*
    )

file(GLOB_RECURSE SHADERS
    ${SL_PROJECT_ROOT}/data/shaders/*.vert
    ${SL_PROJECT_ROOT}/data/shaders/*.frag
    )

file(GLOB_RECURSE CALIBRATIONS
    ${SL_PROJECT_ROOT}/data/calibrations/calib_in_params.yml
    ${SL_PROJECT_ROOT}/data/calibrations/aruco_detector_params.yml
    ${SL_PROJECT_ROOT}/data/calibrations/cam_calibration_main.xml
    )

file(GLOB_RECURSE VOC
	${SL_PROJECT_ROOT}/data/calibrations/ORBvoc.bin
	${SL_PROJECT_ROOT}/data/calibrations/voc_fbow.bin
	)

set(APK_ASSETS
    ${CMAKE_CURRENT_LIST_DIR}/src/main/assets)

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
