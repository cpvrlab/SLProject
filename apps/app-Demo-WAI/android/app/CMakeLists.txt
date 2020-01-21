#For more information about using CMake with Android Studio, read the
#documentation : https: //d.android.com/studio/projects/add-native-code.html
#find_package(OpenCV REQUIRED)

set(target native-lib)

set(headers
    ${SL_PROJECT_ROOT}/apps/app-Demo-WAI/source/Gui/AppDemoWaiGui.h
    ${SL_PROJECT_ROOT}/apps/app-Demo-WAI/source/Gui/AppDemoGuiInfosDialog.h
    ${SL_PROJECT_ROOT}/apps/app-Demo-WAI/source/Gui/GUIPreferences.h
    ${SL_PROJECT_ROOT}/apps/app-Demo-WAI/source/GuiPrefab/AppDemoGuiAbout.h
    ${SL_PROJECT_ROOT}/apps/app-Demo-WAI/source/GuiPrefab/AppDemoGuiError.h
    ${SL_PROJECT_ROOT}/apps/app-Demo-WAI/source/GuiPrefab/AppDemoGuiInfosFrameworks.h
    ${SL_PROJECT_ROOT}/apps/app-Demo-WAI/source/GuiPrefab/AppDemoGuiInfosMapNodeTransform.h
    ${SL_PROJECT_ROOT}/apps/app-Demo-WAI/source/GuiPrefab/AppDemoGuiInfosScene.h
    ${SL_PROJECT_ROOT}/apps/app-Demo-WAI/source/GuiPrefab/AppDemoGuiInfosSensors.h
    ${SL_PROJECT_ROOT}/apps/app-Demo-WAI/source/GuiPrefab/AppDemoGuiInfosTracking.h
    ${SL_PROJECT_ROOT}/apps/app-Demo-WAI/source/GuiPrefab/AppDemoGuiProperties.h
    ${SL_PROJECT_ROOT}/apps/app-Demo-WAI/source/GuiPrefab/AppDemoGuiSceneGraph.h
    ${SL_PROJECT_ROOT}/apps/app-Demo-WAI/source/GuiPrefab/AppDemoGuiSlamLoad.h
    ${SL_PROJECT_ROOT}/apps/app-Demo-WAI/source/GuiPrefab/AppDemoGuiStatsDebugTiming.h
    ${SL_PROJECT_ROOT}/apps/app-Demo-WAI/source/GuiPrefab/AppDemoGuiStatsTiming.h
    ${SL_PROJECT_ROOT}/apps/app-Demo-WAI/source/GuiPrefab/AppDemoGuiStatsVideo.h
    ${SL_PROJECT_ROOT}/apps/app-Demo-WAI/source/GuiPrefab/AppDemoGuiTrackedMapping.h
    ${SL_PROJECT_ROOT}/apps/app-Demo-WAI/source/GuiPrefab/AppDemoGuiTransform.h
    ${SL_PROJECT_ROOT}/apps/app-Demo-WAI/source/GuiPrefab/AppDemoGuiUIPrefs.h
    ${SL_PROJECT_ROOT}/apps/app-Demo-WAI/source/GuiPrefab/AppDemoGuiVideoControls.h
    ${SL_PROJECT_ROOT}/apps/app-Demo-WAI/source/GuiPrefab/AppDemoGuiVideoStorage.h
    ${SL_PROJECT_ROOT}/apps/app-Demo-WAI/source/GuiPrefab/AppDemoGuiSlamParam.h
    ${SL_PROJECT_ROOT}/apps/app-Demo-WAI/source/WAIApp.h
    ${SL_PROJECT_ROOT}/apps/app-Demo-WAI/source/GLSLextractor.h
    ${SL_PROJECT_ROOT}/apps/app-Demo-WAI/source/GLSLHessian.h
    ${SL_PROJECT_ROOT}/apps/app-Demo-WAI/source/AppDirectories.h
    ${SL_PROJECT_ROOT}/apps/app-Demo-WAI/source/AppWAIScene.h
    ${SL_PROJECT_ROOT}/apps/app-Demo-WAI/source/WAICalibration.h
    ${SL_PROJECT_ROOT}/apps/app-Demo-WAI/source/WAIUtils.h
    ${SL_PROJECT_ROOT}/apps/source/wai/WAIMapStorage.h
    ${SL_PROJECT_ROOT}/apps/source/wai/AppWAISlamParamHelper.h
	${SL_PROJECT_ROOT}/apps/source/wai/ImGuiWrapper.h
    ${SL_PROJECT_ROOT}/apps/source/wai/FeatureExtractorFactory.h
	${SL_PROJECT_ROOT}/apps/source/sens/SENS.h
	${SL_PROJECT_ROOT}/apps/source/sens/SENSCamera.h
	${SL_PROJECT_ROOT}/apps/source/sens/SENSFrame.h
	${SL_PROJECT_ROOT}/apps/source/sens/SENSException.h
	${SL_PROJECT_ROOT}/apps/source/sens/SENSUtils.h
	${SL_PROJECT_ROOT}/apps/source/sens/SENSVideoStream.h
	${SL_PROJECT_ROOT}/apps/source/sens/android/SENSNdkCamera.h
	${SL_PROJECT_ROOT}/apps/source/sens/android/SENSNdkCameraUtils.h
    )

set(sources
    ${SL_PROJECT_ROOT}/apps/app-Demo-WAI/source/Gui/AppDemoWaiGui.cpp
    ${SL_PROJECT_ROOT}/apps/app-Demo-WAI/source/Gui/AppDemoGuiInfosDialog.cpp
    ${SL_PROJECT_ROOT}/apps/app-Demo-WAI/source/Gui/GUIPreferences.cpp
    ${SL_PROJECT_ROOT}/apps/app-Demo-WAI/source/GuiPrefab/AppDemoGuiAbout.cpp
    ${SL_PROJECT_ROOT}/apps/app-Demo-WAI/source/GuiPrefab/AppDemoGuiError.cpp
    ${SL_PROJECT_ROOT}/apps/app-Demo-WAI/source/GuiPrefab/AppDemoGuiInfosFrameworks.cpp
    ${SL_PROJECT_ROOT}/apps/app-Demo-WAI/source/GuiPrefab/AppDemoGuiInfosMapNodeTransform.cpp
    ${SL_PROJECT_ROOT}/apps/app-Demo-WAI/source/GuiPrefab/AppDemoGuiInfosScene.cpp
    ${SL_PROJECT_ROOT}/apps/app-Demo-WAI/source/GuiPrefab/AppDemoGuiInfosSensors.cpp
    ${SL_PROJECT_ROOT}/apps/app-Demo-WAI/source/GuiPrefab/AppDemoGuiInfosTracking.cpp
    ${SL_PROJECT_ROOT}/apps/app-Demo-WAI/source/GuiPrefab/AppDemoGuiProperties.cpp
    ${SL_PROJECT_ROOT}/apps/app-Demo-WAI/source/GuiPrefab/AppDemoGuiSceneGraph.cpp
    ${SL_PROJECT_ROOT}/apps/app-Demo-WAI/source/GuiPrefab/AppDemoGuiSlamLoad.cpp
    ${SL_PROJECT_ROOT}/apps/app-Demo-WAI/source/GuiPrefab/AppDemoGuiStatsDebugTiming.cpp
    ${SL_PROJECT_ROOT}/apps/app-Demo-WAI/source/GuiPrefab/AppDemoGuiStatsTiming.cpp
    ${SL_PROJECT_ROOT}/apps/app-Demo-WAI/source/GuiPrefab/AppDemoGuiStatsVideo.cpp
    ${SL_PROJECT_ROOT}/apps/app-Demo-WAI/source/GuiPrefab/AppDemoGuiTrackedMapping.cpp
    ${SL_PROJECT_ROOT}/apps/app-Demo-WAI/source/GuiPrefab/AppDemoGuiTransform.cpp
    ${SL_PROJECT_ROOT}/apps/app-Demo-WAI/source/GuiPrefab/AppDemoGuiUIPrefs.cpp
    ${SL_PROJECT_ROOT}/apps/app-Demo-WAI/source/GuiPrefab/AppDemoGuiVideoControls.cpp
    ${SL_PROJECT_ROOT}/apps/app-Demo-WAI/source/GuiPrefab/AppDemoGuiVideoStorage.cpp
    ${SL_PROJECT_ROOT}/apps/app-Demo-WAI/source/GuiPrefab/AppDemoGuiSlamParam.cpp
    ${SL_PROJECT_ROOT}/apps/app-Demo-WAI/source/WAIApp.cpp
    ${SL_PROJECT_ROOT}/apps/app-Demo-WAI/source/GLSLextractor.cpp
    ${SL_PROJECT_ROOT}/apps/app-Demo-WAI/source/GLSLHessian.cpp
    ${SL_PROJECT_ROOT}/apps/app-Demo-WAI/source/AppWAIScene.cpp
    ${SL_PROJECT_ROOT}/apps/app-Demo-WAI/source/WAICalibration.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/main/cpp/AppDemoWAIAndroidJNI.cpp
    ${SL_PROJECT_ROOT}/apps/source/wai/WAIMapStorage.cpp
	${SL_PROJECT_ROOT}/apps/source/wai/ImGuiWrapper.cpp
    ${SL_PROJECT_ROOT}/apps/source/wai/FeatureExtractorFactory.cpp
	${SL_PROJECT_ROOT}/apps/source/sens/SENSUtils.cpp
	${SL_PROJECT_ROOT}/apps/source/sens/SENSVideoStream.cpp
	${SL_PROJECT_ROOT}/apps/source/sens/android/SENSNdkCamera.cpp
	${SL_PROJECT_ROOT}/apps/source/sens/android/SENSNdkCameraUtils.cpp
    )


add_library(${target}
    SHARED
    ${sources}
    ${headers}
    )

target_include_directories(${target}
    PRIVATE
    ${SL_PROJECT_ROOT}/apps/source/wai/
    PUBLIC
    ${SL_PROJECT_ROOT}/apps/app-Demo-WAI/source
    ${SL_PROJECT_ROOT}/apps/app-Demo-WAI/source/Gui
    ${SL_PROJECT_ROOT}/apps/app-Demo-WAI/source/GuiPrefab
    ${SL_PROJECT_ROOT}/lib-WAI/source
    ${OpenCV_INCLUDE_DIR}
    ${g2o_INCLUDE_DIR}
    ${SL_PROJECT_ROOT}/externals/eigen
    ${SL_PROJECT_ROOT}/externals/DBoW2
	${SL_PROJECT_ROOT}/apps/source/sens/
	${SL_PROJECT_ROOT}/apps/source/sens/android
    ${DEFAULT_INCLUDE_DIRECTORIES}
    )

target_link_libraries(${target}
    PRIVATE
		android
		log
		camera2ndk
		mediandk

        lib-WAI
        lib-SLProject
    PUBLIC
        libc++_shared.so
        ${log-lib}
        z

    INTERFACE
    )

#
#Copy APK contents
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
    ${SL_PROJECT_ROOT}/data/calibrations/ORBvoc.bin
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
    )

file(COPY ${FONTS}          DESTINATION ${APK_ASSETS}/images/fonts)
file(COPY ${TEXTURES}       DESTINATION ${APK_ASSETS}/images/textures)
file(COPY ${VIDEOS}         DESTINATION ${APK_ASSETS}/videos)
file(COPY ${MODELS}         DESTINATION ${APK_ASSETS}/models)
file(COPY ${SHADERS}        DESTINATION ${APK_ASSETS}/shaders)
file(COPY ${CALIBRATIONS}   DESTINATION ${APK_ASSETS}/calibrations)
file(COPY ${CONFIG}         DESTINATION ${APK_ASSETS}/config)
