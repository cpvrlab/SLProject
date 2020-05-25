# 
# CMake configuration for Demo-SL (app-Demo-SLProject) application for ios
#

#find system frameworks (libraries)	
find_library(AVFOUNDATION AVFoundation required)
find_library(COREGRAPHICS CoreGraphics required)
find_library(COREVIDEO CoreVideo required)
find_library(COREMOTION CoreMotion required)
find_library(COREMEDIA CoreMedia required)
find_library(UIKIT UIKit required)
find_library(OPENGLES OpenGLES required)
find_library(GLKIT GLKit required)
find_library(CORELOCATION CoreLocation required)

set(target Demo-SL)
set(DEVELOPMENT_TEAM_ID "858Y9EWZ4B")               # <== Set to your team ID from Apple
set(APP_BUNDLE_IDENTIFIER "ch.bfh.ti.cpvrlab.demo-sl") # <== Set to your app's bundle identifier (Attention the erlebar one is with "cpvr")
set(PROVISIONING_PROFILE "Demo SL Development")
set(CODE_SIGN_IDENTITY "Apple Development: Michael Göttlicher (KMX9985DR7)")  # <== Set to your preferred code sign identity, to see list: /usr/bin/env xcrun security find-identity -v -p codesigning

set(DEPLOYMENT_TARGET 8.0)                                  # <== Set your deployment target version of iOS
set(DEVICE_FAMILY "1,2")                                      # <== Set to "1" to target iPhone, set to "2" to target iPad, set to "1,2" to target both

include(BundleUtilities)

set(PRODUCT_NAME ${target})
set(EXECUTABLE_NAME ${target})
set(MACOSX_BUNDLE_EXECUTABLE_NAME ${target})
set(MACOSX_BUNDLE_INFO_STRING ${APP_BUNDLE_IDENTIFIER})
set(MACOSX_BUNDLE_GUI_IDENTIFIER ${APP_BUNDLE_IDENTIFIER})
set(MACOSX_BUNDLE_BUNDLE_NAME ${APP_BUNDLE_IDENTIFIER})
#set(MACOSX_BUNDLE_ICON_FILE "")
set(MACOSX_BUNDLE_LONG_VERSION_STRING "1.0")
set(MACOSX_BUNDLE_SHORT_VERSION_STRING "1.0")
set(MACOSX_BUNDLE_BUNDLE_VERSION "1.0")
set(MACOSX_BUNDLE_COPYRIGHT "Copyright Berner Fachhochschule (Marcus Hudritsch)")
set(MACOSX_DEPLOYMENT_TARGET ${DEPLOYMENT_TARGET})

# Turn on ARC (TODO: MOVE TO COMPILE OPTIONS!)
#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fobjc-arc")

set(include_path "${CMAKE_CURRENT_SOURCE_DIR}")
set(source_path "${CMAKE_CURRENT_SOURCE_DIR}")
set(compile_definitions SL_STARTSCENE=SID_Minimal)

set(headers
    ${SL_PROJECT_ROOT}/apps/app-Demo-SLProject/include/AppDemoGui.h
    ${SL_PROJECT_ROOT}/apps/app-Demo-SLProject/include/AppDemoSceneView.h
	${SL_PROJECT_ROOT}/apps/source/CVCapture.h
	${SL_PROJECT_ROOT}/apps/source/SLApplication.h
	${SL_PROJECT_ROOT}/apps/source/SLInterface.h
    ${SL_PROJECT_ROOT}/apps/source/SLProjectScene.h
	${CMAKE_CURRENT_SOURCE_DIR}/AppDelegate.h
	${CMAKE_CURRENT_SOURCE_DIR}/Utils_iOS.h
	${CMAKE_CURRENT_SOURCE_DIR}/ViewController.h
    )

set(sources
    ${SL_PROJECT_ROOT}/apps/app-Demo-SLProject/source/AppDemoGui.cpp
    ${SL_PROJECT_ROOT}/apps/app-Demo-SLProject/source/AppDemoLoad.cpp
    ${SL_PROJECT_ROOT}/apps/app-Demo-SLProject/source/AppDemoVideo.cpp
    ${SL_PROJECT_ROOT}/apps/app-Demo-SLProject/source/AppDemoSceneView.cpp
	${SL_PROJECT_ROOT}/apps/source/CVCapture.cpp
	${SL_PROJECT_ROOT}/apps/source/SLApplication.cpp
	${SL_PROJECT_ROOT}/apps/source/SLInterface.cpp
    ${SL_PROJECT_ROOT}/apps/source/SLProjectScene.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/AppDelegate.mm
	${CMAKE_CURRENT_SOURCE_DIR}/main.m
	${CMAKE_CURRENT_SOURCE_DIR}/Utils_iOS.mm
	${CMAKE_CURRENT_SOURCE_DIR}/ViewController.mm
	)

# define resources
set(XIB_FILES
	${CMAKE_CURRENT_SOURCE_DIR}/Base.lproj/ViewController_iPad.xib
	${CMAKE_CURRENT_SOURCE_DIR}/Base.lproj/ViewController_iPhone.xib
	)


#calibrations
set(CALIBRATIONS 
	${SL_PROJECT_ROOT}/data/calibrations/voc_fbow.bin
	${SL_PROJECT_ROOT}/data/calibrations/ORBvoc.bin
	${SL_PROJECT_ROOT}/data/calibrations/aruco_detector_params.yml
	${SL_PROJECT_ROOT}/data/calibrations/haarcascade_frontalface_alt2.xml
	${SL_PROJECT_ROOT}/data/calibrations/lbfmodel.yaml
	${SL_PROJECT_ROOT}/data/calibrations/calib_in_params.yml
	)

#models
file(GLOB MODELS_3DS
	${SL_PROJECT_ROOT}/data/models/3DS
	#${SL_PROJECT_ROOT}/data/models/3DS/Halloween
	)
	
file(GLOB MODELS_DAE
	${SL_PROJECT_ROOT}/data/models/DAE
	#${SL_PROJECT_ROOT}/data/models/DAE/AstroBoy
	#${SL_PROJECT_ROOT}/data/models/DAE/Crate
	#${SL_PROJECT_ROOT}/data/models/DAE/Hands
	#${SL_PROJECT_ROOT}/data/models/DAE/Sintel
	#${SL_PROJECT_ROOT}/data/models/DAE/SkinnedCube
	#${SL_PROJECT_ROOT}/data/models/DAE/Table
	)
	
file(GLOB MODELS_FBX
	${SL_PROJECT_ROOT}/data/models/FBX	
	#${SL_PROJECT_ROOT}/data/models/FBX/Axes/*
	#${SL_PROJECT_ROOT}/data/models/FBX/Brunnen.fbx
	#${SL_PROJECT_ROOT}/data/models/FBX/Christoffel/*
	#${SL_PROJECT_ROOT}/data/models/FBX/Duck/*
	#${SL_PROJECT_ROOT}/data/models/FBX/Sunglasses.fbx
	#${SL_PROJECT_ROOT}/data/models/FBX/Suzanne/*
	#${SL_PROJECT_ROOT}/data/models/FBX/Teapot/*
	)	
	
file(GLOB SHADERS ${SL_PROJECT_ROOT}/data/shaders/*)
file(GLOB FONTS ${SL_PROJECT_ROOT}/data/images/fonts/*)
file(GLOB TEXTURES ${SL_PROJECT_ROOT}/data/images/textures/*)
file(GLOB VIDEOS ${SL_PROJECT_ROOT}/data/videos/*)

set(resources
	${XIB_FILES}
	${CALIBRATIONS}
	${SHADERS}
	${FONTS}
	${TEXTURES}
	${VIDEOS}
	${MODELS_3DS}
	${MODELS_DAE}
	${MODELS_FBX}
	)
	
# Group resource files
source_group(Resources FILES ${XIB_FILES})
source_group(Resources\\data\\calibrations FILES ${CALIBRATIONS})
source_group(Resources\\data\\shaders FILES ${SHADERS})
source_group(Resources\\data\\images\\fonts FILES ${FONTS})
source_group(Resources\\data\\images\\textures FILES ${TEXTURES})
source_group(Resources\\data\\models FILES ${MODELS_3DS})
source_group(Resources\\data\\models FILES ${MODELS_DAE})
source_group(Resources\\data\\models FILES ${MODELS_FBX})
source_group(Resources\\data\\videos FILES ${VIDEOS})

add_executable(
        ${target}
		MACOSX_BUNDLE
        ${headers}
        ${sources}
		${resources}
    )

set_target_properties(
        ${target}
        PROPERTIES
        ${DEFAULT_PROJECT_OPTIONS}
        FOLDER "apps"
		
        XCODE_ATTRIBUTE_DEBUG_INFORMATION_FORMAT "dwarf-with-dsym"
        XCODE_ATTRIBUTE_GCC_PREFIX_HEADER "${CMAKE_CURRENT_SOURCE_DIR}/Prefix.pch" #this is a precompiled header!
        RESOURCE "${resources}"
        XCODE_ATTRIBUTE_GCC_PRECOMPILE_PREFIX_HEADER "YES"
        XCODE_ATTRIBUTE_IPHONEOS_DEPLOYMENT_TARGET ${DEPLOYMENT_TARGET}
        XCODE_ATTRIBUTE_CODE_SIGN_IDENTITY ${CODE_SIGN_IDENTITY}
        XCODE_ATTRIBUTE_DEVELOPMENT_TEAM ${DEVELOPMENT_TEAM_ID}
	    XCODE_ATTRIBUTE_PROVISIONING_PROFILE_SPECIFIER ${PROVISIONING_PROFILE}
        XCODE_ATTRIBUTE_TARGETED_DEVICE_FAMILY ${DEVICE_FAMILY}
        MACOSX_BUNDLE_INFO_PLIST ${CMAKE_CURRENT_SOURCE_DIR}/plist.in
        XCODE_ATTRIBUTE_CLANG_ENABLE_OBJC_ARC YES
        XCODE_ATTRIBUTE_COMBINE_HIDPI_IMAGES NO
        XCODE_ATTRIBUTE_INSTALL_PATH "$(LOCAL_APPS_DIR)"
        XCODE_ATTRIBUTE_ENABLE_TESTABILITY NO
        XCODE_ATTRIBUTE_GCC_SYMBOLS_PRIVATE_EXTERN YES
    )
	
target_include_directories(
        ${target}
        PRIVATE
        ${SL_PROJECT_ROOT}/apps/app-Demo-SLProject/include/
        ${SL_PROJECT_ROOT}/apps/source
        ${SL_PROJECT_ROOT}/lib-SLProject/include
        ${SL_PROJECT_ROOT}/externals/lib-SLExternal
        ${SL_PROJECT_ROOT}/externals/lib-SLExternal/imgui
        ${SL_PROJECT_ROOT}/externals/lib-SLExternal/spa
        ${SL_PROJECT_ROOT}/externals/lib-SLExternal/dirent
        ${SL_PROJECT_ROOT}/externals/lib-SLExternal/gl3w
		${SL_PROJECT_ROOT}/externals/lib-SLExternal/gl3w/GL
        ${OpenCV_INCLUDE_DIR}
        ${glfw_INCLUDE_DIR}
        PUBLIC
        INTERFACE
    )

target_link_libraries(
        ${target}
        #PRIVATE
		PUBLIC
        lib-SLProject
        ${PlatformLinkLibs}
        lib-SLExternal
        lib-Utils
        ${OpenCV_LIBS}

		${AVFOUNDATION}
		${COREGRAPHICS}
		${COREVIDEO}
		${COREMOTION}
		${COREMEDIA}
		${UIKIT}
		${OPENGLES}
		${GLKIT}
		${CORELOCATION}
		
        INTERFACE
    )

target_compile_definitions(${target}
    PRIVATE
    ${compile_definitions}
    PUBLIC
    ${DEFAULT_COMPILE_DEFINITIONS}
    INTERFACE
    )

target_compile_options(${target}
    PRIVATE
    PUBLIC
    ${DEFAULT_COMPILE_OPTIONS}
    INTERFACE
    )

target_link_libraries(${target}
    PRIVATE
    PUBLIC
    ${DEFAULT_LINKER_OPTIONS}
    INTERFACE
    )
