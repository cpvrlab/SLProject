# 
# CMake configuration for ErlebAR application for ios
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
find_library(ARKIT ARKit required)

set(target ErlebAR)
set(DEVELOPMENT_TEAM_ID "858Y9EWZ4B")               # <== Set to your team ID from Apple
set(APP_BUNDLE_IDENTIFIER "ch.bfh.ti.cpvr.erlebar") # <== Set to your app's bundle identifier (Attention the erlebar one is with "cpvr")
set(PROVISIONING_PROFILE "ErlebAR - Development")
set(CODE_SIGN_IDENTITY "Apple Development: Michael Göttlicher (KMX9985DR7)")  # <== Set to your preferred code sign identity, to see list: /usr/bin/env xcrun security find-identity -v -p codesigning

# set(DEPLOYMENT_TARGET 11.0)                                  # <== Set your deployment target version of iOS
set(DEVICE_FAMILY "1,2")                                      # <== Set to "1" to target iPhone, set to "2" to target iPad, set to "1,2" to target both

#include(BundleUtilities)

set(PRODUCT_NAME ${target})
set(EXECUTABLE_NAME ${target})
# MACOSX_BUNDLE_... variables are used in plist.in to generate the Info.plist file used by xcode. (see https://cmake.org/cmake/help/v3.17/prop_tgt/MACOSX_BUNDLE_INFO_PLIST.html)
set(MACOSX_BUNDLE_EXECUTABLE_NAME ${target})
set(MACOSX_BUNDLE_INFO_STRING ${APP_BUNDLE_IDENTIFIER})
set(MACOSX_BUNDLE_GUI_IDENTIFIER ${APP_BUNDLE_IDENTIFIER})
set(MACOSX_BUNDLE_BUNDLE_NAME ${APP_BUNDLE_IDENTIFIER})
set(MACOSX_BUNDLE_LONG_VERSION_STRING "1.0")
set(MACOSX_BUNDLE_SHORT_VERSION_STRING "1.0")
set(MACOSX_BUNDLE_BUNDLE_VERSION "1.0")
set(MACOSX_BUNDLE_COPYRIGHT "Copyright Berner Fachhochschule (Marcus Hudritsch)")
# set(MACOSX_DEPLOYMENT_TARGET ${DEPLOYMENT_TARGET})

#set(include_path "${CMAKE_CURRENT_SOURCE_DIR}")
#set(source_path "${CMAKE_CURRENT_SOURCE_DIR}")
set(compile_definitions SL_STARTSCENE=SID_Minimal)

set(headers
	${erlebar_headers}
	${CMAKE_CURRENT_SOURCE_DIR}/ErlebARAppDelegate.h
	${CMAKE_CURRENT_SOURCE_DIR}/ErlebARViewController.h
    ${SL_PROJECT_ROOT}/apps/source/sens/iOS/SENSiOSCamera.h
	${SL_PROJECT_ROOT}/apps/source/sens/iOS/SENSiOSCameraDelegate.h
	${SL_PROJECT_ROOT}/apps/source/sens/iOS/SENSiOSGps.h
	${SL_PROJECT_ROOT}/apps/source/sens/iOS/SENSiOSGpsDelegate.h
	${SL_PROJECT_ROOT}/apps/source/sens/iOS/SENSiOSOrientation.h
	${SL_PROJECT_ROOT}/apps/source/sens/iOS/SENSiOSOrientationDelegate.h
    ${SL_PROJECT_ROOT}/apps/source/sens/iOS/SENSiOSARCore.h
    ${SL_PROJECT_ROOT}/apps/source/sens/iOS/SENSiOSARCoreDelegate.h
    )

set(sources
	${erlebar_sources}
    ${CMAKE_CURRENT_SOURCE_DIR}/ErlebARAppDelegate.mm
	${CMAKE_CURRENT_SOURCE_DIR}/main.m
	${CMAKE_CURRENT_SOURCE_DIR}/ErlebARViewController.mm
    ${SL_PROJECT_ROOT}/apps/source/sens/iOS/SENSiOSCamera.mm
	${SL_PROJECT_ROOT}/apps/source/sens/iOS/SENSiOSCameraDelegate.mm
	${SL_PROJECT_ROOT}/apps/source/sens/iOS/SENSiOSGps.mm
	${SL_PROJECT_ROOT}/apps/source/sens/iOS/SENSiOSGpsDelegate.mm
	${SL_PROJECT_ROOT}/apps/source/sens/iOS/SENSiOSOrientation.mm
	${SL_PROJECT_ROOT}/apps/source/sens/iOS/SENSiOSOrientationDelegate.mm
    ${SL_PROJECT_ROOT}/apps/source/sens/iOS/SENSiOSARCore.mm
    ${SL_PROJECT_ROOT}/apps/source/sens/iOS/SENSiOSARCoreDelegate.mm
	)
 
# Group source files
set(headers_sources ${sources} ${headers})
source_group_by_path("${CMAKE_CURRENT_SOURCE_DIR}" "\\\\.h$|\\\\.hpp$" "" ${headers_sources})
#source_group_by_path("${CMAKE_CURRENT_SOURCE_DIR}" "\\\\.h$|\\\\.hpp$" "Header Files" ${headers})
#source_group_by_path("${CMAKE_CURRENT_SOURCE_DIR}" "\\\\.cpp$|\\\\.c$|\\\\.h$|\\\\.hpp$" "Source Files" ${sources})

#copy defined resources to build directory and add get folder reference using file(GLOB) (this is a secret trick that nobody knows)
include(CopyResourcesErlebAR)
set(DATA_DIR "${CMAKE_BINARY_DIR}/data")
copy_resources_erlebar("${DATA_DIR}")
file(GLOB DATA "${DATA_DIR}")

# define additional Xcode app resources
set(CONFIG_FILES
	${CMAKE_CURRENT_SOURCE_DIR}/Base.lproj/ViewController_iPad.xib
	${CMAKE_CURRENT_SOURCE_DIR}/Base.lproj/ViewController_iPhone.xib
	${CMAKE_CURRENT_SOURCE_DIR}/LaunchScreen.storyboard
	)

set(ASSET_CATALOG
	${CMAKE_CURRENT_SOURCE_DIR}/Images/Images.xcassets
    ${CMAKE_CURRENT_SOURCE_DIR}/Images/erleb-AR_logo_rounded.png
	)

# Group data resource files in Xcode Resource directory
source_group(Resources FILES ${CONFIG_FILES})
source_group(Resources FILES ${ASSET_CATALOG})
source_group(Resources FILES ${DATA})

set(RESOURCES
	${ASSET_CATALOG}
	${CONFIG_FILES}
	${DATA}
	)

add_executable(
        ${target}
		MACOSX_BUNDLE
        ${headers}
        ${sources}
		${RESOURCES}
    )

set_target_properties(
        ${target}
        PROPERTIES
        ${DEFAULT_PROJECT_OPTIONS}
        FOLDER "apps"
		
        XCODE_ATTRIBUTE_DEBUG_INFORMATION_FORMAT "dwarf-with-dsym"
        XCODE_ATTRIBUTE_GCC_PREFIX_HEADER "${CMAKE_CURRENT_SOURCE_DIR}/Prefix.pch" #this is a precompiled header!
        RESOURCE "${RESOURCES}"
        XCODE_ATTRIBUTE_GCC_PRECOMPILE_PREFIX_HEADER "YES"
        # XCODE_ATTRIBUTE_IPHONEOS_DEPLOYMENT_TARGET ${DEPLOYMENT_TARGET}
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
		XCODE_ATTRIBUTE_ASSETCATALOG_COMPILER_APPICON_NAME "AppIcon" #defines icon name in asset catalog (images.xcassets)
		# apple requires storyboard launchscreens now so the following will not come into store:
		# XCODE_ATTRIBUTE_ASSETCATALOG_COMPILER_LAUNCHIMAGE_NAME "LaunchImage"  #defines launch screen image name in asset catalog (images.xcassets)
    )
	
target_include_directories(
        ${target}
        PRIVATE
		${erlebar_include_directories}
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
		${ARKIT}
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
	
	
