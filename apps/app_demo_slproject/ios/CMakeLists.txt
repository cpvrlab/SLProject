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

#if this variable is not defined, we make automatic signing with "iPhone Developer"
if(XCODE_CODESIGNIDENTITY)
	message(STATUS "manual signing")
	set(APP_BUNDLE_IDENTIFIER "ch.bfh.ti.cpvrlab.demo-sl") # <== Set to your app's bundle identifier (Attention the erlebar one is with "cpvr")
	
	set(XCODE_PROVISIONINGPROFILE "Demo SL Development")
	set(XCODE_DEVELOPMENTTEAM "858Y9EWZ4B")
else()
	message(STATUS "automatic signing")
	#set(APP_BUNDLE_IDENTIFIER "com.${target}") # <== Set to your app's bundle identifier

    set(XCODE_CODESIGNIDENTITY "iPhone Developer")
	set(XCODE_DEVELOPMENTTEAM ${XCODE_DEVELOPMENTTEAM})
endif()

# set new xcode build system
set(CMAKE_XCODE_BUILD_SYSTEM "12")


#set(DEPLOYMENT_TARGET 12.0)                                  # <== Set your deployment target version of iOS
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
    ${SL_PROJECT_ROOT}/apps/app_demo_slproject/include/AppDemoGui.h
    ${SL_PROJECT_ROOT}/apps/app_demo_slproject/include/AppDemoSceneView.h
	${SL_PROJECT_ROOT}/apps/source/CVCapture.h
	${SL_PROJECT_ROOT}/apps/source/AppDemo.h
	${SL_PROJECT_ROOT}/apps/source/SLInterface.h
    ${CMAKE_CURRENT_SOURCE_DIR}/AppDelegate.h
	${CMAKE_CURRENT_SOURCE_DIR}/ViewController.h
    )

set(sources
    ${SL_PROJECT_ROOT}/apps/app_demo_slproject/source/AppDemoGui.cpp
    ${SL_PROJECT_ROOT}/apps/app_demo_slproject/source/AppDemoLoad.cpp
    ${SL_PROJECT_ROOT}/apps/app_demo_slproject/source/AppDemoVideo.cpp
    ${SL_PROJECT_ROOT}/apps/app_demo_slproject/source/AppDemoSceneView.cpp
	${SL_PROJECT_ROOT}/apps/source/CVCapture.cpp
	${SL_PROJECT_ROOT}/apps/source/AppDemo.cpp
	${SL_PROJECT_ROOT}/apps/source/SLInterface.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/AppDelegate.mm
	${CMAKE_CURRENT_SOURCE_DIR}/main.m
	${CMAKE_CURRENT_SOURCE_DIR}/ViewController.mm
	)

#copy defined resources to build directory and add get folder reference using file(GLOB) (this is a secret trick that nobody knows)
include(CopyResourcesAppDemoSLProject)
set(DATA_DIR "${CMAKE_BINARY_DIR}/data")
copy_resources_slprojectdemo("${DATA_DIR}")
file(GLOB DATA "${DATA_DIR}")

# define additional Xcode app resources
set(XIB_FILES
	${CMAKE_CURRENT_SOURCE_DIR}/Base.lproj/ViewController_iPad.xib
	${CMAKE_CURRENT_SOURCE_DIR}/Base.lproj/ViewController_iPhone.xib
    ${CMAKE_CURRENT_SOURCE_DIR}/LaunchScreenSLDemo.storyboard
	)

set(ASSET_CATALOG
	${CMAKE_CURRENT_SOURCE_DIR}/Images/Images.xcassets
    ${CMAKE_CURRENT_SOURCE_DIR}/Images/LaunchImage_1024x768.png
	)

# Group data resource files in Xcode Resource directory
source_group(Resources FILES ${XIB_FILES})
source_group(Resources FILES ${ASSET_CATALOG})
source_group(Resources FILES ${DATA})

set(RESOURCES
	${ASSET_CATALOG}
	${XIB_FILES}
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

if(XCODE_PROVISIONINGPROFILE)
	set_xcode_property(${target} PROVISIONING_PROFILE_SPECIFIER ${XCODE_PROVISIONINGPROFILE})
endif()
set_xcode_property(${target} CODE_SIGN_IDENTITY ${XCODE_CODESIGNIDENTITY})
set_xcode_property(${target} DEVELOPMENT_TEAM ${XCODE_DEVELOPMENTTEAM})

target_include_directories(
        ${target}
        PRIVATE
        ${SL_PROJECT_ROOT}/apps/app_demo_slproject/include/
        ${SL_PROJECT_ROOT}/apps/source
        ${SL_PROJECT_ROOT}/modules/sl/source
        ${SL_PROJECT_ROOT}/modules/sl/externals/imgui
        ${SL_PROJECT_ROOT}/modules/sl/externals/spa
        ${SL_PROJECT_ROOT}/modules/utils/externals/dirent
        ${SL_PROJECT_ROOT}/modules/sl/externals/gl3w
		${SL_PROJECT_ROOT}/modules/sl/externals/gl3w/GL
        ${OpenCV_INCLUDE_DIR}
        ${glfw_INCLUDE_DIR}
        PUBLIC
        INTERFACE
    )

target_link_libraries(
        ${target}
        PRIVATE
		
        PUBLIC
        sl
        ${PlatformLinkLibs}
        sl_external
        sl_utils
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


