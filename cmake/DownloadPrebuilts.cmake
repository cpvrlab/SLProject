#
# CMake options downloading and installing prebuilt libs
#

#
# Download and install OpenCV from pallas.bfh.ch
#

set(OpenCV_VERSION)
set(OpenCV_DIR)
set(OpenCV_LINK_DIR)
set(OpenCV_INCLUDE_DIR)
set(OpenCV_LINK_LIBS
        opencv_aruco
        opencv_calib3d
        opencv_features2d
        opencv_face
        opencv_flann
        opencv_highgui
        opencv_imgcodecs
        opencv_objdetect
        opencv_video
        opencv_imgproc
        opencv_videoio
        opencv_xfeatures2d
        opencv_core
        )
set(OpenCV_LIBS)

set(g2o_DIR)
set(g2o_INCLUDE_DIR)
set(g2o_LINK_DIR)
set(g2o_LINK_LIBS
        g2o_core
        g2o_solver_dense
        g2o_solver_eigen
        g2o_stuff
        g2o_types_sba
        g2o_types_sim3
        g2o_types_slam3d
        g2o_types_slam3d_addons
    )

set(assimp_DIR)
set(assimp_LINK_DIR)
set(assimp_INCLUDE_DIR)

set(PREBUILT_PATH "${SL_PROJECT_ROOT}/externals/prebuilt")
set(PREBUILT_URL "http://pallas.bfh.ch/libs/SLProject/_lib/prebuilt")

#==============================================================================
if("${SYSTEM_NAME_UPPER}" STREQUAL "LINUX")
    set(OpenCV_VERSION "4.1.1")
    set(OpenCV_DIR "${PREBUILT_PATH}/linux_opencv_${OpenCV_VERSION}")
    set(OpenCV_LINK_DIR "${OpenCV_DIR}/${CMAKE_BUILD_TYPE}")
    set(OpenCV_INCLUDE_DIR "${OpenCV_DIR}/include")

    # new include directory structure for opencv 4
    if ("${OpenCV_VERSION}" MATCHES "^4\.[0-9]+\.[0-9]+$")
        set(OpenCV_INCLUDE_DIR "${OpenCV_INCLUDE_DIR}/opencv4")
    endif()

    set(OpenCV_LIBS ${OpenCV_LINK_LIBS})
    set(OpenCV_LIBS_DEBUG ${OpenCV_LIBS})

    set(g2o_DIR ${PREBUILT_PATH}/linux_g2o)
    set(g2o_INCLUDE_DIR ${g2o_DIR}/include)
    set(g2o_LINK_DIR ${g2o_DIR}/${CMAKE_BUILD_TYPE})
    set(g2o_LIBS ${g2o_LINK_LIBS})

elseif("${SYSTEM_NAME_UPPER}" STREQUAL "WINDOWS") #----------------------------
    #OpenCV
    set(OpenCV_VERSION "4.1.2")
    set(OpenCV_PREBUILT_DIR "win64_opencv_${OpenCV_VERSION}")
    set(OpenCV_DIR "${PREBUILT_PATH}/${OpenCV_PREBUILT_DIR}")
    set(OpenCV_LINK_DIR "${OpenCV_DIR}/lib")
    set(OpenCV_INCLUDE_DIR "${OpenCV_DIR}/include")
    set(OpenCV_PREBUILT_ZIP "${OpenCV_PREBUILT_DIR}.zip")

    if (NOT EXISTS "${OpenCV_DIR}")
        file(DOWNLOAD "${PREBUILT_URL}/${OpenCV_PREBUILT_ZIP}" "${PREBUILT_PATH}/${OpenCV_PREBUILT_ZIP}")
        execute_process(COMMAND ${CMAKE_COMMAND} -E tar xzf
            "${PREBUILT_PATH}/${OpenCV_PREBUILT_ZIP}"
            WORKING_DIRECTORY "${PREBUILT_PATH}")
        file(REMOVE "${PREBUILT_PATH}/${OpenCV_PREBUILT_ZIP}")
		
		if( NOT EXISTS "${OpenCV_DIR}" )
			message( SEND_ERROR "Downloading Prebuilds failed! OpenCV prebuilds for version ${OpenCV_VERSION} do not extist! Build required version yourself to location ${OpenCV_DIR} using script in directory externals/prebuild_scipts or try another OpenCV version." )
		endif()
    endif ()

    string(REPLACE "." "" OpenCV_LIBS_POSTFIX ${OpenCV_VERSION})

    foreach(lib ${OpenCV_LINK_LIBS})
        set(OpenCV_LIBS
                ${OpenCV_LIBS}
                optimized ${lib}${OpenCV_LIBS_POSTFIX}
                debug ${lib}${OpenCV_LIBS_POSTFIX}d)
        file(GLOB OpenCV_LIBS_to_copy_debug
                ${OpenCV_LIBS_to_copy_debug}
                ${OpenCV_DIR}/lib/${lib}*d.dll
                )
        file(GLOB OpenCV_LIBS_to_copy_release
                ${OpenCV_LIBS_to_copy_release}
                ${OpenCV_DIR}/lib/${lib}*.dll
                )
    endforeach(lib)

    # Set working dir for VS
    set(DEFAULT_PROJECT_OPTIONS ${DEFAULT_PROJECT_OPTIONS}
            VS_DEBUGGER_WORKING_DIRECTORY ${CMAKE_RUNTIME_OUTPUT_DIRECTORY})

    # For MSVS copy them to working dir
    if ("${CMAKE_CXX_COMPILER_ID}" MATCHES "MSVC")
        file(COPY ${OpenCV_LIBS_to_copy_debug} DESTINATION ${CMAKE_BINARY_DIR}/Debug)
        file(COPY ${OpenCV_LIBS_to_copy_release} DESTINATION ${CMAKE_BINARY_DIR}/Release)
		file(COPY ${OpenCV_LIBS_to_copy_release} DESTINATION ${CMAKE_BINARY_DIR}/RelWithDebInfo)
    endif()

    #G2O
    set(g2o_DIR ${PREBUILT_PATH}/win64_g2o)
    set(g2o_INCLUDE_DIR ${g2o_DIR}/include)
    set(g2o_LINK_DIR ${g2o_DIR}/lib)

    foreach(lib ${g2o_LINK_LIBS})
        add_library(${lib} SHARED IMPORTED)
        set_target_properties(${lib} PROPERTIES
            IMPORTED_IMPLIB_DEBUG "${g2o_LINK_DIR}/${lib}_d.lib"
            IMPORTED_IMPLIB "${g2o_LINK_DIR}/${lib}.lib"
            IMPORTED_LOCATION_DEBUG "${g2o_LINK_DIR}/${lib}_d.dll"
            IMPORTED_LOCATION "${g2o_LINK_DIR}/${lib}.dll"
            INTERFACE_INCLUDE_DIRECTORIES "${g2o_INCLUDE_DIR}"
        )
        set(g2o_LIBS
            ${g2o_LIBS}
            ${lib}
        )
    endforeach(lib)
    
    set(g2o_PREBUILT_ZIP "win64_g2o.zip")
    set(g2o_URL ${PREBUILT_URL}/${g2o_PREBUILT_ZIP})
      
    if (NOT EXISTS "${g2o_DIR}")
        file(DOWNLOAD "${PREBUILT_URL}/${g2o_PREBUILT_ZIP}" "${PREBUILT_PATH}/${g2o_PREBUILT_ZIP}")
        execute_process(COMMAND ${CMAKE_COMMAND} -E tar xzf
            "${PREBUILT_PATH}/${g2o_PREBUILT_ZIP}"
            WORKING_DIRECTORY "${PREBUILT_PATH}")
        file(REMOVE "${PREBUILT_PATH}/${g2o_PREBUILT_ZIP}")
    endif()

    # For MSVS copy g2o dlls to working dir
    if ("${CMAKE_CXX_COMPILER_ID}" MATCHES "MSVC")
		foreach(lib ${g2o_LINK_LIBS})
			file(GLOB g2o_dll_to_copy_debug
				${g2o_dll_to_copy_debug}
				${g2o_DIR}/bin/${lib}*d.dll
				)
			file(GLOB g2o_dll_to_copy_release
				${g2o_dll_to_copy_release}
				${g2o_DIR}/bin/${lib}*.dll
				)
		endforeach(lib)

        #message(STATUS "Copy g2o debug DLLs: ${g2o_dll_to_copy_debug}")
        file(COPY ${g2o_dll_to_copy_debug} DESTINATION ${CMAKE_BINARY_DIR}/Debug)
        #message(STATUS "Copy g2o release DLLs: ${g2o_dll_to_copy_release}")
        file(COPY ${g2o_dll_to_copy_release} DESTINATION ${CMAKE_BINARY_DIR}/Release)
		file(COPY ${g2o_dll_to_copy_release} DESTINATION ${CMAKE_BINARY_DIR}/RelWithDebInfo)
    endif()

    #assimp for windows
    set(assimp_VERSION "5.0")
    set(assimp_PREBUILT_DIR "win64_assimp_${assimp_VERSION}")
    set(assimp_DIR "${PREBUILT_PATH}/${assimp_PREBUILT_DIR}")
    set(assimp_LINK_DIR "${assimp_DIR}/lib")
    set(assimp_INCLUDE_DIR "${assimp_DIR}/include")
    set(assimp_PREBUILT_ZIP "${assimp_PREBUILT_DIR}.zip")
    set(assimp_LINK_LIBS assimp-mt)

    if (NOT EXISTS "${assimp_DIR}")
        file(DOWNLOAD "${PREBUILT_URL}/${assimp_PREBUILT_ZIP}" "${PREBUILT_PATH}/${assimp_PREBUILT_ZIP}")
        execute_process(COMMAND ${CMAKE_COMMAND} -E tar xzf
                "${PREBUILT_PATH}/${assimp_PREBUILT_ZIP}"
                WORKING_DIRECTORY "${PREBUILT_PATH}")
        file(REMOVE "${PREBUILT_PATH}/${assimp_PREBUILT_ZIP}")

        if( NOT EXISTS "${assimp_DIR}" )
            message( SEND_ERROR "Downloading Prebuilds failed! assimp prebuilds for version ${assimp_VERSION} do not extist!" )
        endif()
    endif ()

    set(assimp_LIBS
            ${assimp_LIBS}
            optimized assimp-mt
            debug assimp-mtd)

    file(GLOB assimp_LIBS_to_copy_debug
            ${assimp_LIBS_to_copy_debug}
            ${assimp_DIR}/lib/assimp-mtd.dll
            )
    file(GLOB assimp_LIBS_to_copy_release
            ${assimp_LIBS_to_copy_release}
            ${assimp_DIR}/lib/assimp-mt.dll
            )

    # Set working dir for VS
    set(DEFAULT_PROJECT_OPTIONS ${DEFAULT_PROJECT_OPTIONS}
            VS_DEBUGGER_WORKING_DIRECTORY ${CMAKE_RUNTIME_OUTPUT_DIRECTORY})

    # For MSVS copy them to working dir
    if ("${CMAKE_CXX_COMPILER_ID}" MATCHES "MSVC")
        file(COPY ${assimp_LIBS_to_copy_debug} DESTINATION ${CMAKE_BINARY_DIR}/Debug)
        file(COPY ${assimp_LIBS_to_copy_release} DESTINATION ${CMAKE_BINARY_DIR}/Release)
        file(COPY ${assimp_LIBS_to_copy_release} DESTINATION ${CMAKE_BINARY_DIR}/RelWithDebInfo)
    endif()



elseif("${SYSTEM_NAME_UPPER}" STREQUAL "DARWIN") #-----------------------------
    # Download first for iOS
    set(OpenCV_VERSION "3.4.0")
    set(OpenCV_PREBUILT_DIR "iosV8_opencv_${OpenCV_VERSION}")
    set(OpenCV_DIR "${PREBUILT_PATH}/${OpenCV_PREBUILT_DIR}")
    set(OpenCV_LINK_DIR "${OpenCV_DIR}/${CMAKE_BUILD_TYPE}")
    set(OpenCV_INCLUDE_DIR "${OpenCV_DIR}/include")
    set(OpenCV_PREBUILT_ZIP "${OpenCV_PREBUILT_DIR}.zip")

    if (NOT EXISTS "${OpenCV_DIR}")
        message(STATUS "OpenCV_DIR: ${OpenCV_DIR}")
        message(STATUS "${PREBUILT_URL}/${OpenCV_PREBUILT_ZIP}")
        message(STATUS "Download to: ${PREBUILT_PATH}/${OpenCV_PREBUILT_ZIP}")
        file(DOWNLOAD "${PREBUILT_URL}/${OpenCV_PREBUILT_ZIP}" "${PREBUILT_PATH}/${OpenCV_PREBUILT_ZIP}")
        execute_process(COMMAND ${CMAKE_COMMAND} -E tar xzf
            "${PREBUILT_PATH}/${OpenCV_PREBUILT_ZIP}"
            WORKING_DIRECTORY "${PREBUILT_PATH}")
        file(REMOVE "${PREBUILT_PATH}/${OpenCV_PREBUILT_ZIP}")
    endif ()

    # Now download for MacOS
    set(OpenCV_VERSION "4.1.1")
    set(OpenCV_PREBUILT_DIR "mac64_opencv_${OpenCV_VERSION}")
    set(OpenCV_DIR "${PREBUILT_PATH}/${OpenCV_PREBUILT_DIR}")
    set(OpenCV_LINK_DIR "${OpenCV_DIR}/${CMAKE_BUILD_TYPE}")
    set(OpenCV_INCLUDE_DIR "${OpenCV_DIR}/include")
    set(OpenCV_PREBUILT_ZIP "${OpenCV_PREBUILT_DIR}.zip")

    # new include directory structure for opencv 4
    if ("${OpenCV_VERSION}" MATCHES "^4\.[0-9]+\.[0-9]+$")
        set(OpenCV_INCLUDE_DIR "${OpenCV_INCLUDE_DIR}/opencv4")
    endif()

    if (NOT EXISTS "${OpenCV_DIR}")
        message(STATUS "OpenCV_DIR: ${OpenCV_DIR}")
        message(STATUS "Download from: ${PREBUILT_URL}/${OpenCV_PREBUILT_ZIP}")
        message(STATUS "Download to: ${PREBUILT_PATH}/${OpenCV_PREBUILT_ZIP}")
        file(DOWNLOAD "${PREBUILT_URL}/${OpenCV_PREBUILT_ZIP}" "${PREBUILT_PATH}/${OpenCV_PREBUILT_ZIP}")
        execute_process(COMMAND ${CMAKE_COMMAND} -E tar xzf
            "${PREBUILT_PATH}/${OpenCV_PREBUILT_ZIP}"
            WORKING_DIRECTORY "${PREBUILT_PATH}")
        file(REMOVE "${PREBUILT_PATH}/${OpenCV_PREBUILT_ZIP}")
    endif ()

    foreach(lib ${OpenCV_LINK_LIBS})
        set(OpenCV_LIBS
                ${OpenCV_LIBS}
                optimized ${lib}
                debug ${lib})
    endforeach(lib)

    file(GLOB OpenCV_LIBS_to_copy_debug
            ${OpenCV_LIBS_to_copy_debug}
            ${OpenCV_DIR}/Debug/libopencv_*.dylib
            )
    file(GLOB OpenCV_LIBS_to_copy_release
            ${OpenCV_LIBS_to_copy_release}
            ${OpenCV_DIR}/Release/libopencv_*.dylib
            )

    if(${CMAKE_GENERATOR} STREQUAL Xcode)
        file(COPY ${OpenCV_LIBS_to_copy_debug} DESTINATION ${CMAKE_BINARY_DIR}/Debug)
        file(COPY ${OpenCV_LIBS_to_copy_release} DESTINATION ${CMAKE_BINARY_DIR}/Release)
    endif()

    # Copy plist file with camera access description beside executable
    # This is needed for security purpose since MacOS Mohave
    set(MACOS_PLIST_FILE
        ${SL_PROJECT_ROOT}/data/config/info.plist)

    #message(STATUS "MACOS_PLIST_FILE: ${MACOS_PLIST_FILE}")
    #message(STATUS "CMAKE_BINARY_DIR: ${CMAKE_BINARY_DIR}")
    if(${CMAKE_GENERATOR} STREQUAL Xcode)
        file(COPY ${MACOS_PLIST_FILE} DESTINATION ${CMAKE_BINARY_DIR}/Debug)
        file(COPY ${MACOS_PLIST_FILE} DESTINATION ${CMAKE_BINARY_DIR}/Release)
    else()
        file(COPY ${MACOS_PLIST_FILE} DESTINATION ${CMAKE_BINARY_DIR})
    endif()

    #G2O
    set(g2o_DIR ${PREBUILT_PATH}/mac64_g2o)
    set(g2o_PREBUILT_ZIP "mac64_g2o.zip")
    set(g2o_URL ${PREBUILT_URL}/${g2o_PREBUILT_ZIP})
    set(g2o_INCLUDE_DIR ${g2o_DIR}/include)
    set(g2o_LINK_DIR ${g2o_DIR}/${CMAKE_BUILD_TYPE})

    #message(STATUS "g2o_DIR: ${g2o_DIR}")
    #message(STATUS "g2o_LINK_DIR: ${g2o_LINK_DIR}")
    #message(STATUS "g2o_URL: ${g2o_URL}")

    if (NOT EXISTS "${g2o_DIR}")
        file(DOWNLOAD "${PREBUILT_URL}/${g2o_PREBUILT_ZIP}" "${PREBUILT_PATH}/${g2o_PREBUILT_ZIP}")
        execute_process(COMMAND ${CMAKE_COMMAND} -E tar xzf
            "${PREBUILT_PATH}/${g2o_PREBUILT_ZIP}"
            WORKING_DIRECTORY "${PREBUILT_PATH}")
        file(REMOVE "${PREBUILT_PATH}/${g2o_PREBUILT_ZIP}")
    endif ()

    foreach(lib ${g2o_LINK_LIBS})
        add_library(lib${lib} SHARED IMPORTED)
        set_target_properties(lib${lib} PROPERTIES IMPORTED_LOCATION "${g2o_LINK_DIR}/lib${lib}.dylib")
        #message(STATUS "IMPORTED_LOCATION: ${g2o_LINK_DIR}/lib${lib}.dylib")
        set(g2o_LIBS
            ${g2o_LIBS}
            lib${lib}
            #optimized ${lib}
            #debug ${lib}
            )
    endforeach(lib)

    #assimp for macos
    set(assimp_VERSION "5.0")
    set(assimp_PREBUILT_DIR "mac64_assimp_${assimp_VERSION}")
    set(assimp_DIR "${PREBUILT_PATH}/${assimp_PREBUILT_DIR}")
    set(assimp_LINK_DIR "${assimp_DIR}/${CMAKE_BUILD_TYPE}")
    #message(STATUS "assimp_LINK_DIR ${assimp_LINK_DIR}")

    set(assimp_INCLUDE_DIR "${assimp_DIR}/include")
    set(assimp_PREBUILT_ZIP "${assimp_PREBUILT_DIR}.zip")

    if (NOT EXISTS "${assimp_DIR}")
        file(DOWNLOAD "${PREBUILT_URL}/${assimp_PREBUILT_ZIP}" "${PREBUILT_PATH}/${assimp_PREBUILT_ZIP}")
        execute_process(COMMAND ${CMAKE_COMMAND} -E tar xzf
                "${PREBUILT_PATH}/${assimp_PREBUILT_ZIP}"
                WORKING_DIRECTORY "${PREBUILT_PATH}")
        file(REMOVE "${PREBUILT_PATH}/${assimp_PREBUILT_ZIP}")

        if( NOT EXISTS "${assimp_DIR}" )
            message( SEND_ERROR "Downloading Prebuilds failed! assimp prebuilds for version ${assimp_VERSION} do not extist!" )
        endif()
    endif ()

    set(assimp_LIBS
            ${assimp_LIBS}
            optimized libassimp.dylib
            debug libassimpd.dylib)

    file(GLOB assimp_LIBS_to_copy_debug
            ${assimp_LIBS_to_copy_debug}
            ${assimp_DIR}/Debug/libassimpd*.dylib
            ${assimp_DIR}/Debug/libIrrXMLd.dylib
            )
    file(GLOB assimp_LIBS_to_copy_release
            ${assimp_LIBS_to_copy_release}
            ${assimp_DIR}/Release/libassimp*.dylib
            ${assimp_DIR}/Release/libIrrXML.dylib
            )

    if(${CMAKE_GENERATOR} STREQUAL Xcode)
        file(COPY ${assimp_LIBS_to_copy_debug} DESTINATION ${CMAKE_BINARY_DIR}/Debug)
        file(COPY ${assimp_LIBS_to_copy_release} DESTINATION ${CMAKE_BINARY_DIR}/Release)
    endif()

    # Copy plist file with camera access description beside executable
    # This is needed for security purpose since MacOS Mohave
    set(MACOS_PLIST_FILE
            ${SL_PROJECT_ROOT}/data/config/info.plist)

    #message(STATUS "MACOS_PLIST_FILE: ${MACOS_PLIST_FILE}")
    #message(STATUS "CMAKE_BINARY_DIR: ${CMAKE_BINARY_DIR}")
    if(${CMAKE_GENERATOR} STREQUAL Xcode)
        file(COPY ${MACOS_PLIST_FILE} DESTINATION ${CMAKE_BINARY_DIR}/Debug)
        file(COPY ${MACOS_PLIST_FILE} DESTINATION ${CMAKE_BINARY_DIR}/Release)
    else()
        file(COPY ${MACOS_PLIST_FILE} DESTINATION ${CMAKE_BINARY_DIR})
    endif()

elseif("${SYSTEM_NAME_UPPER}" STREQUAL "ANDROID") #---------------------------
    set(OpenCV_VERSION "4.1.1")
    set(OpenCV_PREBUILT_DIR "andV8_opencv_${OpenCV_VERSION}")
    set(OpenCV_DIR "${PREBUILT_PATH}/${OpenCV_PREBUILT_DIR}")
    set(OpenCV_LINK_DIR "${OpenCV_DIR}/${CMAKE_BUILD_TYPE}/${ANDROID_ABI}")
    set(OpenCV_INCLUDE_DIR "${OpenCV_DIR}/include")
    set(OpenCV_PREBUILT_ZIP "${OpenCV_PREBUILT_DIR}.zip")

    if (NOT EXISTS "${OpenCV_DIR}")
        file(DOWNLOAD "${PREBUILT_URL}/${OpenCV_PREBUILT_ZIP}" "${PREBUILT_PATH}/${OpenCV_PREBUILT_ZIP}")
        execute_process(COMMAND ${CMAKE_COMMAND} -E tar xzf
            "${PREBUILT_PATH}/${OpenCV_PREBUILT_ZIP}"
            WORKING_DIRECTORY "${PREBUILT_PATH}")
        file(REMOVE "${PREBUILT_PATH}/${OpenCV_PREBUILT_ZIP}")
    endif ()

    set(OpenCV_LINK_LIBS
        ${OpenCV_LINK_LIBS}
        cpufeatures
        IlmImf
        libjasper
        libpng
        libprotobuf
        libtiff
        libwebp
        tegra_hal)

    # new link libraries for opencv 4
    if ("${OpenCV_VERSION}" MATCHES "^4\.[0-9]+\.[0-9]+$")
        set(OpenCV_LINK_LIBS
            ${OpenCV_LINK_LIBS}
            ittnotify
            libjpeg-turbo
            quirc)
    else()
        set(OpenCV_LINK_LIBS
            ${OpenCV_LINK_LIBS}
            libjpeg)
    endif()

    foreach(lib ${OpenCV_LINK_LIBS})
        add_library(lib_${lib} STATIC IMPORTED)
        set_target_properties(lib_${lib} PROPERTIES IMPORTED_LOCATION ${OpenCV_LINK_DIR}/lib${lib}.a)
        set(OpenCV_LIBS
                ${OpenCV_LIBS}
                lib_${lib})
    endforeach(lib)

    set(OpenCV_LIBS_DEBUG ${OpenCV_LIBS})

    #G2O
    set(g2o_PREBUILT_DIR "andV8_g2o")
    set(g2o_DIR ${PREBUILT_PATH}/andV8_g2o)
    set(g2o_INCLUDE_DIR ${g2o_DIR}/include)
    set(g2o_LINK_DIR ${g2o_DIR}/${CMAKE_BUILD_TYPE}/${ANDROID_ABI})
    set(g2o_PREBUILT_ZIP "${g2o_PREBUILT_DIR}.zip")

    if (NOT EXISTS "${g2o_DIR}")
        file(DOWNLOAD "${PREBUILT_URL}/${g2o_PREBUILT_ZIP}" "${PREBUILT_PATH}/${g2o_PREBUILT_ZIP}")
        execute_process(COMMAND ${CMAKE_COMMAND} -E tar xzf
            "${PREBUILT_PATH}/${g2o_PREBUILT_ZIP}"
            WORKING_DIRECTORY "${PREBUILT_PATH}")
        file(REMOVE "${PREBUILT_PATH}/${g2o_PREBUILT_ZIP}")
    endif ()

    foreach(lib ${g2o_LINK_LIBS})
        add_library(lib_${lib} SHARED IMPORTED)
        set_target_properties(lib_${lib} PROPERTIES
            IMPORTED_LOCATION "${g2o_LINK_DIR}/lib${lib}.so"
        )
        set(g2o_LIBS
            ${g2o_LIBS}
            lib_${lib}
        )
    endforeach(lib)
endif()
#==============================================================================

link_directories(${OpenCV_LINK_DIR})
link_directories(${g2o_LINK_DIR})
link_directories(${assimp_LINK_DIR})
