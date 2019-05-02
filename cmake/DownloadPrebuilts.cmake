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

set(PREBUILT_PATH "${SL_PROJECT_ROOT}/externals/prebuilt")
set(PREBUILT_URL "http://pallas.bfh.ch/libs/SLProject/_lib/prebuilt")

#==============================================================================
if("${SYSTEM_NAME_UPPER}" STREQUAL "LINUX")
    set(OpenCV_VERSION "3.4.1")
    set(OpenCV_DIR "${PREBUILT_PATH}/linux_opencv_${OpenCV_VERSION}")
    set(OpenCV_LINK_DIR "${OpenCV_DIR}/${CMAKE_BUILD_TYPE}")
    set(OpenCV_INCLUDE_DIR "${OpenCV_DIR}/include")
    set(OpenCV_LIBS ${OpenCV_LINK_LIBS})
    set(OpenCV_LIBS_DEBUG ${OpenCV_LIBS})

elseif("${SYSTEM_NAME_UPPER}" STREQUAL "WINDOWS") #----------------------------
    set(OpenCV_VERSION "3.4.1")
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

    file(GLOB usedCVLibs_Debug
        ${OpenCV_DIR}/lib/opencv_aruco*d.dll
        ${OpenCV_DIR}/lib/opencv_calib3d*d.dll
        ${OpenCV_DIR}/lib/opencv_core*d.dll
        ${OpenCV_DIR}/lib/opencv_features2d*d.dll
        ${OpenCV_DIR}/lib/opencv_face*d.dll
        ${OpenCV_DIR}/lib/opencv_flann*d.dll
        ${OpenCV_DIR}/lib/opencv_highgui*d.dll
        ${OpenCV_DIR}/lib/opencv_imgproc*d.dll
        ${OpenCV_DIR}/lib/opencv_imgcodecs*d.dll
        ${OpenCV_DIR}/lib/opencv_objdetect*d.dll
        ${OpenCV_DIR}/lib/opencv_video*d.dll
        ${OpenCV_DIR}/lib/opencv_videoio*d.dll
        ${OpenCV_DIR}/lib/opencv_xfeatures2d*d.dll
        )
    file(GLOB usedCVLibs_Release
        ${OpenCV_DIR}/lib/opencv_aruco*.dll
        ${OpenCV_DIR}/lib/opencv_calib3d*.dll
        ${OpenCV_DIR}/lib/opencv_core*.dll
        ${OpenCV_DIR}/lib/opencv_features2d*.dll
        ${OpenCV_DIR}/lib/opencv_face*.dll
        ${OpenCV_DIR}/lib/opencv_flann*.dll
        ${OpenCV_DIR}/lib/opencv_highgui*.dll
        ${OpenCV_DIR}/lib/opencv_imgproc*.dll
        ${OpenCV_DIR}/lib/opencv_imgcodecs*.dll
        ${OpenCV_DIR}/lib/opencv_objdetect*.dll
        ${OpenCV_DIR}/lib/opencv_video*.dll
        ${OpenCV_DIR}/lib/opencv_videoio*.dll
        ${OpenCV_DIR}/lib/opencv_xfeatures2d*.dll
        )

    if ("${CMAKE_CXX_COMPILER_ID}" MATCHES "MSVC")
        file(COPY ${OpenCV_LIBS_to_copy_debug} DESTINATION ${CMAKE_BINARY_DIR}/Debug)
        file(COPY ${OpenCV_LIBS_to_copy_release} DESTINATION ${CMAKE_BINARY_DIR}/Release)
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
        file(DOWNLOAD "${PREBUILT_URL}/${OpenCV_PREBUILT_ZIP}" "${PREBUILT_PATH}/${OpenCV_PREBUILT_ZIP}")
        execute_process(COMMAND ${CMAKE_COMMAND} -E tar xzf
            "${PREBUILT_PATH}/${OpenCV_PREBUILT_ZIP}"
            WORKING_DIRECTORY "${PREBUILT_PATH}")
        file(REMOVE "${PREBUILT_PATH}/${OpenCV_PREBUILT_ZIP}")
    endif ()

    # Now download for MacOS
    set(OpenCV_VERSION "4.0.0")
    set(PREBUILT_OPENCV_DIR "mac64_opencv_${OpenCV_VERSION}")
    set(OpenCV_DIR "${PREBUILT_PATH}/${PREBUILT_OPENCV_DIR}")
    set(OpenCV_LINK_DIR "${OpenCV_DIR}/${CMAKE_BUILD_TYPE}")
    set(OpenCV_INCLUDE_DIR "${OpenCV_DIR}/include")
    set(OpenCV_PREBUILT_ZIP "${OpenCV_PREBUILT_DIR}.zip")

    if (NOT EXISTS "${OpenCV_DIR}")
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

elseif("${SYSTEM_NAME_UPPER}" STREQUAL "ANDROID") #---------------------------
    set(OpenCV_VERSION "3.4.1")
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
        libjpeg
        libpng
        libprotobuf
        libtiff
        libwebp
        tegra_hal
        z)

    foreach(lib ${OpenCV_LINK_LIBS})
        add_library(lib_${lib} STATIC IMPORTED)
        set_target_properties(lib_${lib} PROPERTIES IMPORTED_LOCATION ${OpenCV_LINK_DIR}/lib${lib}.a)
        set(OpenCV_LIBS
            ${OpenCV_LIBS}
            lib_${lib})
    endforeach(lib)

    set(OpenCV_LIBS_DEBUG ${OpenCV_LIBS})

endif()
#==============================================================================

link_directories(${OpenCV_LINK_DIR})
