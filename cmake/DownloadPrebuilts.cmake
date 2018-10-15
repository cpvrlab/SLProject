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
    opencv_core
    opencv_features2d
    opencv_face
    opencv_flann
    opencv_highgui
    opencv_imgproc
    opencv_imgcodecs
    opencv_objdetect
    opencv_video
    opencv_videoio
    opencv_xfeatures2d
    )

set(OpenCV_LIBS)
set(PREBUILT_PATH "${CMAKE_CURRENT_SOURCE_DIR}/_lib/prebuilt")
set(PREBUILT_URL "http://pallas.bfh.ch/libs/SLProject/_lib/prebuilt")

#==============================================================================
if("${SYSTEM_NAME_UPPER}" STREQUAL "LINUX")
    set(OpenCV_VERSION "3.4.1")
    set(OpenCV_DIR "${CMAKE_CURRENT_SOURCE_DIR}/_lib/prebuilt/linux_opencv_${OpenCV_VERSION}")
    set(OpenCV_LINK_DIR "${OpenCV_DIR}/${CMAKE_BUILD_TYPE}")
    set(OpenCV_INCLUDE_DIR "${OpenCV_DIR}/include")
    set(OpenCV_LIBS ${OpenCV_LINK_LIBS})
    set(OpenCV_LIBS_DEBUG ${OpenCV_LIBS})

elseif("${SYSTEM_NAME_UPPER}" STREQUAL "WINDOWS") #----------------------------
    set(OpenCV_VERSION "3.4.1")
    set(PREBUILT_OPENCV_DIR "win64_opencv_${OpenCV_VERSION}")
    set(OpenCV_DIR "${PREBUILT_PATH}/${PREBUILT_OPENCV_DIR}")
    set(OpenCV_LINK_DIR "${OpenCV_DIR}/lib")
    set(OpenCV_INCLUDE_DIR "${OpenCV_DIR}/include")
    set(PREBUILT_ZIP "${PREBUILT_OPENCV_DIR}.zip")
	
    set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "../../_bin-win64")
    set(CMAKE_LIBRARY_OUTPUT_DIRECTORY "../../_bin-win64")
    set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY "../../_bin-win64")

    foreach(OUTPUTCONFIG ${CMAKE_CONFIGURATION_TYPES})
        string( TOUPPER ${OUTPUTCONFIG} OUTPUTCONFIGUP)
        set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_${OUTPUTCONFIGUP} "../../_bin-win64")
        set(CMAKE_LIBRARY_OUTPUT_DIRECTORY_${OUTPUTCONFIGUP} "../../_bin-win64")
        set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY_${OUTPUTCONFIGUP} "../../_bin-win64")
    endforeach( OUTPUTCONFIG CMAKE_CONFIGURATION_TYPES )

    if (NOT EXISTS "${OpenCV_DIR}")
        file(DOWNLOAD "${PREBUILT_URL}/${PREBUILT_ZIP}" "${PREBUILT_PATH}/${PREBUILT_ZIP}")
        execute_process(COMMAND ${CMAKE_COMMAND} -E tar xzf
            "${PREBUILT_PATH}/${PREBUILT_ZIP}"
            WORKING_DIRECTORY "${PREBUILT_PATH}")
        file(REMOVE "${PREBUILT_PATH}/${PREBUILT_ZIP}")
    endif ()

    string(REPLACE "." "" OpenCV_LIBS_POSTFIX ${OpenCV_VERSION})

    foreach(lib ${OpenCV_LINK_LIBS})
        set(OpenCV_LIBS
            ${OpenCV_LIBS}
            optimized ${lib}${OpenCV_LIBS_POSTFIX}
            debug ${lib}${OpenCV_LIBS_POSTFIX}d)
    endforeach(lib)

elseif("${SYSTEM_NAME_UPPER}" STREQUAL "DARWIN") #-----------------------------
    set(OpenCV_VERSION "3.4.1")
    set(PREBUILT_OPENCV_DIR "mac64_opencv_${OpenCV_VERSION}")
    set(OpenCV_DIR "${PREBUILT_PATH}/${PREBUILT_OPENCV_DIR}")
    set(OpenCV_LINK_DIR "${OpenCV_DIR}/${CMAKE_BUILD_TYPE}")
    set(OpenCV_INCLUDE_DIR "${OpenCV_DIR}/include")
    set(PREBUILT_ZIP "${PREBUILT_OPENCV_DIR}.zip")

    if (NOT EXISTS "${OpenCV_DIR}")
        file(DOWNLOAD "${PREBUILT_URL}/${PREBUILT_ZIP}" "${PREBUILT_PATH}/${PREBUILT_ZIP}")
        execute_process(COMMAND ${CMAKE_COMMAND} -E tar xzf
            "${PREBUILT_PATH}/${PREBUILT_ZIP}"
            WORKING_DIRECTORY "${PREBUILT_PATH}")
        file(REMOVE "${PREBUILT_PATH}/${PREBUILT_ZIP}")
    endif ()

    foreach(lib ${OpenCV_LINK_LIBS})
        set(OpenCV_LIBS
            ${OpenCV_LIBS}
            optimized ${lib}
            debug ${lib})
    endforeach(lib)

elseif("${SYSTEM_NAME_UPPER}" STREQUAL "ANDROID") #---------------------------
    set(OpenCV_VERSION "3.4.1")
    set(PREBUILT_OPENCV_DIR "andV8_opencv_${OpenCV_VERSION}")
    set(OpenCV_DIR "${PREBUILT_PATH}/${PREBUILT_OPENCV_DIR}")
    set(OpenCV_LINK_DIR "${OpenCV_DIR}/${CMAKE_BUILD_TYPE}")
    set(OpenCV_INCLUDE_DIR "${OpenCV_DIR}/include")
    set(PREBUILT_ZIP "${PREBUILT_OPENCV_DIR}.zip")
	
    if (NOT EXISTS "${OpenCV_DIR}")
        file(DOWNLOAD "${PREBUILT_URL}/${PREBUILT_ZIP}" "${PREBUILT_PATH}/${PREBUILT_ZIP}")
        execute_process(COMMAND ${CMAKE_COMMAND} -E tar xzf
            "${PREBUILT_PATH}/${PREBUILT_ZIP}"
            WORKING_DIRECTORY "${PREBUILT_PATH}")
        file(REMOVE "${PREBUILT_PATH}/${PREBUILT_ZIP}")
    endif ()

    foreach(lib ${OpenCV_LINK_LIBS})
        add_library(lib_${lib} SHARED IMPORTED)
        set_target_properties(lib_${lib} PROPERTIES IMPORTED_LOCATION ${OpenCV_LINK_DIR}/lib${lib}.so)
        set(OpenCV_LIBS
            ${OpenCV_LIBS}
            lib_${lib})
    endforeach(lib)

    set(OpenCV_LIBS_DEBUG ${OpenCV_LIBS})
endif()
#==============================================================================

link_directories(${OpenCV_LINK_DIR})

