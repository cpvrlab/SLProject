#
# CMake options downloading and installing prebuilt libs
#

#
# Download and install prebuilt libraries from pallas.ti.bfh.ch
#

set(OpenCV_VERSION)
set(OpenCV_DIR)
set(OpenCV_LINK_DIR)
set(OpenCV_INCLUDE_DIR)
set(OpenCV_LINK_LIBS
        opencv_aruco
        opencv_calib3d
        opencv_dnn
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
set(assimp_LINK_LIBS
        assimp
        IrrXML)

set(vk_DIR)
set(vk_INCLUDE_DIR)
set(vk_LINK_DIR)
set(vk_LINK_LIBS
        vulkan-1
        )

set(glfw_DIR)
set(glfw_INCLUDE_DIR)
set(glfw_LINK_DIR)
set(glfw_LINK_LIBS)

set(openssl_DIR)
set(openssl_INCLUDE_DIR)
set(openssl_LINK_DIR)
set(openssl_LINK_LIBS
        ssl
        crypto
        )

set(PREBUILT_PATH "${SL_PROJECT_ROOT}/externals/prebuilt")
set(PREBUILT_URL "http://pallas.ti.bfh.ch/libs/SLProject/_lib/prebuilt/")

function (build_external_lib SCRIPT_NAME SCRIPT_ARG LIB_NAME) 
    if (NOT EXISTS "${PREBUILT_PATH}/${LIB_NAME}")        
        execute_process(COMMAND "./${SCRIPT_NAME}" "${SCRIPT_ARG}"
            WORKING_DIRECTORY "${SL_PROJECT_ROOT}/externals/prebuild_scripts"
            )
    endif ()
endfunction ()

function (download_lib LIB_NAME)
    set(LIB_PREBUILT_DIR "${PREBUILT_PATH}/${LIB_NAME}")
    set(LIB_ZIP "${LIB_NAME}.zip")
    set(LIB_LOCK_PATH "${PREBUILT_PATH}/${LIB_NAME}.lock")

    if (NOT EXISTS "${LIB_PREBUILT_DIR}")
        if (NOT EXISTS "${LIB_LOCK_PATH}")
            # Lock the zip so only one CMake process downloads it
            # CLion for example runs one CMake process for every configuration in parallel,
            # which leads to file writing errors when all processes try to write the file simultaneously.
            file(WRITE "${LIB_LOCK_PATH}" "")

            message(STATUS "Downloading ${LIB_ZIP}")
            file(DOWNLOAD "${PREBUILT_URL}/${LIB_ZIP}" "${PREBUILT_PATH}/${LIB_ZIP}")
            execute_process(COMMAND ${CMAKE_COMMAND} -E tar xzf
                    "${PREBUILT_PATH}/${LIB_ZIP}"
                    WORKING_DIRECTORY "${PREBUILT_PATH}")
            file(REMOVE "${PREBUILT_PATH}/${LIB_ZIP}")

            file(REMOVE "${LIB_LOCK_PATH}")

            if (NOT EXISTS "${LIB_PREBUILT_DIR}")
                message(SEND_ERROR "Error downloading ${LIB_ZIPT}! Build required version yourself to location ${LIB_PREBUILT_DIR} using script in directory externals/prebuild_scripts or try another version.")
            endif ()
        else ()
            message(STATUS "${LIB_ZIP} is being downloaded by another CMake process")
        endif ()
    endif ()
endfunction()

function (copy_dlls LIBS)
    if (NOT ("${CMAKE_CXX_COMPILER_ID}" MATCHES "MSVC" OR "${CMAKE_CXX_SIMULATE_ID}" MATCHES "MSVC"))
        return ()
    endif ()

    foreach (LIB ${LIBS})
        get_target_property(DLL_PATH ${LIB} LOCATION_${CMAKE_BUILD_TYPE})
        get_filename_component(DLL_FILENAME ${DLL_PATH} NAME)
        message(STATUS "Copying ${DLL_FILENAME}")
        file(COPY ${DLL_PATH} DESTINATION ${CMAKE_BINARY_DIR}/${CMAKE_BUILD_TYPE})
    endforeach ()
endfunction ()

function(copy_dylibs LIBS)
    if (NOT COPY_LIBS_TO_CONFIG_FOLDER)
        return ()
    endif ()

    foreach (LIB ${LIBS})
        get_target_property(DYLIB_PATH ${LIB} LOCATION_${CMAKE_BUILD_TYPE})
        get_filename_component(DYLIB_FILENAME ${DYLIB_PATH} NAME)
        message(STATUS "Copying ${DYLIB_FILENAME}")
        file(COPY ${DYLIB_PATH} DESTINATION ${CMAKE_BINARY_DIR}/${CMAKE_BUILD_TYPE})
    endforeach ()
endfunction ()

#=======================================================================================================================
if ("${SYSTEM_NAME_UPPER}" STREQUAL "LINUX")

    ####################
    # OpenCV for Linux #
    ####################

    set(OpenCV_VERSION "4.7.0")
    set(OpenCV_PREBUILT_DIR "linux_opencv_${OpenCV_VERSION}")
    set(OpenCV_DIR "${PREBUILT_PATH}/${OpenCV_PREBUILT_DIR}")
    set(OpenCV_LINK_DIR "${OpenCV_DIR}/${CMAKE_BUILD_TYPE}")
    set(OpenCV_INCLUDE_DIR "${OpenCV_DIR}/include")

    # new include directory structure for opencv 4
    if ("${OpenCV_VERSION}" MATCHES "^4\.[0-9]+\.[0-9]+$")
        set(OpenCV_INCLUDE_DIR "${OpenCV_INCLUDE_DIR}/opencv4")
    endif ()

    set(OpenCV_LIBS ${OpenCV_LINK_LIBS})
    set(OpenCV_LIBS_DEBUG ${OpenCV_LIBS})
    
    build_external_lib("build_opencv_w_contrib_for_linux.sh" "${OpenCV_VERSION}" "${OpenCV_PREBUILT_DIR}")

    #################
    # g2o for Linux #
    #################
    
    set(g2o_PREBUILT_DIR "linux_g2o")
    set(g2o_DIR "${PREBUILT_PATH}/${g2o_PREBUILT_DIR}")
    set(g2o_INCLUDE_DIR "${g2o_DIR}/include")

    foreach (lib ${g2o_LINK_LIBS})
        add_library(${lib} SHARED IMPORTED)
        set_target_properties(${lib} PROPERTIES
                IMPORTED_LOCATION "${g2o_DIR}/Release/lib${lib}.so"
                IMPORTED_LOCATION_DEBUG "${g2o_DIR}/Debug/lib${lib}.so"
                INTERFACE_INCLUDE_DIRECTORIES "${g2o_INCLUDE_DIR}")
        set(g2o_LIBS ${g2o_LIBS} ${lib})
    endforeach (lib)
    
    build_external_lib("build_g2o_for_linux.sh" "" "${g2o_PREBUILT_DIR}")

    ####################
    # Assimp for Linux #
    ####################

    set(assimp_VERSION "v5.0.0")
    set(assimp_PREBUILT_DIR "linux_assimp_${assimp_VERSION}")
    set(assimp_DIR "${PREBUILT_PATH}/${assimp_PREBUILT_DIR}")
    set(assimp_INCLUDE_DIR ${assimp_DIR}/include)
    
    add_library(assimp::assimp SHARED IMPORTED)
    set_target_properties(assimp::assimp PROPERTIES
            IMPORTED_LOCATION "${assimp_DIR}/Release/libassimp.so"
            IMPORTED_LOCATION_DEBUG "${assimp_DIR}/Debug/libassimp.so"
            INTERFACE_INCLUDE_DIRECTORIES "${assimp_INCLUDE_DIR}"
            )
    add_library(assimp::irrxml STATIC IMPORTED)
    set_target_properties(assimp::irrxml PROPERTIES
            IMPORTED_LOCATION "${assimp_DIR}/Release/libIrrXML.a"
            IMPORTED_LOCATION_DEBUG "${assimp_DIR}/Debug/libIrrXML.a"
            INTERFACE_INCLUDE_DIRECTORIES "${assimp_INCLUDE_DIR}"
            )

    set(assimp_LIBS assimp::assimp assimp::irrxml)
    
    build_external_lib("build_assimp_for_linux.sh" "${assimp_VERSION}" "${assimp_PREBUILT_DIR}")

    #####################
    # OpenSSL for Linux #
    #####################

    set(openssl_PREBUILT_DIR "linux_openssl")
    set(openssl_DIR "${PREBUILT_PATH}/${openssl_PREBUILT_DIR}")
    set(openssl_INCLUDE_DIR ${openssl_DIR}/include)

    foreach (lib ${openssl_LINK_LIBS})
        add_library(${lib} STATIC IMPORTED)
        set_target_properties(${lib}
                PROPERTIES
                IMPORTED_LOCATION "${openssl_DIR}/lib/lib${lib}.a"
                INTERFACE_INCLUDE_DIRECTORIES "${openssl_INCLUDE_DIR}")
        set(openssl_LIBS ${openssl_LIBS} ${lib})
    endforeach (lib)
    
    build_external_lib("build_openssl_for_linux.sh" "OpenSSL_1_1_1h" "${openssl_PREBUILT_DIR}")

    ####################
    # Vulkan for Linux #
    ####################

    set(vk_VERSION "1.2.135.0")
    set(vk_DIR ${PREBUILT_PATH}/linux_vulkan_${vk_VERSION})
    set(vk_PREBUILT_ZIP "linux_vulkan_${vk_VERSION}.zip")
    set(vk_URL ${PREBUILT_URL}/${vk_PREBUILT_ZIP})

    if (NOT EXISTS "${vk_DIR}")
        file(DOWNLOAD "${vk_URL}" "${PREBUILT_PATH}/${vk_PREBUILT_ZIP}")
        execute_process(COMMAND ${CMAKE_COMMAND} -E tar xzf
                "${PREBUILT_PATH}/${vk_PREBUILT_ZIP}"
                WORKING_DIRECTORY "${PREBUILT_PATH}")
        file(REMOVE "${PREBUILT_PATH}/${vk_PREBUILT_ZIP}")
    endif ()

    set(vk_INCLUDE_DIR ${vk_DIR}/x86_64/include)
    set(vk_LINK_DIR ${vk_DIR}/x86_64/lib)   #don't forget to add the this link dir down at the bottom

    add_library(libvulkan SHARED IMPORTED)
    set_target_properties(libvulkan PROPERTIES IMPORTED_LOCATION "${vk_LINK_DIR}/libvulkan.so")
    set(vk_LIBS libvulkan)

    ##################
    # GLFW for Linux #
    ##################

    set(glfw_VERSION "3.3.2")
    set(glfw_PREBUILT_DIR "linux_glfw_${glfw_VERSION}")
    set(glfw_DIR "${PREBUILT_PATH}/${glfw_PREBUILT_DIR}")
    set(glfw_INCLUDE_DIR "${glfw_DIR}/include")
    
    add_library(glfw STATIC IMPORTED)
    set_target_properties(glfw PROPERTIES
            IMPORTED_LOCATION "${glfw_DIR}/Release/libglfw3.a"
            IMPORTED_LOCATION_DEBUG "${glfw_DIR}/Debug/libglfw3.a"
            INTERFACE_INCLUDE_DIRECTORIES "${glfw_INCLUDE_DIR}"
            )
    set(glfw_LIBS glfw)
    
    build_external_lib("build_glfw_for_linux.sh" "${glfw_VERSION}" "${glfw_PREBUILT_DIR}")

    ####################
    # ktx for Linux    #
    ####################

    set(ktx_VERSION "v4.0.0-beta7")
    set(ktx_PREBUILT_DIR "linux_ktx_${ktx_VERSION}")
    set(ktx_DIR "${PREBUILT_PATH}/${ktx_PREBUILT_DIR}")
    
    add_library(KTX::ktx SHARED IMPORTED)
    set_target_properties(KTX::ktx
            PROPERTIES
            IMPORTED_LOCATION "${ktx_DIR}/release/libktx.so"
            INTERFACE_INCLUDE_DIRECTORIES "${ktx_DIR}/include"
            )
    #IMPORTED_LOCATION_<CONFIG> does not seem to work on linux???!!
    set(ktx_LIBS KTX::ktx)
    
    build_external_lib("build_ktx_for_linux.sh" "${ktx_VERSION}" "${ktx_PREBUILT_DIR}")

    #######################
    # MediaPipe for Linux #
    #######################

    set(MediaPipe_VERSION "v0.8.11")
    set(MediaPipe_DIR ${PREBUILT_PATH}/linux_mediapipe_${MediaPipe_VERSION})

    add_library(MediaPipe::MediaPipe SHARED IMPORTED)
    set_target_properties(MediaPipe::MediaPipe
            PROPERTIES
            IMPORTED_LOCATION "${MediaPipe_DIR}/release/libmediapipe.so"
            IMPORTED_LOCATION_DEBUG "${MediaPipe_DIR}/debug/libmediapipe.so"
            INTERFACE_INCLUDE_DIRECTORIES "${MediaPipe_DIR}/include"
            )
    set(MediaPipe_LIBS MediaPipe::MediaPipe)
    
    link_directories(${OpenCV_LINK_DIR})
    link_directories(${g2o_LINK_DIR})
    link_directories(${assimp_LINK_DIR})
    link_directories(${vk_LINK_DIR})
    link_directories(${glfw_LINK_DIR})

elseif ("${SYSTEM_NAME_UPPER}" STREQUAL "WINDOWS") #---------------------------------------------------------------------

    ######################
    # OpenCV for Windows #
    ######################

    set(OpenCV_VERSION "4.7.0")  #live video info retrieval does not work on windows. Video file loading works. (the only one that is usable)
    #set(OpenCV_VERSION "4.5.4")  #live video info retrieval does not work on windows. Video file loading works. (the only one that is usable)
    #set(OpenCV_VERSION "4.1.2")  #live video info retrieval does not work on windows. Video file loading works. (the only one that is usable)
    #set(OpenCV_VERSION "4.3.0") #live video info retrieval does not work on windows. Video file loading does not work.
    #set(OpenCV_VERSION "3.4.1") #live video info retrieval works on windows. Video file loading does not work.
    set(OpenCV_PREBUILT_DIR "win64_opencv_${OpenCV_VERSION}")
    set(OpenCV_DIR "${PREBUILT_PATH}/${OpenCV_PREBUILT_DIR}")
    set(OpenCV_LIB_DIR "${OpenCV_DIR}/lib")
    set(OpenCV_INCLUDE_DIR "${OpenCV_DIR}/include")

    string(REPLACE "." "" OpenCV_LIBS_POSTFIX ${OpenCV_VERSION})

    foreach (lib ${OpenCV_LINK_LIBS})
        add_library(${lib} SHARED IMPORTED)
        set_target_properties(${lib} PROPERTIES
                IMPORTED_IMPLIB "${OpenCV_LIB_DIR}/${lib}${OpenCV_LIBS_POSTFIX}.lib"
                IMPORTED_LOCATION "${OpenCV_LIB_DIR}/${lib}${OpenCV_LIBS_POSTFIX}.dll"
                IMPORTED_IMPLIB_DEBUG "${OpenCV_LIB_DIR}/${lib}${OpenCV_LIBS_POSTFIX}d.lib"
                IMPORTED_LOCATION_DEBUG "${OpenCV_LIB_DIR}/${lib}${OpenCV_LIBS_POSTFIX}d.dll"
                INTERFACE_INCLUDE_DIRECTORIES "${OpenCV_INCLUDE_DIR}"
                )
        set(OpenCV_LIBS ${OpenCV_LIBS} ${lib})
    endforeach (lib)

    download_lib("${OpenCV_PREBUILT_DIR}")
    copy_dlls("${OpenCV_LIBS}")

    ###################
    # g2o for Windows #
    ###################

    if (SL_BUILD_WAI)
        set(g2o_PREBUILT_DIR "win64_g2o")
        set(g2o_DIR "${PREBUILT_PATH}/${g2o_PREBUILT_DIR}")
        set(g2o_INCLUDE_DIR "${g2o_DIR}/include")
        set(g2o_LIB_DIR "${g2o_DIR}/lib")
        set(g2o_BIN_DIR "${g2o_DIR}/bin")
    
        foreach (lib ${g2o_LINK_LIBS})
            add_library(${lib} SHARED IMPORTED)
            set_target_properties(${lib} PROPERTIES
                    IMPORTED_IMPLIB "${g2o_LIB_DIR}/${lib}.lib"
                    IMPORTED_LOCATION "${g2o_BIN_DIR}/${lib}.dll"
                    IMPORTED_IMPLIB_DEBUG "${g2o_LIB_DIR}/${lib}_d.lib"
                    IMPORTED_LOCATION_DEBUG "${g2o_BIN_DIR}/${lib}_d.dll"
                    INTERFACE_INCLUDE_DIRECTORIES "${g2o_INCLUDE_DIR}"
                    )
            set(g2o_LIBS ${g2o_LIBS} ${lib})
        endforeach (lib)
    
        download_lib("${g2o_PREBUILT_DIR}")
        copy_dlls("${g2o_LIBS}")
    endif ()

    ######################
    # Assimp for Windows #
    ######################

    if (SL_BUILD_WITH_ASSIMP)
        set(assimp_VERSION "5.0")
        set(assimp_PREBUILT_DIR "win64_assimp_${assimp_VERSION}")
        set(assimp_DIR "${PREBUILT_PATH}/${assimp_PREBUILT_DIR}")
        set(assimp_LIB_DIR "${assimp_DIR}/lib")
        set(assimp_INCLUDE_DIR "${assimp_DIR}/include")
    
        add_library(assimp SHARED IMPORTED)
        set_target_properties(assimp PROPERTIES
                IMPORTED_IMPLIB "${assimp_LIB_DIR}/assimp-mt.lib"
                IMPORTED_LOCATION "${assimp_LIB_DIR}/assimp-mt.dll"
                IMPORTED_IMPLIB_DEBUG "${assimp_LIB_DIR}/assimp-mtd.lib"
                IMPORTED_LOCATION_DEBUG "${assimp_LIB_DIR}/assimp-mtd.dll"
                INTERFACE_INCLUDE_DIRECTORIES "${assimp_INCLUDE_DIR}"
                )
        set(assimp_LIBS assimp)
    
        download_lib("${assimp_PREBUILT_DIR}")
        copy_dlls("${assimp_LIBS}")
    endif ()

    #######################
    # OpenSSL for Windows #
    #######################

    if (SL_BUILD_WITH_OPENSSL)
        set(openssl_VERSION "1.1.1h")
        set(openssl_PREBUILT_DIR "win64_openssl_${openssl_VERSION}")
        set(openssl_DIR "${PREBUILT_PATH}/${openssl_PREBUILT_DIR}")
        set(openssl_LIB_DIR "${openssl_DIR}/lib")
        set(openssl_INCLUDE_DIR "${openssl_DIR}/include")
    
        add_library(crypto STATIC IMPORTED)
        set_target_properties(crypto PROPERTIES
                IMPORTED_LOCATION "${openssl_LIB_DIR}/libcrypto_static.lib"
                INTERFACE_INCLUDE_DIRECTORIES "${openssl_INCLUDE_DIR}"
                )
        add_library(ssl STATIC IMPORTED)
        set_target_properties(ssl PROPERTIES
                IMPORTED_LOCATION "${openssl_LIB_DIR}/libssl_static.lib"
                INTERFACE_INCLUDE_DIRECTORIES "${openssl_INCLUDE_DIR}"
                )
        set(openssl_LIBS crypto ssl)
    
        download_lib("${openssl_PREBUILT_DIR}")
    endif ()

    ######################
    # Vulkan for Windows #
    ######################

    if (SL_BUILD_VULKAN_APPS)
        set(vk_VERSION "1.2.162.1")
        set(vk_PREBUILT_DIR "win64_vulkan_${vk_VERSION}")
        set(vk_DIR "${PREBUILT_PATH}/${vk_PREBUILT_DIR}")
        set(vk_INCLUDE_DIR "${vk_DIR}/Include")
        set(vk_LIB_DIR "${vk_DIR}/Lib")
    
        foreach (lib ${vk_LINK_LIBS})
            add_library(${lib} STATIC IMPORTED)
            set_target_properties(${lib} PROPERTIES
                    IMPORTED_LOCATION "${vk_LIB_DIR}/${lib}.lib"
                    INTERFACE_INCLUDE_DIRECTORIES "${vk_INCLUDE_DIR}"
                    )
            set(vk_LIBS ${vk_LIBS} ${lib})
        endforeach (lib)
    
        download_lib("${vk_PREBUILT_DIR}")
        copy_dlls("${vk_LIBS}")
    endif ()

    ####################
    # GLFW for Windows #
    ####################

    set(glfw_VERSION "3.3.2")
    set(glfw_PREBUILT_DIR "win64_glfw_${glfw_VERSION}")
    set(glfw_DIR "${PREBUILT_PATH}/${glfw_PREBUILT_DIR}")
    set(glfw_INCLUDE_DIR "${glfw_DIR}/include")
    set(glfw_LINK_DIR "${glfw_DIR}/lib-vc2019")

    add_library(glfw3dll SHARED IMPORTED)
    set_target_properties(glfw3dll PROPERTIES
            IMPORTED_IMPLIB "${glfw_LINK_DIR}/glfw3dll.lib"
            IMPORTED_LOCATION "${glfw_LINK_DIR}/glfw3.dll"
            INTERFACE_INCLUDE_DIRECTORIES "${glfw_INCLUDE_DIR}"
            )
    set(glfw_LIBS glfw3dll)

    download_lib("${glfw_PREBUILT_DIR}")
    copy_dlls("${glfw_LIBS}")

    ###################
    # KTX for Windows #
    ###################

    if (SL_BUILD_WITH_KTX)
        set(ktx_VERSION "v4.0.0-beta7")
        set(ktx_PREBUILT_DIR "win64_ktx_${ktx_VERSION}")
        set(ktx_DIR "${PREBUILT_PATH}/${ktx_PREBUILT_DIR}")
    
        add_library(KTX::ktx SHARED IMPORTED)
        set_target_properties(KTX::ktx
                PROPERTIES
                IMPORTED_IMPLIB "${ktx_DIR}/release/ktx.lib"
                IMPORTED_LOCATION "${ktx_DIR}/release/ktx.dll"
                INTERFACE_INCLUDE_DIRECTORIES "${ktx_DIR}/include"
                )
        set(ktx_LIBS KTX::ktx)
    
        download_lib("${ktx_PREBUILT_DIR}")
        copy_dlls("${ktx_LIBS}")
    endif ()

    #########################
    # MediaPipe for Windows #
    #########################

    if (SL_BUILD_WITH_MEDIAPIPE)
        set(MediaPipe_VERSION "v0.8.11")
        set(MediaPipe_PREBUILT_DIR "win64_mediapipe_${MediaPipe_VERSION}")
        set(MediaPipe_DIR "${PREBUILT_PATH}/${MediaPipe_PREBUILT_DIR}")
    
        add_library(MediaPipe::MediaPipe SHARED IMPORTED)
        set_target_properties(MediaPipe::MediaPipe
                PROPERTIES
                IMPORTED_IMPLIB "${MediaPipe_DIR}/Release/lib/mediapipe.lib"
                IMPORTED_LOCATION "${MediaPipe_DIR}/Release/bin/mediapipe.dll"
                IMPORTED_IMPLIB_DEBUG "${MediaPipe_DIR}/Debug/lib/mediapipe.lib"
                IMPORTED_LOCATION_DEBUG "${MediaPipe_DIR}/Debug/bin/mediapipe.dll"
                INTERFACE_INCLUDE_DIRECTORIES "${MediaPipe_DIR}/include")
        set(MediaPipe_LIBS MediaPipe::MediaPipe)
    
        download_lib("${MediaPipe_PREBUILT_DIR}")
        copy_dlls("${MediaPipe_LIBS}")
    endif ()

elseif ("${SYSTEM_NAME_UPPER}" STREQUAL "DARWIN" AND
        "${CMAKE_SYSTEM_PROCESSOR}" STREQUAL "x86_64") #----------------------------------------------------------------

    message(STATUS "Configure prebuilts for MacOS-x86_64")

    set(COPY_LIBS_TO_CONFIG_FOLDER TRUE)

    ###########################
    # OpenCV for MacOS-x86_64 #
    ###########################

    # Now download for MacOS
    #set(OpenCV_VERSION "3.4.1")
    #set(OpenCV_VERSION "4.1.1")
    #set(OpenCV_VERSION "4.5.0")
    set(OpenCV_VERSION "4.5.2")
    #set(OpenCV_VERSION "4.7.0") // does not work with ffmpeg
    set(OpenCV_PREBUILT_DIR "mac64_opencv_${OpenCV_VERSION}")
    set(OpenCV_DIR "${PREBUILT_PATH}/${OpenCV_PREBUILT_DIR}")
    set(OpenCV_INCLUDE_DIR "${OpenCV_DIR}/include")

    # new include directory structure for opencv 4
    if ("${OpenCV_VERSION}" MATCHES "^4\.[0-9]+\.[0-9]+$")
        set(OpenCV_INCLUDE_DIR "${OpenCV_INCLUDE_DIR}/opencv4")
    endif ()

    foreach (lib ${OpenCV_LINK_LIBS})
        add_library(${lib} SHARED IMPORTED)
        set_target_properties(${lib} PROPERTIES
                IMPORTED_LOCATION "${OpenCV_DIR}/release/lib${lib}.dylib"
                IMPORTED_LOCATION_DEBUG "${OpenCV_DIR}/debug/lib${lib}.dylib"
                INTERFACE_INCLUDE_DIRECTORIES "${OpenCV_INCLUDE_DIR}")
        set(OpenCV_LIBS ${OpenCV_LIBS} ${lib})
    endforeach (lib)

    download_lib("${OpenCV_PREBUILT_DIR}")
    copy_dylibs("${OpenCV_LIBS}")

    # Copy plist file with camera access description beside executable
    # This is needed for security purpose since MacOS Mohave
    set(MACOS_PLIST_FILE ${SL_PROJECT_ROOT}/data/config/info.plist)
    if (${CMAKE_GENERATOR} STREQUAL Xcode)
        file(COPY ${MACOS_PLIST_FILE} DESTINATION ${CMAKE_BINARY_DIR}/Debug)
        file(COPY ${MACOS_PLIST_FILE} DESTINATION ${CMAKE_BINARY_DIR}/Release)
    else ()
        file(COPY ${MACOS_PLIST_FILE} DESTINATION ${CMAKE_BINARY_DIR})
    endif ()

    ########################
    # g2o for MacOS-x86_64 #
    ########################

    set(g2o_PREBUILT_DIR "mac64_g2o")
    set(g2o_DIR "${PREBUILT_PATH}/${g2o_PREBUILT_DIR}")
    set(g2o_INCLUDE_DIR "${g2o_DIR}/include")

    foreach (lib ${g2o_LINK_LIBS})
        add_library(${lib} SHARED IMPORTED)
        set_target_properties(${lib} PROPERTIES
                IMPORTED_LOCATION "${g2o_DIR}/Debug/lib${lib}.dylib"
                INTERFACE_INCLUDE_DIRECTORIES "${g2o_INCLUDE_DIR}")
        set(g2o_LIBS ${g2o_LIBS} ${lib})
    endforeach (lib)

    download_lib("${g2o_PREBUILT_DIR}")
    copy_dylibs("${g2o_LIBS}")

    ###########################
    # Assimp for MacOS-x86_64 #
    ###########################

    set(assimp_VERSION "5.0")
    set(assimp_PREBUILT_DIR "mac64_assimp_${assimp_VERSION}")
    set(assimp_DIR "${PREBUILT_PATH}/${assimp_PREBUILT_DIR}")
    set(assimp_INCLUDE_DIR "${assimp_DIR}/include")

    foreach (lib ${assimp_LINK_LIBS})
        add_library(${lib} SHARED IMPORTED)
        set_target_properties(${lib} PROPERTIES
                IMPORTED_LOCATION ${assimp_DIR}/Release/lib${lib}.dylib
                IMPORTED_LOCATION_DEBUG ${assimp_DIR}/Debug/lib${lib}d.dylib
                INTERFACE_INCLUDE_DIRECTORIES "${assimp_INCLUDE_DIR}")
        set(assimp_LIBS ${assimp_LIBS} ${lib})
    endforeach ()

    download_lib("${assimp_PREBUILT_DIR}")
    copy_dylibs("${assimp_LIBS}")

    ###########################
    # Vulkan for MacOS-x86_64 #
    ###########################

    #set(vk_VERSION "1.2.135.0")
    #set(vk_VERSIONLIBNAME "1.2.135")
    set(vk_VERSION "1.2.162.1")
    set(vk_VERSIONLIBNAME "1.2.162")
    set(vk_PREBUILT_DIR "mac64_vulkan_${vk_VERSION}")
    set(vk_DIR "${PREBUILT_PATH}/${vk_PREBUILT_DIR}")
    set(vk_INCLUDE_DIR ${vk_DIR}/macOS/include)
    set(vk_LINK_DIR ${vk_DIR}/macOS/lib)   #don't forget to add the this link dir down at the bottom

    add_library(libvulkan SHARED IMPORTED)
    set_target_properties(libvulkan PROPERTIES IMPORTED_LOCATION "${vk_LINK_DIR}/libvulkan.dylib")
    set(vk_LIBS libvulkan)

    if (${CMAKE_GENERATOR} STREQUAL Xcode)
        file(COPY ${vk_LINK_DIR}/libvulkan.dylib DESTINATION ${CMAKE_BINARY_DIR}/Debug)
        file(COPY ${vk_LINK_DIR}/libMoltenVK.dylib DESTINATION ${CMAKE_BINARY_DIR}/Debug)
        file(COPY ${vk_LINK_DIR}/libshaderc_shared.1.dylib DESTINATION ${CMAKE_BINARY_DIR}/Debug)
        file(COPY ${vk_LINK_DIR}/libSPIRV-Tools-shared.dylib DESTINATION ${CMAKE_BINARY_DIR}/Debug)
        file(COPY ${vk_LINK_DIR}/libVkLayer_api_dump.dylib DESTINATION ${CMAKE_BINARY_DIR}/Debug)
        file(COPY ${vk_LINK_DIR}/libVkLayer_khronos_validation.dylib DESTINATION ${CMAKE_BINARY_DIR}/Debug)
        file(COPY ${vk_LINK_DIR}/libvulkan.${vk_VERSIONLIBNAME}.dylib DESTINATION ${CMAKE_BINARY_DIR}/Debug)
        file(COPY ${vk_LINK_DIR}/libshaderc_shared.dylib DESTINATION ${CMAKE_BINARY_DIR}/Debug)
        file(COPY ${vk_LINK_DIR}/libvulkan.1.dylib DESTINATION ${CMAKE_BINARY_DIR}/Debug)
        file(COPY ${vk_LINK_DIR}/libvulkan.dylib DESTINATION ${CMAKE_BINARY_DIR}/Debug)

        file(COPY ${vk_LINK_DIR}/libvulkan.dylib DESTINATION ${CMAKE_BINARY_DIR}/Release)
        file(COPY ${vk_LINK_DIR}/libMoltenVK.dylib DESTINATION ${CMAKE_BINARY_DIR}/Release)
        file(COPY ${vk_LINK_DIR}/libshaderc_shared.1.dylib DESTINATION ${CMAKE_BINARY_DIR}/Release)
        file(COPY ${vk_LINK_DIR}/libSPIRV-Tools-shared.dylib DESTINATION ${CMAKE_BINARY_DIR}/Release)
        file(COPY ${vk_LINK_DIR}/libVkLayer_api_dump.dylib DESTINATION ${CMAKE_BINARY_DIR}/Release)
        file(COPY ${vk_LINK_DIR}/libVkLayer_khronos_validation.dylib DESTINATION ${CMAKE_BINARY_DIR}/Release)
        file(COPY ${vk_LINK_DIR}/libvulkan.${vk_VERSIONLIBNAME}.dylib DESTINATION ${CMAKE_BINARY_DIR}/Release)
        file(COPY ${vk_LINK_DIR}/libshaderc_shared.dylib DESTINATION ${CMAKE_BINARY_DIR}/Release)
        file(COPY ${vk_LINK_DIR}/libvulkan.1.dylib DESTINATION ${CMAKE_BINARY_DIR}/Release)
        file(COPY ${vk_LINK_DIR}/libvulkan.dylib DESTINATION ${CMAKE_BINARY_DIR}/Release)
    endif ()

    download_lib("${vk_PREBUILT_DIR}")

    #########################
    # GLFW for MacOS-x86_64 #
    #########################

    set(glfw_VERSION "3.3.2")
    set(glfw_PREBUILT_DIR "mac64_glfw_${glfw_VERSION}")
    set(glfw_DIR "${PREBUILT_PATH}/${glfw_PREBUILT_DIR}")
    set(glfw_INCLUDE_DIR "${glfw_DIR}/include")

    add_library(glfw SHARED IMPORTED)
    set_target_properties(glfw PROPERTIES
            IMPORTED_LOCATION "${glfw_DIR}/Release/libglfw.3.dylib"
            INTERFACE_INCLUDE_DIRECTORIES "${glfw_INCLUDE_DIR}")
    set(glfw_LIBS glfw)

    download_lib("${glfw_PREBUILT_DIR}")
    copy_dylibs("${glfw_LIBS}")

    ########################
    # KTX for MacOS-x86_64 #
    ########################

    set(ktx_VERSION "v4.0.0-beta7-cpvr")
    set(ktx_PREBUILT_DIR "mac64_ktx_${ktx_VERSION}")
    set(ktx_DIR "${PREBUILT_PATH}/${ktx_PREBUILT_DIR}")
    set(ktx_INCLUDE_DIR "${ktx_DIR}/include")

    add_library(KTX::ktx SHARED IMPORTED)
    set_target_properties(KTX::ktx
            PROPERTIES
            IMPORTED_LOCATION "${ktx_DIR}/release/libktx.dylib"
            IMPORTED_LOCATION_DEBUG "${ktx_DIR}/debug/libktx.dylib"
            INTERFACE_INCLUDE_DIRECTORIES "${ktx_INCLUDE_DIR}")
    set(ktx_LIBS KTX::ktx)

    download_lib("${ktx_PREBUILT_DIR}")
    copy_dylibs("${ktx_LIBS}")

    ############################
    # openssl for MacOS-x86_64 #
    ############################

    set(openssl_VERSION "1.1.1g")
    set(openssl_PREBUILT_DIR "mac64_openssl_${openssl_VERSION}")
    set(openssl_DIR "${PREBUILT_PATH}/${openssl_PREBUILT_DIR}")
    set(openssl_INCLUDE_DIR ${openssl_DIR}/include)

    foreach (lib ${openssl_LINK_LIBS})
        add_library(${lib} STATIC IMPORTED)
        set_target_properties(${lib}
                PROPERTIES
                IMPORTED_LOCATION "${openssl_DIR}/Release/lib${lib}.a"
                INTERFACE_INCLUDE_DIRECTORIES "${openssl_INCLUDE_DIR}")
        set(openssl_LIBS ${openssl_LIBS} ${lib})
    endforeach (lib)

    download_lib("${openssl_PREBUILT_DIR}")

    ##############################
    # MediaPipe for MacOS-x86_64 #
    ##############################

    set(MediaPipe_VERSION "v0.8.11")
    set(MediaPipe_PREBUILT_DIR "mac64_mediapipe_${MediaPipe_VERSION}")
    set(MediaPipe_DIR "${PREBUILT_PATH}/${MediaPipe_PREBUILT_DIR}")
    set(MediaPipe_INCLUDE_DIR "${MediaPipe_DIR}/include")

    add_library(MediaPipe::MediaPipe SHARED IMPORTED)
    set_target_properties(MediaPipe::MediaPipe
            PROPERTIES
            IMPORTED_LOCATION "${MediaPipe_DIR}/release/libmediapipe.dylib"
            IMPORTED_LOCATION_DEBUG "${MediaPipe_DIR}/debug/libmediapipe.dylib"
            INTERFACE_INCLUDE_DIRECTORIES "${MediaPipe_INCLUDE_DIR}")
    set(MediaPipe_LIBS MediaPipe::MediaPipe)

    download_lib("${MediaPipe_PREBUILT_DIR}")
    copy_dylibs("${MediaPipe_LIBS}")

elseif ("${SYSTEM_NAME_UPPER}" STREQUAL "DARWIN" AND
        "${CMAKE_SYSTEM_PROCESSOR}" STREQUAL "arm64") #-----------------------------------------------------------------

    message(STATUS "Configure prebuilts for MacOS-arm64 -----------------------------------")

    set(COPY_LIBS_TO_CONFIG_FOLDER TRUE)

    ##########################
    # OpenCV for MacOS-arm64 #
    ##########################

    set(OpenCV_VERSION "4.7.0")
    set(OpenCV_PREBUILT_DIR "macArm64_opencv_${OpenCV_VERSION}")
    set(OpenCV_DIR "${PREBUILT_PATH}/${OpenCV_PREBUILT_DIR}")
    set(OpenCV_INCLUDE_DIR "${OpenCV_DIR}/include")

    # new include directory structure for opencv 4
    if ("${OpenCV_VERSION}" MATCHES "^4\.[0-9]+\.[0-9]+$")
        set(OpenCV_INCLUDE_DIR "${OpenCV_INCLUDE_DIR}/opencv4")
    endif ()

    foreach (lib ${OpenCV_LINK_LIBS})
        add_library(${lib} SHARED IMPORTED)
        set_target_properties(${lib} PROPERTIES
                IMPORTED_LOCATION "${OpenCV_DIR}/release/lib${lib}.dylib"
                IMPORTED_LOCATION_DEBUG "${OpenCV_DIR}/debug/lib${lib}.dylib"
                INTERFACE_INCLUDE_DIRECTORIES "${OpenCV_INCLUDE_DIR}")
        set(OpenCV_LIBS ${OpenCV_LIBS} ${lib})
    endforeach (lib)

    download_lib("${OpenCV_PREBUILT_DIR}")
    copy_dylibs("${OpenCV_LIBS}")

    # Copy plist file with camera access description beside executable
    # This is needed for security purpose since MacOS Mohave
    set(MACOS_PLIST_FILE
            ${SL_PROJECT_ROOT}/data/config/info.plist)
    if (${CMAKE_GENERATOR} STREQUAL Xcode)
        file(COPY ${MACOS_PLIST_FILE} DESTINATION ${CMAKE_BINARY_DIR}/Debug)
        file(COPY ${MACOS_PLIST_FILE} DESTINATION ${CMAKE_BINARY_DIR}/Release)
    else ()
        file(COPY ${MACOS_PLIST_FILE} DESTINATION ${CMAKE_BINARY_DIR})
    endif ()

    #######################
    # g2o for MacOS-arm64 #
    #######################

    set(g2o_PREBUILT_DIR "macArm64_g2o")
    set(g2o_DIR "${PREBUILT_PATH}/${g2o_PREBUILT_DIR}")
    set(g2o_INCLUDE_DIR "${g2o_DIR}/include")
    
    foreach (lib ${g2o_LINK_LIBS})
        add_library(${lib} SHARED IMPORTED)
        set_target_properties(${lib} PROPERTIES
                IMPORTED_LOCATION "${g2o_DIR}/Debug/lib${lib}.dylib"
                INTERFACE_INCLUDE_DIRECTORIES "${g2o_INCLUDE_DIR}")
        set(g2o_LIBS ${g2o_LIBS} ${lib})
    endforeach (lib)

    download_lib("${g2o_PREBUILT_DIR}")
    copy_dylibs("${g2o_LIBS}")

    ##########################
    # Assimp for MacOS-arm64 #
    ##########################

    set(assimp_VERSION "v5.0.1")
    set(assimp_PREBUILT_DIR "macArm64_assimp_${assimp_VERSION}")
    set(assimp_DIR "${PREBUILT_PATH}/${assimp_PREBUILT_DIR}")
    set(assimp_INCLUDE_DIR "${assimp_DIR}/include")

    foreach (lib ${assimp_LINK_LIBS})
        add_library(${lib} SHARED IMPORTED)
        set_target_properties(${lib} PROPERTIES
                IMPORTED_LOCATION ${assimp_DIR}/Release/lib${lib}.dylib
                IMPORTED_LOCATION_DEBUG ${assimp_DIR}/Debug/lib${lib}d.dylib
                INTERFACE_INCLUDE_DIRECTORIES "${assimp_INCLUDE_DIR}")
        set(assimp_LIBS ${assimp_LIBS} ${lib})
    endforeach ()

    download_lib("${assimp_PREBUILT_DIR}")
    copy_dylibs("${assimp_LIBS}")

    ###########################
    # openssl for MacOS-arm64 #
    ###########################

    set(openssl_VERSION "1.1.1g")
    set(openssl_PREBUILT_DIR "macArm64_openssl_${openssl_VERSION}")
    set(openssl_DIR "${PREBUILT_PATH}/${openssl_PREBUILT_DIR}")

    foreach (lib ${openssl_LINK_LIBS})
        add_library(${lib} STATIC IMPORTED)
        set_target_properties(${lib}
                PROPERTIES
                IMPORTED_LOCATION "${openssl_DIR}/Release/lib${lib}.a"
                INTERFACE_INCLUDE_DIRECTORIES "${openssl_DIR}/include")
        set(openssl_LIBS ${openssl_LIBS} ${lib})
    endforeach (lib)

    download_lib("${openssl_PREBUILT_DIR}")

    ########################
    # GLFW for MacOS-arm64 #
    ########################

    set(glfw_VERSION "3.3.2")
    set(glfw_PREBUILT_DIR "macArm64_glfw_${glfw_VERSION}")
    set(glfw_DIR "${PREBUILT_PATH}/${glfw_PREBUILT_DIR}")
    set(glfw_INCLUDE_DIR "${glfw_DIR}/include")

    add_library(glfw SHARED IMPORTED)
    set_target_properties(glfw PROPERTIES
            IMPORTED_LOCATION "${glfw_DIR}/Release/libglfw.3.3.dylib"
            INTERFACE_INCLUDE_DIRECTORIES "${glfw_INCLUDE_DIR}")
    set(glfw_LIBS glfw)

    download_lib("${glfw_PREBUILT_DIR}")
    copy_dylibs("${glfw_LIBS}")

    #######################
    # KTX for MacOS-arm64 #
    #######################

    set(ktx_VERSION "v4.0.0-beta7-cpvr")
    set(ktx_PREBUILT_DIR "macArm64_ktx_${ktx_VERSION}")
    set(ktx_DIR "${PREBUILT_PATH}/${ktx_PREBUILT_DIR}")
    set(ktx_INCLUDE_DIR "${ktx_DIR}/include")

    add_library(KTX::ktx SHARED IMPORTED)
    set_target_properties(KTX::ktx
            PROPERTIES
            IMPORTED_LOCATION "${ktx_DIR}/release/libktx.dylib"
            IMPORTED_LOCATION_DEBUG "${ktx_DIR}/debug/libktx.dylib"
            INTERFACE_INCLUDE_DIRECTORIES "${ktx_INCLUDE_DIR}")
    set(ktx_LIBS KTX::ktx)

    download_lib("${ktx_PREBUILT_DIR}")
    copy_dylibs("${ktx_LIBS}")

    #############################
    # MediaPipe for MacOS-arm64 #
    #############################

    set(MediaPipe_VERSION "v0.8.11")
    set(MediaPipe_PREBUILT_DIR "macArm64_mediapipe_${MediaPipe_VERSION}")
    set(MediaPipe_DIR "${PREBUILT_PATH}/${MediaPipe_PREBUILT_DIR}")
    set(MediaPipe_INCLUDE_DIR "${MediaPipe_DIR}/include")

    add_library(MediaPipe::MediaPipe SHARED IMPORTED)
    set_target_properties(MediaPipe::MediaPipe
            PROPERTIES
            IMPORTED_LOCATION "${MediaPipe_DIR}/release/libmediapipe.dylib"
            IMPORTED_LOCATION_DEBUG "${MediaPipe_DIR}/debug/libmediapipe.dylib"
            INTERFACE_INCLUDE_DIRECTORIES "${MediaPipe_INCLUDE_DIR}")
    set(MediaPipe_LIBS MediaPipe::MediaPipe)

    download_lib("${MediaPipe_PREBUILT_DIR}")
    copy_dylibs("${MediaPipe_LIBS}")

elseif ("${SYSTEM_NAME_UPPER}" STREQUAL "IOS") #------------------------------------------------------------------------

    message(STATUS "Configure prebuilts for iOS_arm64 -------------------------------------")

    ##################
    # OpenCV for iOS #
    ##################

    set(OpenCV_VERSION "4.5.0")
    set(OpenCV_PREBUILT_DIR "iosV8_opencv_${OpenCV_VERSION}")
    set(OpenCV_DIR "${PREBUILT_PATH}/${OpenCV_PREBUILT_DIR}")
    set(OpenCV_LINK_DIR "${OpenCV_DIR}/${CMAKE_BUILD_TYPE}")
    set(OpenCV_INCLUDE_DIR "${OpenCV_DIR}/include/opencv4")
    
    foreach (lib ${OpenCV_LINK_LIBS})
        add_library(${lib} STATIC IMPORTED)
        set_target_properties(${lib}
                PROPERTIES
                IMPORTED_LOCATION "${OpenCV_DIR}/release/lib${lib}.a"
                IMPORTED_LOCATION_DEBUG "${OpenCV_DIR}/debug/lib${lib}.a"
                INTERFACE_INCLUDE_DIRECTORIES "${OpenCV_DIR}/include/opencv4"
                )
        set(OpenCV_LIBS ${OpenCV_LIBS} ${lib})
    endforeach (lib)

    #add special libs
    set(OpenCV_LINK_LIBS_IOS
            libwebp
            libjpeg-turbo
            libpng
            libtiff
            zlib
            )

    foreach (lib ${OpenCV_LINK_LIBS_IOS})
        add_library(${lib} STATIC IMPORTED)
        set_target_properties(${lib}
                PROPERTIES
                IMPORTED_LOCATION "${OpenCV_DIR}/release/opencv4/3rdparty/lib${lib}.a"
                IMPORTED_LOCATION_DEBUG "${OpenCV_DIR}/debug/opencv4/3rdparty/lib${lib}.a"
                )
        set(OpenCV_LIBS ${OpenCV_LIBS} ${lib})
    endforeach (lib)

    download_lib(${OpenCV_PREBUILT_DIR})

    ###############
    # g2o for iOS #
    ###############

    set(g2o_PREBUILT_DIR "iosV8_g2o")
    set(g2o_DIR "${PREBUILT_PATH}/${g2o_PREBUILT_DIR}")
    set(g2o_INCLUDE_DIR "${g2o_DIR}/include")

    foreach (lib ${g2o_LINK_LIBS})
        add_library(${lib} SHARED IMPORTED)
        set_target_properties(${lib} PROPERTIES
                IMPORTED_LOCATION "${g2o_DIR}/Release/lib${lib}.a"
                INTERFACE_INCLUDE_DIRECTORIES "${g2o_INCLUDE_DIR}")
        set(g2o_LIBS ${g2o_LIBS} ${lib})
    endforeach (lib)

    download_lib(${g2o_PREBUILT_DIR})

    ##################
    # Assimp for iOS #
    ##################

    set(assimp_VERSION "5.0")
    set(assimp_PREBUILT_DIR "iosV8_assimp_${assimp_VERSION}")
    set(assimp_DIR "${PREBUILT_PATH}/${assimp_PREBUILT_DIR}")
    set(assimp_INCLUDE_DIR "${assimp_DIR}/include")

    foreach (lib ${assimp_LINK_LIBS})
        add_library(${lib} STATIC IMPORTED)
        set_target_properties(${lib} PROPERTIES
                IMPORTED_LOCATION ${assimp_DIR}/Release/lib${lib}.a
                IMPORTED_LOCATION_DEBUG ${assimp_DIR}/Debug/lib${lib}d.a
                INTERFACE_INCLUDE_DIRECTORIES "${assimp_INCLUDE_DIR}")
        set(assimp_LIBS ${assimp_LIBS} ${lib})
    endforeach ()

    download_lib("${assimp_PREBUILT_DIR}")

    ###################
    # openssl for iOS #
    ###################

    set(openssl_VERSION "1.1.1g")
    set(openssl_PREBUILT_DIR "iosV8_openssl_${openssl_VERSION}")
    set(openssl_DIR "${PREBUILT_PATH}/${openssl_PREBUILT_DIR}")
    set(openssl_INCLUDE_DIR ${openssl_DIR}/include)

    foreach (lib ${openssl_LINK_LIBS})
        add_library(${lib} STATIC IMPORTED)
        set_target_properties(${lib}
                PROPERTIES
                IMPORTED_LOCATION "${openssl_DIR}/release/lib${lib}.a"
                INTERFACE_INCLUDE_DIRECTORIES "${openssl_INCLUDE_DIR}")
        set(openssl_LIBS ${openssl_LIBS} ${lib})
    endforeach (lib)

    download_lib("${openssl_PREBUILT_DIR}")

    ###############
    # KTX for iOS #
    ###############

    set(ktx_VERSION "v4.0.0-beta7-cpvr")
    set(ktx_PREBUILT_DIR "iosV8_ktx_${ktx_VERSION}")
    set(ktx_DIR "${PREBUILT_PATH}/${ktx_PREBUILT_DIR}")
    set(ktx_INCLUDE_DIR "${ktx_DIR}/include")

    add_library(KTX::ktx STATIC IMPORTED)
    set_target_properties(KTX::ktx
            PROPERTIES
            IMPORTED_LOCATION "${ktx_DIR}/release/libktx.a"
            IMPORTED_LOCATION_DEBUG "${ktx_DIR}/debug/libktx.a"
            INTERFACE_INCLUDE_DIRECTORIES "${ktx_INCLUDE_DIR}")

    add_library(KTX::zstd STATIC IMPORTED)
    set_target_properties(KTX::zstd
            PROPERTIES
            IMPORTED_LOCATION "${ktx_DIR}/release/libzstd.a"
            IMPORTED_LOCATION_DEBUG "${ktx_DIR}/debug/libzstd.a"
            )

    set(ktx_LIBS KTX::ktx KTX::zstd)

    download_lib("${ktx_PREBUILT_DIR}")

elseif ("${SYSTEM_NAME_UPPER}" STREQUAL "ANDROID") #---------------------------------------------------------------------

    ######################
    # OpenCV for Android #
    ######################

    set(OpenCV_VERSION "4.5.0")
    #set(OpenCV_VERSION "3.4.1")
    set(OpenCV_PREBUILT_DIR "andV8_opencv_${OpenCV_VERSION}")
    set(OpenCV_DIR "${PREBUILT_PATH}/${OpenCV_PREBUILT_DIR}")
    set(OpenCV_LINK_DIR "${OpenCV_DIR}/${CMAKE_BUILD_TYPE}/${ANDROID_ABI}")
    set(OpenCV_INCLUDE_DIR "${OpenCV_DIR}/include")

    set(OpenCV_LINK_LIBS
            ${OpenCV_LINK_LIBS}
            cpufeatures
            IlmImf
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

        # new link libraries for opencv 4.5
        if ("${OpenCV_VERSION}" MATCHES "^4\.[5-9]+\.[0-9]+$")
            set(OpenCV_LINK_LIBS
                    ${OpenCV_LINK_LIBS}
                    ade
                    libopenjp2)
        else ()
            set(OpenCV_LINK_LIBS
                    ${OpenCV_LINK_LIBS}
                    libjasper)
        endif ()
    else ()
        set(OpenCV_LINK_LIBS
                ${OpenCV_LINK_LIBS}
                libjpeg
                libjasper)
    endif ()

    foreach (lib ${OpenCV_LINK_LIBS})
        add_library(${lib} STATIC IMPORTED)
        set_target_properties(${lib} PROPERTIES
            IMPORTED_LOCATION "${OpenCV_LINK_DIR}/lib${lib}.a"
            INTERFACE_INCLUDE_DIRECTORIES "${OpenCV_INCLUDE_DIR}")
        set(OpenCV_LIBS ${OpenCV_LIBS} ${lib})
    endforeach (lib)

    download_lib("${OpenCV_PREBUILT_DIR}")

    ###################
    # g2o for Android #
    ###################

    set(g2o_PREBUILT_DIR "andV8_g2o")
    set(g2o_DIR "${PREBUILT_PATH}/${g2o_PREBUILT_DIR}")
    set(g2o_INCLUDE_DIR "${g2o_DIR}/include")
    set(g2o_LINK_DIR "${g2o_DIR}/${CMAKE_BUILD_TYPE}/${ANDROID_ABI}")

    foreach (lib ${g2o_LINK_LIBS})
        add_library(${lib} SHARED IMPORTED)
        set_target_properties(${lib} PROPERTIES
                IMPORTED_LOCATION "${g2o_LINK_DIR}/lib${lib}.so"
                INTERFACE_INCLUDE_DIRECTORIES "${g2o_INCLUDE_DIR}")
        set(g2o_LIBS ${g2o_LIBS} ${lib})
    endforeach (lib)

    download_lib("${g2o_PREBUILT_DIR}")

    ######################
    # assimp for Android #
    ######################

    set(assimp_VERSION "v5.0.0")
    set(assimp_PREBUILT_DIR "andV8_assimp_${assimp_VERSION}")
    set(assimp_DIR "${PREBUILT_PATH}/${assimp_PREBUILT_DIR}")
    set(assimp_INCLUDE_DIR "${assimp_DIR}/include")
    set(assimp_LINK_DIR "${assimp_DIR}/Release/${ANDROID_ABI}")

    add_library(ASSIMP::assimp SHARED IMPORTED)
    set_target_properties(ASSIMP::assimp PROPERTIES
            IMPORTED_LOCATION "${assimp_LINK_DIR}/libassimp.so"
            INTERFACE_INCLUDE_DIRECTORIES "${assimp_INCLUDE_DIR}")
    set(assimp_LIBS ${assimp_LIBS} ASSIMP::assimp)

    download_lib("${assimp_PREBUILT_DIR}")

    #######################
    # openssl for Android #
    #######################

    set(openssl_VERSION "1.1.1h")
    set(openssl_PREBUILT_DIR "andV8_openssl_${openssl_VERSION}")
    set(openssl_DIR "${PREBUILT_PATH}/${openssl_PREBUILT_DIR}")
    set(openssl_INCLUDE_DIR "${openssl_DIR}/include")
    set(openssl_LINK_DIR "${openssl_DIR}/lib")

    add_library(crypto STATIC IMPORTED)
    set_target_properties(crypto PROPERTIES
            IMPORTED_LOCATION "${openssl_LINK_DIR}/libcrypto.a"
            INTERFACE_INCLUDE_DIRECTORIES "${openssl_INCLUDE_DIR}")
    
    add_library(ssl STATIC IMPORTED)
    set_target_properties(ssl PROPERTIES
            IMPORTED_LOCATION "${openssl_LINK_DIR}/libssl.a"
            INTERFACE_INCLUDE_DIRECTORIES "${openssl_INCLUDE_DIR}")
    
    set(openssl_LIBS ssl crypto)

    download_lib("${openssl_PREBUILT_DIR}")

    ###################
    # KTX for Android #
    ###################

    set(ktx_VERSION "v4.0.0-beta7-cpvr")
    set(ktx_PREBUILT_DIR "andV8_ktx_${ktx_VERSION}")
    set(ktx_DIR "${PREBUILT_PATH}/${ktx_PREBUILT_DIR}")

    add_library(KTX::ktx SHARED IMPORTED)
    set_target_properties(KTX::ktx
            PROPERTIES
            IMPORTED_LOCATION "${ktx_DIR}/release/libktx.so"
            IMPORTED_LOCATION_DEBUG "${ktx_DIR}/debug/libktx.so"
            INTERFACE_INCLUDE_DIRECTORIES "${ktx_DIR}/include")
    set(ktx_LIBS KTX::ktx)

    download_lib("${ktx_PREBUILT_DIR}")

    #########################
    # MediaPipe for Android #
    #########################

    set(MediaPipe_VERSION "v0.8.11")
    set(MediaPipe_PREBUILT_DIR "andV8_mediapipe_${MediaPipe_VERSION}")
    set(MediaPipe_DIR ${PREBUILT_PATH}/${MediaPipe_PREBUILT_DIR})

    download_lib(${MediaPipe_PREBUILT_DIR})

    add_library(MediaPipe::MediaPipe SHARED IMPORTED)
    set_target_properties(MediaPipe::MediaPipe
        PROPERTIES
        IMPORTED_LOCATION_RELEASE "${MediaPipe_DIR}/release/libmediapipe.so"
        IMPORTED_LOCATION_DEBUG "${MediaPipe_DIR}/debug/libmediapipe.so"
        INTERFACE_INCLUDE_DIRECTORIES "${MediaPipe_DIR}/include"
    ) 

    add_library(MediaPipe::OpenCV_Java4 SHARED IMPORTED)
    set_target_properties(MediaPipe::OpenCV_Java4
        PROPERTIES
        IMPORTED_LOCATION_RELEASE "${MediaPipe_DIR}/release/libopencv_java4.so"
        IMPORTED_LOCATION_DEBUG "${MediaPipe_DIR}/debug/libopencv_java4.so"
    )

    set(MediaPipe_LIBS MediaPipe::MediaPipe MediaPipe::OpenCV_Java4)

elseif ("${SYSTEM_NAME_UPPER}" STREQUAL "EMSCRIPTEN")
    
    #########################
    # OpenCV for Emscripten #
    #########################

    set(OpenCV_VERSION "4.7.0")
    set(OpenCV_PREBUILT_DIR "emscripten_opencv_${OpenCV_VERSION}")
    set(OpenCV_DIR "${PREBUILT_PATH}/${OpenCV_PREBUILT_DIR}")
    set(OpenCV_INCLUDE_DIR "${OpenCV_DIR}/include")

    # new include directory structure for opencv 4
    if ("${OpenCV_VERSION}" MATCHES "^4\.[0-9]+\.[0-9]+$")
        set(OpenCV_INCLUDE_DIR "${OpenCV_INCLUDE_DIR}/opencv4")
    endif ()

    # some OpenCV modules cannot be built for Emscripten
    list(REMOVE_ITEM OpenCV_LINK_LIBS
            "opencv_highgui"
            "opencv_imgcodecs"
            "opencv_videoio")

    foreach (lib ${OpenCV_LINK_LIBS})
        add_library(${lib} STATIC IMPORTED)
        set_target_properties(${lib} PROPERTIES
            IMPORTED_LOCATION "${OpenCV_DIR}/release/lib${lib}.a"
            IMPORTED_LOCATION_DEBUG "${OpenCV_DIR}/debug/lib${lib}.a"
            INTERFACE_INCLUDE_DIRECTORIES "${OpenCV_INCLUDE_DIR}")
        set(OpenCV_LIBS ${OpenCV_LIBS} ${lib})
    endforeach (lib)

    download_lib(${OpenCV_PREBUILT_DIR})

    #########################
    # Assimp for Emscripten #
    #########################

    set(assimp_VERSION "v5.0.0")
    set(assimp_PREBUILT_DIR "emscripten_assimp_${assimp_VERSION}")
    set(assimp_DIR "${PREBUILT_PATH}/${assimp_PREBUILT_DIR}")
    set(assimp_INCLUDE_DIR "${assimp_DIR}/include")

    foreach (lib ${assimp_LINK_LIBS})
        add_library(${lib} STATIC IMPORTED)
        set_target_properties(${lib} PROPERTIES
                IMPORTED_LOCATION "${assimp_DIR}/lib/lib${lib}.a"
                INTERFACE_INCLUDE_DIRECTORIES "${assimp_INCLUDE_DIR}")
        set(assimp_LIBS ${assimp_LIBS} ${lib})
    endforeach ()

    download_lib(${assimp_PREBUILT_DIR})

    ######################
    # KTX for Emscripten #
    ######################

    set(ktx_VERSION "v4.0.0-beta7")
    set(ktx_PREBUILT_DIR "emscripten_ktx_${ktx_VERSION}")
    set(ktx_DIR "${PREBUILT_PATH}/${ktx_PREBUILT_DIR}")
    set(ktx_INCLUDE_DIR "${ktx_DIR}/include")

    add_library(KTX::ktx STATIC IMPORTED)
    set_target_properties(KTX::ktx
            PROPERTIES
            IMPORTED_LOCATION "${ktx_DIR}/release/libktx.a"
            INTERFACE_INCLUDE_DIRECTORIES "${ktx_INCLUDE_DIR}")
    set(ktx_LIBS KTX::ktx)

    download_lib(${ktx_PREBUILT_DIR})
endif ()
#==============================================================================
