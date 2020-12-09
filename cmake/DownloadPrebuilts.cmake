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
        crypto
        ssl
        )

set(PREBUILT_PATH "${SL_PROJECT_ROOT}/externals/prebuilt")
set(PREBUILT_URL "http://pallas.ti.bfh.ch/libs/SLProject/_lib/prebuilt")

#=======================================================================================================================
if("${SYSTEM_NAME_UPPER}" STREQUAL "LINUX")

    ####################
    # OpenCV for Linux #
    ####################

    set(OpenCV_VERSION "4.5.0")
    set(OpenCV_DIR "${PREBUILT_PATH}/linux_opencv_${OpenCV_VERSION}")
    set(OpenCV_LINK_DIR "${OpenCV_DIR}/${CMAKE_BUILD_TYPE}")
    set(OpenCV_INCLUDE_DIR "${OpenCV_DIR}/include")

    # new include directory structure for opencv 4
    if ("${OpenCV_VERSION}" MATCHES "^4\.[0-9]+\.[0-9]+$")
        set(OpenCV_INCLUDE_DIR "${OpenCV_INCLUDE_DIR}/opencv4")
    endif()

    set(OpenCV_LIBS ${OpenCV_LINK_LIBS})
    set(OpenCV_LIBS_DEBUG ${OpenCV_LIBS})

    #################
    # g2o for Linux #
    #################

    set(g2o_DIR ${PREBUILT_PATH}/linux_g2o)
    set(g2o_INCLUDE_DIR ${g2o_DIR}/include)
    set(g2o_LINK_DIR ${g2o_DIR}/${CMAKE_BUILD_TYPE})
    set(g2o_LIBS ${g2o_LINK_LIBS})

    ####################
    # Assimp for Linux #
    ####################

    set(assimp_VERSION "v5.0.0")
    set(assimp_DIR ${PREBUILT_PATH}/linux_assimp_${assimp_VERSION})
    set(assimp_INCLUDE_DIR ${assimp_DIR}/include)
    set(assimp_LINK_DIR ${assimp_DIR}/${CMAKE_BUILD_TYPE})
    set(assimp_LIBS assimp)

    #####################
    # OpenSSL for Linux #
    #####################

    set(openssl_VERSION "1.1.1h")
    set(openssl_DIR ${PREBUILT_PATH}/linux_openssl)
    set(openssl_INCLUDE_DIR ${openssl_DIR}/include)
    set(openssl_LINK_DIR "")
    set(openssl_LIBS ssl crypto)

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
    endif()

    set(vk_INCLUDE_DIR ${vk_DIR}/x86_64/include)
    set(vk_LINK_DIR ${vk_DIR}/x86_64/lib)   #don't forget to add the this link dir down at the bottom

    add_library(libvulkan SHARED IMPORTED)
    set_target_properties(libvulkan PROPERTIES IMPORTED_LOCATION "${vk_LINK_DIR}/libvulkan.so")
    set(vk_LIBS libvulkan)

    ####################
    # GLFW for Linux #
    ####################

    set(glfw_VERSION "3.3.2")
    set(glfw_DIR ${PREBUILT_PATH}/linux_glfw_${glfw_VERSION})
    set(glfw_INCLUDE_DIR ${glfw_DIR}/include)
    set(glfw_LINK_DIR ${glfw_DIR}/${CMAKE_BUILD_TYPE})
    set(glfw_LIBS glfw3)

elseif("${SYSTEM_NAME_UPPER}" STREQUAL "WINDOWS") #---------------------------------------------------------------------

    ######################
    # OpenCV for Windows #
    #######################
	#version 4 slows down video capture. There are others with the same problem: http://www.emgu.com/forum/viewtopic.php?f=7&t=21526
    set(OpenCV_VERSION "4.1.2")  #live video info retrieval does not work on windows. Video file loading works. (the only one that is usable)
    #set(OpenCV_VERSION "4.3.0") #live video info retrieval does not work on windows. Video file loading does not work.
    #set(OpenCV_VERSION "3.4.1") #live video info retrieval works on windows. Video file loading does not work.
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

    ###################
    # g2o for Windows #
    ###################

    set(g2o_DIR ${PREBUILT_PATH}/win64_g2o)
    set(g2o_INCLUDE_DIR ${g2o_DIR}/include)
    set(g2o_LINK_DIR ${g2o_DIR}/lib)   #don't forget to add the this link dir down at the bottom

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

    ######################
    # Assimp for Windows #
    ######################

    set(assimp_VERSION "5.0")
    set(assimp_PREBUILT_DIR "win64_assimp_${assimp_VERSION}")
    set(assimp_DIR "${PREBUILT_PATH}/${assimp_PREBUILT_DIR}")
    set(assimp_LINK_DIR "${assimp_DIR}/lib")   #don't forget to add the this link dir down at the bottom
    set(assimp_INCLUDE_DIR "${assimp_DIR}/include")
    set(assimp_PREBUILT_ZIP "${assimp_PREBUILT_DIR}.zip")
    #set(assimp_LINK_LIBS_WIN assimp-mt)

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

    #######################
    # OpenSSL for windows #
    ######################

    set(openssl_VERSION "1.1.1h")
    set(openssl_PREBUILT_DIR "win64_openssl")
    set(openssl_DIR ${PREBUILT_PATH}/win64_openssl)
    set(openssl_INCLUDE_DIR ${openssl_DIR}/include)
    set(openssl_LINK_DIR ${openssl_DIR}/${CMAKE_BUILD_TYPE})
    set(openssl_LIBS ssl crypto)
    set(openssl_PREBUILT_ZIP "${openssl_PREBUILT_DIR}.zip")

    if (NOT EXISTS "${openssl_DIR}")
        file(DOWNLOAD "${PREBUILT_URL}/${openssl_PREBUILT_ZIP}" "${PREBUILT_PATH}/${openssl_PREBUILT_ZIP}")
        execute_process(COMMAND ${CMAKE_COMMAND} -E tar xzf
            "${PREBUILT_PATH}/${openssl_PREBUILT_ZIP}"
            WORKING_DIRECTORY "${PREBUILT_PATH}")
        file(REMOVE "${PREBUILT_PATH}/${openssl_PREBUILT_ZIP}")
    endif ()
    link_directories(${openssl_LINK_DIR})


    ######################
    # Vulkan for Windows #
    ######################

    set(vk_VERSION "1.2.131.2")
    set(vk_DIR ${PREBUILT_PATH}/win64_vulkan_${vk_VERSION})
    set(vk_PREBUILT_ZIP "win64_vulkan_${vk_VERSION}.zip")
    set(vk_URL ${PREBUILT_URL}/${vk_PREBUILT_ZIP})

    if (NOT EXISTS "${vk_DIR}")
        file(DOWNLOAD "${vk_URL}" "${PREBUILT_PATH}/${vk_PREBUILT_ZIP}")
        execute_process(COMMAND ${CMAKE_COMMAND} -E tar xzf
                "${PREBUILT_PATH}/${vk_PREBUILT_ZIP}"
                WORKING_DIRECTORY "${PREBUILT_PATH}")
        file(REMOVE "${PREBUILT_PATH}/${vk_PREBUILT_ZIP}")
    endif()

    set(vk_INCLUDE_DIR ${vk_DIR}/Include)
    set(vk_LINK_DIR ${vk_DIR}/Lib)   #don't forget to add the this link dir down at the bottom

    foreach(lib ${vk_LINK_LIBS})
        add_library(${lib} SHARED IMPORTED)
        set_target_properties(${lib} PROPERTIES
                IMPORTED_IMPLIB "${vk_LINK_DIR}/${lib}.lib"
                INTERFACE_INCLUDE_DIRECTORIES "${vk_INCLUDE_DIR}"
                )
        set(vk_LIBS
                ${vk_LIBS}
                ${lib}
                )
    endforeach(lib)

    ####################
    # GLFW for Windows #
    ####################

    set(glfw_VERSION "3.3.2")
    set(glfw_DIR ${PREBUILT_PATH}/win64_glfw_${glfw_VERSION})
    set(glfw_PREBUILT_ZIP "win64_glfw_${glfw_VERSION}.zip")
    set(glfw_URL ${PREBUILT_URL}/${glfw_PREBUILT_ZIP})

    if (NOT EXISTS "${glfw_DIR}")
        file(DOWNLOAD "${glfw_URL}" "${PREBUILT_PATH}/${glfw_PREBUILT_ZIP}")
        execute_process(COMMAND ${CMAKE_COMMAND} -E tar xzf
                "${PREBUILT_PATH}/${glfw_PREBUILT_ZIP}"
                WORKING_DIRECTORY "${PREBUILT_PATH}")
        file(REMOVE "${PREBUILT_PATH}/${glfw_PREBUILT_ZIP}")
    endif()

    set(glfw_INCLUDE_DIR  ${glfw_DIR}/include)
    set(glfw_LINK_DIR ${glfw_DIR}/lib-vc2019) # don't forget to add the this link dir down at the bottom

    add_library(glfw3dll SHARED IMPORTED)
    set_target_properties(glfw3dll PROPERTIES
            IMPORTED_IMPLIB "${glfw_LINK_DIR}/glfw3dll.lib"
            IMPORTED_LOCATION "${glfw_LINK_DIR}/glfw3.dll"
            INTERFACE_INCLUDE_DIRECTORIES "${glfw_INCLUDE_DIR}"
            )

    set(glfw_LIBS glfw3dll)

    # Set working dir for VS
    set(DEFAULT_PROJECT_OPTIONS ${DEFAULT_PROJECT_OPTIONS}
            VS_DEBUGGER_WORKING_DIRECTORY ${CMAKE_RUNTIME_OUTPUT_DIRECTORY})

    # For MSVS copy them to working dir
    if ("${CMAKE_CXX_COMPILER_ID}" MATCHES "MSVC")
        file(COPY ${glfw_LINK_DIR}/glfw3.dll DESTINATION ${CMAKE_BINARY_DIR}/Debug)
        file(COPY ${glfw_LINK_DIR}/glfw3.dll DESTINATION ${CMAKE_BINARY_DIR}/Release)
        file(COPY ${glfw_LINK_DIR}/glfw3.dll DESTINATION ${CMAKE_BINARY_DIR}/RelWithDebInfo)
    endif()

elseif("${SYSTEM_NAME_UPPER}" STREQUAL "DARWIN") #----------------------------------------------------------------------

	set(COPY_LIBS_TO_CONFIG_FOLDER TRUE)
	
    ####################
    # OpenCV for MacOS #
    ####################

    # Now download for MacOS
    #set(OpenCV_VERSION "4.1.1")
	#set(OpenCV_VERSION "3.4.1")
    set(OpenCV_VERSION "4.5.0")
    set(OpenCV_PREBUILT_DIR "mac64_opencv_${OpenCV_VERSION}")
    set(OpenCV_DIR "${PREBUILT_PATH}/${OpenCV_PREBUILT_DIR}")
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
        add_library(${lib} SHARED IMPORTED)
        set_target_properties(${lib} 
			PROPERTIES 
			IMPORTED_LOCATION_DEBUG "${OpenCV_DIR}/debug/lib${lib}.dylib"
			IMPORTED_LOCATION_RELEASE "${OpenCV_DIR}/release/lib${lib}.dylib")
			
		#message(STATUS ${lib})
        set(OpenCV_LIBS
                ${OpenCV_LIBS}
                optimized ${lib}
                debug ${lib})
    endforeach(lib)
	
	if (COPY_LIBS_TO_CONFIG_FOLDER)
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
			file(COPY ${OpenCV_LIBS_to_copy_release} DESTINATION ${CMAKE_BINARY_DIR}/RelWithDebInfo)
		endif()
	endif()

    # Copy plist file with camera access description beside executable
    # This is needed for security purpose since MacOS Mohave
    set(MACOS_PLIST_FILE
        ${SL_PROJECT_ROOT}/data/config/info.plist)
    if(${CMAKE_GENERATOR} STREQUAL Xcode)
        file(COPY ${MACOS_PLIST_FILE} DESTINATION ${CMAKE_BINARY_DIR}/Debug)
        file(COPY ${MACOS_PLIST_FILE} DESTINATION ${CMAKE_BINARY_DIR}/Release)
    else()
        file(COPY ${MACOS_PLIST_FILE} DESTINATION ${CMAKE_BINARY_DIR})
    endif()

    #################
    # g2o for MacOS #
    #################
	
    #Download g2o for MacOS
    set(g2o_DIR ${PREBUILT_PATH}/mac64_g2o)
    set(g2o_PREBUILT_ZIP "mac64_g2o.zip")
    set(g2o_URL ${PREBUILT_URL}/${g2o_PREBUILT_ZIP})
    set(g2o_INCLUDE_DIR ${g2o_DIR}/include)
	set(g2o_LINK_DIR ${g2o_DIR}) 

    if (NOT EXISTS "${g2o_DIR}")
        file(DOWNLOAD "${PREBUILT_URL}/${g2o_PREBUILT_ZIP}" "${PREBUILT_PATH}/${g2o_PREBUILT_ZIP}")
        execute_process(COMMAND ${CMAKE_COMMAND} -E tar xzf
            "${PREBUILT_PATH}/${g2o_PREBUILT_ZIP}"
            WORKING_DIRECTORY "${PREBUILT_PATH}")
        file(REMOVE "${PREBUILT_PATH}/${g2o_PREBUILT_ZIP}")
    endif ()

	#message(STATUS "g2o_LINK_DIR: ${g2o_LINK_DIR}")
    foreach(lib ${g2o_LINK_LIBS})
        add_library(lib${lib} SHARED IMPORTED)
        set_target_properties(lib${lib} 
			PROPERTIES 
			IMPORTED_LOCATION_DEBUG "${g2o_DIR}/Debug/lib${lib}.dylib"
			IMPORTED_LOCATION_RELEASE "${g2o_DIR}/Release/lib${lib}.dylib")
			
        set(g2o_LIBS
            ${g2o_LIBS}
            lib${lib}
            )
    endforeach(lib)
	
	if (COPY_TO_CONFIG_FOLDER)	
	    file(GLOB g2o_LIBS_to_copy_debug
	            ${g2o_LIBS_to_copy_debug}
	            ${g2o_DIR}/Debug/lib${lib}.dylib
	            )
	    file(GLOB g2o_LIBS_to_copy_release
	            ${g2o_LIBS_to_copy_release}
	            ${g2o_DIR}/Release/lib${lib}.dylib
	            )

	    if(${CMAKE_GENERATOR} STREQUAL Xcode)
	        file(COPY ${g2o_LIBS_to_copy_debug} DESTINATION ${CMAKE_BINARY_DIR}/Debug)
	        file(COPY ${g2o_LIBS_to_copy_release} DESTINATION ${CMAKE_BINARY_DIR}/Release)
	        file(COPY ${g2o_LIBS_to_copy_release} DESTINATION ${CMAKE_BINARY_DIR}/RelWithDebInfo)
	    endif()
	endif()

    ####################
    # Assimp for MacOS #
    ####################

    # Download now for macos
    set(assimp_VERSION "5.0")
    set(assimp_PREBUILT_DIR "mac64_assimp_${assimp_VERSION}")
    set(assimp_DIR "${PREBUILT_PATH}/${assimp_PREBUILT_DIR}")
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

	foreach(lib ${assimp_LINK_LIBS})
		add_library(${lib} SHARED IMPORTED)
		set_target_properties(${lib} 
			PROPERTIES
			IMPORTED_LOCATION_DEBUG ${assimp_DIR}/Debug/lib${lib}d.dylib
			IMPORTED_LOCATION_RELEASE ${assimp_DIR}/Release/lib${lib}.dylib )
			
	    set(assimp_LIBS
	        ${assimp_LIBS}
			${lib})	
	endforeach()

	if (COPY_LIBS_TO_CONFIG_FOLDER)
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
	        file(COPY ${assimp_LIBS_to_copy_release} DESTINATION ${CMAKE_BINARY_DIR}/RelWithDebInfo)
	    endif()
	endif()

    ##################
    # Assimp for MacOS #
    ##################

    set(openssl_VERSION "1.1.1h")
    set(openssl_PREBUILT_DIR "mac64_openssl")
    set(openssl_DIR ${PREBUILT_PATH}/mac64_openssl)
    set(openssl_INCLUDE_DIR ${openssl_DIR}/include)
    set(openssl_LINK_DIR ${openssl_DIR}/lib)
    set(openssl_LIBS ssl crypto)
    set(openssl_PREBUILT_ZIP "${openssl_PREBUILT_DIR}.zip")

    if (NOT EXISTS "${openssl_DIR}")
        file(DOWNLOAD "${PREBUILT_URL}/${openssl_PREBUILT_ZIP}" "${PREBUILT_PATH}/${openssl_PREBUILT_ZIP}")
        execute_process(COMMAND ${CMAKE_COMMAND} -E tar xzf
            "${PREBUILT_PATH}/${openssl_PREBUILT_ZIP}"
            WORKING_DIRECTORY "${PREBUILT_PATH}")
        file(REMOVE "${PREBUILT_PATH}/${openssl_PREBUILT_ZIP}")
    endif ()
    link_directories(${openssl_LINK_DIR})


    # Copy plist file with camera access description beside executable
    # This is needed for security purpose since MacOS Mohave
    set(MACOS_PLIST_FILE ${SL_PROJECT_ROOT}/data/config/info.plist)
    if(${CMAKE_GENERATOR} STREQUAL Xcode)
        file(COPY ${MACOS_PLIST_FILE} DESTINATION ${CMAKE_BINARY_DIR}/Debug)
        file(COPY ${MACOS_PLIST_FILE} DESTINATION ${CMAKE_BINARY_DIR}/Release)
    else()
        file(COPY ${MACOS_PLIST_FILE} DESTINATION ${CMAKE_BINARY_DIR})
    endif()

    ####################
    # Vulkan for MacOS #
    ####################

    set(vk_VERSION "1.2.135.0")
	set(vk_VERSIONLIBNAME "1.2.135")
    set(vk_DIR ${PREBUILT_PATH}/mac64_vulkan_${vk_VERSION})
    set(vk_PREBUILT_ZIP "mac64_vulkan_${vk_VERSION}.zip")
    set(vk_URL ${PREBUILT_URL}/${vk_PREBUILT_ZIP})

    if (NOT EXISTS "${vk_DIR}")
        file(DOWNLOAD "${vk_URL}" "${PREBUILT_PATH}/${vk_PREBUILT_ZIP}")
        execute_process(COMMAND ${CMAKE_COMMAND} -E tar xzf
                "${PREBUILT_PATH}/${vk_PREBUILT_ZIP}"
                WORKING_DIRECTORY "${PREBUILT_PATH}")
        file(REMOVE "${PREBUILT_PATH}/${vk_PREBUILT_ZIP}")
    endif()

    set(vk_INCLUDE_DIR ${vk_DIR}/macOS/include)
    set(vk_LINK_DIR ${vk_DIR}/macOS/lib)   #don't forget to add the this link dir down at the bottom

    add_library(libvulkan SHARED IMPORTED)
    set_target_properties(libvulkan PROPERTIES IMPORTED_LOCATION "${vk_LINK_DIR}/libvulkan.dylib")
    set(vk_LIBS libvulkan)

    if(${CMAKE_GENERATOR} STREQUAL Xcode)
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
    endif()

    ##################
    # GLFW for MacOS #
    ##################

    set(glfw_VERSION "3.3.2")
    set(glfw_DIR ${PREBUILT_PATH}/mac64_glfw_${glfw_VERSION})
    set(glfw_PREBUILT_ZIP "mac64_glfw_${glfw_VERSION}.zip")
    set(glfw_URL ${PREBUILT_URL}/${glfw_PREBUILT_ZIP})

    if (NOT EXISTS "${glfw_DIR}")
        file(DOWNLOAD "${glfw_URL}" "${PREBUILT_PATH}/${glfw_PREBUILT_ZIP}")
        execute_process(COMMAND ${CMAKE_COMMAND} -E tar xzf
                "${PREBUILT_PATH}/${glfw_PREBUILT_ZIP}"
                WORKING_DIRECTORY "${PREBUILT_PATH}")
        file(REMOVE "${PREBUILT_PATH}/${glfw_PREBUILT_ZIP}")
    endif()

    set(glfw_INCLUDE_DIR  ${glfw_DIR}/include)
    set(glfw_LINK_DIR ${glfw_DIR}/lib-macos)   #don't forget to add the this link dir down at the bottom

    add_library(libglfw.3 SHARED IMPORTED)
    set_target_properties(libglfw.3 PROPERTIES IMPORTED_LOCATION "${glfw_LINK_DIR}/libglfw.3.dylib")
    set(glfw_LIBS libglfw.3)
	
	if (COPY_LIBS_TO_CONFIG_FOLDER)
	    if(${CMAKE_GENERATOR} STREQUAL Xcode)
	        file(COPY ${glfw_LINK_DIR}/libglfw.3.dylib DESTINATION ${CMAKE_BINARY_DIR}/Debug)
	        file(COPY ${glfw_LINK_DIR}/libglfw.3.dylib DESTINATION ${CMAKE_BINARY_DIR}/Release)
	    endif()
	endif()

    #####################
    # openssl for MacOS #
    #####################

    set(openssl_VERSION "1.1.1g")
    set(openssl_DIR ${PREBUILT_PATH}/mac64_openssl_${openssl_VERSION})
    set(openssl_PREBUILT_ZIP "mac64_openssl_${openssl_VERSION}.zip")
    set(openssl_URL ${PREBUILT_URL}/${openssl_PREBUILT_ZIP})

    if (NOT EXISTS "${openssl_DIR}")
        file(DOWNLOAD "${openssl_URL}" "${PREBUILT_PATH}/${openssl_PREBUILT_ZIP}")
        execute_process(COMMAND ${CMAKE_COMMAND} -E tar xzf
                "${PREBUILT_PATH}/${openssl_PREBUILT_ZIP}"
                WORKING_DIRECTORY "${PREBUILT_PATH}")
        file(REMOVE "${PREBUILT_PATH}/${openssl_PREBUILT_ZIP}")
    endif()

    set(openssl_INCLUDE_DIR  ${openssl_DIR}/include)
    set(openssl_LINK_DIR ${openssl_DIR}/release)   #don't forget to add the this link dir down at the bottom
    link_directories(${openssl_LINK_DIR})

    foreach(lib ${openssl_LINK_LIBS})
        add_library(${lib} STATIC IMPORTED)
        set_target_properties(${lib}
                PROPERTIES
                #we use Release libs for both configurations
                IMPORTED_LOCATION_DEBUG "${openssl_DIR}/Release/lib${lib}.a"
                IMPORTED_LOCATION_RELEASE "${openssl_DIR}/Release/lib${lib}.a"
                INTERFACE_INCLUDE_DIRECTORIES "${openssl_INCLUDE_DIR}"
                )

        set(openssl_LIBS
                ${openssl_LIBS}
                ${lib}
                )
    endforeach(lib)
	
elseif("${SYSTEM_NAME_UPPER}" STREQUAL "IOS") #-------------------------------------------------------------------------
		
    ##################
    # OpenCV for iOS #
    ##################

    # Download first for iOS
    set(OpenCV_VERSION "4.2.0")
    set(OpenCV_PREBUILT_DIR "iosV8_opencv_${OpenCV_VERSION}")
    set(OpenCV_DIR "${PREBUILT_PATH}/${OpenCV_PREBUILT_DIR}")
    set(OpenCV_LINK_DIR "${OpenCV_DIR}/${CMAKE_BUILD_TYPE}")   # don't forget to add the this link dir down at the bottom
	    set(OpenCV_INCLUDE_DIR "${OpenCV_DIR}/include/opencv4")
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
	
    foreach(lib ${OpenCV_LINK_LIBS})
        add_library(${lib} STATIC IMPORTED)
        set_target_properties(${lib} 
			PROPERTIES 
			IMPORTED_LOCATION_DEBUG "${OpenCV_DIR}/debug/lib${lib}.a"
			IMPORTED_LOCATION_RELEASE "${OpenCV_DIR}/release/lib${lib}.a"
			INTERFACE_INCLUDE_DIRECTORIES "${OpenCV_DIR}/include/opencv4"
		)
			
		#ATTENTION: debug and optimized seams to mess things up in ios
        #set(OpenCV_LIBS
        #        ${OpenCV_LIBS}
        #        optimized ${lib}
        #        debug ${lib})
		set(OpenCV_LIBS
		    ${OpenCV_LIBS}
		    ${lib})
    endforeach(lib)
	
	
	#add special libs
	set(OpenCV_LINK_LIBS_IOS
		libwebp
		libjpeg-turbo
		libpng
		zlib
	)
	
    foreach(lib ${OpenCV_LINK_LIBS_IOS})
        add_library(${lib} STATIC IMPORTED)
        set_target_properties(${lib} 
			PROPERTIES 
			IMPORTED_LOCATION_DEBUG "${OpenCV_DIR}/debug/opencv4/3rdparty/lib${lib}.a"
			IMPORTED_LOCATION_RELEASE "${OpenCV_DIR}/release/opencv4/3rdparty/lib${lib}.a"
		)
			
		set(OpenCV_LIBS
		    ${OpenCV_LIBS}
		    ${lib})
    endforeach(lib)
	
    ###############
    # g2o for iOS #
    ###############

    set(g2o_DIR ${PREBUILT_PATH}/iosV8_g2o)
    set(g2o_PREBUILT_ZIP "iosV8_g2o.zip")
    set(g2o_URL ${PREBUILT_URL}/${g2o_PREBUILT_ZIP})
    set(g2o_INCLUDE_DIR ${g2o_DIR}/include)

    if (NOT EXISTS "${g2o_DIR}")
        message(STATUS "g2o_DIR: ${g2o_DIR}")
        message(STATUS "${PREBUILT_URL}/${g2o_PREBUILT_ZIP}")
        message(STATUS "Download to: ${PREBUILT_PATH}/${g2o_PREBUILT_ZIP}")
        file(DOWNLOAD "${PREBUILT_URL}/${g2o_PREBUILT_ZIP}" "${PREBUILT_PATH}/${g2o_PREBUILT_ZIP}")
        execute_process(COMMAND ${CMAKE_COMMAND} -E tar xzf
                "${PREBUILT_PATH}/${g2o_PREBUILT_ZIP}"
                WORKING_DIRECTORY "${PREBUILT_PATH}")
        file(REMOVE "${PREBUILT_PATH}/${g2o_PREBUILT_ZIP}")
    endif ()
	
    foreach(lib ${g2o_LINK_LIBS})
        add_library(${lib} STATIC IMPORTED)
        set_target_properties(${lib} 
			PROPERTIES 
			#we use Release libs for both configurations
			IMPORTED_LOCATION_DEBUG "${g2o_DIR}/Release/lib${lib}.a"
			IMPORTED_LOCATION_RELEASE "${g2o_DIR}/Release/lib${lib}.a"
			INTERFACE_INCLUDE_DIRECTORIES "${g2o_INCLUDE_DIR}"
		)
				
        set(g2o_LIBS
            ${g2o_LIBS}
            ${lib}
            )
    endforeach(lib)
	
    ##################
    # Assimp for iOS #
    ##################

    # Download first for iOS
    set(assimp_VERSION "5.0")
    set(assimp_PREBUILT_DIR "iosV8_assimp_${assimp_VERSION}")
    set(assimp_DIR "${PREBUILT_PATH}/${assimp_PREBUILT_DIR}")
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
	
	foreach(lib ${assimp_LINK_LIBS})
		add_library(${lib} STATIC IMPORTED)
		set_target_properties(${lib} 
			PROPERTIES
			IMPORTED_LOCATION_DEBUG "${assimp_DIR}/Debug/lib${lib}d.a"
			IMPORTED_LOCATION_RELEASE "${assimp_DIR}/Release/lib${lib}.a" 
			INTERFACE_INCLUDE_DIRECTORIES "${assimp_DIR}/include" 
		)
			
	    set(assimp_LIBS
	        ${assimp_LIBS}
			${lib})	
	endforeach()

    ###################
    # openssl for iOS #
    ###################

    set(openssl_VERSION "1.1.1g")
    set(openssl_DIR ${PREBUILT_PATH}/iosV8_openssl_${openssl_VERSION})
    set(openssl_PREBUILT_ZIP "iosV8_openssl_${openssl_VERSION}.zip")
    set(openssl_URL ${PREBUILT_URL}/${openssl_PREBUILT_ZIP})

    if (NOT EXISTS "${openssl_DIR}")
        file(DOWNLOAD "${openssl_URL}" "${PREBUILT_PATH}/${openssl_PREBUILT_ZIP}")
        execute_process(COMMAND ${CMAKE_COMMAND} -E tar xzf
                "${PREBUILT_PATH}/${openssl_PREBUILT_ZIP}"
                WORKING_DIRECTORY "${PREBUILT_PATH}")
        file(REMOVE "${PREBUILT_PATH}/${openssl_PREBUILT_ZIP}")
    endif()

    set(openssl_INCLUDE_DIR  ${openssl_DIR}/include)
    set(openssl_LINK_DIR ${openssl_DIR}/release)   #don't forget to add the this link dir down at the bottom
    link_directories(${openssl_LINK_DIR})

    foreach(lib ${openssl_LINK_LIBS})
        add_library(${lib} STATIC IMPORTED)
        set_target_properties(${lib}
                PROPERTIES
                #we use Release libs for both configurations
                IMPORTED_LOCATION_DEBUG "${openssl_DIR}/Release/lib${lib}.a"
                IMPORTED_LOCATION_RELEASE "${openssl_DIR}/Release/lib${lib}.a"
                INTERFACE_INCLUDE_DIRECTORIES "${openssl_INCLUDE_DIR}"
                )

        set(openssl_LIBS
                ${openssl_LIBS}
                ${lib}
                )
    endforeach(lib)
	
elseif("${SYSTEM_NAME_UPPER}" STREQUAL "ANDROID") #---------------------------------------------------------------------

    ######################
    # OpenCV for Android #
    ######################

    set(OpenCV_VERSION "4.5.0")
    #set(OpenCV_VERSION "3.4.1")
    set(OpenCV_PREBUILT_DIR "andV8_opencv_${OpenCV_VERSION}")
    set(OpenCV_DIR "${PREBUILT_PATH}/${OpenCV_PREBUILT_DIR}")
    set(OpenCV_LINK_DIR "${OpenCV_DIR}/${CMAKE_BUILD_TYPE}/${ANDROID_ABI}")   #don't forget to add the this link dir down at the bottom
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
        else()
            set(OpenCV_LINK_LIBS
                ${OpenCV_LINK_LIBS}
                libjasper)
        endif()
    else()
        set(OpenCV_LINK_LIBS
            ${OpenCV_LINK_LIBS}
            libjpeg
            libjasper)
    endif()

    foreach(lib ${OpenCV_LINK_LIBS})
        add_library(lib_${lib} STATIC IMPORTED)
        set_target_properties(lib_${lib} PROPERTIES IMPORTED_LOCATION ${OpenCV_LINK_DIR}/lib${lib}.a)
        set(OpenCV_LIBS
                ${OpenCV_LIBS}
                lib_${lib})
    endforeach(lib)

    set(OpenCV_LIBS_DEBUG ${OpenCV_LIBS})

    ###################
    # g2o for Android #
    ###################

    set(g2o_PREBUILT_DIR "andV8_g2o")
    set(g2o_DIR ${PREBUILT_PATH}/andV8_g2o)
    set(g2o_INCLUDE_DIR ${g2o_DIR}/include)
    set(g2o_LINK_DIR ${g2o_DIR}/${CMAKE_BUILD_TYPE}/${ANDROID_ABI})   #don't forget to add the this link dir down at the bottom
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

    ######################
    # assimp for Android #
    ######################

    set(assimp_VERSION "v5.0.0")
    set(assimp_PREBUILT_DIR "andV8_assimp_${assimp_VERSION}")
    set(assimp_DIR ${PREBUILT_PATH}/${assimp_PREBUILT_DIR})
    set(assimp_INCLUDE_DIR ${assimp_DIR}/include)
    set(assimp_LINK_DIR ${assimp_DIR}/${CMAKE_BUILD_TYPE}/${ANDROID_ABI})  #don't forget to add the this link dir down at the bottom
    set(assimp_PREBUILT_ZIP "${assimp_PREBUILT_DIR}.zip")

    if (NOT EXISTS "${assimp_DIR}")
        file(DOWNLOAD "${PREBUILT_URL}/${assimp_PREBUILT_ZIP}" "${PREBUILT_PATH}/${assimp_PREBUILT_ZIP}")
        execute_process(COMMAND ${CMAKE_COMMAND} -E tar xzf
            "${PREBUILT_PATH}/${assimp_PREBUILT_ZIP}"
            WORKING_DIRECTORY "${PREBUILT_PATH}")
        file(REMOVE "${PREBUILT_PATH}/${assimp_PREBUILT_ZIP}")
    endif ()

    foreach(lib ${assimp_LINK_LIBS})
        add_library(lib_${lib} STATIC IMPORTED)
        set_target_properties(lib_${lib} PROPERTIES
            IMPORTED_LOCATION "${assimp_LINK_DIR}/lib${lib}.a"
        )
        set(assimp_LIBS
            ${assimp_LIBS}
            lib_${lib}
        )
    endforeach(lib)

    #######################
    # openssl for Android #
    #######################

    set(openssl_VERSION "1.1.1h")
    set(openssl_PREBUILT_DIR "andV8_openssl")
    set(openssl_DIR ${PREBUILT_PATH}/andV8_openssl)
    set(openssl_INCLUDE_DIR ${openssl_DIR}/include)
    set(openssl_LINK_DIR ${openssl_DIR}/lib)
    set(openssl_LIBS ssl crypto)
    set(openssl_PREBUILT_ZIP "${openssl_PREBUILT_DIR}.zip")
    link_directories(${openssl_LINK_DIR})

    if (NOT EXISTS "${openssl_DIR}")
        file(DOWNLOAD "${PREBUILT_URL}/${openssl_PREBUILT_ZIP}" "${PREBUILT_PATH}/${openssl_PREBUILT_ZIP}")
        execute_process(COMMAND ${CMAKE_COMMAND} -E tar xzf
            "${PREBUILT_PATH}/${openssl_PREBUILT_ZIP}"
            WORKING_DIRECTORY "${PREBUILT_PATH}")
        file(REMOVE "${PREBUILT_PATH}/${openssl_PREBUILT_ZIP}")
    endif ()


endif()
#==============================================================================

link_directories(${OpenCV_LINK_DIR})
link_directories(${g2o_LINK_DIR})
link_directories(${assimp_LINK_DIR})
link_directories(${vk_LINK_DIR})
link_directories(${glfw_LINK_DIR})
