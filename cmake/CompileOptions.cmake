#
# Platform and architecture setup
#
# Get upper case system name
string(TOUPPER ${CMAKE_SYSTEM_NAME} SYSTEM_NAME_UPPER)
message(STATUS "SYSTEM_NAME_UPPER: ${SYSTEM_NAME_UPPER} (CompileOptions.cmake)")

# Determine architecture (32/64 bit)
set(X64 OFF)
if(CMAKE_SIZEOF_VOID_P EQUAL 8)
    set(X64 ON)
endif()

set(CMAKE_C_FLAGS_DEBUG "${CMAKE_C_FLAGS_DEBUG} -D_DEBUG")
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -D_DEBUG")
set(CMAKE_BUILD_PARALLEL_LEVEL 16)

set(DEFAULT_PROJECT_OPTIONS
    DEBUG_POSTFIX             "-debug"
    RELEASE_POSTFIX           "-release"
    CXX_STANDARD              14
    LINKER_LANGUAGE           "CXX"
    POSITION_INDEPENDENT_CODE ON
    CXX_VISIBILITY_PRESET     "hidden"
    CXX_EXTENSIONS            Off
    ENABLE_EXPORTS            ON
    )

set(DEFAULT_INCLUDE_DIRECTORIES)
set(DEFAULT_LIBRARIES)

set(DEFAULT_COMPILE_DEFINITIONS
		SYSTEM_${SYSTEM_NAME_UPPER}
		SL_PROJECT_ROOT="${SL_PROJECT_ROOT}"
		SL_GIT_BRANCH="${GitBranch}"
		SL_GIT_COMMIT="${GitCommit}"
		SL_GIT_DATE="${GitDate}"
		GL_SILENCE_DEPRECATION
    )
message(STATUS "DEFAULT_COMPILE_DEFINITIONS: ${DEFAULT_COMPILE_DEFINITIONS}")

if (SL_BUILD_WAI)
	set(DEFAULT_COMPILE_DEFINITIONS
	    ${DEFAULT_COMPILE_DEFINITIONS}
	    SL_BUILD_WAI)
endif()

if (SL_BUILD_WITH_KTX)
    set(DEFAULT_COMPILE_DEFINITIONS
        ${DEFAULT_COMPILE_DEFINITIONS}
        SL_BUILD_WITH_KTX)
endif()

if (SL_BUILD_WITH_OPENSSL)
    set(DEFAULT_COMPILE_DEFINITIONS
        ${DEFAULT_COMPILE_DEFINITIONS}
        SL_BUILD_WITH_OPENSSL)
endif()

if (SL_BUILD_WITH_ASSIMP)
    set(DEFAULT_COMPILE_DEFINITIONS
        ${DEFAULT_COMPILE_DEFINITIONS}
        SL_BUILD_WITH_ASSIMP)
endif()

if (SL_BUILD_WITH_MEDIAPIPE)
	set(DEFAULT_COMPILE_DEFINITIONS
		${DEFAULT_COMPILE_DEFINITIONS}
		SL_BUILD_WITH_MEDIAPIPE)
endif()

# MSVC compiler options
if ("${CMAKE_CXX_COMPILER_ID}" MATCHES "MSVC")
    set(DEFAULT_COMPILE_DEFINITIONS ${DEFAULT_COMPILE_DEFINITIONS}
            _SCL_SECURE_NO_WARNINGS  # Calling any one of the potentially unsafe methods in the Standard C++ Library
            _CRT_SECURE_NO_WARNINGS  # Calling any one of the potentially unsafe methods in the CRT Library
            NOMINMAX #No min max makros (use the ones in std)
            )
    set(MSVC_COMPILE_FLAGS "/MP")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${MSVC_COMPILE_FLAGS}" )
endif ()

# Compile options for compiling using clang-cl on Windows
# clang-cl is a drop-in replacement for cl (the MSVC compiler)
if ("${CMAKE_CXX_COMPILER_ID}" MATCHES "Clang" AND "${CMAKE_CXX_SIMULATE_ID}" MATCHES "MSVC")
	add_compile_definitions(_WINDOWS)

	# NOMINMAX disables the macros "min" and "max" from a Windows header that clang-cl for some reason includes
	# These macros override std::min and std::max from the C++ standard library, which blows everything up
	add_compile_definitions(NOMINMAX)

	# Set additional flags
	# /EHs              Enable Exception handling (https://docs.microsoft.com/en-us/cpp/build/reference/eh-exception-handling-model?view=msvc-170)
	# -march=native     Enable AVX intrinsics for fbow (won't compile on machines without AVX support)
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /EHsc -march=native")
endif ()

set(DEFAULT_COMPILE_OPTIONS)

# MSVC compiler options
if ("${CMAKE_CXX_COMPILER_ID}" MATCHES "MSVC")
    set(DEFAULT_COMPILE_OPTIONS ${DEFAULT_COMPILE_OPTIONS}
        	/MP         # -> build with multiple processes
        	/W3         # -> warning level 3
        	/wd4251     # -> disable warning: 'identifier': class 'type' needs to have dll-interface to be used by clients of class 'type2'
        	/wd4592     # -> disable warning: 'identifier': symbol will be dynamically initialized (implementation limitation)
        	/wd4804     # -> disable warning: unsichere Verwendung des Typs "bool" in einer Operation	C:\Users\hsm4\Documents\GitHub\SLProject	C:\Users\hsm4\Documents\GitHub\SLProject\lib-SLProject\include\SLMaterial.h	88
        	/wd26495	# -> disable warning: C26495 MEMBER_UNINIT
			/wd26812	# -> disable warning: C26812: ' enum class ' vor ' ENUM ' bevorzugen (Enum. 3)
			/wd26451    # -> disable warning: C26451: Aritmetic overflow
			/bigobj

        $<$<CONFIG:Release>:
        /Gw           # -> whole program global optimization
        /GS-          # -> buffer security check: no
        /GL           # -> whole program optimization: enable link-time code generation (disables Zi)
        /GF           # -> enable string pooling
        >
        # No manual c++11 enable for MSVC as all supported MSVC versions for cmake-init have C++11 implicitly enabled (MSVC >=2013)
    )

	set(EXTERNAL_LIB_COMPILE_OPTIONS ${EXTERNAL_LIB_COMPILE_OPTIONS}
			/MP           # -> build with multiple processes
			#/W0           # -> warning level 0 all off
			#/w

			$<$<CONFIG:Release>:
			/Gw           # -> whole program global optimization
			/GS-          # -> buffer security check: no
			/GL           # -> whole program optimization: enable link-time code generation (disables Zi)
			/GF           # -> enable string pooling
			>
			# No manual c++11 enable for MSVC as all supported MSVC versions for cmake-init have C++11 implicitly enabled (MSVC >=2013)
			)
endif ()

# GCC and Clang compiler options
if ("${CMAKE_CXX_COMPILER_ID}" MATCHES "GNU" OR "${CMAKE_CXX_COMPILER_ID}" MATCHES "Clang")
    set(DEFAULT_COMPILE_OPTIONS ${DEFAULT_COMPILE_OPTIONS}
            -Werror=return-type
            -Wno-c++98-compat
            -Wno-c++98-compat-pedantic
            -Wno-covered-switch-default
            -Wno-double-promotion
            -Wno-exit-time-destructors
            -Wno-global-constructors
            -Wno-gnu-zero-variadic-macro-arguments
            -Wno-documentation
            -Wno-missing-variable-declarations
            -Wno-newline-eof
            -Wno-old-style-cast
            -Wno-switch-enum
            -Wno-unused-macros
            -Wno-unused-function
            -Wno-unused-parameter
            -Wno-unused-variable
            -Wno-unused-private-field
            -Wno-unused-value
            -Wno-used-but-marked-unused
            -Wno-extra-tokens
            -Wno-reorder
            -Wno-switch
            -Wno-char-subscripts
            -Wno-injected-class-name
            -Wno-format-security
			-Wno-invalid-noreturn

            $<$<CXX_COMPILER_ID:GNU>:
                -Wmaybe-uninitialized
                $<$<VERSION_GREATER:$<CXX_COMPILER_VERSION>,4.8>:
                    -Wreturn-local-addr
                >
            >

            $<$<CXX_COMPILER_ID:Clang>:

            >

            $<$<PLATFORM_ID:Darwin>:
                -pthread
            >
    )

	if ("${SYSTEM_NAME_UPPER}" STREQUAL "IOS")
		#iOS minimum supported version (deployment target version of iOS)
		set(DEPLOYMENT_TARGET 12.0)

		set(DEFAULT_PROJECT_OPTIONS
			${DEFAULT_PROJECT_OPTIONS}
			XCODE_ATTRIBUTE_IPHONEOS_DEPLOYMENT_TARGET ${DEPLOYMENT_TARGET}
		)

	    set(DEFAULT_COMPILE_OPTIONS ${DEFAULT_COMPILE_OPTIONS}
			-fobjc-arc #enable automatic reference counting
		)

		set(DEFAULT_COMPILE_DEFINITIONS
				${DEFAULT_COMPILE_DEFINITIONS}
				TARGET_OS_IOS #disable GLSLExtractor
				GLES_SILENCE_DEPRECATION
		)
		#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${MSVC_COMPILE_FLAGS} -fobjc-arc" )
	endif()

	# set correct architecture for MacOS (x86_64 or arm64): Needs cmake 3.19.2
	if("${SYSTEM_NAME_UPPER}" STREQUAL "DARWIN")
		set(CMAKE_OSX_ARCHITECTURES ${CMAKE_SYSTEM_PROCESSOR})
	endif()
endif ()


# Clang only compiler options
#if ("${CMAKE_CXX_COMPILER_ID}" MATCHES "Clang")
#	if("${SYSTEM_NAME_UPPER}" STREQUAL "DARWIN")
#		set(DEFAULT_COMPILE_OPTIONS ${DEFAULT_COMPILE_OPTIONS}
#				-fsanitize=address
#			)
#	endif ()
#endif ()

set(EXTERNAL_LIB_COMPILE_OPTIONS)

#
# Linker options
#

set(DEFAULT_LINKER_OPTIONS)

# Use pthreads on mingw and linux
if("${CMAKE_CXX_COMPILER_ID}" MATCHES "GNU" OR "${CMAKE_SYSTEM_NAME}" MATCHES "Linux")
    set(DEFAULT_LINKER_OPTIONS
        -pthread
    )
endif()

# Clang only linker options
#if ("${CMAKE_CXX_COMPILER_ID}" MATCHES "Clang")
#	if("${SYSTEM_NAME_UPPER}" STREQUAL "DARWIN")
#		set(DEFAULT_LINKER_OPTIONS ${DEFAULT_LINKER_OPTIONS}
#				-fsanitize=address
#			)
#	endif ()
#endif ()

#
# Options for Emscripten
#

if ("${SYSTEM_NAME_UPPER}" MATCHES "EMSCRIPTEN")
	add_compile_options(
			"-sUSE_PTHREADS"
			#"-fsanitize=address"
	)
	add_link_options(
			# The Wasm heap has a limited size.
			# Enable growing the heap when allocating more than the initial heap size.
			"-sALLOW_MEMORY_GROWTH=1"

			# Enable assertions that provide information about errors.
			"-sASSERTIONS"

			# Enable support for pthreads, which are implemented using a pool of web workers.
			"-sUSE_PTHREADS"

			# Run the entire application in a web worker, so we can block while fetching.
			# Functions that have to run on the main thread are proxied to it using messages.
			"-sPROXY_TO_PTHREAD"

			# The canvas that WebGL renders to has to be on the same thread as the WebGL context.
			# This flags enables support for transferring canvases to a worker.
			"-sOFFSCREENCANVAS_SUPPORT=1"

			# Transfer the app canvas to the worker.
			"-sOFFSCREENCANVASES_TO_PTHREAD='#canvas'"

			# Strings are always decoded on the main thread, which accesses it using a SharedArrayBuffer.
			# Chrome doesn't support decoding from SharedArrayBuffers, so we disable the JavaScript TextDecoder API.
			# See https://github.com/emscripten-core/emscripten/issues/18034
			"-sTEXTDECODER=0"

			# Enable running global object destructors, calling atexit, flushing stdio streams, etc. when exiting.
			"-sEXIT_RUNTIME=0"
	)
endif ()