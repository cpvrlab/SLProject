add_subdirectory(lib-assimp)
add_subdirectory(lib-SLExternal)

option(SL_BUILD_UTILS
	   "Specifies if lib-Utils should be built"
	   ON)
if (SL_BUILD_UTILS)
    message(STATUS "Building lib-Utils")
    add_subdirectory(utils)
endif()
