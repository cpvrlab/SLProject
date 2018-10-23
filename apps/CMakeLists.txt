if(NOT "${CMAKE_SYSTEM_NAME}" MATCHES "Android")
    add_subdirectory(exercices)
    add_subdirectory(app-Demo-Node)
endif()

add_subdirectory(app-Demo-SLProject)
