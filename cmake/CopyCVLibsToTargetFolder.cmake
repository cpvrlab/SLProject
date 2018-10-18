if("${SYSTEM_NAME_UPPER}" STREQUAL "WINDOWS")
    #message(STATUS "*** OpenCV_LINK_DIR: ${OpenCV_LINK_DIR}")
    #message(STATUS "*** TARGET_FILE_DIR: $<TARGET_FILE_DIR:${target}")
    message("Copied OpenCV DLLs on Windows")
    add_custom_command(TARGET ${target} POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_directory ${OpenCV_LINK_DIR} $<TARGET_FILE_DIR:${target}>)
endif()

#if(${CMAKE_GENERATOR} STREQUAL Xcode)
#if("${SYSTEM_NAME_UPPER}" STREQUAL "DARWIN")
#    message("Copied OpenCV DLLs from: ${OpenCV_LINK_DIR} to: $<TARGET_FILE_DIR:${target}")
#    add_custom_command(TARGET ${target} POST_BUILD
#        COMMAND ${CMAKE_COMMAND} -E copy_directory "${OpenCV_LINK_DIR}" $<TARGET_FILE_DIR:${target}>)
#endif()
