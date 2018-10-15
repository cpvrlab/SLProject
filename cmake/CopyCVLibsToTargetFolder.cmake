if("${SYSTEM_NAME_UPPER}" STREQUAL "WINDOWS")
    message(STATUS "OpenCV_LINK_DIR ${OpenCV_LINK_DIR}")
    add_custom_command(TARGET ${target} POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_directory ${OpenCV_LINK_DIR} $<TARGET_FILE_DIR:${target}>)
endif()