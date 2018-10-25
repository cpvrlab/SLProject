#
# Get GIT branch name and commit id
#

IF(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/.git)
    FIND_PACKAGE(Git)
    IF(GIT_FOUND)

        EXECUTE_PROCESS(
            COMMAND ${GIT_EXECUTABLE} rev-parse --abbrev-ref HEAD
            WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
            OUTPUT_VARIABLE "GitBranch"
            ERROR_QUIET
            OUTPUT_STRIP_TRAILING_WHITESPACE)
        #MESSAGE(STATUS "GitBranch: ${GitBranch}")

        EXECUTE_PROCESS(
            COMMAND ${GIT_EXECUTABLE} rev-parse --short HEAD
            WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
            OUTPUT_VARIABLE "GitCommit"
            ERROR_QUIET
            OUTPUT_STRIP_TRAILING_WHITESPACE)
        #MESSAGE(STATUS "GitCommit: ${GitCommit}")

        EXECUTE_PROCESS(
            COMMAND ${GIT_EXECUTABLE} log -1 --date=local --pretty=format:%cd
            WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
            OUTPUT_VARIABLE "GitDate"
            ERROR_QUIET
            OUTPUT_STRIP_TRAILING_WHITESPACE)
        #MESSAGE(STATUS "GitDate: ${GitDate}")

    ELSE(GIT_FOUND)
        SET(GitBranch "unknown")
        SET(GitCommit "unknown")
        SET(GitDate   "unknown")
    ENDIF(GIT_FOUND)
ENDIF()
