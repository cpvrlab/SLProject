#
# Download data from pallas.ti.bfh.ch
#

set(DATA_DIR "${SL_PROJECT_ROOT}/data")
set(DATA_ZIP_PATH "${SL_PROJECT_ROOT}/data.zip")
set(DATA_LOCK_PATH "${DATA_DIR}/data.lock")
set(DATA_URL "http://pallas.ti.bfh.ch/data/SLProject/data.zip")
set(DATA_MODELS_PATH "${SL_PROJECT_ROOT}/data/models")

if (NOT EXISTS "${DATA_MODELS_PATH}")
    if (NOT EXISTS "${DATA_LOCK_PATH}")
        # Lock the zip so only one CMake process downloads it
        # CLion for example runs one CMake process for every configuration in parallel,
        # which leads to file writing errors when all processes try to write the file simultaneously.
        file(TOUCH "${DATA_LOCK_PATH}")

        message(STATUS "Downloading data zip...")
        file(DOWNLOAD "${DATA_URL}" "${DATA_ZIP_PATH}" SHOW_PROGRESS)
        execute_process(COMMAND ${CMAKE_COMMAND} -E tar xzf "${DATA_ZIP_PATH}" WORKING_DIRECTORY "${SL_PROJECT_ROOT}")
        file(REMOVE "${DATA_ZIP_PATH}")
        
        file(REMOVE "${DATA_LOCK_PATH}")
    else ()
        message(STATUS "Data zip is being downloaded by another CMake process")
    endif ()
else ()
    message(STATUS "Data directory is present")
endif ()