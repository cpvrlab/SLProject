#
# Download 'data' directory from pallas.ti.bfh.ch
#

set(DATA_DIR "${SL_PROJECT_ROOT}/data")
set(DATA_ZIP_PATH "${SL_PROJECT_ROOT}/data.zip")
set(DATA_URL "http://pallas.ti.bfh.ch/data/SLProject-Emscripten/data.zip")

if (NOT EXISTS "${DATA_DIR}")
    message(STATUS "Downloading data...")
    file(DOWNLOAD "${DATA_URL}" "${DATA_ZIP_PATH}" SHOW_PROGRESS)
    execute_process(COMMAND ${CMAKE_COMMAND} -E tar xzf
            "${DATA_ZIP_PATH}"
            WORKING_DIRECTORY "${SL_PROJECT_ROOT}")
    file(REMOVE "${DATA_ZIP_PATH}")
endif ()