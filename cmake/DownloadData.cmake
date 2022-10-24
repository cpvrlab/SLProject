#
# Download data from pallas.ti.bfh.ch
#

set(MODELS_DIR "${SL_PROJECT_ROOT}/data/models")
set(MODELS_ZIP_PATH "${SL_PROJECT_ROOT}/data.zip")
set(MODELS_URL "http://pallas.ti.bfh.ch/data/SLProject/models.zip")

if (NOT EXISTS "${MODELS_DIR}")
    message(STATUS "Downloading models...")
    file(DOWNLOAD "${MODELS_URL}" "${MODELS_ZIP_PATH}" SHOW_PROGRESS)
    execute_process(COMMAND ${CMAKE_COMMAND} -E tar xzf
            "${MODELS_ZIP_PATH}"
            WORKING_DIRECTORY "${SL_PROJECT_ROOT}/data")
    file(REMOVE "${MODELS_ZIP_PATH}")
endif ()