/*!
 * \file    peak_environment_inspector.hpp
 *
 * \author  IDS Imaging Development Systems GmbH
 * \date    2019-05-01
 * \since   1.0
 *
 * Copyright (c) 2019, IDS Imaging Development Systems GmbH. All rights reserved.
 */

#pragma once


#include <peak/backend/peak_backend.h>
#include <peak/dll_interface/peak_dll_interface_util.hpp>

#include <string>
#include <vector>


namespace peak
{
/*!
 * \brief The "core" namespace contains the direct GenAPI and GenTL wrapper.
 */
namespace core
{

/*!
 * \brief Allows to inspect the environment the application is running in.
 */
class EnvironmentInspector final
{
public:
    EnvironmentInspector() = delete;
    ~EnvironmentInspector() = delete;
    EnvironmentInspector(const EnvironmentInspector& other) = delete;
    EnvironmentInspector& operator=(const EnvironmentInspector& other) = delete;
    EnvironmentInspector(EnvironmentInspector&& other) = delete;
    EnvironmentInspector& operator=(EnvironmentInspector&& other) = delete;

    /*!
     * \brief Collects producer library paths found in the current environment.
     *
     * Select a *.cti file from this list of paths and use it with ProducerLibrary::Open().
     *
     * \note This function depends on the architecture your application is compiled for. This means you are getting
     *       the paths for the 32-bit producer libraries if your application is compiled for a 32-bit system and the
     *       paths for the 64-bit producer libraries if your application is compiled for a 64-bit system.
     *
     * \return Producer library paths found in the current environment
     *
     * \since 1.0
     *
     * \throws InternalErrorException An internal error has occurred.
     * \throws NotFoundException The environment variable GENICAM_GENTL32_PATH / GENICAM_GENTL64_PATH was not
     *                           found or was empty when scanning for environment ProducerLibraries.
     */
    static std::vector<std::string> CollectCTIPaths();
};

} /* namespace core */
} /* namespace peak */

/* Implementation */
namespace peak
{
namespace core
{

inline std::vector<std::string> EnvironmentInspector::CollectCTIPaths()
{
    CallAndCheckCInterfaceFunction([&] { return PEAK_C_ABI_PREFIX PEAK_EnvironmentInspector_UpdateCTIPaths(); });

    auto ctiPaths = QueryStringArrayFromCInterfaceFunction(
        [&](size_t* numCtiPaths) {
            return PEAK_C_ABI_PREFIX PEAK_EnvironmentInspector_GetNumCTIPaths(numCtiPaths);
        },
        [&](size_t index, char* ctiPath, size_t* ctiPathSize) {
            return PEAK_C_ABI_PREFIX PEAK_EnvironmentInspector_GetCTIPath(index, ctiPath, ctiPathSize);
        });

    return ctiPaths;
}

} /* namespace core */
} /* namespace peak */
