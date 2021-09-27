/*!
 * \file    peak_library.hpp
 *
 * \author  IDS Imaging Development Systems GmbH
 * \date    2019-05-01
 * \since   1.0
 *
 * Copyright (c) 2019, IDS Imaging Development Systems GmbH. All rights reserved.
 */

#pragma once


#include <peak/version/peak_version.hpp>


/*!
 * \brief The "peak" namespace directly contains convenience functionality like the DeviceManager as well as
 * the global Library::Initialize() / Library::Close() functions, which must be called before/after usage of the
 * library.
 */
namespace peak
{

/*!
 * \brief A collection of global library functions.
 *
 * Currently, this includes Library::Initialize() / Library::Close(), which must be called before/after usage of the
 * library, and Library::Version(), to retrieve version information about the the library.
 */
class Library final
{
public:
    /*!
     * \anchor InitCloseLib1
     * \name Initialize / Close Library
     * \{
     */
    /*!
     * \brief Initializes the IDS peak API library.
     *
     * This function **must** be called prior to any other function call to allow global initialization of library
     * internals. This function is necessary since automated initialization functionality like within DllMain on MS
     * Windows platforms is very limited.
     *
     * Calling this function multiple times is ok, but note that these calls are reference counted, so you have to call
     * Close() as many times as Initialize().
     *
     * \since 1.0
     * \since 2.0 Added reference counting
     *
     * \note Calling any other function before this will result in a core::NotInitializedException.
     *
     * \throws std::runtime_error when dynamic loading is active and and error occured during loading the dynamic library.
     */
    static void Initialize();
    /*!
     * \brief Closes the IDS peak API library and cleans up any resources that are still in use.
     *
     * This function should be called after no function of the library is needed anymore, before unloading the library.
     * It cleans up any resources still in use. If an DeviceManager::Update is in progress, the update will finish
     * before the library is closed.
     *
     * \note Calling any other function (except Initialize()) after this will result in a core::NotInitializedException.
     *
     * \note Calls to Initialize() and Close() are reference counted, so you have to call Close() as many times as you
     *       called Initialize().
     *
     * \warning It is of particular importance on _MS Windows_ platforms, when using "Run-Time Dynamic Linking",
     *          especially when unloading the DLL with FreeLibrary(). Otherwise it is very likely to run into the
     *          "Loader-Lock Deadlock Problem" during DLL unloading.
     *
     * \since 1.0
     * \since 2.0 Added reference counting
     *
     * \throws std::runtime_error when dynamic loading is active and and error occured during loading the dynamic library.
     */
    static void Close();
    //!\}

    /*!
     * \brief Returns the library version.
     *
     * \return Library version
     *
     * \since 1.0
     *
     * \throws std::runtime_error when dynamic loading is active and and error occured during loading the dynamic library.
     */
    static core::Version Version();

private:
    Library() = delete;
    ~Library() = delete;
    Library(const Library& other) = delete;
    Library& operator=(const Library& other) = delete;
    Library(Library&& other) = delete;
    Library& operator=(Library&& other) = delete;
};

} /* namespace peak */

#include <peak/backend/peak_backend.h>
#include <peak/dll_interface/peak_dll_interface_util.hpp>
#include <peak/peak_device_manager.hpp>


/* Implementation */
namespace peak
{

inline void Library::Initialize()
{
    CallAndCheckCInterfaceFunction([] { return PEAK_C_ABI_PREFIX PEAK_Library_Initialize(); });
}

inline void Library::Close()
{
    auto& deviceManager = DeviceManager::Instance();
    CallAndCheckCInterfaceFunction([] { return PEAK_C_ABI_PREFIX PEAK_Library_Close(); });

    if (QueryNumericFromCInterfaceFunction<PEAK_BOOL8>([](PEAK_BOOL8* isInitialized) {
            return PEAK_C_ABI_PREFIX PEAK_Library_IsInitialized(isInitialized);
        })
        == 0)
    {
        deviceManager.Reset(DeviceManager::ResetPolicy::IgnoreOpenDevices);
    }
}

inline core::Version Library::Version()
{
    uint32_t versionMajor = QueryNumericFromCInterfaceFunction<uint32_t>(
        PEAK_C_ABI_PREFIX PEAK_Library_GetVersionMajor);
    uint32_t versionMinor = QueryNumericFromCInterfaceFunction<uint32_t>(
        PEAK_C_ABI_PREFIX PEAK_Library_GetVersionMinor);
    uint32_t versionSubminor = QueryNumericFromCInterfaceFunction<uint32_t>(
        PEAK_C_ABI_PREFIX PEAK_Library_GetVersionSubminor);

    return core::Version(versionMajor, versionMinor, versionSubminor);
}

} /* namespace peak */
