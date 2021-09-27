/*!
 * \file    peak_device_manager.hpp
 *
 * \author  IDS Imaging Development Systems GmbH
 * \date    2019-05-01
 * \since   1.0
 *
 * Copyright (c) 2019, IDS Imaging Development Systems GmbH. All rights reserved.
 */

#pragma once

#include <peak/common/peak_timeout.hpp>
#include <peak/device/peak_device.hpp>
#include <peak/device/peak_device_descriptor.hpp>
#include <peak/environment/peak_environment_inspector.hpp>
#include <peak/exception/peak_exception.hpp>
#include <peak/generic/peak_t_callback_manager.hpp>
#include <peak/interface/peak_interface.hpp>
#include <peak/interface/peak_interface_descriptor.hpp>
#include <peak/producer_library/peak_producer_library.hpp>
#include <peak/system/peak_system.hpp>
#include <peak/system/peak_system_descriptor.hpp>

#include <type_traits>
#include <unordered_map>
#include <unordered_set>
#include <algorithm>
#include <atomic>
#include <exception>
#include <functional>
#include <future>
#include <iterator>
#include <memory>
#include <mutex>
#include <utility>
#include <vector>

namespace peak
{

/*!
 * \brief The global DeviceManager (singleton) searches all installed producer libraries (*.cti) and enumerates the
 * modules contained in them.
 *
 * Retrieve the global DeviceManager with DeviceManager::Instance().
 *
 * When Update() is called, it searches for all producer libraries contained in the directories found in the official
 * GenICam GenTL environment variable GENICAM_GENTL{32/64}_PATH. It then openes all found
 * \link core::ProducerLibrary ProducerLibraries\endlink, their \link core::System Systems\endlink, their
 * \link core::Interface Interfaces\endlink, and lists all available \link core::DeviceDescriptor DeviceDescriptors\endlink.
 *
 * \code
 * auto& deviceManager = peak::DeviceManager::Instance();
 * deviceManager.Update();
 * deviceDescriptors = deviceManager.Devices();
 * \endcode
 *
 * \note The found producer libraries depend on the architecture your application is compiled for. This means you are
 *       getting 32-bit producer libraries if your application is compiled for a 32-bit system and 64-bit producer
 *       libraries if your application is compiled for a 64-bit system.
 *
 * The DeviceManager is a singleton. Therefore, after it was used, it stays active until program termination. That
 * means the opened \link core::ProducerLibrary ProducerLibraries\endlink, \link core::System Systems\endlink, and
 * \link core::Interface Interfaces \endlink stay open. To close them before and start from scratch during runtime, call
 * Reset().
 */
class DeviceManager final
{
public:
    /*! \brief Enum holding the possible update policies. */
    enum class UpdatePolicy
    {
        ScanEnvironmentForProducerLibraries,
        DontScanEnvironmentForProducerLibraries
    };
    /*! \brief Enum holding the possible reset policies. */
    enum class ResetPolicy
    {
        ErrorOnOpenDevices,
        IgnoreOpenDevices
    };

    /*! The type of system found callbacks. */
    using SystemFoundCallback = std::function<void(const std::shared_ptr<const core::System>& foundSystem)>;
    /*! The type of system found callback handles. */
    using SystemFoundCallbackHandle = SystemFoundCallback*;
    /*! The type of interface found callbacks. */
    using InterfaceFoundCallback = std::function<void(const std::shared_ptr<const core::Interface>& foundInterface)>;
    /*! The type of interface found callback handles. */
    using InterfaceFoundCallbackHandle = InterfaceFoundCallback*;
    /*! The type of interface lost callbacks. */
    using InterfaceLostCallback = std::function<void(const std::string& lostInterfaceKey)>;
    /*! The type of interface lost callback handles. */
    using InterfaceLostCallbackHandle = InterfaceLostCallback*;
    /*! The type of device found callbacks. */
    using DeviceFoundCallback = std::function<void(const std::shared_ptr<core::DeviceDescriptor>& foundDevice)>;
    /*! The type of device found callback handles. */
    using DeviceFoundCallbackHandle = DeviceFoundCallback*;
    /*! The type of device lost callbacks. */
    using DeviceLostCallback = std::function<void(const std::string& lostDeviceKey)>;
    /*! The type of device lost callback handles. */
    using DeviceLostCallbackHandle = DeviceLostCallback*;
    /*! The type of update error callbacks. */
    using UpdateErrorCallback = std::function<void(const std::string& errorDescription)>;

    /*!
     * \brief Returns the global DeviceManager.
     *
     * \since 1.0
     */
    static DeviceManager& Instance();

    /*!
     * \brief Adds the given producer library (CTI).
     *
     * This function can be used to add producer libraries manually. This is useful when the desired producer
     * library is not registered at the GenTL environment variable.
     *
     * \param[in] ctiPath The path to the producer library (CTI) to add
     *
     * \since 1.0
     *
     * \throws core::InternalErrorException An internal error has occurred.
     *
     * \note Changes only take effect when they are
     *       applied before calling Update(). Changes applied during a running update only take effect with the next
     *       call to Update().
     */
    void AddProducerLibrary(const std::string& ctiPath);

    /*!
     * \brief Starts an update.
     *
     * If there is already an update in progress, it waits for the previous update to finish before starting the new
     * update.
     *
     * \param[in] updatePolicy The update policy to use.
     * \param[in] errorCallback A thread-safe callback getting called for each error occurring during the update.
     *
     * \since 1.0
     *
     * \throws core::NotInitializedException The library was not initialized before use.
     * \throws core::InternalErrorException An internal error has occurred.
     * \throws core::NotFoundException The environment variable GENICAM_GENTL32_PATH / GENICAM_GENTL64_PATH was not
     *                                 found or was empty when scanning for environment ProducerLibraries.
     */
    void Update(UpdatePolicy updatePolicy = UpdatePolicy::ScanEnvironmentForProducerLibraries,
        const UpdateErrorCallback& errorCallback = UpdateErrorCallback());
    /*!
     * \brief Resets the DeviceManager.
     *
     * All \link core::Interface Interfaces \endlink and \link core::System Systems \endlink are closed, additional
     * ProducerLibraries (added via AddProducerLibrary()) are cleared. Any registered callbacks stay active and the
     * DeviceLostCallback and InterfaceLostCallback are called for all devices / interfaces.
     *
     * \note Can't reset the DeviceManager while there are open \link core::Device Devices\endlink. Close all devices before
     *       calling this method. Otherwise, a core::InternalErrorException is thrown.
     *
     * \param[in] resetPolicy With the default ResetPolicy::ErrorOnOpenDevices, an exception is thrown if there are
     *            any open devices managed by the DeviceManager. With ResetPolicy::IgnoreOpenDevices, no such exception
     *            is thrown.
     *
     * \since 1.0
     *
     * \throws core::InternalErrorException If there are open devices (with ResetPolicy::ErrorOnOpenDevices).
     */
    void Reset(ResetPolicy resetPolicy = ResetPolicy::ErrorOnOpenDevices);
    /*!
     * \brief Returns the interface update timeout.
     *
     * \return Interface update timeout in milliseconds
     *
     * \since 1.0
     *
     * \throws core::InternalErrorException An internal error has occurred.
     */
    core::Timeout InterfaceUpdateTimeout() const;
    /*!
     * \brief Sets the interface update timeout.
     *
     * \param[in] timeout_ms The time to wait for new interfaces in milliseconds.
     *
     * \since 1.0
     *
     * \throws core::InternalErrorException An internal error has occurred.
     *
     * \note Changes only take effect when they are
     *       applied before calling Update(). Changes applied during a running update only take effect with the next
     *       call to Update().
     */
    void SetInterfaceUpdateTimeout(core::Timeout timeout_ms);
    /*!
     * \brief Returns the device update timeout.
     *
     * \return Device update timeout in milliseconds
     *
     * \since 1.0
     *
     * \throws core::InternalErrorException An internal error has occurred.
     */
    core::Timeout DeviceUpdateTimeout() const;
    /*!
     * \brief Sets the device update timeout.
     *
     * \param[in] timeout_ms The time to wait for new devices in milliseconds.
     *
     * \since 1.0
     *
     * \throws core::InternalErrorException An internal error has occurred.
     *
     * \note Changes only take effect when they are
     *       applied before calling Update(). Changes applied during a running update only take effect with the next
     *       call to Update().
     */
    void SetDeviceUpdateTimeout(core::Timeout timeout_ms);

    /*!
     * \brief Returns the found and opened systems.
     *
     * \return Found and opened systems
     *
     * \since 1.0
     *
     * \throws core::InternalErrorException An internal error has occurred.
     */
    std::vector<std::shared_ptr<const core::System>> Systems() const;
    /*!
     * \brief Returns the found and opened interfaces.
     *
     * \return Found and opened interfaces
     *
     * \since 1.0
     *
     * \throws core::InternalErrorException An internal error has occurred.
     */
    std::vector<std::shared_ptr<const core::Interface>> Interfaces() const;
    /*!
     * \brief Returns the found devices.
     *
     * \return Found devices
     *
     * \since 1.0
     *
     * \throws core::InternalErrorException An internal error has occurred.
     */
    std::vector<std::shared_ptr<core::DeviceDescriptor>> Devices() const;

    /*!
     * \brief Registers a callback for signaling a found system.
     *
     * This function registers a callback which gets called every time a new system is found. Pass the callback
     * handle returned by this function to UnregisterSystemFoundCallback() to unregister the callback.
     *
     * \param[in] callback The callback to call if a new system is found.
     *
     * \return Callback handle
     *
     * \since 1.0
     *
     * \throws core::InternalErrorException An internal error has occurred.
     *
     * \note The registered callback is executed in the thread calling Update().
     */
    SystemFoundCallbackHandle RegisterSystemFoundCallback(const SystemFoundCallback& callback);
    /*!
     * \brief Unregisters a system found callback.
     *
     * This function unregisters a system found callback by taking its handle.
     *
     * \param[in] callbackHandle The handle of the callback to unregister.
     *
     * \since 1.0
     *
     * \throws core::InternalErrorException An internal error has occurred.
     */
    void UnregisterSystemFoundCallback(SystemFoundCallbackHandle callbackHandle);
    /*!
     * \brief Registers a callback for signaling a found interface.
     *
     * This function registers a callback which gets called every time a new interface is found. Pass the callback
     * handle returned by this function to UnregisterInterfaceFoundCallback() to unregister the callback.
     *
     * \param[in] callback The callback to call if a new interface is found.
     *
     * \return Callback handle
     *
     * \since 1.0
     *
     * \throws core::InternalErrorException An internal error has occurred.
     *
     * \note The registered callback is executed in the thread calling Update().
     */
    InterfaceFoundCallbackHandle RegisterInterfaceFoundCallback(const InterfaceFoundCallback& callback);
    /*!
     * \brief Unregisters a interface found callback.
     *
     * This function unregisters a interface found callback by taking its handle.
     *
     * \param[in] callbackHandle The handle of the callback to unregister.
     *
     * \since 1.0
     *
     * \throws core::InternalErrorException An internal error has occurred.
     */
    void UnregisterInterfaceFoundCallback(InterfaceFoundCallbackHandle callbackHandle);
    /*!
     * \brief Registers a callback for signaling a lost interface.
     *
     * This function registers a callback which gets called every time an interface is lost. Pass the callback
     * handle returned by this function to UnregisterInterfaceLostCallback() to unregister the callback.
     *
     * \param[in] callback The callback to call if an interface is lost.
     *
     * \return Callback handle
     *
     * \since 1.0
     *
     * \throws core::InternalErrorException An internal error has occurred.
     *
     * \note The registered callback is executed in the thread calling Update().
     */
    InterfaceLostCallbackHandle RegisterInterfaceLostCallback(const InterfaceLostCallback& callback);
    /*!
     * \brief Unregisters an interface lost callback.
     *
     * This function unregisters an interface lost callback by taking its handle.
     *
     * \param[in] callbackHandle The handle of the callback to unregister.
     *
     * \since 1.0
     *
     * \throws core::InternalErrorException An internal error has occurred.
     */
    void UnregisterInterfaceLostCallback(InterfaceLostCallbackHandle callbackHandle);
    /*!
     * \brief Registers a callback for signaling a found device.
     *
     * This function registers a callback which gets called every time a new device is found. Pass the callback
     * handle returned by this function to UnregisterDeviceFoundCallback() to unregister the callback.
     *
     * \param[in] callback The callback to call if a new device is found.
     *
     * \return Callback handle
     *
     * \since 1.0
     *
     * \throws core::InternalErrorException An internal error has occurred.
     *
     * \note The registered callback is executed in the thread calling Update().
     */
    DeviceFoundCallbackHandle RegisterDeviceFoundCallback(const DeviceFoundCallback& callback);
    /*!
     * \brief Unregisters a device found callback.
     *
     * This function unregisters a device found callback by taking its handle.
     *
     * \param[in] callbackHandle The handle of the callback to unregister.
     *
     * \since 1.0
     *
     * \throws core::InternalErrorException An internal error has occurred.
     */
    void UnregisterDeviceFoundCallback(DeviceFoundCallbackHandle callbackHandle);
    /*!
     * \brief Registers a callback for signaling a lost device.
     *
     * This function registers a callback which gets called every time a device is lost. Pass the callback
     * handle returned by this function to UnregisterDeviceLostCallback() to unregister the callback.
     *
     * \param[in] callback The callback to call if a device is lost.
     *
     * \return Callback handle
     *
     * \since 1.0
     *
     * \throws core::InternalErrorException An internal error has occurred.
     *
     * \note The registered callback is executed in the thread calling Update().
     */
    DeviceLostCallbackHandle RegisterDeviceLostCallback(const DeviceLostCallback& callback);
    /*!
     * \brief Unregisters a device lost callback.
     *
     * This function unregisters a device lost callback by taking its handle.
     *
     * \param[in] callbackHandle The handle of the callback to unregister.
     *
     * \since 1.0
     *
     * \throws core::InternalErrorException An internal error has occurred.
     */
    void UnregisterDeviceLostCallback(DeviceLostCallbackHandle callbackHandle);

private:
    DeviceManager(); // private ctor, since this is a singleton
    ~DeviceManager() = default; // private dtor, since this is a singleton

    DeviceManager(const DeviceManager& other) = delete;
    DeviceManager& operator=(const DeviceManager& other) = delete;
    DeviceManager(DeviceManager&& other) = delete;
    DeviceManager& operator=(DeviceManager&& other) = delete;

    void GetLostDevicesFromLostInterfaces(
        const std::unordered_set<std::string>& lostInterfaces, std::vector<std::string>& lostDevices);

    void TriggerSystemFoundCallbacks(const std::shared_ptr<const core::System>& foundSystem) const;
    void TriggerInterfaceFoundCallbacks(const std::shared_ptr<const core::Interface>& foundInterface) const;
    void TriggerInterfaceLostCallbacks(const std::string& lostInterfaceKey) const;
    void TriggerDeviceFoundCallbacks(const std::shared_ptr<core::DeviceDescriptor>& foundDevice) const;
    void TriggerDeviceLostCallbacks(const std::string& lostDeviceKey) const;

    void CheckDevicesOpened() const;

    std::mutex m_updateMutex;
    std::atomic<core::Timeout> m_interfaceUpdateTimeout_ms{ 100 };
    std::atomic<core::Timeout> m_deviceUpdateTimeout_ms{ 300 };

    std::vector<std::string> m_producerLibrariesToAdd;

    std::unordered_map<std::string, std::shared_ptr<core::ProducerLibrary>> m_producerLibraries;
    std::vector<std::shared_ptr<const core::System>> m_systems;
    std::vector<std::shared_ptr<const core::Interface>> m_interfaces;
    std::vector<std::shared_ptr<core::DeviceDescriptor>> m_devices;

    struct OpenedDevice
    {
        std::weak_ptr<core::Device> device;
        std::string displayName;
    };
    std::vector<OpenedDevice> m_openedDevices;

    core::TTriggerCallbackManager<SystemFoundCallbackHandle, SystemFoundCallback> m_systemFoundCallbackManager;
    core::TTriggerCallbackManager<InterfaceFoundCallbackHandle, InterfaceFoundCallback> m_interfaceFoundCallbackManager;
    core::TTriggerCallbackManager<InterfaceLostCallbackHandle, InterfaceLostCallback> m_interfaceLostCallbackManager;
    core::TTriggerCallbackManager<DeviceFoundCallbackHandle, DeviceFoundCallback> m_deviceFoundCallbackManager;
    core::TTriggerCallbackManager<DeviceLostCallbackHandle, DeviceLostCallback> m_deviceLostCallbackManager;

    std::mutex m_producerLibrariesToAddMutex;
    mutable std::mutex m_systemsMutex;
    mutable std::mutex m_interfacesMutex;
    mutable std::mutex m_devicesMutex;
    mutable std::mutex m_openedDevicesMutex;

#ifndef _MSC_VER
    template <typename T,
        typename std::enable_if<(noexcept(std::declval<T>()())
                                    || (std::is_pointer<typename std::decay<T>::type>::value && __cplusplus < 201703L)
                                    // NOTE pre c++17 there is no way to check weather a function pointer is noexcept
                                    )
                && std::is_same<void, decltype(std::declval<T>()())>::value && std::is_move_constructible<T>::value
                && std::is_nothrow_move_constructible<T>::value,
            int>::type = 0>
#else
    template <typename T>
#endif
    class RAIIGuard
    {
        T m_fun;

    public:
        RAIIGuard(T&& fun) noexcept
            : m_fun(std::forward<T>(fun))
        {}
        RAIIGuard(RAIIGuard const&) = delete;
        RAIIGuard(RAIIGuard&& other) noexcept
            : m_fun(std::move(other.m_fun))
        {}
        ~RAIIGuard() noexcept
        {
            m_fun();
        }
    };
    template <typename T>
    static RAIIGuard<T> makeRAIIGuard(T&& fun) noexcept
    {
        return { std::forward<T>(fun) };
    }
};

} /* namespace peak */

/* Implementation */
namespace peak
{

inline DeviceManager& DeviceManager::Instance()
{
    static DeviceManager deviceManager{};
    return deviceManager;
}

inline DeviceManager::DeviceManager()
{
    if (QueryNumericFromCInterfaceFunction<PEAK_BOOL8>([](PEAK_BOOL8* isInitialized) {
            return PEAK_C_ABI_PREFIX PEAK_Library_IsInitialized(isInitialized);
        })
        == 0)
    {
        throw core::NotInitializedException(
            "IDS peak API library not initialized. Call peak::Library::Initialize() / "
            "PEAK_C_ABI_PREFIX PEAK_InitializeLibrary() before anything else.");
    }
}

inline void DeviceManager::AddProducerLibrary(const std::string& ctiPath)
{
    std::lock_guard<std::mutex> lock(m_producerLibrariesToAddMutex);

    m_producerLibrariesToAdd.emplace_back(ctiPath);
}

inline void DeviceManager::Update(UpdatePolicy updatePolicy /*= UpdatePolicy::ScanEnvironmentForProducerLibraries*/,
    const UpdateErrorCallback& errorCallback /*= UpdateErrorCallback()*/)
{
    std::lock_guard<std::mutex> lockUpdate(m_updateMutex);

    std::vector<std::shared_ptr<core::System>> allFoundSystems;
    std::vector<std::shared_ptr<core::Interface>> allFoundInterfaces;
    std::vector<std::string> allLostInterfaces;
    std::vector<std::shared_ptr<core::DeviceDescriptor>> allFoundDevices;
    std::vector<std::string> allLostDevices;

    auto interfaceUpdateTimeout_ms = m_interfaceUpdateTimeout_ms.load();
    auto deviceUpdateTimeout_ms = m_deviceUpdateTimeout_ms.load();

    std::vector<std::string> ctiPaths;
    // We look for all producer libraries in the current environment if we have to
    if (updatePolicy == UpdatePolicy::ScanEnvironmentForProducerLibraries)
    {
        ctiPaths = core::EnvironmentInspector::CollectCTIPaths();
    }
    // We look for the manually added producer libraries
    {
        std::lock_guard<std::mutex> lock(m_producerLibrariesToAddMutex);

        ctiPaths.reserve(ctiPaths.size() + m_producerLibrariesToAdd.size());
        std::move(
            std::begin(m_producerLibrariesToAdd), std::end(m_producerLibrariesToAdd), std::back_inserter(ctiPaths));

        m_producerLibrariesToAdd.clear();
    }
    for (const auto& ctiPath : ctiPaths)
    {
        if (!m_producerLibraries.count(ctiPath))
        {
            try
            {
                auto producerLibrary = core::ProducerLibrary::Open(ctiPath);
                auto system = producerLibrary->System()->OpenSystem();

                m_producerLibraries.emplace(ctiPath, producerLibrary);
                allFoundSystems.emplace_back(system);
            }
            catch (const core::NotInitializedException&)
            {
                // TODO: check this on creation of DeviceManager?
                throw; // if this exception occurs, nothing else will work. so we stop here
            }
            catch (const core::Exception& ex)
            {
                if (errorCallback)
                {
                    try
                    {
                        errorCallback("[CTI - " + ctiPath + "]: " + ex.what());
                    }
                    catch (...)
                    {}
                }
            }
        }
    }

    auto interfaceUpdateFunc = [&interfaceUpdateTimeout_ms, &errorCallback](const std::shared_ptr<core::System>& system)
        -> std::tuple<std::vector<std::shared_ptr<core::Interface>>, std::vector<std::string>> {
        std::vector<std::shared_ptr<core::Interface>> foundInterfaces;
        std::vector<std::string> lostInterfaces;

        try
        {
            auto foundCallbackHandle = system->RegisterInterfaceFoundCallback(
                [&foundInterfaces](const std::shared_ptr<core::InterfaceDescriptor>& foundInterface) {
                    foundInterfaces.emplace_back(foundInterface->OpenInterface());
                });
            auto const foundCallbackCleanupGuard = makeRAIIGuard([system, &foundCallbackHandle]() noexcept {
                try
                {
                    system->UnregisterInterfaceFoundCallback(foundCallbackHandle);
                }
                catch (const core::Exception&)
                {
                    std::terminate();
                }
            });

            auto keysById = std::unordered_map<std::string, std::string>();
            for (const auto& iface : system->Interfaces())
            {
                keysById.emplace(iface->ID(), iface->Key());
            }
            auto lostCallbackHandle = system->RegisterInterfaceLostCallback(
                [&lostInterfaces, &keysById](const std::string& lostInterfaceId) {
                    auto idKeyPairIt = keysById.find(lostInterfaceId);

                    if (idKeyPairIt != keysById.end())
                    {
                        lostInterfaces.emplace_back((*idKeyPairIt).second);
                    }
                });
            auto const lostCallbackCleanupGuard = makeRAIIGuard([system, &lostCallbackHandle]() noexcept {
                try
                {
                    system->UnregisterInterfaceLostCallback(lostCallbackHandle);
                }
                catch (const core::Exception&)
                {
                    std::terminate();
                }
            });

            system->UpdateInterfaces(interfaceUpdateTimeout_ms);
        }
        catch (const core::Exception& ex)
        {
            if (errorCallback)
            {
                try
                {
                    errorCallback("[System - " + system->Key() + "]: " + ex.what());
                }
                catch (...)
                {}
            }
        }

        return std::make_tuple(foundInterfaces, lostInterfaces);
    };

    // We do an update on all available systems
    const std::launch launchPolicy =
        std::launch::async; // switching this to std::launch::deferred makes debugging easier
    std::vector<std::future<std::tuple<std::vector<std::shared_ptr<core::Interface>>, std::vector<std::string>>>>
        interfaceUpdateFutures;
    for (const auto& system : m_systems)
    {
        interfaceUpdateFutures.emplace_back(
            std::async(launchPolicy, interfaceUpdateFunc, std::const_pointer_cast<core::System>(system)));
    }
    for (const auto& system : allFoundSystems)
    {
        interfaceUpdateFutures.emplace_back(std::async(launchPolicy, interfaceUpdateFunc, system));
    }
    for (auto& future : interfaceUpdateFutures)
    {
        auto foundLostInterfacesTuple = future.get();
        auto foundInterfaces = std::get<0>(foundLostInterfacesTuple);
        auto lostInterfaces = std::get<1>(foundLostInterfacesTuple);
        allFoundInterfaces.reserve(allFoundInterfaces.size() + foundInterfaces.size());
        std::move(std::begin(foundInterfaces), std::end(foundInterfaces), std::back_inserter(allFoundInterfaces));
        allLostInterfaces.reserve(allLostInterfaces.size() + lostInterfaces.size());
        std::move(std::begin(lostInterfaces), std::end(lostInterfaces), std::back_inserter(allLostInterfaces));
    }

    auto deviceUpdateFunc = [this, &deviceUpdateTimeout_ms, &errorCallback](
                                const std::shared_ptr<core::Interface>& iface)
        -> std::tuple<std::vector<std::shared_ptr<core::DeviceDescriptor>>, std::vector<std::string>> {
        std::vector<std::shared_ptr<core::DeviceDescriptor>> foundDevices;
        std::vector<std::string> lostDevices;

        try
        {
            auto foundCallbackHandle = iface->RegisterDeviceFoundCallback(
                [this, &foundDevices](const std::shared_ptr<core::DeviceDescriptor>& foundDevice) {
                    foundDevice->RegisterDeviceOpenedCallback(
                        [this](const std::shared_ptr<core::Device>& openedDevice) {
                            std::lock_guard<std::mutex> lock(m_openedDevicesMutex);

                            // clear out closed devices
                            m_openedDevices.erase(std::remove_if(std::begin(m_openedDevices), std::end(m_openedDevices),
                                                      [](const OpenedDevice& dev) { return dev.device.expired(); }),
                                std::end(m_openedDevices));

                            // add the new one
                            OpenedDevice device;
                            device.device = openedDevice;
                            device.displayName = openedDevice->DisplayName();
                            m_openedDevices.push_back(device);
                        });

                    foundDevices.emplace_back(foundDevice);
                });
            auto const foundCallbackCleanupGuard = makeRAIIGuard([iface, &foundCallbackHandle]() noexcept {
                try
                {
                    iface->UnregisterDeviceFoundCallback(foundCallbackHandle);
                }
                catch (const core::Exception&)
                {
                    std::terminate();
                }
            });

            auto keysById = std::unordered_map<std::string, std::string>();
            for (const auto& device : iface->Devices())
            {
                keysById.emplace(device->ID(), device->Key());
            }
            auto lostCallbackHandle = iface->RegisterDeviceLostCallback(
                [&lostDevices, &keysById](const std::string& lostDeviceId) {
                    auto idKeyPairIt = keysById.find(lostDeviceId);

                    if (idKeyPairIt != keysById.end())
                    {
                        lostDevices.emplace_back((*idKeyPairIt).second);
                    }
                });
            auto const lostCallbackCleanupGuard = makeRAIIGuard([iface, &lostCallbackHandle]() noexcept {
                try
                {
                    iface->UnregisterDeviceLostCallback(lostCallbackHandle);
                }
                catch (const core::Exception&)
                {
                    std::terminate();
                }
            });

            iface->UpdateDevices(deviceUpdateTimeout_ms);
        }
        catch (const core::Exception& ex)
        {
            if (errorCallback)
            {
                try
                {
                    errorCallback("[Interface - " + iface->Key() + "]: " + ex.what());
                }
                catch (...)
                {}
            }
        }

        return std::make_tuple(foundDevices, lostDevices);
    };

    // We do an update on all available interfaces
    std::vector<std::future<std::tuple<std::vector<std::shared_ptr<core::DeviceDescriptor>>, std::vector<std::string>>>>
        deviceUpdateFutures;
    for (const auto& iface : m_interfaces)
    {
        deviceUpdateFutures.emplace_back(
            std::async(launchPolicy, deviceUpdateFunc, std::const_pointer_cast<core::Interface>(iface)));
    }
    for (const auto& iface : allFoundInterfaces)
    {
        deviceUpdateFutures.emplace_back(std::async(launchPolicy, deviceUpdateFunc, iface));
    }
    for (auto& future : deviceUpdateFutures)
    {
        auto foundLostDevicesTuple = future.get();
        auto foundDevices = std::get<0>(foundLostDevicesTuple);
        auto lostDevices = std::get<1>(foundLostDevicesTuple);
        allFoundDevices.reserve(allFoundDevices.size() + foundDevices.size());
        std::move(std::begin(foundDevices), std::end(foundDevices), std::back_inserter(allFoundDevices));
        allLostDevices.reserve(allLostDevices.size() + lostDevices.size());
        std::move(std::begin(lostDevices), std::end(lostDevices), std::back_inserter(allLostDevices));
    }

    // We trigger all callbacks
    for (const auto& foundSystem : allFoundSystems)
    {
        TriggerSystemFoundCallbacks(foundSystem);
    }

    for (const auto& foundInterface : allFoundInterfaces)
    {
        TriggerInterfaceFoundCallbacks(foundInterface);
    }
    std::unordered_set<std::string> allLostInterfacesLookUp;
    for (const auto& lostInterfaceKey : allLostInterfaces)
    {
        TriggerInterfaceLostCallbacks(lostInterfaceKey);
        allLostInterfacesLookUp.emplace(lostInterfaceKey);
    }

    for (const auto& foundDevice : allFoundDevices)
    {
        TriggerDeviceFoundCallbacks(foundDevice);
    }
    GetLostDevicesFromLostInterfaces(allLostInterfacesLookUp, allLostDevices);
    for (const auto& lostDeviceKey : allLostDevices)
    {
        TriggerDeviceLostCallbacks(lostDeviceKey);
    }

    // We apply the update changes to our internal lists
    {
        std::lock_guard<std::mutex> lock(m_systemsMutex);
        std::lock_guard<std::mutex> lock2(m_interfacesMutex);
        std::lock_guard<std::mutex> lock3(m_devicesMutex);

        m_systems.reserve(m_systems.size() + allFoundSystems.size());
        std::move(std::begin(allFoundSystems), std::end(allFoundSystems), std::back_inserter(m_systems));

        for (const auto& lostInterfaceKey : allLostInterfaces)
        {
            m_interfaces.erase(std::find_if(std::begin(m_interfaces), std::end(m_interfaces),
                [&lostInterfaceKey](const std::shared_ptr<const core::Interface>& interfaceToCheck) -> bool {
                    return interfaceToCheck->Key() == lostInterfaceKey;
                }));
        }
        m_interfaces.reserve(m_interfaces.size() + allFoundInterfaces.size());
        std::move(std::begin(allFoundInterfaces), std::end(allFoundInterfaces), std::back_inserter(m_interfaces));

        for (const auto& lostDeviceKey : allLostDevices)
        {
            m_devices.erase(std::find_if(std::begin(m_devices), std::end(m_devices),
                [&lostDeviceKey](const std::shared_ptr<core::DeviceDescriptor>& deviceToCheck) -> bool {
                    return deviceToCheck->Key() == lostDeviceKey;
                }));
        }
        m_devices.reserve(m_devices.size() + allFoundDevices.size());
        std::move(std::begin(allFoundDevices), std::end(allFoundDevices), std::back_inserter(m_devices));
    }
}

inline void DeviceManager::Reset(ResetPolicy resetPolicy)
{
    std::lock_guard<std::mutex> lockUpdate(m_updateMutex);

    if (resetPolicy == ResetPolicy::ErrorOnOpenDevices)
    {
        // check if any devices are running
        CheckDevicesOpened();
    }

    {
        std::lock_guard<std::mutex> lock(m_systemsMutex);
        std::lock_guard<std::mutex> lock2(m_interfacesMutex);
        std::lock_guard<std::mutex> lock3(m_devicesMutex);

        for (const auto& lostDevice : m_devices)
        {
            TriggerDeviceLostCallbacks(lostDevice->Key());
        }
        m_devices.clear();

        for (const auto& lostInterface : m_interfaces)
        {
            TriggerInterfaceLostCallbacks(lostInterface->Key());
        }
        m_interfaces.clear();

        m_systems.clear();
        m_producerLibraries.clear();
    }

    {
        std::lock_guard<std::mutex> lock(m_producerLibrariesToAddMutex);
        m_producerLibrariesToAdd.clear();
    }
}

inline void DeviceManager::CheckDevicesOpened() const
{
    size_t openDevicesFound = 0;
    std::string deviceDisplayNames;

    {
        std::lock_guard<std::mutex> lock(m_openedDevicesMutex);

        for (const auto& openedDevice : m_openedDevices)
        {
            auto openDevice = openedDevice.device.lock();
            if (openDevice)
            {
                ++openDevicesFound;
                deviceDisplayNames += openedDevice.displayName;
                deviceDisplayNames += " ";
            }
        }
    }

    if (openDevicesFound > 0)
    {
        throw core::InternalErrorException(
            "Can't reset the device manager while there are devices open. The following devices ("
            + std::to_string(openDevicesFound) + ") are open: " + deviceDisplayNames);
    }
}

inline core::Timeout DeviceManager::InterfaceUpdateTimeout() const
{
    return m_interfaceUpdateTimeout_ms;
}

inline void DeviceManager::SetInterfaceUpdateTimeout(core::Timeout timeout_ms)
{
    m_interfaceUpdateTimeout_ms = timeout_ms;
}

inline core::Timeout DeviceManager::DeviceUpdateTimeout() const
{
    return m_deviceUpdateTimeout_ms;
}

inline void DeviceManager::SetDeviceUpdateTimeout(core::Timeout timeout_ms)
{
    m_deviceUpdateTimeout_ms = timeout_ms;
}

inline std::vector<std::shared_ptr<const core::System>> DeviceManager::Systems() const
{
    std::lock_guard<std::mutex> lock(m_systemsMutex);

    return m_systems;
}

inline std::vector<std::shared_ptr<const core::Interface>> DeviceManager::Interfaces() const
{
    std::lock_guard<std::mutex> lock(m_interfacesMutex);

    return m_interfaces;
}

inline std::vector<std::shared_ptr<core::DeviceDescriptor>> DeviceManager::Devices() const
{
    std::lock_guard<std::mutex> lock(m_devicesMutex);

    return m_devices;
}

inline DeviceManager::SystemFoundCallbackHandle DeviceManager::RegisterSystemFoundCallback(
    const DeviceManager::SystemFoundCallback& callback)
{
    return m_systemFoundCallbackManager.RegisterCallback(callback);
}

inline void DeviceManager::UnregisterSystemFoundCallback(DeviceManager::SystemFoundCallbackHandle callbackHandle)
{
    m_systemFoundCallbackManager.UnregisterCallback(callbackHandle);
}

inline DeviceManager::InterfaceFoundCallbackHandle DeviceManager::RegisterInterfaceFoundCallback(
    const DeviceManager::InterfaceFoundCallback& callback)
{
    return m_interfaceFoundCallbackManager.RegisterCallback(callback);
}

inline void DeviceManager::UnregisterInterfaceFoundCallback(DeviceManager::InterfaceFoundCallbackHandle callbackHandle)
{
    m_interfaceFoundCallbackManager.UnregisterCallback(callbackHandle);
}

inline DeviceManager::InterfaceLostCallbackHandle DeviceManager::RegisterInterfaceLostCallback(
    const DeviceManager::InterfaceLostCallback& callback)
{
    return m_interfaceLostCallbackManager.RegisterCallback(callback);
}

inline void DeviceManager::UnregisterInterfaceLostCallback(DeviceManager::InterfaceLostCallbackHandle callbackHandle)
{
    m_interfaceLostCallbackManager.UnregisterCallback(callbackHandle);
}

inline DeviceManager::DeviceFoundCallbackHandle DeviceManager::RegisterDeviceFoundCallback(
    const DeviceManager::DeviceFoundCallback& callback)
{
    return m_deviceFoundCallbackManager.RegisterCallback(callback);
}

inline void DeviceManager::UnregisterDeviceFoundCallback(DeviceManager::DeviceFoundCallbackHandle callbackHandle)
{
    m_deviceFoundCallbackManager.UnregisterCallback(callbackHandle);
}

inline DeviceManager::DeviceLostCallbackHandle DeviceManager::RegisterDeviceLostCallback(
    const DeviceManager::DeviceLostCallback& callback)
{
    return m_deviceLostCallbackManager.RegisterCallback(callback);
}

inline void DeviceManager::UnregisterDeviceLostCallback(DeviceManager::DeviceLostCallbackHandle callbackHandle)
{
    m_deviceLostCallbackManager.UnregisterCallback(callbackHandle);
}

inline void DeviceManager::GetLostDevicesFromLostInterfaces(
    const std::unordered_set<std::string>& lostInterfaces, std::vector<std::string>& lostDevices)
{
    std::lock_guard<std::mutex> lock(m_devicesMutex);

    for (const auto& device : m_devices)
    {
        if (lostInterfaces.count(device->ParentInterface()->Key()))
        {
            lostDevices.emplace_back(device->Key());
        }
    }
}

inline void DeviceManager::TriggerSystemFoundCallbacks(const std::shared_ptr<const core::System>& foundSystem) const
{
    m_systemFoundCallbackManager.TriggerCallbacks(foundSystem);
}

inline void DeviceManager::TriggerInterfaceFoundCallbacks(
    const std::shared_ptr<const core::Interface>& foundInterface) const
{
    m_interfaceFoundCallbackManager.TriggerCallbacks(foundInterface);
}

inline void DeviceManager::TriggerInterfaceLostCallbacks(const std::string& lostInterfaceKey) const
{
    m_interfaceLostCallbackManager.TriggerCallbacks(lostInterfaceKey);
}

inline void DeviceManager::TriggerDeviceFoundCallbacks(const std::shared_ptr<core::DeviceDescriptor>& foundDevice) const
{
    m_deviceFoundCallbackManager.TriggerCallbacks(foundDevice);
}

inline void DeviceManager::TriggerDeviceLostCallbacks(const std::string& lostDeviceKey) const
{
    m_deviceLostCallbackManager.TriggerCallbacks(lostDeviceKey);
}

} /* namespace peak */
