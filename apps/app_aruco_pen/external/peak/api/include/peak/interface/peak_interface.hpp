/*!
 * \file    peak_interface.hpp
 *
 * \author  IDS Imaging Development Systems GmbH
 * \date    2019-05-01
 * \since   1.0
 *
 * Copyright (c) 2019, IDS Imaging Development Systems GmbH. All rights reserved.
 */

#pragma once


#include <peak/backend/peak_backend.h>
#include <peak/common/peak_common_structs.hpp>
#include <peak/common/peak_event_supporting_module.hpp>
#include <peak/device/peak_device_descriptor.hpp>
#include <peak/dll_interface/peak_dll_interface_util.hpp>
#include <peak/generic/peak_t_callback_manager.hpp>

#include <unordered_map>
#include <cstdint>
#include <functional>
#include <memory>
#include <mutex>
#include <string>
#include <vector>


namespace peak
{
namespace core
{

class System;

struct DeviceFoundCallbackContainer;

/*!
 * \brief Represents a GenTL Interface module.
 *
 * This class allows to query information about a GenTL Interface module and to enumerate its
 * \link DeviceDescriptor DeviceDescriptors\endlink, which allow you to open the corresponding Device.
 */
class Interface
    : public EventSupportingModule
    , public std::enable_shared_from_this<Interface>
{
public:
    /*! The type of device found callbacks. */
    using DeviceFoundCallback = std::function<void(const std::shared_ptr<DeviceDescriptor>& foundDevice)>;
    /*! The type of device found callback handles. */
    using DeviceFoundCallbackHandle = DeviceFoundCallback*;
    /*! The type of device lost callbacks. */
    using DeviceLostCallback = std::function<void(const std::string& lostDeviceId)>;
    /*! The type of device lost callback handles. */
    using DeviceLostCallbackHandle = DeviceLostCallback*;

    Interface() = delete;
    ~Interface() override;
    Interface(const Interface& other) = delete;
    Interface& operator=(const Interface& other) = delete;
    Interface(Interface&& other) = delete;
    Interface& operator=(Interface&& other) = delete;

    /*! @copydoc ProducerLibrary::Key() */
    std::string Key() const;

    /*! @copydoc SystemDescriptor::Info() */
    RawInformation Info(int32_t infoCommand) const;
    /*! @copydoc System::ID() */
    std::string ID() const;
    /*! @copydoc InterfaceDescriptor::DisplayName() */
    std::string DisplayName() const;
    /*! @copydoc SystemDescriptor::TLType() */
    std::string TLType() const;

    /*!
     * \brief Searches for devices.
     *
     * This function triggers an update of the internal device list. The callbacks registered on the interface
     * will be triggered if an device is found or lost.
     *
     * \param[in] timeout_ms The time to wait for new devices in milliseconds. In any case the
     *                       GenTL Producer must make sure that this operation is completed in a
     *                       reasonable amount of time depending on the underlying technology.
     *
     * \since 1.0
     *
     * \throws InternalErrorException An internal error has occurred.
     */
    void UpdateDevices(Timeout timeout_ms);
    /*!
     * \brief Returns the device list.
     *
     * \return Device list
     *
     * \since 1.0
     *
     * \throws InternalErrorException An internal error has occurred.
     */
    std::vector<std::shared_ptr<DeviceDescriptor>> Devices() const;
    /*! @copydoc InterfaceDescriptor::ParentSystem() */
    std::shared_ptr<System> ParentSystem() const;

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
     * \throws InternalErrorException An internal error has occurred.
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
     * \throws InternalErrorException An internal error has occurred.
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
     * \throws InternalErrorException An internal error has occurred.
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
     * \throws InternalErrorException An internal error has occurred.
     */
    void UnregisterDeviceLostCallback(DeviceLostCallbackHandle callbackHandle);

private:
    struct DeviceFoundCallbackContainer
    {
        std::weak_ptr<Interface> _Interface;
        Interface::DeviceFoundCallback Callback;
    };
    static void PEAK_CALL_CONV DeviceFoundCallbackCWrapper(
        PEAK_DEVICE_DESCRIPTOR_HANDLE foundDevice, void* context);
    static void PEAK_CALL_CONV DeviceLostCallbackCWrapper(
        const char* lostDeviceId, size_t lostDeviceIdSize, void* context);
    std::shared_ptr<DeviceDescriptor> GetOrAddFoundDevice(PEAK_DEVICE_DESCRIPTOR_HANDLE foundDevice);

    PEAK_MODULE_HANDLE ModuleHandle() const override;
    PEAK_EVENT_SUPPORTING_MODULE_HANDLE EventSupportingModuleHandle() const override;

    void InitializeUpdateMechanismIfNecessary();

    friend ClassCreator<Interface>;
    Interface(PEAK_INTERFACE_HANDLE interfaceHandle, const std::weak_ptr<System>& parentSystem);
    PEAK_INTERFACE_HANDLE m_backendHandle;

    std::weak_ptr<System> m_parentSystem;

    std::unique_ptr<TCallbackManager<PEAK_DEVICE_FOUND_CALLBACK_HANDLE, DeviceFoundCallbackContainer>>
        m_deviceFoundCallbackManager;
    std::unique_ptr<TCallbackManager<PEAK_DEVICE_LOST_CALLBACK_HANDLE, DeviceLostCallback>>
        m_deviceLostCallbackManager;

    std::vector<std::shared_ptr<DeviceDescriptor>> m_devices;
    std::unordered_map<std::string, std::shared_ptr<DeviceDescriptor>> m_devicesByKey;
    std::unordered_map<std::string, std::string> m_devicesKeyById;
    mutable std::mutex m_devicesMutex;
    std::once_flag m_updateMechanismInitializedFlag;

    std::string m_key;
};

} /* namespace core */
} /* namespace peak */

/* Implementation */
namespace peak
{
namespace core
{

inline Interface::Interface(PEAK_INTERFACE_HANDLE interfaceHandle, const std::weak_ptr<System>& parentSystem)
    : m_backendHandle(interfaceHandle)
    , m_parentSystem(parentSystem)
    , m_deviceFoundCallbackManager()
    , m_deviceLostCallbackManager()
    , m_key(QueryStringFromCInterfaceFunction([&](char* key, size_t* keySize) {
        return PEAK_C_ABI_PREFIX PEAK_Interface_GetKey(interfaceHandle, key, keySize);
    }))
{
    m_deviceFoundCallbackManager =
        std::make_unique<TCallbackManager<PEAK_DEVICE_FOUND_CALLBACK_HANDLE, DeviceFoundCallbackContainer>>(
            [&](void* callbackContext) {
                return QueryNumericFromCInterfaceFunction<PEAK_DEVICE_FOUND_CALLBACK_HANDLE>(
                    [&](PEAK_DEVICE_FOUND_CALLBACK_HANDLE* deviceFoundCallbackHandle) {
                        return PEAK_C_ABI_PREFIX PEAK_Interface_RegisterDeviceFoundCallback(
                            m_backendHandle, DeviceFoundCallbackCWrapper, callbackContext, deviceFoundCallbackHandle);
                    });
            },
            [&](PEAK_DEVICE_FOUND_CALLBACK_HANDLE callbackHandle) {
                CallAndCheckCInterfaceFunction([&] {
                    return PEAK_C_ABI_PREFIX PEAK_Interface_UnregisterDeviceFoundCallback(
                        m_backendHandle, callbackHandle);
                });
            });

    m_deviceLostCallbackManager =
        std::make_unique<TCallbackManager<PEAK_DEVICE_LOST_CALLBACK_HANDLE, DeviceLostCallback>>(
            [&](void* callbackContext) {
                return QueryNumericFromCInterfaceFunction<PEAK_DEVICE_LOST_CALLBACK_HANDLE>(
                    [&](PEAK_DEVICE_LOST_CALLBACK_HANDLE* deviceLostCallbackHandle) {
                        return PEAK_C_ABI_PREFIX PEAK_Interface_RegisterDeviceLostCallback(
                            m_backendHandle, DeviceLostCallbackCWrapper, callbackContext, deviceLostCallbackHandle);
                    });
            },
            [&](PEAK_DEVICE_LOST_CALLBACK_HANDLE callbackHandle) {
                CallAndCheckCInterfaceFunction([&] {
                    return PEAK_C_ABI_PREFIX PEAK_Interface_UnregisterDeviceLostCallback(
                        m_backendHandle, callbackHandle);
                });
            });
}

inline Interface::~Interface()
{
    try
    {
        m_deviceFoundCallbackManager->UnregisterAllCallbacks();
        m_deviceLostCallbackManager->UnregisterAllCallbacks();
    }
    catch (const Exception&)
    {}

    (void)PEAK_C_ABI_PREFIX PEAK_Interface_Destruct(m_backendHandle);
}

inline std::string Interface::Key() const
{
    return m_key;
}

inline RawInformation Interface::Info(int32_t infoCommand) const
{
    return QueryRawInformationFromCInterfaceFunction([&](int32_t* dataType, uint8_t* buffer, size_t* bufferSize) {
        return PEAK_C_ABI_PREFIX PEAK_Interface_GetInfo(
            m_backendHandle, infoCommand, dataType, buffer, bufferSize);
    });
}

inline std::string Interface::ID() const
{
    return QueryStringFromCInterfaceFunction([&](char* id, size_t* idSize) {
        return PEAK_C_ABI_PREFIX PEAK_Interface_GetID(m_backendHandle, id, idSize);
    });
}

inline std::string Interface::DisplayName() const
{
    return QueryStringFromCInterfaceFunction([&](char* displayName, size_t* displayNameSize) {
        return PEAK_C_ABI_PREFIX PEAK_Interface_GetDisplayName(m_backendHandle, displayName, displayNameSize);
    });
}

inline std::string Interface::TLType() const
{
    return QueryStringFromCInterfaceFunction([&](char* tlType, size_t* tlTypeSize) {
        return PEAK_C_ABI_PREFIX PEAK_Interface_GetTLType(m_backendHandle, tlType, tlTypeSize);
    });
}

inline void Interface::UpdateDevices(Timeout timeout_ms)
{
    InitializeUpdateMechanismIfNecessary();

    CallAndCheckCInterfaceFunction(
        [&] { return PEAK_C_ABI_PREFIX PEAK_Interface_UpdateDevices(m_backendHandle, timeout_ms); });
}

inline std::vector<std::shared_ptr<DeviceDescriptor>> Interface::Devices() const
{
    std::lock_guard<std::mutex> lock(m_devicesMutex);
    return m_devices;
}

inline std::shared_ptr<System> Interface::ParentSystem() const
{
    return LockOrThrow(m_parentSystem);
}

inline Interface::DeviceFoundCallbackHandle Interface::RegisterDeviceFoundCallback(
    const Interface::DeviceFoundCallback& callback)
{
    return reinterpret_cast<DeviceFoundCallbackHandle>(
        m_deviceFoundCallbackManager->RegisterCallback(DeviceFoundCallbackContainer{ shared_from_this(), callback }));
}

inline void Interface::UnregisterDeviceFoundCallback(Interface::DeviceFoundCallbackHandle callbackHandle)
{
    m_deviceFoundCallbackManager->UnregisterCallback(
        reinterpret_cast<PEAK_DEVICE_FOUND_CALLBACK_HANDLE>(callbackHandle));
}

inline Interface::DeviceLostCallbackHandle Interface::RegisterDeviceLostCallback(
    const Interface::DeviceLostCallback& callback)
{
    return reinterpret_cast<DeviceLostCallbackHandle>(m_deviceLostCallbackManager->RegisterCallback(callback));
}

inline void Interface::UnregisterDeviceLostCallback(Interface::DeviceLostCallbackHandle callbackHandle)
{
    m_deviceLostCallbackManager->UnregisterCallback(
        reinterpret_cast<PEAK_DEVICE_LOST_CALLBACK_HANDLE>(callbackHandle));
}

inline void PEAK_CALL_CONV Interface::DeviceFoundCallbackCWrapper(
    PEAK_DEVICE_DESCRIPTOR_HANDLE foundDevice, void* context)
{
    auto callbackContainer = static_cast<DeviceFoundCallbackContainer*>(context);

    auto interfaceDescriptor = callbackContainer->_Interface.lock()->GetOrAddFoundDevice(foundDevice);

    callbackContainer->Callback(interfaceDescriptor);
}

inline void PEAK_CALL_CONV Interface::DeviceLostCallbackCWrapper(
    const char* lostDeviceId, size_t lostDeviceIdSize, void* context)
{
    auto callback = static_cast<Interface::DeviceLostCallback*>(context);

    callback->operator()(std::string(lostDeviceId, lostDeviceIdSize - 1));
}

inline std::shared_ptr<DeviceDescriptor> Interface::GetOrAddFoundDevice(PEAK_DEVICE_DESCRIPTOR_HANDLE foundDevice)
{
    std::lock_guard<std::mutex> lock(m_devicesMutex);

    const auto deviceKey = QueryStringFromCInterfaceFunction([&](char* key, size_t* keySize) {
        return PEAK_C_ABI_PREFIX PEAK_DeviceDescriptor_GetKey(foundDevice, key, keySize);
    });
    auto keyDevicePairIterator = m_devicesByKey.find(deviceKey);
    if (keyDevicePairIterator == m_devicesByKey.end())
    {
        auto deviceDescriptor = std::static_pointer_cast<DeviceDescriptor>(
            std::make_shared<ClassCreator<DeviceDescriptor>>(
                foundDevice, std::weak_ptr<Interface>{ shared_from_this() }));
        m_devices.emplace_back(deviceDescriptor);
        m_devicesByKey.emplace(deviceKey, deviceDescriptor);
        m_devicesKeyById.emplace(deviceDescriptor->ID(), deviceKey);

        return deviceDescriptor;
    }

    return keyDevicePairIterator->second;
}

inline PEAK_MODULE_HANDLE Interface::ModuleHandle() const
{
    auto moduleHandle = QueryNumericFromCInterfaceFunction<PEAK_MODULE_HANDLE>(
        [&](PEAK_MODULE_HANDLE* _moduleHandle) {
            return PEAK_C_ABI_PREFIX PEAK_Interface_ToModule(m_backendHandle, _moduleHandle);
        });

    return moduleHandle;
}

inline PEAK_EVENT_SUPPORTING_MODULE_HANDLE Interface::EventSupportingModuleHandle() const
{
    auto eventSupportingModuleHandle = QueryNumericFromCInterfaceFunction<PEAK_EVENT_SUPPORTING_MODULE_HANDLE>(
        [&](PEAK_EVENT_SUPPORTING_MODULE_HANDLE* _eventSupportingModuleHandle) {
            return PEAK_C_ABI_PREFIX PEAK_Interface_ToEventSupportingModule(
                m_backendHandle, _eventSupportingModuleHandle);
        });

    return eventSupportingModuleHandle;
}

inline void Interface::InitializeUpdateMechanismIfNecessary()
{
    std::call_once(m_updateMechanismInitializedFlag, [&] {
        (void)RegisterDeviceFoundCallback([](const std::shared_ptr<DeviceDescriptor>&) {
            // Registering an empty callback is enough since DeviceFoundCallbackCWrapper() handles everything else.
        });
        (void)RegisterDeviceLostCallback([&](const std::string& lostDeviceId) {
            std::lock_guard<std::mutex> lock(m_devicesMutex);
            const auto lostDeviceKey = m_devicesKeyById.at(lostDeviceId);
            m_devices.erase(std::remove_if(std::begin(m_devices), std::end(m_devices),
                                [lostDeviceKey](const std::shared_ptr<DeviceDescriptor>& device) {
                                    return device->Key() == lostDeviceKey;
                                }),
                std::end(m_devices));
            m_devicesByKey.erase(lostDeviceKey);
            m_devicesKeyById.erase(lostDeviceId);
        });
    });
}

} /* namespace core */
} /* namespace peak */
