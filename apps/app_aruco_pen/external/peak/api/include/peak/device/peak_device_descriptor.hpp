/*!
 * \file    peak_device_descriptor.hpp
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
#include <peak/common/peak_module_descriptor.hpp>
#include <peak/device/peak_common_device_enums.hpp>
#include <peak/device/peak_device.hpp>
#include <peak/dll_interface/peak_dll_interface_util.hpp>
#include <peak/generic/peak_t_callback_manager.hpp>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>


namespace peak
{
namespace core
{

/*!
 * \brief Different modes to open the device with.
 *
 * See GenTL DEVICE_ACCESS_FLAGS.
 */
enum class DeviceAccessType
{
    ReadOnly = 2,
    Control,
    Exclusive,

    Custom = 1000
};

/*!
 * \brief Different types of information about the device.
 *
 * See GenTL DEVICE_INFO_CMD.
 */
enum class DeviceInformationRole
{
    ID = 0,
    VendorName,
    ModelName,
    TLType,
    DisplayName,
    AccessStatus,
    UserDefinedName,
    SerialNumber,
    Version,
    TimestampTickFrequency,

    Custom = 1000
};

class Interface;

/*!
 * \brief Encapsulates the GenTL functions associated with a GenTL Device module's ID.
 *
 * This class allows to query information about a GenTL Device module without opening it. Furthermore, it enables
 * you to open this GenTL Device module.
 *
 * ### Opening a Device
 *
 * When you try to open a device with an access mode that is not allowed, a BadAccessException is thrown. To avoid
 * that, ask the device if the desired access mode is available with IsOpenable(). By default, IsOpenable
 * asks for the highest access mode, DeviceAccessType::Exclusive:
 *
 * \code
 * if (deviceDescriptor->IsOpenable())
 * {
 *     device = deviceDescriptor->OpenDevice(DeviceAccessType::Control);
 * }
 * \endcode
 *
 * ### %Device Information Monitoring
 *
 * With [this group of methods](@ref DeviceInformationMonitoring), you can conveniently monitor for changes of
 * information about the device.
 *
 * Register a callback using RegisterInformationChangedCallback() and configure which information you want to
 * monitor with AddInformationRoleToMonitoring() and RemoveInformationRoleFromMonitoring().
 *
 * By default, the following DeviceInformationRoles are monitored:
 * * AccessStatus
 * * UserDefinedName
 * * TimestampTickFrequency
 */
class DeviceDescriptor : public ModuleDescriptor
{
public:
    /*! The type of changed callbacks. */
    using InformationChangedCallback = std::function<void(const std::vector<DeviceInformationRole>& changedRoles)>;
    /*! The type of changed callback handles. */
    using InformationChangedCallbackHandle = InformationChangedCallback*;
    /*! The type of device opened callbacks. */
    using DeviceOpenedCallback = std::function<void(const std::shared_ptr<core::Device>& openedDevice)>;
    /*! The type of device opened callback handles. */
    using DeviceOpenedCallbackHandle = DeviceOpenedCallback*;

    DeviceDescriptor() = delete;
    ~DeviceDescriptor() override;
    DeviceDescriptor(const DeviceDescriptor& other) = delete;
    DeviceDescriptor& operator=(const DeviceDescriptor& other) = delete;
    DeviceDescriptor(DeviceDescriptor&& other) = delete;
    DeviceDescriptor& operator=(DeviceDescriptor&& other) = delete;

    /*! @copydoc ProducerLibrary::Key() */
    std::string Key() const;

    /*! @copydoc SystemDescriptor::Info() */
    RawInformation Info(int32_t infoCommand) const;
    /*!
     * \brief Returns the display name.
     *
     * \return Display name
     *
     * \since 1.0
     *
     * \throws NotFoundException Device could not be found.
     * \throws InternalErrorException An internal error has occurred.
     */
    std::string DisplayName() const;
    /*!
     * \brief Returns the vendor name.
     *
     * \return Vendor name
     *
     * \since 1.0
     *
     * \throws NotFoundException Device could not be found.
     * \throws InternalErrorException An internal error has occurred.
     */
    std::string VendorName() const;
    /*!
     * \brief Returns the model name.
     *
     * \return Model name
     *
     * \since 1.0
     *
     * \throws NotFoundException Device could not be found.
     * \throws InternalErrorException An internal error has occurred.
     */
    std::string ModelName() const;
    /*!
     * \brief Returns the version.
     *
     * \return Version
     *
     * \since 1.0
     *
     * \throws NotFoundException Device could not be found.
     * \throws InternalErrorException An internal error has occurred.
     */
    std::string Version() const;
    /*! @copydoc SystemDescriptor::TLType() */
    std::string TLType() const;
    /*!
     * \brief Returns the user defined name.
     *
     * \return User defined name
     *
     * \since 1.0
     *
     * \throws NotFoundException Device could not be found.
     * \throws InternalErrorException An internal error has occurred.
     */
    std::string UserDefinedName() const;
    /*!
     * \brief Returns the serial number.
     *
     * \return Serial number
     *
     * \since 1.0
     *
     * \throws NotFoundException Device could not be found.
     * \throws InternalErrorException An internal error has occurred.
     */
    std::string SerialNumber() const;
    /*!
     * \brief Returns the access status.
     *
     * \return Access status
     *
     * \since 1.0
     *
     * \throws NotFoundException Device could not be found.
     * \throws InternalErrorException An internal error has occurred.
     */
    DeviceAccessStatus AccessStatus() const;
    /*!
     * \brief Returns the timestamp tick frequency.
     *
     * \return Timestamp tick frequency in Hz
     *
     * \since 1.0
     *
     * \throws NotFoundException Device could not be found.
     * \throws InternalErrorException An internal error has occurred.
     */
    uint64_t TimestampTickFrequency() const;

    /*!
     * \brief Returns the parent interface.
     *
     * \return Parent interface
     *
     * \since 1.0
     *
     * \throws NotFoundException Device could not be found.
     * \throws InternalErrorException An internal error has occurred.
     */
    std::shared_ptr<Interface> ParentInterface() const;

    /*!
     * \anchor DeviceOpen
     * \name Open the device
     */
    ///\{
    /*!
     * \brief Checks whether the device can be opened with a specific access type.
     *
     * If the device can be opened with a higher access type, it can also be opened with a lower access type.
     *
     * \code
     * if (deviceDescriptor->IsOpenable())
     * {
     *     device = deviceDescriptor->OpenDevice(DeviceAccessType::Control);
     * }
     * \endcode
     *
     * \param[in] accessType The access type to check.
     *
     * \return True, if the device can be opened with the given access type.
     * \return False otherwise.
     *
     * \since 1.0
     *
     * \throws InternalErrorException An internal error has occurred.
     */
    bool IsOpenable(DeviceAccessType accessType = DeviceAccessType::Exclusive) const;
    /*!
     * \brief Opens the Device.
     *
     * \param[in] accessType The access type the device should be opened for.
     *
     * \return Opened Device
     *
     * \since 1.0
     *
     * \throws NotFoundException Device could not be found.
     * \throws BadAccessException Access denied
     * \throws InternalErrorException An internal error has occurred.
     */
    std::shared_ptr<Device> OpenDevice(DeviceAccessType accessType);
    /*!
     * \brief Returns the Device that was opened with this DeviceDescriptor.
     *
     * \return Opened Device
     *
     * \since 1.0
     *
     * \throws BadAccessException Device is not open
     */
    std::shared_ptr<Device> OpenedDevice() const;
    /*!
     * \brief Registers a callback for signaling when the Device is opened with this DeviceDescriptor.
     *
     * Pass the callback handle returned by this function to UnregisterDeviceOpenedCallback() to unregister the
     * callback.
     *
     * \param[in] callback The callback to call when the Device is opened.
     *
     * \return Callback handle
     *
     * \since 1.0
     *
     * \throws InternalErrorException An internal error has occurred.
     */
    DeviceOpenedCallbackHandle RegisterDeviceOpenedCallback(const DeviceOpenedCallback& callback);
    /*!
     * \brief Unregisters a device opened callback.
     *
     * This function unregisters a device opened callback by taking its handle.
     *
     * \param[in] callbackHandle The handle of the callback to unregister.
     *
     * \since 1.0
     *
     * \throws InternalErrorException An internal error has occurred.
     */
    void UnregisterDeviceOpenedCallback(DeviceOpenedCallbackHandle callbackHandle);
    ///\}

    /*!
     * \anchor DeviceInformationMonitoring
     * \name Device Information Monitoring
     */
    ///\{
    /*!
     * \brief Returns the monitoring update interval.
     *
     * \return Monitoring update interval in milliseconds
     *
     * \since 1.0
     *
     * \throws InternalErrorException An internal error has occurred.
     */
    uint64_t MonitoringUpdateInterval() const;
    /*!
     * \brief Sets the monitoring update interval.
     *
     * \param[in] monitoringUpdateInterval_ms The monitoring update interval to set in milliseconds.
     *
     * \since 1.0
     *
     * \throws InternalErrorException An internal error has occurred.
     */
    void SetMonitoringUpdateInterval(uint64_t monitoringUpdateInterval_ms);
    /*!
     * \brief Checks whether the given information role is monitored.
     *
     * \param[in] informationRole The information role to check.
     *
     * \return True, if the given information role is monitored.
     * \return False otherwise.
     *
     * \since 1.0
     *
     * \throws InternalErrorException An internal error has occurred.
     */
    bool IsInformationRoleMonitored(DeviceInformationRole informationRole) const;
    /*!
     * \brief Adds the given information role to the monitoring.
     *
     * \param[in] informationRole The information role to add to the monitoring.
     *
     * \since 1.0
     *
     * \throws InternalErrorException An internal error has occurred.
     */
    void AddInformationRoleToMonitoring(DeviceInformationRole informationRole);
    /*!
     * \brief Removes the given information role from the monitoring.
     *
     * \param[in] informationRole The information role to remove from the monitoring.
     *
     * \since 1.0
     *
     * \throws InternalErrorException An internal error has occurred.
     */
    void RemoveInformationRoleFromMonitoring(DeviceInformationRole informationRole);
    /*!
     * \brief Registers a callback for signaling changed information.
     *
     * This function registers a callback which gets called every time information have changed. Pass the callback
     * handle returned by this function to UnregisterInformationChangedCallback() to unregister the callback.
     *
     * \param[in] callback The callback to call if information have changed.
     *
     * \return Callback handle
     *
     * \since 1.0
     *
     * \throws InternalErrorException An internal error has occurred.
     */
    InformationChangedCallbackHandle RegisterInformationChangedCallback(const InformationChangedCallback& callback);
    /*!
     * \brief Unregisters an information changed callback.
     *
     * This function unregisters an information changed callback by taking its handle.
     *
     * \param[in] callbackHandle The handle of the callback to unregister.
     *
     * \since 1.0
     *
     * \throws InternalErrorException An internal error has occurred.
     */
    void UnregisterInformationChangedCallback(InformationChangedCallbackHandle callbackHandle);
    ///\}

private:
    static void PEAK_CALL_CONV InformationChangedCallbackCWrapper(
        const PEAK_DEVICE_INFORMATION_ROLE* changedRoles, size_t changedRolesSize, void* context);

    friend ClassCreator<DeviceDescriptor>;
    DeviceDescriptor(
        PEAK_DEVICE_DESCRIPTOR_HANDLE deviceDescriptorHandle, const std::weak_ptr<Interface>& parentInterface);
    PEAK_MODULE_DESCRIPTOR_HANDLE ModuleDescriptorHandle() const override;

    PEAK_DEVICE_DESCRIPTOR_HANDLE m_backendHandle;

    std::weak_ptr<Interface> m_parentInterface;
    std::weak_ptr<Device> m_openedDevice;

    std::unique_ptr<
        TCallbackManager<PEAK_DEVICE_DESCRIPTOR_INFORMATION_CHANGED_CALLBACK_HANDLE, InformationChangedCallback>>
        m_informationChangedCallbackManager;

    TTriggerCallbackManager<DeviceOpenedCallbackHandle, DeviceOpenedCallback> m_deviceOpenedCallbackManager;

    std::string m_key;

    friend class FirmwareUpdater;
};

} /* namespace core */
} /* namespace peak */

/* Implementation */
namespace peak
{
namespace core
{

inline std::string ToString(DeviceAccessType entry)
{
    std::string entryString;

    if (entry == DeviceAccessType::ReadOnly)
    {
        entryString = "ReadOnly";
    }
    else if (entry == DeviceAccessType::Control)
    {
        entryString = "Control";
    }
    else if (entry == DeviceAccessType::Exclusive)
    {
        entryString = "Exclusive";
    }
    else if (entry >= DeviceAccessType::Custom)
    {
        entryString = "Custom";
    }

    return entryString;
}

inline std::string ToString(DeviceInformationRole entry)
{
    std::string entryString;

    if (entry == DeviceInformationRole::ID)
    {
        entryString = "ID";
    }
    else if (entry == DeviceInformationRole::VendorName)
    {
        entryString = "VendorName";
    }
    else if (entry == DeviceInformationRole::ModelName)
    {
        entryString = "ModelName";
    }
    else if (entry == DeviceInformationRole::TLType)
    {
        entryString = "TLType";
    }
    else if (entry == DeviceInformationRole::DisplayName)
    {
        entryString = "DisplayName";
    }
    else if (entry == DeviceInformationRole::AccessStatus)
    {
        entryString = "AccessStatus";
    }
    else if (entry == DeviceInformationRole::UserDefinedName)
    {
        entryString = "UserDefinedName";
    }
    else if (entry == DeviceInformationRole::SerialNumber)
    {
        entryString = "SerialNumber";
    }
    else if (entry == DeviceInformationRole::Version)
    {
        entryString = "Version";
    }
    else if (entry == DeviceInformationRole::TimestampTickFrequency)
    {
        entryString = "TimestampTickFrequency";
    }
    else if (entry >= DeviceInformationRole::Custom)
    {
        entryString = "Custom";
    }

    return entryString;
}

inline DeviceDescriptor::DeviceDescriptor(
    PEAK_DEVICE_DESCRIPTOR_HANDLE deviceDescriptorHandle, const std::weak_ptr<Interface>& parentInterface)
    : m_backendHandle(deviceDescriptorHandle)
    , m_parentInterface(parentInterface)
    , m_informationChangedCallbackManager()
    , m_key(QueryStringFromCInterfaceFunction([&](char* key, size_t* keySize) {
        return PEAK_C_ABI_PREFIX PEAK_DeviceDescriptor_GetKey(deviceDescriptorHandle, key, keySize);
    }))
{
    m_informationChangedCallbackManager = std::make_unique<
        TCallbackManager<PEAK_DEVICE_DESCRIPTOR_INFORMATION_CHANGED_CALLBACK_HANDLE, InformationChangedCallback>>(
        [&](void* callbackContext) {
            return QueryNumericFromCInterfaceFunction<PEAK_DEVICE_DESCRIPTOR_INFORMATION_CHANGED_CALLBACK_HANDLE>(
                [&](PEAK_DEVICE_DESCRIPTOR_INFORMATION_CHANGED_CALLBACK_HANDLE*
                        deviceDescriptorChangedCallbackHandle) {
                    return PEAK_C_ABI_PREFIX PEAK_DeviceDescriptor_RegisterInformationChangedCallback(
                        m_backendHandle, InformationChangedCallbackCWrapper, callbackContext,
                        deviceDescriptorChangedCallbackHandle);
                });
        },
        [&](PEAK_DEVICE_DESCRIPTOR_INFORMATION_CHANGED_CALLBACK_HANDLE callbackHandle) {
            CallAndCheckCInterfaceFunction([&] {
                return PEAK_C_ABI_PREFIX PEAK_DeviceDescriptor_UnregisterInformationChangedCallback(
                    m_backendHandle, callbackHandle);
            });
        });
}

inline DeviceDescriptor::~DeviceDescriptor()
{
    try
    {
        m_informationChangedCallbackManager->UnregisterAllCallbacks();
    }
    catch (const Exception&)
    {}
}

inline std::string DeviceDescriptor::Key() const
{
    return m_key;
}

inline RawInformation DeviceDescriptor::Info(int32_t infoCommand) const
{
    return QueryRawInformationFromCInterfaceFunction([&](int32_t* dataType, uint8_t* buffer, size_t* bufferSize) {
        return PEAK_C_ABI_PREFIX PEAK_DeviceDescriptor_GetInfo(
            m_backendHandle, infoCommand, dataType, buffer, bufferSize);
    });
}

inline std::string DeviceDescriptor::DisplayName() const
{
    return QueryStringFromCInterfaceFunction([&](char* displayName, size_t* displayNameSize) {
        return PEAK_C_ABI_PREFIX PEAK_DeviceDescriptor_GetDisplayName(
            m_backendHandle, displayName, displayNameSize);
    });
}

inline std::string DeviceDescriptor::VendorName() const
{
    return QueryStringFromCInterfaceFunction([&](char* vendorName, size_t* vendorNameSize) {
        return PEAK_C_ABI_PREFIX PEAK_DeviceDescriptor_GetVendorName(m_backendHandle, vendorName, vendorNameSize);
    });
}

inline std::string DeviceDescriptor::ModelName() const
{
    return QueryStringFromCInterfaceFunction([&](char* modelName, size_t* modelNameSize) {
        return PEAK_C_ABI_PREFIX PEAK_DeviceDescriptor_GetModelName(m_backendHandle, modelName, modelNameSize);
    });
}

inline std::string DeviceDescriptor::Version() const
{
    return QueryStringFromCInterfaceFunction([&](char* version, size_t* versionSize) {
        return PEAK_C_ABI_PREFIX PEAK_DeviceDescriptor_GetVersion(m_backendHandle, version, versionSize);
    });
}

inline std::string DeviceDescriptor::TLType() const
{
    return QueryStringFromCInterfaceFunction([&](char* tlType, size_t* tlTypeSize) {
        return PEAK_C_ABI_PREFIX PEAK_DeviceDescriptor_GetTLType(m_backendHandle, tlType, tlTypeSize);
    });
}

inline std::string DeviceDescriptor::UserDefinedName() const
{
    return QueryStringFromCInterfaceFunction([&](char* userDefinedName, size_t* userDefinedNameSize) {
        return PEAK_C_ABI_PREFIX PEAK_DeviceDescriptor_GetUserDefinedName(
            m_backendHandle, userDefinedName, userDefinedNameSize);
    });
}

inline std::string DeviceDescriptor::SerialNumber() const
{
    return QueryStringFromCInterfaceFunction([&](char* serialNumber, size_t* serialNumberSize) {
        return PEAK_C_ABI_PREFIX PEAK_DeviceDescriptor_GetSerialNumber(
            m_backendHandle, serialNumber, serialNumberSize);
    });
}

inline DeviceAccessStatus DeviceDescriptor::AccessStatus() const
{
    return static_cast<DeviceAccessStatus>(QueryNumericFromCInterfaceFunction<PEAK_DEVICE_ACCESS_STATUS>(
        [&](PEAK_DEVICE_ACCESS_STATUS* accessStatus) {
            return PEAK_C_ABI_PREFIX PEAK_DeviceDescriptor_GetAccessStatus(m_backendHandle, accessStatus);
        }));
}

inline uint64_t DeviceDescriptor::TimestampTickFrequency() const
{
    return QueryNumericFromCInterfaceFunction<uint64_t>([&](uint64_t* timestampTickFrequency) {
        return PEAK_C_ABI_PREFIX PEAK_DeviceDescriptor_GetTimestampTickFrequency(
            m_backendHandle, timestampTickFrequency);
    });
}

inline std::shared_ptr<Interface> DeviceDescriptor::ParentInterface() const
{
    return LockOrThrow(m_parentInterface);
}

inline bool DeviceDescriptor::IsOpenable(DeviceAccessType accessType) const
{
    return QueryNumericFromCInterfaceFunction<PEAK_BOOL8>([&](PEAK_BOOL8* isAccessTypeAvailable) {
        return PEAK_C_ABI_PREFIX PEAK_DeviceDescriptor_GetIsOpenable(
            m_backendHandle, static_cast<PEAK_DEVICE_ACCESS_TYPE>(accessType), isAccessTypeAvailable);
    }) > 0;
}

inline std::shared_ptr<Device> DeviceDescriptor::OpenDevice(DeviceAccessType accessType)
{
    auto deviceHandle = QueryNumericFromCInterfaceFunction<PEAK_DEVICE_HANDLE>(
        [&](PEAK_DEVICE_HANDLE* _deviceHandle) {
            return PEAK_C_ABI_PREFIX PEAK_DeviceDescriptor_OpenDevice(
                m_backendHandle, static_cast<PEAK_DEVICE_ACCESS_TYPE>(accessType), _deviceHandle);
        });

    auto device = std::make_shared<ClassCreator<Device>>(deviceHandle, m_parentInterface);
    m_openedDevice = device;
    m_deviceOpenedCallbackManager.TriggerCallbacks(device);

    return device;
}

inline std::shared_ptr<Device> DeviceDescriptor::OpenedDevice() const
{
    return LockOrThrowOpenedModule(m_openedDevice);
}

inline DeviceDescriptor::DeviceOpenedCallbackHandle DeviceDescriptor::RegisterDeviceOpenedCallback(
    const DeviceOpenedCallback& callback)
{
    return m_deviceOpenedCallbackManager.RegisterCallback(callback);
}

inline void DeviceDescriptor::UnregisterDeviceOpenedCallback(DeviceOpenedCallbackHandle callback)
{
    return m_deviceOpenedCallbackManager.UnregisterCallback(callback);
}

inline uint64_t DeviceDescriptor::MonitoringUpdateInterval() const
{
    return QueryNumericFromCInterfaceFunction<uint64_t>([&](uint64_t* monitoringUpdateInterval_ms) {
        return PEAK_C_ABI_PREFIX PEAK_DeviceDescriptor_GetMonitoringUpdateInterval(
            m_backendHandle, monitoringUpdateInterval_ms);
    });
}

inline void DeviceDescriptor::SetMonitoringUpdateInterval(uint64_t monitoringUpdateInterval_ms)
{
    return CallAndCheckCInterfaceFunction([&] {
        return PEAK_C_ABI_PREFIX PEAK_DeviceDescriptor_SetMonitoringUpdateInterval(
            m_backendHandle, monitoringUpdateInterval_ms);
    });
}

inline bool DeviceDescriptor::IsInformationRoleMonitored(DeviceInformationRole informationRole) const
{
    return QueryNumericFromCInterfaceFunction<PEAK_BOOL8>([&](PEAK_BOOL8* isInformationRoleMonitored) {
        return PEAK_C_ABI_PREFIX PEAK_DeviceDescriptor_IsInformationRoleMonitored(
            m_backendHandle, static_cast<PEAK_DEVICE_INFORMATION_ROLE>(informationRole), isInformationRoleMonitored);
    }) > 0;
}

inline void DeviceDescriptor::AddInformationRoleToMonitoring(DeviceInformationRole informationRole)
{
    return CallAndCheckCInterfaceFunction([&] {
        return PEAK_C_ABI_PREFIX PEAK_DeviceDescriptor_AddInformationRoleToMonitoring(
            m_backendHandle, static_cast<PEAK_DEVICE_INFORMATION_ROLE>(informationRole));
    });
}

inline void DeviceDescriptor::RemoveInformationRoleFromMonitoring(DeviceInformationRole informationRole)
{
    return CallAndCheckCInterfaceFunction([&] {
        return PEAK_C_ABI_PREFIX PEAK_DeviceDescriptor_RemoveInformationRoleFromMonitoring(
            m_backendHandle, static_cast<PEAK_DEVICE_INFORMATION_ROLE>(informationRole));
    });
}

inline DeviceDescriptor::InformationChangedCallbackHandle DeviceDescriptor::RegisterInformationChangedCallback(
    const DeviceDescriptor::InformationChangedCallback& callback)
{
    return reinterpret_cast<InformationChangedCallbackHandle>(
        m_informationChangedCallbackManager->RegisterCallback(callback));
}

inline void DeviceDescriptor::UnregisterInformationChangedCallback(InformationChangedCallbackHandle callbackHandle)
{
    m_informationChangedCallbackManager->UnregisterCallback(
        reinterpret_cast<PEAK_DEVICE_DESCRIPTOR_INFORMATION_CHANGED_CALLBACK_HANDLE>(callbackHandle));
}

inline void PEAK_CALL_CONV DeviceDescriptor::InformationChangedCallbackCWrapper(
    const PEAK_DEVICE_INFORMATION_ROLE* changedRoles, size_t changedRolesSize, void* context)
{
    auto callback = static_cast<DeviceDescriptor::InformationChangedCallback*>(context);
    std::vector<DeviceInformationRole> changedRolesVec;
    changedRolesVec.reserve(changedRolesSize);
    for (size_t i = 0; i < changedRolesSize; ++i)
    {
        changedRolesVec.emplace_back(static_cast<DeviceInformationRole>(*(changedRoles + i)));
    }
    callback->operator()(changedRolesVec);
}

inline PEAK_MODULE_DESCRIPTOR_HANDLE DeviceDescriptor::ModuleDescriptorHandle() const
{
    auto moduleDescriptorHandle = QueryNumericFromCInterfaceFunction<PEAK_MODULE_DESCRIPTOR_HANDLE>(
        [&](PEAK_MODULE_DESCRIPTOR_HANDLE* _moduleDescriptorHandle) {
            return PEAK_C_ABI_PREFIX PEAK_DeviceDescriptor_ToModuleDescriptor(
                m_backendHandle, _moduleDescriptorHandle);
        });

    return moduleDescriptorHandle;
}

} /* namespace core */
} /* namespace peak */
