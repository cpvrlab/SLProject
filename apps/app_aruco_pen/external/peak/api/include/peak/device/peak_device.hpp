/*!
 * \file    peak_device.hpp
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
#include <peak/data_stream/peak_data_stream_descriptor.hpp>
#include <peak/device/peak_common_device_enums.hpp>
#include <peak/device/peak_remote_device.hpp>
#include <peak/dll_interface/peak_dll_interface_util.hpp>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>


namespace peak
{
namespace core
{

class Interface;

/*!
 * \brief Represents a GenTL Device module, i.e. a local virtual Device proxy for the actual hardware RemoteDevice.
 *
 * This class allows to query information about a GenTL Device module and to enumerate its
 * \link DataStreamDescriptor DataStreamDescriptors\endlink, which allows to open the corresponding DataStream.
 *
 * Additionally, it allows access to its RemoteDevice, i.e. the actual hardware device. To change hardware settings,
 * you typically need to work with the NodeMap of the RemoteDevice. E.g. to change the exposure time:
 *
 * \code
 * // get the (first) node map of the remote device
 * auto remoteNodeMap = device->RemoteDevice()->NodeMaps().at(0);
 * // change exposure time
 * remoteNodeMap->FindNode<peak::core::nodes::FloatNode>("ExposureTime")->SetValue(1.5);
 * \endcode
 */
class Device
    : public EventSupportingModule
    , public std::enable_shared_from_this<Device>
{
public:
    Device() = delete;
    ~Device() override;
    Device(const Device& other) = delete;
    Device& operator=(const Device& other) = delete;
    Device(Device&& other) = delete;
    Device& operator=(Device&& other) = delete;

    /*! @copydoc ProducerLibrary::Key() */
    std::string Key() const;

    /*! @copydoc SystemDescriptor::Info() */
    RawInformation Info(int32_t infoCommand) const;
    /*! @copydoc System::ID() */
    std::string ID() const;
    /*! @copydoc DeviceDescriptor::DisplayName() */
    std::string DisplayName() const;
    /*! @copydoc DeviceDescriptor::VendorName() */
    std::string VendorName() const;
    /*! @copydoc DeviceDescriptor::ModelName() */
    std::string ModelName() const;
    /*! @copydoc DeviceDescriptor::Version() */
    std::string Version() const;
    /*! @copydoc SystemDescriptor::TLType() */
    std::string TLType() const;
    /*! @copydoc DeviceDescriptor::UserDefinedName() */
    std::string UserDefinedName() const;
    /*! @copydoc DeviceDescriptor::SerialNumber() */
    std::string SerialNumber() const;
    /*! @copydoc DeviceDescriptor::AccessStatus() */
    DeviceAccessStatus AccessStatus() const;
    /*! @copydoc DeviceDescriptor::TimestampTickFrequency() */
    uint64_t TimestampTickFrequency() const;

    /*!
     * \brief Returns the remote device of the device.
     *
     * This function returns the remote device which provides the access to the physical device. In contrast, this class
     * acts only as a proxy for the physical device.
     *
     * \return Remote device
     *
     * \since 1.0
     *
     * \throws InternalErrorException An internal error has occurred.
     */
    std::shared_ptr<class RemoteDevice> RemoteDevice() const;
    /*!
     * \brief Returns the data stream list of the device.
     *
     * \return Data stream list
     *
     * \since 1.0
     *
     * \throws InternalErrorException An internal error has occurred.
     */
    std::vector<std::shared_ptr<DataStreamDescriptor>> DataStreams() const;
    /*! @copydoc DeviceDescriptor::ParentInterface() */
    std::shared_ptr<Interface> ParentInterface() const;

private:
    PEAK_MODULE_HANDLE ModuleHandle() const override;
    PEAK_EVENT_SUPPORTING_MODULE_HANDLE EventSupportingModuleHandle() const override;

    friend ClassCreator<Device>;
    Device(PEAK_DEVICE_HANDLE deviceHandle, const std::weak_ptr<Interface>& parentInterface);
    PEAK_DEVICE_HANDLE m_backendHandle;

    std::weak_ptr<Interface> m_parentInterface;

    void Initialize() const override;
    mutable std::vector<std::shared_ptr<DataStreamDescriptor>> m_dataStreams;
    mutable std::shared_ptr<class RemoteDevice> m_remoteDevice;

    std::string m_key;
};

} /* namespace core */
} /* namespace peak */

/* Implementation */
namespace peak
{
namespace core
{

inline Device::Device(PEAK_DEVICE_HANDLE deviceHandle, const std::weak_ptr<Interface>& parentInterface)
    : m_backendHandle(deviceHandle)
    , m_parentInterface(parentInterface)
    , m_key(QueryStringFromCInterfaceFunction([&](char* key, size_t* keySize) {
        return PEAK_C_ABI_PREFIX PEAK_Device_GetKey(deviceHandle, key, keySize);
    }))
{}

inline Device::~Device()
{
    (void)PEAK_C_ABI_PREFIX PEAK_Device_Destruct(m_backendHandle);
}

inline std::string Device::Key() const
{
    return m_key;
}

inline RawInformation Device::Info(int32_t infoCommand) const
{
    return QueryRawInformationFromCInterfaceFunction([&](int32_t* dataType, uint8_t* buffer, size_t* bufferSize) {
        return PEAK_C_ABI_PREFIX PEAK_Device_GetInfo(m_backendHandle, infoCommand, dataType, buffer, bufferSize);
    });
}

inline std::string Device::ID() const
{
    return QueryStringFromCInterfaceFunction([&](char* id, size_t* idSize) {
        return PEAK_C_ABI_PREFIX PEAK_Device_GetID(m_backendHandle, id, idSize);
    });
}

inline std::string Device::DisplayName() const
{
    return QueryStringFromCInterfaceFunction([&](char* displayName, size_t* displayNameSize) {
        return PEAK_C_ABI_PREFIX PEAK_Device_GetDisplayName(m_backendHandle, displayName, displayNameSize);
    });
}

inline std::string Device::VendorName() const
{
    return QueryStringFromCInterfaceFunction([&](char* vendorName, size_t* vendorNameSize) {
        return PEAK_C_ABI_PREFIX PEAK_Device_GetVendorName(m_backendHandle, vendorName, vendorNameSize);
    });
}

inline std::string Device::ModelName() const
{
    return QueryStringFromCInterfaceFunction([&](char* modelName, size_t* modelNameSize) {
        return PEAK_C_ABI_PREFIX PEAK_Device_GetModelName(m_backendHandle, modelName, modelNameSize);
    });
}

inline std::string Device::Version() const
{
    return QueryStringFromCInterfaceFunction([&](char* version, size_t* versionSize) {
        return PEAK_C_ABI_PREFIX PEAK_Device_GetVersion(m_backendHandle, version, versionSize);
    });
}

inline std::string Device::TLType() const
{
    return QueryStringFromCInterfaceFunction([&](char* tlType, size_t* tlTypeSize) {
        return PEAK_C_ABI_PREFIX PEAK_Device_GetTLType(m_backendHandle, tlType, tlTypeSize);
    });
}

inline std::string Device::UserDefinedName() const
{
    return QueryStringFromCInterfaceFunction([&](char* userDefinedName, size_t* userDefinedNameSize) {
        return PEAK_C_ABI_PREFIX PEAK_Device_GetUserDefinedName(
            m_backendHandle, userDefinedName, userDefinedNameSize);
    });
}

inline std::string Device::SerialNumber() const
{
    return QueryStringFromCInterfaceFunction([&](char* serialNumber, size_t* serialNumberSize) {
        return PEAK_C_ABI_PREFIX PEAK_Device_GetSerialNumber(m_backendHandle, serialNumber, serialNumberSize);
    });
}

inline DeviceAccessStatus Device::AccessStatus() const
{
    return static_cast<DeviceAccessStatus>(QueryNumericFromCInterfaceFunction<PEAK_DEVICE_ACCESS_STATUS>(
        [&](PEAK_DEVICE_ACCESS_STATUS* accessStatus) {
            return PEAK_C_ABI_PREFIX PEAK_Device_GetAccessStatus(m_backendHandle, accessStatus);
        }));
}

inline uint64_t Device::TimestampTickFrequency() const
{
    return QueryNumericFromCInterfaceFunction<uint64_t>([&](uint64_t* timestampTickFrequency) {
        return PEAK_C_ABI_PREFIX PEAK_Device_GetTimestampTickFrequency(m_backendHandle, timestampTickFrequency);
    });
}

inline std::shared_ptr<class RemoteDevice> Device::RemoteDevice() const
{
    InitializeIfNecessary();

    return m_remoteDevice;
}

inline std::vector<std::shared_ptr<DataStreamDescriptor>> Device::DataStreams() const
{
    InitializeIfNecessary();

    return m_dataStreams;
}

inline std::shared_ptr<Interface> Device::ParentInterface() const
{
    return LockOrThrow(m_parentInterface);
}

inline PEAK_MODULE_HANDLE Device::ModuleHandle() const
{
    auto moduleHandle = QueryNumericFromCInterfaceFunction<PEAK_MODULE_HANDLE>(
        [&](PEAK_MODULE_HANDLE* _moduleHandle) {
            return PEAK_C_ABI_PREFIX PEAK_Device_ToModule(m_backendHandle, _moduleHandle);
        });

    return moduleHandle;
}

inline PEAK_EVENT_SUPPORTING_MODULE_HANDLE Device::EventSupportingModuleHandle() const
{
    auto eventSupportingModuleHandle = QueryNumericFromCInterfaceFunction<PEAK_EVENT_SUPPORTING_MODULE_HANDLE>(
        [&](PEAK_EVENT_SUPPORTING_MODULE_HANDLE* _eventSupportingModuleHandle) {
            return PEAK_C_ABI_PREFIX PEAK_Device_ToEventSupportingModule(
                m_backendHandle, _eventSupportingModuleHandle);
        });

    return eventSupportingModuleHandle;
}

inline void Device::Initialize() const
{
    EventSupportingModule::Initialize(); // init data of parent classes (i.e. Module) as well

    // init data streams
    auto numDataStreams = QueryNumericFromCInterfaceFunction<size_t>([&](size_t* _numDataStreams) {
        return PEAK_C_ABI_PREFIX PEAK_Device_GetNumDataStreams(m_backendHandle, _numDataStreams);
    });

    std::vector<std::shared_ptr<DataStreamDescriptor>> dataStreams;
    for (size_t x = 0; x < numDataStreams; ++x)
    {
        auto dataStreamDescriptorHandle = QueryNumericFromCInterfaceFunction<PEAK_DATA_STREAM_DESCRIPTOR_HANDLE>(
            [&](PEAK_DATA_STREAM_DESCRIPTOR_HANDLE* _dataStreamDescriptorHandle) {
                return PEAK_C_ABI_PREFIX PEAK_Device_GetDataStream(
                    m_backendHandle, x, _dataStreamDescriptorHandle);
            });

        dataStreams.emplace_back(std::make_shared<ClassCreator<DataStreamDescriptor>>(
            dataStreamDescriptorHandle, std::const_pointer_cast<Device>(shared_from_this())));
    }

    m_dataStreams = dataStreams;

    // init remote device
    auto remoteDeviceHandle = QueryNumericFromCInterfaceFunction<PEAK_REMOTE_DEVICE_HANDLE>(
        [&](PEAK_REMOTE_DEVICE_HANDLE* _remoteDeviceHandle) {
            return PEAK_C_ABI_PREFIX PEAK_Device_GetRemoteDevice(m_backendHandle, _remoteDeviceHandle);
        });

    m_remoteDevice = std::make_shared<ClassCreator<peak::core::RemoteDevice>>(
        remoteDeviceHandle, std::const_pointer_cast<Device>(shared_from_this()));
}

} /* namespace core */
} /* namespace peak */
