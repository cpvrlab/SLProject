/*!
 * \file    peak_remote_device.hpp
 *
 * \author  IDS Imaging Development Systems GmbH
 * \date    2019-05-01
 * \since   1.0
 *
 * Copyright (c) 2019, IDS Imaging Development Systems GmbH. All rights reserved.
 */

#pragma once


#include <peak/backend/peak_backend.h>
#include <peak/common/peak_module.hpp>
#include <peak/dll_interface/peak_dll_interface_util.hpp>

#include <memory>


namespace peak
{
namespace core
{

class Device;

/*!
 * \brief Allows to access the physical Device.
 *
 * Access the NodeMap and Port of the physical device. This class doesn't have any additional functionality,
 * but helps separate the local virtual device proxy from that actual physical hardware device. I.e. the NodeMap of
 * this RemoteDevice is created from the XML file retrieved from the physical device, while the NodeMap of the proxy
 * Device is created by the ProducerLibrary (CTI).
 */
class RemoteDevice : public Module
{
public:
    RemoteDevice() = delete;
    ~RemoteDevice() override = default;
    RemoteDevice(const class RemoteDevice& other) = delete;
    RemoteDevice& operator=(const class RemoteDevice& other) = delete;
    RemoteDevice(class RemoteDevice&& other) = delete;
    RemoteDevice& operator=(class RemoteDevice&& other) = delete;

    /*!
     * \brief Returns the local device (device proxy) of the remote device.
     *
     * \return Local device of the remote device
     *
     * \since 1.0
     *
     * \throws InternalErrorException An internal error has occurred.
     */
    std::shared_ptr<Device> LocalDevice() const;

private:
    friend ClassCreator<RemoteDevice>;
    RemoteDevice(PEAK_REMOTE_DEVICE_HANDLE remoteDeviceHandle, const std::weak_ptr<Device>& localDevice);
    PEAK_MODULE_HANDLE ModuleHandle() const override;

    PEAK_REMOTE_DEVICE_HANDLE m_backendHandle;

    std::weak_ptr<Device> m_localDevice;
};

} /* namespace core */
} /* namespace peak */

/* Implementation */
namespace peak
{
namespace core
{

inline RemoteDevice::RemoteDevice(
    PEAK_REMOTE_DEVICE_HANDLE remoteDeviceHandle, const std::weak_ptr<Device>& localDevice)
    : m_backendHandle(remoteDeviceHandle)
    , m_localDevice(localDevice)
{}

inline std::shared_ptr<Device> RemoteDevice::LocalDevice() const
{
    return LockOrThrow(m_localDevice);
}

inline PEAK_MODULE_HANDLE RemoteDevice::ModuleHandle() const
{
    auto moduleHandle = QueryNumericFromCInterfaceFunction<PEAK_MODULE_HANDLE>(
        [&](PEAK_MODULE_HANDLE* _moduleHandle) {
            return PEAK_C_ABI_PREFIX PEAK_RemoteDevice_ToModule(m_backendHandle, _moduleHandle);
        });

    return moduleHandle;
}

} /* namespace core */
} /* namespace peak */
