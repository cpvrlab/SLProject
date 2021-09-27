/*!
 * \file    peak_common_device_enums.hpp
 *
 * \author  IDS Imaging Development Systems GmbH
 * \date    2019-05-01
 * \since   1.0
 *
 * Copyright (c) 2019, IDS Imaging Development Systems GmbH. All rights reserved.
 */

#pragma once


namespace peak
{
namespace core
{

/*!
 * \brief Current accessibility of the device.
 *
 * See GenTL DEVICE_ACCESS_STATUS.
 */
enum class DeviceAccessStatus
{
    ReadWrite = 1,
    ReadOnly,
    NoAccess,
    Busy,
    OpenReadWrite,
    OpenReadOnly,

    Custom = 1000
};

inline std::string ToString(DeviceAccessStatus entry)
{
    std::string entryString;

    if (entry == DeviceAccessStatus::ReadWrite)
    {
        entryString = "ReadWrite";
    }
    else if (entry == DeviceAccessStatus::ReadOnly)
    {
        entryString = "ReadOnly";
    }
    else if (entry == DeviceAccessStatus::NoAccess)
    {
        entryString = "NoAccess";
    }
    else if (entry == DeviceAccessStatus::Busy)
    {
        entryString = "Busy";
    }
    else if (entry == DeviceAccessStatus::OpenReadWrite)
    {
        entryString = "OpenReadWrite";
    }
    else if (entry == DeviceAccessStatus::OpenReadOnly)
    {
        entryString = "OpenReadOnly";
    }
    else if (entry >= DeviceAccessStatus::Custom)
    {
        entryString = "Custom";
    }

    return entryString;
}

} /* namespace core */
} /* namespace peak */
