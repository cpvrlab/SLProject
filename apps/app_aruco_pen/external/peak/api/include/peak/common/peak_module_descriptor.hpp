/*!
 * \file    peak_module_descriptor.hpp
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

#include <cstddef>
#include <string>


namespace peak
{
namespace core
{

/*!
 * \brief The base class for all openable modules.
 *
 */
class ModuleDescriptor
{
public:
    ModuleDescriptor() = default;
    virtual ~ModuleDescriptor() = default;
    ModuleDescriptor(const ModuleDescriptor& other) = delete;
    ModuleDescriptor& operator=(const ModuleDescriptor& other) = delete;
    ModuleDescriptor(ModuleDescriptor&& other) = delete;
    ModuleDescriptor& operator=(ModuleDescriptor&& other) = delete;

    /*!
     * \brief Returns the ID of the described module.
     *
     * \return ID
     *
     * \since 1.0
     *
     * \throws InternalErrorException An internal error has occurred.
     */
    std::string ID() const;

protected:
    virtual PEAK_MODULE_DESCRIPTOR_HANDLE ModuleDescriptorHandle() const = 0;
};

} /* namespace core */
} /* namespace peak */

/* Implementation */
namespace peak
{
namespace core
{

inline std::string ModuleDescriptor::ID() const
{
    auto moduleDescriptorHandle = ModuleDescriptorHandle();

    return QueryStringFromCInterfaceFunction([&](char* id, size_t* idSize) {
        return PEAK_C_ABI_PREFIX PEAK_ModuleDescriptor_GetID(moduleDescriptorHandle, id, idSize);
    });
}

} /* namespace core */
} /* namespace peak */
