/*!
 * \file    peak_interface_descriptor.hpp
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
#include <peak/dll_interface/peak_dll_interface_util.hpp>
#include <peak/interface/peak_interface.hpp>

#include <cstdint>
#include <memory>
#include <string>


namespace peak
{
namespace core
{

class System;

/*!
 * \brief Encapsulates the GenTL functions associated with a GenTL Interface module's ID.
 *
 * This class allows to query information about a GenTL Interface module without opening it. Furthermore, it allows
 * to open the corresponding Interface module.
 *
 */
class InterfaceDescriptor : public ModuleDescriptor
{
public:
    InterfaceDescriptor() = delete;
    ~InterfaceDescriptor() override = default;
    InterfaceDescriptor(const InterfaceDescriptor& other) = delete;
    InterfaceDescriptor& operator=(const InterfaceDescriptor& other) = delete;
    InterfaceDescriptor(InterfaceDescriptor&& other) = delete;
    InterfaceDescriptor& operator=(InterfaceDescriptor&& other) = delete;

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
     * \throws NotFoundException Interface could not be found.
     * \throws InternalErrorException An internal error has occurred.
     */
    std::string DisplayName() const;
    /*! @copydoc SystemDescriptor::TLType() */
    std::string TLType() const;

    /*!
     * \brief Returns the parent system.
     *
     * \return Parent system
     *
     * \since 1.0
     *
     * \throws NotFoundException Interface could not be found.
     * \throws InternalErrorException An internal error has occurred.
     */
    std::shared_ptr<System> ParentSystem() const;

    /*!
     * \brief Opens the interface.
     *
     * \return Opened interface
     *
     * \since 1.0
     *
     * \throws BadAccessException Access denied
     * \throws InternalErrorException An internal error has occurred.
     */
    std::shared_ptr<Interface> OpenInterface();
    /*!
     * \brief Returns the Interface that was opened with this InterfaceDescriptor.
     *
     * \return Opened Interface
     *
     * \since 1.0
     *
     * \throws BadAccessException Interface is not open
     */
    std::shared_ptr<Interface> OpenedInterface() const;

private:
    PEAK_MODULE_DESCRIPTOR_HANDLE ModuleDescriptorHandle() const override;

    friend ClassCreator<InterfaceDescriptor>;
    InterfaceDescriptor(
        PEAK_INTERFACE_DESCRIPTOR_HANDLE interfaceDescriptorHandle, const std::weak_ptr<System>& parentSystem);
    PEAK_INTERFACE_DESCRIPTOR_HANDLE m_backendHandle;

    std::weak_ptr<System> m_parentSystem;
    std::weak_ptr<Interface> m_openedInterface;

    std::string m_key;
};

} /* namespace core */
} /* namespace peak */

/* Implementation */
namespace peak
{
namespace core
{

inline InterfaceDescriptor::InterfaceDescriptor(
    PEAK_INTERFACE_DESCRIPTOR_HANDLE interfaceDescriptorHandle, const std::weak_ptr<System>& parentSystem)
    : m_backendHandle(interfaceDescriptorHandle)
    , m_parentSystem(parentSystem)
    , m_key(QueryStringFromCInterfaceFunction([&](char* key, size_t* keySize) {
        return PEAK_C_ABI_PREFIX PEAK_InterfaceDescriptor_GetKey(interfaceDescriptorHandle, key, keySize);
    }))
{}

inline std::string InterfaceDescriptor::Key() const
{
    return m_key;
}

inline RawInformation InterfaceDescriptor::Info(int32_t infoCommand) const
{
    return QueryRawInformationFromCInterfaceFunction([&](int32_t* dataType, uint8_t* buffer, size_t* bufferSize) {
        return PEAK_C_ABI_PREFIX PEAK_InterfaceDescriptor_GetInfo(
            m_backendHandle, infoCommand, dataType, buffer, bufferSize);
    });
}

inline std::string InterfaceDescriptor::DisplayName() const
{
    return QueryStringFromCInterfaceFunction([&](char* displayName, size_t* displayNameSize) {
        return PEAK_C_ABI_PREFIX PEAK_InterfaceDescriptor_GetDisplayName(
            m_backendHandle, displayName, displayNameSize);
    });
}

inline std::string InterfaceDescriptor::TLType() const
{
    return QueryStringFromCInterfaceFunction([&](char* tlType, size_t* tlTypeSize) {
        return PEAK_C_ABI_PREFIX PEAK_InterfaceDescriptor_GetTLType(m_backendHandle, tlType, tlTypeSize);
    });
}

inline std::shared_ptr<System> InterfaceDescriptor::ParentSystem() const
{
    return LockOrThrow(m_parentSystem);
}

inline std::shared_ptr<Interface> InterfaceDescriptor::OpenInterface()
{
    auto interfaceHandle = QueryNumericFromCInterfaceFunction<PEAK_INTERFACE_HANDLE>(
        [&](PEAK_INTERFACE_HANDLE* _interfaceHandle) {
            return PEAK_C_ABI_PREFIX PEAK_InterfaceDescriptor_OpenInterface(m_backendHandle, _interfaceHandle);
        });

    auto interface_ = std::make_shared<ClassCreator<Interface>>(interfaceHandle, m_parentSystem);
    m_openedInterface = interface_;
    return interface_;
}

inline std::shared_ptr<Interface> InterfaceDescriptor::OpenedInterface() const
{
    return LockOrThrowOpenedModule(m_openedInterface);
}

inline PEAK_MODULE_DESCRIPTOR_HANDLE InterfaceDescriptor::ModuleDescriptorHandle() const
{
    auto moduleDescriptorHandle = QueryNumericFromCInterfaceFunction<PEAK_MODULE_DESCRIPTOR_HANDLE>(
        [&](PEAK_MODULE_DESCRIPTOR_HANDLE* _moduleDescriptorHandle) {
            return PEAK_C_ABI_PREFIX PEAK_InterfaceDescriptor_ToModuleDescriptor(
                m_backendHandle, _moduleDescriptorHandle);
        });

    return moduleDescriptorHandle;
}

} /* namespace core */
} /* namespace peak */
