/*!
 * \file    peak_system_descriptor.hpp
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
#include <peak/system/peak_common_system_enums.hpp>
#include <peak/system/peak_system.hpp>

#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <vector>


namespace peak
{
namespace core
{

class ProducerLibrary;

/*!
 * \brief Encapsulates the GenTL functions associated with the GenTL System module's ID.
 *
 * This class allows to query information about the GenTL System module without opening it. Furthermore, it enables
 * you to open the GenTL System module.
 *
 */
class SystemDescriptor : public ModuleDescriptor
{
public:
    SystemDescriptor() = delete;
    ~SystemDescriptor() override = default;
    SystemDescriptor(const SystemDescriptor& other) = delete;
    SystemDescriptor& operator=(const SystemDescriptor& other) = delete;
    SystemDescriptor(SystemDescriptor&& other) = delete;
    SystemDescriptor& operator=(SystemDescriptor&& other) = delete;

    /*! @copydoc ProducerLibrary::Key() */
    std::string Key() const;

    /*!
     * \brief Delivers information based on the given GenTL info command.
     *
     * \param[in] infoCommand The GenTL info command.
     *
     * This function can be used to query information going beyond the predefined info functions, based on the GenTL
     * info commands of the corresponding module.
     *
     * Example (error handling is omitted):
     *
     * \code
     * #include <peak/thirdparty/GenTL.h>
     * auto info = object->Info(GenTL::XX_INFO_XX);
     *
     * // Cast depending on the delivered data type
     * if (info.dataType == GenTL::INFO_DATATYPE_UINT64)
     * {
     *     uint64_t uint64Var = *reinterpret_cast<const uint64_t*>(info.data.data());
     *     // Do something with the information
     * }
     * else if (infoDataTypeVar == GenTL::INFO_DATATYPE_INT64)
     * {
     *     ...
     * }
     * ...
     * \endcode
     *
     * \return Raw information according to the passed info command
     *
     * \since 1.0
     *
     * \throws InternalErrorException An internal error has occurred.
     */
    RawInformation Info(int32_t infoCommand) const;
    /*!
     * \brief Returns the display name.
     *
     * \return Display name
     *
     * \since 1.0
     *
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
     * \throws InternalErrorException An internal error has occurred.
     */
    std::string Version() const;
    /*!
     * \brief Returns the TL (transport layer) type.
     *
     * \return TL type
     *
     * \since 1.0
     *
     * \throws InternalErrorException An internal error has occurred.
     */
    std::string TLType() const;
    /*!
     * \brief Returns the file name of the GenTL producer library this system belongs to.
     *
     * \return File name of the GenTL producer library
     *
     * \since 1.0
     *
     * \throws InternalErrorException An internal error has occurred.
     */
    std::string CTIFileName() const;
    /*!
     * \brief Returns the full path of the GenTL producer library this system belongs to.
     *
     * \return Full path of the GenTL producer library
     *
     * \since 1.0
     *
     * \throws InternalErrorException An internal error has occurred.
     */
    std::string CTIFullPath() const;
    /*!
     * \brief Returns the GenTL major version of the GenTL producer library this system belongs to.
     *
     * \return GenTL major version of the GenTL producer library
     *
     * \since 1.0
     *
     * \throws InternalErrorException An internal error has occurred.
     */
    uint32_t GenTLVersionMajor() const;
    /*!
     * \brief Returns the GenTL minor version of the GenTL producer library this system belongs to.
     *
     * \return GenTL minor version of the GenTL producer library
     *
     * \since 1.0
     *
     * \throws InternalErrorException An internal error has occurred.
     */
    uint32_t GenTLVersionMinor() const;
    /*!
     * \brief Returns the character encoding of the GenTL producer library this system belongs to.
     *
     * \return Character encoding of the GenTL producer library
     *
     * \since 1.0
     *
     * \throws InternalErrorException An internal error has occurred.
     */
    peak::core::CharacterEncoding CharacterEncoding() const;
    /*!
     * \brief Returns the parent library.
     *
     * \return Parent library
     *
     * \since 1.0
     *
     * \throws InternalErrorException An internal error has occurred.
     */
    std::shared_ptr<ProducerLibrary> ParentLibrary() const;
    /*!
     * \brief Opens the system.
     *
     * \return Opened system
     *
     * \since 1.0
     *
     * \throws BadAccessException Access denied
     * \throws InternalErrorException An internal error has occurred.
     */
    std::shared_ptr<System> OpenSystem();
    /*!
     * \brief Returns the System that was opened with this SystemDescriptor.
     *
     * \return Opened System
     *
     * \since 1.0
     *
     * \throws BadAccessException System is not open
     */
    std::shared_ptr<System> OpenedSystem() const;

private:
    PEAK_MODULE_DESCRIPTOR_HANDLE ModuleDescriptorHandle() const override;

    friend ClassCreator<SystemDescriptor>;
    SystemDescriptor(
        PEAK_SYSTEM_DESCRIPTOR_HANDLE systemDescriptorHandle, const std::weak_ptr<ProducerLibrary>& parentLibrary);
    PEAK_SYSTEM_DESCRIPTOR_HANDLE m_backendHandle;

    std::weak_ptr<ProducerLibrary> m_parentLibrary;
    std::weak_ptr<System> m_openedSystem;

    std::string m_key;
};

} /* namespace core */
} /* namespace peak */

/* Implementation */
namespace peak
{
namespace core
{

inline SystemDescriptor::SystemDescriptor(
    PEAK_SYSTEM_DESCRIPTOR_HANDLE systemDescriptorHandle, const std::weak_ptr<ProducerLibrary>& parentLibrary)
    : m_backendHandle(systemDescriptorHandle)
    , m_parentLibrary(parentLibrary)
    , m_key(QueryStringFromCInterfaceFunction([&](char* key, size_t* keySize) {
        return PEAK_C_ABI_PREFIX PEAK_SystemDescriptor_GetKey(systemDescriptorHandle, key, keySize);
    }))
{}

inline std::string SystemDescriptor::Key() const
{
    return m_key;
}

inline RawInformation SystemDescriptor::Info(int32_t infoCommand) const
{
    return QueryRawInformationFromCInterfaceFunction([&](int32_t* dataType, uint8_t* buffer, size_t* bufferSize) {
        return PEAK_C_ABI_PREFIX PEAK_SystemDescriptor_GetInfo(
            m_backendHandle, infoCommand, dataType, buffer, bufferSize);
    });
}

inline std::string SystemDescriptor::DisplayName() const
{
    return QueryStringFromCInterfaceFunction([&](char* displayName, size_t* displayNameSize) {
        return PEAK_C_ABI_PREFIX PEAK_SystemDescriptor_GetDisplayName(
            m_backendHandle, displayName, displayNameSize);
    });
}

inline std::string SystemDescriptor::VendorName() const
{
    return QueryStringFromCInterfaceFunction([&](char* vendorName, size_t* vendorNameSize) {
        return PEAK_C_ABI_PREFIX PEAK_SystemDescriptor_GetVendorName(m_backendHandle, vendorName, vendorNameSize);
    });
}

inline std::string SystemDescriptor::ModelName() const
{
    return QueryStringFromCInterfaceFunction([&](char* modelName, size_t* modelNameSize) {
        return PEAK_C_ABI_PREFIX PEAK_SystemDescriptor_GetModelName(m_backendHandle, modelName, modelNameSize);
    });
}

inline std::string SystemDescriptor::Version() const
{
    return QueryStringFromCInterfaceFunction([&](char* version, size_t* versionSize) {
        return PEAK_C_ABI_PREFIX PEAK_SystemDescriptor_GetVersion(m_backendHandle, version, versionSize);
    });
}

inline std::string SystemDescriptor::TLType() const
{
    return QueryStringFromCInterfaceFunction([&](char* tlType, size_t* tlTypeSize) {
        return PEAK_C_ABI_PREFIX PEAK_SystemDescriptor_GetTLType(m_backendHandle, tlType, tlTypeSize);
    });
}

inline std::string SystemDescriptor::CTIFileName() const
{
    return QueryStringFromCInterfaceFunction([&](char* ctiFileName, size_t* ctiFileNameSize) {
        return PEAK_C_ABI_PREFIX PEAK_SystemDescriptor_GetCTIFileName(
            m_backendHandle, ctiFileName, ctiFileNameSize);
    });
}

inline std::string SystemDescriptor::CTIFullPath() const
{
    return QueryStringFromCInterfaceFunction([&](char* ctiFullPath, size_t* ctiFullPathSize) {
        return PEAK_C_ABI_PREFIX PEAK_SystemDescriptor_GetCTIFullPath(
            m_backendHandle, ctiFullPath, ctiFullPathSize);
    });
}

inline uint32_t SystemDescriptor::GenTLVersionMajor() const
{
    return QueryNumericFromCInterfaceFunction<uint32_t>([&](uint32_t* gentlVersionMajor) {
        return PEAK_C_ABI_PREFIX PEAK_SystemDescriptor_GetGenTLVersionMajor(m_backendHandle, gentlVersionMajor);
    });
}

inline uint32_t SystemDescriptor::GenTLVersionMinor() const
{
    return QueryNumericFromCInterfaceFunction<uint32_t>([&](uint32_t* gentlVersionMinor) {
        return PEAK_C_ABI_PREFIX PEAK_SystemDescriptor_GetGenTLVersionMinor(m_backendHandle, gentlVersionMinor);
    });
}

inline peak::core::CharacterEncoding SystemDescriptor::CharacterEncoding() const
{
    return static_cast<peak::core::CharacterEncoding>(
        QueryNumericFromCInterfaceFunction<PEAK_CHARACTER_ENCODING>(
            [&](PEAK_CHARACTER_ENCODING* characterEncoding) {
                return PEAK_C_ABI_PREFIX PEAK_SystemDescriptor_GetCharacterEncoding(
                    m_backendHandle, characterEncoding);
            }));
}

inline std::shared_ptr<ProducerLibrary> SystemDescriptor::ParentLibrary() const
{
    return LockOrThrow(m_parentLibrary);
}

inline std::shared_ptr<System> SystemDescriptor::OpenSystem()
{
    auto systemHandle = QueryNumericFromCInterfaceFunction<PEAK_SYSTEM_HANDLE>(
        [&](PEAK_SYSTEM_HANDLE* _systemHandle) {
            return PEAK_C_ABI_PREFIX PEAK_SystemDescriptor_OpenSystem(m_backendHandle, _systemHandle);
        });

    auto system = std::make_shared<ClassCreator<System>>(systemHandle, m_parentLibrary);
    m_openedSystem = system;
    return system;
}

inline std::shared_ptr<System> SystemDescriptor::OpenedSystem() const
{
    return LockOrThrowOpenedModule(m_openedSystem);
}

inline PEAK_MODULE_DESCRIPTOR_HANDLE SystemDescriptor::ModuleDescriptorHandle() const
{
    auto moduleDescriptorHandle = QueryNumericFromCInterfaceFunction<PEAK_MODULE_DESCRIPTOR_HANDLE>(
        [&](PEAK_MODULE_DESCRIPTOR_HANDLE* _moduleDescriptorHandle) {
            return PEAK_C_ABI_PREFIX PEAK_SystemDescriptor_ToModuleDescriptor(
                m_backendHandle, _moduleDescriptorHandle);
        });

    return moduleDescriptorHandle;
}

} /* namespace core */
} /* namespace peak */
