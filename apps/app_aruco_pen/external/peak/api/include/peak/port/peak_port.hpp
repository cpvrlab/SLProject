/*!
 * \file    peak_port.hpp
 *
 * \author  IDS Imaging Development Systems GmbH
 * \date    2019-05-01
 * \since   1.0
 *
 * Copyright (c) 2019, IDS Imaging Development Systems GmbH. All rights reserved.
 */

#pragma once


#include <peak/backend/peak_backend.h>
#include <peak/common/peak_common_enums.hpp>
#include <peak/common/peak_common_structs.hpp>
#include <peak/dll_interface/peak_dll_interface_util.hpp>
#include <peak/error_handling/peak_error_handling.hpp>
#include <peak/generic/peak_init_once.hpp>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>


namespace peak
{
namespace core
{

class PortURL;

/*!
 * \brief Represents a GenTL port.
 *
 * This class allows to query information about a GenTL port and to enumerate its URLs.
 *
 */
class Port
    : public InitOnce
    , public std::enable_shared_from_this<Port>
{
public:
    Port() = delete;
    ~Port() = default;
    Port(const class Port& other) = delete;
    Port& operator=(const class Port& other) = delete;
    Port(class Port&& other) = delete;
    Port& operator=(class Port&& other) = delete;

    /*! @copydoc SystemDescriptor::Info() */
    RawInformation Info(int32_t infoCommand) const;
    /*!
     * \brief Returns the ID.
     *
     * \return ID
     *
     * \since 1.0
     *
     * \throws InternalErrorException An internal error has occurred.
     */
    std::string ID() const;
    /*!
     * \brief Returns the name.
     *
     * \return Name
     *
     * \since 1.0
     *
     * \throws InternalErrorException An internal error has occurred.
     */
    std::string Name() const;
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
    /*! @copydoc SystemDescriptor::TLType() */
    std::string TLType() const;
    /*!
     * \brief Returns the module name.
     *
     * \return Module name
     *
     * \since 1.0
     *
     * \throws InternalErrorException An internal error has occurred.
     */
    std::string ModuleName() const;
    /*!
     * \brief Returns the data endianness.
     *
     * \return Data endianness
     *
     * \since 1.0
     *
     * \throws InternalErrorException An internal error has occurred.
     */
    Endianness DataEndianness() const;
    /*!
     * \brief Checks whether the port is readable.
     *
     * \return True, if the port is readable
     * \return False otherwise.
     *
     * \since 1.0
     *
     * \throws InternalErrorException An internal error has occurred.
     */
    bool IsReadable() const;
    /*!
     * \brief Checks whether the port is writable.
     *
     * \return True, if the port is writable.
     * \return False otherwise.
     *
     * \since 1.0
     *
     * \throws InternalErrorException An internal error has occurred.
     */
    bool IsWritable() const;
    /*!
     * \brief Checks whether the port is available.
     *
     * \return True, if the port is available.
     * \return False otherwise.
     *
     * \since 1.0
     *
     * \throws InternalErrorException An internal error has occurred.
     */
    bool IsAvailable() const;
    /*!
     * \brief Checks whether the port is implemented.
     *
     * \return True, if the port is implemented.
     * \return False otherwise.
     *
     * \since 1.0
     *
     * \throws InternalErrorException An internal error has occurred.
     */
    bool IsImplemented() const;

    /*!
     * \brief Reads a given amount of bytes at a given address.
     *
     * \param[in] address The address to read at.
     * \param[in] numBytes The amount of bytes to read.
     *
     * \return Read bytes
     *
     * \since 1.0
     *
     * \throws InvalidAddressException Address is invalid
     * \throws InternalErrorException An internal error has occurred.
     */
    std::vector<uint8_t> Read(uint64_t address, size_t numBytes) const;
    /*!
     * \brief Writes a given amount of bytes at a given address.
     *
     * \param[in] address The address to write at.
     * \param[in] bytes The bytes to write.
     *
     * \since 1.0
     *
     * \throws InvalidAddressException Address is invalid
     * \throws OutOfRangeException The given value is out of range
     * \throws InternalErrorException An internal error has occurred.
     */
    void Write(uint64_t address, const std::vector<uint8_t>& bytes) const;
    /*!
     * \brief Returns the URLs.
     *
     * \return URLs
     *
     * \since 1.0
     *
     * \throws InternalErrorException An internal error has occurred.
     */
    std::vector<std::shared_ptr<PortURL>> URLs() const;

private:
    friend class ClassCreator<Port>;
    explicit Port(PEAK_PORT_HANDLE portHandle);
    PEAK_PORT_HANDLE m_backendHandle;

    void Initialize() const override;
    mutable std::vector<std::shared_ptr<PortURL>> m_portURLs;
};

} /* namespace core */
} /* namespace peak */

#include <peak/port/peak_port_url.hpp>


/* Implementation */
namespace peak
{
namespace core
{

inline Port::Port(PEAK_PORT_HANDLE portHandle)
    : m_backendHandle(portHandle)
{}

inline RawInformation Port::Info(int32_t infoCommand) const
{
    return QueryRawInformationFromCInterfaceFunction([&](int32_t* dataType, uint8_t* buffer, size_t* bufferSize) {
        return PEAK_C_ABI_PREFIX PEAK_Port_GetInfo(m_backendHandle, infoCommand, dataType, buffer, bufferSize);
    });
}

inline std::string Port::ID() const
{
    return QueryStringFromCInterfaceFunction(
        [&](char* id, size_t* idSize) { return PEAK_C_ABI_PREFIX PEAK_Port_GetID(m_backendHandle, id, idSize); });
}

inline std::string Port::Name() const
{
    return QueryStringFromCInterfaceFunction([&](char* name, size_t* nameSize) {
        return PEAK_C_ABI_PREFIX PEAK_Port_GetName(m_backendHandle, name, nameSize);
    });
}

inline std::string Port::VendorName() const
{
    return QueryStringFromCInterfaceFunction([&](char* vendorName, size_t* vendorNameSize) {
        return PEAK_C_ABI_PREFIX PEAK_Port_GetVendorName(m_backendHandle, vendorName, vendorNameSize);
    });
}

inline std::string Port::ModelName() const
{
    return QueryStringFromCInterfaceFunction([&](char* modelName, size_t* modelNameSize) {
        return PEAK_C_ABI_PREFIX PEAK_Port_GetModelName(m_backendHandle, modelName, modelNameSize);
    });
}

inline std::string Port::Version() const
{
    return QueryStringFromCInterfaceFunction([&](char* version, size_t* versionSize) {
        return PEAK_C_ABI_PREFIX PEAK_Port_GetVersion(m_backendHandle, version, versionSize);
    });
}

inline std::string Port::TLType() const
{
    return QueryStringFromCInterfaceFunction([&](char* tlType, size_t* tlTypeSize) {
        return PEAK_C_ABI_PREFIX PEAK_Port_GetTLType(m_backendHandle, tlType, tlTypeSize);
    });
}

inline std::string Port::ModuleName() const
{
    return QueryStringFromCInterfaceFunction([&](char* moduleName, size_t* moduleNameSize) {
        return PEAK_C_ABI_PREFIX PEAK_Port_GetModuleName(m_backendHandle, moduleName, moduleNameSize);
    });
}

inline Endianness Port::DataEndianness() const
{
    return static_cast<Endianness>(
        QueryNumericFromCInterfaceFunction<PEAK_ENDIANNESS>([&](PEAK_ENDIANNESS* dataEndianness) {
            return PEAK_C_ABI_PREFIX PEAK_Port_GetDataEndianness(m_backendHandle, dataEndianness);
        }));
}

inline bool Port::IsReadable() const
{
    return QueryNumericFromCInterfaceFunction<PEAK_BOOL8>([&](PEAK_BOOL8* isReadable) {
        return PEAK_C_ABI_PREFIX PEAK_Port_GetIsReadable(m_backendHandle, isReadable);
    }) > 0;
}

inline bool Port::IsWritable() const
{
    return QueryNumericFromCInterfaceFunction<PEAK_BOOL8>([&](PEAK_BOOL8* isWritable) {
        return PEAK_C_ABI_PREFIX PEAK_Port_GetIsWritable(m_backendHandle, isWritable);
    }) > 0;
}

inline bool Port::IsAvailable() const
{
    return QueryNumericFromCInterfaceFunction<PEAK_BOOL8>([&](PEAK_BOOL8* isAvailable) {
        return PEAK_C_ABI_PREFIX PEAK_Port_GetIsAvailable(m_backendHandle, isAvailable);
    }) > 0;
}

inline bool Port::IsImplemented() const
{
    return QueryNumericFromCInterfaceFunction<PEAK_BOOL8>([&](PEAK_BOOL8* isImplemented) {
        return PEAK_C_ABI_PREFIX PEAK_Port_GetIsImplemented(m_backendHandle, isImplemented);
    }) > 0;
}

inline std::vector<uint8_t> Port::Read(uint64_t address, size_t numBytes) const
{
    std::vector<uint8_t> bytes(numBytes);
    ExecuteAndMapReturnCodes(
        [&] { return PEAK_C_ABI_PREFIX PEAK_Port_Read(m_backendHandle, address, bytes.data(), bytes.size()); });

    return bytes;
}

inline void Port::Write(uint64_t address, const std::vector<uint8_t>& bytes) const
{
    ExecuteAndMapReturnCodes(
        [&] { return PEAK_C_ABI_PREFIX PEAK_Port_Write(m_backendHandle, address, bytes.data(), bytes.size()); });
}

inline std::vector<std::shared_ptr<PortURL>> Port::URLs() const
{
    InitializeIfNecessary();
    return m_portURLs;
}

inline void Port::Initialize() const
{
    auto numUrls = QueryNumericFromCInterfaceFunction<size_t>(
        [&](size_t* _numUrls) { return PEAK_C_ABI_PREFIX PEAK_Port_GetNumURLs(m_backendHandle, _numUrls); });

    std::vector<std::shared_ptr<PortURL>> urls;
    for (size_t x = 0; x < numUrls; ++x)
    {
        auto portUrlHandle = QueryNumericFromCInterfaceFunction<PEAK_PORT_URL_HANDLE>(
            [&](PEAK_PORT_URL_HANDLE* _portUrlHandle) {
                return PEAK_C_ABI_PREFIX PEAK_Port_GetURL(m_backendHandle, x, _portUrlHandle);
            });

        urls.emplace_back(
            std::make_shared<ClassCreator<PortURL>>(portUrlHandle, std::const_pointer_cast<Port>(shared_from_this())));
    }

    m_portURLs = urls;
}

} /* namespace core */
} /* namespace peak */
