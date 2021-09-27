/*!
 * \file    peak_port_url.hpp
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

/*!
 * \brief Possible URL schemes for the port.
 *
 * See GenTL URL_SCHEME_IDS.
 */
enum class PortURLScheme
{
    Local,
    HTTP,
    File,

    Custom = 1000
};

class Port;

/*!
 * \brief Represents a GenTL port URL.
 *
 * This class allows to query information about a GenTL port URL.
 *
 */
class PortURL
{
public:
    PortURL() = delete;
    ~PortURL() = default;
    PortURL(const PortURL& other) = delete;
    PortURL& operator=(const PortURL& other) = delete;
    PortURL(PortURL&& other) = delete;
    PortURL& operator=(PortURL&& other) = delete;

    /*! @copydoc SystemDescriptor::Info() */
    RawInformation Info(int32_t infoCommand) const;
    /*!
     * \brief Returns the URL.
     *
     * \return URL
     *
     * \since 1.0
     *
     * \throws InternalErrorException An internal error has occurred.
     */
    std::string URL() const;
    /*!
     * \brief Returns the scheme.
     *
     * \return Scheme
     *
     * \since 1.0
     *
     * \throws InternalErrorException An internal error has occurred.
     */
    PortURLScheme Scheme() const;
    /*!
     * \brief Returns the file name.
     *
     * \return File name
     *
     * \since 1.0
     *
     * \throws InternalErrorException An internal error has occurred.
     */
    std::string FileName() const;
    /*!
     * \brief Returns the file register address.
     *
     * \return File register address
     *
     * \since 1.0
     *
     * \throws InternalErrorException An internal error has occurred.
     */
    uint64_t FileRegisterAddress() const;
    /*!
     * \brief Returns the file size.
     *
     * \return File size
     *
     * \since 1.0
     *
     * \throws InternalErrorException An internal error has occurred.
     */
    uint64_t FileSize() const;
    /*!
     * \brief Returns the file SHA1 hash.
     *
     * \return File SHA1 hash
     *
     * \since 1.0
     *
     * \throws InternalErrorException An internal error has occurred.
     */
    std::vector<uint8_t> FileSHA1Hash() const;
    /*!
     * \brief Returns the file major version.
     *
     * \return File major version
     *
     * \since 1.0
     *
     * \throws InternalErrorException An internal error has occurred.
     */
    int32_t FileVersionMajor() const;
    /*!
     * \brief Returns the file major version.
     *
     * \return File major version
     *
     * \since 1.0
     *
     * \throws InternalErrorException An internal error has occurred.
     */
    int32_t FileVersionMinor() const;
    /*!
     * \brief Returns the file subminor version.
     *
     * \return File subminor version
     *
     * \since 1.0
     *
     * \throws InternalErrorException An internal error has occurred.
     */
    int32_t FileVersionSubminor() const;
    /*!
     * \brief Returns the file schema major version.
     *
     * \return File schema major version
     *
     * \since 1.0
     *
     * \throws InternalErrorException An internal error has occurred.
     */
    int32_t FileSchemaVersionMajor() const;
    /*!
     * \brief Returns the file schema minor version.
     *
     * \return File schema minor version
     *
     * \since 1.0
     *
     * \throws InternalErrorException An internal error has occurred.
     */
    int32_t FileSchemaVersionMinor() const;

    /*!
     * \brief Returns the parent port.
     *
     * \return Parent port
     *
     * \since 1.0
     *
     * \throws InternalErrorException An internal error has occurred.
     */
    std::shared_ptr<class Port> ParentPort() const;

private:
    friend ClassCreator<PortURL>;
    explicit PortURL(PEAK_PORT_URL_HANDLE portUrlHandle, const std::weak_ptr<Port>& parentPort);
    PEAK_PORT_URL_HANDLE m_backendHandle;

    std::weak_ptr<Port> m_parentPort;
};

} /* namespace core */
} /* namespace peak */

#include <peak/port/peak_port.hpp>


/* Implementation */
namespace peak
{
namespace core
{

inline PortURL::PortURL(PEAK_PORT_URL_HANDLE portUrlHandle, const std::weak_ptr<Port>& parentPort)
    : m_backendHandle(portUrlHandle)
    , m_parentPort(parentPort)
{}

inline RawInformation PortURL::Info(int32_t infoCommand) const
{
    return QueryRawInformationFromCInterfaceFunction([&](int32_t* dataType, uint8_t* buffer, size_t* bufferSize) {
        return PEAK_C_ABI_PREFIX PEAK_PortURL_GetInfo(m_backendHandle, infoCommand, dataType, buffer, bufferSize);
    });
}

inline std::string PortURL::URL() const
{
    return QueryStringFromCInterfaceFunction([&](char* url, size_t* urlSize) {
        return PEAK_C_ABI_PREFIX PEAK_PortURL_GetURL(m_backendHandle, url, urlSize);
    });
}

inline PortURLScheme PortURL::Scheme() const
{
    return static_cast<PortURLScheme>(
        QueryNumericFromCInterfaceFunction<PEAK_PORT_URL_SCHEME>([&](PEAK_PORT_URL_SCHEME* scheme) {
            return PEAK_C_ABI_PREFIX PEAK_PortURL_GetScheme(m_backendHandle, scheme);
        }));
}

inline std::string PortURL::FileName() const
{
    return QueryStringFromCInterfaceFunction([&](char* fileName, size_t* fileNameSize) {
        return PEAK_C_ABI_PREFIX PEAK_PortURL_GetFileName(m_backendHandle, fileName, fileNameSize);
    });
}

inline uint64_t PortURL::FileRegisterAddress() const
{
    return QueryNumericFromCInterfaceFunction<uint64_t>([&](uint64_t* fileRegisterAddress) {
        return PEAK_C_ABI_PREFIX PEAK_PortURL_GetFileRegisterAddress(m_backendHandle, fileRegisterAddress);
    });
}

inline uint64_t PortURL::FileSize() const
{
    return QueryNumericFromCInterfaceFunction<uint64_t>([&](uint64_t* fileSize) {
        return PEAK_C_ABI_PREFIX PEAK_PortURL_GetFileSize(m_backendHandle, fileSize);
    });
}

inline std::vector<uint8_t> PortURL::FileSHA1Hash() const
{
    return QueryNumericArrayFromCInterfaceFunction<uint8_t>([&](uint8_t* fileSha1Hash, size_t* fileSha1HashSize) {
        return PEAK_C_ABI_PREFIX PEAK_PortURL_GetFileSHA1Hash(m_backendHandle, fileSha1Hash, fileSha1HashSize);
    });
}

inline int32_t PortURL::FileVersionMajor() const
{
    return QueryNumericFromCInterfaceFunction<int32_t>([&](int32_t* fileVersionMajor) {
        return PEAK_C_ABI_PREFIX PEAK_PortURL_GetFileVersionMajor(m_backendHandle, fileVersionMajor);
    });
}

inline int32_t PortURL::FileVersionMinor() const
{
    return QueryNumericFromCInterfaceFunction<int32_t>([&](int32_t* fileVersionMinor) {
        return PEAK_C_ABI_PREFIX PEAK_PortURL_GetFileVersionMinor(m_backendHandle, fileVersionMinor);
    });
}

inline int32_t PortURL::FileVersionSubminor() const
{
    return QueryNumericFromCInterfaceFunction<int32_t>([&](int32_t* fileVersionSubminor) {
        return PEAK_C_ABI_PREFIX PEAK_PortURL_GetFileVersionSubminor(m_backendHandle, fileVersionSubminor);
    });
}

inline int32_t PortURL::FileSchemaVersionMajor() const
{
    return QueryNumericFromCInterfaceFunction<int32_t>([&](int32_t* fileSchemaVersionMajor) {
        return PEAK_C_ABI_PREFIX PEAK_PortURL_GetFileVersionMajor(m_backendHandle, fileSchemaVersionMajor);
    });
}

inline int32_t PortURL::FileSchemaVersionMinor() const
{
    return QueryNumericFromCInterfaceFunction<int32_t>([&](int32_t* fileSchemaVersionMinor) {
        return PEAK_C_ABI_PREFIX PEAK_PortURL_GetFileVersionMinor(m_backendHandle, fileSchemaVersionMinor);
    });
}

inline std::shared_ptr<Port> PortURL::ParentPort() const
{
    return m_parentPort.lock();
}

inline std::string ToString(PortURLScheme entry)
{
    std::string entryString;

    if (entry == PortURLScheme::Local)
    {
        entryString = "Local";
    }
    else if (entry == PortURLScheme::HTTP)
    {
        entryString = "HTTP";
    }
    else if (entry == PortURLScheme::File)
    {
        entryString = "File";
    }
    else if (entry >= PortURLScheme::Custom)
    {
        entryString = "Custom";
    }

    return entryString;
}

} /* namespace core */
} /* namespace peak */
