/*!
 * \file    peak_producer_library.hpp
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
#include <peak/generic/peak_init_once.hpp>
#include <peak/system/peak_system_descriptor.hpp>

#include <cstdint>
#include <memory>
#include <string>
#include <vector>


namespace peak
{
namespace core
{

/*!
 * \brief Represents a GenTL producer library (CTI).
 *
 * This class allows to load and initialize a GenTL producer library (CTI) and access its functionality. Each
 * ProducerLibrary contains exactly one SystemDescriptor, which allows you to open the corresponding System.
 */
class ProducerLibrary
    : public InitOnce
    , public std::enable_shared_from_this<ProducerLibrary>
{
public:
    ProducerLibrary() = delete;
    ~ProducerLibrary();
    ProducerLibrary(const ProducerLibrary& other) = delete;
    ProducerLibrary& operator=(const ProducerLibrary& other) = delete;
    ProducerLibrary(ProducerLibrary&& other) = delete;
    ProducerLibrary& operator=(ProducerLibrary&& other) = delete;

    /*!
     * \brief Opens the given producer library (CTI).
     *
     * \param[in] ctiPath The path of the producer library (CTI) to open.
     *
     * \return Producer library
     *
     * \since 1.0
     *
     * \throws InternalErrorException An internal error has occurred.
     */
    static std::shared_ptr<ProducerLibrary> Open(const std::string& ctiPath);

    /*!
     * \brief Returns the unique key.
     *
     * The returned key is unique even across different producer libraries.
     *
     * \return Unique key
     *
     * \since 1.0
     *
     * \throws InternalErrorException An internal error has occurred.
     */
    std::string Key() const;

    /*!
     * \brief Returns the system descriptor.
     *
     * The returned system descriptor can be used to query information about the GenTL system module
     * without opening it and to open the GenTL system module.
     *
     * \return System descriptor
     *
     * \since 1.0
     *
     * \throws InternalErrorException An internal error has occurred.
     */
    std::shared_ptr<SystemDescriptor> System() const;

private:
    friend ClassCreator<ProducerLibrary>;
    explicit ProducerLibrary(PEAK_PRODUCER_LIBRARY_HANDLE producerLibraryHandle);
    PEAK_PRODUCER_LIBRARY_HANDLE m_backendHandle;

    void Initialize() const override;
    mutable std::shared_ptr<SystemDescriptor> m_system;

    std::string m_key;
};

} /* namespace core */
} /* namespace peak */

/* Implementation */
namespace peak
{
namespace core
{

inline std::shared_ptr<ProducerLibrary> ProducerLibrary::Open(const std::string& ctiPath)
{
    auto producerLibraryHandle = QueryNumericFromCInterfaceFunction<PEAK_PRODUCER_LIBRARY_HANDLE>(
        [&](PEAK_PRODUCER_LIBRARY_HANDLE* _producerLibraryHandle) {
            return PEAK_C_ABI_PREFIX PEAK_ProducerLibrary_Construct(
                ctiPath.c_str(), ctiPath.size() + 1, _producerLibraryHandle);
        });

    return std::make_shared<ClassCreator<ProducerLibrary>>(producerLibraryHandle);
}

inline ProducerLibrary::ProducerLibrary(PEAK_PRODUCER_LIBRARY_HANDLE producerLibraryHandle)
    : m_backendHandle(producerLibraryHandle)
    , m_system()
    , m_key(QueryStringFromCInterfaceFunction([&](char* key, size_t* keySize) {
        return PEAK_C_ABI_PREFIX PEAK_ProducerLibrary_GetKey(producerLibraryHandle, key, keySize);
    }))
{}

inline ProducerLibrary::~ProducerLibrary()
{
    (void)PEAK_C_ABI_PREFIX PEAK_ProducerLibrary_Destruct(m_backendHandle);
}

inline std::string ProducerLibrary::Key() const
{
    return m_key;
}

inline std::shared_ptr<SystemDescriptor> ProducerLibrary::System() const
{
    InitializeIfNecessary();

    return m_system;
}

inline void ProducerLibrary::Initialize() const
{
    auto systemDescriptorHandle = QueryNumericFromCInterfaceFunction<PEAK_SYSTEM_DESCRIPTOR_HANDLE>(
        [&](PEAK_SYSTEM_DESCRIPTOR_HANDLE* _systemDescriptorHandle) {
            return PEAK_C_ABI_PREFIX PEAK_ProducerLibrary_GetSystem(m_backendHandle, _systemDescriptorHandle);
        });

    m_system = std::make_shared<ClassCreator<SystemDescriptor>>(
        systemDescriptorHandle, std::const_pointer_cast<ProducerLibrary>(shared_from_this()));
}

} /* namespace core */
} /* namespace peak */
