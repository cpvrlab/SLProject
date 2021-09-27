/*!
 * \file    peak_data_stream_descriptor.hpp
 *
 * \author  IDS Imaging Development Systems GmbH
 * \date    2019-05-01
 * \since   1.0
 *
 * Copyright (c) 2019, IDS Imaging Development Systems GmbH. All rights reserved.
 */

#pragma once


#include <peak/backend/peak_backend.h>
#include <peak/common/peak_module_descriptor.hpp>
#include <peak/data_stream/peak_data_stream.hpp>
#include <peak/dll_interface/peak_dll_interface_util.hpp>

#include <memory>
#include <string>


namespace peak
{
namespace core
{

class Device;

/*!
 * \brief Encapsulates the GenTL functions associated with a GenTL DataStream module's ID.
 *
 * This class allows to query information about a GenTL DataStream module without opening it.
 * Furthermore, it allows to open this GenTL DataStream module.
 *
 */
class DataStreamDescriptor : public ModuleDescriptor
{
public:
    DataStreamDescriptor() = delete;
    ~DataStreamDescriptor() override = default;
    DataStreamDescriptor(const DataStreamDescriptor& other) = delete;
    DataStreamDescriptor& operator=(const DataStreamDescriptor& other) = delete;
    DataStreamDescriptor(DataStreamDescriptor&& other) = delete;
    DataStreamDescriptor& operator=(DataStreamDescriptor&& other) = delete;

    /*! @copydoc ProducerLibrary::Key() */
    std::string Key() const;

    /*!
     * \brief Returns the parent device.
     *
     * \return Parent device
     *
     * \since 1.0
     *
     * \throws InternalErrorException An internal error has occurred.
     */
    std::shared_ptr<Device> ParentDevice() const;
    /*!
     * \brief Opens the data stream.
     *
     * \return Opened data stream
     *
     * \since 1.0
     *
     * \throws BadAccessException Access denied
     * \throws InternalErrorException An internal error has occurred.
     */
    std::shared_ptr<DataStream> OpenDataStream();
    /*!
     * \brief Returns the DataStream that was opened with this DataStreamDescriptor.
     *
     * \return Opened DataStream
     *
     * \since 1.0
     *
     * \throws BadAccessException DataStream is not open
     */
    std::shared_ptr<DataStream> OpenedDataStream() const;

private:
    PEAK_MODULE_DESCRIPTOR_HANDLE ModuleDescriptorHandle() const override;

    friend ClassCreator<DataStreamDescriptor>;
    DataStreamDescriptor(
        PEAK_DATA_STREAM_DESCRIPTOR_HANDLE dataStreamDescriptorHandle, const std::weak_ptr<Device>& parentDevice);
    PEAK_DATA_STREAM_DESCRIPTOR_HANDLE m_backendHandle;

    std::weak_ptr<Device> m_parentDevice;
    std::weak_ptr<DataStream> m_openedDataStream;

    std::string m_key;
};

} /* namespace core */
} /* namespace peak */

/* Implementation */
namespace peak
{
namespace core
{

inline DataStreamDescriptor::DataStreamDescriptor(
    PEAK_DATA_STREAM_DESCRIPTOR_HANDLE dataStreamDescriptorHandle, const std::weak_ptr<Device>& parentDevice)
    : m_backendHandle(dataStreamDescriptorHandle)
    , m_parentDevice(parentDevice)
    , m_key(QueryStringFromCInterfaceFunction([&](char* key, size_t* keySize) {
        return PEAK_C_ABI_PREFIX PEAK_DataStreamDescriptor_GetKey(dataStreamDescriptorHandle, key, keySize);
    }))
{}

inline std::string DataStreamDescriptor::Key() const
{
    return m_key;
}

inline std::shared_ptr<Device> DataStreamDescriptor::ParentDevice() const
{
    return LockOrThrow(m_parentDevice);
}

inline std::shared_ptr<DataStream> DataStreamDescriptor::OpenDataStream()
{
    auto dataStreamHandle = QueryNumericFromCInterfaceFunction<PEAK_DATA_STREAM_HANDLE>(
        [&](PEAK_DATA_STREAM_HANDLE* _dataStreamHandle) {
            return PEAK_C_ABI_PREFIX PEAK_DataStreamDescriptor_OpenDataStream(m_backendHandle, _dataStreamHandle);
        });

    auto dataStream = std::make_shared<ClassCreator<DataStream>>(dataStreamHandle, m_parentDevice);
    m_openedDataStream = dataStream;
    return dataStream;
}

inline std::shared_ptr<DataStream> DataStreamDescriptor::OpenedDataStream() const
{
    return LockOrThrowOpenedModule(m_openedDataStream);
}

inline PEAK_MODULE_DESCRIPTOR_HANDLE DataStreamDescriptor::ModuleDescriptorHandle() const
{
    auto moduleDescriptorHandle = QueryNumericFromCInterfaceFunction<PEAK_MODULE_DESCRIPTOR_HANDLE>(
        [&](PEAK_MODULE_DESCRIPTOR_HANDLE* _moduleDescriptorHandle) {
            return PEAK_C_ABI_PREFIX PEAK_DataStreamDescriptor_ToModuleDescriptor(
                m_backendHandle, _moduleDescriptorHandle);
        });

    return moduleDescriptorHandle;
}

} /* namespace core */
} /* namespace peak */
