/*!
 * \file    peak_buffer.hpp
 *
 * \author  IDS Imaging Development Systems GmbH
 * \date    2019-05-01
 * \since   1.0
 *
 * Copyright (c) 2019, IDS Imaging Development Systems GmbH. All rights reserved.
 */

#pragma once


#include <peak/backend/peak_backend.h>
#include <peak/buffer/peak_buffer_chunk.hpp>
#include <peak/buffer/peak_buffer_part.hpp>
#include <peak/common/peak_common_structs.hpp>
#include <peak/common/peak_event_supporting_module.hpp>
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
 * \brief Payload type of buffer.
 *
 * See GenTL PAYLOADTYPE_INFO_IDS.
 */
enum class BufferPayloadType
{
    Unknown,
    Image,
    RawData,
    File,
    Chunk,
    JPEG,
    JPEG2000,
    H264,
    ChunkOnly,
    DeviceSpecific,
    MultiPart,

    Custom = 1000
};

/*!
 * \brief The namespace needed to interpret the pixel format value of the buffer.
 *
 * See GenTL PIXELFORMAT_NAMESPACE_IDS.
 */
enum class PixelFormatNamespace
{
    GEV = 1,
    IIDC,
    PFNC16Bit,
    PFNC32Bit,

    Custom = 1000
};

class DataStream;
class NodeMap;

/*!
 * \brief Represents a GenTL buffer module.
 *
 * This class allows to query information about a GenTL buffer module and to enumerate its chunks or parts.
 */
class Buffer
    : public EventSupportingModule
    , public std::enable_shared_from_this<Buffer>
{
public:
    Buffer() = delete;
    ~Buffer() override = default;
    Buffer(const Buffer& other) = delete;
    Buffer& operator=(const Buffer& other) = delete;
    Buffer(Buffer&& other) = delete;
    Buffer& operator=(Buffer&& other) = delete;

    /*!
     * \anchor BufferInfoCommand
     * \name Info Commands
     * \brief Information based on GenTL BUFFER_INFO_CMD.
     */
    //!\{
    /*! @copydoc SystemDescriptor::Info() */
    RawInformation Info(int32_t infoCommand) const;
    /*! @copydoc SystemDescriptor::TLType() */
    std::string TLType() const;
    /*!
     * \brief Returns the base pointer.
     *
     * \return Base pointer
     *
     * \since 1.0
     *
     * \throws InternalErrorException An internal error has occurred.
     */
    void* BasePtr() const;
    /*!
     * \brief Returns the size of the buffer in bytes.
     *
     * \return Buffer size in bytes
     *
     * \since 1.0
     *
     * \throws InternalErrorException An internal error has occurred.
     */
    size_t Size() const;
    /*!
     * \brief Returns the pointer to custom user data attached to the buffer.
     *
     * Pointer to user data provided at buffer announcement using DataStream::AnnounceBuffer() /
     * DataStream::AllocAndAnnounceBuffer(). This pointer can be used to identify or to attach custom user data to the
     * buffer.
     *
     * \return User pointer
     *
     * \since 1.0
     *
     * \throws InternalErrorException An internal error has occurred.
     */
    void* UserPtr() const;
    /*!
     * \brief Returns the payload type.
     *
     * \return Payload type
     *
     * \since 1.0
     *
     * \throws InternalErrorException An internal error has occurred.
     */
    BufferPayloadType PayloadType() const;
    /*!
     * \brief Returns the pixel format of the data.
     *
     * The interpretation of the pixel format depends on the namespace the pixel format belongs to. This can be
     * inquired using PixelFormatNamespace().
     *
     * \return Pixel format
     *
     * \since 1.0
     *
     * \throws InternalErrorException An internal error has occurred.
     */
    uint64_t PixelFormat() const;
    /*!
     * \brief Returns the pixel format namespace, to allow interpretation of the PixelFormat().
     *
     * \return Pixel format namespace
     *
     * \since 1.0
     *
     * \throws InternalErrorException An internal error has occurred.
     */
    peak::core::PixelFormatNamespace PixelFormatNamespace() const;
    /*!
     * \brief Returns the pixel endianness.
     *
     * \return Pixel endianness
     *
     * \since 1.0
     *
     * \throws InternalErrorException An internal error has occurred.
     */
    Endianness PixelEndianness() const;
    /*!
     * \brief Returns the expected data size.
     *
     * \return Expected data size
     *
     * \since 1.0
     *
     * \throws InternalErrorException An internal error has occurred.
     */
    size_t ExpectedDataSize() const;
    /*!
     * \brief Returns the delivered data size.
     *
     * \return Delivered data size
     *
     * \since 1.0
     *
     * \throws InternalErrorException An internal error has occurred.
     */
    size_t DeliveredDataSize() const;
    /*!
     * \brief Returns the frame ID.
     *
     * \return Frame ID
     *
     * \since 1.0
     *
     * \throws InternalErrorException An internal error has occurred.
     */
    uint64_t FrameID() const;
    /*!
     * \brief Returns the offset of the image data from the beginning of the delivered buffer in bytes.
     *
     * \return Image offset in bytes
     *
     * \since 1.0
     *
     * \throws InternalErrorException An internal error has occurred.
     */
    size_t ImageOffset() const;
    /*!
     * \brief Returns the delivered image height.
     *
     * \return Delivered image height
     *
     * \since 1.0
     *
     * \throws InternalErrorException An internal error has occurred.
     */
    size_t DeliveredImageHeight() const;
    /*!
     * \brief Returns the delivered chunk payload size.
     *
     * \return Delivered chunk payload size
     *
     * \since 1.0
     *
     * \throws InternalErrorException An internal error has occurred.
     */
    size_t DeliveredChunkPayloadSize() const;
    /*!
     * \brief Returns the chunk layout ID.
     *
     * \return Chunk layout ID
     *
     * \since 1.0
     *
     * \throws InternalErrorException An internal error has occurred.
     */
    uint64_t ChunkLayoutID() const;
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
     * \brief Returns the width.
     *
     * \return Width
     *
     * \since 1.0
     *
     * \throws InternalErrorException An internal error has occurred.
     */
    size_t Width() const;
    /*!
     * \brief Returns the height.
     *
     * \return Height
     *
     * \since 1.0
     *
     * \throws InternalErrorException An internal error has occurred.
     */
    size_t Height() const;
    /*!
     * \brief Returns the x offset of the data in the buffer in number of pixels from the image origin to handle
     * areas of interest.
     *
     * \return X offset in number of pixels
     *
     * \since 1.0
     *
     * \throws InternalErrorException An internal error has occurred.
     */
    size_t XOffset() const;
    /*!
     * \brief Returns the y offset of the data in the buffer in number of lines from the image origin to handle
     * areas of interest.
     *
     * \return Y offset in number of lines
     *
     * \since 1.0
     *
     * \throws InternalErrorException An internal error has occurred.
     */
    size_t YOffset() const;
    /*!
     * \brief Returns the x padding of the data in the buffer in number of bytes.
     *
     * \return X padding in bytes
     *
     * \since 1.0
     *
     * \throws InternalErrorException An internal error has occurred.
     */
    size_t XPadding() const;
    /*!
     * \brief Returns the y padding of the data in the buffer in number of bytes.
     *
     * \return Y padding in bytes
     *
     * \since 1.0
     *
     * \throws InternalErrorException An internal error has occurred.
     */
    size_t YPadding() const;
    /*!
     * \brief Returns the timestamp.
     *
     * \return Timestamp in ticks
     *
     * \since 1.0
     *
     * \throws InternalErrorException An internal error has occurred.
     */
    uint64_t Timestamp_ticks() const;
    /*!
     * \brief Returns the timestamp.
     *
     * \return Timestamp in nanoseconds
     *
     * \since 1.0
     *
     * \throws InternalErrorException An internal error has occurred.
     */
    uint64_t Timestamp_ns() const;
    /*!
     * \brief Checks whether the buffer is queued  or not.
     *
     * \return True, while the buffer is in the input pool, is currently being filled or is in the output buffer queue.
     * \return False otherwise.
     *
     * \since 1.0
     *
     * \throws InternalErrorException An internal error has occurred.
     */
    bool IsQueued() const;
    /*!
     * \brief Checks whether the buffer is currently being filled.
     *
     * \return True, while the buffer is being filled.
     * \return False otherwise.
     *
     *
     * \since 1.0
     *
     * \throws InternalErrorException An internal error has occurred.
     */
    bool IsAcquiring() const;
    /*!
     * \brief Checks whether the buffer is incomplete or not.
     *
     * Incomplete buffers can happen when an error occurred while the buffer was being filled.
     *
     * \return True if the buffer couldn't be filled completely.
     * \return False if the buffer is complete.
     *
     * \since 1.0
     *
     * \throws InternalErrorException An internal error has occurred.
     */
    bool IsIncomplete() const;
    /*!
     * \brief Checks whether the buffer has new data since the last delivery.
     *
     * \return True when the buffer has new data since the last delivery.
     * \return False otherwise.
     *
     * \since 1.0
     *
     * \throws InternalErrorException An internal error has occurred.
     */
    bool HasNewData() const;
    /*!
     * \brief Checks whether the buffer contains image data.
     *
     * \return True, if the buffer contains an image.
     * \return False otherwise.
     *
     * See GenTL BUFFER_INFO_IMAGEPRESENT.
     *
     * \since 1.0
     *
     * \throws InternalErrorException An internal error has occurred.
     */
    bool HasImage() const;
    /*!
     * \brief Checks whether the buffer contains chunks.
     *
     * If HasChunks() is true, check the BufferChunks using Chunks().
     *
     * \return True, if the buffer has chunks.
     * \return False otherwise.
     *
     * \since 1.0
     *
     * \throws InternalErrorException An internal error has occurred.
     */
    bool HasChunks() const;
    //!\}

    /*!
     * \brief Returns the buffer's chunks if it contains chunks.
     *
     * Check if the buffer has chunks with HasChunks().
     *
     * \return Buffer chunks
     *
     * \since 1.0
     *
     * \throws InternalErrorException An internal error has occurred.
     */
    std::vector<std::shared_ptr<BufferChunk>> Chunks();
    /*!
     * \brief Returns all buffer parts for multipart buffers.
     *
     * \return Buffer parts
     *
     * \since 1.0
     *
     * \throws InternalErrorException An internal error has occurred.
     */
    std::vector<std::shared_ptr<BufferPart>> Parts();
    /*!
     * \brief Returns the parent data stream.
     *
     * \return Parent data stream
     *
     * \since 1.0
     *
     * \throws InternalErrorException An internal error has occurred.
     */
    std::shared_ptr<DataStream> ParentDataStream() const;

private:
    PEAK_MODULE_HANDLE ModuleHandle() const override;
    PEAK_EVENT_SUPPORTING_MODULE_HANDLE EventSupportingModuleHandle() const override;

    void Revoke();

    friend NodeMap;
    friend ClassCreator<Buffer>;
    Buffer(PEAK_BUFFER_HANDLE bufferHandle, const std::weak_ptr<DataStream>& parentDataStream);
    PEAK_BUFFER_HANDLE m_backendHandle;

    std::weak_ptr<DataStream> m_parentDataStream;

    friend class DataStream;
};

} /* namespace core */
} /* namespace peak */

/* Implementation */
namespace peak
{
namespace core
{

inline std::string ToString(BufferPayloadType entry)
{
    std::string entryString;

    if (entry == BufferPayloadType::Unknown)
    {
        entryString = "Unknown";
    }
    else if (entry == BufferPayloadType::Image)
    {
        entryString = "Image";
    }
    else if (entry == BufferPayloadType::RawData)
    {
        entryString = "RawData";
    }
    else if (entry == BufferPayloadType::File)
    {
        entryString = "File";
    }
    else if (entry == BufferPayloadType::Chunk)
    {
        entryString = "Chunk";
    }
    else if (entry == BufferPayloadType::JPEG)
    {
        entryString = "JPEG";
    }
    else if (entry == BufferPayloadType::JPEG2000)
    {
        entryString = "JPEG2000";
    }
    else if (entry == BufferPayloadType::H264)
    {
        entryString = "H264";
    }
    else if (entry == BufferPayloadType::ChunkOnly)
    {
        entryString = "ChunkOnly";
    }
    else if (entry == BufferPayloadType::DeviceSpecific)
    {
        entryString = "DeviceSpecific";
    }
    else if (entry == BufferPayloadType::MultiPart)
    {
        entryString = "MultiPart";
    }
    else if (entry >= BufferPayloadType::Custom)
    {
        entryString = "Custom";
    }

    return entryString;
}

inline std::string ToString(PixelFormatNamespace entry)
{
    std::string entryString;

    if (entry == PixelFormatNamespace::GEV)
    {
        entryString = "GEV";
    }
    else if (entry == PixelFormatNamespace::IIDC)
    {
        entryString = "IIDC";
    }
    else if (entry == PixelFormatNamespace::PFNC16Bit)
    {
        entryString = "PFNC16Bit";
    }
    else if (entry == PixelFormatNamespace::PFNC32Bit)
    {
        entryString = "PFNC32Bit";
    }
    else if (entry >= PixelFormatNamespace::Custom)
    {
        entryString = "Custom";
    }

    return entryString;
}

inline Buffer::Buffer(PEAK_BUFFER_HANDLE bufferHandle, const std::weak_ptr<DataStream>& parentDataStream)
    : m_backendHandle(bufferHandle)
    , m_parentDataStream(parentDataStream)
{}

inline RawInformation Buffer::Info(int32_t infoCommand) const
{
    return QueryRawInformationFromCInterfaceFunction([&](int32_t* dataType, uint8_t* buffer, size_t* bufferSize) {
        return PEAK_C_ABI_PREFIX PEAK_Buffer_GetInfo(m_backendHandle, infoCommand, dataType, buffer, bufferSize);
    });
}

inline std::string Buffer::TLType() const
{
    return QueryStringFromCInterfaceFunction([&](char* tlType, size_t* tlTypeSize) {
        return PEAK_C_ABI_PREFIX PEAK_Buffer_GetTLType(m_backendHandle, tlType, tlTypeSize);
    });
}

inline void* Buffer::BasePtr() const
{
    return QueryNumericFromCInterfaceFunction<void*>(
        [&](void** basePtr) { return PEAK_C_ABI_PREFIX PEAK_Buffer_GetBasePtr(m_backendHandle, basePtr); });
}

inline size_t Buffer::Size() const
{
    return QueryNumericFromCInterfaceFunction<size_t>(
        [&](size_t* size) { return PEAK_C_ABI_PREFIX PEAK_Buffer_GetSize(m_backendHandle, size); });
}

inline void* Buffer::UserPtr() const
{
    return QueryNumericFromCInterfaceFunction<void*>(
        [&](void** userPtr) { return PEAK_C_ABI_PREFIX PEAK_Buffer_GetUserPtr(m_backendHandle, userPtr); });
}

inline BufferPayloadType Buffer::PayloadType() const
{
    return static_cast<BufferPayloadType>(
        QueryNumericFromCInterfaceFunction<PEAK_BUFFER_PAYLOAD_TYPE>([&](PEAK_BUFFER_PAYLOAD_TYPE* payloadType) {
            return PEAK_C_ABI_PREFIX PEAK_Buffer_GetPayloadType(m_backendHandle, payloadType);
        }));
}

inline uint64_t Buffer::PixelFormat() const
{
    return QueryNumericFromCInterfaceFunction<uint64_t>([&](uint64_t* pixelFormat) {
        return PEAK_C_ABI_PREFIX PEAK_Buffer_GetPixelFormat(m_backendHandle, pixelFormat);
    });
}

inline peak::core::PixelFormatNamespace Buffer::PixelFormatNamespace() const
{
    return static_cast<peak::core::PixelFormatNamespace>(
        QueryNumericFromCInterfaceFunction<PEAK_PIXEL_FORMAT_NAMESPACE>(
            [&](PEAK_PIXEL_FORMAT_NAMESPACE* pixelFormatNamespace) {
                return PEAK_C_ABI_PREFIX PEAK_Buffer_GetPixelFormatNamespace(
                    m_backendHandle, pixelFormatNamespace);
            }));
}

inline Endianness Buffer::PixelEndianness() const
{
    return static_cast<Endianness>(
        QueryNumericFromCInterfaceFunction<PEAK_ENDIANNESS>([&](PEAK_ENDIANNESS* pixelEndianness) {
            return PEAK_C_ABI_PREFIX PEAK_Buffer_GetPixelEndianness(m_backendHandle, pixelEndianness);
        }));
}

inline size_t Buffer::ExpectedDataSize() const
{
    return QueryNumericFromCInterfaceFunction<size_t>([&](size_t* expectedDataSize) {
        return PEAK_C_ABI_PREFIX PEAK_Buffer_GetExpectedDataSize(m_backendHandle, expectedDataSize);
    });
}

inline size_t Buffer::DeliveredDataSize() const
{
    return QueryNumericFromCInterfaceFunction<size_t>([&](size_t* deliveredDataSize) {
        return PEAK_C_ABI_PREFIX PEAK_Buffer_GetDeliveredDataSize(m_backendHandle, deliveredDataSize);
    });
}

inline uint64_t Buffer::FrameID() const
{
    return QueryNumericFromCInterfaceFunction<uint64_t>(
        [&](uint64_t* frameId) { return PEAK_C_ABI_PREFIX PEAK_Buffer_GetFrameID(m_backendHandle, frameId); });
}

inline size_t Buffer::ImageOffset() const
{
    return QueryNumericFromCInterfaceFunction<size_t>([&](size_t* imageOffset) {
        return PEAK_C_ABI_PREFIX PEAK_Buffer_GetImageOffset(m_backendHandle, imageOffset);
    });
}

inline size_t Buffer::DeliveredImageHeight() const
{
    return QueryNumericFromCInterfaceFunction<size_t>([&](size_t* deliveredImageHeight) {
        return PEAK_C_ABI_PREFIX PEAK_Buffer_GetDeliveredImageHeight(m_backendHandle, deliveredImageHeight);
    });
}

inline size_t Buffer::DeliveredChunkPayloadSize() const
{
    return QueryNumericFromCInterfaceFunction<size_t>([&](size_t* deliveredChunkPayloadSize) {
        return PEAK_C_ABI_PREFIX PEAK_Buffer_GetDeliveredChunkPayloadSize(
            m_backendHandle, deliveredChunkPayloadSize);
    });
}

inline uint64_t Buffer::ChunkLayoutID() const
{
    return QueryNumericFromCInterfaceFunction<uint64_t>([&](uint64_t* chunkLayoutId) {
        return PEAK_C_ABI_PREFIX PEAK_Buffer_GetChunkLayoutID(m_backendHandle, chunkLayoutId);
    });
}

inline std::string Buffer::FileName() const
{
    return QueryStringFromCInterfaceFunction([&](char* fileName, size_t* fileNameSize) {
        return PEAK_C_ABI_PREFIX PEAK_Buffer_GetFileName(m_backendHandle, fileName, fileNameSize);
    });
}

inline size_t Buffer::Width() const
{
    return QueryNumericFromCInterfaceFunction<size_t>(
        [&](size_t* width) { return PEAK_C_ABI_PREFIX PEAK_Buffer_GetWidth(m_backendHandle, width); });
}

inline size_t Buffer::Height() const
{
    return QueryNumericFromCInterfaceFunction<size_t>(
        [&](size_t* height) { return PEAK_C_ABI_PREFIX PEAK_Buffer_GetHeight(m_backendHandle, height); });
}

inline size_t Buffer::XOffset() const
{
    return QueryNumericFromCInterfaceFunction<size_t>(
        [&](size_t* xOffset) { return PEAK_C_ABI_PREFIX PEAK_Buffer_GetXOffset(m_backendHandle, xOffset); });
}

inline size_t Buffer::YOffset() const
{
    return QueryNumericFromCInterfaceFunction<size_t>(
        [&](size_t* yOffset) { return PEAK_C_ABI_PREFIX PEAK_Buffer_GetYOffset(m_backendHandle, yOffset); });
}

inline size_t Buffer::XPadding() const
{
    return QueryNumericFromCInterfaceFunction<size_t>(
        [&](size_t* xPadding) { return PEAK_C_ABI_PREFIX PEAK_Buffer_GetXPadding(m_backendHandle, xPadding); });
}

inline size_t Buffer::YPadding() const
{
    return QueryNumericFromCInterfaceFunction<size_t>(
        [&](size_t* yPadding) { return PEAK_C_ABI_PREFIX PEAK_Buffer_GetYPadding(m_backendHandle, yPadding); });
}

inline uint64_t Buffer::Timestamp_ticks() const
{
    return QueryNumericFromCInterfaceFunction<uint64_t>([&](uint64_t* timestamp_ticks) {
        return PEAK_C_ABI_PREFIX PEAK_Buffer_GetTimestamp_ticks(m_backendHandle, timestamp_ticks);
    });
}

inline uint64_t Buffer::Timestamp_ns() const
{
    return QueryNumericFromCInterfaceFunction<uint64_t>([&](uint64_t* timestamp_ns) {
        return PEAK_C_ABI_PREFIX PEAK_Buffer_GetTimestamp_ns(m_backendHandle, timestamp_ns);
    });
}

inline bool Buffer::IsQueued() const
{
    return QueryNumericFromCInterfaceFunction<PEAK_BOOL8>([&](PEAK_BOOL8* isQueued) {
        return PEAK_C_ABI_PREFIX PEAK_Buffer_GetIsQueued(m_backendHandle, isQueued);
    }) > 0;
}

inline bool Buffer::IsAcquiring() const
{
    return QueryNumericFromCInterfaceFunction<PEAK_BOOL8>([&](PEAK_BOOL8* isAcquiring) {
        return PEAK_C_ABI_PREFIX PEAK_Buffer_GetIsAcquiring(m_backendHandle, isAcquiring);
    }) > 0;
}

inline bool Buffer::IsIncomplete() const
{
    return QueryNumericFromCInterfaceFunction<PEAK_BOOL8>([&](PEAK_BOOL8* isIncomplete) {
        return PEAK_C_ABI_PREFIX PEAK_Buffer_GetIsIncomplete(m_backendHandle, isIncomplete);
    }) > 0;
}

inline bool Buffer::HasNewData() const
{
    return QueryNumericFromCInterfaceFunction<PEAK_BOOL8>([&](PEAK_BOOL8* hasNewData) {
        return PEAK_C_ABI_PREFIX PEAK_Buffer_GetHasNewData(m_backendHandle, hasNewData);
    }) > 0;
}

inline bool Buffer::HasImage() const
{
    return QueryNumericFromCInterfaceFunction<PEAK_BOOL8>([&](PEAK_BOOL8* hasImage) {
        return PEAK_C_ABI_PREFIX PEAK_Buffer_GetHasImage(m_backendHandle, hasImage);
    }) > 0;
}

inline bool Buffer::HasChunks() const
{
    return QueryNumericFromCInterfaceFunction<PEAK_BOOL8>([&](PEAK_BOOL8* hasChunks) {
        return PEAK_C_ABI_PREFIX PEAK_Buffer_GetHasChunks(m_backendHandle, hasChunks);
    }) > 0;
}

inline std::vector<std::shared_ptr<BufferChunk>> Buffer::Chunks()
{
    CallAndCheckCInterfaceFunction([&] { return PEAK_C_ABI_PREFIX PEAK_Buffer_UpdateChunks(m_backendHandle); });

    auto numChunks = QueryNumericFromCInterfaceFunction<size_t>([&](size_t* _numChunks) {
        return PEAK_C_ABI_PREFIX PEAK_Buffer_GetNumChunks(m_backendHandle, _numChunks);
    });

    std::vector<std::shared_ptr<BufferChunk>> chunks;
    for (size_t x = 0; x < numChunks; ++x)
    {
        auto bufferChunkHandle = QueryNumericFromCInterfaceFunction<PEAK_BUFFER_CHUNK_HANDLE>(
            [&](PEAK_BUFFER_CHUNK_HANDLE* _bufferChunkHandle) {
                return PEAK_C_ABI_PREFIX PEAK_Buffer_GetChunk(m_backendHandle, x, _bufferChunkHandle);
            });

        chunks.emplace_back(std::make_shared<ClassCreator<BufferChunk>>(bufferChunkHandle, shared_from_this()));
    }

    return chunks;
}

inline std::vector<std::shared_ptr<BufferPart>> Buffer::Parts()
{
    CallAndCheckCInterfaceFunction([&] { return PEAK_C_ABI_PREFIX PEAK_Buffer_UpdateParts(m_backendHandle); });

    auto numParts = QueryNumericFromCInterfaceFunction<size_t>(
        [&](size_t* _numParts) { return PEAK_C_ABI_PREFIX PEAK_Buffer_GetNumParts(m_backendHandle, _numParts); });

    std::vector<std::shared_ptr<BufferPart>> parts;
    for (size_t x = 0; x < numParts; ++x)
    {
        auto bufferPartHandle = QueryNumericFromCInterfaceFunction<PEAK_BUFFER_PART_HANDLE>(
            [&](PEAK_BUFFER_PART_HANDLE* _bufferPartHandle) {
                return PEAK_C_ABI_PREFIX PEAK_Buffer_GetPart(m_backendHandle, x, _bufferPartHandle);
            });

        parts.emplace_back(std::make_shared<ClassCreator<BufferPart>>(bufferPartHandle, shared_from_this()));
    }

    return parts;
}

inline std::shared_ptr<DataStream> Buffer::ParentDataStream() const
{
    return LockOrThrow(m_parentDataStream);
}

inline PEAK_MODULE_HANDLE Buffer::ModuleHandle() const
{
    auto moduleHandle = QueryNumericFromCInterfaceFunction<PEAK_MODULE_HANDLE>(
        [&](PEAK_MODULE_HANDLE* _moduleHandle) {
            return PEAK_C_ABI_PREFIX PEAK_Buffer_ToModule(m_backendHandle, _moduleHandle);
        });

    return moduleHandle;
}

inline PEAK_EVENT_SUPPORTING_MODULE_HANDLE Buffer::EventSupportingModuleHandle() const
{
    auto eventSupportingModuleHandle = QueryNumericFromCInterfaceFunction<PEAK_EVENT_SUPPORTING_MODULE_HANDLE>(
        [&](PEAK_EVENT_SUPPORTING_MODULE_HANDLE* _eventSupportingModuleHandle) {
            return PEAK_C_ABI_PREFIX PEAK_Buffer_ToEventSupportingModule(
                m_backendHandle, _eventSupportingModuleHandle);
        });

    return eventSupportingModuleHandle;
}

inline void Buffer::Revoke()
{
    m_parentDataStream.reset();
}

inline bool NodeMap::HasBufferSupportedChunks(const std::shared_ptr<Buffer>& buffer) const
{
    return QueryNumericFromCInterfaceFunction<PEAK_BOOL8>([&](PEAK_BOOL8* hasSupportedChunks) {
        return PEAK_C_ABI_PREFIX PEAK_NodeMap_GetHasBufferSupportedChunks(
            m_backendHandle, buffer->m_backendHandle, hasSupportedChunks);
    }) > 0;
}

inline void NodeMap::UpdateChunkNodes(const std::shared_ptr<Buffer>& buffer)
{
    CallAndCheckCInterfaceFunction([&] {
        return PEAK_C_ABI_PREFIX PEAK_NodeMap_UpdateChunkNodes(m_backendHandle, buffer->m_backendHandle);
    });
}

} /* namespace core */
} /* namespace peak */
