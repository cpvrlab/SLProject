/*!
 * \file    peak_buffer_chunk.hpp
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
#include <cstdint>
#include <memory>


namespace peak
{
namespace core
{

class Buffer;

/*!
 * \brief Represents a buffer chunk.
 *
 * This class allows to query information about a buffer chunk.
 *
 */
class BufferChunk
{
public:
    BufferChunk() = delete;
    ~BufferChunk() = default;
    BufferChunk(const BufferChunk& other) = delete;
    BufferChunk& operator=(const BufferChunk& other) = delete;
    BufferChunk(BufferChunk&& other) = delete;
    BufferChunk& operator=(BufferChunk&& other) = delete;

    /*!
     * \brief Returns the ID.
     *
     * \return ID
     *
     * \since 1.0
     *
     * \throws InternalErrorException An internal error has occurred.
     */
    uint64_t ID() const;
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
     * \brief Returns the size.
     *
     * \return Size
     *
     * \since 1.0
     *
     * \throws InternalErrorException An internal error has occurred.
     */
    size_t Size() const;

    /*!
     * \brief Returns the parent buffer.
     *
     * \return Parent buffer
     *
     * \since 1.0
     *
     * \throws InternalErrorException An internal error has occurred.
     */
    std::shared_ptr<Buffer> ParentBuffer() const;

private:
    friend ClassCreator<BufferChunk>;
    BufferChunk(PEAK_BUFFER_CHUNK_HANDLE bufferChunkHandle, const std::weak_ptr<Buffer>& parentBuffer);
    PEAK_BUFFER_CHUNK_HANDLE m_backendHandle;

    std::weak_ptr<Buffer> m_parentBuffer;
};

} /* namespace core */
} /* namespace peak */

/* Implementation */
namespace peak
{
namespace core
{

inline BufferChunk::BufferChunk(
    PEAK_BUFFER_CHUNK_HANDLE bufferChunkHandle, const std::weak_ptr<Buffer>& parentBuffer)
    : m_backendHandle(bufferChunkHandle)
    , m_parentBuffer(parentBuffer)
{}

inline uint64_t BufferChunk::ID() const
{
    return QueryNumericFromCInterfaceFunction<uint64_t>(
        [&](uint64_t* id) { return PEAK_C_ABI_PREFIX PEAK_BufferChunk_GetID(m_backendHandle, id); });
}

inline void* BufferChunk::BasePtr() const
{
    return QueryNumericFromCInterfaceFunction<void*>(
        [&](void** basePtr) { return PEAK_C_ABI_PREFIX PEAK_BufferChunk_GetBasePtr(m_backendHandle, basePtr); });
}

inline size_t BufferChunk::Size() const
{
    return QueryNumericFromCInterfaceFunction<size_t>(
        [&](size_t* size) { return PEAK_C_ABI_PREFIX PEAK_BufferChunk_GetSize(m_backendHandle, size); });
}

inline std::shared_ptr<Buffer> BufferChunk::ParentBuffer() const
{
    return LockOrThrow(m_parentBuffer);
}

} /* namespace core */
} /* namespace peak */
