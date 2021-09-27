/*!
 * \file    peak_buffer_part.hpp
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


namespace peak
{
namespace core
{

/*!
 * \brief Data type of the buffer part.
 *
 * See GenTL PARTDATATYPE_IDS.
 */
enum class BufferPartType
{
    Unknown,
    Image2D,
    PlaneBiPlanar2D,
    PlaneTriPlanar2D,
    PlaneQuadPlanar2D,
    Image3D,
    PlaneBiPlanar3D,
    PlaneTriPlanar3D,
    PlaneQuadPlanar3D,
    ConfidenceMap,

    Custom = 1000
};

class Buffer;

/*!
 * \brief Represents a buffer part.
 *
 * This class allows to query information about a buffer part.
 */
class BufferPart
{
public:
    BufferPart() = delete;
    ~BufferPart() = default;
    BufferPart(const BufferPart& other) = delete;
    BufferPart& operator=(const BufferPart& other) = delete;
    BufferPart(BufferPart&& other) = delete;
    BufferPart& operator=(BufferPart&& other) = delete;

    /*! @copydoc SystemDescriptor::Info() */
    RawInformation Info(int32_t infoCommand) const;
    /*!
     * \brief Returns the source ID.
     *
     * \return Source ID
     *
     * \since 1.0
     *
     * \throws InternalErrorException An internal error has occurred.
     */
    uint64_t SourceID() const;
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
     * \brief Returns the size in bytes.
     *
     * \return Size in bytes
     *
     * \since 1.0
     *
     * \throws InternalErrorException An internal error has occurred.
     */
    size_t Size() const;
    /*!
     * \brief Returns the type.
     *
     * \return Type
     *
     * \since 1.0
     *
     * \throws InternalErrorException An internal error has occurred.
     */
    BufferPartType Type() const;
    /*!
     * \brief Returns the format.
     *
     * \return Format
     *
     * \since 1.0
     *
     * \throws InternalErrorException An internal error has occurred.
     */
    uint64_t Format() const;
    /*!
     * \brief Returns the format namespace.
     *
     * \return Format namespace
     *
     * \since 1.0
     *
     * \throws InternalErrorException An internal error has occurred.
     */
    uint64_t FormatNamespace() const;
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
     * \brief Returns the x offset.
     *
     * \return X offset
     *
     * \since 1.0
     *
     * \throws InternalErrorException An internal error has occurred.
     */
    size_t XOffset() const;
    /*!
     * \brief Returns the y offset.
     *
     * \return Y offset
     *
     * \since 1.0
     *
     * \throws InternalErrorException An internal error has occurred.
     */
    size_t YOffset() const;
    /*!
     * \brief Returns the x padding.
     *
     * \return X padding
     *
     * \since 1.0
     *
     * \throws InternalErrorException An internal error has occurred.
     */
    size_t XPadding() const;
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
    friend ClassCreator<BufferPart>;
    BufferPart(PEAK_BUFFER_PART_HANDLE bufferPartHandle, const std::weak_ptr<Buffer>& parentBuffer);
    PEAK_BUFFER_PART_HANDLE m_backendHandle;

    std::weak_ptr<Buffer> m_parentBuffer;
};

} /* namespace core */
} /* namespace peak */

/* Implementation */
namespace peak
{
namespace core
{

inline std::string ToString(BufferPartType entry)
{
    std::string entryString;

    if (entry == BufferPartType::Unknown)
    {
        entryString = "Unknown";
    }
    else if (entry == BufferPartType::Image2D)
    {
        entryString = "Image2D";
    }
    else if (entry == BufferPartType::PlaneBiPlanar2D)
    {
        entryString = "PlaneBiPlanar2D";
    }
    else if (entry == BufferPartType::PlaneTriPlanar2D)
    {
        entryString = "PlaneTriPlanar2D";
    }
    else if (entry == BufferPartType::PlaneQuadPlanar2D)
    {
        entryString = "PlaneQuadPlanar2D";
    }
    else if (entry == BufferPartType::Image3D)
    {
        entryString = "Image3D";
    }
    else if (entry == BufferPartType::PlaneBiPlanar3D)
    {
        entryString = "PlaneBiPlanar3D";
    }
    else if (entry == BufferPartType::PlaneTriPlanar3D)
    {
        entryString = "PlaneTriPlanar3D";
    }
    else if (entry == BufferPartType::PlaneQuadPlanar3D)
    {
        entryString = "PlaneQuadPlanar3D";
    }
    else if (entry == BufferPartType::ConfidenceMap)
    {
        entryString = "ConfidenceMap";
    }
    else if (entry >= BufferPartType::Custom)
    {
        entryString = "Custom";
    }

    return entryString;
}

inline BufferPart::BufferPart(PEAK_BUFFER_PART_HANDLE bufferPartHandle, const std::weak_ptr<Buffer>& parentBuffer)
    : m_backendHandle(bufferPartHandle)
    , m_parentBuffer(parentBuffer)
{}

inline RawInformation BufferPart::Info(int32_t infoCommand) const
{
    return QueryRawInformationFromCInterfaceFunction([&](int32_t* dataType, uint8_t* buffer, size_t* bufferSize) {
        return PEAK_C_ABI_PREFIX PEAK_BufferPart_GetInfo(
            m_backendHandle, infoCommand, dataType, buffer, bufferSize);
    });
}

inline uint64_t BufferPart::SourceID() const
{
    return QueryNumericFromCInterfaceFunction<uint64_t>([&](uint64_t* sourceId) {
        return PEAK_C_ABI_PREFIX PEAK_BufferPart_GetSourceID(m_backendHandle, sourceId);
    });
}

inline void* BufferPart::BasePtr() const
{
    return QueryNumericFromCInterfaceFunction<void*>(
        [&](void** basePtr) { return PEAK_C_ABI_PREFIX PEAK_BufferPart_GetBasePtr(m_backendHandle, basePtr); });
}

inline size_t BufferPart::Size() const
{
    return QueryNumericFromCInterfaceFunction<size_t>(
        [&](size_t* size) { return PEAK_C_ABI_PREFIX PEAK_BufferPart_GetSize(m_backendHandle, size); });
}

inline BufferPartType BufferPart::Type() const
{
    return static_cast<BufferPartType>(
        QueryNumericFromCInterfaceFunction<PEAK_BUFFER_PART_TYPE>([&](PEAK_BUFFER_PART_TYPE* type) {
            return PEAK_C_ABI_PREFIX PEAK_BufferPart_GetType(m_backendHandle, type);
        }));
}

inline uint64_t BufferPart::Format() const
{
    return QueryNumericFromCInterfaceFunction<uint64_t>(
        [&](uint64_t* format) { return PEAK_C_ABI_PREFIX PEAK_BufferPart_GetFormat(m_backendHandle, format); });
}

inline uint64_t BufferPart::FormatNamespace() const
{
    return QueryNumericFromCInterfaceFunction<uint64_t>([&](uint64_t* formatNamespace) {
        return PEAK_C_ABI_PREFIX PEAK_BufferPart_GetFormatNamespace(m_backendHandle, formatNamespace);
    });
}

inline size_t BufferPart::Width() const
{
    return QueryNumericFromCInterfaceFunction<size_t>(
        [&](size_t* width) { return PEAK_C_ABI_PREFIX PEAK_BufferPart_GetWidth(m_backendHandle, width); });
}

inline size_t BufferPart::Height() const
{
    return QueryNumericFromCInterfaceFunction<size_t>(
        [&](size_t* height) { return PEAK_C_ABI_PREFIX PEAK_BufferPart_GetWidth(m_backendHandle, height); });
}

inline size_t BufferPart::XOffset() const
{
    return QueryNumericFromCInterfaceFunction<size_t>(
        [&](size_t* xOffset) { return PEAK_C_ABI_PREFIX PEAK_BufferPart_GetXOffset(m_backendHandle, xOffset); });
}

inline size_t BufferPart::YOffset() const
{
    return QueryNumericFromCInterfaceFunction<size_t>(
        [&](size_t* yOffset) { return PEAK_C_ABI_PREFIX PEAK_BufferPart_GetYOffset(m_backendHandle, yOffset); });
}

inline size_t BufferPart::XPadding() const
{
    return QueryNumericFromCInterfaceFunction<size_t>([&](size_t* xPadding) {
        return PEAK_C_ABI_PREFIX PEAK_BufferPart_GetXPadding(m_backendHandle, xPadding);
    });
}

inline size_t BufferPart::DeliveredImageHeight() const
{
    return QueryNumericFromCInterfaceFunction<size_t>([&](size_t* deliveredImageHeight) {
        return PEAK_C_ABI_PREFIX PEAK_BufferPart_GetDeliveredImageHeight(m_backendHandle, deliveredImageHeight);
    });
}

inline std::shared_ptr<Buffer> BufferPart::ParentBuffer() const
{
    return LockOrThrow(m_parentBuffer);
}

} /* namespace core */
} /* namespace peak */
