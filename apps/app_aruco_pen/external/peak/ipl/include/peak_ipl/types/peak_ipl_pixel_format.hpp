/*!
 * \file    peak_ipl_pixel_format.hpp
 *
 * \author  IDS Imaging Development Systems GmbH
 * \date    2019-05-01
 * \since   1.0
 *
 * Copyright (c) 2019, IDS Imaging Development Systems GmbH. All rights reserved.
 */

#pragma once

#include <peak_ipl/backend/peak_ipl_backend.h>
#include <peak_ipl/exception/peak_ipl_exception.hpp>
#include <peak_ipl/types/peak_ipl_simple_types.hpp>

#include <cstddef>
#include <cstdint>

/*!
 * \brief The "peak::ipl" namespace contains the whole image processing library.
 */
namespace peak
{
namespace ipl
{

/*!
 * \brief Represents a pixel format and its specific properties.
 */
class PixelFormat final
{
public:
    PixelFormat()
        : m_name(peak::ipl::PixelFormatName::Invalid)
    {}
    PixelFormat(peak::ipl::PixelFormatName name);
    ~PixelFormat() = default;
    PixelFormat(const PixelFormat& other) = default;
    PixelFormat& operator=(const PixelFormat& other) = default;
    PixelFormat(PixelFormat&& other) = default;
    PixelFormat& operator=(PixelFormat&& other) = default;

    /*!
     * \brief Returns the name of the pixel format as String.
     *
     * \returns Name
     *
     * \since 1.0
     *
     * \throws Exception An internal error has occurred.
     */
    std::string Name() const
    {
        return peak::ipl::ToString(m_name);
    }

    /*!
     * \brief Returns the name of the pixel format as enum value.
     *
     * \returns PixelFormatName
     *
     * \since 1.0
     *
     * \throws Exception An internal error has occurred.
     */
    peak::ipl::PixelFormatName PixelFormatName() const
    {
        return m_name;
    }

    bool operator==(const PixelFormat& other) const
    {
        return m_name == other.m_name;
    }

    /*! \brief Returns the number of significant bits per pixel per channel of the given pixel format.
     *
     * \returns Number of significant bits per pixel per channel.
     *
     * \since 1.0
     *
     * \throws Exception An internal error has occurred.
     */
    size_t NumSignificantBitsPerChannel() const;

    /*! \brief Returns the number of storage bits per pixel per channel of the given pixel format.
     *
     * \returns Number of storage bits per pixel per channel.
     *
     *
     * \since 1.0
     *
     * \throws Exception An internal error has occurred.
     */
    size_t NumStorageBitsPerChannel() const;

    /*! \brief Returns the number of channels of the given pixel format.
     *
     * \returns Number of channels.
     *
     * \since 1.0
     *
     * \throws Exception An internal error has occurred.
     */
    size_t NumChannels() const;

    /*!
     * \brief Returns the maximum value of one pixel channel of the given pixel format.
     *
     * \returns Maximum value of one pixel channel.
     *
     * \since 1.0
     *
     * \throws Exception An internal error has occurred.
     */
    uint32_t MaximumValuePerChannel() const;

    /*!
     * \brief Returns the number of significant bits per pixel of the given pixel format.
     *
     * \returns NumSignificantBits
     *
     * \since 1.0
     *
     * \throws Exception An internal error has occurred.
     */
    size_t NumSignificantBitsPerPixel() const;

    /*!
     * \brief Returns the number of storage bits per pixel of the given pixel format.
     *
     * returns NumStorageBits
     *
     * \since 1.0
     *
     * \throws Exception An internal error has occurred.
     */
    size_t NumStorageBitsPerPixel() const;


    /*! \brief Returns the endianness of the given pixel format.
     *
     * \returns Endianness.
     *
     * \since 1.0
     *
     * \throws Exception An internal error has occurred.
     */
    peak::ipl::Endianness Endianness() const;

    /*! \brief Returns the storage size of the given number of pixels of the given pixel format in bytes.
     *
     * \param[in]  numPixels   The number of pixels.
     *
     * \returns    CalculateStorageSizeOfPixels
     *
     * \since 1.0
     *
     * \throws Exception An internal error has occurred.
     */
    uint64_t CalculateStorageSizeOfPixels(uint64_t numPixels) const;

private:
    peak::ipl::PixelFormatName m_name{};
};

inline PixelFormat::PixelFormat(peak::ipl::PixelFormatName name)
    : m_name(name)
{}

inline size_t PixelFormat::NumSignificantBitsPerChannel() const
{
    size_t numSignificantBitsPerChannel = 0;

    ExecuteAndMapReturnCodes([&] {
        return PEAK_IPL_C_ABI_PREFIX PEAK_IPL_PixelFormat_GetNumSignificantBitsPerChannel(
            static_cast<PEAK_IPL_PIXEL_FORMAT>(m_name), &numSignificantBitsPerChannel);
    });

    return numSignificantBitsPerChannel;
}

inline size_t PixelFormat::NumStorageBitsPerChannel() const
{
    size_t numStorageBitsPerChannel = 0;

    ExecuteAndMapReturnCodes([&] {
        return PEAK_IPL_C_ABI_PREFIX PEAK_IPL_PixelFormat_GetNumStorageBitsPerChannel(
            static_cast<PEAK_IPL_PIXEL_FORMAT>(m_name), &numStorageBitsPerChannel);
    });

    return numStorageBitsPerChannel;
}

inline size_t PixelFormat::NumChannels() const
{
    size_t numChannels = 0;

    ExecuteAndMapReturnCodes([&] {
        return PEAK_IPL_C_ABI_PREFIX PEAK_IPL_PixelFormat_GetNumChannels(
            static_cast<PEAK_IPL_PIXEL_FORMAT>(m_name), &numChannels);
    });

    return numChannels;
}

inline uint32_t PixelFormat::MaximumValuePerChannel() const
{
    uint32_t maximumValuePerChannel = 0;

    ExecuteAndMapReturnCodes([&] {
        return PEAK_IPL_C_ABI_PREFIX PEAK_IPL_PixelFormat_GetMaximumValuePerChannel(
            static_cast<PEAK_IPL_PIXEL_FORMAT>(m_name), &maximumValuePerChannel);
    });

    return maximumValuePerChannel;
}

inline size_t PixelFormat::NumSignificantBitsPerPixel() const
{
    size_t numSignificantBitsPerPixel = 0;

    ExecuteAndMapReturnCodes([&] {
        return PEAK_IPL_C_ABI_PREFIX PEAK_IPL_PixelFormat_GetNumSignificantBitsPerPixel(
            static_cast<PEAK_IPL_PIXEL_FORMAT>(m_name), &numSignificantBitsPerPixel);
    });

    return numSignificantBitsPerPixel;
}

inline size_t PixelFormat::NumStorageBitsPerPixel() const
{
    size_t numStorageBitsPerPixel = 0;

    ExecuteAndMapReturnCodes([&] {
        return PEAK_IPL_C_ABI_PREFIX PEAK_IPL_PixelFormat_GetNumStorageBitsPerPixel(
            static_cast<PEAK_IPL_PIXEL_FORMAT>(m_name), &numStorageBitsPerPixel);
    });

    return numStorageBitsPerPixel;
}

inline peak::ipl::Endianness PixelFormat::Endianness() const
{
    peak::ipl::Endianness endianness = peak::ipl::Endianness::LittleEndian;

    ExecuteAndMapReturnCodes([&] {
        return PEAK_IPL_C_ABI_PREFIX PEAK_IPL_PixelFormat_GetEndianness(
            static_cast<PEAK_IPL_PIXEL_FORMAT>(m_name), reinterpret_cast<PEAK_IPL_ENDIANNESS*>(&endianness));
    });

    return endianness;
}

inline uint64_t PixelFormat::CalculateStorageSizeOfPixels(uint64_t numPixels) const
{
    uint64_t sizeOfPixels = 0;

    ExecuteAndMapReturnCodes([&] {
        return PEAK_IPL_C_ABI_PREFIX PEAK_IPL_PixelFormat_CalculateStorageSizeOfPixels(
            static_cast<PEAK_IPL_PIXEL_FORMAT>(m_name), numPixels, &sizeOfPixels);
    });

    return sizeOfPixels;
}

} /* namespace ipl */
} /* namespace peak */
