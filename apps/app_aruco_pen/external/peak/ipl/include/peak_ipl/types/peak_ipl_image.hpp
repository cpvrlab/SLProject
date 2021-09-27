/*!
 * \file    peak_ipl_image.hpp
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
#include <peak_ipl/types/peak_ipl_pixel_format.hpp>

#include <cstddef>
#include <cstdint>
#include <utility>

namespace
{
class ImageBackendAccessor;
}

/*!
 * \brief The "peak::ipl" namespace contains the whole image processing library.
 */
namespace peak
{
namespace ipl
{

/*!
 * \brief Stores the pixel format, width and height of an image and the pointer to the image buffer.
 */
class Image
{
public:
    /*!
     * \brief Creates an empty image, i.e. with size 0x0 and invalid pixel format.
     *
     * \throws Exception An internal error has occurred.
     *
     * \since 1.1
     */
    Image();
    /*!
     * \brief Creates an image with the given pixel format and size.
     *
     * \param[in] pixelFormat The pixel format.
     * \param[in] width       The width.
     * \param[in] height      The height.
     *
     * \throws OutOfRangeException pixelFormat is valid but width or height is 0.
     * \throws Exception An internal error has occurred.
     *
     * \since 1.0
     */
    Image(const peak::ipl::PixelFormat& pixelFormat, size_t width, size_t height);
    /*!
     * \brief Creates an image with the given pixel format and size from the given buffer.
     *
     * \param[in] pixelFormat The pixel format.
     * \param[in] buffer      The buffer.
     * \param[in] bufferSize  The size of the buffer.
     * \param[in] width       The width.
     * \param[in] height      The height.
     *
     * \note The given buffer does not get copied. This is why the buffer must not be freed before
     *       the image gets freed.
     *
     * \throws InvalidArgumentException bufferSize is too small for the specified pixel format and image dimension.
     * \throws OutOfRangeException pixelFormat is valid but width or height is 0.
     * \throws Exception An internal error has occurred.
     *
     * \since 1.0
     */
    Image(
        const peak::ipl::PixelFormat& pixelFormat, uint8_t* buffer, size_t bufferSize, size_t width, size_t height);
    ~Image();
    Image(const Image& other) = delete;
    Image& operator=(const Image& other) = delete;
    Image(Image&& other) noexcept;
    Image& operator=(Image&& other) noexcept;

    /*!
     * \brief Returns the width.
     *
     * \returns Width
     *
     * \throws Exception An internal error has occurred.
     *
     * \since 1.0
     */
    size_t Width() const;

    /*!
     * \brief Returns the height.
     *
     * \returns Height
     *
     * \throws Exception An internal error has occurred.
     *
     * \since 1.0
     */
    size_t Height() const;

    /*!
     * \brief Returns the pixel pointer to the given pixel position in dependency of the size of the pixel format.
     *
     * \param[in] xPos The x position.
     * \param[in] yPos The y position.
     *
     * \returns PixelPointer
     *
     * \throws OutOfRangeException The pixel position is outside the image.
     * \throws Exception An internal error has occurred.
     *
     * \since 1.0
     */
    uint8_t* PixelPointer(size_t xPos, size_t yPos) const;

    /*!
     * \brief Returns the size of the given image in number of bytes.
     *
     * \returns ByteCount
     *
     * \throws Exception An internal error has occurred.
     *
     * \since 1.0
     */
    size_t ByteCount() const;

    /*!
     * \brief Returns the pixel format.
     *
     * \return Pixel format
     *
     * \throws Exception An internal error has occurred.
     *
     * \since 1.0
     */
    peak::ipl::PixelFormat PixelFormat() const;

    /*!
     * \brief Returns the pointer to the first pixel position in dependency on the size of the pixel format.
     *
     * \returns Buffer data.
     *
     * \throws Exception An internal error has occurred.
     *
     * \since 1.0
     */
    uint8_t* Data() const;

    /*!
     * \brief Returns a new created image containing the data of the current image converted to the given pixel format.
     *
     * \param[in]  outputPixelFormat The output pixel format.
     * \param[in]  conversionMode    The conversion mode.
     *
     * \returns Converted image
     *
     * \throws Exception An internal error has occurred.
     *
     * \since 1.0
     */
    Image ConvertTo(const peak::ipl::PixelFormat& outputPixelFormat,
        peak::ipl::ConversionMode conversionMode = peak::ipl::ConversionMode::Fast) const;

    /*!
     * \brief Saves the data of the current image converted to the given pixel format into a destination buffer
     *        and creates an image from that buffer.
     *
     * \param[in]  outputPixelFormat     The output pixel format.
     * \param[out] outputImageBuffer     Pointer to destination buffer.
     * \param[in]  outputImageBufferSize Size of destination buffer.
     * \param[in]  conversionMode        The conversion mode.
     *
     * \returns Converted image
     *
     * \throws Exception An internal error has occurred.
     *
     * \since 1.1
     */
    Image ConvertTo(const peak::ipl::PixelFormat& outputPixelFormat, uint8_t* outputImageBuffer,
        size_t outputImageBufferSize,
        peak::ipl::ConversionMode conversionMode = peak::ipl::ConversionMode::Fast) const;

    /*!
     * \brief Returns a new created image containing the data of the current image as deep copy.
     *
     * \returns Copied image
     *
     * \throws Exception An internal error has occurred.
     *
     * \since 1.0
     */
    Image Clone() const;

    /*!
     * \brief Checks whether the image is empty.
     *
     * An image can be empty if its data have been moved.
     *
     * \returns True if image is empty
     *
     * \since 1.0
     */
    bool Empty() const;

private:
    friend ImageBackendAccessor;
    explicit Image(PEAK_IPL_IMAGE_HANDLE imageHandle) noexcept;
    PEAK_IPL_IMAGE_HANDLE m_backendHandle{};
};

} /* namespace ipl */
} /* namespace peak */

namespace
{

// helper class to access the C-backend of the Image, which are hidden from the public interface
class ImageBackendAccessor
{
public:
    static PEAK_IPL_IMAGE_HANDLE BackendHandle(const peak::ipl::Image& image)
    {
        return image.m_backendHandle;
    }
    static peak::ipl::Image CreateImage(PEAK_IPL_IMAGE_HANDLE imageHandle)
    {
        return peak::ipl::Image(imageHandle);
    }
};

} /* namespace */

#include <peak_ipl/algorithm/peak_ipl_image_converter.hpp>

namespace peak
{
namespace ipl
{

inline Image::Image()
{
    ExecuteAndMapReturnCodes([&] {
        return PEAK_IPL_C_ABI_PREFIX PEAK_IPL_Image_Construct(PEAK_IPL_PIXEL_FORMAT_INVALID, 0, 0, &m_backendHandle);
    });
}

inline Image::Image(const peak::ipl::PixelFormat& pixelFormat, size_t width, size_t height)
{
    ExecuteAndMapReturnCodes([&] {
        return PEAK_IPL_C_ABI_PREFIX PEAK_IPL_Image_Construct(
            static_cast<PEAK_IPL_PIXEL_FORMAT>(pixelFormat.PixelFormatName()), width, height, &m_backendHandle);
    });
}

inline Image::Image(
    const peak::ipl::PixelFormat& pixelFormat, uint8_t* buffer, size_t bufferSize, size_t width, size_t height)
{
    ExecuteAndMapReturnCodes([&] {
        return PEAK_IPL_C_ABI_PREFIX PEAK_IPL_Image_ConstructFromBuffer(
            static_cast<PEAK_IPL_PIXEL_FORMAT>(pixelFormat.PixelFormatName()), buffer, bufferSize, width, height,
            &m_backendHandle);
    });
}

inline Image::Image(PEAK_IPL_IMAGE_HANDLE imageHandle) noexcept
    : m_backendHandle(imageHandle)
{}

inline Image::~Image()
{
    (void)PEAK_IPL_C_ABI_PREFIX PEAK_IPL_Image_Destruct(m_backendHandle);
}

inline Image::Image(Image&& other) noexcept
{
    *this = std::move(other);
}

inline Image& Image::operator=(Image&& other) noexcept
{
    if (this != &other)
    {
        (void)PEAK_IPL_C_ABI_PREFIX PEAK_IPL_Image_Destruct(m_backendHandle);
        m_backendHandle = other.m_backendHandle;
        other.m_backendHandle = nullptr;
    }

    return *this;
}

inline size_t Image::Width() const
{
    size_t width = 0;

    ExecuteAndMapReturnCodes([&] { return PEAK_IPL_C_ABI_PREFIX PEAK_IPL_Image_GetWidth(m_backendHandle, &width); });

    return width;
}

inline size_t Image::Height() const
{
    size_t height = 0;

    ExecuteAndMapReturnCodes([&] { return PEAK_IPL_C_ABI_PREFIX PEAK_IPL_Image_GetHeight(m_backendHandle, &height); });

    return height;
}

inline uint8_t* Image::PixelPointer(size_t xPos, size_t yPos) const
{
    uint8_t* pixelPosition = nullptr;

    ExecuteAndMapReturnCodes([&] {
        return PEAK_IPL_C_ABI_PREFIX PEAK_IPL_Image_GetPixelPointer(m_backendHandle, xPos, yPos, &pixelPosition);
    });

    return pixelPosition;
}

inline size_t Image::ByteCount() const
{
    size_t byteCount = 0;

    ExecuteAndMapReturnCodes(
        [&] { return PEAK_IPL_C_ABI_PREFIX PEAK_IPL_Image_GetByteCount(m_backendHandle, &byteCount); });

    return byteCount;
}

inline peak::ipl::PixelFormat Image::PixelFormat() const
{
    peak::ipl::PixelFormatName pixelFormatName = PixelFormatName::Invalid;

    ExecuteAndMapReturnCodes([&] {
        return PEAK_IPL_C_ABI_PREFIX PEAK_IPL_Image_GetPixelFormat(
            m_backendHandle, reinterpret_cast<PEAK_IPL_PIXEL_FORMAT*>(&pixelFormatName));
    });

    return peak::ipl::PixelFormat{ pixelFormatName };
}

inline uint8_t* Image::Data() const
{
    uint8_t* data = nullptr;

    ExecuteAndMapReturnCodes([&] { return PEAK_IPL_C_ABI_PREFIX PEAK_IPL_Image_GetData(m_backendHandle, &data); });

    return data;
}

inline Image Image::ConvertTo(
    const peak::ipl::PixelFormat& outputPixelFormat, peak::ipl::ConversionMode conversionMode) const
{
    PEAK_IPL_IMAGE_HANDLE outputImageHandle = nullptr;
    ExecuteAndMapReturnCodes([&] {
        return PEAK_IPL_C_ABI_PREFIX PEAK_IPL_Image_ConvertTo(m_backendHandle,
            static_cast<PEAK_IPL_PIXEL_FORMAT>(outputPixelFormat.PixelFormatName()),
            static_cast<PEAK_IPL_CONVERSION_MODE>(conversionMode), &outputImageHandle);
    });

    return ImageBackendAccessor::CreateImage(outputImageHandle);
}

inline Image Image::ConvertTo(const peak::ipl::PixelFormat& outputPixelFormat, uint8_t* outputImageBuffer,
    size_t outputImageBufferSize, peak::ipl::ConversionMode conversionMode) const
{
    PEAK_IPL_IMAGE_HANDLE outputImageHandle = nullptr;
    ExecuteAndMapReturnCodes([&] {
        return PEAK_IPL_C_ABI_PREFIX PEAK_IPL_Image_ConvertToBuffer(m_backendHandle,
            static_cast<PEAK_IPL_PIXEL_FORMAT>(outputPixelFormat.PixelFormatName()), outputImageBuffer,
            outputImageBufferSize, static_cast<PEAK_IPL_CONVERSION_MODE>(conversionMode), &outputImageHandle);
    });

    return ImageBackendAccessor::CreateImage(outputImageHandle);
}

inline Image Image::Clone() const
{
    PEAK_IPL_IMAGE_HANDLE outputImageHandle = nullptr;
    ExecuteAndMapReturnCodes(
        [&] { return PEAK_IPL_C_ABI_PREFIX PEAK_IPL_Image_Clone(m_backendHandle, &outputImageHandle); });

    return ImageBackendAccessor::CreateImage(outputImageHandle);
}

inline bool Image::Empty() const
{
    return (m_backendHandle == nullptr) || (PixelFormat().PixelFormatName() == PixelFormatName::Invalid);
}

} /* namespace ipl */
} /* namespace peak */
