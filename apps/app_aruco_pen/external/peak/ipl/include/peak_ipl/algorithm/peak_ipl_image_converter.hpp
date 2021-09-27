/*!
 * \file    peak_ipl_image_converter.hpp
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
#include <peak_ipl/types/peak_ipl_simple_types.hpp>

#include <cstdint>
#include <utility>
#include <vector>

/*!
 * \brief The "peak::ipl" namespace contains the whole image processing library.
 */
namespace peak
{
namespace ipl
{

class Image;

/*!
 * \brief Converts images from one PixelFormat to another.
 *
 * \note To speed up processing instances of this class maintain internal memory pools to reuse
 * memory instead of allocating new memory for each conversion. The memory is freed when the
 * instance is destroyed.
 */
class ImageConverter final
{
public:
    ImageConverter();
    ~ImageConverter();
    ImageConverter(const ImageConverter& other) = delete;
    ImageConverter& operator=(const ImageConverter& other) = delete;
    ImageConverter(ImageConverter&& other) noexcept;
    ImageConverter& operator=(ImageConverter&& other) noexcept;

    /*!
     * \brief Returns the current conversion mode.
     *
     * \returns The current conversion mode
     *
     * \throws Exception An internal error has occurred.
     *
     * \since 1.0
     */
    peak::ipl::ConversionMode ConversionMode() const;

    /*!
     * \brief Set conversion mode.
     *
     * \param[in] conversionMode The conversion mode to set.
     *
     * \throws Exception An internal error has occurred.
     *
     * \since 1.0
     */
    void SetConversionMode(peak::ipl::ConversionMode conversionMode);

    /*!
     * \brief Returns the supported output pixel formats for a given input pixel format.
     *
     * \param[in] inputPixelFormat The input pixel format.
     *
     * \returns A vector of supported pixel formats for the given input pixel format
     *
     * \throws Exception An internal error has occurred.
     *
     * \since 1.0
     */
    std::vector<peak::ipl::PixelFormatName> SupportedOutputPixelFormatNames(
        const peak::ipl::PixelFormat& inputPixelFormat) const;

    /*!
     * \brief Converts the input image converted to the given pixel format.
     *
     * \param[in] inputImage        The input image.
     * \param[in] outputPixelFormat The output pixel format.
     *
     * \returns A new created image containing the data of the input image converted to the given pixel format
     *
     * \throws Exception An internal error has occurred.
     *
     * \since 1.0
     */
    Image Convert(const Image& inputImage, const peak::ipl::PixelFormat& outputPixelFormat) const;

    /*! \brief Saves the data of the current image converted to the given pixel format into a destination buffer
     *         and creates an image from that buffer.
     *
     * \param[in]  inputImage            The input image.
     * \param[in]  outputPixelFormat     The output pixel format.
     * \param[out] outputImageBuffer     Pointer to destination buffer.
     * \param[in]  outputImageBufferSize Size of destination buffer.
     *
     * \returns Converted image
     *
     * \throws Exception An internal error has occurred.
     *
     * \since 1.1
     */
    Image Convert(const Image& inputImage, const peak::ipl::PixelFormat& outputPixelFormat,
        uint8_t* outputImageBuffer, size_t outputImageBufferSize) const;

private:
    PEAK_IPL_IMAGE_CONVERTER_HANDLE m_backendHandle{};
};

} /* namespace ipl */
} /* namespace peak */

#include <peak_ipl/types/peak_ipl_image.hpp>

namespace peak
{
namespace ipl
{

inline ImageConverter::ImageConverter()
{
    ExecuteAndMapReturnCodes([&] { return PEAK_IPL_C_ABI_PREFIX PEAK_IPL_ImageConverter_Construct(&m_backendHandle); });
}

inline ImageConverter::~ImageConverter()
{
    (void)PEAK_IPL_C_ABI_PREFIX PEAK_IPL_ImageConverter_Destruct(m_backendHandle);
}

inline ImageConverter::ImageConverter(ImageConverter&& other) noexcept
{
    *this = std::move(other);
}

inline ImageConverter& ImageConverter::operator=(ImageConverter&& other) noexcept
{
    if (this != &other)
    {
        (void)PEAK_IPL_C_ABI_PREFIX PEAK_IPL_ImageConverter_Destruct(m_backendHandle);
        m_backendHandle = other.m_backendHandle;
        other.m_backendHandle = nullptr;
    }

    return *this;
}

inline peak::ipl::ConversionMode ImageConverter::ConversionMode() const
{
    peak::ipl::ConversionMode conversionMode = peak::ipl::ConversionMode::Fast;

    ExecuteAndMapReturnCodes([&] {
        return PEAK_IPL_C_ABI_PREFIX PEAK_IPL_ImageConverter_GetConversionMode(
            m_backendHandle, reinterpret_cast<PEAK_IPL_CONVERSION_MODE*>(&conversionMode));
    });

    return conversionMode;
}

inline void ImageConverter::SetConversionMode(peak::ipl::ConversionMode conversionMode)
{
    ExecuteAndMapReturnCodes([&] {
        return PEAK_IPL_C_ABI_PREFIX PEAK_IPL_ImageConverter_SetConversionMode(
            m_backendHandle, static_cast<PEAK_IPL_CONVERSION_MODE>(conversionMode));
    });
}

inline std::vector<PixelFormatName> ImageConverter::SupportedOutputPixelFormatNames(
    const PixelFormat& inputPixelFormat) const
{
    size_t size = 0;
    ExecuteAndMapReturnCodes([&] {
        return PEAK_IPL_C_ABI_PREFIX PEAK_IPL_ImageConverter_GetSupportedOutputPixelFormats(
            m_backendHandle, static_cast<PEAK_IPL_PIXEL_FORMAT>(inputPixelFormat.PixelFormatName()), nullptr, &size);
    });
    std::vector<PixelFormatName> supportedOutputPixelFormats(size);
    ExecuteAndMapReturnCodes([&] {
        return PEAK_IPL_C_ABI_PREFIX PEAK_IPL_ImageConverter_GetSupportedOutputPixelFormats(m_backendHandle,
            static_cast<PEAK_IPL_PIXEL_FORMAT>(inputPixelFormat.PixelFormatName()),
            reinterpret_cast<PEAK_IPL_PIXEL_FORMAT*>(supportedOutputPixelFormats.data()), &size);
    });

    return supportedOutputPixelFormats;
}

inline Image ImageConverter::Convert(const Image& inputImage, const PixelFormat& outputPixelFormat) const
{
    PEAK_IPL_IMAGE_HANDLE outputImageHandle = nullptr;
    ExecuteAndMapReturnCodes([&] {
        return PEAK_IPL_C_ABI_PREFIX PEAK_IPL_ImageConverter_Convert(m_backendHandle,
            ImageBackendAccessor::BackendHandle(inputImage),
            static_cast<PEAK_IPL_PIXEL_FORMAT>(outputPixelFormat.PixelFormatName()), &outputImageHandle);
    });

    return ImageBackendAccessor::CreateImage(outputImageHandle);
}

inline Image ImageConverter::Convert(const Image& inputImage, const PixelFormat& outputPixelFormat,
    uint8_t* outputImageBuffer, size_t outputImageBufferSize) const
{
    PEAK_IPL_IMAGE_HANDLE outputImageHandle = nullptr;
    ExecuteAndMapReturnCodes([&] {
        return PEAK_IPL_C_ABI_PREFIX PEAK_IPL_ImageConverter_ConvertToBuffer(m_backendHandle,
            ImageBackendAccessor::BackendHandle(inputImage),
            static_cast<PEAK_IPL_PIXEL_FORMAT>(outputPixelFormat.PixelFormatName()), outputImageBuffer,
            outputImageBufferSize, &outputImageHandle);
    });

    return ImageBackendAccessor::CreateImage(outputImageHandle);
}

} /* namespace ipl */
} /* namespace peak */
