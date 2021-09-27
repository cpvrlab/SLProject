/*!
 * \file    peak_ipl_image_writer.hpp
 *
 * \author  IDS Imaging Development Systems GmbH
 * \date    2019-05-16
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
#include <string>

namespace peak
{
namespace ipl
{

class Image;

/*!
 * \brief Writes an image to a file.
 *
 * Supported Formats are currently:
 * JPEG, PNG and BMP and a simple RAW format
 */
class ImageWriter final
{
public:
    ImageWriter() = delete;

    /*!
     * \brief Parameter for the PNG write e.g. compression quality.
     */
    struct PNGParameter final
    {
        PNGParameter()
            : Quality(100)
        {}
        // Compression parameter from 0 to 100 -> 100 means quality level 0 for PNG
        uint32_t Quality = 100;
    };

    /*!
     * \brief Parameter for the JPEG write e.g. compression quality.
     */
    struct JPEGParameter final
    {
        JPEGParameter()
            : Quality(75)
        {}
        // Quality parameter from 0 to 100
        uint32_t Quality = 75;
    };

    /*!
     * \brief Writes the specified image to the filesystem as BMP image.
     *
     * Not all image formats can be written to a BMP file.
     * Currently supported for:
     * Mono8, Mono10, Mono12, RGB8, RGB10, BGR8, BGR10, RGBa8, BGRa8
     * Written as Mono:
     * BayerGR8, BayerRG8, BayerGB8, BayerBG8, BayerGR10, BayerRG10, BayerGB10, BayerBG10, BayerGR12, BayerRG12,
     * BayerGB12, BayerBG12
     * For all other formats an exception is thrown.
     *
     * \param[in] filePath    The path of the file to store the image to.
     * \param[in] imageToSave The image to save.
     *
     * \throws ImageFormatNotSupportedException A file type is not supported for this image pixel format
     * \throws IOException                      Errors during file access e.g. no permissions on this file
     * \throws InvalidArgumentException         Arguments passed are invalid
     * \throws Exception                        An internal error has occurred
     *
     * \since 1.0
     * \since 1.2 Will throw ImageFormatNotSupportedException if imageToSave has packed pixel format
     */
    static void WriteAsBMP(const std::string& filePath, const Image& imageToSave);

    /*!
     * \brief Writes the specified image to the filesystem as PNG image
     *
     * Not all image formats can be written to a PNG file.
     * Currently supported for:
     * Mono8, Mono10, Mono12, RGB8, RGB10, RGB12, RGBa8, RGBa10, RGBa12
     * Written as Mono:
     * BayerGR8, BayerRG8, BayerGB8, BayerBG8, BayerGR10, BayerRG10, BayerGB10, BayerBG10, BayerGR12, BayerRG12,
     * BayerGB12, BayerBG12
     * Written as RGB:
     * BGR8, BGR10, BGR12, BGRa8, BGRa10, RGBa12
     * For all other formats an exception is thrown.
     *
     * \param[in] filePath    The path of the file to store the image to.
     * \param[in] imageToSave The image to save.
     * \param[in] parameter   The parameter for the PNG image e.g. compression.
     *
     * \throws ImageFormatNotSupportedException A file type is not supported for this image pixel format
     * \throws IOException                      Errors during file access e.g. no permissions on this file
     * \throws InvalidArgumentException         Arguments passed are invalid
     * \throws Exception                        An internal error has occurred
     *
     * \since 1.0
     */
    static void WriteAsPNG(
        const std::string& filePath, const Image& imageToSave, const PNGParameter& parameter = PNGParameter());

    /*!
     * \brief Writes the specified image to the filesystem as JPEG image
     *
     * Not all image formats can be written to a JPEG file.
     * Currently supported for:
     * Mono8, RGB8, BGR8, RGBa8, BGRa8
     * Written as Mono: BayerGR8, BayerRG8, BayerGB8, BayerBG8
     * For all other formats an exception is thrown.
     *
     * \param[in] filePath    The path of the file to store the image to.
     * \param[in] imageToSave The image to save.
     * \param[in] parameter   The parameter for the JPEG image e.g. compression.
     *
     * \throws ImageFormatNotSupportedException A file type is not supported for this image pixel format
     * \throws IOException                      Errors during file access e.g. no permissions on this file
     * \throws InvalidArgumentException         Arguments passed are invalid
     * \throws Exception                        An internal error has occurred
     *
     * \since 1.0
     */
    static void WriteAsJPG(
        const std::string& filePath, const Image& imageToSave, const JPEGParameter& parameter = JPEGParameter());

    /*!
     * \brief Writes the specified image to the filesystem as a raw binary image.
     *
     * This is supported for all non-packed pixel formats.
     *
     * \param[in] filePath    The path of the file to store the image to.
     * \param[in] imageToSave The image to save.
     *
     * \throws ImageFormatNotSupportedException A file type is not supported for this image pixel format
     * \throws IOException                      Errors during file access e.g. no permissions on this file
     * \throws InvalidArgumentException         Arguments passed are invalid
     * \throws Exception                        An internal error has occurred
     *
     * \since 1.0
     * \since 1.2 Will throw ImageFormatNotSupportedException if imageToSave has packed pixel format
     */
    static void WriteAsRAW(const std::string& filePath, const Image& imageToSave);

    /*! \brief Writes the specified image to the filesystem. The type is specified by the given file ending in file
     *         name.
     *
     * \param[in] filePath    The path of the file to store the image to.
     * \param[in] imageToSave The image to save.
     *
     * \throws ImageFormatNotSupportedException A file type is not supported for this image pixel format
     * \throws IOException                      Errors during file access e.g. no permissions on this file
     * \throws InvalidArgumentException         Arguments passed are invalid
     * \throws Exception                        An internal error has occurred
     *
     * \since 1.0
     */
    static void Write(const std::string& filePath, const Image& imageToSave);
};

} /* namespace ipl */
} /* namespace peak */

#include <peak_ipl/types/peak_ipl_image.hpp>

namespace peak
{
namespace ipl
{
inline void ImageWriter::WriteAsBMP(const std::string& filePath, const Image& imageToSave)
{
    ExecuteAndMapReturnCodes([&] {
        return PEAK_IPL_C_ABI_PREFIX PEAK_IPL_ImageWriter_WriteAsBMP(
            ImageBackendAccessor::BackendHandle(imageToSave), filePath.c_str(), filePath.size());
    });
}

inline void ImageWriter::WriteAsPNG(
    const std::string& filePath, const Image& imageToSave, const PNGParameter& parameter)
{
    ExecuteAndMapReturnCodes([&] {
        return PEAK_IPL_C_ABI_PREFIX PEAK_IPL_ImageWriter_WriteAsPNG(
            ImageBackendAccessor::BackendHandle(imageToSave), parameter.Quality, filePath.c_str(), filePath.size());
    });
}

inline void ImageWriter::WriteAsJPG(
    const std::string& filePath, const Image& imageToSave, const JPEGParameter& parameter)
{
    ExecuteAndMapReturnCodes([&] {
        return PEAK_IPL_C_ABI_PREFIX PEAK_IPL_ImageWriter_WriteAsJPG(
            ImageBackendAccessor::BackendHandle(imageToSave), parameter.Quality, filePath.c_str(), filePath.size());
    });
}

inline void ImageWriter::WriteAsRAW(const std::string& filePath, const Image& imageToSave)
{
    ExecuteAndMapReturnCodes([&] {
        return PEAK_IPL_C_ABI_PREFIX PEAK_IPL_ImageWriter_WriteAsRAW(
            ImageBackendAccessor::BackendHandle(imageToSave), filePath.c_str(), filePath.size());
    });
}

inline void ImageWriter::Write(const std::string& filePath, const Image& imageToSave)
{
    ExecuteAndMapReturnCodes([&] {
        return PEAK_IPL_C_ABI_PREFIX PEAK_IPL_ImageWriter_Write(
            ImageBackendAccessor::BackendHandle(imageToSave), filePath.c_str(), filePath.size());
    });
}

} /* namespace ipl */
} /* namespace peak */
