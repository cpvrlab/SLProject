/*!
 * \file    peak_ipl_simple_types.hpp
 *
 * \author  IDS Imaging Development Systems GmbH
 * \date    2019-05-01
 * \since   1.0
 *
 * Copyright (c) 2019, IDS Imaging Development Systems GmbH. All rights reserved.
 */

#pragma once

#include <peak_ipl/backend/peak_ipl_backend.h>

#include <string>

/*!
 * \brief The "peak::ipl" namespace contains the whole image processing library.
 */
namespace peak
{
namespace ipl
{

/*!
 *\brief Enum holding the possible conversion modes.
 */
enum class ConversionMode
{
    Fast = PEAK_IPL_CONVERSION_MODE_FAST,
    HighQuality = PEAK_IPL_CONVERSION_MODE_HIGH_QUALITY,
    Classic = PEAK_IPL_CONVERSION_MODE_CLASSIC,
};

/*!
 *\brief Enum holding the possible pixel format names.
 */
enum class PixelFormatName
{
    Invalid = PEAK_IPL_PIXEL_FORMAT_INVALID,

    BayerGR8 = PEAK_IPL_PIXEL_FORMAT_BAYER_GR_8,
    BayerGR10 = PEAK_IPL_PIXEL_FORMAT_BAYER_GR_10,
    BayerGR12 = PEAK_IPL_PIXEL_FORMAT_BAYER_GR_12,

    BayerRG8 = PEAK_IPL_PIXEL_FORMAT_BAYER_RG_8,
    BayerRG10 = PEAK_IPL_PIXEL_FORMAT_BAYER_RG_10,
    BayerRG12 = PEAK_IPL_PIXEL_FORMAT_BAYER_RG_12,

    BayerGB8 = PEAK_IPL_PIXEL_FORMAT_BAYER_GB_8,
    BayerGB10 = PEAK_IPL_PIXEL_FORMAT_BAYER_GB_10,
    BayerGB12 = PEAK_IPL_PIXEL_FORMAT_BAYER_GB_12,

    BayerBG8 = PEAK_IPL_PIXEL_FORMAT_BAYER_BG_8,
    BayerBG10 = PEAK_IPL_PIXEL_FORMAT_BAYER_BG_10,
    BayerBG12 = PEAK_IPL_PIXEL_FORMAT_BAYER_BG_12,

    Mono8 = PEAK_IPL_PIXEL_FORMAT_MONO_8,
    Mono10 = PEAK_IPL_PIXEL_FORMAT_MONO_10,
    Mono12 = PEAK_IPL_PIXEL_FORMAT_MONO_12,

    RGB8 = PEAK_IPL_PIXEL_FORMAT_RGB_8,
    RGB10 = PEAK_IPL_PIXEL_FORMAT_RGB_10,
    RGB12 = PEAK_IPL_PIXEL_FORMAT_RGB_12,

    BGR8 = PEAK_IPL_PIXEL_FORMAT_BGR_8,
    BGR10 = PEAK_IPL_PIXEL_FORMAT_BGR_10,
    BGR12 = PEAK_IPL_PIXEL_FORMAT_BGR_12,

    RGBa8 = PEAK_IPL_PIXEL_FORMAT_RGBA_8,
    RGBa10 = PEAK_IPL_PIXEL_FORMAT_RGBA_10,
    RGBa12 = PEAK_IPL_PIXEL_FORMAT_RGBA_12,

    BGRa8 = PEAK_IPL_PIXEL_FORMAT_BGRA_8,
    BGRa10 = PEAK_IPL_PIXEL_FORMAT_BGRA_10,
    BGRa12 = PEAK_IPL_PIXEL_FORMAT_BGRA_12,

    BayerBG10p = PEAK_IPL_PIXEL_FORMAT_BAYER_BG_10_PACKED,
    BayerBG12p = PEAK_IPL_PIXEL_FORMAT_BAYER_BG_12_PACKED,

    BayerGB10p = PEAK_IPL_PIXEL_FORMAT_BAYER_GB_10_PACKED,
    BayerGB12p = PEAK_IPL_PIXEL_FORMAT_BAYER_GB_12_PACKED,

    BayerGR10p = PEAK_IPL_PIXEL_FORMAT_BAYER_GR_10_PACKED,
    BayerGR12p = PEAK_IPL_PIXEL_FORMAT_BAYER_GR_12_PACKED,

    BayerRG10p = PEAK_IPL_PIXEL_FORMAT_BAYER_RG_10_PACKED,
    BayerRG12p = PEAK_IPL_PIXEL_FORMAT_BAYER_RG_12_PACKED,

    Mono10p = PEAK_IPL_PIXEL_FORMAT_MONO_10_PACKED,
    Mono12p = PEAK_IPL_PIXEL_FORMAT_MONO_12_PACKED,

    RGB10p32 = PEAK_IPL_PIXEL_FORMAT_RGB_10_PACKED_32,

    BGR10p32 = PEAK_IPL_PIXEL_FORMAT_BGR_10_PACKED_32,

    /*!
     * \attention This pixel format is preliminary and its name and value may change in a future product version.
     */
    BayerRG10g40IDS = PEAK_IPL_PIXEL_FORMAT_BAYER_RG_10_GROUPED_40_IDS,
    /*!
     * \attention This pixel format is preliminary and its name and value may change in a future product version.
     */
    BayerGB10g40IDS = PEAK_IPL_PIXEL_FORMAT_BAYER_GB_10_GROUPED_40_IDS,
    /*!
     * \attention This pixel format is preliminary and its name and value may change in a future product version.
     */
    BayerGR10g40IDS = PEAK_IPL_PIXEL_FORMAT_BAYER_GR_10_GROUPED_40_IDS,
    /*!
     * \attention This pixel format is preliminary and its name and value may change in a future product version.
     */
    BayerBG10g40IDS = PEAK_IPL_PIXEL_FORMAT_BAYER_BG_10_GROUPED_40_IDS,

    /*!
     * \attention This pixel format is preliminary and its name and value may change in a future product version.
     */
    BayerRG12g24IDS = PEAK_IPL_PIXEL_FORMAT_BAYER_RG_12_GROUPED_24_IDS,
    /*!
     * \attention This pixel format is preliminary and its name and value may change in a future product version.
     */
    BayerGB12g24IDS = PEAK_IPL_PIXEL_FORMAT_BAYER_GB_12_GROUPED_24_IDS,
    /*!
     * \attention This pixel format is preliminary and its name and value may change in a future product version.
     */
    BayerGR12g24IDS = PEAK_IPL_PIXEL_FORMAT_BAYER_GR_12_GROUPED_24_IDS,
    /*!
     * \attention This pixel format is preliminary and its name and value may change in a future product version.
     */
    BayerBG12g24IDS = PEAK_IPL_PIXEL_FORMAT_BAYER_BG_12_GROUPED_24_IDS,

    /*!
     * \attention This pixel format is preliminary and its name and value may change in a future product version.
     */
    Mono10g40IDS = PEAK_IPL_PIXEL_FORMAT_MONO_10_GROUPED_40_IDS,
    /*!
     * \attention This pixel format is preliminary and its name and value may change in a future product version.
     */
    Mono12g24IDS = PEAK_IPL_PIXEL_FORMAT_MONO_12_GROUPED_24_IDS,
};

/*!
 *\brief Enum holding the possible endianness (the byte order).
 */
enum class Endianness
{
    Unknown = PEAK_IPL_ENDIANNESS_UNKNOWN,
    LittleEndian = PEAK_IPL_ENDIANNESS_LITTLE_ENDIAN,
    BigEndian = PEAK_IPL_ENDIANNESS_BIG_ENDIAN
};

/*!
 *\brief Enum holding the possible orientation.
 */
enum class Orientation
{
    Horizontal = PEAK_IPL_ORIENTATION_HORIZONTAL,
    Vertical = PEAK_IPL_ORIENTATION_VERTICAL
};

inline std::string ToString(ConversionMode entry)
{
    switch (entry)
    {
    case ConversionMode::Fast:
        return "Fast";
    case ConversionMode::HighQuality:
        return "HighQuality";
    case ConversionMode::Classic:
        return "Classic";
    }

    return "INVALID CONVERSION MODE VALUE";
}

inline std::string ToString(PixelFormatName entry)
{
    switch (entry)
    {
    case PixelFormatName::Invalid:
        return "Invalid";
    case PixelFormatName::BayerGR8:
        return "BayerGR8";
    case PixelFormatName::BayerGR10:
        return "BayerGR10";
    case PixelFormatName::BayerGR12:
        return "BayerGR12";
    case PixelFormatName::BayerRG8:
        return "BayerRG8";
    case PixelFormatName::BayerRG10:
        return "BayerRG10";
    case PixelFormatName::BayerRG12:
        return "BayerRG12";
    case PixelFormatName::BayerGB8:
        return "BayerGB8";
    case PixelFormatName::BayerGB10:
        return "BayerGB10";
    case PixelFormatName::BayerGB12:
        return "BayerGB12";
    case PixelFormatName::BayerBG8:
        return "BayerBG8";
    case PixelFormatName::BayerBG10:
        return "BayerBG10";
    case PixelFormatName::BayerBG12:
        return "BayerBG12";
    case PixelFormatName::Mono8:
        return "Mono8";
    case PixelFormatName::Mono10:
        return "Mono10";
    case PixelFormatName::Mono12:
        return "Mono12";
    case PixelFormatName::RGB8:
        return "RGB8";
    case PixelFormatName::RGB10:
        return "RGB10";
    case PixelFormatName::RGB12:
        return "RGB12";
    case PixelFormatName::BGR8:
        return "BGR8";
    case PixelFormatName::BGR10:
        return "BGR10";
    case PixelFormatName::BGR12:
        return "BGR12";
    case PixelFormatName::RGBa8:
        return "RGBa8";
    case PixelFormatName::RGBa10:
        return "RGBa10";
    case PixelFormatName::RGBa12:
        return "RGBa12";
    case PixelFormatName::BGRa8:
        return "BGRa8";
    case PixelFormatName::BGRa10:
        return "BGRa10";
    case PixelFormatName::BGRa12:
        return "BGRa12";
    case PixelFormatName::BayerBG10p:
        return "BayerBG10p";
    case PixelFormatName::BayerBG12p:
        return "BayerBG12p";
    case PixelFormatName::BayerGB10p:
        return "BayerGB10p";
    case PixelFormatName::BayerGB12p:
        return "BayerGB12p";
    case PixelFormatName::BayerGR10p:
        return "BayerGR10p";
    case PixelFormatName::BayerGR12p:
        return "BayerGR12p";
    case PixelFormatName::BayerRG10p:
        return "BayerRG10p";
    case PixelFormatName::BayerRG12p:
        return "BayerRG12p";
    case PixelFormatName::Mono10p:
        return "Mono10p";
    case PixelFormatName::Mono12p:
        return "Mono12p";
    case PixelFormatName::RGB10p32:
        return "RGB10p32";
    case PixelFormatName::BGR10p32:
        return "BGR10p32";
    case PixelFormatName::BayerRG10g40IDS:
        return "BayerRG10g40IDS";
    case PixelFormatName::BayerGB10g40IDS:
        return "BayerGB10g40IDS";
    case PixelFormatName::BayerGR10g40IDS:
        return "BayerGR10g40IDS";
    case PixelFormatName::BayerBG10g40IDS:
        return "BayerBG10g40IDS";
    case PixelFormatName::BayerRG12g24IDS:
        return "BayerRG12g24IDS";
    case PixelFormatName::BayerGB12g24IDS:
        return "BayerGB12g24IDS";
    case PixelFormatName::BayerGR12g24IDS:
        return "BayerGR12g24IDS";
    case PixelFormatName::BayerBG12g24IDS:
        return "BayerBG12g24IDS";
    case PixelFormatName::Mono10g40IDS:
        return "Mono10g40IDS";
    case PixelFormatName::Mono12g24IDS:
        return "Mono12g24IDS";
    }

    return "INVALID PIXEL FORMAT VALUE";
}

inline std::string ToString(Endianness entry)
{
    switch (entry)
    {
    case Endianness::Unknown:
        return "Unknown";
    case Endianness::LittleEndian:
        return "LittleEndian";
    case Endianness::BigEndian:
        return "BigEndian";
    }

    return "INVALID ENDIANNESS VALUE";
}

inline std::string ToString(Orientation entry)
{
    switch (entry)
    {
    case Orientation::Horizontal:
        return "Horizontal";
    case Orientation::Vertical:
        return "Vertical";
    }

    return "INVALID ORIENTATION VALUE";
}

} /* namespace ipl */
} /* namespace peak */
