/*!
 * \file    peak_ipl_backend.h
 *
 * \author  IDS Imaging Development Systems GmbH
 * \date    2019-05-01
 * \since   1.0
 *
 * Copyright (c) 2019, IDS Imaging Development Systems GmbH. All rights reserved.
 */

#pragma once

#include <peak_ipl/backend/peak_ipl_dll_defines.h>

#ifdef __cplusplus
#    include <cstddef>
#    include <cstdint>

extern "C" {
#else
#    include <stddef.h>
#    include <stdint.h>
#endif

typedef int8_t PEAK_IPL_BOOL8;

/*! The enum holding the possible function return codes. */
enum PEAK_IPL_RETURN_CODE_t
{
    PEAK_IPL_RETURN_CODE_SUCCESS = 0,
    PEAK_IPL_RETURN_CODE_ERROR = 1,
    PEAK_IPL_RETURN_CODE_INVALID_HANDLE = 2,
    PEAK_IPL_RETURN_CODE_IO_ERROR = 3,
    PEAK_IPL_RETURN_CODE_BUFFER_TOO_SMALL = 4,
    PEAK_IPL_RETURN_CODE_INVALID_ARGUMENT = 5,
    PEAK_IPL_RETURN_CODE_OUT_OF_RANGE = 6,
    PEAK_IPL_RETURN_CODE_IMAGE_FORMAT_NOT_SUPPORTED = 7,
    PEAK_IPL_RETURN_CODE_FORMAT_INTERPRETATION_ERROR = 8,

};
typedef int32_t PEAK_IPL_RETURN_CODE;

/*! The enum holding the possible conversion modes. */
enum PEAK_IPL_CONVERSION_MODE_t
{
    PEAK_IPL_CONVERSION_MODE_FAST = 1,
    PEAK_IPL_CONVERSION_MODE_HIGH_QUALITY = 2,
    PEAK_IPL_CONVERSION_MODE_CLASSIC = 3,
};
typedef int32_t PEAK_IPL_CONVERSION_MODE;

/*! The enum holding the possible pixel formats. */
enum PEAK_IPL_PIXEL_FORMAT_t
{
    PEAK_IPL_PIXEL_FORMAT_INVALID = 0,

    PEAK_IPL_PIXEL_FORMAT_BAYER_GR_8 = 0x01080008,
    PEAK_IPL_PIXEL_FORMAT_BAYER_GR_10 = 0x0110000C,
    PEAK_IPL_PIXEL_FORMAT_BAYER_GR_12 = 0x01100010,

    PEAK_IPL_PIXEL_FORMAT_BAYER_RG_8 = 0x01080009,
    PEAK_IPL_PIXEL_FORMAT_BAYER_RG_10 = 0x0110000D,
    PEAK_IPL_PIXEL_FORMAT_BAYER_RG_12 = 0x01100011,

    PEAK_IPL_PIXEL_FORMAT_BAYER_GB_8 = 0x0108000A,
    PEAK_IPL_PIXEL_FORMAT_BAYER_GB_10 = 0x0110000E,
    PEAK_IPL_PIXEL_FORMAT_BAYER_GB_12 = 0x01100012,

    PEAK_IPL_PIXEL_FORMAT_BAYER_BG_8 = 0x0108000B,
    PEAK_IPL_PIXEL_FORMAT_BAYER_BG_10 = 0x0110000F,
    PEAK_IPL_PIXEL_FORMAT_BAYER_BG_12 = 0x01100013,

    PEAK_IPL_PIXEL_FORMAT_MONO_8 = 0x01080001,
    PEAK_IPL_PIXEL_FORMAT_MONO_10 = 0x01100003,
    PEAK_IPL_PIXEL_FORMAT_MONO_12 = 0x01100005,

    PEAK_IPL_PIXEL_FORMAT_RGB_8 = 0x02180014,
    PEAK_IPL_PIXEL_FORMAT_RGB_10 = 0x02300018,
    PEAK_IPL_PIXEL_FORMAT_RGB_12 = 0x0230001A,

    PEAK_IPL_PIXEL_FORMAT_BGR_8 = 0x02180015,
    PEAK_IPL_PIXEL_FORMAT_BGR_10 = 0x02300019,
    PEAK_IPL_PIXEL_FORMAT_BGR_12 = 0x0230001B,

    PEAK_IPL_PIXEL_FORMAT_RGBA_8 = 0x02200016,
    PEAK_IPL_PIXEL_FORMAT_RGBA_10 = 0x0240005F,
    PEAK_IPL_PIXEL_FORMAT_RGBA_12 = 0x02400061,

    PEAK_IPL_PIXEL_FORMAT_BGRA_8 = 0x02200017,
    PEAK_IPL_PIXEL_FORMAT_BGRA_10 = 0x0240004C,
    PEAK_IPL_PIXEL_FORMAT_BGRA_12 = 0x0240004E,

    PEAK_IPL_PIXEL_FORMAT_BAYER_BG_10_PACKED = 0x010A0052,
    PEAK_IPL_PIXEL_FORMAT_BAYER_BG_12_PACKED = 0x010C0053,

    PEAK_IPL_PIXEL_FORMAT_BAYER_GB_10_PACKED = 0x010A0054,
    PEAK_IPL_PIXEL_FORMAT_BAYER_GB_12_PACKED = 0x010C0055,

    PEAK_IPL_PIXEL_FORMAT_BAYER_GR_10_PACKED = 0x010A0056,
    PEAK_IPL_PIXEL_FORMAT_BAYER_GR_12_PACKED = 0x010C0057,

    PEAK_IPL_PIXEL_FORMAT_BAYER_RG_10_PACKED = 0x010A0058,
    PEAK_IPL_PIXEL_FORMAT_BAYER_RG_12_PACKED = 0x010C0059,

    PEAK_IPL_PIXEL_FORMAT_MONO_10_PACKED = 0x010A0046,
    PEAK_IPL_PIXEL_FORMAT_MONO_12_PACKED = 0x010C0047,

    PEAK_IPL_PIXEL_FORMAT_RGB_10_PACKED_32 = 0x0220001D,

    PEAK_IPL_PIXEL_FORMAT_BGR_10_PACKED_32 = 0x0220001E,

    /*!
     * \attention This pixel format is preliminary and its name and value may change in a future product version.
     */
    PEAK_IPL_PIXEL_FORMAT_BAYER_RG_10_GROUPED_40_IDS = 0x40000001,
    /*!
     * \attention This pixel format is preliminary and its name and value may change in a future product version.
     */
    PEAK_IPL_PIXEL_FORMAT_BAYER_GB_10_GROUPED_40_IDS = 0x40000002,
    /*!
     * \attention This pixel format is preliminary and its name and value may change in a future product version.
     */
    PEAK_IPL_PIXEL_FORMAT_BAYER_GR_10_GROUPED_40_IDS = 0x40000003,
    /*!
     * \attention This pixel format is preliminary and its name and value may change in a future product version.
     */
    PEAK_IPL_PIXEL_FORMAT_BAYER_BG_10_GROUPED_40_IDS = 0x40000004,
    /*!
     * \attention This pixel format is preliminary and its name and value may change in a future product version.
     */
    PEAK_IPL_PIXEL_FORMAT_BAYER_RG_12_GROUPED_24_IDS = 0x40000011,
    /*!
     * \attention This pixel format is preliminary and its name and value may change in a future product version.
     */
    PEAK_IPL_PIXEL_FORMAT_BAYER_GB_12_GROUPED_24_IDS = 0x40000012,
    /*!
     * \attention This pixel format is preliminary and its name and value may change in a future product version.
     */
    PEAK_IPL_PIXEL_FORMAT_BAYER_GR_12_GROUPED_24_IDS = 0x40000013,
    /*!
     * \attention This pixel format is preliminary and its name and value may change in a future product version.
     */
    PEAK_IPL_PIXEL_FORMAT_BAYER_BG_12_GROUPED_24_IDS = 0x40000014,
    /*!
     * \attention This pixel format is preliminary and its name and value may change in a future product version.
     */
    PEAK_IPL_PIXEL_FORMAT_MONO_10_GROUPED_40_IDS = 0x4000000f,
    /*!
     * \attention This pixel format is preliminary and its name and value may change in a future product version.
     */
    PEAK_IPL_PIXEL_FORMAT_MONO_12_GROUPED_24_IDS = 0x4000001f,
};
typedef int32_t PEAK_IPL_PIXEL_FORMAT;

/*! The enum holding the possible endianness types. */
enum PEAK_IPL_ENDIANNESS_t
{
    PEAK_IPL_ENDIANNESS_UNKNOWN,
    PEAK_IPL_ENDIANNESS_LITTLE_ENDIAN,
    PEAK_IPL_ENDIANNESS_BIG_ENDIAN
};
typedef int32_t PEAK_IPL_ENDIANNESS;

/*! The enum holding the possible orientations. */
enum PEAK_IPL_ORIENTATION_t
{
    PEAK_IPL_ORIENTATION_HORIZONTAL,
    PEAK_IPL_ORIENTATION_VERTICAL
};
typedef int32_t PEAK_IPL_ORIENTATION;

/*! Sensitivity parameter for the hotpixel correction algorithm */
enum PEAK_IPL_HOTPIXELCORRECTION_SENSITIVITY_t
{
    PEAK_IPL_HOTPIXELCORRECTION_SENSITIVITY_INVALID,
    PEAK_IPL_HOTPIXELCORRECTION_SENSITIVITY_LEVEL1,
    PEAK_IPL_HOTPIXELCORRECTION_SENSITIVITY_LEVEL2,
    PEAK_IPL_HOTPIXELCORRECTION_SENSITIVITY_LEVEL3,
    PEAK_IPL_HOTPIXELCORRECTION_SENSITIVITY_LEVEL4,
    PEAK_IPL_HOTPIXELCORRECTION_SENSITIVITY_LEVEL5
};
typedef int32_t PEAK_IPL_HOTPIXELCORRECTION_SENSITIVITY;

/*! Rotation angle for the image transformer rotation algorithm*/
enum PEAK_IPL_IMAGE_TRANSFORMER_ROTATION_ANGLE_t
{
    LIMGTYPEIMAGE_TRANSFORMER_ROTATION_ANGLE_DEGREE_90_COUNTERCLOCKWISE = 90,
    LIMGTYPEIMAGE_TRANSFORMER_ROTATION_ANGLE_DEGREE_90_CLOCKWISE = 270,
    LIMGTYPEIMAGE_TRANSFORMER_ROTATION_ANGLE_DEGREE_180 = 180
};
typedef uint16_t PEAK_IPL_IMAGE_TRANSFORMER_ROTATION_ANGLE;

/*! Position of a pixel in an image */
struct PEAK_IPL_POINT_2D
{
    size_t x;
    size_t y;
};


struct PEAK_IPL_IMAGE_CONVERTER;
/*! The type of image converter handles. */
typedef struct PEAK_IPL_IMAGE_CONVERTER* PEAK_IPL_IMAGE_CONVERTER_HANDLE;

struct PEAK_IPL_IMAGE_transformer;
/*! The type of image transformer handles. */
typedef struct PEAK_IPL_IMAGE_transformer* PEAK_IPL_IMAGE_TRANSFORMER_HANDLE;

struct PEAK_IPL_ADAPTIVE_HOTPIXEL_CORRECTOR;
/*! The type of adaptive hotpixel corrector handles. */
typedef struct PEAK_IPL_ADAPTIVE_HOTPIXEL_CORRECTOR* PEAK_IPL_ADAPTIVE_HOTPIXEL_CORRECTOR_HANDLE;

struct PEAK_IPL_COLOR_CORRECTOR;
/*! The type of color corrector handles. */
typedef struct PEAK_IPL_COLOR_CORRECTOR* PEAK_IPL_COLOR_CORRECTOR_HANDLE;

struct PEAK_IPL_GAMMA_CORRECTOR;
/*! The type of gamma corrector handles. */
typedef struct PEAK_IPL_GAMMA_CORRECTOR* PEAK_IPL_GAMMA_CORRECTOR_HANDLE;

struct PEAK_IPL_IMAGE;
/*! The type of image handles. */
typedef struct PEAK_IPL_IMAGE* PEAK_IPL_IMAGE_HANDLE;

struct PEAK_IPL_HISTOGRAM;
/*! The type of histogram handles. */
typedef struct PEAK_IPL_HISTOGRAM* PEAK_IPL_HISTOGRAM_HANDLE;

struct PEAK_IPL_PIXEL_LINE;
/*! The type of pixel line handles. */
typedef struct PEAK_IPL_PIXEL_LINE* PEAK_IPL_PIXEL_LINE_HANDLE;


#define PEAK_IPL_C_API PEAK_IPL_PUBLIC PEAK_IPL_RETURN_CODE PEAK_IPL_CALL_CONV

#ifndef PEAK_IPL_DYNAMIC_LOADING
/*!
 * \brief Queries the library major version.
 *
 * \param[out] libraryVersionMajor The major version of the library.
 *
 * \return PEAK_IPL_RETURN_CODE_SUCCESS          Success<br>
 *         PEAK_IPL_RETURN_CODE_INVALID_ARGUMENT libraryVersionMajor is a null pointer
 *
 * \since 1.0
 */
PEAK_IPL_C_API PEAK_IPL_Library_GetVersionMajor(uint32_t* libraryVersionMajor);

/*!
 * \brief Queries the library minor version.
 *
 * \param[out] libraryVersionMinor The minor version of the library.
 *
 * \return PEAK_IPL_RETURN_CODE_SUCCESS          Success<br>
 *         PEAK_IPL_RETURN_CODE_INVALID_ARGUMENT libraryVersionMinor is a null pointer
 *
 * \since 1.0
 */
PEAK_IPL_C_API PEAK_IPL_Library_GetVersionMinor(uint32_t* libraryVersionMinor);

/*!
 * \brief Queries the library subminor version.
 *
 * \param[out] libraryVersionSubminor The subminor version of the library.
 *
 * \return PEAK_IPL_RETURN_CODE_SUCCESS          Success<br>
 *         PEAK_IPL_RETURN_CODE_INVALID_ARGUMENT libraryVersionSubminor is a null pointer
 *
 * \since 1.1
 */
PEAK_IPL_C_API PEAK_IPL_Library_GetVersionSubminor(uint32_t* libraryVersionSubminor);

/*!
 * \brief Queries the last error.
 *
 * This function is normally used by applying a two-step procedure. First of all, you call the function with all
 * arguments except of lastErrorDescription.
 * \code
 *   // Error handling is omitted
 *   PEAK_IPL_RETURN_CODE lastErrorCode = PEAK_IPL_RETURN_CODE_SUCCESS;
 *   size_t size = 0;
 *   PEAK_IPL_GetLastError(&lastErrorCode, NULL, &size);
 * \endcode
 * The function then gives you the last error code and the size of the error description. You could stop now if you only
 * want to query the last error code. If you want to query the error description as well, you have to go on.
 * \code
 *   // Error handling is omitted
 *   char errorDescription[size];
 *   PEAK_IPL_GetLastError(&returnCode, errorDescription, &size);
 * \endcode
 *
 * This two-step procedure may not be necessary if you just pass a buffer big enough for holding the description at the
 * first function call.
 *
 * \param[out]    lastErrorCode            The last function error code.
 * \param[out]    lastErrorDescription     The description for the last error.
 * \param[in,out] lastErrorDescriptionSize IN: Size of the given buffer - OUT: Size of the queried data
 *
 * \return PEAK_IPL_RETURN_CODE_SUCCESS          Success<br>
 *         PEAK_IPL_RETURN_CODE_INVALID_ARGUMENT At least one of the arguments is a null pointer<br>
 *         PEAK_IPL_RETURN_CODE_BUFFER_TOO_SMALL lastErrorDescription is valid but lastErrorDescriptionSize is too small
 *
 * \since 1.0
 */
PEAK_IPL_C_API PEAK_IPL_Library_GetLastError(
    PEAK_IPL_RETURN_CODE* lastErrorCode, char* lastErrorDescription, size_t* lastErrorDescriptionSize);

/*!
 * \brief Creates an image converter.
 *
 * \note To speed up processing image converters maintain internal memory pools to reuse
 * memory instead of allocating new memory for each conversion. The memory is freed when the image converter is
 * destroyed using PEAK_IPL_ImageConverter_Destruct().
 *
 * \param[out] imageConverterHandle The handle to the created image converter.
 *
 * \return PEAK_IPL_RETURN_CODE_SUCCESS Success<br>
 *         PEAK_IPL_RETURN_CODE_ERROR   An internal error has occurred.
 *
 * \since 1.0
 */
PEAK_IPL_C_API PEAK_IPL_ImageConverter_Construct(PEAK_IPL_IMAGE_CONVERTER_HANDLE* imageConverterHandle);

/*!
 * \brief Queries the given image converter's current conversion mode.
 *
 * \param[in]  imageConverterHandle The handle to the image converter of interest.
 * \param[out] conversionMode       The current conversion mode.
 *
 * \return PEAK_IPL_RETURN_CODE_SUCCESS          Success<br>
 *         PEAK_IPL_RETURN_CODE_INVALID_HANDLE   imageConverterHandle is invalid<br>
 *         PEAK_IPL_RETURN_CODE_INVALID_ARGUMENT conversionMode is a null pointer<br>
 *         PEAK_IPL_RETURN_CODE_ERROR            An internal error has occurred.
 *
 * \since 1.0
 */
PEAK_IPL_C_API PEAK_IPL_ImageConverter_GetConversionMode(
    PEAK_IPL_IMAGE_CONVERTER_HANDLE imageConverterHandle, PEAK_IPL_CONVERSION_MODE* conversionMode);

/*!
 * \brief Sets the given image converter's conversion mode.
 *
 * \param[in] imageConverterHandle The handle to the image converter of interest.
 * \param[in] conversionMode       The conversion mode to set.
 *
 * \return PEAK_IPL_RETURN_CODE_SUCCESS        Success<br>
 *         PEAK_IPL_RETURN_CODE_INVALID_HANDLE imageConverterHandle is invalid<br>
 *         PEAK_IPL_RETURN_CODE_ERROR          An internal error has occurred.
 *
 * \since 1.0
 */
PEAK_IPL_C_API PEAK_IPL_ImageConverter_SetConversionMode(
    PEAK_IPL_IMAGE_CONVERTER_HANDLE imageConverterHandle, PEAK_IPL_CONVERSION_MODE conversionMode);

/*!
 * \brief Queries the supported output pixel formats for a given input pixel format.
 *
 * For more details on how to apply the two-step procedure this function requires, see also PEAK_IPL_GetLastError().
 *
 * \param[in]     imageConverterHandle   The handle to the image converter of interest.
 * \param[in]     inputPixelFormat       The input pixel format.
 * \param[out]    outputPixelFormats     The list of supported output pixel formats.
 * \param[in,out] outputPixelFormatsSize IN: Size of the given buffer - OUT: Size of the queried data
 *
 * \return PEAK_IPL_RETURN_CODE_SUCCESS          Success<br>
 *         PEAK_IPL_RETURN_CODE_INVALID_HANDLE   imageConverterHandle is invalid<br>
 *         PEAK_IPL_RETURN_CODE_INVALID_ARGUMENT size is a null pointer<br>
 *         PEAK_IPL_RETURN_CODE_BUFFER_TOO_SMALL outputPixelFormats is valid but size is too small<br>
 *         PEAK_IPL_RETURN_CODE_ERROR            An internal error has occurred.
 *
 * \since 1.0
 */
PEAK_IPL_C_API PEAK_IPL_ImageConverter_GetSupportedOutputPixelFormats(PEAK_IPL_IMAGE_CONVERTER_HANDLE imageConverterHandle,
    PEAK_IPL_PIXEL_FORMAT inputPixelFormat, PEAK_IPL_PIXEL_FORMAT* outputPixelFormats, size_t* outputPixelFormatsSize);

/*!
 * \brief Creates a new image containing the data of the input image converted to the given pixel format.
 *
 * \param[in]  imageConverterHandle The handle to the image converter of interest.
 * \param[in]  inputImageHandle     The handle to the created image.
 * \param[in]  outputPixelFormat    The output pixel format.
 * \param[out] outputImageHandle    The handle associated with the image containing the converted data.
 *
 * \return PEAK_IPL_RETURN_CODE_SUCCESS          Success<br>
 *         PEAK_IPL_RETURN_CODE_INVALID_HANDLE   At least one of imageConverterHandle and inputImageHandle is invalid<br>
 *         PEAK_IPL_RETURN_CODE_INVALID_ARGUMENT outputImageHandle is a null pointer<br>
 *         PEAK_IPL_RETURN_CODE_ERROR            An internal error has occurred.
 *
 * \since 1.0
 */
PEAK_IPL_C_API PEAK_IPL_ImageConverter_Convert(PEAK_IPL_IMAGE_CONVERTER_HANDLE imageConverterHandle,
    PEAK_IPL_IMAGE_HANDLE inputImageHandle, PEAK_IPL_PIXEL_FORMAT outputPixelFormat,
    PEAK_IPL_IMAGE_HANDLE* outputImageHandle);

/*!
 * \brief Creates a new image containing the data of the input image converted to the given pixel format.
 *
 * \param[in]  imageConverterHandle      The handle to the image converter of interest.
 * \param[in]  inputImageHandle          The handle to the input image.
 * \param[in]  outputPixelFormat         The output pixel format.
 * \param[out] outputImageBuffer         Pointer to the destination buffer.
 * \param[in]  outputImageBufferSize     Size of the destination buffer.
 * \param[out] outputImageHandle         The handle associated with the image containing the converted data.
 *
 * \return PEAK_IPL_RETURN_CODE_SUCCESS          Success<br>
 *         PEAK_IPL_RETURN_CODE_INVALID_HANDLE   At least one of imageConverterHandle and inputImageHandle is invalid<br>
 *         PEAK_IPL_RETURN_CODE_INVALID_ARGUMENT At least one of outputImageBuffer and outputImageHandle is a null
 *                                              pointer<br>
 *         PEAK_IPL_RETURN_CODE_BUFFER_TOO_SMALL The output buffer size is too small regarding width, height and
 *                                              pixel format<br>
 *         PEAK_IPL_RETURN_CODE_ERROR            An internal error has occurred.
 *
 * \since 1.1
 */
PEAK_IPL_C_API PEAK_IPL_ImageConverter_ConvertToBuffer(PEAK_IPL_IMAGE_CONVERTER_HANDLE imageConverterHandle,
    PEAK_IPL_IMAGE_HANDLE inputImageHandle, PEAK_IPL_PIXEL_FORMAT outputPixelFormat, uint8_t* outputImageBuffer,
    size_t outputImageBufferSize, PEAK_IPL_IMAGE_HANDLE* outputImageHandle);

/*!
 * \brief Destroys the image converter associated with the given handle.
 *
 * \param[in] imageConverterHandle The handle to the image converter of interest.
 *
 * \return PEAK_IPL_RETURN_CODE_SUCCESS        Success<br>
 *         PEAK_IPL_RETURN_CODE_INVALID_HANDLE imageConverterHandle is invalid<br>
 *         PEAK_IPL_RETURN_CODE_ERROR          An internal error has occurred.
 *
 * \since 1.0
 */
PEAK_IPL_C_API PEAK_IPL_ImageConverter_Destruct(PEAK_IPL_IMAGE_CONVERTER_HANDLE imageConverterHandle);

/*!
 * \brief Creates an adaptive hotpixel corrector algorithm.
 *
 * \param[out] adaptiveHotpixelCorrectorHandle The handle to the created adaptive hotpixel corrector algorithm.
 *
 * \return PEAK_IPL_RETURN_CODE_SUCCESS Success<br>
 *         PEAK_IPL_RETURN_CODE_ERROR   Something went wrong
 *
 * \since 1.0
 */
PEAK_IPL_C_API PEAK_IPL_AdaptiveHotpixelCorrector_Construct(
    PEAK_IPL_ADAPTIVE_HOTPIXEL_CORRECTOR_HANDLE* adaptiveHotpixelCorrectorHandle);

/*!
 * \brief Sets the given adaptive hotpixel corrector algorithm's sensitivity.
 *
 * \param[in] adaptiveHotpixelCorrectorHandle The handle to the adaptive hotpixel corrector algorithm of interest.
 * \param[in] sensitivityLevel                The sensitivity level to set.
 *
 * \return PEAK_IPL_RETURN_CODE_SUCCESS        Success<br>
 *         PEAK_IPL_RETURN_CODE_INVALID_HANDLE adaptiveHotpixelCorrectorHandle is invalid<br>
 *         PEAK_IPL_RETURN_CODE_ERROR          Something went wrong
 *
 * \since 1.0
 */
PEAK_IPL_C_API PEAK_IPL_AdaptiveHotpixelCorrector_SetSensitivity(
    PEAK_IPL_ADAPTIVE_HOTPIXEL_CORRECTOR_HANDLE adaptiveHotpixelCorrectorHandle,
    PEAK_IPL_HOTPIXELCORRECTION_SENSITIVITY sensitivityLevel);
/*!
 * \brief Queries the given adaptive hotpixel corrector algorithm's current sensitivity.
 *
 * \param[in]  adaptiveHotpixelCorrectorHandle The handle to the adaptive hotpixel corrector algorithm of interest.
 * \param[out] sensitivityLevel                The current sensitivity level factor.
 *
 * \return PEAK_IPL_RETURN_CODE_SUCCESS          Success<br>
 *         PEAK_IPL_RETURN_CODE_INVALID_HANDLE   adaptiveHotpixelCorrectorHandle is invalid<br>
 *         PEAK_IPL_RETURN_CODE_INVALID_ARGUMENT sensitivityLevel is a null pointer<br>
 *         PEAK_IPL_RETURN_CODE_ERROR            Something went wrong
 *
 * \since 1.0
 */
PEAK_IPL_C_API PEAK_IPL_AdaptiveHotpixelCorrector_GetSensitivity(
    PEAK_IPL_ADAPTIVE_HOTPIXEL_CORRECTOR_HANDLE adaptiveHotpixelCorrectorHandle,
    PEAK_IPL_HOTPIXELCORRECTION_SENSITIVITY* sensitivityLevel);

/*!
 * \brief Sets the given adaptive hotpixel corrector algorithm's gain factor in percent.
 *
 * \param[in] adaptiveHotpixelCorrectorHandle The handle to the adaptive hotpixel corrector algorithm of interest.
 * \param[in] gainFactorPercent               The gain factor in percent to set.
 *
 * \return PEAK_IPL_RETURN_CODE_SUCCESS        Success<br>
 *         PEAK_IPL_RETURN_CODE_INVALID_HANDLE adaptiveHotpixelCorrectorHandle is invalid<br>
 *         PEAK_IPL_RETURN_CODE_ERROR          Something went wrong
 *
 * \since 1.0
 */
PEAK_IPL_C_API PEAK_IPL_AdaptiveHotpixelCorrector_SetGainFactorPercent(
    PEAK_IPL_ADAPTIVE_HOTPIXEL_CORRECTOR_HANDLE adaptiveHotpixelCorrectorHandle, uint32_t gainFactorPercent);

/*!
 * \brief Queries the given adaptive hotpixel corrector algorithm's current gain factor in percent.
 *
 * \param[in]  adaptiveHotpixelCorrectorHandle The handle to the adaptive hotpixel corrector algorithm of interest.
 * \param[out] gainFactorPercent               The current gain factor in percent.
 *
 * \return PEAK_IPL_RETURN_CODE_SUCCESS          Success<br>
 *         PEAK_IPL_RETURN_CODE_INVALID_HANDLE   adaptiveHotpixelCorrectorHandle is invalid<br>
 *         PEAK_IPL_RETURN_CODE_INVALID_ARGUMENT gainFactorPercent is a null pointer<br>
 *         PEAK_IPL_RETURN_CODE_ERROR            Something went wrong
 *
 * \since 1.0
 */
PEAK_IPL_C_API PEAK_IPL_AdaptiveHotpixelCorrector_GetGainFactorPercent(
    PEAK_IPL_ADAPTIVE_HOTPIXEL_CORRECTOR_HANDLE adaptiveHotpixelCorrectorHandle, uint32_t* gainFactorPercent);

/*!
 * \brief Detects hotpixels in the given image.
 *
 * \param[in] adaptiveHotpixelCorrectorHandle The handle to the adaptive hotpixel corrector algorithm of interest.
 * \param[in] inputImageHandle                The handle to the input image.
 *
 * \return PEAK_IPL_RETURN_CODE_SUCCESS                    Success<br>
 *         PEAK_IPL_RETURN_CODE_INVALID_HANDLE             At least one of adaptiveHotpixelCorrectorHandle and
 *                                                        inputImageHandle is invalid<br>
 *         PEAK_IPL_RETURN_CODE_IMAGE_FORMAT_NOT_SUPPORTED Image associated with inputImageHandle has packed pixel
 *                                                        format<br>
 *         PEAK_IPL_RETURN_CODE_ERROR                      Something went wrong
 *
 * \since 1.0
 * \since 1.2 Will return PEAK_IPL_RETURN_CODE_IMAGE_FORMAT_NOT_SUPPORTED if image associated with inputImageHandle has packed pixel format
 */
PEAK_IPL_C_API PEAK_IPL_AdaptiveHotpixelCorrector_Detect(
    PEAK_IPL_ADAPTIVE_HOTPIXEL_CORRECTOR_HANDLE adaptiveHotpixelCorrectorHandle, PEAK_IPL_IMAGE_HANDLE inputImageHandle);

/*!
 * \brief Queries previously detected hotpixels.
 *
 * For more details on how to apply the two-step procedure this function requires, see also PEAK_IPL_GetLastError().
 *
 * \param[in]     adaptiveHotpixelCorrectorHandle The handle to the adaptive hotpixel corrector algorithm of interest.
 * \param[out]    hotpixels                       The list of hotpixels.
 * \param[in,out] hotpixelsSize                   IN: Size of the given buffer - OUT: Size of the queried data
 *
 * \return PEAK_IPL_RETURN_CODE_SUCCESS          Success<br>
 *         PEAK_IPL_RETURN_CODE_INVALID_HANDLE   adaptiveHotpixelCorrectorHandle is invalid<br>
 *         PEAK_IPL_RETURN_CODE_INVALID_ARGUMENT hotpixelsSize is a null pointer<br>
 *         PEAK_IPL_RETURN_CODE_BUFFER_TOO_SMALL hotpixels is valid but size is too small<br>
 *         PEAK_IPL_RETURN_CODE_ERROR            An internal error has occurred.
 *
 * \since 1.0
 */
PEAK_IPL_C_API PEAK_IPL_AdaptiveHotpixelCorrector_GetHotpixels(
    PEAK_IPL_ADAPTIVE_HOTPIXEL_CORRECTOR_HANDLE adaptiveHotpixelCorrectorHandle, PEAK_IPL_POINT_2D* hotpixels,
    size_t* hotpixelsSize);

/*!
 * \brief Corrects the given hotpixels in the given image.
 *
 * \param[in]  adaptiveHotpixelCorrectorHandle The handle to the adaptive hotpixel corrector algorithm of interest.
 * \param[in]  inputImageHandle                The handle to the input image.
 * \param[in]  hotpixels                       The list of hotpixels to be corrected.
 * \param[in]  hotpixelsSize                   Size of the given hotpixels list.
 * \param[out] outputImageHandle               The handle associated with the image containing the corrected data.
 *
 * \return PEAK_IPL_RETURN_CODE_SUCCESS                    Success<br>
 *         PEAK_IPL_RETURN_CODE_INVALID_HANDLE             At least one of adaptiveHotpixelCorrectorHandle and
 *                                                        inputImageHandle is invalid<br>
 *         PEAK_IPL_RETURN_CODE_INVALID_ARGUMENT           outputImageHandle is a null pointer<br>
 *         PEAK_IPL_RETURN_CODE_IMAGE_FORMAT_NOT_SUPPORTED Image associated with inputImageHandle has packed pixel
 *                                                        format<br>
 *         PEAK_IPL_RETURN_CODE_ERROR                      Something went wrong
 *
 * \since 1.0
 * \since 1.2 Will return PEAK_IPL_RETURN_CODE_IMAGE_FORMAT_NOT_SUPPORTED if image associated with inputImageHandle has packed pixel format
 */
PEAK_IPL_C_API PEAK_IPL_AdaptiveHotpixelCorrector_Correct(
    PEAK_IPL_ADAPTIVE_HOTPIXEL_CORRECTOR_HANDLE adaptiveHotpixelCorrectorHandle, PEAK_IPL_IMAGE_HANDLE inputImageHandle,
    const PEAK_IPL_POINT_2D* hotpixels, size_t hotpixelsSize, PEAK_IPL_IMAGE_HANDLE* outputImageHandle);

/*!
 * \brief Detects and corrects hotpixels in the given image.
 *
 * \param[in]  adaptiveHotpixelCorrectorHandle The handle to the adaptive hotpixel corrector algorithm of interest.
 * \param[in]  inputImageHandle                The handle to the input image.
 * \param[out] outputImageHandle               The handle associated with the image containing the corrected data.
 *
 * \return PEAK_IPL_RETURN_CODE_SUCCESS                    Success<br>
 *         PEAK_IPL_RETURN_CODE_INVALID_HANDLE             At least one of adaptiveHotpixelCorrectorHandle and
 *                                                        inputImageHandle is invalid<br>
 *         PEAK_IPL_RETURN_CODE_INVALID_ARGUMENT           outputImageHandle is a null pointer<br>
 *         PEAK_IPL_RETURN_CODE_IMAGE_FORMAT_NOT_SUPPORTED Image associated with inputImageHandle has packed pixel
 *                                                        format<br>
 *         PEAK_IPL_RETURN_CODE_ERROR                      Something went wrong
 *
 * \since 1.0
 * \since 1.2 Will return PEAK_IPL_RETURN_CODE_IMAGE_FORMAT_NOT_SUPPORTED if image associated with inputImageHandle has packed pixel format
 */
PEAK_IPL_C_API PEAK_IPL_AdaptiveHotpixelCorrector_CorrectAdaptive(
    PEAK_IPL_ADAPTIVE_HOTPIXEL_CORRECTOR_HANDLE adaptiveHotpixelCorrectorHandle, PEAK_IPL_IMAGE_HANDLE inputImageHandle,
    PEAK_IPL_IMAGE_HANDLE* outputImageHandle);

/*!
 * \brief Destroys the adaptive hotpixel corrector algorithm associated with the given handle.
 *
 * \param[in] adaptiveHotpixelCorrectorHandle The handle to the adaptive hotpixel corrector algorithm of interest.
 *
 * \return PEAK_IPL_RETURN_CODE_SUCCESS        Success<br>
 *         PEAK_IPL_RETURN_CODE_INVALID_HANDLE adaptiveHotpixelCorrectorHandle is invalid<br>
 *         PEAK_IPL_RETURN_CODE_ERROR          Something went wrong
 *
 * \since 1.0
 */
PEAK_IPL_C_API PEAK_IPL_AdaptiveHotpixelCorrector_Destruct(
    PEAK_IPL_ADAPTIVE_HOTPIXEL_CORRECTOR_HANDLE adaptiveHotpixelCorrectorHandle);

/*!
 * \brief Creates a color corrector.
 *
 * \param[out] colorCorrectorHandle The handle to the created color corrector.
 *
 * \return IMG_RETURN_CODE_SUCCESS Success<br>
 *         IMG_RETURN_CODE_ERROR   Something went wrong
 *
 * \since 1.0
 */
PEAK_IPL_C_API PEAK_IPL_ColorCorrector_Construct(PEAK_IPL_COLOR_CORRECTOR_HANDLE* colorCorrectorHandle);

/*!
 * \brief Sets the factors of the color correction matrix.
 *
 * \param[in] colorCorrectorHandle      The handle to the color corrector.
 * \param[in] colorCorrectorFactors     The color corrector factors that are set.
 *
 * \return PEAK_IPL_RETURN_CODE_SUCCESS        Success<br>
 *         PEAK_IPL_RETURN_CODE_INVALID_HANDLE colorCorrectorHandle is invalid<br>
 *         PEAK_IPL_RETURN_CODE_ERROR          An internal error has occurred.
 *
 * \since 1.0
 */
PEAK_IPL_C_API PEAK_IPL_ColorCorrector_SetColorCorrectionFactors(
    PEAK_IPL_COLOR_CORRECTOR_HANDLE colorCorrectorHandle, float* colorCorrectorFactors);

/*!
 * \brief Returns the color corrector 3x3 matrix factors and its size
 *
 * \param[in]  colorCorrectorHandle             The handle to the color corrector
 * \param[out] colorCorrectorFactors            The current color corrector factors.
 * \param[out] colorCorrectorFactorsSize        The current color corrector factors size, default 3x3 = 9.
 *
 * \return PEAK_IPL_RETURN_CODE_SUCCESS          Success<br>
 *         PEAK_IPL_RETURN_CODE_INVALID_HANDLE   colorCorrectorHandle is invalid<br>
 *         PEAK_IPL_RETURN_CODE_INVALID_ARGUMENT colorCorrectorFactors and/or colorCorrectorFactorsSize is/are a null
 *                                              pointer<br>
 *         PEAK_IPL_RETURN_CODE_ERROR            Something went wrong
 *
 * \since 1.0
 */
PEAK_IPL_C_API PEAK_IPL_ColorCorrector_GetColorCorrectionFactors(PEAK_IPL_COLOR_CORRECTOR_HANDLE colorCorrectorHandle,
    float* colorCorrectorFactors, size_t* colorCorrectorFactorsSize);

/*!
 * \brief Asks the given color corrector whether it supports the given pixel format.
 *
 * \param[in] colorCorrectorHandle The handle to the color corrector to use.
 * \param[in] pixelFormat The pixel format of interest.
 * \param[out] isPixelFormatSupported The flag telling whether the given pixel format is supported.
 *
 * \return PEAK_IPL_RETURN_CODE_SUCCESS        Success<br>
 *         PEAK_IPL_RETURN_CODE_INVALID_HANDLE colorCorrectorHandle and/or imageHandle are/is invalid<br>
 *         PEAK_IPL_RETURN_CODE_ERROR          An internal error has occurred.
 *
 * \since 1.0
 */
PEAK_IPL_C_API PEAK_IPL_ColorCorrector_GetIsPixelFormatSupported(PEAK_IPL_COLOR_CORRECTOR_HANDLE colorCorrectorHandle,
    PEAK_IPL_PIXEL_FORMAT pixelFormat, PEAK_IPL_BOOL8* isPixelFormatSupported);

/*!
 * \brief Corrects the colors of the given image.
 *
 * \param[in]  colorCorrectorHandle The handle to the color corrector to use.
 * \param[in]  imageHandle          The handle to the image on which the color correction matrix is applied.
 * \param[out] outputImageHandle    The handle to the image on which the color correction matrix was applied.
 *
 * \return PEAK_IPL_RETURN_CODE_SUCCESS                    Success<br>
 *         PEAK_IPL_RETURN_CODE_INVALID_HANDLE             colorCorrectorHandle and/or imageHandle are/is invalid<br>
 *         PEAK_IPL_RETURN_CODE_INVALID_ARGUMENT           imageHandle is a null pointer<br>
 *         PEAK_IPL_RETURN_CODE_IMAGE_FORMAT_NOT_SUPPORTED Image associated with imageHandle has unsupported pixel format
 *                                                        (e.g. packed pixel format)<br>
 *         PEAK_IPL_RETURN_CODE_ERROR                      An internal error has occurred.
 *
 *
 * \since 1.0
 * \since 1.2 Will return PEAK_IPL_RETURN_CODE_IMAGE_FORMAT_NOT_SUPPORTED if image associated with imageHandle has packed pixel format
 */
PEAK_IPL_C_API PEAK_IPL_ColorCorrector_Process(PEAK_IPL_COLOR_CORRECTOR_HANDLE colorCorrectorHandle,
    PEAK_IPL_IMAGE_HANDLE imageHandle, PEAK_IPL_IMAGE_HANDLE* outputImageHandle);

/*!
 * \brief Corrects the colors of the given image.
 *
 * \param[in] colorCorrectorHandle The handle to the color corrector to use.
 * \param[in] imageHandle          The handle to the image on which the color correction matrix is applied.
 *
 * \return PEAK_IPL_RETURN_CODE_SUCCESS                    Success<br>
 *         PEAK_IPL_RETURN_CODE_INVALID_HANDLE             colorCorrectorHandle and/or imageHandle are/is invalid<br>
 *         PEAK_IPL_RETURN_CODE_IMAGE_FORMAT_NOT_SUPPORTED Image associated with imageHandle has unsupported pixel format
 *                                                        (e.g. packed pixel format)<br>
 *         PEAK_IPL_RETURN_CODE_ERROR                      Something went wrong
 *
 * \since 1.0
 * \since 1.2 Will return PEAK_IPL_RETURN_CODE_IMAGE_FORMAT_NOT_SUPPORTED if image associated with imageHandle has packed pixel format
 */
PEAK_IPL_C_API PEAK_IPL_ColorCorrector_ProcessInPlace(
    PEAK_IPL_COLOR_CORRECTOR_HANDLE colorCorrectorHandle, PEAK_IPL_IMAGE_HANDLE imageHandle);

/*!
 * \brief Destroys the color corrector associated with the given handle.
 *
 * \param[in] colorCorrectorHandle The handle to the color corrector to destroy.
 *
 * \return IMG_RETURN_CODE_SUCCESS        Success<br>
 *         IMG_RETURN_CODE_INVALID_HANDLE colorCorrectorHandle is invalid<br>
 *         IMG_RETURN_CODE_ERROR          Something went wrong
 *
 * \since 1.0
 */
PEAK_IPL_C_API PEAK_IPL_ColorCorrector_Destruct(PEAK_IPL_COLOR_CORRECTOR_HANDLE colorCorrectorHandle);

/*!
 * \brief Creates a gamma corrector.
 *
 * \param[out] gammaCorrectorHandle The handle to the created gamma corrector.
 *
 * \return IMG_RETURN_CODE_SUCCESS Success<br>
 *         IMG_RETURN_CODE_ERROR   Something went wrong
 *
 * \since 1.2.2
 */
PEAK_IPL_C_API PEAK_IPL_GammaCorrector_Construct(PEAK_IPL_GAMMA_CORRECTOR_HANDLE* gammaCorrectorHandle);

/*!
 * \brief Sets the value of the gamma correction.
 *
 * \param[in] gammaCorrectorHandle      The handle to the gamma corrector.
 * \param[in] gammaValue                The gamma corrector value that are set.
 *
 * \return PEAK_IPL_RETURN_CODE_SUCCESS        Success<br>
 *         PEAK_IPL_RETURN_CODE_INVALID_HANDLE gammaCorrectorHandle is invalid<br>
 *         PEAK_IPL_RETURN_CODE_ERROR          An internal error has occurred.
 *
 * \since 1.2.2
 */
PEAK_IPL_C_API PEAK_IPL_GammaCorrector_SetGammaCorrectionValue(
    PEAK_IPL_GAMMA_CORRECTOR_HANDLE gammaCorrectorHandle, float* gammaValue);

/*!
 * \brief Returns the gamma corrector value.
 *
 * \param[in]  gammaCorrectorHandle             The handle to the gamma corrector
 * \param[out] gammaValue                       The current gamma corrector value.
 *
 * \return PEAK_IPL_RETURN_CODE_SUCCESS          Success<br>
 *         PEAK_IPL_RETURN_CODE_INVALID_HANDLE   gammaCorrectorHandle is invalid<br>
 *         PEAK_IPL_RETURN_CODE_INVALID_ARGUMENT gammaValue is a null pointer<br>
 *         PEAK_IPL_RETURN_CODE_ERROR            Something went wrong
 *
 * \since 1.2.2
 */
PEAK_IPL_C_API PEAK_IPL_GammaCorrector_GetGammaCorrectionValue(
    PEAK_IPL_GAMMA_CORRECTOR_HANDLE gammaCorrectorHandle, float* gammaValue);

/*!
 * \brief Returns the maximum gamma corrector value which can be set.
 *
 * \param[in]  gammaCorrectorHandle             The handle to the gamma corrector
 * \param[out] gammaMax                         The maximum gamma corrector value which can be set.
 *
 * \return PEAK_IPL_RETURN_CODE_SUCCESS          Success<br>
 *         PEAK_IPL_RETURN_CODE_INVALID_HANDLE   gammaCorrectorHandle is invalid<br>
 *         PEAK_IPL_RETURN_CODE_INVALID_ARGUMENT gammaMax is a null pointer<br>
 *         PEAK_IPL_RETURN_CODE_ERROR            Something went wrong
 *
 * \since 1.2.2
 */
PEAK_IPL_C_API PEAK_IPL_GammaCorrector_GetGammaCorrectionMax(
    PEAK_IPL_GAMMA_CORRECTOR_HANDLE gammaCorrectorHandle, float* gammaMax);

/*!
 * \brief Returns the minimum gamma corrector value which can be set.
 *
 * \param[in]  gammaCorrectorHandle             The handle to the gamma corrector
 * \param[out] gammaMin                         The minimum gamma corrector value which can be set.
 *
 * \return PEAK_IPL_RETURN_CODE_SUCCESS          Success<br>
 *         PEAK_IPL_RETURN_CODE_INVALID_HANDLE   gammaCorrectorHandle is invalid<br>
 *         PEAK_IPL_RETURN_CODE_INVALID_ARGUMENT gammaMin is a null pointer<br>
 *         PEAK_IPL_RETURN_CODE_ERROR            Something went wrong
 *
 * \since 1.2.2
 */
PEAK_IPL_C_API PEAK_IPL_GammaCorrector_GetGammaCorrectionMin(
    PEAK_IPL_GAMMA_CORRECTOR_HANDLE gammaCorrectorHandle, float* gammaMin);

/*!
 * \brief Asks the given gamma corrector whether it supports the given pixel format.
 *
 * \param[in] gammaCorrectorHandle The handle to the gamma corrector to use.
 * \param[in] pixelFormat The pixel format of interest.
 * \param[out] isPixelFormatSupported The flag telling whether the given pixel format is supported.
 *
 * \return PEAK_IPL_RETURN_CODE_SUCCESS        Success<br>
 *         PEAK_IPL_RETURN_CODE_INVALID_HANDLE gammaCorrectorHandle and/or imageHandle are/is invalid<br>
 *         PEAK_IPL_RETURN_CODE_ERROR          An internal error has occurred.
 *
 * \since 1.2.2
 */
PEAK_IPL_C_API PEAK_IPL_GammaCorrector_GetIsPixelFormatSupported(PEAK_IPL_GAMMA_CORRECTOR_HANDLE gammaCorrectorHandle,
    PEAK_IPL_PIXEL_FORMAT pixelFormat, PEAK_IPL_BOOL8* isPixelFormatSupported);

/*!
 * \brief Corrects the gammas of the given image.
 *
 * \param[in]  gammaCorrectorHandle The handle to the gamma corrector to use.
 * \param[in]  imageHandle          The handle to the image on which the gamma correction matrix is applied.
 * \param[out] outputImageHandle    The handle to the image on which the gamma correction matrix was applied.
 *
 * \return PEAK_IPL_RETURN_CODE_SUCCESS                    Success<br>
 *         PEAK_IPL_RETURN_CODE_INVALID_HANDLE             gammaCorrectorHandle and/or imageHandle are/is invalid<br>
 *         PEAK_IPL_RETURN_CODE_INVALID_ARGUMENT           imageHandle is a null pointer<br>
 *         PEAK_IPL_RETURN_CODE_IMAGE_FORMAT_NOT_SUPPORTED Image associated with imageHandle has unsupported pixel
 *                                                        format.<br>
 *         PEAK_IPL_RETURN_CODE_ERROR                      An internal error has occurred.
 *
 *
 * \since 1.2.2
 */
PEAK_IPL_C_API PEAK_IPL_GammaCorrector_Process(PEAK_IPL_GAMMA_CORRECTOR_HANDLE gammaCorrectorHandle,
    PEAK_IPL_IMAGE_HANDLE imageHandle, PEAK_IPL_IMAGE_HANDLE* outputImageHandle);

/*!
 * \brief Corrects the gammas of the given image.
 *
 * \param[in] gammaCorrectorHandle The handle to the gamma corrector to use.
 * \param[in] imageHandle          The handle to the image on which the gamma correction matrix is applied.
 *
 * \return PEAK_IPL_RETURN_CODE_SUCCESS                    Success<br>
 *         PEAK_IPL_RETURN_CODE_INVALID_HANDLE             gammaCorrectorHandle and/or imageHandle are/is invalid<br>
 *         PEAK_IPL_RETURN_CODE_IMAGE_FORMAT_NOT_SUPPORTED Image associated with imageHandle has unsupported pixel
 *                                                        format.<br>
 *         PEAK_IPL_RETURN_CODE_ERROR                      Something went wrong
 *
 * \since 1.2.2
 */
PEAK_IPL_C_API PEAK_IPL_GammaCorrector_ProcessInPlace(
    PEAK_IPL_GAMMA_CORRECTOR_HANDLE gammaCorrectorHandle, PEAK_IPL_IMAGE_HANDLE imageHandle);

/*!
 * \brief Destroys the gamma corrector associated with the given handle.
 *
 * \param[in] gammaCorrectorHandle The handle to the gamma corrector to destroy.
 *
 * \return IMG_RETURN_CODE_SUCCESS        Success<br>
 *         IMG_RETURN_CODE_INVALID_HANDLE gammaCorrectorHandle is invalid<br>
 *         IMG_RETURN_CODE_ERROR          Something went wrong
 *
 * \since 1.2.2
 */
PEAK_IPL_C_API PEAK_IPL_GammaCorrector_Destruct(PEAK_IPL_GAMMA_CORRECTOR_HANDLE gammaCorrectorHandle);

/*!
 * \brief Creates an image.
 *
 * \param[in]  pixelFormat The desired pixel format of the image.
 * \param[in]  width       The desired width of the image.
 * \param[in]  height      The desired height of the image.
 * \param[out] imageHandle The handle to the created image.
 *
 * \return PEAK_IPL_RETURN_CODE_SUCCESS          Success<br>
 *         PEAK_IPL_RETURN_CODE_INVALID_ARGUMENT imageHandle is a null pointer<br>
 *         PEAK_IPL_RETURN_CODE_ERROR            An internal error has occurred.
 *
 * \since 1.0
 */
PEAK_IPL_C_API PEAK_IPL_Image_Construct(
    PEAK_IPL_PIXEL_FORMAT pixelFormat, size_t width, size_t height, PEAK_IPL_IMAGE_HANDLE* imageHandle);

/*!
 * \brief Creates an image from a given buffer.
 *
 * \param[in]  pixelFormat The desired pixel format of the image.
 * \param[in]  buffer      The buffer to use for the image.
 * \param[in]  bufferSize  The size of *buffer.
 * \param[in]  width       The desired width of the image in pixels.
 * \param[in]  height      The desired height of the image in lines.
 * \param[out] imageHandle The handle to the created image.
 *
 * \return PEAK_IPL_RETURN_CODE_SUCCESS          Success<br>
 *         PEAK_IPL_RETURN_CODE_INVALID_ARGUMENT At least one of buffer and imageHandle is a null pointer<br>
 *         PEAK_IPL_RETURN_CODE_ERROR            Something went wrong
 *
 * \since 1.0
 */
PEAK_IPL_C_API PEAK_IPL_Image_ConstructFromBuffer(PEAK_IPL_PIXEL_FORMAT pixelFormat, uint8_t* buffer, uint64_t bufferSize,
    size_t width, size_t height, PEAK_IPL_IMAGE_HANDLE* imageHandle);

/*!
 * \brief Queries the width of the given image.
 *
 * \param[in]  imageHandle The handle to the image of interest.
 * \param[out] width       The width of the image in pixels.
 *
 * \return PEAK_IPL_RETURN_CODE_SUCCESS          Success<br>
 *         PEAK_IPL_RETURN_CODE_INVALID_HANDLE   imageHandle is invalid<br>
 *         PEAK_IPL_RETURN_CODE_INVALID_ARGUMENT width is a null pointer<br>
 *         PEAK_IPL_RETURN_CODE_ERROR            Something  went wrong
 *
 * \since 1.0
 */
PEAK_IPL_C_API PEAK_IPL_Image_GetWidth(PEAK_IPL_IMAGE_HANDLE imageHandle, size_t* width);

/*!
 * \brief Queries the height of the given image.
 *
 * \param[in]  imageHandle The handle to the image of interest.
 * \param[out] height      The height of the image in lines.
 *
 * \return PEAK_IPL_RETURN_CODE_SUCCESS          Success<br>
 *         PEAK_IPL_RETURN_CODE_INVALID_HANDLE   imageHandle is invalid<br>
 *         PEAK_IPL_RETURN_CODE_INVALID_ARGUMENT height is a null pointer<br>
 *         PEAK_IPL_RETURN_CODE_ERROR            Something  went wrong
 *
 * \since 1.0
 */
PEAK_IPL_C_API PEAK_IPL_Image_GetHeight(PEAK_IPL_IMAGE_HANDLE imageHandle, size_t* height);

/*!
 * \brief Queries the pointer to the given pixel position.
 *
 * \param[in]  imageHandle  The handle to the image of interest.
 * \param[in]  xPos         The x position of the pixel of interest.
 * \param[in]  yPos         The y position of the pixel of interest.
 * \param[out] pixelPointer The pointer to the given pixel position.
 *
 * \return PEAK_IPL_RETURN_CODE_SUCCESS          Success<br>
 *         PEAK_IPL_RETURN_CODE_INVALID_HANDLE   imageHandle is invalid<br>
 *         PEAK_IPL_RETURN_CODE_INVALID_ARGUMENT pixelPosition is a null pointer<br>
 *         PEAK_IPL_RETURN_CODE_OUT_OF_RANGE     At least one of xPos and yPos is out of range<br>
 *         PEAK_IPL_RETURN_CODE_ERROR            An internal error has occurred.
 *
 * \since 1.0
 */
PEAK_IPL_C_API PEAK_IPL_Image_GetPixelPointer(
    PEAK_IPL_IMAGE_HANDLE imageHandle, size_t xPos, size_t yPos, uint8_t** pixelPointer);

/*!
 * \brief Queries the size of the given image in number of bytes.
 *
 * \param[in]  imageHandle The handle to the image of interest.
 * \param[out] byteCount   The size of the given image in number of bytes.
 *
 * \return PEAK_IPL_RETURN_CODE_SUCCESS          Success<br>
 *         PEAK_IPL_RETURN_CODE_INVALID_HANDLE   imageHandle is invalid<br>
 *         PEAK_IPL_RETURN_CODE_INVALID_ARGUMENT byteCount is a null pointer<br>
 *         PEAK_IPL_RETURN_CODE_ERROR            An internal error has occurred.
 *
 * \since 1.0
 */
PEAK_IPL_C_API PEAK_IPL_Image_GetByteCount(PEAK_IPL_IMAGE_HANDLE imageHandle, size_t* byteCount);

/*!
 * \brief Queries the pixel format of the given image.
 *
 * \param[in]  imageHandle The handle to the image of interest.
 * \param[out] pixelFormat The pixel format of the given image.
 *
 * \return PEAK_IPL_RETURN_CODE_SUCCESS          Success<br>
 *         PEAK_IPL_RETURN_CODE_INVALID_HANDLE   imageHandle is invalid<br>
 *         PEAK_IPL_RETURN_CODE_INVALID_ARGUMENT pixelFormat is a null pointer<br>
 *         PEAK_IPL_RETURN_CODE_ERROR            An internal error has occurred.
 *
 * \since 1.0
 */
PEAK_IPL_C_API PEAK_IPL_Image_GetPixelFormat(PEAK_IPL_IMAGE_HANDLE imageHandle, PEAK_IPL_PIXEL_FORMAT* pixelFormat);

/*!
 * \brief Queries the data pointer of the given image.
 *
 * \param[in]  imageHandle The handle to the image of interest.
 * \param[out] data        The data pointer of the given image.
 *
 * \return PEAK_IPL_RETURN_CODE_SUCCESS          Success<br>
 *         PEAK_IPL_RETURN_CODE_INVALID_HANDLE   imageHandle is invalid<br>
 *         PEAK_IPL_RETURN_CODE_INVALID_ARGUMENT data is a null pointer<br>
 *         PEAK_IPL_RETURN_CODE_ERROR            An internal error has occurred.
 *
 * \since 1.0
 */
PEAK_IPL_C_API PEAK_IPL_Image_GetData(PEAK_IPL_IMAGE_HANDLE imageHandle, uint8_t** data);

/*!
 * \brief Returns a new created image containing the data of the current image as deep copy.
 *
 * \param[in]  imageHandle       The handle to the image of interest.
 * \param[out] outputImageHandle The handle associated with the image containing the converted data.
 *
 * \return PEAK_IPL_RETURN_CODE_SUCCESS          Success<br>
 *         PEAK_IPL_RETURN_CODE_INVALID_HANDLE   imageHandle is invalid<br>
 *         PEAK_IPL_RETURN_CODE_INVALID_ARGUMENT outputImageHandle is a null pointer<br>
 *         PEAK_IPL_RETURN_CODE_ERROR            An internal error has occurred.
 *
 * \since 1.0
 */
PEAK_IPL_C_API PEAK_IPL_Image_Clone(PEAK_IPL_IMAGE_HANDLE imageHandle, PEAK_IPL_IMAGE_HANDLE* outputImageHandle);

/*!
 * \brief Creates a new image containing the data of the current image converted to the given pixel format.
 *
 * \param[in]  imageHandle       The handle to the image of interest.
 * \param[in]  outputPixelFormat The output pixel format.
 * \param[in]  conversionMode    The conversion mode.
 * \param[out] outputImageHandle The handle associated with the image containing the converted data.
 *
 * \return PEAK_IPL_RETURN_CODE_SUCCESS          Success<br>
 *         PEAK_IPL_RETURN_CODE_INVALID_HANDLE   imageHandle is invalid<br>
 *         PEAK_IPL_RETURN_CODE_INVALID_ARGUMENT outputImageHandle is a null pointer or the mirror direction is
 *                                              invalid<br>
 *         PEAK_IPL_RETURN_CODE_ERROR            An internal error has occurred.
 *
 * \since 1.0
 */
PEAK_IPL_C_API PEAK_IPL_Image_ConvertTo(PEAK_IPL_IMAGE_HANDLE imageHandle, PEAK_IPL_PIXEL_FORMAT outputPixelFormat,
    PEAK_IPL_CONVERSION_MODE conversionMode, PEAK_IPL_IMAGE_HANDLE* outputImageHandle);

/*!
 * \brief Stores the data of the current image converted to the given pixel format into a destination buffer
 *        and creates a new image.
 *
 * \param[in]  imageHandle           The handle to the image of interest.
 * \param[in]  outputPixelFormat     The output pixel format.
 * \param[out] outputImageBuffer     Pointer to the destination buffer.
 * \param[in]  outputImageBufferSize Size of the destination buffer.
 * \param[in]  conversionMode        The conversion mode.
 * \param[out] outputImageHandle     The handle associated with the image containing the converted data.
 *
 * \return PEAK_IPL_RETURN_CODE_SUCCESS          Success<br>
 *         PEAK_IPL_RETURN_CODE_INVALID_HANDLE   imageHandle is invalid<br>
 *         PEAK_IPL_RETURN_CODE_INVALID_ARGUMENT At least one of outputImageBuffer and outputImageHandle is a null
 *                                              pointer<br>
 *         PEAK_IPL_RETURN_CODE_BUFFER_TOO_SMALL The output buffer size is too small regarding width, height and
 *                                              pixel format<br>
 *         PEAK_IPL_RETURN_CODE_ERROR            An internal error has occurred.
 *
 * \since 1.1
 */
PEAK_IPL_C_API PEAK_IPL_Image_ConvertToBuffer(PEAK_IPL_IMAGE_HANDLE imageHandle, PEAK_IPL_PIXEL_FORMAT outputPixelFormat,
    uint8_t* outputImageBuffer, size_t outputImageBufferSize, PEAK_IPL_CONVERSION_MODE conversionMode,
    PEAK_IPL_IMAGE_HANDLE* outputImageHandle);

/*!
 * \brief Destroys the image associated with the given handle.
 *
 * \param[in] imageHandle The handle to the image to destroy.
 *
 * \return PEAK_IPL_RETURN_CODE_SUCCESS        Success<br>
 *         PEAK_IPL_RETURN_CODE_INVALID_HANDLE imageHandle is invalid<br>
 *         PEAK_IPL_RETURN_CODE_ERROR          An internal error has occurred.
 *
 * \since 1.0
 */
PEAK_IPL_C_API PEAK_IPL_Image_Destruct(PEAK_IPL_IMAGE_HANDLE imageHandle);

/*!
 * \brief Creates an image transformer.
 *
 * \note To speed up processing image transformers maintain internal memory pools to reuse
 * memory instead of allocating new memory for each transformation. The memory is freed when the image transformer is
 * destroyed using PEAK_IPL_ImageTransformer_Destruct().
 *
 * \param[out] imageTransformerHandle The handle to the created image transformer.
 *
 * \return PEAK_IPL_RETURN_CODE_SUCCESS Success<br>
 *         PEAK_IPL_RETURN_CODE_ERROR   An internal error has occurred.
 *
 * \since 1.0
 */
PEAK_IPL_C_API PEAK_IPL_ImageTransformer_Construct(PEAK_IPL_IMAGE_TRANSFORMER_HANDLE* imageTransformerHandle);

/*!
 * \brief Creates a new image containing the data of the current image mirrored in up-down direction.
 *
 * If the transformed image is a bayer-format image and the number of rows is even,
 * the format will change. (e.g. BayerBG8 -> BayerGR8)
 *
 * \param[in]  imageTransformerHandle The handle to the image transformer to use.
 * \param[in]  imageHandle            The handle to the created image.
 * \param[out] outputImageHandle      The handle associated with the image containing the converted data.
 *
 * \return PEAK_IPL_RETURN_CODE_SUCCESS                    Success<br>
 *         PEAK_IPL_RETURN_CODE_INVALID_HANDLE             imageHandle is invalid<br>
 *         PEAK_IPL_RETURN_CODE_INVALID_ARGUMENT           outputImageHandle is a null pointer<br>
 *         PEAK_IPL_RETURN_CODE_IMAGE_FORMAT_NOT_SUPPORTED Image associated with imageHandle has packed pixel format<br>
 *         PEAK_IPL_RETURN_CODE_ERROR                      An internal error has occurred.
 *
 * \since 1.0
 * \since 1.2 Will return PEAK_IPL_RETURN_CODE_IMAGE_FORMAT_NOT_SUPPORTED if image associated with imageHandle has packed pixel format
 */
PEAK_IPL_C_API PEAK_IPL_ImageTransformer_MirrorUpDown(PEAK_IPL_IMAGE_TRANSFORMER_HANDLE imageTransformerHandle,
    PEAK_IPL_IMAGE_HANDLE imageHandle, PEAK_IPL_IMAGE_HANDLE* outputImageHandle);

/*!
 * \brief Creates a new image containing the data of the current image mirrored in left-right direction.
 *
 * If the transformed image is a bayer-format image and the number of columns is even,
 * the format will change. (e.g. BayerBG8 -> BayerGB8)
 *
 * \param[in]  imageTransformerHandle The handle to the image transformer to use.
 * \param[in]  imageHandle            The handle to the created image.
 * \param[out] outputImageHandle      The handle associated with the image containing the converted data.
 *
 * \return PEAK_IPL_RETURN_CODE_SUCCESS                    Success<br>
 *         PEAK_IPL_RETURN_CODE_INVALID_HANDLE             imageHandle is invalid<br>
 *         PEAK_IPL_RETURN_CODE_INVALID_ARGUMENT           outputImageHandle is a null pointer<br>
 *         PEAK_IPL_RETURN_CODE_IMAGE_FORMAT_NOT_SUPPORTED Image associated with imageHandle has packed pixel format<br>
 *         PEAK_IPL_RETURN_CODE_ERROR                      An internal error has occurred.
 *
 * \since 1.0
 * \since 1.2 Will return PEAK_IPL_RETURN_CODE_IMAGE_FORMAT_NOT_SUPPORTED if image associated with imageHandle has packed pixel format
 */
PEAK_IPL_C_API PEAK_IPL_ImageTransformer_MirrorLeftRight(PEAK_IPL_IMAGE_TRANSFORMER_HANDLE imageTransformerHandle,
    PEAK_IPL_IMAGE_HANDLE imageHandle, PEAK_IPL_IMAGE_HANDLE* outputImageHandle);

/*!
 * \brief Creates a new image containing the data of the current image mirrored in up-down and left-right direction.
 *
 * If the transformed image is a bayer-format image and the number of rows or columns are even,
 * the format will change. (e.g. BayerBG8 -> BayerRG8)
 *
 * \param[in]  imageTransformerHandle The handle to the image transformer to use.
 * \param[in]  imageHandle            The handle to the created image.
 * \param[out] outputImageHandle      The handle associated with the image containing the converted data.
 *
 * \return PEAK_IPL_RETURN_CODE_SUCCESS                    Success<br>
 *         PEAK_IPL_RETURN_CODE_INVALID_HANDLE             imageHandle is invalid<br>
 *         PEAK_IPL_RETURN_CODE_INVALID_ARGUMENT           outputImageHandle is a null pointer<br>
 *         PEAK_IPL_RETURN_CODE_IMAGE_FORMAT_NOT_SUPPORTED Image associated with imageHandle has packed pixel format<br>
 *         PEAK_IPL_RETURN_CODE_ERROR                      An internal error has occurred.
 *
 * \since 1.0
 * \since 1.2 Will return PEAK_IPL_RETURN_CODE_IMAGE_FORMAT_NOT_SUPPORTED if image associated with imageHandle has packed pixel format
 */
PEAK_IPL_C_API PEAK_IPL_ImageTransformer_MirrorUpDownLeftRight(PEAK_IPL_IMAGE_TRANSFORMER_HANDLE imageTransformerHandle,
    PEAK_IPL_IMAGE_HANDLE imageHandle, PEAK_IPL_IMAGE_HANDLE* outputImageHandle);

/*!
 * \brief Mirrors the image in up-down direction in place i.e. it will change the input image itself.
 *
 * If the transformed image is a bayer-format image and the number of rows is even,
 * the format will change. (e.g. BayerBG8 -> BayerGR8)
 *
 * \param[in]     imageTransformerHandle The handle to the image transformer to use.
 * \param[in,out] imageHandle            The handle to the image to be mirrored.
 *
 * \return PEAK_IPL_RETURN_CODE_SUCCESS                    Success<br>
 *         PEAK_IPL_RETURN_CODE_INVALID_HANDLE             imageTransformerHandle is invalid<br>
 *         PEAK_IPL_RETURN_CODE_IMAGE_FORMAT_NOT_SUPPORTED Image associated with imageHandle has packed pixel format<br>
 *         PEAK_IPL_RETURN_CODE_ERROR                      An internal error has occurred.
 *
 * \since 1.0
 * \since 1.2 Will return PEAK_IPL_RETURN_CODE_IMAGE_FORMAT_NOT_SUPPORTED if image associated with imageHandle has packed pixel format
 */
PEAK_IPL_C_API PEAK_IPL_ImageTransformer_MirrorUpDownInPlace(
    PEAK_IPL_IMAGE_TRANSFORMER_HANDLE imageTransformerHandle, PEAK_IPL_IMAGE_HANDLE imageHandle);

/*!
 * \brief Mirrors the image left-right direction in place i.e. it will change the input image itself.
 *
 * If the transformed image is a bayer-format image and the number of columns is even,
 * the format will change. (e.g. BayerBG8 -> BayerGB8)
 *
 * \param[in]     imageTransformerHandle The handle to the image transformer to use.
 * \param[in,out] imageHandle            The handle to the image to be mirrored.
 *
 * \return PEAK_IPL_RETURN_CODE_SUCCESS                    Success<br>
 *         PEAK_IPL_RETURN_CODE_INVALID_HANDLE             imageTransformerHandle is invalid<br>
 *         PEAK_IPL_RETURN_CODE_IMAGE_FORMAT_NOT_SUPPORTED Image associated with imageHandle has packed pixel format<br>
 *         PEAK_IPL_RETURN_CODE_ERROR                      An internal error has occurred.
 *
 * \since 1.0
 * \since 1.2 Will return PEAK_IPL_RETURN_CODE_IMAGE_FORMAT_NOT_SUPPORTED if image associated with imageHandle has packed pixel format
 */
PEAK_IPL_C_API PEAK_IPL_ImageTransformer_MirrorLeftRightInPlace(
    PEAK_IPL_IMAGE_TRANSFORMER_HANDLE imageTransformerHandle, PEAK_IPL_IMAGE_HANDLE imageHandle);

/*!
 * \brief Mirrors the image in up-down and left-right direction in place i.e. it will change the input image itself.
 *
 * If the transformed image is a bayer-format image and the number of rows or columns are even,
 * the format will change. (e.g. BayerBG8 -> BayerRG8)
 *
 * \param[in]     imageTransformerHandle The handle to the image transformer to use.
 * \param[in,out] imageHandle            The handle to the image to be mirrored.
 *
 * \return PEAK_IPL_RETURN_CODE_SUCCESS                    Success<br>
 *         PEAK_IPL_RETURN_CODE_INVALID_HANDLE             imageTransformerHandle is invalid<br>
 *         PEAK_IPL_RETURN_CODE_IMAGE_FORMAT_NOT_SUPPORTED Image associated with imageHandle has packed pixel format<br>
 *         PEAK_IPL_RETURN_CODE_ERROR                      An internal error has occurred.
 *
 * \since 1.0
 * \since 1.2 Will return PEAK_IPL_RETURN_CODE_IMAGE_FORMAT_NOT_SUPPORTED if image associated with imageHandle has packed pixel format
 */
PEAK_IPL_C_API PEAK_IPL_ImageTransformer_MirrorUpDownLeftRightInPlace(
    PEAK_IPL_IMAGE_TRANSFORMER_HANDLE imageTransformerHandle, PEAK_IPL_IMAGE_HANDLE imageHandle);

/*!
 * \brief Creates a new image containing the data of the current image rotated in direction and
 * value of the rotation angle.
 *
 * If the transformed image is a bayer-format image and the number of rows or columns are even,
 * the format will change. (e.g. 90_COUNTERCLOCKWISE & an image with even width: BayerGB8 -> BayerBG8)
 *
 * \param[in]  imageTransformerHandle The handle to the image transformer to use.
 * \param[in]  imageHandle            The handle to the created image.
 * \param[in]  rotationAngle          The rotation angle for the image transformer rotation algorithm
 * \param[out] outputImageHandle      The handle associated with the image containing the rotated data.
 *
 * \return PEAK_IPL_RETURN_CODE_SUCCESS                    Success<br>
 *         PEAK_IPL_RETURN_CODE_INVALID_HANDLE             imageHandle is invalid<br>
 *         PEAK_IPL_RETURN_CODE_INVALID_ARGUMENT           outputImageHandle is a null pointer<br>
 *         PEAK_IPL_RETURN_CODE_IMAGE_FORMAT_NOT_SUPPORTED Image associated with imageHandle has packed pixel format<br>
 *         PEAK_IPL_RETURN_CODE_ERROR                      An internal error has occurred.
 *
 * \since 1.0
 * \since 1.2 Will return PEAK_IPL_RETURN_CODE_IMAGE_FORMAT_NOT_SUPPORTED if image associated with imageHandle has packed pixel format
 */
PEAK_IPL_C_API PEAK_IPL_ImageTransformer_Rotate(PEAK_IPL_IMAGE_TRANSFORMER_HANDLE imageTransformerHandle,
    PEAK_IPL_IMAGE_HANDLE imageHandle, PEAK_IPL_IMAGE_HANDLE* outputImageHandle,
    PEAK_IPL_IMAGE_TRANSFORMER_ROTATION_ANGLE_t rotationAngle);

/*!
 * \brief Rotates the image in in direction and value of the rotation angle in place
 * i.e. it will change the input image itself.
 *
 * If the transformed image is a bayer-format image and the number of rows or columns are even,
 * the format will change. (e.g. 90_COUNTERCLOCKWISE & an image with even width: BayerGB8 -> BayerBG8)
 *
 * \param[in]     imageTransformerHandle The handle to the image transformer to use.
 * \param[in]  rotationAngle          The rotation angle for the image transformer rotation algorithm
 * \param[in,out] imageHandle            The handle to the image to be mirrored.
 *
 * \return PEAK_IPL_RETURN_CODE_SUCCESS                    Success<br>
 *         PEAK_IPL_RETURN_CODE_INVALID_HANDLE             imageTransformerHandle is invalid<br>
 *         PEAK_IPL_RETURN_CODE_IMAGE_FORMAT_NOT_SUPPORTED Image associated with imageHandle has packed pixel format<br>
 *         PEAK_IPL_RETURN_CODE_ERROR                      An internal error has occurred.
 *
 * \since 1.0
 * \since 1.2 Will return PEAK_IPL_RETURN_CODE_IMAGE_FORMAT_NOT_SUPPORTED if image associated with imageHandle has packed pixel format
 */
PEAK_IPL_C_API PEAK_IPL_ImageTransformer_RotateInPlace(PEAK_IPL_IMAGE_TRANSFORMER_HANDLE imageTransformerHandle,
    PEAK_IPL_IMAGE_HANDLE imageHandle, PEAK_IPL_IMAGE_TRANSFORMER_ROTATION_ANGLE_t rotationAngle);

/*!
 * \brief Destroys the image transformer associated with the given handle.
 *
 * \param[in] imageTransformerHandle The handle to the image converter of interest.
 *
 * \return PEAK_IPL_RETURN_CODE_SUCCESS        Success<br>
 *         PEAK_IPL_RETURN_CODE_INVALID_HANDLE imageTransformerHandle is invalid<br>
 *         PEAK_IPL_RETURN_CODE_ERROR          An internal error has occurred.
 *
 * \since 1.0
 */
PEAK_IPL_C_API PEAK_IPL_ImageTransformer_Destruct(PEAK_IPL_IMAGE_TRANSFORMER_HANDLE imageTransformerHandle);

/*!
 * \brief Creates a histogram from image data.
 *
 * \param[in]  imageHandle     The handle to the image of interest.
 * \param[out] histogramHandle The handle to the created histogram.
 *
 * \return PEAK_IPL_RETURN_CODE_SUCCESS                    Success<br>
 *         PEAK_IPL_RETURN_CODE_INVALID_HANDLE             imageHandle is invalid<br>
 *         PEAK_IPL_RETURN_CODE_INVALID_ARGUMENT           histogramHandle is a null pointer<br>
 *         PEAK_IPL_RETURN_CODE_IMAGE_FORMAT_NOT_SUPPORTED Image associated with imageHandle has packed pixel format<br>
 *         PEAK_IPL_RETURN_CODE_ERROR                      An internal error has occurred.
 *
 * \since 1.0
 * \since 1.2 Will return PEAK_IPL_RETURN_CODE_IMAGE_FORMAT_NOT_SUPPORTED if image associated with imageHandle has packed pixel format
 */
PEAK_IPL_C_API PEAK_IPL_Histogram_Construct(PEAK_IPL_IMAGE_HANDLE imageHandle, PEAK_IPL_HISTOGRAM_HANDLE* histogramHandle);

/*!
 * \brief Queries the pixel format of the given histogram.
 *
 * \param[in]  histogramHandle The handle to the histogram of interest.
 * \param[out] pixelFormat     The pixel format.
 *
 * \return PEAK_IPL_RETURN_CODE_SUCCESS          Success<br>
 *         PEAK_IPL_RETURN_CODE_INVALID_HANDLE   histogramHandle is invalid<br>
 *         PEAK_IPL_RETURN_CODE_INVALID_ARGUMENT pixelFormat is a null pointer<br>
 *         PEAK_IPL_RETURN_CODE_ERROR            An internal error has occurred.
 *
 * \since 1.0
 */
PEAK_IPL_C_API PEAK_IPL_Histogram_GetPixelFormat(
    PEAK_IPL_HISTOGRAM_HANDLE histogramHandle, PEAK_IPL_PIXEL_FORMAT* pixelFormat);

/*!
 * \brief Queries the number of the given histogram's channels.
 *
 * \param[in]  histogramHandle The handle to the histogram of interest.
 * \param[out] numChannels     The number of channels.
 *
 * \return PEAK_IPL_RETURN_CODE_SUCCESS          Success<br>
 *         PEAK_IPL_RETURN_CODE_INVALID_HANDLE   histogramHandle is invalid<br>
 *         PEAK_IPL_RETURN_CODE_INVALID_ARGUMENT numChannels is a null pointer<br>
 *         PEAK_IPL_RETURN_CODE_ERROR            An internal error has occurred.
 *
 * \since 1.0
 */
PEAK_IPL_C_API PEAK_IPL_Histogram_GetNumChannels(PEAK_IPL_HISTOGRAM_HANDLE histogramHandle, size_t* numChannels);

/*!
 * \brief Queries the pixel sum for the channel with the given index.
 *
 * \param[in]  histogramHandle The handle to the histogram of interest.
 * \param[in]  channelIndex    The index of the channel of interest.
 * \param[out] pixelSum        The pixel sum of the given channel.
 *
 * \return PEAK_IPL_RETURN_CODE_SUCCESS          Success<br>
 *         PEAK_IPL_RETURN_CODE_INVALID_HANDLE   histogramHandle is invalid<br>
 *         PEAK_IPL_RETURN_CODE_INVALID_ARGUMENT pixelSum is a null pointer<br>
 *         PEAK_IPL_RETURN_CODE_OUT_OF_RANGE     channelIndex is out of range<br>
 *         PEAK_IPL_RETURN_CODE_ERROR            An internal error has occurred.
 *
 * \since 1.0
 */
PEAK_IPL_C_API PEAK_IPL_Histogram_GetPixelSumForChannel(
    PEAK_IPL_HISTOGRAM_HANDLE histogramHandle, size_t channelIndex, uint64_t* pixelSum);

/*!
 * \brief Queries the pixel count for the channel with the given index.
 *
 * \param[in]  histogramHandle The handle to the histogram of interest.
 * \param[in]  channelIndex    The index of the channel of interest.
 * \param[out] pixelCount      The pixel count of the given channel.
 *
 * \return PEAK_IPL_RETURN_CODE_SUCCESS          Success<br>
 *         PEAK_IPL_RETURN_CODE_INVALID_HANDLE   histogramHandle is invalid<br>
 *         PEAK_IPL_RETURN_CODE_INVALID_ARGUMENT pixelCount is a null pointer<br>
 *         PEAK_IPL_RETURN_CODE_OUT_OF_RANGE     channelIndex is out of range<br>
 *         PEAK_IPL_RETURN_CODE_ERROR            An internal error has occurred.
 *
 * \since 1.0
 */
PEAK_IPL_C_API PEAK_IPL_Histogram_GetPixelCountForChannel(
    PEAK_IPL_HISTOGRAM_HANDLE histogramHandle, size_t channelIndex, uint64_t* pixelCount);

/*!
 * \brief Queries the bin list of the channel with the given index.
 *
 * For more details on how to apply the two-step procedure this function requires, see also PEAK_IPL_GetLastError().
 *
 * \param[in]     histogramHandle The handle to the histogram of interest.
 * \param[in]     channelIndex    The index of the channel of interest.
 * \param[out]    bins            The bin list of the given channel.
 * \param[in,out] binsSize        IN: Size of the given buffer - OUT: Size of the queried data
 *
 * \return PEAK_IPL_RETURN_CODE_SUCCESS          Success<br>
 *         PEAK_IPL_RETURN_CODE_INVALID_HANDLE   histogramHandle is invalid<br>
 *         PEAK_IPL_RETURN_CODE_INVALID_ARGUMENT At least one of binList and binsSize is a null pointer<br>
 *         PEAK_IPL_RETURN_CODE_OUT_OF_RANGE     channelIndex is out of range<br>
 *         PEAK_IPL_RETURN_CODE_BUFFER_TOO_SMALL binList is valid but binsSize is too small<br>
 *         PEAK_IPL_RETURN_CODE_ERROR            An internal error has occurred.
 *
 * \since 1.0
 */
PEAK_IPL_C_API PEAK_IPL_Histogram_GetBinsForChannel(
    PEAK_IPL_HISTOGRAM_HANDLE histogramHandle, size_t channelIndex, uint64_t* bins, size_t* binsSize);

/*!
 * \brief Destroys the histogram associated with the given handle.
 *
 * \param[in] histogramHandle The handle to the histogram to destroy.
 *
 * \return PEAK_IPL_RETURN_CODE_SUCCESS        Success<br>
 *         PEAK_IPL_RETURN_CODE_INVALID_HANDLE histogramHandle is invalid<br>
 *         PEAK_IPL_RETURN_CODE_ERROR          An internal error has occurred.
 *
 * \since 1.0
 */
PEAK_IPL_C_API PEAK_IPL_Histogram_Destruct(PEAK_IPL_HISTOGRAM_HANDLE histogramHandle);

/*!
 * \brief Creates a pixel line depending on the given orientation.
 *
 * A pixel line is a whole line or a whole column of the image's pixels.
 *
 * \param[in] imageHandle        The image to create the pixel line from.
 * \param[in] orientation        The orientation of the line.
 * \param[in] offset             The offset from the reference border.
 *                               The reference border depends on the given orientation
                                 (vertical: left - horizontal: top).
 * \param[out] pixelLineHandle The handle to the created pixel line.
 *
 * \return PEAK_IPL_RETURN_CODE_SUCCESS                    Success<br>
 *         PEAK_IPL_RETURN_CODE_INVALID_HANDLE             imageHandle is invalid<br>
 *         PEAK_IPL_RETURN_CODE_INVALID_ARGUMENT           pixelLineHandle is a null pointer<br>
 *         PEAK_IPL_RETURN_CODE_IMAGE_FORMAT_NOT_SUPPORTED Image associated with imageHandle has packed pixel format<br>
 *         PEAK_IPL_RETURN_CODE_OUT_OF_RANGE               offset is out of range<br>
 *         PEAK_IPL_RETURN_CODE_ERROR                      An internal error has occurred.
 *
 * \since 1.0
 * \since 1.2 Will return PEAK_IPL_RETURN_CODE_IMAGE_FORMAT_NOT_SUPPORTED if image associated with inputImageHandle has
 *            packed pixel format
 */
PEAK_IPL_C_API PEAK_IPL_PixelLine_Construct(PEAK_IPL_IMAGE_HANDLE imageHandle, PEAK_IPL_ORIENTATION orientation,
    size_t offset, PEAK_IPL_PIXEL_LINE_HANDLE* pixelLineHandle);

/*!
 * \brief Queries the pixel format of the given pixel line.
 *
 * \param[in]  pixelLineHandle The handle to the pixel line of interest.
 * \param[out] pixelFormat       The pixel format.
 *
 * \return PEAK_IPL_RETURN_CODE_SUCCESS          Success<br>
 *         PEAK_IPL_RETURN_CODE_INVALID_HANDLE   pixelLineHandle is invalid<br>
 *         PEAK_IPL_RETURN_CODE_INVALID_ARGUMENT pixelFormat is a null pointer<br>
 *         PEAK_IPL_RETURN_CODE_ERROR            An internal error has occurred.
 *
 * \since 1.0
 */
PEAK_IPL_C_API PEAK_IPL_PixelLine_GetPixelFormat(
    PEAK_IPL_PIXEL_LINE_HANDLE pixelLineHandle, PEAK_IPL_PIXEL_FORMAT* pixelFormat);

/*!
 * \brief Queries the orientation of the given pixel line.
 *
 * \param[in]  pixelLineHandle The handle to the pixel line of interest.
 * \param[out] orientation       The orientation.
 *
 * \return PEAK_IPL_RETURN_CODE_SUCCESS          Success<br>
 *         PEAK_IPL_RETURN_CODE_INVALID_HANDLE   pixelLineHandle is invalid<br>
 *         PEAK_IPL_RETURN_CODE_INVALID_ARGUMENT orientation is a null pointer<br>
 *         PEAK_IPL_RETURN_CODE_ERROR            An internal error has occurred.
 *
 * \since 1.0
 */
PEAK_IPL_C_API PEAK_IPL_PixelLine_GetOrientation(
    PEAK_IPL_PIXEL_LINE_HANDLE pixelLineHandle, PEAK_IPL_ORIENTATION* orientation);

/*!
 * \brief Queries the offset of the given pixel line.
 *
 * \param[in]  pixelLineHandle The handle to the pixel line of interest.
 * \param[out] offset            The offset.
 *                               The reference border depends on the orientation of the collection line
                                 (vertical: left - horizontal: top).
 *
 * \return PEAK_IPL_RETURN_CODE_SUCCESS          Success<br>
 *         PEAK_IPL_RETURN_CODE_INVALID_HANDLE   pixelLineHandle is invalid<br>
 *         PEAK_IPL_RETURN_CODE_INVALID_ARGUMENT offset is a null pointer<br>
 *         PEAK_IPL_RETURN_CODE_ERROR            An internal error has occurred.
 *
 * \since 1.0
 */
PEAK_IPL_C_API PEAK_IPL_PixelLine_GetOffset(PEAK_IPL_PIXEL_LINE_HANDLE pixelLineHandle, size_t* offset);

/*!
 * \brief Queries the number of the given pixel line's channels.
 *
 * \param[in]  pixelLineHandle The handle to the pixel line of interest.
 * \param[out] numChannels       The number of channels.
 *
 * \return PEAK_IPL_RETURN_CODE_SUCCESS          Success<br>
 *         PEAK_IPL_RETURN_CODE_INVALID_HANDLE   pixelLineHandle is invalid<br>
 *         PEAK_IPL_RETURN_CODE_INVALID_ARGUMENT numChannels is a null pointer<br>
 *         PEAK_IPL_RETURN_CODE_ERROR            An internal error has occurred.
 *
 * \since 1.0
 */
PEAK_IPL_C_API PEAK_IPL_PixelLine_GetNumChannels(PEAK_IPL_PIXEL_LINE_HANDLE pixelLineHandle, size_t* numChannels);

/*!
 * \brief Queries the value list for the channel with the given index.
 *
 * For more details on how to apply the two-step procedure this function requires, see also PEAK_IPL_GetLastError().
 *
 * \param[in]     pixelLineHandle The handle to the pixel line of interest.
 * \param[in]     channelIndex      The index of the channel of interest.
 * \param[out]    values            The value list of the given channel.
 * \param[in,out] valuesSize        IN: Size of the given buffer - OUT: Size of the queried data
 *
 * \return PEAK_IPL_RETURN_CODE_SUCCESS          Success<br>
 *         PEAK_IPL_RETURN_CODE_INVALID_HANDLE   pixelLineHandle is invalid<br>
 *         PEAK_IPL_RETURN_CODE_INVALID_ARGUMENT At least one of values and valuesSize is a null pointer<br>
 *         PEAK_IPL_RETURN_CODE_OUT_OF_RANGE     channelIndex is out of range<br>
 *         PEAK_IPL_RETURN_CODE_BUFFER_TOO_SMALL values is valid but valuesSize is too small<br>
 *         PEAK_IPL_RETURN_CODE_ERROR            An internal error has occurred.
 *
 * \since 1.0
 */
PEAK_IPL_C_API PEAK_IPL_PixelLine_GetValuesForChannel(
    PEAK_IPL_PIXEL_LINE_HANDLE pixelLineHandle, size_t channelIndex, uint32_t* values, size_t* valuesSize);

/*!
 * \brief Destroys the pixel line associated with the given handle.
 *
 * \param[in] pixelLineHandle The handle to the pixel line to destroy.
 *
 * \return PEAK_IPL_RETURN_CODE_SUCCESS        Success<br>
 *         PEAK_IPL_RETURN_CODE_INVALID_HANDLE pixelLineHandle is invalid<br>
 *         PEAK_IPL_RETURN_CODE_ERROR          An internal error has occurred.
 *
 * \since 1.0
 */
PEAK_IPL_C_API PEAK_IPL_PixelLine_Destruct(PEAK_IPL_PIXEL_LINE_HANDLE pixelLineHandle);

/*!
 * \brief Queries the number of channels of the given pixel format.
 *
 * \param[in]  pixelFormat The pixel format of interest.
 * \param[out] numChannels The number of channels.
 *
 * \return PEAK_IPL_RETURN_CODE_SUCCESS          Success<br>
 *         PEAK_IPL_RETURN_CODE_INVALID_ARGUMENT numChannels is a null pointer<br>
 *         PEAK_IPL_RETURN_CODE_ERROR            An internal error has occurred.
 *
 * \since 1.0
 */
PEAK_IPL_C_API PEAK_IPL_PixelFormat_GetNumChannels(PEAK_IPL_PIXEL_FORMAT pixelFormat, size_t* numChannels);

/*!
 * \brief Queries the number of significant bits per pixel per channel of the given pixel format.
 *
 * \param[in]  pixelFormat        The pixel format of interest.
 * \param[out] numSignificantBits The number of significant bits per pixel per channel.
 *
 * \return PEAK_IPL_RETURN_CODE_SUCCESS          Success<br>
 *         PEAK_IPL_RETURN_CODE_INVALID_ARGUMENT numSignificantBits is a null pointer<br>
 *         PEAK_IPL_RETURN_CODE_ERROR            An internal error has occurred.
 *
 * \since 1.0
 */
PEAK_IPL_C_API PEAK_IPL_PixelFormat_GetNumSignificantBitsPerChannel(
    PEAK_IPL_PIXEL_FORMAT pixelFormat, size_t* numSignificantBits);

/*!
 * \brief Queries the number of storage bits per pixel per channel of the given pixel format.
 *
 * \param[in]  pixelFormat    The pixel format of interest.
 * \param[out] numStorageBits The number of storage bits per pixel per channel.
 *
 * \return PEAK_IPL_RETURN_CODE_SUCCESS          Success<br>
 *         PEAK_IPL_RETURN_CODE_INVALID_ARGUMENT numStorageBits is a null pointer<br>
 *         PEAK_IPL_RETURN_CODE_ERROR            An internal error has occurred.
 *
 * \since 1.0
 */
PEAK_IPL_C_API PEAK_IPL_PixelFormat_GetNumStorageBitsPerChannel(PEAK_IPL_PIXEL_FORMAT pixelFormat, size_t* numStorageBits);

/*!
 * \brief Queries the maximum value of one pixel channel of the given pixel format.
 *
 * \param[in]  pixelFormat         The pixel format of interest.
 * \param[out] channelMaximumValue The maximum value of one pixel channel.
 *
 * \return PEAK_IPL_RETURN_CODE_SUCCESS          Success<br>
 *         PEAK_IPL_RETURN_CODE_INVALID_ARGUMENT channelMaximumValue is a null pointer<br>
 *         PEAK_IPL_RETURN_CODE_ERROR            An internal error has occurred.
 *
 * \since 1.0
 */
PEAK_IPL_C_API PEAK_IPL_PixelFormat_GetMaximumValuePerChannel(
    PEAK_IPL_PIXEL_FORMAT pixelFormat, uint32_t* channelMaximumValue);

/*!
 * \brief Queries the number of significant bits per pixel of the given pixel format.
 *
 * \param[in]  pixelFormat        The pixel format of interest.
 * \param[out] numSignificantBits The number of significant bits per pixel.
 *
 * \return PEAK_IPL_RETURN_CODE_SUCCESS          Success<br>
 *         PEAK_IPL_RETURN_CODE_INVALID_ARGUMENT numSignificantBits is a null pointer<br>
 *         PEAK_IPL_RETURN_CODE_ERROR            An internal error has occurred.
 *
 * \since 1.0
 */
PEAK_IPL_C_API PEAK_IPL_PixelFormat_GetNumSignificantBitsPerPixel(
    PEAK_IPL_PIXEL_FORMAT pixelFormat, size_t* numSignificantBits);

/*!
 * \brief Queries the number of storage bits per pixel of the given pixel format.
 *
 * \param[in]  pixelFormat    The pixel format of interest.
 * \param[out] numStorageBits The number of significant bits per pixel.
 *
 * \return PEAK_IPL_RETURN_CODE_SUCCESS          Success<br>
 *         PEAK_IPL_RETURN_CODE_INVALID_ARGUMENT numStorageBits is a null pointer<br>
 *         PEAK_IPL_RETURN_CODE_ERROR            An internal error has occurred.
 *
 * \since 1.0
 */
PEAK_IPL_C_API PEAK_IPL_PixelFormat_GetNumStorageBitsPerPixel(PEAK_IPL_PIXEL_FORMAT pixelFormat, size_t* numStorageBits);

/*!
 * \brief Queries the endianness of the given pixel format.
 *
 * \param[in]  pixelFormat The pixel format of interest.
 * \param[out] endianness  The endianness.
 *
 * \return PEAK_IPL_RETURN_CODE_SUCCESS          Success<br>
 *         PEAK_IPL_RETURN_CODE_INVALID_ARGUMENT endianness is a null pointer<br>
 *         PEAK_IPL_RETURN_CODE_ERROR            An internal error has occurred.
 *
 * \since 1.0
 */
PEAK_IPL_C_API PEAK_IPL_PixelFormat_GetEndianness(PEAK_IPL_PIXEL_FORMAT pixelFormat, PEAK_IPL_ENDIANNESS* endianness);

/*!
 * \brief Calculates the storage size of the given number of pixels of the given pixel format in bytes.
 *
 * \param[in]  pixelFormat The pixel format of interest.
 * \param[in]  numPixels   The number of pixels.
 * \param[out] size        The size of the pixels in bytes.
 *
 * \return PEAK_IPL_RETURN_CODE_SUCCESS          Success<br>
 *         PEAK_IPL_RETURN_CODE_INVALID_ARGUMENT size is a null pointer<br>
 *         PEAK_IPL_RETURN_CODE_ERROR            An internal error has occurred.
 *
 * \since 1.0
 */
PEAK_IPL_C_API PEAK_IPL_PixelFormat_CalculateStorageSizeOfPixels(
    PEAK_IPL_PIXEL_FORMAT pixelFormat, uint64_t numPixels, uint64_t* size);

/*!
 * \brief Writes an image as raw data binary file.
 *
 * This is supported by all formats.
 *
 * \param[in] inputImageHandle The handle to the image to save.
 * \param[in] filePath         The file to write the given image to.
 * \param[in] filePathSize     The size of the given string.
 *
 * \return PEAK_IPL_RETURN_CODE_SUCCESS                    Success<br>
 *         PEAK_IPL_RETURN_CODE_INVALID_ARGUMENT           filePath is a null pointer<br>
 *         PEAK_IPL_RETURN_CODE_IO_ERROR                   Errors during file access e.g. no permissions on this file
 *         PEAK_IPL_RETURN_CODE_IMAGE_FORMAT_NOT_SUPPORTED A file type is not supported for this image pixel format
 *         PEAK_IPL_RETURN_CODE_ERROR                      Something else went wrong
 *
 * \since 1.0
 * \since 1.2 Will return PEAK_IPL_RETURN_CODE_IMAGE_FORMAT_NOT_SUPPORTED if image associated with inputImageHandle has packed pixel format
 */
PEAK_IPL_C_API PEAK_IPL_ImageWriter_WriteAsRAW(
    PEAK_IPL_IMAGE_HANDLE inputImageHandle, const char* filePath, size_t filePathSize);

/*!
 * \brief Writes an image as BMP image file.
 *
 * Supported for the following formats:
 * MONO_8, MONO_10, MONO_12, RGB_8, RGB_10, BGR_8, BGR_10, RGBA_8, BGRA_8, RGB_10_PACKED_32, BGR_10_PACKED_32
 * Written as MONO:
 * BAYER_GR_8, BAYER_RG_8, BAYER_GB_8, BAYER_BG_8, BAYER_GR_10, BAYER_RG_10, BAYER_GB_10, BAYER_BG_10
 *
 * \param[in] inputImageHandle The handle to the image to save.
 * \param[in] filePath         The file to write the given image to.
 * \param[in] filePathSize     The size of the given string.
 *
 * \return PEAK_IPL_RETURN_CODE_SUCCESS                    Success<br>
 *         PEAK_IPL_RETURN_CODE_INVALID_ARGUMENT           filePath is a null pointer<br>
 *         PEAK_IPL_RETURN_CODE_IO_ERROR                   Errors during file access e.g. no permissions on this file
 *         PEAK_IPL_RETURN_CODE_IMAGE_FORMAT_NOT_SUPPORTED A file type is not supported for this image pixel format
 *         PEAK_IPL_RETURN_CODE_ERROR                      Something else went wrong
 *
 * \since 1.0
 * \since 1.2 Will return PEAK_IPL_RETURN_CODE_IMAGE_FORMAT_NOT_SUPPORTED if image associated with inputImageHandle has packed pixel format
 */
PEAK_IPL_C_API PEAK_IPL_ImageWriter_WriteAsBMP(
    PEAK_IPL_IMAGE_HANDLE inputImageHandle, const char* filePath, size_t filePathSize);

/*!
 * \brief Writes an image as PNG image file.
 *
 * Supported for following formats:
 * MONO_8, MONO_10, MONO_12, RGB_8, RGB_10, RGB_12, RGBA_8, RGBA_10, RGBA_12
 * Written as MONO:
 * BAYER_GR_8, BAYER_RG_8, BAYER_GB_8, BAYER_BG_8, BAYER_GR_10, BAYER_RG_10, BAYER_GB_10, BAYER_BG_10, BAYER_GR_12,
 * BAYER_RG_12, BAYER_GB_12, BAYER_BG_12
 * Written as RGB:
 * BGR_8, BGR_10, BGR_12, BGRA_8, BGRA_10, BGRA_12
 *
 * \param[in] inputImageHandle The handle to the image to save.
 * \param[in] quality          Specifies the output image quality from 0 to 100.
 * \param[in] filePath         The file to write the given image to.
 * \param[in] filePathSize     The size of the given string.
 *
 * \return PEAK_IPL_RETURN_CODE_SUCCESS                    Success<br>
 *         PEAK_IPL_RETURN_CODE_INVALID_ARGUMENT           filePath is a null pointer<br>
 *         PEAK_IPL_RETURN_CODE_IO_ERROR                   Errors during file access e.g. no permissions on this file
 *         PEAK_IPL_RETURN_CODE_IMAGE_FORMAT_NOT_SUPPORTED A file type is not supported for this image pixel format
 *         PEAK_IPL_RETURN_CODE_ERROR                      Something else went wrong
 *
 * \since 1.0
 * \since 1.2 Will return PEAK_IPL_RETURN_CODE_IMAGE_FORMAT_NOT_SUPPORTED if image associated with inputImageHandle has packed pixel format
 */
PEAK_IPL_C_API PEAK_IPL_ImageWriter_WriteAsPNG(
    PEAK_IPL_IMAGE_HANDLE inputImageHandle, uint32_t quality, const char* filePath, size_t filePathSize);

/*!
 * \brief Writes an image as JPEG image file.
 *
 * Supported for the following formats:
 * MONO_8, RGB_8, BGR_8, RGBA_8, BGRA_8
 * Written as MONO:
 * BAYER_GR_8, BAYER_RG_8, BAYER_GB_8, BAYER_BG_8
 *
 * \param[in] inputImageHandle The handle to the image to save.
 * \param[in] quality          Specifies the output image quality from 0 to 100.
 * \param[in] filePath         The file to write the given image to.
 * \param[in] filePathSize     The size of the given string.
 *
 * \return PEAK_IPL_RETURN_CODE_SUCCESS                    Success<br>
 *         PEAK_IPL_RETURN_CODE_INVALID_ARGUMENT           filePath is a null pointer<br>
 *         PEAK_IPL_RETURN_CODE_IO_ERROR                   Errors during file access e.g. no permissions on this file
 *         PEAK_IPL_RETURN_CODE_IMAGE_FORMAT_NOT_SUPPORTED A file type is not supported for this image pixel format
 *         PEAK_IPL_RETURN_CODE_ERROR                      Something else went wrong
 *
 * \since 1.0
 * \since 1.2 Will return PEAK_IPL_RETURN_CODE_IMAGE_FORMAT_NOT_SUPPORTED if image associated with inputImageHandle has packed pixel format
 */
PEAK_IPL_C_API PEAK_IPL_ImageWriter_WriteAsJPG(
    PEAK_IPL_IMAGE_HANDLE inputImageHandle, uint32_t quality, const char* filePath, size_t filePathSize);

/*!
 * \brief Writes an image to file and selects the format by file ending. In case the format has parameters, the default
 *        parameters are chosen.
 *
 * \param[in] inputImageHandle The handle to the image to save.
 * \param[in] filePath         The file to write the given image to.
 * \param[in] filePathSize     The size of the given string.
 *
 * \return PEAK_IPL_RETURN_CODE_SUCCESS                    Success<br>
 *         PEAK_IPL_RETURN_CODE_INVALID_ARGUMENT           filePath is a null pointer<br>
 *         PEAK_IPL_RETURN_CODE_IO_ERROR                   Errors during file access e.g. no permissions on this file
 *         PEAK_IPL_RETURN_CODE_IMAGE_FORMAT_NOT_SUPPORTED A file type is not supported for this image pixel format
 *         PEAK_IPL_RETURN_CODE_ERROR                      Something else went wrong
 *
 * \since 1.0
 * \since 1.2 Will return PEAK_IPL_RETURN_CODE_IMAGE_FORMAT_NOT_SUPPORTED if image associated with inputImageHandle has packed pixel format
 */
PEAK_IPL_C_API PEAK_IPL_ImageWriter_Write(
    PEAK_IPL_IMAGE_HANDLE inputImageHandle, const char* filePath, size_t filePathSize);

/*!
 * \brief Reads an image from a file. Image type is determined by the file ending.
 *
 * \param[in]  filePath          The file to read from.
 * \param[in]  filePathSize      The size of the given string.
 * \param[out] outputImageHandle The handle to the read image.
 *
 * \return PEAK_IPL_RETURN_CODE_SUCCESS                    Success<br>
 *         PEAK_IPL_RETURN_CODE_INVALID_ARGUMENT           filePath and/or outputImageHandle are/is a null pointer<br>
 *         PEAK_IPL_RETURN_CODE_IO_ERROR                   Errors during file access e.g. no permissions on this file
 *         PEAK_IPL_RETURN_CODE_IMAGE_FORMAT_NOT_SUPPORTED An image format of this file is not supported
 *         PEAK_IPL_RETURN_CODE_ERROR                      Something went wrong
 *
 * \since 1.0
 */
PEAK_IPL_C_API PEAK_IPL_ImageReader_Read(
    const char* filePath, size_t filePathSize, PEAK_IPL_IMAGE_HANDLE* outputImageHandle);

/*!
 * \brief Reads an image from a file. Image type is determined by the file ending.
 *
 * \param[in]  filePath          The file to read from.
 * \param[in]  filePathSize      The size of the given string.
 * \param[in]  pixelFormat       The pixel format of the data to read.
 * \param[out] outputImageHandle The handle to read image.
 *
 * \return PEAK_IPL_RETURN_CODE_SUCCESS                     Success<br>
 *         PEAK_IPL_RETURN_CODE_INVALID_ARGUMENT            filePath and/or outputImageHandle are/is a null pointer<br>
 *         PEAK_IPL_RETURN_CODE_IO_ERROR                    Errors during file access e.g. no permissions on this file
 *         PEAK_IPL_RETURN_CODE_IMAGE_FORMAT_NOT_SUPPORTED  An image format of this file is not supported
 *         PEAK_IPL_RETURN_CODE_FORMAT_INTERPRETATION_ERROR Can not interpret this file with the given pixel format
 *         PEAK_IPL_RETURN_CODE_ERROR                       Something went wrong
 *
 * \since 1.0
 * \since 1.2 Will return PEAK_IPL_RETURN_CODE_IMAGE_FORMAT_NOT_SUPPORTED if pixelFormat is packed pixel format
 */
PEAK_IPL_C_API PEAK_IPL_ImageReaderRead_ReadAsPixelFormat(const char* filePath, size_t filePathSize,
    PEAK_IPL_PIXEL_FORMAT pixelFormat, PEAK_IPL_IMAGE_HANDLE* outputImageHandle);

#    ifdef __cplusplus
} // extern "C"
#    endif

#else
#    ifdef __cplusplus
} // extern "C"
#    endif

#    include <peak_ipl/backend/peak_ipl_dynamic_loader.h>
#endif // !PEAK_IPL_DYNAMIC_LOADING
