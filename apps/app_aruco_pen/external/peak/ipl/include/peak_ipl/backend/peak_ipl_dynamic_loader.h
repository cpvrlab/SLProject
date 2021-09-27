/*!
 * \file    peak_ipl_dynamic_loader.h
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
#include <cstdint>

#ifdef __linux__
    #include <dlfcn.h>
#else
    #include <vector>
    #include <windows.h>
    #include <tchar.h>
#endif
 
#include <stdexcept>

namespace peak
{
namespace ipl
{
namespace dynamic
{

typedef PEAK_IPL_RETURN_CODE (*dyn_PEAK_IPL_Library_GetVersionMajor)(uint32_t * libraryVersionMajor);
typedef PEAK_IPL_RETURN_CODE (*dyn_PEAK_IPL_Library_GetVersionMinor)(uint32_t * libraryVersionMinor);
typedef PEAK_IPL_RETURN_CODE (*dyn_PEAK_IPL_Library_GetVersionSubminor)(uint32_t * libraryVersionSubminor);
typedef PEAK_IPL_RETURN_CODE (*dyn_PEAK_IPL_Library_GetLastError)(PEAK_IPL_RETURN_CODE * lastErrorCode, char * lastErrorDescription, size_t * lastErrorDescriptionSize);
typedef PEAK_IPL_RETURN_CODE (*dyn_PEAK_IPL_ImageConverter_Construct)(PEAK_IPL_IMAGE_CONVERTER_HANDLE * imageConverterHandle);
typedef PEAK_IPL_RETURN_CODE (*dyn_PEAK_IPL_ImageConverter_GetConversionMode)(PEAK_IPL_IMAGE_CONVERTER_HANDLE imageConverterHandle, PEAK_IPL_CONVERSION_MODE * conversionMode);
typedef PEAK_IPL_RETURN_CODE (*dyn_PEAK_IPL_ImageConverter_SetConversionMode)(PEAK_IPL_IMAGE_CONVERTER_HANDLE imageConverterHandle, PEAK_IPL_CONVERSION_MODE conversionMode);
typedef PEAK_IPL_RETURN_CODE (*dyn_PEAK_IPL_ImageConverter_GetSupportedOutputPixelFormats)(PEAK_IPL_IMAGE_CONVERTER_HANDLE imageConverterHandle, PEAK_IPL_PIXEL_FORMAT inputPixelFormat, PEAK_IPL_PIXEL_FORMAT * outputPixelFormats, size_t * outputPixelFormatsSize);
typedef PEAK_IPL_RETURN_CODE (*dyn_PEAK_IPL_ImageConverter_Convert)(PEAK_IPL_IMAGE_CONVERTER_HANDLE imageConverterHandle, PEAK_IPL_IMAGE_HANDLE inputImageHandle, PEAK_IPL_PIXEL_FORMAT outputPixelFormat, PEAK_IPL_IMAGE_HANDLE * outputImageHandle);
typedef PEAK_IPL_RETURN_CODE (*dyn_PEAK_IPL_ImageConverter_ConvertToBuffer)(PEAK_IPL_IMAGE_CONVERTER_HANDLE imageConverterHandle, PEAK_IPL_IMAGE_HANDLE inputImageHandle, PEAK_IPL_PIXEL_FORMAT outputPixelFormat, uint8_t * outputImageBuffer, size_t outputImageBufferSize, PEAK_IPL_IMAGE_HANDLE * outputImageHandle);
typedef PEAK_IPL_RETURN_CODE (*dyn_PEAK_IPL_ImageConverter_Destruct)(PEAK_IPL_IMAGE_CONVERTER_HANDLE imageConverterHandle);
typedef PEAK_IPL_RETURN_CODE (*dyn_PEAK_IPL_AdaptiveHotpixelCorrector_Construct)(PEAK_IPL_ADAPTIVE_HOTPIXEL_CORRECTOR_HANDLE * adaptiveHotpixelCorrectorHandle);
typedef PEAK_IPL_RETURN_CODE (*dyn_PEAK_IPL_AdaptiveHotpixelCorrector_SetSensitivity)(PEAK_IPL_ADAPTIVE_HOTPIXEL_CORRECTOR_HANDLE adaptiveHotpixelCorrectorHandle, PEAK_IPL_HOTPIXELCORRECTION_SENSITIVITY sensitivityLevel);
typedef PEAK_IPL_RETURN_CODE (*dyn_PEAK_IPL_AdaptiveHotpixelCorrector_GetSensitivity)(PEAK_IPL_ADAPTIVE_HOTPIXEL_CORRECTOR_HANDLE adaptiveHotpixelCorrectorHandle, PEAK_IPL_HOTPIXELCORRECTION_SENSITIVITY * sensitivityLevel);
typedef PEAK_IPL_RETURN_CODE (*dyn_PEAK_IPL_AdaptiveHotpixelCorrector_SetGainFactorPercent)(PEAK_IPL_ADAPTIVE_HOTPIXEL_CORRECTOR_HANDLE adaptiveHotpixelCorrectorHandle, uint32_t gainFactorPercent);
typedef PEAK_IPL_RETURN_CODE (*dyn_PEAK_IPL_AdaptiveHotpixelCorrector_GetGainFactorPercent)(PEAK_IPL_ADAPTIVE_HOTPIXEL_CORRECTOR_HANDLE adaptiveHotpixelCorrectorHandle, uint32_t * gainFactorPercent);
typedef PEAK_IPL_RETURN_CODE (*dyn_PEAK_IPL_AdaptiveHotpixelCorrector_Detect)(PEAK_IPL_ADAPTIVE_HOTPIXEL_CORRECTOR_HANDLE adaptiveHotpixelCorrectorHandle, PEAK_IPL_IMAGE_HANDLE inputImageHandle);
typedef PEAK_IPL_RETURN_CODE (*dyn_PEAK_IPL_AdaptiveHotpixelCorrector_GetHotpixels)(PEAK_IPL_ADAPTIVE_HOTPIXEL_CORRECTOR_HANDLE adaptiveHotpixelCorrectorHandle, PEAK_IPL_POINT_2D * hotpixels, size_t * hotpixelsSize);
typedef PEAK_IPL_RETURN_CODE (*dyn_PEAK_IPL_AdaptiveHotpixelCorrector_Correct)(PEAK_IPL_ADAPTIVE_HOTPIXEL_CORRECTOR_HANDLE adaptiveHotpixelCorrectorHandle, PEAK_IPL_IMAGE_HANDLE inputImageHandle, const PEAK_IPL_POINT_2D * hotpixels, size_t hotpixelsSize, PEAK_IPL_IMAGE_HANDLE * outputImageHandle);
typedef PEAK_IPL_RETURN_CODE (*dyn_PEAK_IPL_AdaptiveHotpixelCorrector_CorrectAdaptive)(PEAK_IPL_ADAPTIVE_HOTPIXEL_CORRECTOR_HANDLE adaptiveHotpixelCorrectorHandle, PEAK_IPL_IMAGE_HANDLE inputImageHandle, PEAK_IPL_IMAGE_HANDLE * outputImageHandle);
typedef PEAK_IPL_RETURN_CODE (*dyn_PEAK_IPL_AdaptiveHotpixelCorrector_Destruct)(PEAK_IPL_ADAPTIVE_HOTPIXEL_CORRECTOR_HANDLE adaptiveHotpixelCorrectorHandle);
typedef PEAK_IPL_RETURN_CODE (*dyn_PEAK_IPL_ColorCorrector_Construct)(PEAK_IPL_COLOR_CORRECTOR_HANDLE * colorCorrectorHandle);
typedef PEAK_IPL_RETURN_CODE (*dyn_PEAK_IPL_ColorCorrector_SetColorCorrectionFactors)(PEAK_IPL_COLOR_CORRECTOR_HANDLE colorCorrectorHandle, float * colorCorrectorFactors);
typedef PEAK_IPL_RETURN_CODE (*dyn_PEAK_IPL_ColorCorrector_GetColorCorrectionFactors)(PEAK_IPL_COLOR_CORRECTOR_HANDLE colorCorrectorHandle, float * colorCorrectorFactors, size_t * colorCorrectorFactorsSize);
typedef PEAK_IPL_RETURN_CODE (*dyn_PEAK_IPL_ColorCorrector_GetIsPixelFormatSupported)(PEAK_IPL_COLOR_CORRECTOR_HANDLE colorCorrectorHandle, PEAK_IPL_PIXEL_FORMAT pixelFormat, PEAK_IPL_BOOL8 * isPixelFormatSupported);
typedef PEAK_IPL_RETURN_CODE (*dyn_PEAK_IPL_ColorCorrector_Process)(PEAK_IPL_COLOR_CORRECTOR_HANDLE colorCorrectorHandle, PEAK_IPL_IMAGE_HANDLE imageHandle, PEAK_IPL_IMAGE_HANDLE * outputImageHandle);
typedef PEAK_IPL_RETURN_CODE (*dyn_PEAK_IPL_ColorCorrector_ProcessInPlace)(PEAK_IPL_COLOR_CORRECTOR_HANDLE colorCorrectorHandle, PEAK_IPL_IMAGE_HANDLE imageHandle);
typedef PEAK_IPL_RETURN_CODE (*dyn_PEAK_IPL_ColorCorrector_Destruct)(PEAK_IPL_COLOR_CORRECTOR_HANDLE colorCorrectorHandle);
typedef PEAK_IPL_RETURN_CODE (*dyn_PEAK_IPL_GammaCorrector_Construct)(PEAK_IPL_GAMMA_CORRECTOR_HANDLE * gammaCorrectorHandle);
typedef PEAK_IPL_RETURN_CODE (*dyn_PEAK_IPL_GammaCorrector_SetGammaCorrectionValue)(PEAK_IPL_GAMMA_CORRECTOR_HANDLE gammaCorrectorHandle, float * gammaValue);
typedef PEAK_IPL_RETURN_CODE (*dyn_PEAK_IPL_GammaCorrector_GetGammaCorrectionValue)(PEAK_IPL_GAMMA_CORRECTOR_HANDLE gammaCorrectorHandle, float * gammaValue);
typedef PEAK_IPL_RETURN_CODE (*dyn_PEAK_IPL_GammaCorrector_GetGammaCorrectionMax)(PEAK_IPL_GAMMA_CORRECTOR_HANDLE gammaCorrectorHandle, float * gammaMax);
typedef PEAK_IPL_RETURN_CODE (*dyn_PEAK_IPL_GammaCorrector_GetGammaCorrectionMin)(PEAK_IPL_GAMMA_CORRECTOR_HANDLE gammaCorrectorHandle, float * gammaMin);
typedef PEAK_IPL_RETURN_CODE (*dyn_PEAK_IPL_GammaCorrector_GetIsPixelFormatSupported)(PEAK_IPL_GAMMA_CORRECTOR_HANDLE gammaCorrectorHandle, PEAK_IPL_PIXEL_FORMAT pixelFormat, PEAK_IPL_BOOL8 * isPixelFormatSupported);
typedef PEAK_IPL_RETURN_CODE (*dyn_PEAK_IPL_GammaCorrector_Process)(PEAK_IPL_GAMMA_CORRECTOR_HANDLE gammaCorrectorHandle, PEAK_IPL_IMAGE_HANDLE imageHandle, PEAK_IPL_IMAGE_HANDLE * outputImageHandle);
typedef PEAK_IPL_RETURN_CODE (*dyn_PEAK_IPL_GammaCorrector_ProcessInPlace)(PEAK_IPL_GAMMA_CORRECTOR_HANDLE gammaCorrectorHandle, PEAK_IPL_IMAGE_HANDLE imageHandle);
typedef PEAK_IPL_RETURN_CODE (*dyn_PEAK_IPL_GammaCorrector_Destruct)(PEAK_IPL_GAMMA_CORRECTOR_HANDLE gammaCorrectorHandle);
typedef PEAK_IPL_RETURN_CODE (*dyn_PEAK_IPL_Image_Construct)(PEAK_IPL_PIXEL_FORMAT pixelFormat, size_t width, size_t height, PEAK_IPL_IMAGE_HANDLE * imageHandle);
typedef PEAK_IPL_RETURN_CODE (*dyn_PEAK_IPL_Image_ConstructFromBuffer)(PEAK_IPL_PIXEL_FORMAT pixelFormat, uint8_t * buffer, uint64_t bufferSize, size_t width, size_t height, PEAK_IPL_IMAGE_HANDLE * imageHandle);
typedef PEAK_IPL_RETURN_CODE (*dyn_PEAK_IPL_Image_GetWidth)(PEAK_IPL_IMAGE_HANDLE imageHandle, size_t * width);
typedef PEAK_IPL_RETURN_CODE (*dyn_PEAK_IPL_Image_GetHeight)(PEAK_IPL_IMAGE_HANDLE imageHandle, size_t * height);
typedef PEAK_IPL_RETURN_CODE (*dyn_PEAK_IPL_Image_GetPixelPointer)(PEAK_IPL_IMAGE_HANDLE imageHandle, size_t xPos, size_t yPos, uint8_t * * pixelPointer);
typedef PEAK_IPL_RETURN_CODE (*dyn_PEAK_IPL_Image_GetByteCount)(PEAK_IPL_IMAGE_HANDLE imageHandle, size_t * byteCount);
typedef PEAK_IPL_RETURN_CODE (*dyn_PEAK_IPL_Image_GetPixelFormat)(PEAK_IPL_IMAGE_HANDLE imageHandle, PEAK_IPL_PIXEL_FORMAT * pixelFormat);
typedef PEAK_IPL_RETURN_CODE (*dyn_PEAK_IPL_Image_GetData)(PEAK_IPL_IMAGE_HANDLE imageHandle, uint8_t * * data);
typedef PEAK_IPL_RETURN_CODE (*dyn_PEAK_IPL_Image_Clone)(PEAK_IPL_IMAGE_HANDLE imageHandle, PEAK_IPL_IMAGE_HANDLE * outputImageHandle);
typedef PEAK_IPL_RETURN_CODE (*dyn_PEAK_IPL_Image_ConvertTo)(PEAK_IPL_IMAGE_HANDLE imageHandle, PEAK_IPL_PIXEL_FORMAT outputPixelFormat, PEAK_IPL_CONVERSION_MODE conversionMode, PEAK_IPL_IMAGE_HANDLE * outputImageHandle);
typedef PEAK_IPL_RETURN_CODE (*dyn_PEAK_IPL_Image_ConvertToBuffer)(PEAK_IPL_IMAGE_HANDLE imageHandle, PEAK_IPL_PIXEL_FORMAT outputPixelFormat, uint8_t * outputImageBuffer, size_t outputImageBufferSize, PEAK_IPL_CONVERSION_MODE conversionMode, PEAK_IPL_IMAGE_HANDLE * outputImageHandle);
typedef PEAK_IPL_RETURN_CODE (*dyn_PEAK_IPL_Image_Destruct)(PEAK_IPL_IMAGE_HANDLE imageHandle);
typedef PEAK_IPL_RETURN_CODE (*dyn_PEAK_IPL_ImageTransformer_Construct)(PEAK_IPL_IMAGE_TRANSFORMER_HANDLE * imageTransformerHandle);
typedef PEAK_IPL_RETURN_CODE (*dyn_PEAK_IPL_ImageTransformer_MirrorUpDown)(PEAK_IPL_IMAGE_TRANSFORMER_HANDLE imageTransformerHandle, PEAK_IPL_IMAGE_HANDLE imageHandle, PEAK_IPL_IMAGE_HANDLE * outputImageHandle);
typedef PEAK_IPL_RETURN_CODE (*dyn_PEAK_IPL_ImageTransformer_MirrorLeftRight)(PEAK_IPL_IMAGE_TRANSFORMER_HANDLE imageTransformerHandle, PEAK_IPL_IMAGE_HANDLE imageHandle, PEAK_IPL_IMAGE_HANDLE * outputImageHandle);
typedef PEAK_IPL_RETURN_CODE (*dyn_PEAK_IPL_ImageTransformer_MirrorUpDownLeftRight)(PEAK_IPL_IMAGE_TRANSFORMER_HANDLE imageTransformerHandle, PEAK_IPL_IMAGE_HANDLE imageHandle, PEAK_IPL_IMAGE_HANDLE * outputImageHandle);
typedef PEAK_IPL_RETURN_CODE (*dyn_PEAK_IPL_ImageTransformer_MirrorUpDownInPlace)(PEAK_IPL_IMAGE_TRANSFORMER_HANDLE imageTransformerHandle, PEAK_IPL_IMAGE_HANDLE imageHandle);
typedef PEAK_IPL_RETURN_CODE (*dyn_PEAK_IPL_ImageTransformer_MirrorLeftRightInPlace)(PEAK_IPL_IMAGE_TRANSFORMER_HANDLE imageTransformerHandle, PEAK_IPL_IMAGE_HANDLE imageHandle);
typedef PEAK_IPL_RETURN_CODE (*dyn_PEAK_IPL_ImageTransformer_MirrorUpDownLeftRightInPlace)(PEAK_IPL_IMAGE_TRANSFORMER_HANDLE imageTransformerHandle, PEAK_IPL_IMAGE_HANDLE imageHandle);
typedef PEAK_IPL_RETURN_CODE (*dyn_PEAK_IPL_ImageTransformer_Rotate)(PEAK_IPL_IMAGE_TRANSFORMER_HANDLE imageTransformerHandle, PEAK_IPL_IMAGE_HANDLE imageHandle, PEAK_IPL_IMAGE_HANDLE * outputImageHandle, PEAK_IPL_IMAGE_TRANSFORMER_ROTATION_ANGLE_t rotationAngle);
typedef PEAK_IPL_RETURN_CODE (*dyn_PEAK_IPL_ImageTransformer_RotateInPlace)(PEAK_IPL_IMAGE_TRANSFORMER_HANDLE imageTransformerHandle, PEAK_IPL_IMAGE_HANDLE imageHandle, PEAK_IPL_IMAGE_TRANSFORMER_ROTATION_ANGLE_t rotationAngle);
typedef PEAK_IPL_RETURN_CODE (*dyn_PEAK_IPL_ImageTransformer_Destruct)(PEAK_IPL_IMAGE_TRANSFORMER_HANDLE imageTransformerHandle);
typedef PEAK_IPL_RETURN_CODE (*dyn_PEAK_IPL_Histogram_Construct)(PEAK_IPL_IMAGE_HANDLE imageHandle, PEAK_IPL_HISTOGRAM_HANDLE * histogramHandle);
typedef PEAK_IPL_RETURN_CODE (*dyn_PEAK_IPL_Histogram_GetPixelFormat)(PEAK_IPL_HISTOGRAM_HANDLE histogramHandle, PEAK_IPL_PIXEL_FORMAT * pixelFormat);
typedef PEAK_IPL_RETURN_CODE (*dyn_PEAK_IPL_Histogram_GetNumChannels)(PEAK_IPL_HISTOGRAM_HANDLE histogramHandle, size_t * numChannels);
typedef PEAK_IPL_RETURN_CODE (*dyn_PEAK_IPL_Histogram_GetPixelSumForChannel)(PEAK_IPL_HISTOGRAM_HANDLE histogramHandle, size_t channelIndex, uint64_t * pixelSum);
typedef PEAK_IPL_RETURN_CODE (*dyn_PEAK_IPL_Histogram_GetPixelCountForChannel)(PEAK_IPL_HISTOGRAM_HANDLE histogramHandle, size_t channelIndex, uint64_t * pixelCount);
typedef PEAK_IPL_RETURN_CODE (*dyn_PEAK_IPL_Histogram_GetBinsForChannel)(PEAK_IPL_HISTOGRAM_HANDLE histogramHandle, size_t channelIndex, uint64_t * bins, size_t * binsSize);
typedef PEAK_IPL_RETURN_CODE (*dyn_PEAK_IPL_Histogram_Destruct)(PEAK_IPL_HISTOGRAM_HANDLE histogramHandle);
typedef PEAK_IPL_RETURN_CODE (*dyn_PEAK_IPL_PixelLine_Construct)(PEAK_IPL_IMAGE_HANDLE imageHandle, PEAK_IPL_ORIENTATION orientation, size_t offset, PEAK_IPL_PIXEL_LINE_HANDLE * pixelLineHandle);
typedef PEAK_IPL_RETURN_CODE (*dyn_PEAK_IPL_PixelLine_GetPixelFormat)(PEAK_IPL_PIXEL_LINE_HANDLE pixelLineHandle, PEAK_IPL_PIXEL_FORMAT * pixelFormat);
typedef PEAK_IPL_RETURN_CODE (*dyn_PEAK_IPL_PixelLine_GetOrientation)(PEAK_IPL_PIXEL_LINE_HANDLE pixelLineHandle, PEAK_IPL_ORIENTATION * orientation);
typedef PEAK_IPL_RETURN_CODE (*dyn_PEAK_IPL_PixelLine_GetOffset)(PEAK_IPL_PIXEL_LINE_HANDLE pixelLineHandle, size_t * offset);
typedef PEAK_IPL_RETURN_CODE (*dyn_PEAK_IPL_PixelLine_GetNumChannels)(PEAK_IPL_PIXEL_LINE_HANDLE pixelLineHandle, size_t * numChannels);
typedef PEAK_IPL_RETURN_CODE (*dyn_PEAK_IPL_PixelLine_GetValuesForChannel)(PEAK_IPL_PIXEL_LINE_HANDLE pixelLineHandle, size_t channelIndex, uint32_t * values, size_t * valuesSize);
typedef PEAK_IPL_RETURN_CODE (*dyn_PEAK_IPL_PixelLine_Destruct)(PEAK_IPL_PIXEL_LINE_HANDLE pixelLineHandle);
typedef PEAK_IPL_RETURN_CODE (*dyn_PEAK_IPL_PixelFormat_GetNumChannels)(PEAK_IPL_PIXEL_FORMAT pixelFormat, size_t * numChannels);
typedef PEAK_IPL_RETURN_CODE (*dyn_PEAK_IPL_PixelFormat_GetNumSignificantBitsPerChannel)(PEAK_IPL_PIXEL_FORMAT pixelFormat, size_t * numSignificantBits);
typedef PEAK_IPL_RETURN_CODE (*dyn_PEAK_IPL_PixelFormat_GetNumStorageBitsPerChannel)(PEAK_IPL_PIXEL_FORMAT pixelFormat, size_t * numStorageBits);
typedef PEAK_IPL_RETURN_CODE (*dyn_PEAK_IPL_PixelFormat_GetMaximumValuePerChannel)(PEAK_IPL_PIXEL_FORMAT pixelFormat, uint32_t * channelMaximumValue);
typedef PEAK_IPL_RETURN_CODE (*dyn_PEAK_IPL_PixelFormat_GetNumSignificantBitsPerPixel)(PEAK_IPL_PIXEL_FORMAT pixelFormat, size_t * numSignificantBits);
typedef PEAK_IPL_RETURN_CODE (*dyn_PEAK_IPL_PixelFormat_GetNumStorageBitsPerPixel)(PEAK_IPL_PIXEL_FORMAT pixelFormat, size_t * numStorageBits);
typedef PEAK_IPL_RETURN_CODE (*dyn_PEAK_IPL_PixelFormat_GetEndianness)(PEAK_IPL_PIXEL_FORMAT pixelFormat, PEAK_IPL_ENDIANNESS * endianness);
typedef PEAK_IPL_RETURN_CODE (*dyn_PEAK_IPL_PixelFormat_CalculateStorageSizeOfPixels)(PEAK_IPL_PIXEL_FORMAT pixelFormat, uint64_t numPixels, uint64_t * size);
typedef PEAK_IPL_RETURN_CODE (*dyn_PEAK_IPL_ImageWriter_WriteAsRAW)(PEAK_IPL_IMAGE_HANDLE inputImageHandle, const char * filePath, size_t filePathSize);
typedef PEAK_IPL_RETURN_CODE (*dyn_PEAK_IPL_ImageWriter_WriteAsBMP)(PEAK_IPL_IMAGE_HANDLE inputImageHandle, const char * filePath, size_t filePathSize);
typedef PEAK_IPL_RETURN_CODE (*dyn_PEAK_IPL_ImageWriter_WriteAsPNG)(PEAK_IPL_IMAGE_HANDLE inputImageHandle, uint32_t quality, const char * filePath, size_t filePathSize);
typedef PEAK_IPL_RETURN_CODE (*dyn_PEAK_IPL_ImageWriter_WriteAsJPG)(PEAK_IPL_IMAGE_HANDLE inputImageHandle, uint32_t quality, const char * filePath, size_t filePathSize);
typedef PEAK_IPL_RETURN_CODE (*dyn_PEAK_IPL_ImageWriter_Write)(PEAK_IPL_IMAGE_HANDLE inputImageHandle, const char * filePath, size_t filePathSize);
typedef PEAK_IPL_RETURN_CODE (*dyn_PEAK_IPL_ImageReader_Read)(const char * filePath, size_t filePathSize, PEAK_IPL_IMAGE_HANDLE * outputImageHandle);
typedef PEAK_IPL_RETURN_CODE (*dyn_PEAK_IPL_ImageReaderRead_ReadAsPixelFormat)(const char * filePath, size_t filePathSize, PEAK_IPL_PIXEL_FORMAT pixelFormat, PEAK_IPL_IMAGE_HANDLE * outputImageHandle);

                        
class DynamicLoader
{
private:
    DynamicLoader();
    
    static DynamicLoader& instance()
    {
        static DynamicLoader dynamicLoader{};
        return dynamicLoader;
    }
    bool loadLib(const char* file);
    void unload();
    bool setPointers(bool load);

public:
    ~DynamicLoader();
    
    static bool isLoaded();
    
    static PEAK_IPL_RETURN_CODE PEAK_IPL_Library_GetVersionMajor(uint32_t * libraryVersionMajor);
    static PEAK_IPL_RETURN_CODE PEAK_IPL_Library_GetVersionMinor(uint32_t * libraryVersionMinor);
    static PEAK_IPL_RETURN_CODE PEAK_IPL_Library_GetVersionSubminor(uint32_t * libraryVersionSubminor);
    static PEAK_IPL_RETURN_CODE PEAK_IPL_Library_GetLastError(PEAK_IPL_RETURN_CODE * lastErrorCode, char * lastErrorDescription, size_t * lastErrorDescriptionSize);
    static PEAK_IPL_RETURN_CODE PEAK_IPL_ImageConverter_Construct(PEAK_IPL_IMAGE_CONVERTER_HANDLE * imageConverterHandle);
    static PEAK_IPL_RETURN_CODE PEAK_IPL_ImageConverter_GetConversionMode(PEAK_IPL_IMAGE_CONVERTER_HANDLE imageConverterHandle, PEAK_IPL_CONVERSION_MODE * conversionMode);
    static PEAK_IPL_RETURN_CODE PEAK_IPL_ImageConverter_SetConversionMode(PEAK_IPL_IMAGE_CONVERTER_HANDLE imageConverterHandle, PEAK_IPL_CONVERSION_MODE conversionMode);
    static PEAK_IPL_RETURN_CODE PEAK_IPL_ImageConverter_GetSupportedOutputPixelFormats(PEAK_IPL_IMAGE_CONVERTER_HANDLE imageConverterHandle, PEAK_IPL_PIXEL_FORMAT inputPixelFormat, PEAK_IPL_PIXEL_FORMAT * outputPixelFormats, size_t * outputPixelFormatsSize);
    static PEAK_IPL_RETURN_CODE PEAK_IPL_ImageConverter_Convert(PEAK_IPL_IMAGE_CONVERTER_HANDLE imageConverterHandle, PEAK_IPL_IMAGE_HANDLE inputImageHandle, PEAK_IPL_PIXEL_FORMAT outputPixelFormat, PEAK_IPL_IMAGE_HANDLE * outputImageHandle);
    static PEAK_IPL_RETURN_CODE PEAK_IPL_ImageConverter_ConvertToBuffer(PEAK_IPL_IMAGE_CONVERTER_HANDLE imageConverterHandle, PEAK_IPL_IMAGE_HANDLE inputImageHandle, PEAK_IPL_PIXEL_FORMAT outputPixelFormat, uint8_t * outputImageBuffer, size_t outputImageBufferSize, PEAK_IPL_IMAGE_HANDLE * outputImageHandle);
    static PEAK_IPL_RETURN_CODE PEAK_IPL_ImageConverter_Destruct(PEAK_IPL_IMAGE_CONVERTER_HANDLE imageConverterHandle);
    static PEAK_IPL_RETURN_CODE PEAK_IPL_AdaptiveHotpixelCorrector_Construct(PEAK_IPL_ADAPTIVE_HOTPIXEL_CORRECTOR_HANDLE * adaptiveHotpixelCorrectorHandle);
    static PEAK_IPL_RETURN_CODE PEAK_IPL_AdaptiveHotpixelCorrector_SetSensitivity(PEAK_IPL_ADAPTIVE_HOTPIXEL_CORRECTOR_HANDLE adaptiveHotpixelCorrectorHandle, PEAK_IPL_HOTPIXELCORRECTION_SENSITIVITY sensitivityLevel);
    static PEAK_IPL_RETURN_CODE PEAK_IPL_AdaptiveHotpixelCorrector_GetSensitivity(PEAK_IPL_ADAPTIVE_HOTPIXEL_CORRECTOR_HANDLE adaptiveHotpixelCorrectorHandle, PEAK_IPL_HOTPIXELCORRECTION_SENSITIVITY * sensitivityLevel);
    static PEAK_IPL_RETURN_CODE PEAK_IPL_AdaptiveHotpixelCorrector_SetGainFactorPercent(PEAK_IPL_ADAPTIVE_HOTPIXEL_CORRECTOR_HANDLE adaptiveHotpixelCorrectorHandle, uint32_t gainFactorPercent);
    static PEAK_IPL_RETURN_CODE PEAK_IPL_AdaptiveHotpixelCorrector_GetGainFactorPercent(PEAK_IPL_ADAPTIVE_HOTPIXEL_CORRECTOR_HANDLE adaptiveHotpixelCorrectorHandle, uint32_t * gainFactorPercent);
    static PEAK_IPL_RETURN_CODE PEAK_IPL_AdaptiveHotpixelCorrector_Detect(PEAK_IPL_ADAPTIVE_HOTPIXEL_CORRECTOR_HANDLE adaptiveHotpixelCorrectorHandle, PEAK_IPL_IMAGE_HANDLE inputImageHandle);
    static PEAK_IPL_RETURN_CODE PEAK_IPL_AdaptiveHotpixelCorrector_GetHotpixels(PEAK_IPL_ADAPTIVE_HOTPIXEL_CORRECTOR_HANDLE adaptiveHotpixelCorrectorHandle, PEAK_IPL_POINT_2D * hotpixels, size_t * hotpixelsSize);
    static PEAK_IPL_RETURN_CODE PEAK_IPL_AdaptiveHotpixelCorrector_Correct(PEAK_IPL_ADAPTIVE_HOTPIXEL_CORRECTOR_HANDLE adaptiveHotpixelCorrectorHandle, PEAK_IPL_IMAGE_HANDLE inputImageHandle, const PEAK_IPL_POINT_2D * hotpixels, size_t hotpixelsSize, PEAK_IPL_IMAGE_HANDLE * outputImageHandle);
    static PEAK_IPL_RETURN_CODE PEAK_IPL_AdaptiveHotpixelCorrector_CorrectAdaptive(PEAK_IPL_ADAPTIVE_HOTPIXEL_CORRECTOR_HANDLE adaptiveHotpixelCorrectorHandle, PEAK_IPL_IMAGE_HANDLE inputImageHandle, PEAK_IPL_IMAGE_HANDLE * outputImageHandle);
    static PEAK_IPL_RETURN_CODE PEAK_IPL_AdaptiveHotpixelCorrector_Destruct(PEAK_IPL_ADAPTIVE_HOTPIXEL_CORRECTOR_HANDLE adaptiveHotpixelCorrectorHandle);
    static PEAK_IPL_RETURN_CODE PEAK_IPL_ColorCorrector_Construct(PEAK_IPL_COLOR_CORRECTOR_HANDLE * colorCorrectorHandle);
    static PEAK_IPL_RETURN_CODE PEAK_IPL_ColorCorrector_SetColorCorrectionFactors(PEAK_IPL_COLOR_CORRECTOR_HANDLE colorCorrectorHandle, float * colorCorrectorFactors);
    static PEAK_IPL_RETURN_CODE PEAK_IPL_ColorCorrector_GetColorCorrectionFactors(PEAK_IPL_COLOR_CORRECTOR_HANDLE colorCorrectorHandle, float * colorCorrectorFactors, size_t * colorCorrectorFactorsSize);
    static PEAK_IPL_RETURN_CODE PEAK_IPL_ColorCorrector_GetIsPixelFormatSupported(PEAK_IPL_COLOR_CORRECTOR_HANDLE colorCorrectorHandle, PEAK_IPL_PIXEL_FORMAT pixelFormat, PEAK_IPL_BOOL8 * isPixelFormatSupported);
    static PEAK_IPL_RETURN_CODE PEAK_IPL_ColorCorrector_Process(PEAK_IPL_COLOR_CORRECTOR_HANDLE colorCorrectorHandle, PEAK_IPL_IMAGE_HANDLE imageHandle, PEAK_IPL_IMAGE_HANDLE * outputImageHandle);
    static PEAK_IPL_RETURN_CODE PEAK_IPL_ColorCorrector_ProcessInPlace(PEAK_IPL_COLOR_CORRECTOR_HANDLE colorCorrectorHandle, PEAK_IPL_IMAGE_HANDLE imageHandle);
    static PEAK_IPL_RETURN_CODE PEAK_IPL_ColorCorrector_Destruct(PEAK_IPL_COLOR_CORRECTOR_HANDLE colorCorrectorHandle);
    static PEAK_IPL_RETURN_CODE PEAK_IPL_GammaCorrector_Construct(PEAK_IPL_GAMMA_CORRECTOR_HANDLE * gammaCorrectorHandle);
    static PEAK_IPL_RETURN_CODE PEAK_IPL_GammaCorrector_SetGammaCorrectionValue(PEAK_IPL_GAMMA_CORRECTOR_HANDLE gammaCorrectorHandle, float * gammaValue);
    static PEAK_IPL_RETURN_CODE PEAK_IPL_GammaCorrector_GetGammaCorrectionValue(PEAK_IPL_GAMMA_CORRECTOR_HANDLE gammaCorrectorHandle, float * gammaValue);
    static PEAK_IPL_RETURN_CODE PEAK_IPL_GammaCorrector_GetGammaCorrectionMax(PEAK_IPL_GAMMA_CORRECTOR_HANDLE gammaCorrectorHandle, float * gammaMax);
    static PEAK_IPL_RETURN_CODE PEAK_IPL_GammaCorrector_GetGammaCorrectionMin(PEAK_IPL_GAMMA_CORRECTOR_HANDLE gammaCorrectorHandle, float * gammaMin);
    static PEAK_IPL_RETURN_CODE PEAK_IPL_GammaCorrector_GetIsPixelFormatSupported(PEAK_IPL_GAMMA_CORRECTOR_HANDLE gammaCorrectorHandle, PEAK_IPL_PIXEL_FORMAT pixelFormat, PEAK_IPL_BOOL8 * isPixelFormatSupported);
    static PEAK_IPL_RETURN_CODE PEAK_IPL_GammaCorrector_Process(PEAK_IPL_GAMMA_CORRECTOR_HANDLE gammaCorrectorHandle, PEAK_IPL_IMAGE_HANDLE imageHandle, PEAK_IPL_IMAGE_HANDLE * outputImageHandle);
    static PEAK_IPL_RETURN_CODE PEAK_IPL_GammaCorrector_ProcessInPlace(PEAK_IPL_GAMMA_CORRECTOR_HANDLE gammaCorrectorHandle, PEAK_IPL_IMAGE_HANDLE imageHandle);
    static PEAK_IPL_RETURN_CODE PEAK_IPL_GammaCorrector_Destruct(PEAK_IPL_GAMMA_CORRECTOR_HANDLE gammaCorrectorHandle);
    static PEAK_IPL_RETURN_CODE PEAK_IPL_Image_Construct(PEAK_IPL_PIXEL_FORMAT pixelFormat, size_t width, size_t height, PEAK_IPL_IMAGE_HANDLE * imageHandle);
    static PEAK_IPL_RETURN_CODE PEAK_IPL_Image_ConstructFromBuffer(PEAK_IPL_PIXEL_FORMAT pixelFormat, uint8_t * buffer, uint64_t bufferSize, size_t width, size_t height, PEAK_IPL_IMAGE_HANDLE * imageHandle);
    static PEAK_IPL_RETURN_CODE PEAK_IPL_Image_GetWidth(PEAK_IPL_IMAGE_HANDLE imageHandle, size_t * width);
    static PEAK_IPL_RETURN_CODE PEAK_IPL_Image_GetHeight(PEAK_IPL_IMAGE_HANDLE imageHandle, size_t * height);
    static PEAK_IPL_RETURN_CODE PEAK_IPL_Image_GetPixelPointer(PEAK_IPL_IMAGE_HANDLE imageHandle, size_t xPos, size_t yPos, uint8_t * * pixelPointer);
    static PEAK_IPL_RETURN_CODE PEAK_IPL_Image_GetByteCount(PEAK_IPL_IMAGE_HANDLE imageHandle, size_t * byteCount);
    static PEAK_IPL_RETURN_CODE PEAK_IPL_Image_GetPixelFormat(PEAK_IPL_IMAGE_HANDLE imageHandle, PEAK_IPL_PIXEL_FORMAT * pixelFormat);
    static PEAK_IPL_RETURN_CODE PEAK_IPL_Image_GetData(PEAK_IPL_IMAGE_HANDLE imageHandle, uint8_t * * data);
    static PEAK_IPL_RETURN_CODE PEAK_IPL_Image_Clone(PEAK_IPL_IMAGE_HANDLE imageHandle, PEAK_IPL_IMAGE_HANDLE * outputImageHandle);
    static PEAK_IPL_RETURN_CODE PEAK_IPL_Image_ConvertTo(PEAK_IPL_IMAGE_HANDLE imageHandle, PEAK_IPL_PIXEL_FORMAT outputPixelFormat, PEAK_IPL_CONVERSION_MODE conversionMode, PEAK_IPL_IMAGE_HANDLE * outputImageHandle);
    static PEAK_IPL_RETURN_CODE PEAK_IPL_Image_ConvertToBuffer(PEAK_IPL_IMAGE_HANDLE imageHandle, PEAK_IPL_PIXEL_FORMAT outputPixelFormat, uint8_t * outputImageBuffer, size_t outputImageBufferSize, PEAK_IPL_CONVERSION_MODE conversionMode, PEAK_IPL_IMAGE_HANDLE * outputImageHandle);
    static PEAK_IPL_RETURN_CODE PEAK_IPL_Image_Destruct(PEAK_IPL_IMAGE_HANDLE imageHandle);
    static PEAK_IPL_RETURN_CODE PEAK_IPL_ImageTransformer_Construct(PEAK_IPL_IMAGE_TRANSFORMER_HANDLE * imageTransformerHandle);
    static PEAK_IPL_RETURN_CODE PEAK_IPL_ImageTransformer_MirrorUpDown(PEAK_IPL_IMAGE_TRANSFORMER_HANDLE imageTransformerHandle, PEAK_IPL_IMAGE_HANDLE imageHandle, PEAK_IPL_IMAGE_HANDLE * outputImageHandle);
    static PEAK_IPL_RETURN_CODE PEAK_IPL_ImageTransformer_MirrorLeftRight(PEAK_IPL_IMAGE_TRANSFORMER_HANDLE imageTransformerHandle, PEAK_IPL_IMAGE_HANDLE imageHandle, PEAK_IPL_IMAGE_HANDLE * outputImageHandle);
    static PEAK_IPL_RETURN_CODE PEAK_IPL_ImageTransformer_MirrorUpDownLeftRight(PEAK_IPL_IMAGE_TRANSFORMER_HANDLE imageTransformerHandle, PEAK_IPL_IMAGE_HANDLE imageHandle, PEAK_IPL_IMAGE_HANDLE * outputImageHandle);
    static PEAK_IPL_RETURN_CODE PEAK_IPL_ImageTransformer_MirrorUpDownInPlace(PEAK_IPL_IMAGE_TRANSFORMER_HANDLE imageTransformerHandle, PEAK_IPL_IMAGE_HANDLE imageHandle);
    static PEAK_IPL_RETURN_CODE PEAK_IPL_ImageTransformer_MirrorLeftRightInPlace(PEAK_IPL_IMAGE_TRANSFORMER_HANDLE imageTransformerHandle, PEAK_IPL_IMAGE_HANDLE imageHandle);
    static PEAK_IPL_RETURN_CODE PEAK_IPL_ImageTransformer_MirrorUpDownLeftRightInPlace(PEAK_IPL_IMAGE_TRANSFORMER_HANDLE imageTransformerHandle, PEAK_IPL_IMAGE_HANDLE imageHandle);
    static PEAK_IPL_RETURN_CODE PEAK_IPL_ImageTransformer_Rotate(PEAK_IPL_IMAGE_TRANSFORMER_HANDLE imageTransformerHandle, PEAK_IPL_IMAGE_HANDLE imageHandle, PEAK_IPL_IMAGE_HANDLE * outputImageHandle, PEAK_IPL_IMAGE_TRANSFORMER_ROTATION_ANGLE_t rotationAngle);
    static PEAK_IPL_RETURN_CODE PEAK_IPL_ImageTransformer_RotateInPlace(PEAK_IPL_IMAGE_TRANSFORMER_HANDLE imageTransformerHandle, PEAK_IPL_IMAGE_HANDLE imageHandle, PEAK_IPL_IMAGE_TRANSFORMER_ROTATION_ANGLE_t rotationAngle);
    static PEAK_IPL_RETURN_CODE PEAK_IPL_ImageTransformer_Destruct(PEAK_IPL_IMAGE_TRANSFORMER_HANDLE imageTransformerHandle);
    static PEAK_IPL_RETURN_CODE PEAK_IPL_Histogram_Construct(PEAK_IPL_IMAGE_HANDLE imageHandle, PEAK_IPL_HISTOGRAM_HANDLE * histogramHandle);
    static PEAK_IPL_RETURN_CODE PEAK_IPL_Histogram_GetPixelFormat(PEAK_IPL_HISTOGRAM_HANDLE histogramHandle, PEAK_IPL_PIXEL_FORMAT * pixelFormat);
    static PEAK_IPL_RETURN_CODE PEAK_IPL_Histogram_GetNumChannels(PEAK_IPL_HISTOGRAM_HANDLE histogramHandle, size_t * numChannels);
    static PEAK_IPL_RETURN_CODE PEAK_IPL_Histogram_GetPixelSumForChannel(PEAK_IPL_HISTOGRAM_HANDLE histogramHandle, size_t channelIndex, uint64_t * pixelSum);
    static PEAK_IPL_RETURN_CODE PEAK_IPL_Histogram_GetPixelCountForChannel(PEAK_IPL_HISTOGRAM_HANDLE histogramHandle, size_t channelIndex, uint64_t * pixelCount);
    static PEAK_IPL_RETURN_CODE PEAK_IPL_Histogram_GetBinsForChannel(PEAK_IPL_HISTOGRAM_HANDLE histogramHandle, size_t channelIndex, uint64_t * bins, size_t * binsSize);
    static PEAK_IPL_RETURN_CODE PEAK_IPL_Histogram_Destruct(PEAK_IPL_HISTOGRAM_HANDLE histogramHandle);
    static PEAK_IPL_RETURN_CODE PEAK_IPL_PixelLine_Construct(PEAK_IPL_IMAGE_HANDLE imageHandle, PEAK_IPL_ORIENTATION orientation, size_t offset, PEAK_IPL_PIXEL_LINE_HANDLE * pixelLineHandle);
    static PEAK_IPL_RETURN_CODE PEAK_IPL_PixelLine_GetPixelFormat(PEAK_IPL_PIXEL_LINE_HANDLE pixelLineHandle, PEAK_IPL_PIXEL_FORMAT * pixelFormat);
    static PEAK_IPL_RETURN_CODE PEAK_IPL_PixelLine_GetOrientation(PEAK_IPL_PIXEL_LINE_HANDLE pixelLineHandle, PEAK_IPL_ORIENTATION * orientation);
    static PEAK_IPL_RETURN_CODE PEAK_IPL_PixelLine_GetOffset(PEAK_IPL_PIXEL_LINE_HANDLE pixelLineHandle, size_t * offset);
    static PEAK_IPL_RETURN_CODE PEAK_IPL_PixelLine_GetNumChannels(PEAK_IPL_PIXEL_LINE_HANDLE pixelLineHandle, size_t * numChannels);
    static PEAK_IPL_RETURN_CODE PEAK_IPL_PixelLine_GetValuesForChannel(PEAK_IPL_PIXEL_LINE_HANDLE pixelLineHandle, size_t channelIndex, uint32_t * values, size_t * valuesSize);
    static PEAK_IPL_RETURN_CODE PEAK_IPL_PixelLine_Destruct(PEAK_IPL_PIXEL_LINE_HANDLE pixelLineHandle);
    static PEAK_IPL_RETURN_CODE PEAK_IPL_PixelFormat_GetNumChannels(PEAK_IPL_PIXEL_FORMAT pixelFormat, size_t * numChannels);
    static PEAK_IPL_RETURN_CODE PEAK_IPL_PixelFormat_GetNumSignificantBitsPerChannel(PEAK_IPL_PIXEL_FORMAT pixelFormat, size_t * numSignificantBits);
    static PEAK_IPL_RETURN_CODE PEAK_IPL_PixelFormat_GetNumStorageBitsPerChannel(PEAK_IPL_PIXEL_FORMAT pixelFormat, size_t * numStorageBits);
    static PEAK_IPL_RETURN_CODE PEAK_IPL_PixelFormat_GetMaximumValuePerChannel(PEAK_IPL_PIXEL_FORMAT pixelFormat, uint32_t * channelMaximumValue);
    static PEAK_IPL_RETURN_CODE PEAK_IPL_PixelFormat_GetNumSignificantBitsPerPixel(PEAK_IPL_PIXEL_FORMAT pixelFormat, size_t * numSignificantBits);
    static PEAK_IPL_RETURN_CODE PEAK_IPL_PixelFormat_GetNumStorageBitsPerPixel(PEAK_IPL_PIXEL_FORMAT pixelFormat, size_t * numStorageBits);
    static PEAK_IPL_RETURN_CODE PEAK_IPL_PixelFormat_GetEndianness(PEAK_IPL_PIXEL_FORMAT pixelFormat, PEAK_IPL_ENDIANNESS * endianness);
    static PEAK_IPL_RETURN_CODE PEAK_IPL_PixelFormat_CalculateStorageSizeOfPixels(PEAK_IPL_PIXEL_FORMAT pixelFormat, uint64_t numPixels, uint64_t * size);
    static PEAK_IPL_RETURN_CODE PEAK_IPL_ImageWriter_WriteAsRAW(PEAK_IPL_IMAGE_HANDLE inputImageHandle, const char * filePath, size_t filePathSize);
    static PEAK_IPL_RETURN_CODE PEAK_IPL_ImageWriter_WriteAsBMP(PEAK_IPL_IMAGE_HANDLE inputImageHandle, const char * filePath, size_t filePathSize);
    static PEAK_IPL_RETURN_CODE PEAK_IPL_ImageWriter_WriteAsPNG(PEAK_IPL_IMAGE_HANDLE inputImageHandle, uint32_t quality, const char * filePath, size_t filePathSize);
    static PEAK_IPL_RETURN_CODE PEAK_IPL_ImageWriter_WriteAsJPG(PEAK_IPL_IMAGE_HANDLE inputImageHandle, uint32_t quality, const char * filePath, size_t filePathSize);
    static PEAK_IPL_RETURN_CODE PEAK_IPL_ImageWriter_Write(PEAK_IPL_IMAGE_HANDLE inputImageHandle, const char * filePath, size_t filePathSize);
    static PEAK_IPL_RETURN_CODE PEAK_IPL_ImageReader_Read(const char * filePath, size_t filePathSize, PEAK_IPL_IMAGE_HANDLE * outputImageHandle);
    static PEAK_IPL_RETURN_CODE PEAK_IPL_ImageReaderRead_ReadAsPixelFormat(const char * filePath, size_t filePathSize, PEAK_IPL_PIXEL_FORMAT pixelFormat, PEAK_IPL_IMAGE_HANDLE * outputImageHandle);
       
private:
    void* m_handle = nullptr;
    dyn_PEAK_IPL_Library_GetVersionMajor m_PEAK_IPL_Library_GetVersionMajor{};
    dyn_PEAK_IPL_Library_GetVersionMinor m_PEAK_IPL_Library_GetVersionMinor{};
    dyn_PEAK_IPL_Library_GetVersionSubminor m_PEAK_IPL_Library_GetVersionSubminor{};
    dyn_PEAK_IPL_Library_GetLastError m_PEAK_IPL_Library_GetLastError{};
    dyn_PEAK_IPL_ImageConverter_Construct m_PEAK_IPL_ImageConverter_Construct{};
    dyn_PEAK_IPL_ImageConverter_GetConversionMode m_PEAK_IPL_ImageConverter_GetConversionMode{};
    dyn_PEAK_IPL_ImageConverter_SetConversionMode m_PEAK_IPL_ImageConverter_SetConversionMode{};
    dyn_PEAK_IPL_ImageConverter_GetSupportedOutputPixelFormats m_PEAK_IPL_ImageConverter_GetSupportedOutputPixelFormats{};
    dyn_PEAK_IPL_ImageConverter_Convert m_PEAK_IPL_ImageConverter_Convert{};
    dyn_PEAK_IPL_ImageConverter_ConvertToBuffer m_PEAK_IPL_ImageConverter_ConvertToBuffer{};
    dyn_PEAK_IPL_ImageConverter_Destruct m_PEAK_IPL_ImageConverter_Destruct{};
    dyn_PEAK_IPL_AdaptiveHotpixelCorrector_Construct m_PEAK_IPL_AdaptiveHotpixelCorrector_Construct{};
    dyn_PEAK_IPL_AdaptiveHotpixelCorrector_SetSensitivity m_PEAK_IPL_AdaptiveHotpixelCorrector_SetSensitivity{};
    dyn_PEAK_IPL_AdaptiveHotpixelCorrector_GetSensitivity m_PEAK_IPL_AdaptiveHotpixelCorrector_GetSensitivity{};
    dyn_PEAK_IPL_AdaptiveHotpixelCorrector_SetGainFactorPercent m_PEAK_IPL_AdaptiveHotpixelCorrector_SetGainFactorPercent{};
    dyn_PEAK_IPL_AdaptiveHotpixelCorrector_GetGainFactorPercent m_PEAK_IPL_AdaptiveHotpixelCorrector_GetGainFactorPercent{};
    dyn_PEAK_IPL_AdaptiveHotpixelCorrector_Detect m_PEAK_IPL_AdaptiveHotpixelCorrector_Detect{};
    dyn_PEAK_IPL_AdaptiveHotpixelCorrector_GetHotpixels m_PEAK_IPL_AdaptiveHotpixelCorrector_GetHotpixels{};
    dyn_PEAK_IPL_AdaptiveHotpixelCorrector_Correct m_PEAK_IPL_AdaptiveHotpixelCorrector_Correct{};
    dyn_PEAK_IPL_AdaptiveHotpixelCorrector_CorrectAdaptive m_PEAK_IPL_AdaptiveHotpixelCorrector_CorrectAdaptive{};
    dyn_PEAK_IPL_AdaptiveHotpixelCorrector_Destruct m_PEAK_IPL_AdaptiveHotpixelCorrector_Destruct{};
    dyn_PEAK_IPL_ColorCorrector_Construct m_PEAK_IPL_ColorCorrector_Construct{};
    dyn_PEAK_IPL_ColorCorrector_SetColorCorrectionFactors m_PEAK_IPL_ColorCorrector_SetColorCorrectionFactors{};
    dyn_PEAK_IPL_ColorCorrector_GetColorCorrectionFactors m_PEAK_IPL_ColorCorrector_GetColorCorrectionFactors{};
    dyn_PEAK_IPL_ColorCorrector_GetIsPixelFormatSupported m_PEAK_IPL_ColorCorrector_GetIsPixelFormatSupported{};
    dyn_PEAK_IPL_ColorCorrector_Process m_PEAK_IPL_ColorCorrector_Process{};
    dyn_PEAK_IPL_ColorCorrector_ProcessInPlace m_PEAK_IPL_ColorCorrector_ProcessInPlace{};
    dyn_PEAK_IPL_ColorCorrector_Destruct m_PEAK_IPL_ColorCorrector_Destruct{};
    dyn_PEAK_IPL_GammaCorrector_Construct m_PEAK_IPL_GammaCorrector_Construct{};
    dyn_PEAK_IPL_GammaCorrector_SetGammaCorrectionValue m_PEAK_IPL_GammaCorrector_SetGammaCorrectionValue{};
    dyn_PEAK_IPL_GammaCorrector_GetGammaCorrectionValue m_PEAK_IPL_GammaCorrector_GetGammaCorrectionValue{};
    dyn_PEAK_IPL_GammaCorrector_GetGammaCorrectionMax m_PEAK_IPL_GammaCorrector_GetGammaCorrectionMax{};
    dyn_PEAK_IPL_GammaCorrector_GetGammaCorrectionMin m_PEAK_IPL_GammaCorrector_GetGammaCorrectionMin{};
    dyn_PEAK_IPL_GammaCorrector_GetIsPixelFormatSupported m_PEAK_IPL_GammaCorrector_GetIsPixelFormatSupported{};
    dyn_PEAK_IPL_GammaCorrector_Process m_PEAK_IPL_GammaCorrector_Process{};
    dyn_PEAK_IPL_GammaCorrector_ProcessInPlace m_PEAK_IPL_GammaCorrector_ProcessInPlace{};
    dyn_PEAK_IPL_GammaCorrector_Destruct m_PEAK_IPL_GammaCorrector_Destruct{};
    dyn_PEAK_IPL_Image_Construct m_PEAK_IPL_Image_Construct{};
    dyn_PEAK_IPL_Image_ConstructFromBuffer m_PEAK_IPL_Image_ConstructFromBuffer{};
    dyn_PEAK_IPL_Image_GetWidth m_PEAK_IPL_Image_GetWidth{};
    dyn_PEAK_IPL_Image_GetHeight m_PEAK_IPL_Image_GetHeight{};
    dyn_PEAK_IPL_Image_GetPixelPointer m_PEAK_IPL_Image_GetPixelPointer{};
    dyn_PEAK_IPL_Image_GetByteCount m_PEAK_IPL_Image_GetByteCount{};
    dyn_PEAK_IPL_Image_GetPixelFormat m_PEAK_IPL_Image_GetPixelFormat{};
    dyn_PEAK_IPL_Image_GetData m_PEAK_IPL_Image_GetData{};
    dyn_PEAK_IPL_Image_Clone m_PEAK_IPL_Image_Clone{};
    dyn_PEAK_IPL_Image_ConvertTo m_PEAK_IPL_Image_ConvertTo{};
    dyn_PEAK_IPL_Image_ConvertToBuffer m_PEAK_IPL_Image_ConvertToBuffer{};
    dyn_PEAK_IPL_Image_Destruct m_PEAK_IPL_Image_Destruct{};
    dyn_PEAK_IPL_ImageTransformer_Construct m_PEAK_IPL_ImageTransformer_Construct{};
    dyn_PEAK_IPL_ImageTransformer_MirrorUpDown m_PEAK_IPL_ImageTransformer_MirrorUpDown{};
    dyn_PEAK_IPL_ImageTransformer_MirrorLeftRight m_PEAK_IPL_ImageTransformer_MirrorLeftRight{};
    dyn_PEAK_IPL_ImageTransformer_MirrorUpDownLeftRight m_PEAK_IPL_ImageTransformer_MirrorUpDownLeftRight{};
    dyn_PEAK_IPL_ImageTransformer_MirrorUpDownInPlace m_PEAK_IPL_ImageTransformer_MirrorUpDownInPlace{};
    dyn_PEAK_IPL_ImageTransformer_MirrorLeftRightInPlace m_PEAK_IPL_ImageTransformer_MirrorLeftRightInPlace{};
    dyn_PEAK_IPL_ImageTransformer_MirrorUpDownLeftRightInPlace m_PEAK_IPL_ImageTransformer_MirrorUpDownLeftRightInPlace{};
    dyn_PEAK_IPL_ImageTransformer_Rotate m_PEAK_IPL_ImageTransformer_Rotate{};
    dyn_PEAK_IPL_ImageTransformer_RotateInPlace m_PEAK_IPL_ImageTransformer_RotateInPlace{};
    dyn_PEAK_IPL_ImageTransformer_Destruct m_PEAK_IPL_ImageTransformer_Destruct{};
    dyn_PEAK_IPL_Histogram_Construct m_PEAK_IPL_Histogram_Construct{};
    dyn_PEAK_IPL_Histogram_GetPixelFormat m_PEAK_IPL_Histogram_GetPixelFormat{};
    dyn_PEAK_IPL_Histogram_GetNumChannels m_PEAK_IPL_Histogram_GetNumChannels{};
    dyn_PEAK_IPL_Histogram_GetPixelSumForChannel m_PEAK_IPL_Histogram_GetPixelSumForChannel{};
    dyn_PEAK_IPL_Histogram_GetPixelCountForChannel m_PEAK_IPL_Histogram_GetPixelCountForChannel{};
    dyn_PEAK_IPL_Histogram_GetBinsForChannel m_PEAK_IPL_Histogram_GetBinsForChannel{};
    dyn_PEAK_IPL_Histogram_Destruct m_PEAK_IPL_Histogram_Destruct{};
    dyn_PEAK_IPL_PixelLine_Construct m_PEAK_IPL_PixelLine_Construct{};
    dyn_PEAK_IPL_PixelLine_GetPixelFormat m_PEAK_IPL_PixelLine_GetPixelFormat{};
    dyn_PEAK_IPL_PixelLine_GetOrientation m_PEAK_IPL_PixelLine_GetOrientation{};
    dyn_PEAK_IPL_PixelLine_GetOffset m_PEAK_IPL_PixelLine_GetOffset{};
    dyn_PEAK_IPL_PixelLine_GetNumChannels m_PEAK_IPL_PixelLine_GetNumChannels{};
    dyn_PEAK_IPL_PixelLine_GetValuesForChannel m_PEAK_IPL_PixelLine_GetValuesForChannel{};
    dyn_PEAK_IPL_PixelLine_Destruct m_PEAK_IPL_PixelLine_Destruct{};
    dyn_PEAK_IPL_PixelFormat_GetNumChannels m_PEAK_IPL_PixelFormat_GetNumChannels{};
    dyn_PEAK_IPL_PixelFormat_GetNumSignificantBitsPerChannel m_PEAK_IPL_PixelFormat_GetNumSignificantBitsPerChannel{};
    dyn_PEAK_IPL_PixelFormat_GetNumStorageBitsPerChannel m_PEAK_IPL_PixelFormat_GetNumStorageBitsPerChannel{};
    dyn_PEAK_IPL_PixelFormat_GetMaximumValuePerChannel m_PEAK_IPL_PixelFormat_GetMaximumValuePerChannel{};
    dyn_PEAK_IPL_PixelFormat_GetNumSignificantBitsPerPixel m_PEAK_IPL_PixelFormat_GetNumSignificantBitsPerPixel{};
    dyn_PEAK_IPL_PixelFormat_GetNumStorageBitsPerPixel m_PEAK_IPL_PixelFormat_GetNumStorageBitsPerPixel{};
    dyn_PEAK_IPL_PixelFormat_GetEndianness m_PEAK_IPL_PixelFormat_GetEndianness{};
    dyn_PEAK_IPL_PixelFormat_CalculateStorageSizeOfPixels m_PEAK_IPL_PixelFormat_CalculateStorageSizeOfPixels{};
    dyn_PEAK_IPL_ImageWriter_WriteAsRAW m_PEAK_IPL_ImageWriter_WriteAsRAW{};
    dyn_PEAK_IPL_ImageWriter_WriteAsBMP m_PEAK_IPL_ImageWriter_WriteAsBMP{};
    dyn_PEAK_IPL_ImageWriter_WriteAsPNG m_PEAK_IPL_ImageWriter_WriteAsPNG{};
    dyn_PEAK_IPL_ImageWriter_WriteAsJPG m_PEAK_IPL_ImageWriter_WriteAsJPG{};
    dyn_PEAK_IPL_ImageWriter_Write m_PEAK_IPL_ImageWriter_Write{};
    dyn_PEAK_IPL_ImageReader_Read m_PEAK_IPL_ImageReader_Read{};
    dyn_PEAK_IPL_ImageReaderRead_ReadAsPixelFormat m_PEAK_IPL_ImageReaderRead_ReadAsPixelFormat{};

};

inline void* import_function(void *module, const char* proc_name)
{
#ifdef __linux__
    return dlsym(module, proc_name);
#else
    return GetProcAddress(static_cast<HMODULE>(module), proc_name);
#endif
}
            
inline DynamicLoader::DynamicLoader()
{
#if defined _WIN32 || defined _WIN64
    size_t sz = 0;
    if (_wgetenv_s(&sz, NULL, 0, L"IDS_PEAK_SDK_PATH") == 0)
    {
        std::vector<wchar_t> env_ids_peak_ipl(sz);
        if (_wgetenv_s(&sz, env_ids_peak_ipl.data(), sz, L"IDS_PEAK_SDK_PATH") == 0)
        {
            if (_wgetenv_s(&sz, NULL, 0, L"PATH") == 0)
            {
                std::vector<wchar_t> env_path(sz);
                if (_wgetenv_s(&sz, env_path.data(), sz, L"PATH") == 0)
                {
                    std::wstring ids_peak_ipl_path(env_ids_peak_ipl.data());
#ifdef _WIN64
                    ids_peak_ipl_path.append(L"\\ipl\\lib\\x86_64");
#else
                    ids_peak_ipl_path.append(L"\\ipl\\lib\\x86_32");
#endif
                    std::wstring path_var(env_path.data());
                    path_var.append(L";").append(ids_peak_ipl_path);
                    _wputenv_s(L"PATH", path_var.c_str());
                }
            }
        }
    }
    
    loadLib("ids_peak_ipl.dll");
#else
    loadLib("libids_peak_ipl.so");
#endif
}

inline DynamicLoader::~DynamicLoader()
{
    if(m_handle != nullptr)
    {
        unload();
    }
}

inline bool DynamicLoader::isLoaded()
{
    auto&& inst = instance();
    return inst.m_handle != nullptr;
}

inline void DynamicLoader::unload()
{
    setPointers(false);
    
    if (m_handle != nullptr)
    {
#ifdef __linux__
        dlclose(m_handle);
#else
        FreeLibrary(static_cast<HMODULE>(m_handle));
#endif
    }
    m_handle = nullptr;
}


inline bool DynamicLoader::loadLib(const char* file)
{
    bool ret = false;
    
    if (file)
    {
#ifdef __linux__
        m_handle = dlopen(file, RTLD_NOW);
#else
        m_handle = LoadLibraryA(file);
#endif
        if (m_handle != nullptr)
        {
            try {
                setPointers(true);
                ret = true;
            } catch (const std::exception&) {
                unload();
                throw;
            }
        }
        else
        {
            throw std::runtime_error(std::string("Lib load failed: ") + file);
        }
    }
    else
    {
        throw std::runtime_error("Filename empty");
    }

    return ret;
}

inline bool DynamicLoader::setPointers(bool load)
{

    m_PEAK_IPL_Library_GetVersionMajor = (dyn_PEAK_IPL_Library_GetVersionMajor) (load ?  import_function(m_handle, "PEAK_IPL_Library_GetVersionMajor") : nullptr);
    if(m_PEAK_IPL_Library_GetVersionMajor == nullptr && load)
    {
        throw std::runtime_error("Failed to load PEAK_IPL_Library_GetVersionMajor");
    }        

    m_PEAK_IPL_Library_GetVersionMinor = (dyn_PEAK_IPL_Library_GetVersionMinor) (load ?  import_function(m_handle, "PEAK_IPL_Library_GetVersionMinor") : nullptr);
    if(m_PEAK_IPL_Library_GetVersionMinor == nullptr && load)
    {
        throw std::runtime_error("Failed to load PEAK_IPL_Library_GetVersionMinor");
    }        

    m_PEAK_IPL_Library_GetVersionSubminor = (dyn_PEAK_IPL_Library_GetVersionSubminor) (load ?  import_function(m_handle, "PEAK_IPL_Library_GetVersionSubminor") : nullptr);
    if(m_PEAK_IPL_Library_GetVersionSubminor == nullptr && load)
    {
        throw std::runtime_error("Failed to load PEAK_IPL_Library_GetVersionSubminor");
    }        

    m_PEAK_IPL_Library_GetLastError = (dyn_PEAK_IPL_Library_GetLastError) (load ?  import_function(m_handle, "PEAK_IPL_Library_GetLastError") : nullptr);
    if(m_PEAK_IPL_Library_GetLastError == nullptr && load)
    {
        throw std::runtime_error("Failed to load PEAK_IPL_Library_GetLastError");
    }        

    m_PEAK_IPL_ImageConverter_Construct = (dyn_PEAK_IPL_ImageConverter_Construct) (load ?  import_function(m_handle, "PEAK_IPL_ImageConverter_Construct") : nullptr);
    if(m_PEAK_IPL_ImageConverter_Construct == nullptr && load)
    {
        throw std::runtime_error("Failed to load PEAK_IPL_ImageConverter_Construct");
    }        

    m_PEAK_IPL_ImageConverter_GetConversionMode = (dyn_PEAK_IPL_ImageConverter_GetConversionMode) (load ?  import_function(m_handle, "PEAK_IPL_ImageConverter_GetConversionMode") : nullptr);
    if(m_PEAK_IPL_ImageConverter_GetConversionMode == nullptr && load)
    {
        throw std::runtime_error("Failed to load PEAK_IPL_ImageConverter_GetConversionMode");
    }        

    m_PEAK_IPL_ImageConverter_SetConversionMode = (dyn_PEAK_IPL_ImageConverter_SetConversionMode) (load ?  import_function(m_handle, "PEAK_IPL_ImageConverter_SetConversionMode") : nullptr);
    if(m_PEAK_IPL_ImageConverter_SetConversionMode == nullptr && load)
    {
        throw std::runtime_error("Failed to load PEAK_IPL_ImageConverter_SetConversionMode");
    }        

    m_PEAK_IPL_ImageConverter_GetSupportedOutputPixelFormats = (dyn_PEAK_IPL_ImageConverter_GetSupportedOutputPixelFormats) (load ?  import_function(m_handle, "PEAK_IPL_ImageConverter_GetSupportedOutputPixelFormats") : nullptr);
    if(m_PEAK_IPL_ImageConverter_GetSupportedOutputPixelFormats == nullptr && load)
    {
        throw std::runtime_error("Failed to load PEAK_IPL_ImageConverter_GetSupportedOutputPixelFormats");
    }        

    m_PEAK_IPL_ImageConverter_Convert = (dyn_PEAK_IPL_ImageConverter_Convert) (load ?  import_function(m_handle, "PEAK_IPL_ImageConverter_Convert") : nullptr);
    if(m_PEAK_IPL_ImageConverter_Convert == nullptr && load)
    {
        throw std::runtime_error("Failed to load PEAK_IPL_ImageConverter_Convert");
    }        

    m_PEAK_IPL_ImageConverter_ConvertToBuffer = (dyn_PEAK_IPL_ImageConverter_ConvertToBuffer) (load ?  import_function(m_handle, "PEAK_IPL_ImageConverter_ConvertToBuffer") : nullptr);
    if(m_PEAK_IPL_ImageConverter_ConvertToBuffer == nullptr && load)
    {
        throw std::runtime_error("Failed to load PEAK_IPL_ImageConverter_ConvertToBuffer");
    }        

    m_PEAK_IPL_ImageConverter_Destruct = (dyn_PEAK_IPL_ImageConverter_Destruct) (load ?  import_function(m_handle, "PEAK_IPL_ImageConverter_Destruct") : nullptr);
    if(m_PEAK_IPL_ImageConverter_Destruct == nullptr && load)
    {
        throw std::runtime_error("Failed to load PEAK_IPL_ImageConverter_Destruct");
    }        

    m_PEAK_IPL_AdaptiveHotpixelCorrector_Construct = (dyn_PEAK_IPL_AdaptiveHotpixelCorrector_Construct) (load ?  import_function(m_handle, "PEAK_IPL_AdaptiveHotpixelCorrector_Construct") : nullptr);
    if(m_PEAK_IPL_AdaptiveHotpixelCorrector_Construct == nullptr && load)
    {
        throw std::runtime_error("Failed to load PEAK_IPL_AdaptiveHotpixelCorrector_Construct");
    }        

    m_PEAK_IPL_AdaptiveHotpixelCorrector_SetSensitivity = (dyn_PEAK_IPL_AdaptiveHotpixelCorrector_SetSensitivity) (load ?  import_function(m_handle, "PEAK_IPL_AdaptiveHotpixelCorrector_SetSensitivity") : nullptr);
    if(m_PEAK_IPL_AdaptiveHotpixelCorrector_SetSensitivity == nullptr && load)
    {
        throw std::runtime_error("Failed to load PEAK_IPL_AdaptiveHotpixelCorrector_SetSensitivity");
    }        

    m_PEAK_IPL_AdaptiveHotpixelCorrector_GetSensitivity = (dyn_PEAK_IPL_AdaptiveHotpixelCorrector_GetSensitivity) (load ?  import_function(m_handle, "PEAK_IPL_AdaptiveHotpixelCorrector_GetSensitivity") : nullptr);
    if(m_PEAK_IPL_AdaptiveHotpixelCorrector_GetSensitivity == nullptr && load)
    {
        throw std::runtime_error("Failed to load PEAK_IPL_AdaptiveHotpixelCorrector_GetSensitivity");
    }        

    m_PEAK_IPL_AdaptiveHotpixelCorrector_SetGainFactorPercent = (dyn_PEAK_IPL_AdaptiveHotpixelCorrector_SetGainFactorPercent) (load ?  import_function(m_handle, "PEAK_IPL_AdaptiveHotpixelCorrector_SetGainFactorPercent") : nullptr);
    if(m_PEAK_IPL_AdaptiveHotpixelCorrector_SetGainFactorPercent == nullptr && load)
    {
        throw std::runtime_error("Failed to load PEAK_IPL_AdaptiveHotpixelCorrector_SetGainFactorPercent");
    }        

    m_PEAK_IPL_AdaptiveHotpixelCorrector_GetGainFactorPercent = (dyn_PEAK_IPL_AdaptiveHotpixelCorrector_GetGainFactorPercent) (load ?  import_function(m_handle, "PEAK_IPL_AdaptiveHotpixelCorrector_GetGainFactorPercent") : nullptr);
    if(m_PEAK_IPL_AdaptiveHotpixelCorrector_GetGainFactorPercent == nullptr && load)
    {
        throw std::runtime_error("Failed to load PEAK_IPL_AdaptiveHotpixelCorrector_GetGainFactorPercent");
    }        

    m_PEAK_IPL_AdaptiveHotpixelCorrector_Detect = (dyn_PEAK_IPL_AdaptiveHotpixelCorrector_Detect) (load ?  import_function(m_handle, "PEAK_IPL_AdaptiveHotpixelCorrector_Detect") : nullptr);
    if(m_PEAK_IPL_AdaptiveHotpixelCorrector_Detect == nullptr && load)
    {
        throw std::runtime_error("Failed to load PEAK_IPL_AdaptiveHotpixelCorrector_Detect");
    }        

    m_PEAK_IPL_AdaptiveHotpixelCorrector_GetHotpixels = (dyn_PEAK_IPL_AdaptiveHotpixelCorrector_GetHotpixels) (load ?  import_function(m_handle, "PEAK_IPL_AdaptiveHotpixelCorrector_GetHotpixels") : nullptr);
    if(m_PEAK_IPL_AdaptiveHotpixelCorrector_GetHotpixels == nullptr && load)
    {
        throw std::runtime_error("Failed to load PEAK_IPL_AdaptiveHotpixelCorrector_GetHotpixels");
    }        

    m_PEAK_IPL_AdaptiveHotpixelCorrector_Correct = (dyn_PEAK_IPL_AdaptiveHotpixelCorrector_Correct) (load ?  import_function(m_handle, "PEAK_IPL_AdaptiveHotpixelCorrector_Correct") : nullptr);
    if(m_PEAK_IPL_AdaptiveHotpixelCorrector_Correct == nullptr && load)
    {
        throw std::runtime_error("Failed to load PEAK_IPL_AdaptiveHotpixelCorrector_Correct");
    }        

    m_PEAK_IPL_AdaptiveHotpixelCorrector_CorrectAdaptive = (dyn_PEAK_IPL_AdaptiveHotpixelCorrector_CorrectAdaptive) (load ?  import_function(m_handle, "PEAK_IPL_AdaptiveHotpixelCorrector_CorrectAdaptive") : nullptr);
    if(m_PEAK_IPL_AdaptiveHotpixelCorrector_CorrectAdaptive == nullptr && load)
    {
        throw std::runtime_error("Failed to load PEAK_IPL_AdaptiveHotpixelCorrector_CorrectAdaptive");
    }        

    m_PEAK_IPL_AdaptiveHotpixelCorrector_Destruct = (dyn_PEAK_IPL_AdaptiveHotpixelCorrector_Destruct) (load ?  import_function(m_handle, "PEAK_IPL_AdaptiveHotpixelCorrector_Destruct") : nullptr);
    if(m_PEAK_IPL_AdaptiveHotpixelCorrector_Destruct == nullptr && load)
    {
        throw std::runtime_error("Failed to load PEAK_IPL_AdaptiveHotpixelCorrector_Destruct");
    }        

    m_PEAK_IPL_ColorCorrector_Construct = (dyn_PEAK_IPL_ColorCorrector_Construct) (load ?  import_function(m_handle, "PEAK_IPL_ColorCorrector_Construct") : nullptr);
    if(m_PEAK_IPL_ColorCorrector_Construct == nullptr && load)
    {
        throw std::runtime_error("Failed to load PEAK_IPL_ColorCorrector_Construct");
    }        

    m_PEAK_IPL_ColorCorrector_SetColorCorrectionFactors = (dyn_PEAK_IPL_ColorCorrector_SetColorCorrectionFactors) (load ?  import_function(m_handle, "PEAK_IPL_ColorCorrector_SetColorCorrectionFactors") : nullptr);
    if(m_PEAK_IPL_ColorCorrector_SetColorCorrectionFactors == nullptr && load)
    {
        throw std::runtime_error("Failed to load PEAK_IPL_ColorCorrector_SetColorCorrectionFactors");
    }        

    m_PEAK_IPL_ColorCorrector_GetColorCorrectionFactors = (dyn_PEAK_IPL_ColorCorrector_GetColorCorrectionFactors) (load ?  import_function(m_handle, "PEAK_IPL_ColorCorrector_GetColorCorrectionFactors") : nullptr);
    if(m_PEAK_IPL_ColorCorrector_GetColorCorrectionFactors == nullptr && load)
    {
        throw std::runtime_error("Failed to load PEAK_IPL_ColorCorrector_GetColorCorrectionFactors");
    }        

    m_PEAK_IPL_ColorCorrector_GetIsPixelFormatSupported = (dyn_PEAK_IPL_ColorCorrector_GetIsPixelFormatSupported) (load ?  import_function(m_handle, "PEAK_IPL_ColorCorrector_GetIsPixelFormatSupported") : nullptr);
    if(m_PEAK_IPL_ColorCorrector_GetIsPixelFormatSupported == nullptr && load)
    {
        throw std::runtime_error("Failed to load PEAK_IPL_ColorCorrector_GetIsPixelFormatSupported");
    }        

    m_PEAK_IPL_ColorCorrector_Process = (dyn_PEAK_IPL_ColorCorrector_Process) (load ?  import_function(m_handle, "PEAK_IPL_ColorCorrector_Process") : nullptr);
    if(m_PEAK_IPL_ColorCorrector_Process == nullptr && load)
    {
        throw std::runtime_error("Failed to load PEAK_IPL_ColorCorrector_Process");
    }        

    m_PEAK_IPL_ColorCorrector_ProcessInPlace = (dyn_PEAK_IPL_ColorCorrector_ProcessInPlace) (load ?  import_function(m_handle, "PEAK_IPL_ColorCorrector_ProcessInPlace") : nullptr);
    if(m_PEAK_IPL_ColorCorrector_ProcessInPlace == nullptr && load)
    {
        throw std::runtime_error("Failed to load PEAK_IPL_ColorCorrector_ProcessInPlace");
    }        

    m_PEAK_IPL_ColorCorrector_Destruct = (dyn_PEAK_IPL_ColorCorrector_Destruct) (load ?  import_function(m_handle, "PEAK_IPL_ColorCorrector_Destruct") : nullptr);
    if(m_PEAK_IPL_ColorCorrector_Destruct == nullptr && load)
    {
        throw std::runtime_error("Failed to load PEAK_IPL_ColorCorrector_Destruct");
    }        

    m_PEAK_IPL_GammaCorrector_Construct = (dyn_PEAK_IPL_GammaCorrector_Construct) (load ?  import_function(m_handle, "PEAK_IPL_GammaCorrector_Construct") : nullptr);
    if(m_PEAK_IPL_GammaCorrector_Construct == nullptr && load)
    {
        throw std::runtime_error("Failed to load PEAK_IPL_GammaCorrector_Construct");
    }        

    m_PEAK_IPL_GammaCorrector_SetGammaCorrectionValue = (dyn_PEAK_IPL_GammaCorrector_SetGammaCorrectionValue) (load ?  import_function(m_handle, "PEAK_IPL_GammaCorrector_SetGammaCorrectionValue") : nullptr);
    if(m_PEAK_IPL_GammaCorrector_SetGammaCorrectionValue == nullptr && load)
    {
        throw std::runtime_error("Failed to load PEAK_IPL_GammaCorrector_SetGammaCorrectionValue");
    }        

    m_PEAK_IPL_GammaCorrector_GetGammaCorrectionValue = (dyn_PEAK_IPL_GammaCorrector_GetGammaCorrectionValue) (load ?  import_function(m_handle, "PEAK_IPL_GammaCorrector_GetGammaCorrectionValue") : nullptr);
    if(m_PEAK_IPL_GammaCorrector_GetGammaCorrectionValue == nullptr && load)
    {
        throw std::runtime_error("Failed to load PEAK_IPL_GammaCorrector_GetGammaCorrectionValue");
    }        

    m_PEAK_IPL_GammaCorrector_GetGammaCorrectionMax = (dyn_PEAK_IPL_GammaCorrector_GetGammaCorrectionMax) (load ?  import_function(m_handle, "PEAK_IPL_GammaCorrector_GetGammaCorrectionMax") : nullptr);
    if(m_PEAK_IPL_GammaCorrector_GetGammaCorrectionMax == nullptr && load)
    {
        throw std::runtime_error("Failed to load PEAK_IPL_GammaCorrector_GetGammaCorrectionMax");
    }        

    m_PEAK_IPL_GammaCorrector_GetGammaCorrectionMin = (dyn_PEAK_IPL_GammaCorrector_GetGammaCorrectionMin) (load ?  import_function(m_handle, "PEAK_IPL_GammaCorrector_GetGammaCorrectionMin") : nullptr);
    if(m_PEAK_IPL_GammaCorrector_GetGammaCorrectionMin == nullptr && load)
    {
        throw std::runtime_error("Failed to load PEAK_IPL_GammaCorrector_GetGammaCorrectionMin");
    }        

    m_PEAK_IPL_GammaCorrector_GetIsPixelFormatSupported = (dyn_PEAK_IPL_GammaCorrector_GetIsPixelFormatSupported) (load ?  import_function(m_handle, "PEAK_IPL_GammaCorrector_GetIsPixelFormatSupported") : nullptr);
    if(m_PEAK_IPL_GammaCorrector_GetIsPixelFormatSupported == nullptr && load)
    {
        throw std::runtime_error("Failed to load PEAK_IPL_GammaCorrector_GetIsPixelFormatSupported");
    }        

    m_PEAK_IPL_GammaCorrector_Process = (dyn_PEAK_IPL_GammaCorrector_Process) (load ?  import_function(m_handle, "PEAK_IPL_GammaCorrector_Process") : nullptr);
    if(m_PEAK_IPL_GammaCorrector_Process == nullptr && load)
    {
        throw std::runtime_error("Failed to load PEAK_IPL_GammaCorrector_Process");
    }        

    m_PEAK_IPL_GammaCorrector_ProcessInPlace = (dyn_PEAK_IPL_GammaCorrector_ProcessInPlace) (load ?  import_function(m_handle, "PEAK_IPL_GammaCorrector_ProcessInPlace") : nullptr);
    if(m_PEAK_IPL_GammaCorrector_ProcessInPlace == nullptr && load)
    {
        throw std::runtime_error("Failed to load PEAK_IPL_GammaCorrector_ProcessInPlace");
    }        

    m_PEAK_IPL_GammaCorrector_Destruct = (dyn_PEAK_IPL_GammaCorrector_Destruct) (load ?  import_function(m_handle, "PEAK_IPL_GammaCorrector_Destruct") : nullptr);
    if(m_PEAK_IPL_GammaCorrector_Destruct == nullptr && load)
    {
        throw std::runtime_error("Failed to load PEAK_IPL_GammaCorrector_Destruct");
    }        

    m_PEAK_IPL_Image_Construct = (dyn_PEAK_IPL_Image_Construct) (load ?  import_function(m_handle, "PEAK_IPL_Image_Construct") : nullptr);
    if(m_PEAK_IPL_Image_Construct == nullptr && load)
    {
        throw std::runtime_error("Failed to load PEAK_IPL_Image_Construct");
    }        

    m_PEAK_IPL_Image_ConstructFromBuffer = (dyn_PEAK_IPL_Image_ConstructFromBuffer) (load ?  import_function(m_handle, "PEAK_IPL_Image_ConstructFromBuffer") : nullptr);
    if(m_PEAK_IPL_Image_ConstructFromBuffer == nullptr && load)
    {
        throw std::runtime_error("Failed to load PEAK_IPL_Image_ConstructFromBuffer");
    }        

    m_PEAK_IPL_Image_GetWidth = (dyn_PEAK_IPL_Image_GetWidth) (load ?  import_function(m_handle, "PEAK_IPL_Image_GetWidth") : nullptr);
    if(m_PEAK_IPL_Image_GetWidth == nullptr && load)
    {
        throw std::runtime_error("Failed to load PEAK_IPL_Image_GetWidth");
    }        

    m_PEAK_IPL_Image_GetHeight = (dyn_PEAK_IPL_Image_GetHeight) (load ?  import_function(m_handle, "PEAK_IPL_Image_GetHeight") : nullptr);
    if(m_PEAK_IPL_Image_GetHeight == nullptr && load)
    {
        throw std::runtime_error("Failed to load PEAK_IPL_Image_GetHeight");
    }        

    m_PEAK_IPL_Image_GetPixelPointer = (dyn_PEAK_IPL_Image_GetPixelPointer) (load ?  import_function(m_handle, "PEAK_IPL_Image_GetPixelPointer") : nullptr);
    if(m_PEAK_IPL_Image_GetPixelPointer == nullptr && load)
    {
        throw std::runtime_error("Failed to load PEAK_IPL_Image_GetPixelPointer");
    }        

    m_PEAK_IPL_Image_GetByteCount = (dyn_PEAK_IPL_Image_GetByteCount) (load ?  import_function(m_handle, "PEAK_IPL_Image_GetByteCount") : nullptr);
    if(m_PEAK_IPL_Image_GetByteCount == nullptr && load)
    {
        throw std::runtime_error("Failed to load PEAK_IPL_Image_GetByteCount");
    }        

    m_PEAK_IPL_Image_GetPixelFormat = (dyn_PEAK_IPL_Image_GetPixelFormat) (load ?  import_function(m_handle, "PEAK_IPL_Image_GetPixelFormat") : nullptr);
    if(m_PEAK_IPL_Image_GetPixelFormat == nullptr && load)
    {
        throw std::runtime_error("Failed to load PEAK_IPL_Image_GetPixelFormat");
    }        

    m_PEAK_IPL_Image_GetData = (dyn_PEAK_IPL_Image_GetData) (load ?  import_function(m_handle, "PEAK_IPL_Image_GetData") : nullptr);
    if(m_PEAK_IPL_Image_GetData == nullptr && load)
    {
        throw std::runtime_error("Failed to load PEAK_IPL_Image_GetData");
    }        

    m_PEAK_IPL_Image_Clone = (dyn_PEAK_IPL_Image_Clone) (load ?  import_function(m_handle, "PEAK_IPL_Image_Clone") : nullptr);
    if(m_PEAK_IPL_Image_Clone == nullptr && load)
    {
        throw std::runtime_error("Failed to load PEAK_IPL_Image_Clone");
    }        

    m_PEAK_IPL_Image_ConvertTo = (dyn_PEAK_IPL_Image_ConvertTo) (load ?  import_function(m_handle, "PEAK_IPL_Image_ConvertTo") : nullptr);
    if(m_PEAK_IPL_Image_ConvertTo == nullptr && load)
    {
        throw std::runtime_error("Failed to load PEAK_IPL_Image_ConvertTo");
    }        

    m_PEAK_IPL_Image_ConvertToBuffer = (dyn_PEAK_IPL_Image_ConvertToBuffer) (load ?  import_function(m_handle, "PEAK_IPL_Image_ConvertToBuffer") : nullptr);
    if(m_PEAK_IPL_Image_ConvertToBuffer == nullptr && load)
    {
        throw std::runtime_error("Failed to load PEAK_IPL_Image_ConvertToBuffer");
    }        

    m_PEAK_IPL_Image_Destruct = (dyn_PEAK_IPL_Image_Destruct) (load ?  import_function(m_handle, "PEAK_IPL_Image_Destruct") : nullptr);
    if(m_PEAK_IPL_Image_Destruct == nullptr && load)
    {
        throw std::runtime_error("Failed to load PEAK_IPL_Image_Destruct");
    }        

    m_PEAK_IPL_ImageTransformer_Construct = (dyn_PEAK_IPL_ImageTransformer_Construct) (load ?  import_function(m_handle, "PEAK_IPL_ImageTransformer_Construct") : nullptr);
    if(m_PEAK_IPL_ImageTransformer_Construct == nullptr && load)
    {
        throw std::runtime_error("Failed to load PEAK_IPL_ImageTransformer_Construct");
    }        

    m_PEAK_IPL_ImageTransformer_MirrorUpDown = (dyn_PEAK_IPL_ImageTransformer_MirrorUpDown) (load ?  import_function(m_handle, "PEAK_IPL_ImageTransformer_MirrorUpDown") : nullptr);
    if(m_PEAK_IPL_ImageTransformer_MirrorUpDown == nullptr && load)
    {
        throw std::runtime_error("Failed to load PEAK_IPL_ImageTransformer_MirrorUpDown");
    }        

    m_PEAK_IPL_ImageTransformer_MirrorLeftRight = (dyn_PEAK_IPL_ImageTransformer_MirrorLeftRight) (load ?  import_function(m_handle, "PEAK_IPL_ImageTransformer_MirrorLeftRight") : nullptr);
    if(m_PEAK_IPL_ImageTransformer_MirrorLeftRight == nullptr && load)
    {
        throw std::runtime_error("Failed to load PEAK_IPL_ImageTransformer_MirrorLeftRight");
    }        

    m_PEAK_IPL_ImageTransformer_MirrorUpDownLeftRight = (dyn_PEAK_IPL_ImageTransformer_MirrorUpDownLeftRight) (load ?  import_function(m_handle, "PEAK_IPL_ImageTransformer_MirrorUpDownLeftRight") : nullptr);
    if(m_PEAK_IPL_ImageTransformer_MirrorUpDownLeftRight == nullptr && load)
    {
        throw std::runtime_error("Failed to load PEAK_IPL_ImageTransformer_MirrorUpDownLeftRight");
    }        

    m_PEAK_IPL_ImageTransformer_MirrorUpDownInPlace = (dyn_PEAK_IPL_ImageTransformer_MirrorUpDownInPlace) (load ?  import_function(m_handle, "PEAK_IPL_ImageTransformer_MirrorUpDownInPlace") : nullptr);
    if(m_PEAK_IPL_ImageTransformer_MirrorUpDownInPlace == nullptr && load)
    {
        throw std::runtime_error("Failed to load PEAK_IPL_ImageTransformer_MirrorUpDownInPlace");
    }        

    m_PEAK_IPL_ImageTransformer_MirrorLeftRightInPlace = (dyn_PEAK_IPL_ImageTransformer_MirrorLeftRightInPlace) (load ?  import_function(m_handle, "PEAK_IPL_ImageTransformer_MirrorLeftRightInPlace") : nullptr);
    if(m_PEAK_IPL_ImageTransformer_MirrorLeftRightInPlace == nullptr && load)
    {
        throw std::runtime_error("Failed to load PEAK_IPL_ImageTransformer_MirrorLeftRightInPlace");
    }        

    m_PEAK_IPL_ImageTransformer_MirrorUpDownLeftRightInPlace = (dyn_PEAK_IPL_ImageTransformer_MirrorUpDownLeftRightInPlace) (load ?  import_function(m_handle, "PEAK_IPL_ImageTransformer_MirrorUpDownLeftRightInPlace") : nullptr);
    if(m_PEAK_IPL_ImageTransformer_MirrorUpDownLeftRightInPlace == nullptr && load)
    {
        throw std::runtime_error("Failed to load PEAK_IPL_ImageTransformer_MirrorUpDownLeftRightInPlace");
    }        

    m_PEAK_IPL_ImageTransformer_Rotate = (dyn_PEAK_IPL_ImageTransformer_Rotate) (load ?  import_function(m_handle, "PEAK_IPL_ImageTransformer_Rotate") : nullptr);
    if(m_PEAK_IPL_ImageTransformer_Rotate == nullptr && load)
    {
        throw std::runtime_error("Failed to load PEAK_IPL_ImageTransformer_Rotate");
    }        

    m_PEAK_IPL_ImageTransformer_RotateInPlace = (dyn_PEAK_IPL_ImageTransformer_RotateInPlace) (load ?  import_function(m_handle, "PEAK_IPL_ImageTransformer_RotateInPlace") : nullptr);
    if(m_PEAK_IPL_ImageTransformer_RotateInPlace == nullptr && load)
    {
        throw std::runtime_error("Failed to load PEAK_IPL_ImageTransformer_RotateInPlace");
    }        

    m_PEAK_IPL_ImageTransformer_Destruct = (dyn_PEAK_IPL_ImageTransformer_Destruct) (load ?  import_function(m_handle, "PEAK_IPL_ImageTransformer_Destruct") : nullptr);
    if(m_PEAK_IPL_ImageTransformer_Destruct == nullptr && load)
    {
        throw std::runtime_error("Failed to load PEAK_IPL_ImageTransformer_Destruct");
    }        

    m_PEAK_IPL_Histogram_Construct = (dyn_PEAK_IPL_Histogram_Construct) (load ?  import_function(m_handle, "PEAK_IPL_Histogram_Construct") : nullptr);
    if(m_PEAK_IPL_Histogram_Construct == nullptr && load)
    {
        throw std::runtime_error("Failed to load PEAK_IPL_Histogram_Construct");
    }        

    m_PEAK_IPL_Histogram_GetPixelFormat = (dyn_PEAK_IPL_Histogram_GetPixelFormat) (load ?  import_function(m_handle, "PEAK_IPL_Histogram_GetPixelFormat") : nullptr);
    if(m_PEAK_IPL_Histogram_GetPixelFormat == nullptr && load)
    {
        throw std::runtime_error("Failed to load PEAK_IPL_Histogram_GetPixelFormat");
    }        

    m_PEAK_IPL_Histogram_GetNumChannels = (dyn_PEAK_IPL_Histogram_GetNumChannels) (load ?  import_function(m_handle, "PEAK_IPL_Histogram_GetNumChannels") : nullptr);
    if(m_PEAK_IPL_Histogram_GetNumChannels == nullptr && load)
    {
        throw std::runtime_error("Failed to load PEAK_IPL_Histogram_GetNumChannels");
    }        

    m_PEAK_IPL_Histogram_GetPixelSumForChannel = (dyn_PEAK_IPL_Histogram_GetPixelSumForChannel) (load ?  import_function(m_handle, "PEAK_IPL_Histogram_GetPixelSumForChannel") : nullptr);
    if(m_PEAK_IPL_Histogram_GetPixelSumForChannel == nullptr && load)
    {
        throw std::runtime_error("Failed to load PEAK_IPL_Histogram_GetPixelSumForChannel");
    }        

    m_PEAK_IPL_Histogram_GetPixelCountForChannel = (dyn_PEAK_IPL_Histogram_GetPixelCountForChannel) (load ?  import_function(m_handle, "PEAK_IPL_Histogram_GetPixelCountForChannel") : nullptr);
    if(m_PEAK_IPL_Histogram_GetPixelCountForChannel == nullptr && load)
    {
        throw std::runtime_error("Failed to load PEAK_IPL_Histogram_GetPixelCountForChannel");
    }        

    m_PEAK_IPL_Histogram_GetBinsForChannel = (dyn_PEAK_IPL_Histogram_GetBinsForChannel) (load ?  import_function(m_handle, "PEAK_IPL_Histogram_GetBinsForChannel") : nullptr);
    if(m_PEAK_IPL_Histogram_GetBinsForChannel == nullptr && load)
    {
        throw std::runtime_error("Failed to load PEAK_IPL_Histogram_GetBinsForChannel");
    }        

    m_PEAK_IPL_Histogram_Destruct = (dyn_PEAK_IPL_Histogram_Destruct) (load ?  import_function(m_handle, "PEAK_IPL_Histogram_Destruct") : nullptr);
    if(m_PEAK_IPL_Histogram_Destruct == nullptr && load)
    {
        throw std::runtime_error("Failed to load PEAK_IPL_Histogram_Destruct");
    }        

    m_PEAK_IPL_PixelLine_Construct = (dyn_PEAK_IPL_PixelLine_Construct) (load ?  import_function(m_handle, "PEAK_IPL_PixelLine_Construct") : nullptr);
    if(m_PEAK_IPL_PixelLine_Construct == nullptr && load)
    {
        throw std::runtime_error("Failed to load PEAK_IPL_PixelLine_Construct");
    }        

    m_PEAK_IPL_PixelLine_GetPixelFormat = (dyn_PEAK_IPL_PixelLine_GetPixelFormat) (load ?  import_function(m_handle, "PEAK_IPL_PixelLine_GetPixelFormat") : nullptr);
    if(m_PEAK_IPL_PixelLine_GetPixelFormat == nullptr && load)
    {
        throw std::runtime_error("Failed to load PEAK_IPL_PixelLine_GetPixelFormat");
    }        

    m_PEAK_IPL_PixelLine_GetOrientation = (dyn_PEAK_IPL_PixelLine_GetOrientation) (load ?  import_function(m_handle, "PEAK_IPL_PixelLine_GetOrientation") : nullptr);
    if(m_PEAK_IPL_PixelLine_GetOrientation == nullptr && load)
    {
        throw std::runtime_error("Failed to load PEAK_IPL_PixelLine_GetOrientation");
    }        

    m_PEAK_IPL_PixelLine_GetOffset = (dyn_PEAK_IPL_PixelLine_GetOffset) (load ?  import_function(m_handle, "PEAK_IPL_PixelLine_GetOffset") : nullptr);
    if(m_PEAK_IPL_PixelLine_GetOffset == nullptr && load)
    {
        throw std::runtime_error("Failed to load PEAK_IPL_PixelLine_GetOffset");
    }        

    m_PEAK_IPL_PixelLine_GetNumChannels = (dyn_PEAK_IPL_PixelLine_GetNumChannels) (load ?  import_function(m_handle, "PEAK_IPL_PixelLine_GetNumChannels") : nullptr);
    if(m_PEAK_IPL_PixelLine_GetNumChannels == nullptr && load)
    {
        throw std::runtime_error("Failed to load PEAK_IPL_PixelLine_GetNumChannels");
    }        

    m_PEAK_IPL_PixelLine_GetValuesForChannel = (dyn_PEAK_IPL_PixelLine_GetValuesForChannel) (load ?  import_function(m_handle, "PEAK_IPL_PixelLine_GetValuesForChannel") : nullptr);
    if(m_PEAK_IPL_PixelLine_GetValuesForChannel == nullptr && load)
    {
        throw std::runtime_error("Failed to load PEAK_IPL_PixelLine_GetValuesForChannel");
    }        

    m_PEAK_IPL_PixelLine_Destruct = (dyn_PEAK_IPL_PixelLine_Destruct) (load ?  import_function(m_handle, "PEAK_IPL_PixelLine_Destruct") : nullptr);
    if(m_PEAK_IPL_PixelLine_Destruct == nullptr && load)
    {
        throw std::runtime_error("Failed to load PEAK_IPL_PixelLine_Destruct");
    }        

    m_PEAK_IPL_PixelFormat_GetNumChannels = (dyn_PEAK_IPL_PixelFormat_GetNumChannels) (load ?  import_function(m_handle, "PEAK_IPL_PixelFormat_GetNumChannels") : nullptr);
    if(m_PEAK_IPL_PixelFormat_GetNumChannels == nullptr && load)
    {
        throw std::runtime_error("Failed to load PEAK_IPL_PixelFormat_GetNumChannels");
    }        

    m_PEAK_IPL_PixelFormat_GetNumSignificantBitsPerChannel = (dyn_PEAK_IPL_PixelFormat_GetNumSignificantBitsPerChannel) (load ?  import_function(m_handle, "PEAK_IPL_PixelFormat_GetNumSignificantBitsPerChannel") : nullptr);
    if(m_PEAK_IPL_PixelFormat_GetNumSignificantBitsPerChannel == nullptr && load)
    {
        throw std::runtime_error("Failed to load PEAK_IPL_PixelFormat_GetNumSignificantBitsPerChannel");
    }        

    m_PEAK_IPL_PixelFormat_GetNumStorageBitsPerChannel = (dyn_PEAK_IPL_PixelFormat_GetNumStorageBitsPerChannel) (load ?  import_function(m_handle, "PEAK_IPL_PixelFormat_GetNumStorageBitsPerChannel") : nullptr);
    if(m_PEAK_IPL_PixelFormat_GetNumStorageBitsPerChannel == nullptr && load)
    {
        throw std::runtime_error("Failed to load PEAK_IPL_PixelFormat_GetNumStorageBitsPerChannel");
    }        

    m_PEAK_IPL_PixelFormat_GetMaximumValuePerChannel = (dyn_PEAK_IPL_PixelFormat_GetMaximumValuePerChannel) (load ?  import_function(m_handle, "PEAK_IPL_PixelFormat_GetMaximumValuePerChannel") : nullptr);
    if(m_PEAK_IPL_PixelFormat_GetMaximumValuePerChannel == nullptr && load)
    {
        throw std::runtime_error("Failed to load PEAK_IPL_PixelFormat_GetMaximumValuePerChannel");
    }        

    m_PEAK_IPL_PixelFormat_GetNumSignificantBitsPerPixel = (dyn_PEAK_IPL_PixelFormat_GetNumSignificantBitsPerPixel) (load ?  import_function(m_handle, "PEAK_IPL_PixelFormat_GetNumSignificantBitsPerPixel") : nullptr);
    if(m_PEAK_IPL_PixelFormat_GetNumSignificantBitsPerPixel == nullptr && load)
    {
        throw std::runtime_error("Failed to load PEAK_IPL_PixelFormat_GetNumSignificantBitsPerPixel");
    }        

    m_PEAK_IPL_PixelFormat_GetNumStorageBitsPerPixel = (dyn_PEAK_IPL_PixelFormat_GetNumStorageBitsPerPixel) (load ?  import_function(m_handle, "PEAK_IPL_PixelFormat_GetNumStorageBitsPerPixel") : nullptr);
    if(m_PEAK_IPL_PixelFormat_GetNumStorageBitsPerPixel == nullptr && load)
    {
        throw std::runtime_error("Failed to load PEAK_IPL_PixelFormat_GetNumStorageBitsPerPixel");
    }        

    m_PEAK_IPL_PixelFormat_GetEndianness = (dyn_PEAK_IPL_PixelFormat_GetEndianness) (load ?  import_function(m_handle, "PEAK_IPL_PixelFormat_GetEndianness") : nullptr);
    if(m_PEAK_IPL_PixelFormat_GetEndianness == nullptr && load)
    {
        throw std::runtime_error("Failed to load PEAK_IPL_PixelFormat_GetEndianness");
    }        

    m_PEAK_IPL_PixelFormat_CalculateStorageSizeOfPixels = (dyn_PEAK_IPL_PixelFormat_CalculateStorageSizeOfPixels) (load ?  import_function(m_handle, "PEAK_IPL_PixelFormat_CalculateStorageSizeOfPixels") : nullptr);
    if(m_PEAK_IPL_PixelFormat_CalculateStorageSizeOfPixels == nullptr && load)
    {
        throw std::runtime_error("Failed to load PEAK_IPL_PixelFormat_CalculateStorageSizeOfPixels");
    }        

    m_PEAK_IPL_ImageWriter_WriteAsRAW = (dyn_PEAK_IPL_ImageWriter_WriteAsRAW) (load ?  import_function(m_handle, "PEAK_IPL_ImageWriter_WriteAsRAW") : nullptr);
    if(m_PEAK_IPL_ImageWriter_WriteAsRAW == nullptr && load)
    {
        throw std::runtime_error("Failed to load PEAK_IPL_ImageWriter_WriteAsRAW");
    }        

    m_PEAK_IPL_ImageWriter_WriteAsBMP = (dyn_PEAK_IPL_ImageWriter_WriteAsBMP) (load ?  import_function(m_handle, "PEAK_IPL_ImageWriter_WriteAsBMP") : nullptr);
    if(m_PEAK_IPL_ImageWriter_WriteAsBMP == nullptr && load)
    {
        throw std::runtime_error("Failed to load PEAK_IPL_ImageWriter_WriteAsBMP");
    }        

    m_PEAK_IPL_ImageWriter_WriteAsPNG = (dyn_PEAK_IPL_ImageWriter_WriteAsPNG) (load ?  import_function(m_handle, "PEAK_IPL_ImageWriter_WriteAsPNG") : nullptr);
    if(m_PEAK_IPL_ImageWriter_WriteAsPNG == nullptr && load)
    {
        throw std::runtime_error("Failed to load PEAK_IPL_ImageWriter_WriteAsPNG");
    }        

    m_PEAK_IPL_ImageWriter_WriteAsJPG = (dyn_PEAK_IPL_ImageWriter_WriteAsJPG) (load ?  import_function(m_handle, "PEAK_IPL_ImageWriter_WriteAsJPG") : nullptr);
    if(m_PEAK_IPL_ImageWriter_WriteAsJPG == nullptr && load)
    {
        throw std::runtime_error("Failed to load PEAK_IPL_ImageWriter_WriteAsJPG");
    }        

    m_PEAK_IPL_ImageWriter_Write = (dyn_PEAK_IPL_ImageWriter_Write) (load ?  import_function(m_handle, "PEAK_IPL_ImageWriter_Write") : nullptr);
    if(m_PEAK_IPL_ImageWriter_Write == nullptr && load)
    {
        throw std::runtime_error("Failed to load PEAK_IPL_ImageWriter_Write");
    }        

    m_PEAK_IPL_ImageReader_Read = (dyn_PEAK_IPL_ImageReader_Read) (load ?  import_function(m_handle, "PEAK_IPL_ImageReader_Read") : nullptr);
    if(m_PEAK_IPL_ImageReader_Read == nullptr && load)
    {
        throw std::runtime_error("Failed to load PEAK_IPL_ImageReader_Read");
    }        

    m_PEAK_IPL_ImageReaderRead_ReadAsPixelFormat = (dyn_PEAK_IPL_ImageReaderRead_ReadAsPixelFormat) (load ?  import_function(m_handle, "PEAK_IPL_ImageReaderRead_ReadAsPixelFormat") : nullptr);
    if(m_PEAK_IPL_ImageReaderRead_ReadAsPixelFormat == nullptr && load)
    {
        throw std::runtime_error("Failed to load PEAK_IPL_ImageReaderRead_ReadAsPixelFormat");
    }        

            
            return true;
}

inline PEAK_IPL_RETURN_CODE DynamicLoader::PEAK_IPL_Library_GetVersionMajor(uint32_t * libraryVersionMajor)
{
    auto& inst = instance();
    if(inst.m_PEAK_IPL_Library_GetVersionMajor)
    {
        return inst.m_PEAK_IPL_Library_GetVersionMajor(libraryVersionMajor);
    }
    else
    {
        throw std::runtime_error("Library not loaded!");
    }
}

inline PEAK_IPL_RETURN_CODE DynamicLoader::PEAK_IPL_Library_GetVersionMinor(uint32_t * libraryVersionMinor)
{
    auto& inst = instance();
    if(inst.m_PEAK_IPL_Library_GetVersionMinor)
    {
        return inst.m_PEAK_IPL_Library_GetVersionMinor(libraryVersionMinor);
    }
    else
    {
        throw std::runtime_error("Library not loaded!");
    }
}

inline PEAK_IPL_RETURN_CODE DynamicLoader::PEAK_IPL_Library_GetVersionSubminor(uint32_t * libraryVersionSubminor)
{
    auto& inst = instance();
    if(inst.m_PEAK_IPL_Library_GetVersionSubminor)
    {
        return inst.m_PEAK_IPL_Library_GetVersionSubminor(libraryVersionSubminor);
    }
    else
    {
        throw std::runtime_error("Library not loaded!");
    }
}

inline PEAK_IPL_RETURN_CODE DynamicLoader::PEAK_IPL_Library_GetLastError(PEAK_IPL_RETURN_CODE * lastErrorCode, char * lastErrorDescription, size_t * lastErrorDescriptionSize)
{
    auto& inst = instance();
    if(inst.m_PEAK_IPL_Library_GetLastError)
    {
        return inst.m_PEAK_IPL_Library_GetLastError(lastErrorCode, lastErrorDescription, lastErrorDescriptionSize);
    }
    else
    {
        throw std::runtime_error("Library not loaded!");
    }
}

inline PEAK_IPL_RETURN_CODE DynamicLoader::PEAK_IPL_ImageConverter_Construct(PEAK_IPL_IMAGE_CONVERTER_HANDLE * imageConverterHandle)
{
    auto& inst = instance();
    if(inst.m_PEAK_IPL_ImageConverter_Construct)
    {
        return inst.m_PEAK_IPL_ImageConverter_Construct(imageConverterHandle);
    }
    else
    {
        throw std::runtime_error("Library not loaded!");
    }
}

inline PEAK_IPL_RETURN_CODE DynamicLoader::PEAK_IPL_ImageConverter_GetConversionMode(PEAK_IPL_IMAGE_CONVERTER_HANDLE imageConverterHandle, PEAK_IPL_CONVERSION_MODE * conversionMode)
{
    auto& inst = instance();
    if(inst.m_PEAK_IPL_ImageConverter_GetConversionMode)
    {
        return inst.m_PEAK_IPL_ImageConverter_GetConversionMode(imageConverterHandle, conversionMode);
    }
    else
    {
        throw std::runtime_error("Library not loaded!");
    }
}

inline PEAK_IPL_RETURN_CODE DynamicLoader::PEAK_IPL_ImageConverter_SetConversionMode(PEAK_IPL_IMAGE_CONVERTER_HANDLE imageConverterHandle, PEAK_IPL_CONVERSION_MODE conversionMode)
{
    auto& inst = instance();
    if(inst.m_PEAK_IPL_ImageConverter_SetConversionMode)
    {
        return inst.m_PEAK_IPL_ImageConverter_SetConversionMode(imageConverterHandle, conversionMode);
    }
    else
    {
        throw std::runtime_error("Library not loaded!");
    }
}

inline PEAK_IPL_RETURN_CODE DynamicLoader::PEAK_IPL_ImageConverter_GetSupportedOutputPixelFormats(PEAK_IPL_IMAGE_CONVERTER_HANDLE imageConverterHandle, PEAK_IPL_PIXEL_FORMAT inputPixelFormat, PEAK_IPL_PIXEL_FORMAT * outputPixelFormats, size_t * outputPixelFormatsSize)
{
    auto& inst = instance();
    if(inst.m_PEAK_IPL_ImageConverter_GetSupportedOutputPixelFormats)
    {
        return inst.m_PEAK_IPL_ImageConverter_GetSupportedOutputPixelFormats(imageConverterHandle, inputPixelFormat, outputPixelFormats, outputPixelFormatsSize);
    }
    else
    {
        throw std::runtime_error("Library not loaded!");
    }
}

inline PEAK_IPL_RETURN_CODE DynamicLoader::PEAK_IPL_ImageConverter_Convert(PEAK_IPL_IMAGE_CONVERTER_HANDLE imageConverterHandle, PEAK_IPL_IMAGE_HANDLE inputImageHandle, PEAK_IPL_PIXEL_FORMAT outputPixelFormat, PEAK_IPL_IMAGE_HANDLE * outputImageHandle)
{
    auto& inst = instance();
    if(inst.m_PEAK_IPL_ImageConverter_Convert)
    {
        return inst.m_PEAK_IPL_ImageConverter_Convert(imageConverterHandle, inputImageHandle, outputPixelFormat, outputImageHandle);
    }
    else
    {
        throw std::runtime_error("Library not loaded!");
    }
}

inline PEAK_IPL_RETURN_CODE DynamicLoader::PEAK_IPL_ImageConverter_ConvertToBuffer(PEAK_IPL_IMAGE_CONVERTER_HANDLE imageConverterHandle, PEAK_IPL_IMAGE_HANDLE inputImageHandle, PEAK_IPL_PIXEL_FORMAT outputPixelFormat, uint8_t * outputImageBuffer, size_t outputImageBufferSize, PEAK_IPL_IMAGE_HANDLE * outputImageHandle)
{
    auto& inst = instance();
    if(inst.m_PEAK_IPL_ImageConverter_ConvertToBuffer)
    {
        return inst.m_PEAK_IPL_ImageConverter_ConvertToBuffer(imageConverterHandle, inputImageHandle, outputPixelFormat, outputImageBuffer, outputImageBufferSize, outputImageHandle);
    }
    else
    {
        throw std::runtime_error("Library not loaded!");
    }
}

inline PEAK_IPL_RETURN_CODE DynamicLoader::PEAK_IPL_ImageConverter_Destruct(PEAK_IPL_IMAGE_CONVERTER_HANDLE imageConverterHandle)
{
    auto& inst = instance();
    if(inst.m_PEAK_IPL_ImageConverter_Destruct)
    {
        return inst.m_PEAK_IPL_ImageConverter_Destruct(imageConverterHandle);
    }
    else
    {
        throw std::runtime_error("Library not loaded!");
    }
}

inline PEAK_IPL_RETURN_CODE DynamicLoader::PEAK_IPL_AdaptiveHotpixelCorrector_Construct(PEAK_IPL_ADAPTIVE_HOTPIXEL_CORRECTOR_HANDLE * adaptiveHotpixelCorrectorHandle)
{
    auto& inst = instance();
    if(inst.m_PEAK_IPL_AdaptiveHotpixelCorrector_Construct)
    {
        return inst.m_PEAK_IPL_AdaptiveHotpixelCorrector_Construct(adaptiveHotpixelCorrectorHandle);
    }
    else
    {
        throw std::runtime_error("Library not loaded!");
    }
}

inline PEAK_IPL_RETURN_CODE DynamicLoader::PEAK_IPL_AdaptiveHotpixelCorrector_SetSensitivity(PEAK_IPL_ADAPTIVE_HOTPIXEL_CORRECTOR_HANDLE adaptiveHotpixelCorrectorHandle, PEAK_IPL_HOTPIXELCORRECTION_SENSITIVITY sensitivityLevel)
{
    auto& inst = instance();
    if(inst.m_PEAK_IPL_AdaptiveHotpixelCorrector_SetSensitivity)
    {
        return inst.m_PEAK_IPL_AdaptiveHotpixelCorrector_SetSensitivity(adaptiveHotpixelCorrectorHandle, sensitivityLevel);
    }
    else
    {
        throw std::runtime_error("Library not loaded!");
    }
}

inline PEAK_IPL_RETURN_CODE DynamicLoader::PEAK_IPL_AdaptiveHotpixelCorrector_GetSensitivity(PEAK_IPL_ADAPTIVE_HOTPIXEL_CORRECTOR_HANDLE adaptiveHotpixelCorrectorHandle, PEAK_IPL_HOTPIXELCORRECTION_SENSITIVITY * sensitivityLevel)
{
    auto& inst = instance();
    if(inst.m_PEAK_IPL_AdaptiveHotpixelCorrector_GetSensitivity)
    {
        return inst.m_PEAK_IPL_AdaptiveHotpixelCorrector_GetSensitivity(adaptiveHotpixelCorrectorHandle, sensitivityLevel);
    }
    else
    {
        throw std::runtime_error("Library not loaded!");
    }
}

inline PEAK_IPL_RETURN_CODE DynamicLoader::PEAK_IPL_AdaptiveHotpixelCorrector_SetGainFactorPercent(PEAK_IPL_ADAPTIVE_HOTPIXEL_CORRECTOR_HANDLE adaptiveHotpixelCorrectorHandle, uint32_t gainFactorPercent)
{
    auto& inst = instance();
    if(inst.m_PEAK_IPL_AdaptiveHotpixelCorrector_SetGainFactorPercent)
    {
        return inst.m_PEAK_IPL_AdaptiveHotpixelCorrector_SetGainFactorPercent(adaptiveHotpixelCorrectorHandle, gainFactorPercent);
    }
    else
    {
        throw std::runtime_error("Library not loaded!");
    }
}

inline PEAK_IPL_RETURN_CODE DynamicLoader::PEAK_IPL_AdaptiveHotpixelCorrector_GetGainFactorPercent(PEAK_IPL_ADAPTIVE_HOTPIXEL_CORRECTOR_HANDLE adaptiveHotpixelCorrectorHandle, uint32_t * gainFactorPercent)
{
    auto& inst = instance();
    if(inst.m_PEAK_IPL_AdaptiveHotpixelCorrector_GetGainFactorPercent)
    {
        return inst.m_PEAK_IPL_AdaptiveHotpixelCorrector_GetGainFactorPercent(adaptiveHotpixelCorrectorHandle, gainFactorPercent);
    }
    else
    {
        throw std::runtime_error("Library not loaded!");
    }
}

inline PEAK_IPL_RETURN_CODE DynamicLoader::PEAK_IPL_AdaptiveHotpixelCorrector_Detect(PEAK_IPL_ADAPTIVE_HOTPIXEL_CORRECTOR_HANDLE adaptiveHotpixelCorrectorHandle, PEAK_IPL_IMAGE_HANDLE inputImageHandle)
{
    auto& inst = instance();
    if(inst.m_PEAK_IPL_AdaptiveHotpixelCorrector_Detect)
    {
        return inst.m_PEAK_IPL_AdaptiveHotpixelCorrector_Detect(adaptiveHotpixelCorrectorHandle, inputImageHandle);
    }
    else
    {
        throw std::runtime_error("Library not loaded!");
    }
}

inline PEAK_IPL_RETURN_CODE DynamicLoader::PEAK_IPL_AdaptiveHotpixelCorrector_GetHotpixels(PEAK_IPL_ADAPTIVE_HOTPIXEL_CORRECTOR_HANDLE adaptiveHotpixelCorrectorHandle, PEAK_IPL_POINT_2D * hotpixels, size_t * hotpixelsSize)
{
    auto& inst = instance();
    if(inst.m_PEAK_IPL_AdaptiveHotpixelCorrector_GetHotpixels)
    {
        return inst.m_PEAK_IPL_AdaptiveHotpixelCorrector_GetHotpixels(adaptiveHotpixelCorrectorHandle, hotpixels, hotpixelsSize);
    }
    else
    {
        throw std::runtime_error("Library not loaded!");
    }
}

inline PEAK_IPL_RETURN_CODE DynamicLoader::PEAK_IPL_AdaptiveHotpixelCorrector_Correct(PEAK_IPL_ADAPTIVE_HOTPIXEL_CORRECTOR_HANDLE adaptiveHotpixelCorrectorHandle, PEAK_IPL_IMAGE_HANDLE inputImageHandle, const PEAK_IPL_POINT_2D * hotpixels, size_t hotpixelsSize, PEAK_IPL_IMAGE_HANDLE * outputImageHandle)
{
    auto& inst = instance();
    if(inst.m_PEAK_IPL_AdaptiveHotpixelCorrector_Correct)
    {
        return inst.m_PEAK_IPL_AdaptiveHotpixelCorrector_Correct(adaptiveHotpixelCorrectorHandle, inputImageHandle, hotpixels, hotpixelsSize, outputImageHandle);
    }
    else
    {
        throw std::runtime_error("Library not loaded!");
    }
}

inline PEAK_IPL_RETURN_CODE DynamicLoader::PEAK_IPL_AdaptiveHotpixelCorrector_CorrectAdaptive(PEAK_IPL_ADAPTIVE_HOTPIXEL_CORRECTOR_HANDLE adaptiveHotpixelCorrectorHandle, PEAK_IPL_IMAGE_HANDLE inputImageHandle, PEAK_IPL_IMAGE_HANDLE * outputImageHandle)
{
    auto& inst = instance();
    if(inst.m_PEAK_IPL_AdaptiveHotpixelCorrector_CorrectAdaptive)
    {
        return inst.m_PEAK_IPL_AdaptiveHotpixelCorrector_CorrectAdaptive(adaptiveHotpixelCorrectorHandle, inputImageHandle, outputImageHandle);
    }
    else
    {
        throw std::runtime_error("Library not loaded!");
    }
}

inline PEAK_IPL_RETURN_CODE DynamicLoader::PEAK_IPL_AdaptiveHotpixelCorrector_Destruct(PEAK_IPL_ADAPTIVE_HOTPIXEL_CORRECTOR_HANDLE adaptiveHotpixelCorrectorHandle)
{
    auto& inst = instance();
    if(inst.m_PEAK_IPL_AdaptiveHotpixelCorrector_Destruct)
    {
        return inst.m_PEAK_IPL_AdaptiveHotpixelCorrector_Destruct(adaptiveHotpixelCorrectorHandle);
    }
    else
    {
        throw std::runtime_error("Library not loaded!");
    }
}

inline PEAK_IPL_RETURN_CODE DynamicLoader::PEAK_IPL_ColorCorrector_Construct(PEAK_IPL_COLOR_CORRECTOR_HANDLE * colorCorrectorHandle)
{
    auto& inst = instance();
    if(inst.m_PEAK_IPL_ColorCorrector_Construct)
    {
        return inst.m_PEAK_IPL_ColorCorrector_Construct(colorCorrectorHandle);
    }
    else
    {
        throw std::runtime_error("Library not loaded!");
    }
}

inline PEAK_IPL_RETURN_CODE DynamicLoader::PEAK_IPL_ColorCorrector_SetColorCorrectionFactors(PEAK_IPL_COLOR_CORRECTOR_HANDLE colorCorrectorHandle, float * colorCorrectorFactors)
{
    auto& inst = instance();
    if(inst.m_PEAK_IPL_ColorCorrector_SetColorCorrectionFactors)
    {
        return inst.m_PEAK_IPL_ColorCorrector_SetColorCorrectionFactors(colorCorrectorHandle, colorCorrectorFactors);
    }
    else
    {
        throw std::runtime_error("Library not loaded!");
    }
}

inline PEAK_IPL_RETURN_CODE DynamicLoader::PEAK_IPL_ColorCorrector_GetColorCorrectionFactors(PEAK_IPL_COLOR_CORRECTOR_HANDLE colorCorrectorHandle, float * colorCorrectorFactors, size_t * colorCorrectorFactorsSize)
{
    auto& inst = instance();
    if(inst.m_PEAK_IPL_ColorCorrector_GetColorCorrectionFactors)
    {
        return inst.m_PEAK_IPL_ColorCorrector_GetColorCorrectionFactors(colorCorrectorHandle, colorCorrectorFactors, colorCorrectorFactorsSize);
    }
    else
    {
        throw std::runtime_error("Library not loaded!");
    }
}

inline PEAK_IPL_RETURN_CODE DynamicLoader::PEAK_IPL_ColorCorrector_GetIsPixelFormatSupported(PEAK_IPL_COLOR_CORRECTOR_HANDLE colorCorrectorHandle, PEAK_IPL_PIXEL_FORMAT pixelFormat, PEAK_IPL_BOOL8 * isPixelFormatSupported)
{
    auto& inst = instance();
    if(inst.m_PEAK_IPL_ColorCorrector_GetIsPixelFormatSupported)
    {
        return inst.m_PEAK_IPL_ColorCorrector_GetIsPixelFormatSupported(colorCorrectorHandle, pixelFormat, isPixelFormatSupported);
    }
    else
    {
        throw std::runtime_error("Library not loaded!");
    }
}

inline PEAK_IPL_RETURN_CODE DynamicLoader::PEAK_IPL_ColorCorrector_Process(PEAK_IPL_COLOR_CORRECTOR_HANDLE colorCorrectorHandle, PEAK_IPL_IMAGE_HANDLE imageHandle, PEAK_IPL_IMAGE_HANDLE * outputImageHandle)
{
    auto& inst = instance();
    if(inst.m_PEAK_IPL_ColorCorrector_Process)
    {
        return inst.m_PEAK_IPL_ColorCorrector_Process(colorCorrectorHandle, imageHandle, outputImageHandle);
    }
    else
    {
        throw std::runtime_error("Library not loaded!");
    }
}

inline PEAK_IPL_RETURN_CODE DynamicLoader::PEAK_IPL_ColorCorrector_ProcessInPlace(PEAK_IPL_COLOR_CORRECTOR_HANDLE colorCorrectorHandle, PEAK_IPL_IMAGE_HANDLE imageHandle)
{
    auto& inst = instance();
    if(inst.m_PEAK_IPL_ColorCorrector_ProcessInPlace)
    {
        return inst.m_PEAK_IPL_ColorCorrector_ProcessInPlace(colorCorrectorHandle, imageHandle);
    }
    else
    {
        throw std::runtime_error("Library not loaded!");
    }
}

inline PEAK_IPL_RETURN_CODE DynamicLoader::PEAK_IPL_ColorCorrector_Destruct(PEAK_IPL_COLOR_CORRECTOR_HANDLE colorCorrectorHandle)
{
    auto& inst = instance();
    if(inst.m_PEAK_IPL_ColorCorrector_Destruct)
    {
        return inst.m_PEAK_IPL_ColorCorrector_Destruct(colorCorrectorHandle);
    }
    else
    {
        throw std::runtime_error("Library not loaded!");
    }
}

inline PEAK_IPL_RETURN_CODE DynamicLoader::PEAK_IPL_GammaCorrector_Construct(PEAK_IPL_GAMMA_CORRECTOR_HANDLE * gammaCorrectorHandle)
{
    auto& inst = instance();
    if(inst.m_PEAK_IPL_GammaCorrector_Construct)
    {
        return inst.m_PEAK_IPL_GammaCorrector_Construct(gammaCorrectorHandle);
    }
    else
    {
        throw std::runtime_error("Library not loaded!");
    }
}

inline PEAK_IPL_RETURN_CODE DynamicLoader::PEAK_IPL_GammaCorrector_SetGammaCorrectionValue(PEAK_IPL_GAMMA_CORRECTOR_HANDLE gammaCorrectorHandle, float * gammaValue)
{
    auto& inst = instance();
    if(inst.m_PEAK_IPL_GammaCorrector_SetGammaCorrectionValue)
    {
        return inst.m_PEAK_IPL_GammaCorrector_SetGammaCorrectionValue(gammaCorrectorHandle, gammaValue);
    }
    else
    {
        throw std::runtime_error("Library not loaded!");
    }
}

inline PEAK_IPL_RETURN_CODE DynamicLoader::PEAK_IPL_GammaCorrector_GetGammaCorrectionValue(PEAK_IPL_GAMMA_CORRECTOR_HANDLE gammaCorrectorHandle, float * gammaValue)
{
    auto& inst = instance();
    if(inst.m_PEAK_IPL_GammaCorrector_GetGammaCorrectionValue)
    {
        return inst.m_PEAK_IPL_GammaCorrector_GetGammaCorrectionValue(gammaCorrectorHandle, gammaValue);
    }
    else
    {
        throw std::runtime_error("Library not loaded!");
    }
}

inline PEAK_IPL_RETURN_CODE DynamicLoader::PEAK_IPL_GammaCorrector_GetGammaCorrectionMax(PEAK_IPL_GAMMA_CORRECTOR_HANDLE gammaCorrectorHandle, float * gammaMax)
{
    auto& inst = instance();
    if(inst.m_PEAK_IPL_GammaCorrector_GetGammaCorrectionMax)
    {
        return inst.m_PEAK_IPL_GammaCorrector_GetGammaCorrectionMax(gammaCorrectorHandle, gammaMax);
    }
    else
    {
        throw std::runtime_error("Library not loaded!");
    }
}

inline PEAK_IPL_RETURN_CODE DynamicLoader::PEAK_IPL_GammaCorrector_GetGammaCorrectionMin(PEAK_IPL_GAMMA_CORRECTOR_HANDLE gammaCorrectorHandle, float * gammaMin)
{
    auto& inst = instance();
    if(inst.m_PEAK_IPL_GammaCorrector_GetGammaCorrectionMin)
    {
        return inst.m_PEAK_IPL_GammaCorrector_GetGammaCorrectionMin(gammaCorrectorHandle, gammaMin);
    }
    else
    {
        throw std::runtime_error("Library not loaded!");
    }
}

inline PEAK_IPL_RETURN_CODE DynamicLoader::PEAK_IPL_GammaCorrector_GetIsPixelFormatSupported(PEAK_IPL_GAMMA_CORRECTOR_HANDLE gammaCorrectorHandle, PEAK_IPL_PIXEL_FORMAT pixelFormat, PEAK_IPL_BOOL8 * isPixelFormatSupported)
{
    auto& inst = instance();
    if(inst.m_PEAK_IPL_GammaCorrector_GetIsPixelFormatSupported)
    {
        return inst.m_PEAK_IPL_GammaCorrector_GetIsPixelFormatSupported(gammaCorrectorHandle, pixelFormat, isPixelFormatSupported);
    }
    else
    {
        throw std::runtime_error("Library not loaded!");
    }
}

inline PEAK_IPL_RETURN_CODE DynamicLoader::PEAK_IPL_GammaCorrector_Process(PEAK_IPL_GAMMA_CORRECTOR_HANDLE gammaCorrectorHandle, PEAK_IPL_IMAGE_HANDLE imageHandle, PEAK_IPL_IMAGE_HANDLE * outputImageHandle)
{
    auto& inst = instance();
    if(inst.m_PEAK_IPL_GammaCorrector_Process)
    {
        return inst.m_PEAK_IPL_GammaCorrector_Process(gammaCorrectorHandle, imageHandle, outputImageHandle);
    }
    else
    {
        throw std::runtime_error("Library not loaded!");
    }
}

inline PEAK_IPL_RETURN_CODE DynamicLoader::PEAK_IPL_GammaCorrector_ProcessInPlace(PEAK_IPL_GAMMA_CORRECTOR_HANDLE gammaCorrectorHandle, PEAK_IPL_IMAGE_HANDLE imageHandle)
{
    auto& inst = instance();
    if(inst.m_PEAK_IPL_GammaCorrector_ProcessInPlace)
    {
        return inst.m_PEAK_IPL_GammaCorrector_ProcessInPlace(gammaCorrectorHandle, imageHandle);
    }
    else
    {
        throw std::runtime_error("Library not loaded!");
    }
}

inline PEAK_IPL_RETURN_CODE DynamicLoader::PEAK_IPL_GammaCorrector_Destruct(PEAK_IPL_GAMMA_CORRECTOR_HANDLE gammaCorrectorHandle)
{
    auto& inst = instance();
    if(inst.m_PEAK_IPL_GammaCorrector_Destruct)
    {
        return inst.m_PEAK_IPL_GammaCorrector_Destruct(gammaCorrectorHandle);
    }
    else
    {
        throw std::runtime_error("Library not loaded!");
    }
}

inline PEAK_IPL_RETURN_CODE DynamicLoader::PEAK_IPL_Image_Construct(PEAK_IPL_PIXEL_FORMAT pixelFormat, size_t width, size_t height, PEAK_IPL_IMAGE_HANDLE * imageHandle)
{
    auto& inst = instance();
    if(inst.m_PEAK_IPL_Image_Construct)
    {
        return inst.m_PEAK_IPL_Image_Construct(pixelFormat, width, height, imageHandle);
    }
    else
    {
        throw std::runtime_error("Library not loaded!");
    }
}

inline PEAK_IPL_RETURN_CODE DynamicLoader::PEAK_IPL_Image_ConstructFromBuffer(PEAK_IPL_PIXEL_FORMAT pixelFormat, uint8_t * buffer, uint64_t bufferSize, size_t width, size_t height, PEAK_IPL_IMAGE_HANDLE * imageHandle)
{
    auto& inst = instance();
    if(inst.m_PEAK_IPL_Image_ConstructFromBuffer)
    {
        return inst.m_PEAK_IPL_Image_ConstructFromBuffer(pixelFormat, buffer, bufferSize, width, height, imageHandle);
    }
    else
    {
        throw std::runtime_error("Library not loaded!");
    }
}

inline PEAK_IPL_RETURN_CODE DynamicLoader::PEAK_IPL_Image_GetWidth(PEAK_IPL_IMAGE_HANDLE imageHandle, size_t * width)
{
    auto& inst = instance();
    if(inst.m_PEAK_IPL_Image_GetWidth)
    {
        return inst.m_PEAK_IPL_Image_GetWidth(imageHandle, width);
    }
    else
    {
        throw std::runtime_error("Library not loaded!");
    }
}

inline PEAK_IPL_RETURN_CODE DynamicLoader::PEAK_IPL_Image_GetHeight(PEAK_IPL_IMAGE_HANDLE imageHandle, size_t * height)
{
    auto& inst = instance();
    if(inst.m_PEAK_IPL_Image_GetHeight)
    {
        return inst.m_PEAK_IPL_Image_GetHeight(imageHandle, height);
    }
    else
    {
        throw std::runtime_error("Library not loaded!");
    }
}

inline PEAK_IPL_RETURN_CODE DynamicLoader::PEAK_IPL_Image_GetPixelPointer(PEAK_IPL_IMAGE_HANDLE imageHandle, size_t xPos, size_t yPos, uint8_t * * pixelPointer)
{
    auto& inst = instance();
    if(inst.m_PEAK_IPL_Image_GetPixelPointer)
    {
        return inst.m_PEAK_IPL_Image_GetPixelPointer(imageHandle, xPos, yPos, pixelPointer);
    }
    else
    {
        throw std::runtime_error("Library not loaded!");
    }
}

inline PEAK_IPL_RETURN_CODE DynamicLoader::PEAK_IPL_Image_GetByteCount(PEAK_IPL_IMAGE_HANDLE imageHandle, size_t * byteCount)
{
    auto& inst = instance();
    if(inst.m_PEAK_IPL_Image_GetByteCount)
    {
        return inst.m_PEAK_IPL_Image_GetByteCount(imageHandle, byteCount);
    }
    else
    {
        throw std::runtime_error("Library not loaded!");
    }
}

inline PEAK_IPL_RETURN_CODE DynamicLoader::PEAK_IPL_Image_GetPixelFormat(PEAK_IPL_IMAGE_HANDLE imageHandle, PEAK_IPL_PIXEL_FORMAT * pixelFormat)
{
    auto& inst = instance();
    if(inst.m_PEAK_IPL_Image_GetPixelFormat)
    {
        return inst.m_PEAK_IPL_Image_GetPixelFormat(imageHandle, pixelFormat);
    }
    else
    {
        throw std::runtime_error("Library not loaded!");
    }
}

inline PEAK_IPL_RETURN_CODE DynamicLoader::PEAK_IPL_Image_GetData(PEAK_IPL_IMAGE_HANDLE imageHandle, uint8_t * * data)
{
    auto& inst = instance();
    if(inst.m_PEAK_IPL_Image_GetData)
    {
        return inst.m_PEAK_IPL_Image_GetData(imageHandle, data);
    }
    else
    {
        throw std::runtime_error("Library not loaded!");
    }
}

inline PEAK_IPL_RETURN_CODE DynamicLoader::PEAK_IPL_Image_Clone(PEAK_IPL_IMAGE_HANDLE imageHandle, PEAK_IPL_IMAGE_HANDLE * outputImageHandle)
{
    auto& inst = instance();
    if(inst.m_PEAK_IPL_Image_Clone)
    {
        return inst.m_PEAK_IPL_Image_Clone(imageHandle, outputImageHandle);
    }
    else
    {
        throw std::runtime_error("Library not loaded!");
    }
}

inline PEAK_IPL_RETURN_CODE DynamicLoader::PEAK_IPL_Image_ConvertTo(PEAK_IPL_IMAGE_HANDLE imageHandle, PEAK_IPL_PIXEL_FORMAT outputPixelFormat, PEAK_IPL_CONVERSION_MODE conversionMode, PEAK_IPL_IMAGE_HANDLE * outputImageHandle)
{
    auto& inst = instance();
    if(inst.m_PEAK_IPL_Image_ConvertTo)
    {
        return inst.m_PEAK_IPL_Image_ConvertTo(imageHandle, outputPixelFormat, conversionMode, outputImageHandle);
    }
    else
    {
        throw std::runtime_error("Library not loaded!");
    }
}

inline PEAK_IPL_RETURN_CODE DynamicLoader::PEAK_IPL_Image_ConvertToBuffer(PEAK_IPL_IMAGE_HANDLE imageHandle, PEAK_IPL_PIXEL_FORMAT outputPixelFormat, uint8_t * outputImageBuffer, size_t outputImageBufferSize, PEAK_IPL_CONVERSION_MODE conversionMode, PEAK_IPL_IMAGE_HANDLE * outputImageHandle)
{
    auto& inst = instance();
    if(inst.m_PEAK_IPL_Image_ConvertToBuffer)
    {
        return inst.m_PEAK_IPL_Image_ConvertToBuffer(imageHandle, outputPixelFormat, outputImageBuffer, outputImageBufferSize, conversionMode, outputImageHandle);
    }
    else
    {
        throw std::runtime_error("Library not loaded!");
    }
}

inline PEAK_IPL_RETURN_CODE DynamicLoader::PEAK_IPL_Image_Destruct(PEAK_IPL_IMAGE_HANDLE imageHandle)
{
    auto& inst = instance();
    if(inst.m_PEAK_IPL_Image_Destruct)
    {
        return inst.m_PEAK_IPL_Image_Destruct(imageHandle);
    }
    else
    {
        throw std::runtime_error("Library not loaded!");
    }
}

inline PEAK_IPL_RETURN_CODE DynamicLoader::PEAK_IPL_ImageTransformer_Construct(PEAK_IPL_IMAGE_TRANSFORMER_HANDLE * imageTransformerHandle)
{
    auto& inst = instance();
    if(inst.m_PEAK_IPL_ImageTransformer_Construct)
    {
        return inst.m_PEAK_IPL_ImageTransformer_Construct(imageTransformerHandle);
    }
    else
    {
        throw std::runtime_error("Library not loaded!");
    }
}

inline PEAK_IPL_RETURN_CODE DynamicLoader::PEAK_IPL_ImageTransformer_MirrorUpDown(PEAK_IPL_IMAGE_TRANSFORMER_HANDLE imageTransformerHandle, PEAK_IPL_IMAGE_HANDLE imageHandle, PEAK_IPL_IMAGE_HANDLE * outputImageHandle)
{
    auto& inst = instance();
    if(inst.m_PEAK_IPL_ImageTransformer_MirrorUpDown)
    {
        return inst.m_PEAK_IPL_ImageTransformer_MirrorUpDown(imageTransformerHandle, imageHandle, outputImageHandle);
    }
    else
    {
        throw std::runtime_error("Library not loaded!");
    }
}

inline PEAK_IPL_RETURN_CODE DynamicLoader::PEAK_IPL_ImageTransformer_MirrorLeftRight(PEAK_IPL_IMAGE_TRANSFORMER_HANDLE imageTransformerHandle, PEAK_IPL_IMAGE_HANDLE imageHandle, PEAK_IPL_IMAGE_HANDLE * outputImageHandle)
{
    auto& inst = instance();
    if(inst.m_PEAK_IPL_ImageTransformer_MirrorLeftRight)
    {
        return inst.m_PEAK_IPL_ImageTransformer_MirrorLeftRight(imageTransformerHandle, imageHandle, outputImageHandle);
    }
    else
    {
        throw std::runtime_error("Library not loaded!");
    }
}

inline PEAK_IPL_RETURN_CODE DynamicLoader::PEAK_IPL_ImageTransformer_MirrorUpDownLeftRight(PEAK_IPL_IMAGE_TRANSFORMER_HANDLE imageTransformerHandle, PEAK_IPL_IMAGE_HANDLE imageHandle, PEAK_IPL_IMAGE_HANDLE * outputImageHandle)
{
    auto& inst = instance();
    if(inst.m_PEAK_IPL_ImageTransformer_MirrorUpDownLeftRight)
    {
        return inst.m_PEAK_IPL_ImageTransformer_MirrorUpDownLeftRight(imageTransformerHandle, imageHandle, outputImageHandle);
    }
    else
    {
        throw std::runtime_error("Library not loaded!");
    }
}

inline PEAK_IPL_RETURN_CODE DynamicLoader::PEAK_IPL_ImageTransformer_MirrorUpDownInPlace(PEAK_IPL_IMAGE_TRANSFORMER_HANDLE imageTransformerHandle, PEAK_IPL_IMAGE_HANDLE imageHandle)
{
    auto& inst = instance();
    if(inst.m_PEAK_IPL_ImageTransformer_MirrorUpDownInPlace)
    {
        return inst.m_PEAK_IPL_ImageTransformer_MirrorUpDownInPlace(imageTransformerHandle, imageHandle);
    }
    else
    {
        throw std::runtime_error("Library not loaded!");
    }
}

inline PEAK_IPL_RETURN_CODE DynamicLoader::PEAK_IPL_ImageTransformer_MirrorLeftRightInPlace(PEAK_IPL_IMAGE_TRANSFORMER_HANDLE imageTransformerHandle, PEAK_IPL_IMAGE_HANDLE imageHandle)
{
    auto& inst = instance();
    if(inst.m_PEAK_IPL_ImageTransformer_MirrorLeftRightInPlace)
    {
        return inst.m_PEAK_IPL_ImageTransformer_MirrorLeftRightInPlace(imageTransformerHandle, imageHandle);
    }
    else
    {
        throw std::runtime_error("Library not loaded!");
    }
}

inline PEAK_IPL_RETURN_CODE DynamicLoader::PEAK_IPL_ImageTransformer_MirrorUpDownLeftRightInPlace(PEAK_IPL_IMAGE_TRANSFORMER_HANDLE imageTransformerHandle, PEAK_IPL_IMAGE_HANDLE imageHandle)
{
    auto& inst = instance();
    if(inst.m_PEAK_IPL_ImageTransformer_MirrorUpDownLeftRightInPlace)
    {
        return inst.m_PEAK_IPL_ImageTransformer_MirrorUpDownLeftRightInPlace(imageTransformerHandle, imageHandle);
    }
    else
    {
        throw std::runtime_error("Library not loaded!");
    }
}

inline PEAK_IPL_RETURN_CODE DynamicLoader::PEAK_IPL_ImageTransformer_Rotate(PEAK_IPL_IMAGE_TRANSFORMER_HANDLE imageTransformerHandle, PEAK_IPL_IMAGE_HANDLE imageHandle, PEAK_IPL_IMAGE_HANDLE * outputImageHandle, PEAK_IPL_IMAGE_TRANSFORMER_ROTATION_ANGLE_t rotationAngle)
{
    auto& inst = instance();
    if(inst.m_PEAK_IPL_ImageTransformer_Rotate)
    {
        return inst.m_PEAK_IPL_ImageTransformer_Rotate(imageTransformerHandle, imageHandle, outputImageHandle, rotationAngle);
    }
    else
    {
        throw std::runtime_error("Library not loaded!");
    }
}

inline PEAK_IPL_RETURN_CODE DynamicLoader::PEAK_IPL_ImageTransformer_RotateInPlace(PEAK_IPL_IMAGE_TRANSFORMER_HANDLE imageTransformerHandle, PEAK_IPL_IMAGE_HANDLE imageHandle, PEAK_IPL_IMAGE_TRANSFORMER_ROTATION_ANGLE_t rotationAngle)
{
    auto& inst = instance();
    if(inst.m_PEAK_IPL_ImageTransformer_RotateInPlace)
    {
        return inst.m_PEAK_IPL_ImageTransformer_RotateInPlace(imageTransformerHandle, imageHandle, rotationAngle);
    }
    else
    {
        throw std::runtime_error("Library not loaded!");
    }
}

inline PEAK_IPL_RETURN_CODE DynamicLoader::PEAK_IPL_ImageTransformer_Destruct(PEAK_IPL_IMAGE_TRANSFORMER_HANDLE imageTransformerHandle)
{
    auto& inst = instance();
    if(inst.m_PEAK_IPL_ImageTransformer_Destruct)
    {
        return inst.m_PEAK_IPL_ImageTransformer_Destruct(imageTransformerHandle);
    }
    else
    {
        throw std::runtime_error("Library not loaded!");
    }
}

inline PEAK_IPL_RETURN_CODE DynamicLoader::PEAK_IPL_Histogram_Construct(PEAK_IPL_IMAGE_HANDLE imageHandle, PEAK_IPL_HISTOGRAM_HANDLE * histogramHandle)
{
    auto& inst = instance();
    if(inst.m_PEAK_IPL_Histogram_Construct)
    {
        return inst.m_PEAK_IPL_Histogram_Construct(imageHandle, histogramHandle);
    }
    else
    {
        throw std::runtime_error("Library not loaded!");
    }
}

inline PEAK_IPL_RETURN_CODE DynamicLoader::PEAK_IPL_Histogram_GetPixelFormat(PEAK_IPL_HISTOGRAM_HANDLE histogramHandle, PEAK_IPL_PIXEL_FORMAT * pixelFormat)
{
    auto& inst = instance();
    if(inst.m_PEAK_IPL_Histogram_GetPixelFormat)
    {
        return inst.m_PEAK_IPL_Histogram_GetPixelFormat(histogramHandle, pixelFormat);
    }
    else
    {
        throw std::runtime_error("Library not loaded!");
    }
}

inline PEAK_IPL_RETURN_CODE DynamicLoader::PEAK_IPL_Histogram_GetNumChannels(PEAK_IPL_HISTOGRAM_HANDLE histogramHandle, size_t * numChannels)
{
    auto& inst = instance();
    if(inst.m_PEAK_IPL_Histogram_GetNumChannels)
    {
        return inst.m_PEAK_IPL_Histogram_GetNumChannels(histogramHandle, numChannels);
    }
    else
    {
        throw std::runtime_error("Library not loaded!");
    }
}

inline PEAK_IPL_RETURN_CODE DynamicLoader::PEAK_IPL_Histogram_GetPixelSumForChannel(PEAK_IPL_HISTOGRAM_HANDLE histogramHandle, size_t channelIndex, uint64_t * pixelSum)
{
    auto& inst = instance();
    if(inst.m_PEAK_IPL_Histogram_GetPixelSumForChannel)
    {
        return inst.m_PEAK_IPL_Histogram_GetPixelSumForChannel(histogramHandle, channelIndex, pixelSum);
    }
    else
    {
        throw std::runtime_error("Library not loaded!");
    }
}

inline PEAK_IPL_RETURN_CODE DynamicLoader::PEAK_IPL_Histogram_GetPixelCountForChannel(PEAK_IPL_HISTOGRAM_HANDLE histogramHandle, size_t channelIndex, uint64_t * pixelCount)
{
    auto& inst = instance();
    if(inst.m_PEAK_IPL_Histogram_GetPixelCountForChannel)
    {
        return inst.m_PEAK_IPL_Histogram_GetPixelCountForChannel(histogramHandle, channelIndex, pixelCount);
    }
    else
    {
        throw std::runtime_error("Library not loaded!");
    }
}

inline PEAK_IPL_RETURN_CODE DynamicLoader::PEAK_IPL_Histogram_GetBinsForChannel(PEAK_IPL_HISTOGRAM_HANDLE histogramHandle, size_t channelIndex, uint64_t * bins, size_t * binsSize)
{
    auto& inst = instance();
    if(inst.m_PEAK_IPL_Histogram_GetBinsForChannel)
    {
        return inst.m_PEAK_IPL_Histogram_GetBinsForChannel(histogramHandle, channelIndex, bins, binsSize);
    }
    else
    {
        throw std::runtime_error("Library not loaded!");
    }
}

inline PEAK_IPL_RETURN_CODE DynamicLoader::PEAK_IPL_Histogram_Destruct(PEAK_IPL_HISTOGRAM_HANDLE histogramHandle)
{
    auto& inst = instance();
    if(inst.m_PEAK_IPL_Histogram_Destruct)
    {
        return inst.m_PEAK_IPL_Histogram_Destruct(histogramHandle);
    }
    else
    {
        throw std::runtime_error("Library not loaded!");
    }
}

inline PEAK_IPL_RETURN_CODE DynamicLoader::PEAK_IPL_PixelLine_Construct(PEAK_IPL_IMAGE_HANDLE imageHandle, PEAK_IPL_ORIENTATION orientation, size_t offset, PEAK_IPL_PIXEL_LINE_HANDLE * pixelLineHandle)
{
    auto& inst = instance();
    if(inst.m_PEAK_IPL_PixelLine_Construct)
    {
        return inst.m_PEAK_IPL_PixelLine_Construct(imageHandle, orientation, offset, pixelLineHandle);
    }
    else
    {
        throw std::runtime_error("Library not loaded!");
    }
}

inline PEAK_IPL_RETURN_CODE DynamicLoader::PEAK_IPL_PixelLine_GetPixelFormat(PEAK_IPL_PIXEL_LINE_HANDLE pixelLineHandle, PEAK_IPL_PIXEL_FORMAT * pixelFormat)
{
    auto& inst = instance();
    if(inst.m_PEAK_IPL_PixelLine_GetPixelFormat)
    {
        return inst.m_PEAK_IPL_PixelLine_GetPixelFormat(pixelLineHandle, pixelFormat);
    }
    else
    {
        throw std::runtime_error("Library not loaded!");
    }
}

inline PEAK_IPL_RETURN_CODE DynamicLoader::PEAK_IPL_PixelLine_GetOrientation(PEAK_IPL_PIXEL_LINE_HANDLE pixelLineHandle, PEAK_IPL_ORIENTATION * orientation)
{
    auto& inst = instance();
    if(inst.m_PEAK_IPL_PixelLine_GetOrientation)
    {
        return inst.m_PEAK_IPL_PixelLine_GetOrientation(pixelLineHandle, orientation);
    }
    else
    {
        throw std::runtime_error("Library not loaded!");
    }
}

inline PEAK_IPL_RETURN_CODE DynamicLoader::PEAK_IPL_PixelLine_GetOffset(PEAK_IPL_PIXEL_LINE_HANDLE pixelLineHandle, size_t * offset)
{
    auto& inst = instance();
    if(inst.m_PEAK_IPL_PixelLine_GetOffset)
    {
        return inst.m_PEAK_IPL_PixelLine_GetOffset(pixelLineHandle, offset);
    }
    else
    {
        throw std::runtime_error("Library not loaded!");
    }
}

inline PEAK_IPL_RETURN_CODE DynamicLoader::PEAK_IPL_PixelLine_GetNumChannels(PEAK_IPL_PIXEL_LINE_HANDLE pixelLineHandle, size_t * numChannels)
{
    auto& inst = instance();
    if(inst.m_PEAK_IPL_PixelLine_GetNumChannels)
    {
        return inst.m_PEAK_IPL_PixelLine_GetNumChannels(pixelLineHandle, numChannels);
    }
    else
    {
        throw std::runtime_error("Library not loaded!");
    }
}

inline PEAK_IPL_RETURN_CODE DynamicLoader::PEAK_IPL_PixelLine_GetValuesForChannel(PEAK_IPL_PIXEL_LINE_HANDLE pixelLineHandle, size_t channelIndex, uint32_t * values, size_t * valuesSize)
{
    auto& inst = instance();
    if(inst.m_PEAK_IPL_PixelLine_GetValuesForChannel)
    {
        return inst.m_PEAK_IPL_PixelLine_GetValuesForChannel(pixelLineHandle, channelIndex, values, valuesSize);
    }
    else
    {
        throw std::runtime_error("Library not loaded!");
    }
}

inline PEAK_IPL_RETURN_CODE DynamicLoader::PEAK_IPL_PixelLine_Destruct(PEAK_IPL_PIXEL_LINE_HANDLE pixelLineHandle)
{
    auto& inst = instance();
    if(inst.m_PEAK_IPL_PixelLine_Destruct)
    {
        return inst.m_PEAK_IPL_PixelLine_Destruct(pixelLineHandle);
    }
    else
    {
        throw std::runtime_error("Library not loaded!");
    }
}

inline PEAK_IPL_RETURN_CODE DynamicLoader::PEAK_IPL_PixelFormat_GetNumChannels(PEAK_IPL_PIXEL_FORMAT pixelFormat, size_t * numChannels)
{
    auto& inst = instance();
    if(inst.m_PEAK_IPL_PixelFormat_GetNumChannels)
    {
        return inst.m_PEAK_IPL_PixelFormat_GetNumChannels(pixelFormat, numChannels);
    }
    else
    {
        throw std::runtime_error("Library not loaded!");
    }
}

inline PEAK_IPL_RETURN_CODE DynamicLoader::PEAK_IPL_PixelFormat_GetNumSignificantBitsPerChannel(PEAK_IPL_PIXEL_FORMAT pixelFormat, size_t * numSignificantBits)
{
    auto& inst = instance();
    if(inst.m_PEAK_IPL_PixelFormat_GetNumSignificantBitsPerChannel)
    {
        return inst.m_PEAK_IPL_PixelFormat_GetNumSignificantBitsPerChannel(pixelFormat, numSignificantBits);
    }
    else
    {
        throw std::runtime_error("Library not loaded!");
    }
}

inline PEAK_IPL_RETURN_CODE DynamicLoader::PEAK_IPL_PixelFormat_GetNumStorageBitsPerChannel(PEAK_IPL_PIXEL_FORMAT pixelFormat, size_t * numStorageBits)
{
    auto& inst = instance();
    if(inst.m_PEAK_IPL_PixelFormat_GetNumStorageBitsPerChannel)
    {
        return inst.m_PEAK_IPL_PixelFormat_GetNumStorageBitsPerChannel(pixelFormat, numStorageBits);
    }
    else
    {
        throw std::runtime_error("Library not loaded!");
    }
}

inline PEAK_IPL_RETURN_CODE DynamicLoader::PEAK_IPL_PixelFormat_GetMaximumValuePerChannel(PEAK_IPL_PIXEL_FORMAT pixelFormat, uint32_t * channelMaximumValue)
{
    auto& inst = instance();
    if(inst.m_PEAK_IPL_PixelFormat_GetMaximumValuePerChannel)
    {
        return inst.m_PEAK_IPL_PixelFormat_GetMaximumValuePerChannel(pixelFormat, channelMaximumValue);
    }
    else
    {
        throw std::runtime_error("Library not loaded!");
    }
}

inline PEAK_IPL_RETURN_CODE DynamicLoader::PEAK_IPL_PixelFormat_GetNumSignificantBitsPerPixel(PEAK_IPL_PIXEL_FORMAT pixelFormat, size_t * numSignificantBits)
{
    auto& inst = instance();
    if(inst.m_PEAK_IPL_PixelFormat_GetNumSignificantBitsPerPixel)
    {
        return inst.m_PEAK_IPL_PixelFormat_GetNumSignificantBitsPerPixel(pixelFormat, numSignificantBits);
    }
    else
    {
        throw std::runtime_error("Library not loaded!");
    }
}

inline PEAK_IPL_RETURN_CODE DynamicLoader::PEAK_IPL_PixelFormat_GetNumStorageBitsPerPixel(PEAK_IPL_PIXEL_FORMAT pixelFormat, size_t * numStorageBits)
{
    auto& inst = instance();
    if(inst.m_PEAK_IPL_PixelFormat_GetNumStorageBitsPerPixel)
    {
        return inst.m_PEAK_IPL_PixelFormat_GetNumStorageBitsPerPixel(pixelFormat, numStorageBits);
    }
    else
    {
        throw std::runtime_error("Library not loaded!");
    }
}

inline PEAK_IPL_RETURN_CODE DynamicLoader::PEAK_IPL_PixelFormat_GetEndianness(PEAK_IPL_PIXEL_FORMAT pixelFormat, PEAK_IPL_ENDIANNESS * endianness)
{
    auto& inst = instance();
    if(inst.m_PEAK_IPL_PixelFormat_GetEndianness)
    {
        return inst.m_PEAK_IPL_PixelFormat_GetEndianness(pixelFormat, endianness);
    }
    else
    {
        throw std::runtime_error("Library not loaded!");
    }
}

inline PEAK_IPL_RETURN_CODE DynamicLoader::PEAK_IPL_PixelFormat_CalculateStorageSizeOfPixels(PEAK_IPL_PIXEL_FORMAT pixelFormat, uint64_t numPixels, uint64_t * size)
{
    auto& inst = instance();
    if(inst.m_PEAK_IPL_PixelFormat_CalculateStorageSizeOfPixels)
    {
        return inst.m_PEAK_IPL_PixelFormat_CalculateStorageSizeOfPixels(pixelFormat, numPixels, size);
    }
    else
    {
        throw std::runtime_error("Library not loaded!");
    }
}

inline PEAK_IPL_RETURN_CODE DynamicLoader::PEAK_IPL_ImageWriter_WriteAsRAW(PEAK_IPL_IMAGE_HANDLE inputImageHandle, const char * filePath, size_t filePathSize)
{
    auto& inst = instance();
    if(inst.m_PEAK_IPL_ImageWriter_WriteAsRAW)
    {
        return inst.m_PEAK_IPL_ImageWriter_WriteAsRAW(inputImageHandle, filePath, filePathSize);
    }
    else
    {
        throw std::runtime_error("Library not loaded!");
    }
}

inline PEAK_IPL_RETURN_CODE DynamicLoader::PEAK_IPL_ImageWriter_WriteAsBMP(PEAK_IPL_IMAGE_HANDLE inputImageHandle, const char * filePath, size_t filePathSize)
{
    auto& inst = instance();
    if(inst.m_PEAK_IPL_ImageWriter_WriteAsBMP)
    {
        return inst.m_PEAK_IPL_ImageWriter_WriteAsBMP(inputImageHandle, filePath, filePathSize);
    }
    else
    {
        throw std::runtime_error("Library not loaded!");
    }
}

inline PEAK_IPL_RETURN_CODE DynamicLoader::PEAK_IPL_ImageWriter_WriteAsPNG(PEAK_IPL_IMAGE_HANDLE inputImageHandle, uint32_t quality, const char * filePath, size_t filePathSize)
{
    auto& inst = instance();
    if(inst.m_PEAK_IPL_ImageWriter_WriteAsPNG)
    {
        return inst.m_PEAK_IPL_ImageWriter_WriteAsPNG(inputImageHandle, quality, filePath, filePathSize);
    }
    else
    {
        throw std::runtime_error("Library not loaded!");
    }
}

inline PEAK_IPL_RETURN_CODE DynamicLoader::PEAK_IPL_ImageWriter_WriteAsJPG(PEAK_IPL_IMAGE_HANDLE inputImageHandle, uint32_t quality, const char * filePath, size_t filePathSize)
{
    auto& inst = instance();
    if(inst.m_PEAK_IPL_ImageWriter_WriteAsJPG)
    {
        return inst.m_PEAK_IPL_ImageWriter_WriteAsJPG(inputImageHandle, quality, filePath, filePathSize);
    }
    else
    {
        throw std::runtime_error("Library not loaded!");
    }
}

inline PEAK_IPL_RETURN_CODE DynamicLoader::PEAK_IPL_ImageWriter_Write(PEAK_IPL_IMAGE_HANDLE inputImageHandle, const char * filePath, size_t filePathSize)
{
    auto& inst = instance();
    if(inst.m_PEAK_IPL_ImageWriter_Write)
    {
        return inst.m_PEAK_IPL_ImageWriter_Write(inputImageHandle, filePath, filePathSize);
    }
    else
    {
        throw std::runtime_error("Library not loaded!");
    }
}

inline PEAK_IPL_RETURN_CODE DynamicLoader::PEAK_IPL_ImageReader_Read(const char * filePath, size_t filePathSize, PEAK_IPL_IMAGE_HANDLE * outputImageHandle)
{
    auto& inst = instance();
    if(inst.m_PEAK_IPL_ImageReader_Read)
    {
        return inst.m_PEAK_IPL_ImageReader_Read(filePath, filePathSize, outputImageHandle);
    }
    else
    {
        throw std::runtime_error("Library not loaded!");
    }
}

inline PEAK_IPL_RETURN_CODE DynamicLoader::PEAK_IPL_ImageReaderRead_ReadAsPixelFormat(const char * filePath, size_t filePathSize, PEAK_IPL_PIXEL_FORMAT pixelFormat, PEAK_IPL_IMAGE_HANDLE * outputImageHandle)
{
    auto& inst = instance();
    if(inst.m_PEAK_IPL_ImageReaderRead_ReadAsPixelFormat)
    {
        return inst.m_PEAK_IPL_ImageReaderRead_ReadAsPixelFormat(filePath, filePathSize, pixelFormat, outputImageHandle);
    }
    else
    {
        throw std::runtime_error("Library not loaded!");
    }
}

} /* namespace dynamic */
} /* namespace ipl */
} /* namespace peak */

