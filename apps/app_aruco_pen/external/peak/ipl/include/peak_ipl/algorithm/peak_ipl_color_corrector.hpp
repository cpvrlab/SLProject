/*!
 * \file    peak_ipl_color_corrector.hpp
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
#include <peak_ipl/types/peak_ipl_image.hpp>
#include <peak_ipl/types/peak_ipl_pixel_format.hpp>
#include <peak_ipl/types/peak_ipl_simple_types.hpp>

#include <type_traits>
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
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
 * \brief The Factors of the Color Correction Matrix.
 */
struct ColorCorrectionFactors
{
    ColorCorrectionFactors(float facRR, float facGR, float facBR, float facRG, float facGG, float facBG, float facRB,
        float facGB, float facBB)
        : factorRR(facRR)
        , factorGR(facGR)
        , factorBR(facBR)
        , factorRG(facRG)
        , factorGG(facGG)
        , factorBG(facBG)
        , factorRB(facRB)
        , factorGB(facGB)
        , factorBB(facBB)
    {}

    ColorCorrectionFactors() = default;
    ~ColorCorrectionFactors() = default;
    ColorCorrectionFactors(ColorCorrectionFactors&& o) = default;
    ColorCorrectionFactors& operator=(ColorCorrectionFactors&& o) = default;
    ColorCorrectionFactors& operator=(const ColorCorrectionFactors& o) = default;
    ColorCorrectionFactors(const ColorCorrectionFactors& o) = default;

    bool operator==(const ColorCorrectionFactors& other) const
    {
        return (isEqual(factorRR, other.factorRR) && isEqual(factorGR, other.factorGR)
            && isEqual(factorBR, other.factorBR) && isEqual(factorRG, other.factorRG)
            && isEqual(factorGG, other.factorGG) && isEqual(factorBG, other.factorBG)
            && isEqual(factorRB, other.factorRB) && isEqual(factorGB, other.factorGB)
            && isEqual(factorBB, other.factorBB));
    }

    float factorRR;
    float factorGR;
    float factorBR;
    float factorRG;
    float factorGG;
    float factorBG;
    float factorRB;
    float factorGB;
    float factorBB;

private:
    template <class T>
    static typename std::enable_if<!std::numeric_limits<T>::is_integer, bool>::type isEqual(T x, T y)
    {
        return std::fabs(x - y) <= std::numeric_limits<T>::epsilon();
    }
};

/*!
 * \brief Applies a 3x3 color correction matrix to the data.
 */
class ColorCorrector final
{
public:
    ColorCorrector();
    ~ColorCorrector();
    ColorCorrector(const ColorCorrector& other) = delete;
    ColorCorrector& operator=(const ColorCorrector& other) = delete;
    ColorCorrector(ColorCorrector&& other);
    ColorCorrector& operator=(ColorCorrector&& other);

    /*!
     * \brief Sets the values of the color correction matrix.
     *
     * The matrix is row-wise sorted:
     * <table>
     *  <tr><td>factorRR</td><td>factorGR</td><td>factorBR</td></tr>
     *  <tr><td>factorRG</td><td>factorGG</td><td>factorBG</td></tr>
     *  <tr><td>factorRB</td><td>factorGB</td><td>factorBB</td></tr>
     * </table>
     *
     * \param[in] colorCorrectorFactors The factors of the color correction matrix.
     *
     * \since 1.0
     */
    void SetColorCorrectionFactors(peak::ipl::ColorCorrectionFactors colorCorrectorFactors);

    /*!
     * \brief Returns the factors of the color correction matrix.
     *
     * \returns colorCorrectorFactors The factors of the color correction matrix
     *
     * \since 1.0
     */
    peak::ipl::ColorCorrectionFactors ColorCorrectionFactors();

    /*!
     * \brief Returns whether the color corrector supports the given pixel format.
     *
     * \returns Flag whether the given pixel format is supported.
     *
     * \param[in] pixelFormatName The pixel format of interest.
     *
     * \since 1.0
     */
    bool IsPixelFormatSupported(PixelFormatName pixelFormatName) const;

    /*!
     * \brief Corrects the colors of the given image by applying a 3x3 color correction matrix to the data
     *        in place i.e. it will change the input image.
     *
     * \param[in] image Image to process.
     *
     * \throws ImageFormatNotSupportedException image has unsupported pixel format (e.g. packed pixel format)
     *
     * \since 1.0
     * \since 1.2 Will throw ImageFormatNotSupportedException if image has packed pixel format
     */
    void ProcessInPlace(Image& image) const;

    /*! \brief Corrects the colors of the given image by applying a 3x3 color correction matrix to the data.
     *
     * \param[in] inputImage Image to process.
     *
     * \returns A new created image containing the color corrected pixels
     *
     * \throws ImageFormatNotSupportedException image has unsupported pixel format (e.g. packed pixel format)
     *
     * \since 1.0
     * \since 1.2 Will throw ImageFormatNotSupportedException if inputImage has packed pixel format
     */
    Image Process(const Image& inputImage) const;

private:
    PEAK_IPL_COLOR_CORRECTOR_HANDLE m_backendHandle{};
};

inline ColorCorrector::ColorCorrector()
{
    ExecuteAndMapReturnCodes([&] { return PEAK_IPL_C_ABI_PREFIX PEAK_IPL_ColorCorrector_Construct(&m_backendHandle); });
}

inline ColorCorrector::~ColorCorrector()
{
    (void)PEAK_IPL_C_ABI_PREFIX PEAK_IPL_ColorCorrector_Destruct(m_backendHandle);
}

inline ColorCorrector::ColorCorrector(ColorCorrector&& other)
{
    *this = std::move(other);
}

inline ColorCorrector& ColorCorrector::operator=(ColorCorrector&& other)
{
    if (this != &other)
    {
        (void)PEAK_IPL_C_ABI_PREFIX PEAK_IPL_ColorCorrector_Destruct(m_backendHandle);
        m_backendHandle = other.m_backendHandle;
        other.m_backendHandle = nullptr;
    }

    return *this;
}

inline void ColorCorrector::SetColorCorrectionFactors(peak::ipl::ColorCorrectionFactors colorCorrectorFactors)
{
    ExecuteAndMapReturnCodes([&] {
        return PEAK_IPL_C_ABI_PREFIX PEAK_IPL_ColorCorrector_SetColorCorrectionFactors(
            m_backendHandle, reinterpret_cast<float*>(&colorCorrectorFactors));
    });
}

inline peak::ipl::ColorCorrectionFactors ColorCorrector::ColorCorrectionFactors()
{
    size_t colorCorrectorFactorsSize = 0;

    ExecuteAndMapReturnCodes([&] {
        return PEAK_IPL_C_ABI_PREFIX PEAK_IPL_ColorCorrector_GetColorCorrectionFactors(
            m_backendHandle, nullptr, &colorCorrectorFactorsSize);
    });

    peak::ipl::ColorCorrectionFactors colorCorrectorFactors;

    ExecuteAndMapReturnCodes([&] {
        return PEAK_IPL_C_ABI_PREFIX PEAK_IPL_ColorCorrector_GetColorCorrectionFactors(
            m_backendHandle, reinterpret_cast<float*>(&colorCorrectorFactors), &colorCorrectorFactorsSize);
    });

    return colorCorrectorFactors;
}

inline bool ColorCorrector::IsPixelFormatSupported(PixelFormatName pixelFormatName) const
{
    PEAK_IPL_BOOL8 isPixelFormatSupported = 0;

    ExecuteAndMapReturnCodes([&] {
        return PEAK_IPL_C_ABI_PREFIX PEAK_IPL_ColorCorrector_GetIsPixelFormatSupported(m_backendHandle,
            static_cast<PEAK_IPL_PIXEL_FORMAT>(pixelFormatName),
            reinterpret_cast<PEAK_IPL_BOOL8*>(&isPixelFormatSupported));
    });

    return isPixelFormatSupported > 0;
}

inline void ColorCorrector::ProcessInPlace(Image& image) const
{
    ExecuteAndMapReturnCodes([&] {
        return PEAK_IPL_C_ABI_PREFIX PEAK_IPL_ColorCorrector_ProcessInPlace(
            m_backendHandle, ImageBackendAccessor::BackendHandle(image));
    });
}

inline Image ColorCorrector::Process(const Image& inputImage) const
{
    PEAK_IPL_IMAGE_HANDLE outputImageHandle = nullptr;
    ExecuteAndMapReturnCodes([&] {
        return PEAK_IPL_C_ABI_PREFIX PEAK_IPL_ColorCorrector_Process(
            m_backendHandle, ImageBackendAccessor::BackendHandle(inputImage), &outputImageHandle);
    });

    return ImageBackendAccessor::CreateImage(outputImageHandle);
}

} /* namespace ipl */
} /* namespace peak */
