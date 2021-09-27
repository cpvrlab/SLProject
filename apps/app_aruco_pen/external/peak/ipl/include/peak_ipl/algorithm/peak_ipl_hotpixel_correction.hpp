/*!
 * \file    peak_ipl_hotpixel_correction.hpp
 *
 * \author  IDS Imaging Development Systems GmbH
 * \date    2019-05-01
 * \since   1.0
 *
 * Copyright (c) 2019, IDS Imaging Development Systems GmbH. All rights reserved.
 */

#pragma once

#include <peak_ipl/backend/peak_ipl_backend.h>

#include <cstdint>
#include <vector>

namespace peak
{
namespace ipl
{

class Image;

/*!
 * \brief 2D position in an image.
 */
struct Point2D
{
    size_t x;
    size_t y;
};

/*!
 * \brief Algorithm for (adaptive) hotpixel detection and correction.
 *
 * This algorithm corrects hotpixels either manually by correcting a predefined list of pixels or adaptively by
 * automatically updating the list of pixels to correct. In manual mode, pass a list of hotpixels to Correct(). The
 * hotpixels can obtained using Detect(), or from a different source, e.g. retrieved directly from the camera from an
 * initial hotpixel calibration. In adaptive mode, just keep passing new images to CorrectAdaptive(). Hotpixels are then
 * detected and corrected automatically.
 */
class HotpixelCorrection final
{
public:
    /*!
     * \brief SensitivityLevel parameter for the HotpixelCorrection algorithm.
     *
     * Higher sensitivity levels mean more hotpixels will be detected and corrected, but can also lead to more
     * false-positives.
     */
    enum class SensitivityLevel
    {
        Invalid,
        SensitivityLevel1,
        SensitivityLevel2,
        SensitivityLevel3, // default
        SensitivityLevel4,
        SensitivityLevel5
    };

    HotpixelCorrection();
    ~HotpixelCorrection();
    HotpixelCorrection(const HotpixelCorrection& other) = delete;
    HotpixelCorrection& operator=(const HotpixelCorrection& other) = delete;
    HotpixelCorrection(HotpixelCorrection&& other) noexcept;
    HotpixelCorrection& operator=(HotpixelCorrection&& other) noexcept;

    /*!
     * \brief Sets the sensitivity of the hotpixel detection.
     *
     * \param[in] sensitivityLevel The sensitivity level to set.
     *
     * \throws Exception An internal error has occurred.
     *
     * \since 1.0
     */
    void SetSensitivity(SensitivityLevel sensitivityLevel = SensitivityLevel::SensitivityLevel3);

    /*!
     * \brief Returns the current sensitivity.
     *
     * \returns SensitivityLevel
     *
     * \throws Exception An internal error has occurred.
     *
     * \since 1.0
     */
    SensitivityLevel Sensitivity() const;

    /*!
     * \brief Sets the gain factor in percent.
     *
     * \param[in] gainFactorPercent The gain factor in percent to set.
     *
     * \throws Exception An internal error has occurred.
     *
     * \since 1.0
     */
    void SetGainFactorPercent(uint32_t gainFactorPercent = 100);

    /*!
     * \brief Returns the current gain factor in percent.
     *
     * \returns Gain factor in percent
     *
     * \throws Exception An internal error has occurred.
     *
     * \since 1.0
     */
    uint32_t GainFactorPercent() const;


    /*!
     * \brief Detects hotpixels in the given image.
     *
     * \param[in] inputImage The input image.
     *
     * \returns List of detected hotpixels
     *
     * \throws ImageFormatNotSupportedException inputImage has packed pixel format
     * \throws Exception An internal error has occurred.
     *
     * \since 1.0
     * \since 1.2 Will throw ImageFormatNotSupportedException if inputImage has packed pixel format
     */
    std::vector<Point2D> Detect(const Image& inputImage) const;

    /*!
     * \brief Corrects the given hotpixels in the given image.
     *
     * Pass a list of hotpixels, either returned by Detect(), or from a different source (e.g. retrieved directly from
     * the camera from an initial hotpixel calibration).
     *
     * \param[in] inputImage The input image.
     * \param[in] hotpixels  The list of hotpixels to be corrected.
     *
     * \returns Corrected image
     *
     * \throws ImageFormatNotSupportedException inputImage has packed pixel format
     * \throws Exception An internal error has occurred.
     *
     * \since 1.0
     * \since 1.2 Will throw ImageFormatNotSupportedException if inputImage has packed pixel format
     */
    Image Correct(const Image& inputImage, const std::vector<Point2D>& hotpixels) const;

    /*!
     * \brief Corrects the given image adaptively.
     *
     * For each new passed to this method, first the list of hotpixels is adapted, then all pixels in the adapted
     * hotpixel list are corrected.
     *
     * \param[in] inputImage The input image.
     *
     * \returns Corrected image
     *
     * \throws ImageFormatNotSupportedException inputImage has packed pixel format
     * \throws Exception An internal error has occurred.
     *
     * \since 1.0
     * \since 1.2 Will throw ImageFormatNotSupportedException if inputImage has packed pixel format
     */
    Image CorrectAdaptive(const Image& inputImage);

private:
    PEAK_IPL_ADAPTIVE_HOTPIXEL_CORRECTOR_HANDLE m_backendHandle{};
};

} /* namespace ipl */
} /* namespace peak */

#include <peak_ipl/types/peak_ipl_image.hpp>

namespace peak
{
namespace ipl
{

inline HotpixelCorrection::HotpixelCorrection()
{
    ExecuteAndMapReturnCodes(
        [&] { return PEAK_IPL_C_ABI_PREFIX PEAK_IPL_AdaptiveHotpixelCorrector_Construct(&m_backendHandle); });

    // make sure default params are set
    SetSensitivity();
    SetGainFactorPercent();
}

inline HotpixelCorrection::~HotpixelCorrection()
{
    (void)PEAK_IPL_C_ABI_PREFIX PEAK_IPL_AdaptiveHotpixelCorrector_Destruct(m_backendHandle);
}

inline HotpixelCorrection::HotpixelCorrection(HotpixelCorrection&& other) noexcept
{
    *this = std::move(other);
}

inline HotpixelCorrection& HotpixelCorrection::operator=(HotpixelCorrection&& other) noexcept
{
    if (this != &other)
    {
        (void)PEAK_IPL_C_ABI_PREFIX PEAK_IPL_AdaptiveHotpixelCorrector_Destruct(m_backendHandle);
        m_backendHandle = other.m_backendHandle;
        other.m_backendHandle = nullptr;
    }

    return *this;
}

inline void HotpixelCorrection::SetSensitivity(SensitivityLevel sensitivityLevel)
{
    ExecuteAndMapReturnCodes([&] {
        return PEAK_IPL_C_ABI_PREFIX PEAK_IPL_AdaptiveHotpixelCorrector_SetSensitivity(
            m_backendHandle, static_cast<PEAK_IPL_HOTPIXELCORRECTION_SENSITIVITY>(sensitivityLevel));
    });
}

inline HotpixelCorrection::SensitivityLevel HotpixelCorrection::Sensitivity() const
{
    PEAK_IPL_HOTPIXELCORRECTION_SENSITIVITY sensitivityLevel;
    ExecuteAndMapReturnCodes([&] {
        return PEAK_IPL_C_ABI_PREFIX PEAK_IPL_AdaptiveHotpixelCorrector_GetSensitivity(
            m_backendHandle, &sensitivityLevel);
    });

    return static_cast<SensitivityLevel>(sensitivityLevel);
}

inline void HotpixelCorrection::SetGainFactorPercent(uint32_t gainFactorPercent)
{
    ExecuteAndMapReturnCodes([&] {
        return PEAK_IPL_C_ABI_PREFIX PEAK_IPL_AdaptiveHotpixelCorrector_SetGainFactorPercent(
            m_backendHandle, gainFactorPercent);
    });
}

inline uint32_t HotpixelCorrection::GainFactorPercent() const
{
    uint32_t gainFactorPercent;
    ExecuteAndMapReturnCodes([&] {
        return PEAK_IPL_C_ABI_PREFIX PEAK_IPL_AdaptiveHotpixelCorrector_GetGainFactorPercent(
            m_backendHandle, &gainFactorPercent);
    });

    return gainFactorPercent;
}

inline std::vector<Point2D> HotpixelCorrection::Detect(const Image& inputImage) const
{
    ExecuteAndMapReturnCodes([&] {
        return PEAK_IPL_C_ABI_PREFIX PEAK_IPL_AdaptiveHotpixelCorrector_Detect(
            m_backendHandle, ImageBackendAccessor::BackendHandle(inputImage));
    });

    size_t size = 0;
    ExecuteAndMapReturnCodes([&] {
        return PEAK_IPL_C_ABI_PREFIX PEAK_IPL_AdaptiveHotpixelCorrector_GetHotpixels(m_backendHandle, nullptr, &size);
    });
    std::vector<Point2D> hotpixels(size);
    ExecuteAndMapReturnCodes([&] {
        return PEAK_IPL_C_ABI_PREFIX PEAK_IPL_AdaptiveHotpixelCorrector_GetHotpixels(
            m_backendHandle, reinterpret_cast<PEAK_IPL_POINT_2D*>(hotpixels.data()), &size);
    });

    return hotpixels;
}

inline Image HotpixelCorrection::Correct(const Image& inputImage, const std::vector<Point2D>& hotpixels) const
{
    PEAK_IPL_IMAGE_HANDLE outputImageHandle = nullptr;
    ExecuteAndMapReturnCodes([&] {
        return PEAK_IPL_C_ABI_PREFIX PEAK_IPL_AdaptiveHotpixelCorrector_Correct(m_backendHandle,
            ImageBackendAccessor::BackendHandle(inputImage),
            reinterpret_cast<const PEAK_IPL_POINT_2D*>(hotpixels.data()), hotpixels.size(), &outputImageHandle);
    });

    return ImageBackendAccessor::CreateImage(outputImageHandle);
}

inline Image HotpixelCorrection::CorrectAdaptive(const Image& inputImage)
{
    PEAK_IPL_IMAGE_HANDLE outputImageHandle = nullptr;
    ExecuteAndMapReturnCodes([&] {
        return PEAK_IPL_C_ABI_PREFIX PEAK_IPL_AdaptiveHotpixelCorrector_CorrectAdaptive(
            m_backendHandle, ImageBackendAccessor::BackendHandle(inputImage), &outputImageHandle);
    });

    return ImageBackendAccessor::CreateImage(outputImageHandle);
}


} /* namespace ipl */
} /* namespace peak */
