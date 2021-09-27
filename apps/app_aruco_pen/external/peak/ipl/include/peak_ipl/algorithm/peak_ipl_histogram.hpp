/*!
 * \file    peak_ipl_histogram.hpp
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
#include <peak_ipl/types/peak_ipl_simple_types.hpp>

#include <cstddef>
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

/*!
 * \brief Stores the histogram values.
 */
class Histogram final
{
public:
    /*!
     * \brief Stores the values of the histogram.
     */
    struct HistogramChannel
    {
        uint64_t PixelSum;
        uint64_t PixelCount;
        std::vector<uint64_t> Bins;
    };

public:
    Histogram() = delete;
    /*!
     * \brief Constructor.
     *
     * \param[in] image Image to process.
     *
     * \throws ImageFormatNotSupportedException image has packed pixel format
     *
     * \since 1.0
     * \since 1.2 Will throw ImageFormatNotSupportedException if image has packed pixel format
     */
    explicit Histogram(const Image& image);
    ~Histogram();
    Histogram(const Histogram& other) = delete;
    Histogram& operator=(const Histogram& other) = delete;
    Histogram(Histogram&& other);
    Histogram& operator=(Histogram&& other);

    /*!
     * \brief Returns the pixel format of the histogram.
     *
     * \returns PixelFormat
     *
     * \throws Exception An internal error has occurred.
     *
     * \since 1.0
     */
    peak::ipl::PixelFormat PixelFormat() const;

    /*!
     * \brief Returns a vector containing the bin list of each channel.
     *
     * For more details on how to apply the two-step procedure this function requires, see also PEAK_IPL_GetLastError().
     *
     * \returns Channels
     *
     * \throws Exception An internal error has occurred.
     *
     * \since 1.0
     */
    std::vector<HistogramChannel> Channels() const;

private:
    PEAK_IPL_HISTOGRAM_HANDLE m_backendHandle{};
};

inline Histogram::Histogram(const Image& image)
{
    ExecuteAndMapReturnCodes([&] {
        return PEAK_IPL_C_ABI_PREFIX PEAK_IPL_Histogram_Construct(
            ImageBackendAccessor::BackendHandle(image), &m_backendHandle);
    });
}

inline Histogram::~Histogram()
{
    (void)PEAK_IPL_C_ABI_PREFIX PEAK_IPL_Histogram_Destruct(m_backendHandle);
}

inline Histogram::Histogram(Histogram&& other)
{
    *this = std::move(other);
}

inline Histogram& Histogram::operator=(Histogram&& other)
{
    if (this != &other)
    {
        (void)PEAK_IPL_C_ABI_PREFIX PEAK_IPL_Histogram_Destruct(m_backendHandle);
        m_backendHandle = other.m_backendHandle;
        other.m_backendHandle = nullptr;
    }

    return *this;
}

inline peak::ipl::PixelFormat Histogram::PixelFormat() const
{
    PixelFormatName pixelFormatName = PixelFormatName::Invalid;

    ExecuteAndMapReturnCodes([&] {
        return PEAK_IPL_C_ABI_PREFIX PEAK_IPL_Histogram_GetPixelFormat(
            m_backendHandle, reinterpret_cast<PEAK_IPL_PIXEL_FORMAT*>(&pixelFormatName));
    });

    return peak::ipl::PixelFormat{ pixelFormatName };
}

inline std::vector<Histogram::HistogramChannel> Histogram::Channels() const
{
    std::vector<HistogramChannel> channels;

    size_t numChannels = 0;
    ExecuteAndMapReturnCodes(
        [&] { return PEAK_IPL_C_ABI_PREFIX PEAK_IPL_Histogram_GetNumChannels(m_backendHandle, &numChannels); });

    for (size_t ch = 0; ch < numChannels; ++ch)
    {
        uint64_t pixelSum = 0;
        ExecuteAndMapReturnCodes([&] {
            return PEAK_IPL_C_ABI_PREFIX PEAK_IPL_Histogram_GetPixelSumForChannel(m_backendHandle, ch, &pixelSum);
        });

        uint64_t pixelCount = 0;
        ExecuteAndMapReturnCodes([&] {
            return PEAK_IPL_C_ABI_PREFIX PEAK_IPL_Histogram_GetPixelCountForChannel(m_backendHandle, ch, &pixelCount);
        });

        size_t binListSize = 0;
        ExecuteAndMapReturnCodes([&] {
            return PEAK_IPL_C_ABI_PREFIX PEAK_IPL_Histogram_GetBinsForChannel(m_backendHandle, ch, nullptr, &binListSize);
        });
        std::vector<uint64_t> binList(binListSize);
        ExecuteAndMapReturnCodes([&] {
            return PEAK_IPL_C_ABI_PREFIX PEAK_IPL_Histogram_GetBinsForChannel(
                m_backendHandle, ch, binList.data(), &binListSize);
        });

        channels.emplace_back(HistogramChannel{ pixelSum, pixelCount, std::move(binList) });
    }

    return channels;
}

} /* namespace ipl */
} /* namespace peak */
