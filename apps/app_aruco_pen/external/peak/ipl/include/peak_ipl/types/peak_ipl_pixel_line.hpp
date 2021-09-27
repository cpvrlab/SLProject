/*!
 * \file    peak_ipl_PIXEL_LINE.hpp
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
 * \brief Represents the values of a horizontal or vertical line of pixels in an image.
 */
class PixelLine
{
public:
    /*!
     * \brief Represents the values of the pixel line.
     */
    struct Channel
    {
        std::vector<uint32_t> Values;
    };

public:
    PixelLine() = delete;
    /*! \brief Constructor.
     *
     * \param[in] image The image to process.
     * \param[in] orientation The orientation of the pixel line.
     * \param[in] offset Y offset if orientation = peak::ipl::Orientation::Horizontal; X offset if orientation = peak::ipl::Orientation::Vertical
     *
     * \throws ImageFormatNotSupportedException image has packed pixel format
     *
     * \since 1.0
     * \since 1.2 Will throw ImageFormatNotSupportedException if image has packed pixel format
     */
    explicit PixelLine(const Image& image, peak::ipl::Orientation orientation, size_t offset);
    virtual ~PixelLine();
    PixelLine(const PixelLine& other) = delete;
    PixelLine& operator=(const PixelLine& other) = delete;
    PixelLine(PixelLine&& other);
    PixelLine& operator=(PixelLine&& other);

    /*! \brief Returns the pixel format of the given pixel line.
     *
     * \returns PixelFormatName
     *
     * \since 1.0
     *
     * \throws Exception An internal error has occurred.
     */
    peak::ipl::PixelFormatName PixelFormatName() const;

    /*! \brief Returns the orientation of the given pixel line.
     *
     * \returns Orientation
     *
     * \since 1.0
     *
     * \throws Exception An internal error has occurred.
     */
    peak::ipl::Orientation Orientation() const;

    /*! \brief Returns the offset of the given pixel line.
     * (Vertical: Left - Horizontal: Top).
     *
     * \returns Offset
     *
     * \since 1.0
     *
     * \throws Exception An internal error has occurred.
     */
    size_t Offset() const;

    /*! \brief Returns the number of the given pixel line's channels.
     *
     * \returns Channels
     *
     * \since 1.0
     *
     * \throws Exception An internal error has occurred.
     */
    std::vector<Channel> Channels() const;

private:
    PEAK_IPL_PIXEL_LINE_HANDLE m_backendHandle{};
};

/*!
 * \brief Represents the values of a horizontal line of pixels in an image.
 */
class PixelRow : public PixelLine
{
public:
    PixelRow() = delete;
    explicit PixelRow(const Image& image, size_t row);
    virtual ~PixelRow() override = default;
    PixelRow(const PixelRow& other) = delete;
    PixelRow& operator=(const PixelRow& other) = delete;
    PixelRow(PixelRow&& other) = default;
    PixelRow& operator=(PixelRow&& other) = default;
};

/*!
 * \brief Represents the values of a vertical line of pixels in an image.
 */
class PixelColumn : public PixelLine
{
public:
    PixelColumn() = delete;
    explicit PixelColumn(const Image& image, size_t column);
    virtual ~PixelColumn() override = default;
    PixelColumn(const PixelColumn& other) = delete;
    PixelColumn& operator=(const PixelColumn& other) = delete;
    PixelColumn(PixelColumn&& other) = default;
    PixelColumn& operator=(PixelColumn&& other) = default;
};

inline PixelRow::PixelRow(const Image& image, size_t row)
    : PixelLine(image, peak::ipl::Orientation::Horizontal, row)
{}

inline PixelColumn::PixelColumn(const Image& image, size_t column)
    : PixelLine(image, peak::ipl::Orientation::Vertical, column)
{}


inline PixelLine::PixelLine(const Image& image, peak::ipl::Orientation orientation, size_t offset)
{
    ExecuteAndMapReturnCodes([&] {
        return PEAK_IPL_C_ABI_PREFIX PEAK_IPL_PixelLine_Construct(ImageBackendAccessor::BackendHandle(image),
            static_cast<PEAK_IPL_ORIENTATION>(orientation), offset, &m_backendHandle);
    });
}

inline PixelLine::~PixelLine()
{
    (void)PEAK_IPL_C_ABI_PREFIX PEAK_IPL_PixelLine_Destruct(m_backendHandle);
}

inline PixelLine::PixelLine(PixelLine&& other)
{
    *this = std::move(other);
}

inline PixelLine& PixelLine::operator=(PixelLine&& other)
{
    if (this != &other)
    {
        (void)PEAK_IPL_C_ABI_PREFIX PEAK_IPL_PixelLine_Destruct(m_backendHandle);
        m_backendHandle = other.m_backendHandle;
        other.m_backendHandle = nullptr;
    }

    return *this;
}

inline peak::ipl::PixelFormatName PixelLine::PixelFormatName() const
{
    peak::ipl::PixelFormatName pixelFormatName = peak::ipl::PixelFormatName::Invalid;

    ExecuteAndMapReturnCodes([&] {
        return PEAK_IPL_C_ABI_PREFIX PEAK_IPL_PixelLine_GetPixelFormat(
            m_backendHandle, reinterpret_cast<PEAK_IPL_PIXEL_FORMAT*>(&pixelFormatName));
    });

    return pixelFormatName;
}

inline peak::ipl::Orientation PixelLine::Orientation() const
{
    peak::ipl::Orientation orientation = peak::ipl::Orientation::Horizontal;

    ExecuteAndMapReturnCodes([&] {
        return PEAK_IPL_C_ABI_PREFIX PEAK_IPL_PixelLine_GetOrientation(
            m_backendHandle, reinterpret_cast<PEAK_IPL_ORIENTATION*>(&orientation));
    });

    return orientation;
}

inline size_t PixelLine::Offset() const
{
    size_t offset = 0;

    ExecuteAndMapReturnCodes(
        [&] { return PEAK_IPL_C_ABI_PREFIX PEAK_IPL_PixelLine_GetOffset(m_backendHandle, &offset); });

    return offset;
}

inline std::vector<PixelLine::Channel> PixelLine::Channels() const
{
    std::vector<Channel> channels;

    size_t numChannels = 0;
    ExecuteAndMapReturnCodes(
        [&] { return PEAK_IPL_C_ABI_PREFIX PEAK_IPL_PixelLine_GetNumChannels(m_backendHandle, &numChannels); });

    for (size_t ch = 0; ch < numChannels; ++ch)
    {
        size_t valueListSize = 0;
        ExecuteAndMapReturnCodes([&] {
            return PEAK_IPL_C_ABI_PREFIX PEAK_IPL_PixelLine_GetValuesForChannel(
                m_backendHandle, ch, nullptr, &valueListSize);
        });
        std::vector<uint32_t> valueList(valueListSize);
        ExecuteAndMapReturnCodes([&] {
            return PEAK_IPL_C_ABI_PREFIX PEAK_IPL_PixelLine_GetValuesForChannel(
                m_backendHandle, ch, valueList.data(), &valueListSize);
        });

        channels.emplace_back(Channel{ std::move(valueList) });
    }

    return channels;
}

} /* namespace ipl */
} /* namespace peak */
