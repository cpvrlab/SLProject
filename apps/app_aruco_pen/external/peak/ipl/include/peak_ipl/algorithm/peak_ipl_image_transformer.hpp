/*!
 * \file    peak_ipl_image_transformer.hpp
 *
 * \author  IDS Imaging Development Systems GmbH
 * \date    2019-05-15
 * \since   1.0
 *
 * Copyright (c) 2019, IDS Imaging Development Systems GmbH. All rights reserved.
 */

#pragma once

#include <peak_ipl/backend/peak_ipl_backend.h>
#include <peak_ipl/exception/peak_ipl_exception.hpp>
#include <peak_ipl/types/peak_ipl_pixel_format.hpp>
#include <peak_ipl/types/peak_ipl_simple_types.hpp>

/*!
 * \brief The "peak::ipl" namespace contains the whole image processing library.
 */
namespace peak
{
namespace ipl
{

class Image;

/*!
 * \brief Performs transformations like mirror and rotate on images.
 *
 * \note To speed up processing instances of this class maintain internal memory pools to reuse
 * memory instead of allocating new memory for each transformation. The memory is freed when the
 * instance is destroyed.
 */
class ImageTransformer final
{
public:
    /*!
     * \brief Angle parameter for the Rotation algorithm.
     *
     * The enum holding the possible rotation angles and the rotation direction.
     */
    enum class RotationAngle : uint16_t
    {
        Degree90Counterclockwise = 90,
        Degree90Clockwise = 270,
        Degree180 = 180
    };

    ImageTransformer();
    ~ImageTransformer();
    ImageTransformer(const ImageTransformer& other) = delete;
    ImageTransformer& operator=(const ImageTransformer& other) = delete;
    ImageTransformer(ImageTransformer&& other);
    ImageTransformer& operator=(ImageTransformer&& other);

    /*!
     * \brief Mirrors the input image in up-down direction.
     *
     * If the transformed image is a bayer-format image and the number of rows is even,
     * the format will change. (e.g. BayerBG8 -> BayerGR8)
     *
     * \param[in]  inputImage           The handle to the created image.
     *
     * \returns A new created image containing the data of the input image mirrored in up-down direction
     *
     * \throws ImageFormatNotSupportedException inputImage has packed pixel format
     * \throws Exception An internal error has occurred.
     *
     * \since 1.0
     * \since 1.2 Will throw ImageFormatNotSupportedException if inputImage has packed pixel format
     */
    Image MirrorUpDown(const Image& inputImage) const;

    /*!
     * \brief Mirrors the input image in left-right direction.
     *
     * If the transformed image is a bayer-format image and the number of columns is even,
     * the format will change. (e.g. BayerBG8 -> BayerGB8)
     *
     * \param[in]  inputImage           The handle to the created image.
     *
     * \returns A new created image containing the data of the input image mirrored in left-right direction
     *
     * \throws ImageFormatNotSupportedException inputImage has packed pixel format
     * \throws Exception An internal error has occurred.
     *
     * \since 1.0
     * \since 1.2 Will throw ImageFormatNotSupportedException if inputImage has packed pixel format
     */
    Image MirrorLeftRight(const Image& inputImage) const;

    /*!
     * \brief Mirrors the input image in up-down and left-right direction.
     *
     * If the transformed image is a bayer-format image and the number of rows or columns are even,
     * the format will change. (e.g. BayerBG8 -> BayerRG8)
     *
     * \param[in]  inputImage           The handle to the created image.
     *
     * \returns A new created image containing the data of the input image mirrored in up-down and left-right direction
     *
     * \throws ImageFormatNotSupportedException inputImage has packed pixel format
     * \throws Exception An internal error has occurred.
     *
     * \since 1.0
     * \since 1.2 Will throw ImageFormatNotSupportedException if inputImage has packed pixel format
     */
    Image MirrorUpDownLeftRight(const Image& inputImage) const;


    /*!
     * \brief Mirrors the given image in up-down direction in place i.e. it will change the input image itself.
     *
     * If the transformed image is a bayer-format image and the number of rows is even,
     * the format will change. (e.g. BayerBG8 -> BayerGR8)
     *
     * \param[in,out]  image           The handle to the image to mirror
     *
     * \throws ImageFormatNotSupportedException image has packed pixel format
     * \throws Exception An internal error has occurred.
     *
     * \since 1.0
     * \since 1.2 Will throw ImageFormatNotSupportedException if image has packed pixel format
     */
    void MirrorUpDownInPlace(Image& image) const;

    /*!
     * \brief Mirrors the given image in left-right direction in place i.e. it will change the input image itself.
     *
     * If the transformed image is a bayer-format image and the number of columns is even,
     * the format will change. (e.g. BayerBG8 -> BayerGB8)
     *
     * \param[in,out]  image           The handle to the image to mirror
     *
     * \throws ImageFormatNotSupportedException image has packed pixel format
     * \throws Exception An internal error has occurred.
     *
     * \since 1.0
     * \since 1.2 Will throw ImageFormatNotSupportedException if image has packed pixel format
     */
    void MirrorLeftRightInPlace(Image& image) const;

    /*!
     * \brief Mirrors the given image in up-down and left-right direction in place i.e. it will change the input
     *         image itself.
     *
     * If the transformed image is a bayer-format image and the number of rows or columns are even,
     * the format will change. (e.g. BayerBG8 -> BayerRG8)
     *
     * \param[in,out]  image           The handle to the image to mirror
     *
     * \throws ImageFormatNotSupportedException image has packed pixel format
     * \throws Exception An internal error has occurred.
     *
     * \since 1.0
     * \since 1.2 Will throw ImageFormatNotSupportedException if image has packed pixel format
     */
    void MirrorUpDownLeftRightInPlace(Image& image) const;

    /*!
     * \brief Rotate the input image with the given rotationAngle
     *
     * If the transformed image is a bayer-format image and the number of rows or columns are even,
     * the format will change. (e.g. BayerBG8 -> BayerRG8)
     *
     * \param[in]  inputImage           The handle to the created image.
     *
     * \param[in]  rotationAngle        The rotation angle.
     *
     * \returns A new created image containing the data of the input image rotated with the given rotationAngle
     *
     * \throws ImageFormatNotSupportedException inputImage has packed pixel format
     * \throws Exception An internal error has occurred.
     *
     * \since 1.2
     */
    Image Rotate(const Image& inputImage, RotationAngle rotationAngle) const;

    /*!
     * \brief Rotate the input image with the given rotationAngle
     *
     * If the transformed image is a bayer-format image and the number of rows or columns are even,
     * the format will change. (e.g. BayerBG8 -> BayerRG8)
     *
     * \param[in,out]  image           The handle to the created image.
     *
     * \param[in]  rotationAngle        The rotation angle.
     *
     * \throws ImageFormatNotSupportedException image has packed pixel format
     * \throws Exception An internal error has occurred.
     *
     * \since 1.2
     */
    void RotateInPlace(Image& image, RotationAngle rotationAngle) const;

private:
    PEAK_IPL_IMAGE_TRANSFORMER_HANDLE m_backendHandle{};
};

} /* namespace ipl */
} /* namespace peak */

#include <peak_ipl/types/peak_ipl_image.hpp>


namespace peak
{
namespace ipl
{

inline ImageTransformer::ImageTransformer()
{
    ExecuteAndMapReturnCodes([&] { return PEAK_IPL_C_ABI_PREFIX PEAK_IPL_ImageTransformer_Construct(&m_backendHandle); });
}


inline ImageTransformer::~ImageTransformer()
{
    (void)PEAK_IPL_C_ABI_PREFIX PEAK_IPL_ImageTransformer_Destruct(m_backendHandle);
}

inline ImageTransformer::ImageTransformer(ImageTransformer&& other)
{
    *this = std::move(other);
}

inline ImageTransformer& ImageTransformer::operator=(ImageTransformer&& other)
{
    if (this != &other)
    {
        (void)PEAK_IPL_C_ABI_PREFIX PEAK_IPL_ImageTransformer_Destruct(m_backendHandle);
        m_backendHandle = other.m_backendHandle;
        other.m_backendHandle = nullptr;
    }

    return *this;
}

inline Image ImageTransformer::MirrorUpDown(const Image& inputImage) const
{
    PEAK_IPL_IMAGE_HANDLE outputImageHandle = nullptr;
    ExecuteAndMapReturnCodes([&] {
        return PEAK_IPL_C_ABI_PREFIX PEAK_IPL_ImageTransformer_MirrorUpDown(
            m_backendHandle, ImageBackendAccessor::BackendHandle(inputImage), &outputImageHandle);
    });

    return ImageBackendAccessor::CreateImage(outputImageHandle);
}

inline Image ImageTransformer::MirrorLeftRight(const Image& inputImage) const
{
    PEAK_IPL_IMAGE_HANDLE outputImageHandle = nullptr;
    ExecuteAndMapReturnCodes([&] {
        return PEAK_IPL_C_ABI_PREFIX PEAK_IPL_ImageTransformer_MirrorLeftRight(
            m_backendHandle, ImageBackendAccessor::BackendHandle(inputImage), &outputImageHandle);
    });

    return ImageBackendAccessor::CreateImage(outputImageHandle);
}

inline Image ImageTransformer::MirrorUpDownLeftRight(const Image& inputImage) const
{
    PEAK_IPL_IMAGE_HANDLE outputImageHandle = nullptr;
    ExecuteAndMapReturnCodes([&] {
        return PEAK_IPL_C_ABI_PREFIX PEAK_IPL_ImageTransformer_MirrorUpDownLeftRight(
            m_backendHandle, ImageBackendAccessor::BackendHandle(inputImage), &outputImageHandle);
    });

    return ImageBackendAccessor::CreateImage(outputImageHandle);
}

inline void ImageTransformer::MirrorUpDownInPlace(Image& image) const
{
    ExecuteAndMapReturnCodes([&] {
        return PEAK_IPL_C_ABI_PREFIX PEAK_IPL_ImageTransformer_MirrorUpDownInPlace(
            m_backendHandle, ImageBackendAccessor::BackendHandle(image));
    });
}

inline void ImageTransformer::MirrorLeftRightInPlace(Image& image) const
{
    ExecuteAndMapReturnCodes([&] {
        return PEAK_IPL_C_ABI_PREFIX PEAK_IPL_ImageTransformer_MirrorLeftRightInPlace(
            m_backendHandle, ImageBackendAccessor::BackendHandle(image));
    });
}

inline void ImageTransformer::MirrorUpDownLeftRightInPlace(Image& image) const
{
    ExecuteAndMapReturnCodes([&] {
        return PEAK_IPL_C_ABI_PREFIX PEAK_IPL_ImageTransformer_MirrorUpDownLeftRightInPlace(
            m_backendHandle, ImageBackendAccessor::BackendHandle(image));
    });
}

inline Image ImageTransformer::Rotate(const Image& inputImage, RotationAngle rotationAngle) const
{
    PEAK_IPL_IMAGE_HANDLE outputImageHandle = nullptr;
    ExecuteAndMapReturnCodes([&] {
        return PEAK_IPL_C_ABI_PREFIX PEAK_IPL_ImageTransformer_Rotate(m_backendHandle,
            ImageBackendAccessor::BackendHandle(inputImage), &outputImageHandle,
            static_cast<PEAK_IPL_IMAGE_TRANSFORMER_ROTATION_ANGLE_t>(rotationAngle));
    });

    return ImageBackendAccessor::CreateImage(outputImageHandle);
}


inline void ImageTransformer::RotateInPlace(Image& image, RotationAngle rotationAngle) const
{
    ExecuteAndMapReturnCodes([&] {
        return PEAK_IPL_C_ABI_PREFIX PEAK_IPL_ImageTransformer_RotateInPlace(m_backendHandle,
            ImageBackendAccessor::BackendHandle(image),
            static_cast<PEAK_IPL_IMAGE_TRANSFORMER_ROTATION_ANGLE_t>(rotationAngle));
    });
}


} /* namespace ipl */
} /* namespace peak */
