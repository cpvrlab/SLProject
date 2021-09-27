/*!
 * \file    peak_buffer_converter.hpp
 *
 * \author  IDS Imaging Development Systems GmbH
 * \date    2019-05-01
 * \since   1.0
 *
 * Copyright (c) 2019, IDS Imaging Development Systems GmbH. All rights reserved.
 */

#pragma once


#include <peak/buffer/peak_buffer.hpp>


namespace peak
{

/*!
 * \brief Converts a core::Buffer into a different type.
 *
 * This function converts the buffer into the format requested as template argument. Implement a template specialization
 * for your custom data type yourself. For an example, have a look at the implementation to convert the buffer to a
 * peak::ipl::Image, i.e. see file converters/peak_buffer_converter_ipl.hpp.
 *
 * Usage:
 * \code
 * auto image = peak::BufferTo<CustomImageType>(buffer);
 * \endcode
 *
 * \param[in] buffer A core::Buffer that contains image data.
 *
 * \return Image data in the requested image container/class.
 *
 * \since 1.0
 *
 * \throws InternalErrorException An internal error has occurred.
 */

template <class OutType>
OutType BufferTo(const std::shared_ptr<core::Buffer>& buffer)
{
    (void)buffer;
    // static_assert that's always false, so we get a compiler error when using a not-implemented specialization
    static_assert(sizeof(OutType) == 0,
        "BufferTo<>() not implemented for this type. Implement a specialization for your own type or include one from "
        "the 'converters' folder.");
}

} /* namespace peak */
