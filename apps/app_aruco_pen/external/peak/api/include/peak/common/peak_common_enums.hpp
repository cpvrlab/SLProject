/*!
 * \file    peak_common_enums.hpp
 *
 * \author  IDS Imaging Development Systems GmbH
 * \date    2019-05-01
 * \since   1.0
 *
 * Copyright (c) 2019, IDS Imaging Development Systems GmbH. All rights reserved.
 */

#pragma once


#include <string>


namespace peak
{
namespace core
{

/*!
 * \brief Endianness of pixel data in a Buffer or the data in a Port.
 *
 * See GenTL PIXELENDIANNESS_IDS.
 */
enum class Endianness
{
    /*!
     * Endianness of the data is unknown to the GenTL Producer.
     */
    Unknown,
    /*!
     * The data is stored in little endian format.
     */
    Little,
    /*!
     * The data is stored in big endian format.
     */
    Big
};

inline std::string ToString(Endianness entry)
{
    std::string entryString;

    if (entry == Endianness::Unknown)
    {
        entryString = "Unknown";
    }
    else if (entry == Endianness::Little)
    {
        entryString = "Little";
    }
    else if (entry == Endianness::Big)
    {
        entryString = "Big";
    }

    return entryString;
}

} /* namespace core */
} /* namespace peak */
