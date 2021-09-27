/*!
 * \file    peak_common_system_enums.hpp
 *
 * \author  IDS Imaging Development Systems GmbH
 * \date    2019-05-01
 * \since   1.0
 *
 * Copyright (c) 2019, IDS Imaging Development Systems GmbH. All rights reserved.
 */

#pragma once


namespace peak
{
namespace core
{

/*!
 * \brief Possible character encodings.
 *
 * See GenTL TL_CHAR_ENCODING_LIST.
 */
enum class CharacterEncoding
{
    ASCII = 0,
    UTF8
};

inline std::string ToString(CharacterEncoding entry)
{
    std::string entryString;

    if (entry == CharacterEncoding::ASCII)
    {
        entryString = "ASCII";
    }
    else if (entry == CharacterEncoding::UTF8)
    {
        entryString = "UTF8";
    }

    return entryString;
}

} /* namespace core */
} /* namespace peak */
