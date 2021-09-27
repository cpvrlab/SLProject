/*!
 * \file    peak_common_structs.hpp
 *
 * \author  IDS Imaging Development Systems GmbH
 * \date    2019-05-01
 * \since   1.0
 *
 * Copyright (c) 2019, IDS Imaging Development Systems GmbH. All rights reserved.
 */

#pragma once


#include <cstdint>
#include <vector>


namespace peak
{
namespace core
{

/*! \brief The struct returned by all raw information functions. */
struct RawInformation
{
    int32_t DataType;
    std::vector<uint8_t> Data;
};

} /* namespace core */
} /* namespace peak */
