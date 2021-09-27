/*!
 * \file    peak_timeout.hpp
 *
 * \author  IDS Imaging Development Systems GmbH
 * \date    2019-05-01
 * \since   1.1
 *
 * Copyright (c) 2019, IDS Imaging Development Systems GmbH. All rights reserved.
 */

#pragma once

#include <peak/backend/peak_backend.h>

#include <chrono>
#include <cstdint>

namespace peak
{
namespace core
{

/*!
 * \brief Represents a timeout value in milliseconds.
 *
 * Use INFINITE_TIMEOUT for an infinite timeout.
 *
 * \since   1.1
 */
class Timeout
{
public:
    /*!
     * \brief The constant defining an infinite timeout.
     *
     * The corresponding function will only return after the operation is completed.
     */
    static const uint64_t INFINITE_TIMEOUT = PEAK_INFINITE_TIMEOUT;

    Timeout(uint64_t timeout = 0) noexcept
        : m_timeout(timeout)
    {}

    Timeout(std::chrono::milliseconds timeout) noexcept
        : m_timeout(timeout)
    {}

    operator uint64_t() const
    {
        return static_cast<uint64_t>(m_timeout.count());
    }

private:
    std::chrono::milliseconds m_timeout;
};

} /* namespace core */
} /* namespace peak */
