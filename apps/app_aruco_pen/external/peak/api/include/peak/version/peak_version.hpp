/*!
 * \file    peak_version.hpp
 *
 * \author  IDS Imaging Development Systems GmbH
 * \date    2019-05-01
 * \since   1.0
 *
 * Copyright (c) 2019, IDS Imaging Development Systems GmbH. All rights reserved.
 */

#pragma once


#include <peak/backend/peak_backend.h>
#include <peak/dll_interface/peak_dll_interface_util.hpp>

#include <cstdint>
#include <sstream>
#include <string>


namespace peak
{
namespace core
{

/*!
 * \brief Implements versioning functionality.
 *
 * This class allows to create and compare different versions.
 *
 */
class Version final
{
public:
    Version() = delete;
    Version(uint32_t major, uint32_t minor, uint32_t subminor);
    ~Version() = default;
    Version(const Version& other) = default;
    Version& operator=(const Version& other) = default;
    Version(Version&& other) = default;
    Version& operator=(Version&& other) = default;

    /*!
     * \brief Returns the version as string.
     *
     * \since 1.0
     */
    std::string ToString() const;
    /*!
     * \brief Returns the major part of the version which is the first part of the version scheme separated by dots.
     *
     * \return <b>x</b>.y.z
     *
     * \since 1.0
     */
    uint32_t Major() const;
    /*!
     * \brief Returns the minor part of the version which is the second part of the version scheme separated by dots.
     *
     * \return x.<b>y</b>.z
     *
     * \since 1.0
     */
    uint32_t Minor() const;
    /*!
     * \brief Returns the subminor part of the version which is the third part of the version scheme separated by dots.
     *
     * \return x.y.<b>z</b>
     *
     * \since 1.0
     */
    uint32_t Subminor() const;

private:
    uint32_t m_major;
    uint32_t m_minor;
    uint32_t m_subminor;
};

bool operator<(const Version& lhs, const Version& rhs);
bool operator>(const Version& lhs, const Version& rhs);
bool operator==(const Version& lhs, const Version& rhs);
bool operator!=(const Version& lhs, const Version& rhs);

} /* namespace core */
} /* namespace peak */

/* Implementation */
namespace peak
{
namespace core
{

inline Version::Version(uint32_t major, uint32_t minor, uint32_t subminor)
    : m_major(major)
    , m_minor(minor)
    , m_subminor(subminor)
{}

inline std::string Version::ToString() const
{
    std::stringstream strStream;

    strStream << Major() << '.' << Minor() << '.' << Subminor();

    return strStream.str();
}

inline uint32_t Version::Major() const
{
    return m_major;
}

inline uint32_t Version::Minor() const
{
    return m_minor;
}

inline uint32_t Version::Subminor() const
{
    return m_subminor;
}

inline bool operator<(const Version& lhs, const Version& rhs)
{
    if (lhs.Major() < rhs.Major())
    {
        return true;
    }
    else if (lhs.Minor() < rhs.Minor())
    {
        return true;
    }
    else if (lhs.Subminor() < rhs.Subminor())
    {
        return true;
    }

    return false;
}

inline bool operator>(const Version& lhs, const Version& rhs)
{
    return rhs < lhs;
}

inline bool operator==(const Version& lhs, const Version& rhs)
{
    return !(lhs < rhs) && !(lhs > rhs);
}

inline bool operator!=(const Version& lhs, const Version& rhs)
{
    return !(lhs == rhs);
}

} /* namespace core */
} /* namespace peak */
