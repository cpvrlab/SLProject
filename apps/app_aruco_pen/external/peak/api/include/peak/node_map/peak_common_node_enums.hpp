/*!
 * \file    peak_common_node_enums.hpp
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
namespace nodes
{

/*!
 * The enum holding the possible use policies of node cache .
 */
enum class NodeCacheUsePolicy
{
    UseCache,
    IgnoreCache
};

/*!
 * Possible node increment types for number nodes (float, integer).
 */
enum class NodeIncrementType
{
    NoIncrement,
    FixedIncrement,
    ListIncrement
};

/*!
 * Possible node representations for number nodes (float, integer).
 */
enum class NodeRepresentation
{
    Linear,
    Logarithmic,
    Boolean,
    PureNumber,
    HexNumber,
    IP4Address,
    MACAddress
};

inline std::string ToString(NodeCacheUsePolicy entry)
{
    std::string entryString;

    if (entry == NodeCacheUsePolicy::UseCache)
    {
        entryString = "UseCache";
    }
    else if (entry == NodeCacheUsePolicy::IgnoreCache)
    {
        entryString = "IgnoreCache";
    }

    return entryString;
}

inline std::string ToString(NodeIncrementType entry)
{
    std::string entryString;

    if (entry == NodeIncrementType::NoIncrement)
    {
        entryString = "NoIncrement";
    }
    else if (entry == NodeIncrementType::FixedIncrement)
    {
        entryString = "FixedIncrement";
    }
    else if (entry == NodeIncrementType::ListIncrement)
    {
        entryString = "ListIncrement";
    }

    return entryString;
}

inline std::string ToString(NodeRepresentation entry)
{
    std::string entryString;

    if (entry == NodeRepresentation::Linear)
    {
        entryString = "Linear";
    }
    else if (entry == NodeRepresentation::Logarithmic)
    {
        entryString = "Logarithmic";
    }
    else if (entry == NodeRepresentation::PureNumber)
    {
        entryString = "PureNumber";
    }
    else if (entry == NodeRepresentation::HexNumber)
    {
        entryString = "HexNumber";
    }
    else if (entry == NodeRepresentation::IP4Address)
    {
        entryString = "IP4Address";
    }
    else if (entry == NodeRepresentation::MACAddress)
    {
        entryString = "MACAddress";
    }

    return entryString;
}

} /* namespace nodes */
} /* namespace core */
} /* namespace peak */
