/*!
 * \file    peak_float_node.hpp
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
#include <peak/node_map/peak_common_node_enums.hpp>
#include <peak/node_map/peak_node.hpp>

#include <cstdint>
#include <string>
#include <vector>


namespace peak
{
namespace core
{
namespace nodes
{

/*!
 * \brief Display notation for numbers in FloatNode.
 */
enum class NodeDisplayNotation
{
    Automatic,
    Fixed,
    Scientific
};

/*!
 * \brief Represents a GenAPI float node.
 */
class FloatNode : public Node
{
public:
    FloatNode() = delete;
    ~FloatNode() override = default;
    FloatNode(const FloatNode& other) = delete;
    FloatNode& operator=(const FloatNode& other) = delete;
    FloatNode(FloatNode&& other) = delete;
    FloatNode& operator=(FloatNode&& other) = delete;

    /*!
     * \brief Returns the minimum.
     *
     * \return Minimum
     *
     * \since 1.0
     *
     * \throws InternalErrorException An internal error has occurred.
     */
    double Minimum() const;
    /*!
     * \brief Returns the maximum.
     *
     * \return Maximum
     *
     * \since 1.0
     *
     * \throws InternalErrorException An internal error has occurred.
     */
    double Maximum() const;
    /*!
     * \brief Returns the increment.
     *
     * \return Increment
     *
     * \since 1.0
     *
     * \throws InternalErrorException An internal error has occurred.
     */
    double Increment() const;
    /*!
     * \brief Returns the increment type.
     *
     * \return Increment type
     *
     * \since 1.0
     *
     * \throws InternalErrorException An internal error has occurred.
     */
    NodeIncrementType IncrementType() const;
    /*!
     * \brief Returns the valid values.
     *
     * This function returns a list of valid values. A numeric node can have a list of valid values if it does not have
     * a constant increment.
     *
     * \return Valid values
     *
     * \since 1.0
     *
     * \throws InternalErrorException An internal error has occurred.
     */
    std::vector<double> ValidValues() const;
    /*!
     * \brief Returns the representation.
     *
     * \return Representation
     *
     * \since 1.0
     *
     * \throws InternalErrorException An internal error has occurred.
     */
    NodeRepresentation Representation() const;
    /*!
     * \brief Returns the unit.
     *
     * \return Unit
     *
     * \since 1.0
     *
     * \throws InternalErrorException An internal error has occurred.
     */
    std::string Unit() const;
    /*!
     * \brief Returns the display notation.
     *
     * \return Display notation
     *
     * \since 1.0
     *
     * \throws InternalErrorException An internal error has occurred.
     */
    NodeDisplayNotation DisplayNotation() const;
    /*!
     * \brief Returns the display precision.
     *
     * \return Display precision
     *
     * \since 1.0
     *
     * \throws InternalErrorException An internal error has occurred.
     */
    int64_t DisplayPrecision() const;
    /*!
     * \brief Checks whether the node has a constant increment.
     *
     * \return True, if the node has a constant increment.
     * \return False otherwise.
     *
     * \since 1.0
     *
     * \throws InternalErrorException An internal error has occurred.
     */
    bool HasConstantIncrement() const;

    /*!
     * \brief Returns the value.
     *
     * \param[in] cacheUsePolicy A flag telling whether the value should be read using the internal cache.
     *
     * \return Value
     *
     * \since 1.0
     *
     * \throws InternalErrorException An internal error has occurred.
     */
    double Value(NodeCacheUsePolicy cacheUsePolicy = NodeCacheUsePolicy::UseCache) const;
    /*!
     * \brief Sets the given value.
     *
     * \param[in] value The value to set.
     *
     * \since 1.0
     *
     * \throws InternalErrorException An internal error has occurred.
     */
    void SetValue(double value);

private:
    friend ClassCreator<FloatNode>;
    FloatNode(PEAK_FLOAT_NODE_HANDLE floatNodeHandle, const std::weak_ptr<NodeMap>& parentNodeMap);
    PEAK_FLOAT_NODE_HANDLE m_backendHandle;
};

} /* namespace nodes */
} /* namespace core */
} /* namespace peak */

/* Implementation */
namespace peak
{
namespace core
{
namespace nodes
{

inline std::string ToString(NodeDisplayNotation entry)
{
    std::string entryString;

    if (entry == NodeDisplayNotation::Automatic)
    {
        entryString = "Automatic";
    }
    else if (entry == NodeDisplayNotation::Fixed)
    {
        entryString = "Fixed";
    }
    else if (entry == NodeDisplayNotation::Scientific)
    {
        entryString = "Scientific";
    }

    return entryString;
}

inline FloatNode::FloatNode(PEAK_FLOAT_NODE_HANDLE floatNodeHandle, const std::weak_ptr<NodeMap>& parentNodeMap)
    : Node(QueryNumericFromCInterfaceFunction<PEAK_NODE_HANDLE>([&](PEAK_NODE_HANDLE* nodeHandle) {
        return PEAK_C_ABI_PREFIX PEAK_FloatNode_ToNode(floatNodeHandle, nodeHandle);
    }),
        parentNodeMap)
    , m_backendHandle(floatNodeHandle)
{}

inline double FloatNode::Minimum() const
{
    return QueryNumericFromCInterfaceFunction<double>(
        [&](double* minimum) { return PEAK_C_ABI_PREFIX PEAK_FloatNode_GetMinimum(m_backendHandle, minimum); });
}

inline double FloatNode::Maximum() const
{
    return QueryNumericFromCInterfaceFunction<double>(
        [&](double* maximum) { return PEAK_C_ABI_PREFIX PEAK_FloatNode_GetMaximum(m_backendHandle, maximum); });
}

inline double FloatNode::Increment() const
{
    return QueryNumericFromCInterfaceFunction<double>([&](double* increment) {
        return PEAK_C_ABI_PREFIX PEAK_FloatNode_GetIncrement(m_backendHandle, increment);
    });
}

inline NodeIncrementType FloatNode::IncrementType() const
{
    return static_cast<NodeIncrementType>(QueryNumericFromCInterfaceFunction<PEAK_NODE_INCREMENT_TYPE>(
        [&](PEAK_NODE_INCREMENT_TYPE* incrementType) {
            return PEAK_C_ABI_PREFIX PEAK_FloatNode_GetIncrementType(m_backendHandle, incrementType);
        }));
}

inline std::vector<double> FloatNode::ValidValues() const
{
    return QueryNumericArrayFromCInterfaceFunction<double>([&](double* validValues, size_t* validValuesSize) {
        return PEAK_C_ABI_PREFIX PEAK_FloatNode_GetValidValues(m_backendHandle, validValues, validValuesSize);
    });
}

inline NodeRepresentation FloatNode::Representation() const
{
    return static_cast<NodeRepresentation>(QueryNumericFromCInterfaceFunction<PEAK_NODE_REPRESENTATION>(
        [&](PEAK_NODE_REPRESENTATION* representation) {
            return PEAK_C_ABI_PREFIX PEAK_FloatNode_GetRepresentation(m_backendHandle, representation);
        }));
}

inline std::string FloatNode::Unit() const
{
    return QueryStringFromCInterfaceFunction([&](char* unit, size_t* unitSize) {
        return PEAK_C_ABI_PREFIX PEAK_FloatNode_GetUnit(m_backendHandle, unit, unitSize);
    });
}

inline NodeDisplayNotation FloatNode::DisplayNotation() const
{
    return static_cast<NodeDisplayNotation>(QueryNumericFromCInterfaceFunction<PEAK_NODE_DISPLAY_NOTATION>(
        [&](PEAK_NODE_DISPLAY_NOTATION* displayNotation) {
            return PEAK_C_ABI_PREFIX PEAK_FloatNode_GetDisplayNotation(m_backendHandle, displayNotation);
        }));
}

inline int64_t FloatNode::DisplayPrecision() const
{
    return QueryNumericFromCInterfaceFunction<int64_t>([&](int64_t* displayPrecision) {
        return PEAK_C_ABI_PREFIX PEAK_FloatNode_GetDisplayPrecision(m_backendHandle, displayPrecision);
    });
}

inline bool FloatNode::HasConstantIncrement() const
{
    return QueryNumericFromCInterfaceFunction<PEAK_BOOL8>([&](PEAK_BOOL8* hasConstantIncrement) {
        return PEAK_C_ABI_PREFIX PEAK_FloatNode_GetHasConstantIncrement(m_backendHandle, hasConstantIncrement);
    }) > 0;
}

inline double FloatNode::Value(NodeCacheUsePolicy cacheUsePolicy /* = NodeCacheUsePolicy::UseCache */) const
{
    return QueryNumericFromCInterfaceFunction<double>([&](double* value) {
        return PEAK_C_ABI_PREFIX PEAK_FloatNode_GetValue(
            m_backendHandle, static_cast<PEAK_NODE_CACHE_USE_POLICY>(cacheUsePolicy), value);
    });
}

inline void FloatNode::SetValue(double value)
{
    CallAndCheckCInterfaceFunction(
        [&] { return PEAK_C_ABI_PREFIX PEAK_FloatNode_SetValue(m_backendHandle, value); });
}

} /* namespace nodes */
} /* namespace core */
} /* namespace peak */
