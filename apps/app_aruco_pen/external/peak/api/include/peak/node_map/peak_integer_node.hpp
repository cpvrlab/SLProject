/*!
 * \file    peak_integer_node.hpp
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
 * \brief Represents a GenAPI integer node.
 *
 */
class IntegerNode : public Node
{
public:
    IntegerNode() = delete;
    ~IntegerNode() override = default;
    IntegerNode(const IntegerNode& other) = delete;
    IntegerNode& operator=(const IntegerNode& other) = delete;
    IntegerNode(IntegerNode&& other) = delete;
    IntegerNode& operator=(IntegerNode&& other) = delete;

    /*! @copydoc FloatNode::Minimum() */
    int64_t Minimum() const;
    /*! @copydoc FloatNode::Maximum() */
    int64_t Maximum() const;
    /*! @copydoc FloatNode::Increment() */
    int64_t Increment() const;
    /*! @copydoc FloatNode::IncrementType() */
    NodeIncrementType IncrementType() const;
    /*! @copydoc FloatNode::ValidValues() */
    std::vector<int64_t> ValidValues() const;
    /*! @copydoc FloatNode::Representation() */
    NodeRepresentation Representation() const;
    /*! @copydoc FloatNode::Unit() */
    std::string Unit() const;

    /*! @copydoc FloatNode::Value() */
    int64_t Value(NodeCacheUsePolicy cacheUsePolicy = NodeCacheUsePolicy::UseCache) const;
    /*! @copydoc FloatNode::SetValue() */
    void SetValue(int64_t value);

private:
    friend ClassCreator<IntegerNode>;
    IntegerNode(PEAK_INTEGER_NODE_HANDLE integerNodeHandle, const std::weak_ptr<NodeMap>& parentNodeMap);
    PEAK_INTEGER_NODE_HANDLE m_backendHandle;
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

inline IntegerNode::IntegerNode(
    PEAK_INTEGER_NODE_HANDLE integerNodeHandle, const std::weak_ptr<NodeMap>& parentNodeMap)
    : Node(QueryNumericFromCInterfaceFunction<PEAK_NODE_HANDLE>([&](PEAK_NODE_HANDLE* nodeHandle) {
        return PEAK_C_ABI_PREFIX PEAK_IntegerNode_ToNode(integerNodeHandle, nodeHandle);
    }),
        parentNodeMap)
    , m_backendHandle(integerNodeHandle)
{}

inline int64_t IntegerNode::Minimum() const
{
    return QueryNumericFromCInterfaceFunction<int64_t>([&](int64_t* minimum) {
        return PEAK_C_ABI_PREFIX PEAK_IntegerNode_GetMinimum(m_backendHandle, minimum);
    });
}

inline int64_t IntegerNode::Maximum() const
{
    return QueryNumericFromCInterfaceFunction<int64_t>([&](int64_t* maximum) {
        return PEAK_C_ABI_PREFIX PEAK_IntegerNode_GetMaximum(m_backendHandle, maximum);
    });
}

inline int64_t IntegerNode::Increment() const
{
    return QueryNumericFromCInterfaceFunction<int64_t>([&](int64_t* increment) {
        return PEAK_C_ABI_PREFIX PEAK_IntegerNode_GetIncrement(m_backendHandle, increment);
    });
}

inline NodeIncrementType IntegerNode::IncrementType() const
{
    return static_cast<NodeIncrementType>(QueryNumericFromCInterfaceFunction<PEAK_NODE_INCREMENT_TYPE>(
        [&](PEAK_NODE_INCREMENT_TYPE* incrementType) {
            return PEAK_C_ABI_PREFIX PEAK_IntegerNode_GetIncrementType(m_backendHandle, incrementType);
        }));
}

inline std::vector<int64_t> IntegerNode::ValidValues() const
{
    return QueryNumericArrayFromCInterfaceFunction<int64_t>([&](int64_t* validValues, size_t* validValuesSize) {
        return PEAK_C_ABI_PREFIX PEAK_IntegerNode_GetValidValues(m_backendHandle, validValues, validValuesSize);
    });
}

inline NodeRepresentation IntegerNode::Representation() const
{
    return static_cast<NodeRepresentation>(QueryNumericFromCInterfaceFunction<PEAK_NODE_REPRESENTATION>(
        [&](PEAK_NODE_REPRESENTATION* representation) {
            return PEAK_C_ABI_PREFIX PEAK_IntegerNode_GetRepresentation(m_backendHandle, representation);
        }));
}

inline std::string IntegerNode::Unit() const
{
    return QueryStringFromCInterfaceFunction([&](char* unit, size_t* unitSize) {
        return PEAK_C_ABI_PREFIX PEAK_IntegerNode_GetUnit(m_backendHandle, unit, unitSize);
    });
}

inline int64_t IntegerNode::Value(NodeCacheUsePolicy cacheUsePolicy /* = NodeCacheUsePolicy::UseCache */) const
{
    return QueryNumericFromCInterfaceFunction<int64_t>([&](int64_t* value) {
        return PEAK_C_ABI_PREFIX PEAK_IntegerNode_GetValue(
            m_backendHandle, static_cast<PEAK_NODE_CACHE_USE_POLICY>(cacheUsePolicy), value);
    });
}

inline void IntegerNode::SetValue(int64_t value)
{
    CallAndCheckCInterfaceFunction(
        [&] { return PEAK_C_ABI_PREFIX PEAK_IntegerNode_SetValue(m_backendHandle, value); });
}

} /* namespace nodes */
} /* namespace core */
} /* namespace peak */
