/*!
 * \file    peak_boolean_node.hpp
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


namespace peak
{
namespace core
{
namespace nodes
{

/*!
 * \brief Represents a GenAPI boolean node.
 *
 */
class BooleanNode : public Node
{
public:
    BooleanNode() = delete;
    ~BooleanNode() override = default;
    BooleanNode(const BooleanNode& other) = delete;
    BooleanNode& operator=(const BooleanNode& other) = delete;
    BooleanNode(BooleanNode&& other) = delete;
    BooleanNode& operator=(BooleanNode&& other) = delete;

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
    bool Value(NodeCacheUsePolicy cacheUsePolicy = NodeCacheUsePolicy::UseCache) const;
    /*!
     * \brief Sets the given value.
     *
     * \param[in] value The value to be set.
     *
     * \since 1.0
     *
     * \throws InternalErrorException An internal error has occurred.
     */
    void SetValue(bool value);

private:
    friend ClassCreator<BooleanNode>;
    BooleanNode(PEAK_BOOLEAN_NODE_HANDLE booleanNodeHandle, const std::weak_ptr<NodeMap>& parentNodeMap);
    PEAK_BOOLEAN_NODE_HANDLE m_backendHandle;
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

inline BooleanNode::BooleanNode(
    PEAK_BOOLEAN_NODE_HANDLE booleanNodeHandle, const std::weak_ptr<NodeMap>& parentNodeMap)
    : Node(QueryNumericFromCInterfaceFunction<PEAK_NODE_HANDLE>([&](PEAK_NODE_HANDLE* nodeHandle) {
        return PEAK_C_ABI_PREFIX PEAK_BooleanNode_ToNode(booleanNodeHandle, nodeHandle);
    }),
        parentNodeMap)
    , m_backendHandle(booleanNodeHandle)
{}

inline bool BooleanNode::Value(NodeCacheUsePolicy cacheUsePolicy /* = NodeCacheUsePolicy::UseCache */) const
{
    return QueryNumericFromCInterfaceFunction<PEAK_BOOL8>([&](PEAK_BOOL8* value) {
        return PEAK_C_ABI_PREFIX PEAK_BooleanNode_GetValue(
            m_backendHandle, static_cast<PEAK_NODE_CACHE_USE_POLICY>(cacheUsePolicy), value);
    }) > 0;
}

inline void BooleanNode::SetValue(bool value)
{
    CallAndCheckCInterfaceFunction(
        [&] { return PEAK_C_ABI_PREFIX PEAK_BooleanNode_SetValue(m_backendHandle, value); });
}

} /* namespace nodes */
} /* namespace core */
} /* namespace peak */
