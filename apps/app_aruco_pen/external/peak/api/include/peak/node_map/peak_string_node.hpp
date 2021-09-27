/*!
 * \file    peak_string_node.hpp
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


namespace peak
{
namespace core
{
namespace nodes
{

/*!
 * \brief Represents a GenAPI string node.
 *
 */
class StringNode : public Node
{
public:
    StringNode() = delete;
    ~StringNode() override = default;
    StringNode(const StringNode& other) = delete;
    StringNode& operator=(const StringNode& other) = delete;
    StringNode(StringNode&& other) = delete;
    StringNode& operator=(StringNode&& other) = delete;

    /*!
     * \brief Returns the maximum length.
     *
     * \return Maximum length
     *
     * \since 1.0
     *
     * \throws InternalErrorException An internal error has occurred.
     */
    int64_t MaximumLength() const;

    /*! @copydoc FloatNode::Value() */
    std::string Value(NodeCacheUsePolicy cacheUsePolicy = NodeCacheUsePolicy::UseCache) const;
    /*! @copydoc FloatNode::SetValue() */
    void SetValue(const std::string& value);

private:
    friend ClassCreator<StringNode>;
    StringNode(PEAK_STRING_NODE_HANDLE stringNodeHandle, const std::weak_ptr<NodeMap>& parentNodeMap);
    PEAK_STRING_NODE_HANDLE m_backendHandle;
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

inline StringNode::StringNode(PEAK_STRING_NODE_HANDLE stringNodeHandle, const std::weak_ptr<NodeMap>& parentNodeMap)
    : Node(QueryNumericFromCInterfaceFunction<PEAK_NODE_HANDLE>([&](PEAK_NODE_HANDLE* nodeHandle) {
        return PEAK_C_ABI_PREFIX PEAK_StringNode_ToNode(stringNodeHandle, nodeHandle);
    }),
        parentNodeMap)
    , m_backendHandle(stringNodeHandle)
{}

inline int64_t StringNode::MaximumLength() const
{
    return QueryNumericFromCInterfaceFunction<int64_t>([&](int64_t* maximumLength) {
        return PEAK_C_ABI_PREFIX PEAK_StringNode_GetMaximumLength(m_backendHandle, maximumLength);
    });
}

inline std::string StringNode::Value(NodeCacheUsePolicy cacheUsePolicy /* = NodeCacheUsePolicy::UseCache */) const
{
    return QueryStringFromCInterfaceFunction([&](char* value, size_t* valueSize) {
        return PEAK_C_ABI_PREFIX PEAK_StringNode_GetValue(
            m_backendHandle, static_cast<PEAK_NODE_CACHE_USE_POLICY>(cacheUsePolicy), value, valueSize);
    });
}

inline void StringNode::SetValue(const std::string& value)
{
    CallAndCheckCInterfaceFunction([&] {
        return PEAK_C_ABI_PREFIX PEAK_StringNode_SetValue(m_backendHandle, value.c_str(), value.size() + 1);
    });
}

} /* namespace nodes */
} /* namespace core */
} /* namespace peak */
