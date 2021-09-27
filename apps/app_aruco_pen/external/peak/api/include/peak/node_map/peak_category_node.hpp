/*!
 * \file    peak_category_node.hpp
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
#include <peak/node_map/peak_node.hpp>

#include <cstddef>
#include <memory>
#include <vector>


namespace peak
{
namespace core
{
namespace nodes
{

/*!
 * \brief Represents a GenAPI category node.
 *
 */
class CategoryNode : public Node
{
public:
    CategoryNode() = delete;
    ~CategoryNode() override = default;
    CategoryNode(const CategoryNode& other) = delete;
    CategoryNode& operator=(const CategoryNode& other) = delete;
    CategoryNode(CategoryNode&& other) = delete;
    CategoryNode& operator=(CategoryNode&& other) = delete;

    /*!
     * \brief Returns the sub nodes.
     *
     * \return Sub nodes
     *
     * \since 1.0
     *
     * \throws InternalErrorException An internal error has occurred.
     */
    std::vector<std::shared_ptr<Node>> SubNodes() const;

private:
    friend ClassCreator<CategoryNode>;
    CategoryNode(PEAK_CATEGORY_NODE_HANDLE categoryNodeHandle, const std::weak_ptr<NodeMap>& parentNodeMap);
    PEAK_CATEGORY_NODE_HANDLE m_backendHandle;
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

inline CategoryNode::CategoryNode(
    PEAK_CATEGORY_NODE_HANDLE categoryNodeHandle, const std::weak_ptr<NodeMap>& parentNodeMap)
    : Node(QueryNumericFromCInterfaceFunction<PEAK_NODE_HANDLE>([&](PEAK_NODE_HANDLE* nodeHandle) {
        return PEAK_C_ABI_PREFIX PEAK_CategoryNode_ToNode(categoryNodeHandle, nodeHandle);
    }),
        parentNodeMap)
    , m_backendHandle(categoryNodeHandle)
{}

inline std::vector<std::shared_ptr<Node>> CategoryNode::SubNodes() const
{
    auto numSubNodes = QueryNumericFromCInterfaceFunction<size_t>([&](size_t* _numSubNodes) {
        return PEAK_C_ABI_PREFIX PEAK_CategoryNode_GetNumSubNodes(m_backendHandle, _numSubNodes);
    });

    std::vector<std::shared_ptr<Node>> subNodes;
    for (size_t x = 0; x < numSubNodes; ++x)
    {
        auto nodeHandle = QueryNumericFromCInterfaceFunction<PEAK_NODE_HANDLE>(
            [&](PEAK_NODE_HANDLE* _nodeHandle) {
                return PEAK_C_ABI_PREFIX PEAK_CategoryNode_GetSubNode(m_backendHandle, x, _nodeHandle);
            });

        auto nodeName = GetNodeName(nodeHandle);

        subNodes.emplace_back(ParentNodeMap()->FindNode(nodeName));
    }

    return subNodes;
}

} /* namespace nodes */
} /* namespace core */
} /* namespace peak */
