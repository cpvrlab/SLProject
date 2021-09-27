/*!
 * \file    peak_node_map.hpp
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
#include <peak/event/peak_event.hpp>
#include <peak/generic/peak_init_once.hpp>

#include <type_traits>
#include <unordered_map>
#include <algorithm>
#include <cstdint>
#include <functional>
#include <iterator>
#include <memory>
#include <mutex>
#include <string>
#include <vector>


namespace peak
{
namespace core
{
namespace nodes
{

class BooleanNode;
class CategoryNode;
class CommandNode;
class EnumerationNode;
class EnumerationEntryNode;
class FloatNode;
class IntegerNode;
class Node;
class RegisterNode;
class StringNode;

} // namespace nodes

class Buffer;

/*!
 * \brief Represents a GenAPI node map.
 *
 * This class allows to interact with the nodes of a module.
 *
 */
class NodeMap
    : public InitOnce
    , public std::enable_shared_from_this<NodeMap>
{
public:
    /*! The type of node filter functions. */
    using NodeFilterFunction = std::function<bool(const std::shared_ptr<const nodes::Node>& nodeToFilter)>;

    /*!
     * \brief Holds a recursive lock on the NodeMap.
     *
     * The lock is released on destruction.
     */
    class ScopedNodeMapLock
    {
    public:
        ScopedNodeMapLock() = delete;
        ~ScopedNodeMapLock();
        ScopedNodeMapLock(const ScopedNodeMapLock& other) = delete;
        ScopedNodeMapLock& operator=(const ScopedNodeMapLock& other) = delete;
        ScopedNodeMapLock(ScopedNodeMapLock&& other) = delete;
        ScopedNodeMapLock& operator=(ScopedNodeMapLock&& other) = delete;

    private:
        friend NodeMap;
        friend ClassCreator<ScopedNodeMapLock>;
        explicit ScopedNodeMapLock(const std::shared_ptr<NodeMap>& nodeMap);
        std::shared_ptr<NodeMap> m_nodeMap;
    };

    NodeMap() = delete;
    ~NodeMap() = default;
    NodeMap(const NodeMap& other) = delete;
    NodeMap& operator=(const NodeMap& other) = delete;
    NodeMap(NodeMap&& other) = delete;
    NodeMap& operator=(NodeMap&& other) = delete;

    /*!
     * \brief Checks whether the node map contains a node with the given name.
     *
     * \param[in] name The name of the node to find.
     *
     * \since 1.2
     *
     * \throws InternalErrorException An internal error has occurred.
     */
    bool HasNode(const std::string& name) const;
    /*!
     * \brief Checks whether the node map contains a node with the given name and type.
     *
     * \param[in] name The name of the node to find.
     *
     * \since 1.2
     *
     * \throws InternalErrorException An internal error has occurred.
     */
    template <class NodeType, typename std::enable_if<std::is_base_of<nodes::Node, NodeType>::value, int>::type = 0>
    bool HasNode(const std::string& name)
    {
        PEAK_NODE_HANDLE nodeHandle{ nullptr };
        if (PEAK_C_ABI_PREFIX PEAK_NodeMap_FindNode(m_backendHandle, name.c_str(), name.size() + 1, &nodeHandle)
            != PEAK_RETURN_CODE_SUCCESS)
        {
            return false;
        }

        return std::dynamic_pointer_cast<NodeType>(FindNode(name)) != nullptr;
    }
    /*!
     * \brief Tries to find a node with the given name.
     *
     * \param[in] name The name of the node to find.
     *
     * \return Found node.
     * \return If no node with the given name exists, a NotFoundException is thrown.
     *
     * \since 1.0
     *
     * \throws NotFoundException There is no node with the given name
     * \throws InternalErrorException An internal error has occurred.
     */
    std::shared_ptr<nodes::Node> FindNode(const std::string& name);
    /*!
     * \brief Tries to find a node with the given name and type.
     *
     * \param[in] name The name of the node to find.
     *
     * \return Found node.
     * \return If no node with the given name exists, a NotFoundException is thrown.
     *
     * \since 1.0
     *
     * \throws NotFoundException There is no node with the given name
     * \throws InvalidCastException The node cannot be cast to the given type
     * \throws InternalErrorException An internal error has occurred.
     */
    template <class NodeType, typename std::enable_if<std::is_base_of<nodes::Node, NodeType>::value, int>::type = 0>
    std::shared_ptr<NodeType> FindNode(const std::string& name)
    {
        auto castedNode = std::dynamic_pointer_cast<NodeType>(FindNode(name));
        if (!castedNode)
        {
            throw InvalidCastException("Invalid node cast!");
        }

        return castedNode;
    }
    /*!
     * \brief Invalidates all nodes.
     *
     * \since 1.0
     *
     * \throws InternalErrorException An internal error has occurred.
     */
    void InvalidateNodes();
    /*!
     * \brief Polls all nodes having a polling time.
     *
     * \param[in] elapsedTime_ms The elapsed time since the last poll in milliseconds.
     *
     * \since 1.0
     *
     * \throws InternalErrorException An internal error has occurred.
     */
    void PollNodes(int64_t elapsedTime_ms);
    /*!
     * \brief Returns the nodes.
     *
     * \return Nodes
     *
     * \since 1.0
     *
     * \throws InternalErrorException An internal error has occurred.
     */
    std::vector<std::shared_ptr<nodes::Node>> Nodes() const;
    /*!
     * \brief Checks if the Buffer contains chunks corresponding to the NodeMap.
     *
     * \param[in] buffer The Buffer to check.
     *
     * \since 1.1
     *
     * \throws InternalErrorException An internal error has occurred.
     */
    bool HasBufferSupportedChunks(const std::shared_ptr<Buffer>& buffer) const;
    /*!
     * \brief Updates chunk information in the NodeMap.
     *
     * \param[in] buffer The Buffer to update from.
     *
     * When chunks are active, pass each new buffer to this method to parse the chunks and update the corresponding
     * chunk nodes in the NodeMap.
     *
     * \since 1.1
     *
     * \throws InternalErrorException An internal error has occurred.
     */
    void UpdateChunkNodes(const std::shared_ptr<Buffer>& buffer);
    /*!
     * \brief Checks if the Event contains data corresponding to the NodeMap.
     *
     * \param[in] event The Event to check.
     *
     * \since 1.2
     *
     * \throws InternalErrorException An internal error has occurred.
     */
    bool HasEventSupportedData(const std::unique_ptr<Event>& event) const;
    /*!
     * \brief Updates event information in the NodeMap.
     *
     * \param[in] event The Event to update from.
     *
     * When events are active, pass each new event to this method to parse the event data and update the corresponding
     * nodes in the NodeMap.
     *
     * \since 1.2
     *
     * \throws InvalidArgumentException The given Event does not have supported data
     * \throws InternalErrorException An internal error has occurred.
     */
    void UpdateEventNodes(const std::unique_ptr<Event>& event);
    /*!
     * \brief Stores the values of streamable nodes to the file at the given file path.
     *
     * \param[in] filePath The path of the file to store to.
     *
     * The stored file uses the GenApi persistence file format. It is not recommended to edit files using this format
     * manually unless you are familiar with the GenApi persistence functionality.
     *
     * \since 1.1
     *
     * \throws InvalidArgumentException The given file path is invalid
     * \throws InternalErrorException An internal error has occurred.
     */
    void StoreToFile(const std::string& filePath) const;
    /*!
     * \brief Loads the values of streamable nodes from the file at the given file path.
     *
     * \param[in] filePath The path of the file to load from.
     *
     * The file to load has to use the GenApi persistence file format. It is not recommended to edit files using this
     * format manually unless you are familiar with the GenApi persistence functionality.
     *
     * \since 1.1
     *
     * \throws InvalidArgumentException The given file path is invalid
     * \throws InternalErrorException An internal error has occurred.
     */
    void LoadFromFile(const std::string& filePath);

    /*!
     * \brief Locks a recursive mutex on the NodeMap.
     *
     * Use this to synchronize NodeMap access from multiple threads.
     *
     * \note Each individual access is already protected by this mutex, so the nodemap doesn't need to
     *       be locked for those. But often, nodemap access consists of multiple calls, e.g.: First,
     *       (1.) change a selector value (e.g. set "GainSelector" node to "DigitalRed"), then (2.)
     *       access a selected node (e.g. set value in "Gain" node). If a different thread changes the
     *       selector value between (1.) and (2.), (2.) will access the wrong selected
     *       node. These kinds of calls should be protected by this lock.
     *
     * \return A ScopedNodeMapLock, which holds the lock and releases it on destruction.
     *
     * \since 1.2
     *
     * \throws InternalErrorException An internal error has occurred.
     */
    std::unique_ptr<ScopedNodeMapLock> Lock();

private:
    std::shared_ptr<nodes::Node> NodeFromHandle(PEAK_NODE_HANDLE nodeHandle) const;

    friend ClassCreator<NodeMap>;
    explicit NodeMap(PEAK_NODE_MAP_HANDLE nodeMapHandle);
    PEAK_NODE_MAP_HANDLE m_backendHandle;

    void Initialize() const override;
    mutable std::vector<std::shared_ptr<nodes::Node>> m_nodes;
    mutable std::unordered_map<std::string, std::shared_ptr<nodes::Node>> m_nodesByName;
};

} /* namespace core */
} /* namespace peak */

#include <peak/node_map/peak_boolean_node.hpp>
#include <peak/node_map/peak_category_node.hpp>
#include <peak/node_map/peak_command_node.hpp>
#include <peak/node_map/peak_enumeration_entry_node.hpp>
#include <peak/node_map/peak_enumeration_node.hpp>
#include <peak/node_map/peak_float_node.hpp>
#include <peak/node_map/peak_integer_node.hpp>
#include <peak/node_map/peak_node.hpp>
#include <peak/node_map/peak_register_node.hpp>
#include <peak/node_map/peak_string_node.hpp>


/* Implementation */
namespace peak
{
namespace core
{

inline NodeMap::NodeMap(PEAK_NODE_MAP_HANDLE nodeMapHandle)
    : m_backendHandle(nodeMapHandle)
    , m_nodes()
    , m_nodesByName()
{}

inline bool NodeMap::HasNode(const std::string& name) const
{
    return QueryNumericFromCInterfaceFunction<PEAK_BOOL8>([&](PEAK_BOOL8* hasNode) {
        return PEAK_C_ABI_PREFIX PEAK_NodeMap_GetHasNode(m_backendHandle, name.c_str(), name.size() + 1, hasNode);
    }) > 0;
}

inline std::shared_ptr<nodes::Node> NodeMap::FindNode(const std::string& name)
{
    InitializeIfNecessary();

    (void)QueryNumericFromCInterfaceFunction<PEAK_NODE_HANDLE>([&](PEAK_NODE_HANDLE* nodeHandle) {
        return PEAK_C_ABI_PREFIX PEAK_NodeMap_FindNode(
            m_backendHandle, name.c_str(), name.size() + 1, nodeHandle);
    });

    return m_nodesByName.at(name);
}

inline void NodeMap::InvalidateNodes()
{
    CallAndCheckCInterfaceFunction(
        [&] { return PEAK_C_ABI_PREFIX PEAK_NodeMap_InvalidateNodes(m_backendHandle); });
}

inline void NodeMap::PollNodes(int64_t elapsedTime_ms)
{
    CallAndCheckCInterfaceFunction(
        [&] { return PEAK_C_ABI_PREFIX PEAK_NodeMap_PollNodes(m_backendHandle, elapsedTime_ms); });
}

inline std::vector<std::shared_ptr<nodes::Node>> NodeMap::Nodes() const
{
    InitializeIfNecessary();

    return m_nodes;
}

inline bool NodeMap::HasEventSupportedData(const std::unique_ptr<Event>& event) const
{
    return QueryNumericFromCInterfaceFunction<PEAK_BOOL8>([&](PEAK_BOOL8* hasSupportedData) {
        return PEAK_C_ABI_PREFIX PEAK_NodeMap_GetHasEventSupportedData(
            m_backendHandle, event->m_backendHandle, hasSupportedData);
    }) > 0;
}

inline void NodeMap::UpdateEventNodes(const std::unique_ptr<Event>& event)
{
    CallAndCheckCInterfaceFunction(
        [&] { return PEAK_C_ABI_PREFIX PEAK_NodeMap_UpdateEventNodes(m_backendHandle, event->m_backendHandle); });
}

inline void NodeMap::StoreToFile(const std::string& filePath) const
{
    CallAndCheckCInterfaceFunction([&] {
        return PEAK_C_ABI_PREFIX PEAK_NodeMap_StoreToFile(m_backendHandle, filePath.c_str(), filePath.size() + 1);
    });
}

inline void NodeMap::LoadFromFile(const std::string& filePath)
{
    InitializeIfNecessary();

    CallAndCheckCInterfaceFunction([&] {
        return PEAK_C_ABI_PREFIX PEAK_NodeMap_LoadFromFile(
            m_backendHandle, filePath.c_str(), filePath.size() + 1);
    });
}

inline NodeMap::ScopedNodeMapLock::ScopedNodeMapLock(const std::shared_ptr<NodeMap>& nodeMap)
    : m_nodeMap(nodeMap)
{
    CallAndCheckCInterfaceFunction(
        [&] { return PEAK_C_ABI_PREFIX PEAK_NodeMap_Lock(m_nodeMap->m_backendHandle); });
}
inline NodeMap::ScopedNodeMapLock::~ScopedNodeMapLock()
{
    CallAndCheckCInterfaceFunction(
        [&] { return PEAK_C_ABI_PREFIX PEAK_NodeMap_Unlock(m_nodeMap->m_backendHandle); });
}

inline std::unique_ptr<NodeMap::ScopedNodeMapLock> NodeMap::Lock()
{
    return std::make_unique<ClassCreator<ScopedNodeMapLock>>(shared_from_this());
}

inline std::shared_ptr<nodes::Node> NodeMap::NodeFromHandle(PEAK_NODE_HANDLE nodeHandle) const
{
    auto nodeType = QueryNumericFromCInterfaceFunction<PEAK_NODE_TYPE>(
        [&](PEAK_NODE_TYPE* _nodeType) { return PEAK_C_ABI_PREFIX PEAK_Node_GetType(nodeHandle, _nodeType); });

    switch (nodeType)
    {
    case PEAK_NODE_TYPE_INTEGER: {
        auto integerNodeHandle = QueryNumericFromCInterfaceFunction<PEAK_INTEGER_NODE_HANDLE>(
            [&](PEAK_INTEGER_NODE_HANDLE* _integerNodeHandle) {
                return PEAK_C_ABI_PREFIX PEAK_Node_ToIntegerNode(nodeHandle, _integerNodeHandle);
            });

        return std::make_shared<ClassCreator<nodes::IntegerNode>>(
            integerNodeHandle, std::const_pointer_cast<core::NodeMap>(shared_from_this()));
    }
    case PEAK_NODE_TYPE_BOOLEAN: {
        auto booleanNodeHandle = QueryNumericFromCInterfaceFunction<PEAK_BOOLEAN_NODE_HANDLE>(
            [&](PEAK_BOOLEAN_NODE_HANDLE* _booleanNodeHandle) {
                return PEAK_C_ABI_PREFIX PEAK_Node_ToBooleanNode(nodeHandle, _booleanNodeHandle);
            });

        return std::make_shared<ClassCreator<nodes::BooleanNode>>(
            booleanNodeHandle, std::const_pointer_cast<core::NodeMap>(shared_from_this()));
    }
    case PEAK_NODE_TYPE_COMMAND: {
        auto commandNodeHandle = QueryNumericFromCInterfaceFunction<PEAK_COMMAND_NODE_HANDLE>(
            [&](PEAK_COMMAND_NODE_HANDLE* _commandNodeHandle) {
                return PEAK_C_ABI_PREFIX PEAK_Node_ToCommandNode(nodeHandle, _commandNodeHandle);
            });

        return std::make_shared<ClassCreator<nodes::CommandNode>>(
            commandNodeHandle, std::const_pointer_cast<core::NodeMap>(shared_from_this()));
    }
    case PEAK_NODE_TYPE_FLOAT: {
        auto floatNodeHandle = QueryNumericFromCInterfaceFunction<PEAK_FLOAT_NODE_HANDLE>(
            [&](PEAK_FLOAT_NODE_HANDLE* _floatNodeHandle) {
                return PEAK_C_ABI_PREFIX PEAK_Node_ToFloatNode(nodeHandle, _floatNodeHandle);
            });

        return std::make_shared<ClassCreator<nodes::FloatNode>>(
            floatNodeHandle, std::const_pointer_cast<core::NodeMap>(shared_from_this()));
    }
    case PEAK_NODE_TYPE_STRING: {
        auto stringNodeHandle = QueryNumericFromCInterfaceFunction<PEAK_STRING_NODE_HANDLE>(
            [&](PEAK_STRING_NODE_HANDLE* _stringNodeHandle) {
                return PEAK_C_ABI_PREFIX PEAK_Node_ToStringNode(nodeHandle, _stringNodeHandle);
            });

        return std::make_shared<ClassCreator<nodes::StringNode>>(
            stringNodeHandle, std::const_pointer_cast<core::NodeMap>(shared_from_this()));
    }
    case PEAK_NODE_TYPE_REGISTER: {
        auto registerNodeHandle = QueryNumericFromCInterfaceFunction<PEAK_REGISTER_NODE_HANDLE>(
            [&](PEAK_REGISTER_NODE_HANDLE* _registerNodeHandle) {
                return PEAK_C_ABI_PREFIX PEAK_Node_ToRegisterNode(nodeHandle, _registerNodeHandle);
            });

        return std::make_shared<ClassCreator<nodes::RegisterNode>>(
            registerNodeHandle, std::const_pointer_cast<core::NodeMap>(shared_from_this()));
    }
    case PEAK_NODE_TYPE_CATEGORY: {
        auto categoryNodeHandle = QueryNumericFromCInterfaceFunction<PEAK_CATEGORY_NODE_HANDLE>(
            [&](PEAK_CATEGORY_NODE_HANDLE* _categoryNodeHandle) {
                return PEAK_C_ABI_PREFIX PEAK_Node_ToCategoryNode(nodeHandle, _categoryNodeHandle);
            });

        return std::make_shared<ClassCreator<nodes::CategoryNode>>(
            categoryNodeHandle, std::const_pointer_cast<core::NodeMap>(shared_from_this()));
    }
    case PEAK_NODE_TYPE_ENUMERATION: {
        auto enumerationNodeHandle = QueryNumericFromCInterfaceFunction<PEAK_ENUMERATION_NODE_HANDLE>(
            [&](PEAK_ENUMERATION_NODE_HANDLE* _enumerationNodeHandle) {
                return PEAK_C_ABI_PREFIX PEAK_Node_ToEnumerationNode(nodeHandle, _enumerationNodeHandle);
            });

        return std::make_shared<ClassCreator<nodes::EnumerationNode>>(
            enumerationNodeHandle, std::const_pointer_cast<core::NodeMap>(shared_from_this()));
    }
    case PEAK_NODE_TYPE_ENUMERATION_ENTRY: {
        auto enumerationEntryNodeHandle = QueryNumericFromCInterfaceFunction<PEAK_ENUMERATION_ENTRY_NODE_HANDLE>(
            [&](PEAK_ENUMERATION_ENTRY_NODE_HANDLE* _enumerationEntryNodeHandle) {
                return PEAK_C_ABI_PREFIX PEAK_Node_ToEnumerationEntryNode(
                    nodeHandle, _enumerationEntryNodeHandle);
            });

        return std::make_shared<ClassCreator<nodes::EnumerationEntryNode>>(
            enumerationEntryNodeHandle, std::const_pointer_cast<core::NodeMap>(shared_from_this()));
    }
    default: {
        return std::make_shared<ClassCreator<nodes::Node>>(
            nodeHandle, std::const_pointer_cast<core::NodeMap>(shared_from_this()));
    }
    }
}

inline void NodeMap::Initialize() const
{
    auto numNodes = QueryNumericFromCInterfaceFunction<size_t>([&](size_t* _numNodes) {
        return PEAK_C_ABI_PREFIX PEAK_NodeMap_GetNumNodes(m_backendHandle, _numNodes);
    });

    for (size_t x = 0; x < numNodes; ++x)
    {
        auto nodeHandle = QueryNumericFromCInterfaceFunction<PEAK_NODE_HANDLE>(
            [&](PEAK_NODE_HANDLE* _nodeHandle) {
                return PEAK_C_ABI_PREFIX PEAK_NodeMap_GetNode(m_backendHandle, x, _nodeHandle);
            });

        auto node = NodeFromHandle(nodeHandle);

        m_nodes.emplace_back(node);
        m_nodesByName.emplace(node->Name(), node);
    }
}

} /* namespace core */
} /* namespace peak */
