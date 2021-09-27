/*!
 * \file    peak_node.hpp
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
#include <peak/generic/peak_t_callback_manager.hpp>

#include <type_traits>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <vector>


namespace peak
{
namespace core
{

class NodeMap;

/*!
 * \brief The "nodes" namespace contains all GenAPI node types.
 */
namespace nodes
{

/*!
 * The node's current access status, i.e. the \<AccessMode\> element.
 * See GenAPI Access Mode.
 */
enum class NodeAccessStatus
{
    NotImplemented,
    NotAvailable,
    WriteOnly,
    ReadOnly,
    ReadWrite
};

/*!
 * The node's caching mode, i.e. the \<Cacheable\> element.
 * See GenAPI Caching.
 */
enum class NodeCachingMode
{
    /*!
     * NoCache means all values are read directly from the device.
     */
    NoCache,
    /*!
     * WriteThrough means that a value written to the camera is written to the cache as well.
     */
    WriteThrough,
    /*!
     * WriteAround means that only read values are written to the cache.
     */
    WriteAround
};

/*!
 * The node's namespace, i.e. the \<node NameSpace=""\> attribute.
 * See GenAPI Node.
 */
enum class NodeNamespace
{
    Custom,
    Standard
};

/*!
 * Different node types
 */
enum class NodeType
{
    Integer,
    Boolean,
    Command,
    Float,
    String,
    Register,
    Category,
    Enumeration,
    EnumerationEntry
};

/*!
 * Visibility of the node
 */
enum class NodeVisibility
{
    Beginner,
    Expert,
    Guru,
    Invisible
};

//! \cond
struct NodeChangedCallbackContainer;
//! \endcond

/*!
 * \brief Represents a GenAPI node.
 *
 * This class allows to interact with a single node of a module. It is the base class for all nodes a module can
 * have.
 *
 */
class Node : public std::enable_shared_from_this<Node>
{
public:
    /*! The type of changed callbacks. */
    using ChangedCallback = std::function<void(const std::shared_ptr<Node>& changedNode)>;
    /*! The type of changed callback handles. */
    using ChangedCallbackHandle = ChangedCallback*;

    Node() = delete;
    virtual ~Node();
    Node(const Node& other) = delete;
    Node& operator=(const Node& other) = delete;
    Node(Node&& other) = delete;
    Node& operator=(Node&& other) = delete;

    /*!
     * \brief Returns the name.
     *
     * \return Name
     *
     * \since 1.0
     *
     * \throws InternalErrorException An internal error has occurred.
     */
    std::string Name() const;
    /*!
     * \brief Returns the display name.
     *
     * \return Display name
     *
     * \since 1.0
     *
     * \throws InternalErrorException An internal error has occurred.
     */
    std::string DisplayName() const;
    /*!
     * \brief Returns the namespace the node belongs to.
     *
     * \return Namespace the node belongs to
     *
     * \since 1.0
     *
     * \throws InternalErrorException An internal error has occurred.
     */
    NodeNamespace Namespace() const;
    /*!
     * \brief Returns the visibility.
     *
     * \return Visibility
     *
     * \since 1.0
     *
     * \throws InternalErrorException An internal error has occurred.
     */
    NodeVisibility Visibility() const;
    /*!
     * \brief Returns the access status.
     *
     * \return Access status
     *
     * \since 1.0
     *
     * \throws InternalErrorException An internal error has occurred.
     */
    NodeAccessStatus AccessStatus() const;
    /*!
     * \brief Checks whether the node is cacheable.
     *
     * \return True, if the node is cacheable.
     * \return False otherwise.
     *
     * \since 1.0
     *
     * \throws InternalErrorException An internal error has occurred.
     */
    bool IsCacheable() const;
    /*!
     * \brief Checks whether the node's access status is cacheable.
     *
     * \return True, if the node's access status is cacheable.
     * \return False otherwise.
     *
     * \since 1.0
     *
     * \throws InternalErrorException An internal error has occurred.
     */
    bool IsAccessStatusCacheable() const;
    /*!
     * \brief Checks whether the node is streamable.
     *
     * \return True, if the node is streamable.
     * \return False otherwise.
     *
     * \since 1.0
     *
     * \throws InternalErrorException An internal error has occurred.
     */
    bool IsStreamable() const;
    /*!
     * \brief Checks whether the node is deprecated.
     *
     * \return True, if the node is deprecated.
     * \return False otherwise.
     *
     * \since 1.0
     *
     * \throws InternalErrorException An internal error has occurred.
     */
    bool IsDeprecated() const;
    /*!
     * \brief Checks whether the node is a feature, i.e. it can be reached via
     * category nodes from a category node named "Root".
     *
     * \return True, if the node is a feature.
     * \return False otherwise.
     *
     * \since 1.2
     *
     * \throws InternalErrorException An internal error has occurred.
     */
    bool IsFeature() const;
    /*!
     * \brief Returns the caching mode.
     *
     * \return Caching mode
     *
     * \since 1.0
     *
     * \throws InternalErrorException An internal error has occurred.
     */
    NodeCachingMode CachingMode() const;
    /*!
     * \brief Returns the polling time.
     *
     * \return Polling time in milliseconds
     *
     * \since 1.0
     *
     * \throws InternalErrorException An internal error has occurred.
     */
    int64_t PollingTime() const;
    /*!
     * \brief Returns the tool tip.
     *
     * \return Tool tip
     *
     * \since 1.0
     *
     * \throws InternalErrorException An internal error has occurred.
     */
    std::string ToolTip() const;
    /*!
     * \brief Returns the description.
     *
     * \return Description
     *
     * \since 1.0
     *
     * \throws InternalErrorException An internal error has occurred.
     */
    std::string Description() const;
    /*!
     * \brief Returns the type.
     *
     * The returned type is necessary to know the type the node has to be casted to if you want to access the full
     * functionality of the node.
     *
     * \return Type
     *
     * \since 1.0
     *
     * \throws InternalErrorException An internal error has occurred.
     */
    NodeType Type() const;
    /*!
     * \brief Returns the parent node map.
     *
     * \return Parent node map
     *
     * \since 1.0
     *
     * \throws InternalErrorException An internal error has occurred.
     */
    std::shared_ptr<NodeMap> ParentNodeMap() const;

    /*!
     * \brief Tries to find an invalidated node with the given name.
     *
     * \param[in] name The name of the invalidated node to find.
     *
     * \return Found node.
     * \return If no invalidated node with the given name exists, a NotFoundException is thrown.
     *
     * \since 1.0
     *
     * \throws NotFoundException There is no invalidated node with the given name
     * \throws InternalErrorException An internal error has occurred.
     */
    std::shared_ptr<Node> FindInvalidatedNode(const std::string& name) const;
    /*!
     * \brief Tries to find an invalidated node with the given name and type.
     *
     * \param[in] name The name of the invalidated node to find.
     *
     * \return Found node.
     * \return If no invalidated node with the given name exists, a NotFoundException is thrown.
     *
     * \since 1.0
     *
     * \throws NotFoundException There is no invalidated node with the given name
     * \throws InvalidCastException The found node cannot be cast to the given type
     * \throws InternalErrorException An internal error has occurred.
     */
    template <class NodeType, typename std::enable_if<std::is_base_of<Node, NodeType>::value, int>::type = 0>
    std::shared_ptr<NodeType> FindInvalidatedNode(const std::string& name) const
    {
        auto castedNode = std::dynamic_pointer_cast<NodeType>(FindInvalidatedNode(name));
        if (!castedNode)
        {
            throw InvalidCastException("Invalid node cast!");
        }

        return castedNode;
    }
    /*!
     * \brief Returns the invalidated nodes.
     *
     * \return Invalidated nodes
     *
     * \since 1.0
     *
     * \throws InternalErrorException An internal error has occurred.
     */
    std::vector<std::shared_ptr<Node>> InvalidatedNodes() const;
    /*!
     * \brief Tries to find an invalidating node with the given name.
     *
     * \param[in] name The name of the invalidating node to find.
     *
     * \return Found node.
     * \return If no invalidating node with the given name exists, a NotFoundException is thrown.
     *
     * \since 1.0
     *
     * \throws NotFoundException There is no invalidating node with the given name
     * \throws InternalErrorException An internal error has occurred.
     */
    std::shared_ptr<Node> FindInvalidatingNode(const std::string& name) const;
    /*!
     * \brief Tries to find an invalidating node with the given name and type.
     *
     * \param[in] name The name of the invalidating node to find.
     *
     * \return Found node.
     * \return If no invalidating node with the given name exists, a NotFoundException is thrown.
     *
     * \since 1.0
     *
     * \throws NotFoundException There is no invalidating node with the given name
     * \throws InvalidCastException The found node cannot be cast to the given type
     * \throws InternalErrorException An internal error has occurred.
     */
    template <class NodeType, typename std::enable_if<std::is_base_of<Node, NodeType>::value, int>::type = 0>
    std::shared_ptr<NodeType> FindInvalidatingNode(const std::string& name) const
    {
        auto castedNode = std::dynamic_pointer_cast<NodeType>(FindInvalidatingNode(name));
        if (!castedNode)
        {
            throw InvalidCastException("Invalid node cast!");
        }

        return castedNode;
    }
    /*!
     * \brief Returns the invalidating nodes.
     *
     * \return Invalidating nodes
     *
     * \since 1.0
     *
     * \throws InternalErrorException An internal error has occurred.
     */
    std::vector<std::shared_ptr<Node>> InvalidatingNodes() const;
    /*!
     * \brief Tries to find a selected node with the given name.
     *
     * \param[in] name The name of the selected node to find.
     *
     * \return Found node.
     * \return If no selected node with the given name exists, a NotFoundException is thrown.
     *
     * \since 1.0
     *
     * \throws NotFoundException There is no selected node with the given name
     * \throws InternalErrorException An internal error has occurred.
     */
    std::shared_ptr<Node> FindSelectedNode(const std::string& name) const;
    /*!
     * \brief Tries to find a selected node with the given name and type.
     *
     * \param[in] name The name of the selected node to find.
     *
     * \return Found node.
     * \return If no selected node with the given name exists, a NotFoundException is thrown.
     *
     * \since 1.0
     *
     * \throws NotFoundException There is no selected node with the given name
     * \throws InvalidCastException The found node cannot be cast to the given type
     * \throws InternalErrorException An internal error has occurred.
     */
    template <class NodeType, typename std::enable_if<std::is_base_of<Node, NodeType>::value, int>::type = 0>
    std::shared_ptr<NodeType> FindSelectedNode(const std::string& name) const
    {
        auto castedNode = std::dynamic_pointer_cast<NodeType>(FindSelectedNode(name));
        if (!castedNode)
        {
            throw InvalidCastException("Invalid node cast!");
        }

        return castedNode;
    }
    /*!
     * \brief Returns the selected nodes.
     *
     * \return Selected nodes
     *
     * \since 1.0
     *
     * \throws InternalErrorException An internal error has occurred.
     */
    std::vector<std::shared_ptr<Node>> SelectedNodes() const;
    /*!
     * \brief Tries to find a selecting node with the given name.
     *
     * \param[in] name The name of the selecting node to find.
     *
     * \return Found node.
     * \return If no selecting node with the given name exists, a NotFoundException is thrown.
     *
     * \since 1.0
     *
     * \throws NotFoundException There is no selecting node with the given name
     * \throws InternalErrorException An internal error has occurred.
     */
    std::shared_ptr<Node> FindSelectingNode(const std::string& name) const;
    /*!
     * \brief Tries to find a selecting node with the given name and type.
     *
     * \param[in] name The name of the selecting node to find.
     *
     * \return Found node.
     * \return If no selecting node with the given name exists, a NotFoundException is thrown.
     *
     * \since 1.0
     *
     * \throws NotFoundException There is no selecting node with the given name
     * \throws InvalidCastException The found node cannot be cast to the given type
     * \throws InternalErrorException An internal error has occurred.
     */
    template <class NodeType, typename std::enable_if<std::is_base_of<Node, NodeType>::value, int>::type = 0>
    std::shared_ptr<NodeType> FindSelectingNode(const std::string& name) const
    {
        auto castedNode = std::dynamic_pointer_cast<NodeType>(FindSelectingNode(name));
        if (!castedNode)
        {
            throw InvalidCastException("Invalid node cast!");
        }

        return castedNode;
    }
    /*!
     * \brief Returns the selecting nodes.
     *
     * \return Selecting nodes
     *
     * \since 1.0
     *
     * \throws InternalErrorException An internal error has occurred.
     */
    std::vector<std::shared_ptr<Node>> SelectingNodes() const;

    /*!
     * \brief Registers a callback for signaling a change to the node.
     *
     * This function registers a callback which gets called every time the node changes. Pass the callback
     * handle returned by this function to UnregisterChangedCallback() to unregister the callback.
     *
     * \param[in] callback The callback to call if the node has changed.
     *
     * \return Callback handle
     *
     * \since 1.0
     *
     * \throws InternalErrorException An internal error has occurred.
     */
    ChangedCallbackHandle RegisterChangedCallback(const ChangedCallback& callback);
    /*!
     * \brief Unregisters a changed callback.
     *
     * This function unregisters a changed callback by taking its handle.
     *
     * \param[in] callbackHandle The handle of the callback to unregister.
     *
     * \since 1.0
     *
     * \throws InternalErrorException An internal error has occurred.
     */
    void UnregisterChangedCallback(ChangedCallbackHandle callbackHandle);

protected:
    //! \cond
    Node(PEAK_NODE_HANDLE nodeHandle, const std::weak_ptr<peak::core::NodeMap>& parentNodeMap);
    //! \endcond
private:
    struct NodeChangedCallbackContainer
    {
        std::shared_ptr<Node> _Node;
        Node::ChangedCallback Callback;
    };
    static void PEAK_CALL_CONV ChangedCallbackCWrapper(PEAK_NODE_HANDLE, void* context);

    friend ClassCreator<Node>;
    PEAK_NODE_HANDLE m_backendHandle;

    std::weak_ptr<NodeMap> m_parentNodeMap;

    std::unique_ptr<TCallbackManager<PEAK_NODE_CHANGED_CALLBACK_HANDLE, NodeChangedCallbackContainer>>
        m_changedCallbackManager;
};

inline std::string GetNodeName(PEAK_NODE_HANDLE nodeHandle)
{
    return QueryStringFromCInterfaceFunction([&](char* nodeName, size_t* nodeNameSize) {
        return PEAK_C_ABI_PREFIX PEAK_Node_GetName(nodeHandle, nodeName, nodeNameSize);
    });
}

} /* namespace nodes */
} /* namespace core */
} /* namespace peak */

#include <peak/node_map/peak_node_map.hpp>

/* Implementation */
namespace peak
{
namespace core
{
namespace nodes
{

inline std::string ToString(NodeAccessStatus entry)
{
    std::string entryString;

    if (entry == NodeAccessStatus::NotImplemented)
    {
        entryString = "NotImplemented";
    }
    else if (entry == NodeAccessStatus::NotAvailable)
    {
        entryString = "NotAvailable";
    }
    else if (entry == NodeAccessStatus::WriteOnly)
    {
        entryString = "WriteOnly";
    }
    else if (entry == NodeAccessStatus::ReadOnly)
    {
        entryString = "ReadOnly";
    }
    else if (entry == NodeAccessStatus::ReadWrite)
    {
        entryString = "ReadWrite";
    }

    return entryString;
}

inline std::string ToString(NodeCachingMode entry)
{
    std::string entryString;

    if (entry == NodeCachingMode::NoCache)
    {
        entryString = "NoCache";
    }
    else if (entry == NodeCachingMode::WriteThrough)
    {
        entryString = "WriteThrough";
    }
    else if (entry == NodeCachingMode::WriteAround)
    {
        entryString = "WriteAround";
    }

    return entryString;
}

inline std::string ToString(NodeNamespace entry)
{
    std::string entryString;

    if (entry == NodeNamespace::Custom)
    {
        entryString = "Custom";
    }
    else if (entry == NodeNamespace::Standard)
    {
        entryString = "Standard";
    }

    return entryString;
}

inline std::string ToString(NodeType entry)
{
    std::string entryString;

    if (entry == NodeType::Integer)
    {
        entryString = "Integer";
    }
    else if (entry == NodeType::Boolean)
    {
        entryString = "Boolean";
    }
    else if (entry == NodeType::Command)
    {
        entryString = "Command";
    }
    else if (entry == NodeType::Float)
    {
        entryString = "Float";
    }
    else if (entry == NodeType::String)
    {
        entryString = "String";
    }
    else if (entry == NodeType::Register)
    {
        entryString = "Register";
    }
    else if (entry == NodeType::Category)
    {
        entryString = "Category";
    }
    else if (entry == NodeType::Enumeration)
    {
        entryString = "Enumeration";
    }
    else if (entry == NodeType::EnumerationEntry)
    {
        entryString = "EnumerationEntry";
    }

    return entryString;
}

inline std::string ToString(NodeVisibility entry)
{
    std::string entryString;

    if (entry == NodeVisibility::Beginner)
    {
        entryString = "Beginner";
    }
    else if (entry == NodeVisibility::Expert)
    {
        entryString = "Expert";
    }
    else if (entry == NodeVisibility::Guru)
    {
        entryString = "Guru";
    }
    else if (entry == NodeVisibility::Invisible)
    {
        entryString = "Invisible";
    }

    return entryString;
}

//! \cond
inline Node::Node(PEAK_NODE_HANDLE nodeHandle, const std::weak_ptr<peak::core::NodeMap>& parentNodeMap)
    : m_backendHandle(nodeHandle)
    , m_parentNodeMap(parentNodeMap)
    , m_changedCallbackManager()
{
    m_changedCallbackManager =
        std::make_unique<TCallbackManager<PEAK_NODE_CHANGED_CALLBACK_HANDLE, NodeChangedCallbackContainer>>(
            [&](void* callbackContext) {
                return QueryNumericFromCInterfaceFunction<PEAK_NODE_CHANGED_CALLBACK_HANDLE>(
                    [&](PEAK_NODE_CHANGED_CALLBACK_HANDLE* nodeChangedCallbackHandle) {
                        return PEAK_C_ABI_PREFIX PEAK_Node_RegisterChangedCallback(
                            m_backendHandle, ChangedCallbackCWrapper, callbackContext, nodeChangedCallbackHandle);
                    });
            },
            [&](PEAK_NODE_CHANGED_CALLBACK_HANDLE callbackHandle) {
                CallAndCheckCInterfaceFunction([&] {
                    return PEAK_C_ABI_PREFIX PEAK_Node_UnregisterChangedCallback(m_backendHandle, callbackHandle);
                });
            });
}
//! \endcond

inline Node::~Node()
{
    m_changedCallbackManager->UnregisterAllCallbacks();
}

inline std::string Node::Name() const
{
    return GetNodeName(m_backendHandle);
}

inline std::string Node::DisplayName() const
{
    return QueryStringFromCInterfaceFunction([&](char* displayName, size_t* displayNameSize) {
        return PEAK_C_ABI_PREFIX PEAK_Node_GetDisplayName(m_backendHandle, displayName, displayNameSize);
    });
}

inline NodeNamespace Node::Namespace() const
{
    return static_cast<NodeNamespace>(
        QueryNumericFromCInterfaceFunction<PEAK_NODE_NAMESPACE>([&](PEAK_NODE_NAMESPACE* _namespace) {
            return PEAK_C_ABI_PREFIX PEAK_Node_GetNamespace(m_backendHandle, _namespace);
        }));
}

inline NodeVisibility Node::Visibility() const
{
    return static_cast<NodeVisibility>(
        QueryNumericFromCInterfaceFunction<PEAK_NODE_VISIBILITY>([&](PEAK_NODE_VISIBILITY* visibility) {
            return PEAK_C_ABI_PREFIX PEAK_Node_GetVisibility(m_backendHandle, visibility);
        }));
}

inline NodeAccessStatus Node::AccessStatus() const
{
    return static_cast<NodeAccessStatus>(
        QueryNumericFromCInterfaceFunction<PEAK_NODE_ACCESS_STATUS>([&](PEAK_NODE_ACCESS_STATUS* accessStatus) {
            return PEAK_C_ABI_PREFIX PEAK_Node_GetAccessStatus(m_backendHandle, accessStatus);
        }));
}

inline bool Node::IsCacheable() const
{
    return QueryNumericFromCInterfaceFunction<PEAK_BOOL8>([&](PEAK_BOOL8* isCacheable) {
        return PEAK_C_ABI_PREFIX PEAK_Node_GetIsCacheable(m_backendHandle, isCacheable);
    }) > 0;
}

inline bool Node::IsAccessStatusCacheable() const
{
    return QueryNumericFromCInterfaceFunction<PEAK_BOOL8>([&](PEAK_BOOL8* isAccessStatusCacheable) {
        return PEAK_C_ABI_PREFIX PEAK_Node_GetIsAccessStatusCacheable(m_backendHandle, isAccessStatusCacheable);
    }) > 0;
}

inline bool Node::IsStreamable() const
{
    return QueryNumericFromCInterfaceFunction<PEAK_BOOL8>([&](PEAK_BOOL8* isStreamable) {
        return PEAK_C_ABI_PREFIX PEAK_Node_GetIsStreamable(m_backendHandle, isStreamable);
    }) > 0;
}

inline bool Node::IsDeprecated() const
{
    return QueryNumericFromCInterfaceFunction<PEAK_BOOL8>([&](PEAK_BOOL8* isDeprecated) {
        return PEAK_C_ABI_PREFIX PEAK_Node_GetIsDeprecated(m_backendHandle, isDeprecated);
    }) > 0;
}

inline bool Node::IsFeature() const
{
    return QueryNumericFromCInterfaceFunction<PEAK_BOOL8>([&](PEAK_BOOL8* isFeature) {
        return PEAK_C_ABI_PREFIX PEAK_Node_GetIsFeature(m_backendHandle, isFeature);
    }) > 0;
}

inline NodeCachingMode Node::CachingMode() const
{
    return static_cast<NodeCachingMode>(
        QueryNumericFromCInterfaceFunction<PEAK_NODE_CACHING_MODE>([&](PEAK_NODE_CACHING_MODE* cachingMode) {
            return PEAK_C_ABI_PREFIX PEAK_Node_GetCachingMode(m_backendHandle, cachingMode);
        }));
}

inline int64_t Node::PollingTime() const
{
    return QueryNumericFromCInterfaceFunction<int64_t>([&](int64_t* pollingTime_ms) {
        return PEAK_C_ABI_PREFIX PEAK_Node_GetPollingTime(m_backendHandle, pollingTime_ms);
    });
}

inline std::string Node::ToolTip() const
{
    return QueryStringFromCInterfaceFunction([&](char* toolTip, size_t* toolTipSize) {
        return PEAK_C_ABI_PREFIX PEAK_Node_GetToolTip(m_backendHandle, toolTip, toolTipSize);
    });
}

inline std::string Node::Description() const
{
    return QueryStringFromCInterfaceFunction([&](char* description, size_t* descriptionSize) {
        return PEAK_C_ABI_PREFIX PEAK_Node_GetDescription(m_backendHandle, description, descriptionSize);
    });
}

inline NodeType Node::Type() const
{
    return static_cast<NodeType>(QueryNumericFromCInterfaceFunction<PEAK_NODE_TYPE>(
        [&](PEAK_NODE_TYPE* type) { return PEAK_C_ABI_PREFIX PEAK_Node_GetType(m_backendHandle, type); }));
}

inline std::shared_ptr<peak::core::NodeMap> Node::ParentNodeMap() const
{
    return LockOrThrow(m_parentNodeMap);
}

inline std::shared_ptr<Node> Node::FindInvalidatedNode(const std::string& name) const
{
    auto nodeHandle = QueryNumericFromCInterfaceFunction<PEAK_NODE_HANDLE>([&](PEAK_NODE_HANDLE* _nodeHandle) {
        return PEAK_C_ABI_PREFIX PEAK_Node_FindInvalidatedNode(
            m_backendHandle, name.c_str(), name.size() + 1, _nodeHandle);
    });

    auto nodeName = GetNodeName(nodeHandle);

    return ParentNodeMap()->FindNode(name);
}

inline std::vector<std::shared_ptr<Node>> Node::InvalidatedNodes() const
{
    auto numInvalidatedNodes = QueryNumericFromCInterfaceFunction<size_t>([&](size_t* _numInvalidatedNodes) {
        return PEAK_C_ABI_PREFIX PEAK_Node_GetNumInvalidatedNodes(m_backendHandle, _numInvalidatedNodes);
    });

    std::vector<std::shared_ptr<Node>> invalidatedNodes;
    for (size_t x = 0; x < numInvalidatedNodes; ++x)
    {
        auto nodeHandle = QueryNumericFromCInterfaceFunction<PEAK_NODE_HANDLE>(
            [&](PEAK_NODE_HANDLE* _nodeHandle) {
                return PEAK_C_ABI_PREFIX PEAK_Node_GetInvalidatedNode(m_backendHandle, x, _nodeHandle);
            });

        auto nodeName = GetNodeName(nodeHandle);

        invalidatedNodes.emplace_back(ParentNodeMap()->FindNode(nodeName));
    }

    return invalidatedNodes;
}

inline std::shared_ptr<Node> Node::FindInvalidatingNode(const std::string& name) const
{
    auto nodeHandle = QueryNumericFromCInterfaceFunction<PEAK_NODE_HANDLE>([&](PEAK_NODE_HANDLE* _nodeHandle) {
        return PEAK_C_ABI_PREFIX PEAK_Node_FindInvalidatingNode(
            m_backendHandle, name.c_str(), name.size() + 1, _nodeHandle);
    });

    auto nodeName = GetNodeName(nodeHandle);

    return ParentNodeMap()->FindNode(name);
}

inline std::vector<std::shared_ptr<Node>> Node::InvalidatingNodes() const
{
    auto numInvalidatingNodes = QueryNumericFromCInterfaceFunction<size_t>([&](size_t* _numInvalidatedNodes) {
        return PEAK_C_ABI_PREFIX PEAK_Node_GetNumInvalidatingNodes(m_backendHandle, _numInvalidatedNodes);
    });

    std::vector<std::shared_ptr<Node>> invalidatingNodes;
    for (size_t x = 0; x < numInvalidatingNodes; ++x)
    {
        auto nodeHandle = QueryNumericFromCInterfaceFunction<PEAK_NODE_HANDLE>(
            [&](PEAK_NODE_HANDLE* _nodeHandle) {
                return PEAK_C_ABI_PREFIX PEAK_Node_GetInvalidatingNode(m_backendHandle, x, _nodeHandle);
            });

        auto nodeName = GetNodeName(nodeHandle);

        invalidatingNodes.emplace_back(ParentNodeMap()->FindNode(nodeName));
    }

    return invalidatingNodes;
}

inline std::shared_ptr<Node> Node::FindSelectedNode(const std::string& name) const
{
    auto nodeHandle = QueryNumericFromCInterfaceFunction<PEAK_NODE_HANDLE>([&](PEAK_NODE_HANDLE* _nodeHandle) {
        return PEAK_C_ABI_PREFIX PEAK_Node_FindSelectedNode(
            m_backendHandle, name.c_str(), name.size() + 1, _nodeHandle);
    });

    auto nodeName = GetNodeName(nodeHandle);

    return ParentNodeMap()->FindNode(name);
}

inline std::vector<std::shared_ptr<Node>> Node::SelectedNodes() const
{
    auto numSelectedNodes = QueryNumericFromCInterfaceFunction<size_t>([&](size_t* _numSelectedNodes) {
        return PEAK_C_ABI_PREFIX PEAK_Node_GetNumSelectedNodes(m_backendHandle, _numSelectedNodes);
    });

    std::vector<std::shared_ptr<Node>> selectedNodes;
    for (size_t x = 0; x < numSelectedNodes; ++x)
    {
        auto nodeHandle = QueryNumericFromCInterfaceFunction<PEAK_NODE_HANDLE>(
            [&](PEAK_NODE_HANDLE* _nodeHandle) {
                return PEAK_C_ABI_PREFIX PEAK_Node_GetSelectedNode(m_backendHandle, x, _nodeHandle);
            });

        auto nodeName = GetNodeName(nodeHandle);

        selectedNodes.emplace_back(ParentNodeMap()->FindNode(nodeName));
    }

    return selectedNodes;
}

inline std::shared_ptr<Node> Node::FindSelectingNode(const std::string& name) const
{
    auto nodeHandle = QueryNumericFromCInterfaceFunction<PEAK_NODE_HANDLE>([&](PEAK_NODE_HANDLE* _nodeHandle) {
        return PEAK_C_ABI_PREFIX PEAK_Node_FindSelectingNode(
            m_backendHandle, name.c_str(), name.size() + 1, _nodeHandle);
    });

    auto nodeName = GetNodeName(nodeHandle);

    return ParentNodeMap()->FindNode(name);
}

inline std::vector<std::shared_ptr<Node>> Node::SelectingNodes() const
{
    auto numSelectingNodes = QueryNumericFromCInterfaceFunction<size_t>([&](size_t* _numSelectingNodes) {
        return PEAK_C_ABI_PREFIX PEAK_Node_GetNumSelectingNodes(m_backendHandle, _numSelectingNodes);
    });

    std::vector<std::shared_ptr<Node>> selectingNodes;
    for (size_t x = 0; x < numSelectingNodes; ++x)
    {
        auto nodeHandle = QueryNumericFromCInterfaceFunction<PEAK_NODE_HANDLE>(
            [&](PEAK_NODE_HANDLE* _nodeHandle) {
                return PEAK_C_ABI_PREFIX PEAK_Node_GetSelectingNode(m_backendHandle, x, _nodeHandle);
            });

        auto nodeName = GetNodeName(nodeHandle);

        selectingNodes.emplace_back(ParentNodeMap()->FindNode(nodeName));
    }

    return selectingNodes;
}

inline Node::ChangedCallbackHandle Node::RegisterChangedCallback(const Node::ChangedCallback& callback)
{
    return reinterpret_cast<ChangedCallbackHandle>(
        m_changedCallbackManager->RegisterCallback(NodeChangedCallbackContainer{ shared_from_this(), callback }));
}

inline void Node::UnregisterChangedCallback(Node::ChangedCallbackHandle callbackHandle)
{
    m_changedCallbackManager->UnregisterCallback(
        reinterpret_cast<PEAK_NODE_CHANGED_CALLBACK_HANDLE>(callbackHandle));
}

inline void PEAK_CALL_CONV Node::ChangedCallbackCWrapper(PEAK_NODE_HANDLE, void* context)
{
    auto callbackContainer = static_cast<NodeChangedCallbackContainer*>(context);

    callbackContainer->Callback(callbackContainer->_Node);
}

} /* namespace nodes */
} /* namespace core */
} /* namespace peak */
