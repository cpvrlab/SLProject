/*!
 * \file    peak_enumeration_node.hpp
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
#include <peak/node_map/peak_enumeration_entry_node.hpp>
#include <peak/node_map/peak_node.hpp>

#include <cstddef>
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
 * \brief Represents a GenAPI enumeration node.
 *
 */
class EnumerationNode : public Node
{
public:
    EnumerationNode() = delete;
    ~EnumerationNode() override = default;
    EnumerationNode(const EnumerationNode& other) = delete;
    EnumerationNode& operator=(const EnumerationNode& other) = delete;
    EnumerationNode(EnumerationNode&& other) = delete;
    EnumerationNode& operator=(EnumerationNode&& other) = delete;

    /*!
     * \brief Returns the current entry.
     *
     * \param[in] cacheUsePolicy A flag telling whether the value should be read using the internal cache.
     *
     * \return Current entry
     *
     * \since 1.0
     *
     * \throws InternalErrorException An internal error has occurred.
     */
    std::shared_ptr<EnumerationEntryNode> CurrentEntry(
        NodeCacheUsePolicy cacheUsePolicy = NodeCacheUsePolicy::UseCache) const;
    /*!
     * \brief Sets the current entry.
     *
     * \param[in] entry The entry to set as current entry.
     *
     * \since 1.0
     *
     * \throws InternalErrorException An internal error has occurred.
     * \throws InvalidArgumentException There is no matching entry in this enumeration node.
     */
    void SetCurrentEntry(const std::shared_ptr<EnumerationEntryNode>& entry);
    /*!
     * \brief Sets the current entry to an entry with the given symbolic value.
     *
     * \param[in] symbolicValue The symbolic value of the entry to set.
     *
     * \since 1.0
     *
     * \throws InternalErrorException An internal error has occurred.
     * \throws InvalidArgumentException There is no entry with this symbolicValue in this enumeration node.
     */
    void SetCurrentEntry(const std::string& symbolicValue);
    /*!
     * \brief Sets the current entry to an entry with the given value.
     *
     * \param[in] value The value of the entry to set.
     *
     * \since 1.0
     *
     * \throws InternalErrorException An internal error has occurred.
     * \throws InvalidArgumentException There is no entry with this value in this enumeration node.
     */
    void SetCurrentEntry(int64_t value);
    /*!
     * \brief Tries to find a entry with the given symbolic value.
     *
     * \param[in] symbolicValue The symbolic value of the entry to find.
     *
     * \return Found entry.
     * \return If no entry with the given name exists, a NotFoundException is thrown.
     *
     * \since 1.0
     *
     * \throws NotFoundException There is no entry with the given name.
     * \throws InternalErrorException An internal error has occurred.
     */
    std::shared_ptr<EnumerationEntryNode> FindEntry(const std::string& symbolicValue) const;
    /*!
     * \brief Tries to find a entry with the given numeric value.
     *
     * \param[in] value The value of the entry to find.
     *
     * \return Found entry.
     * \return If no entry with the given numeric value, a NotFoundException is thrown.
     *
     * \since 1.0
     *
     * \throws NotFoundException There is no entry with the given numeric value.
     * \throws InternalErrorException An internal error has occurred.
     */
    std::shared_ptr<EnumerationEntryNode> FindEntry(int64_t value) const;
    /*!
     * \brief Returns the entries.
     *
     * \return Entries
     *
     * \since 1.0
     *
     * \throws InternalErrorException An internal error has occurred.
     */
    std::vector<std::shared_ptr<EnumerationEntryNode>> Entries() const;

private:
    friend ClassCreator<EnumerationNode>;
    EnumerationNode(PEAK_ENUMERATION_NODE_HANDLE enumerationNodeHandle, const std::weak_ptr<NodeMap>& parentNodeMap);
    PEAK_ENUMERATION_NODE_HANDLE m_backendHandle;
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

inline EnumerationNode::EnumerationNode(
    PEAK_ENUMERATION_NODE_HANDLE enumerationNodeHandle, const std::weak_ptr<NodeMap>& parentNodeMap)
    : Node(QueryNumericFromCInterfaceFunction<PEAK_NODE_HANDLE>([&](PEAK_NODE_HANDLE* nodeHandle) {
        return PEAK_C_ABI_PREFIX PEAK_EnumerationNode_ToNode(enumerationNodeHandle, nodeHandle);
    }),
        parentNodeMap)
    , m_backendHandle(enumerationNodeHandle)
{}

inline std::shared_ptr<EnumerationEntryNode> EnumerationNode::CurrentEntry(
    NodeCacheUsePolicy cacheUsePolicy /* = NodeCacheUsePolicy::UseCache */) const
{
    auto enumerationEntryNodeHandle = QueryNumericFromCInterfaceFunction<PEAK_ENUMERATION_ENTRY_NODE_HANDLE>(
        [&](PEAK_ENUMERATION_ENTRY_NODE_HANDLE* _enumerationEntryNodeHandle) {
            return PEAK_C_ABI_PREFIX PEAK_EnumerationNode_GetCurrentEntry(m_backendHandle,
                static_cast<PEAK_NODE_CACHE_USE_POLICY>(cacheUsePolicy), _enumerationEntryNodeHandle);
        });

    auto nodeHandle = QueryNumericFromCInterfaceFunction<PEAK_NODE_HANDLE>([&](PEAK_NODE_HANDLE* _nodeHandle) {
        return PEAK_C_ABI_PREFIX PEAK_EnumerationEntryNode_ToNode(enumerationEntryNodeHandle, _nodeHandle);
    });

    auto nodeName = GetNodeName(nodeHandle);

    return std::dynamic_pointer_cast<EnumerationEntryNode>(ParentNodeMap()->FindNode(nodeName));
}

inline void EnumerationNode::SetCurrentEntry(const std::shared_ptr<EnumerationEntryNode>& entry)
{
    CallAndCheckCInterfaceFunction([&] {
        return PEAK_C_ABI_PREFIX PEAK_EnumerationNode_SetCurrentEntry(m_backendHandle, entry->m_backendHandle);
    });
}

inline void EnumerationNode::SetCurrentEntry(const std::string& symbolicValue)
{
    CallAndCheckCInterfaceFunction([&] {
        return PEAK_C_ABI_PREFIX PEAK_EnumerationNode_SetCurrentEntryBySymbolicValue(
            m_backendHandle, symbolicValue.c_str(), symbolicValue.size() + 1);
    });
}

inline void EnumerationNode::SetCurrentEntry(int64_t value)
{
    CallAndCheckCInterfaceFunction(
        [&] { return PEAK_C_ABI_PREFIX PEAK_EnumerationNode_SetCurrentEntryByValue(m_backendHandle, value); });
}

inline std::shared_ptr<EnumerationEntryNode> EnumerationNode::FindEntry(const std::string& symbolicValue) const
{
    auto enumerationEntryNodeHandle = QueryNumericFromCInterfaceFunction<PEAK_ENUMERATION_ENTRY_NODE_HANDLE>(
        [&](PEAK_ENUMERATION_ENTRY_NODE_HANDLE* _enumerationEntryNodeHandle) {
            return PEAK_C_ABI_PREFIX PEAK_EnumerationNode_FindEntryBySymbolicValue(
                m_backendHandle, symbolicValue.c_str(), symbolicValue.size() + 1, _enumerationEntryNodeHandle);
        });

    auto nodeHandle = QueryNumericFromCInterfaceFunction<PEAK_NODE_HANDLE>([&](PEAK_NODE_HANDLE* _nodeHandle) {
        return PEAK_C_ABI_PREFIX PEAK_EnumerationEntryNode_ToNode(enumerationEntryNodeHandle, _nodeHandle);
    });

    auto nodeName = GetNodeName(nodeHandle);

    return std::dynamic_pointer_cast<EnumerationEntryNode>(ParentNodeMap()->FindNode(nodeName));
}

inline std::shared_ptr<EnumerationEntryNode> EnumerationNode::FindEntry(int64_t value) const
{
    auto enumerationEntryNodeHandle = QueryNumericFromCInterfaceFunction<PEAK_ENUMERATION_ENTRY_NODE_HANDLE>(
        [&](PEAK_ENUMERATION_ENTRY_NODE_HANDLE* _enumerationEntryNodeHandle) {
            return PEAK_C_ABI_PREFIX PEAK_EnumerationNode_FindEntryByValue(
                m_backendHandle, value, _enumerationEntryNodeHandle);
        });

    auto nodeHandle = QueryNumericFromCInterfaceFunction<PEAK_NODE_HANDLE>([&](PEAK_NODE_HANDLE* _nodeHandle) {
        return PEAK_C_ABI_PREFIX PEAK_EnumerationEntryNode_ToNode(enumerationEntryNodeHandle, _nodeHandle);
    });

    auto nodeName = GetNodeName(nodeHandle);

    return std::dynamic_pointer_cast<EnumerationEntryNode>(ParentNodeMap()->FindNode(nodeName));
}

inline std::vector<std::shared_ptr<EnumerationEntryNode>> EnumerationNode::Entries() const
{
    auto numEntries = QueryNumericFromCInterfaceFunction<size_t>([&](size_t* _numEntries) {
        return PEAK_C_ABI_PREFIX PEAK_EnumerationNode_GetNumEntries(m_backendHandle, _numEntries);
    });

    std::vector<std::shared_ptr<EnumerationEntryNode>> entries;
    for (size_t x = 0; x < numEntries; ++x)
    {
        auto enumerationEntryNodeHandle = QueryNumericFromCInterfaceFunction<PEAK_ENUMERATION_ENTRY_NODE_HANDLE>(
            [&](PEAK_ENUMERATION_ENTRY_NODE_HANDLE* _enumerationEntryNodeHandle) {
                return PEAK_C_ABI_PREFIX PEAK_EnumerationNode_GetEntry(
                    m_backendHandle, x, _enumerationEntryNodeHandle);
            });

        auto nodeHandle = QueryNumericFromCInterfaceFunction<PEAK_NODE_HANDLE>(
            [&](PEAK_NODE_HANDLE* _nodeHandle) {
                return PEAK_C_ABI_PREFIX PEAK_EnumerationEntryNode_ToNode(
                    enumerationEntryNodeHandle, _nodeHandle);
            });

        auto nodeName = GetNodeName(nodeHandle);

        entries.emplace_back(std::dynamic_pointer_cast<EnumerationEntryNode>(ParentNodeMap()->FindNode(nodeName)));
    }

    return entries;
}

} /* namespace nodes */
} /* namespace core */
} /* namespace peak */
