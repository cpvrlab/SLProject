/*!
 * \file    peak_enumeration_entry_node.hpp
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
#include <cstdint>
#include <string>


namespace peak
{
namespace core
{
namespace nodes
{

/*!
 * \brief Represents a GenAPI enumeration entry node.
 *
 */
class EnumerationEntryNode : public Node
{
public:
    EnumerationEntryNode() = delete;
    ~EnumerationEntryNode() override = default;
    EnumerationEntryNode(const EnumerationEntryNode& other) = delete;
    EnumerationEntryNode& operator=(const EnumerationEntryNode& other) = delete;
    EnumerationEntryNode(EnumerationEntryNode&& other) = delete;
    EnumerationEntryNode& operator=(EnumerationEntryNode&& other) = delete;

    /*!
     * \brief Checks whether the node is self clearing.
     *
     * \return True, if the node is self clearing.
     * \return False otherwise.
     *
     * \since 1.0
     *
     * \throws InternalErrorException An internal error has occurred.
     */
    bool IsSelfClearing() const;

    /*!
     * \brief Returns the symbolic value (i.e. name/string value) of the enum entry.
     *
     * \return Symbolic value.
     *
     * \since 1.0
     *
     * \throws InternalErrorException An internal error has occurred.
     */
    std::string SymbolicValue() const;
    /*! \copydoc SymbolicValue */
    std::string StringValue() const
    {
        return SymbolicValue();
    }
    /*!
     * \brief Returns the numeric value of the enum entry.
     *
     * \return Value.
     *
     * \since 1.0
     *
     * \throws InternalErrorException An internal error has occurred.
     */
    int64_t Value() const;
    /*! \copydoc Value */
    int64_t NumericValue() const
    {
        return Value();
    }

private:
    friend ClassCreator<EnumerationEntryNode>;
    friend EnumerationNode;
    EnumerationEntryNode(
        PEAK_ENUMERATION_ENTRY_NODE_HANDLE enumerationEntryNodeHandle, const std::weak_ptr<NodeMap>& parentNodeMap);
    PEAK_ENUMERATION_ENTRY_NODE_HANDLE m_backendHandle;
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

inline EnumerationEntryNode::EnumerationEntryNode(
    PEAK_ENUMERATION_ENTRY_NODE_HANDLE enumerationEntryNodeHandle, const std::weak_ptr<NodeMap>& parentNodeMap)
    : Node(QueryNumericFromCInterfaceFunction<PEAK_NODE_HANDLE>([&](PEAK_NODE_HANDLE* nodeHandle) {
        return PEAK_C_ABI_PREFIX PEAK_EnumerationEntryNode_ToNode(enumerationEntryNodeHandle, nodeHandle);
    }),
        parentNodeMap)
    , m_backendHandle(enumerationEntryNodeHandle)
{}

inline bool EnumerationEntryNode::IsSelfClearing() const
{
    return QueryNumericFromCInterfaceFunction<PEAK_BOOL8>([&](PEAK_BOOL8* isSelfClearing) {
        return PEAK_C_ABI_PREFIX PEAK_EnumerationEntryNode_GetIsSelfClearing(m_backendHandle, isSelfClearing);
    }) > 0;
}

inline std::string EnumerationEntryNode::SymbolicValue() const
{
    return QueryStringFromCInterfaceFunction([&](char* symbolicValue, size_t* symbolicValueSize) {
        return PEAK_C_ABI_PREFIX PEAK_EnumerationEntryNode_GetSymbolicValue(
            m_backendHandle, symbolicValue, symbolicValueSize);
    });
}

inline int64_t EnumerationEntryNode::Value() const
{
    return QueryNumericFromCInterfaceFunction<int64_t>([&](int64_t* value) {
        return PEAK_C_ABI_PREFIX PEAK_EnumerationEntryNode_GetValue(m_backendHandle, value);
    });
}

} /* namespace nodes */
} /* namespace core */
} /* namespace peak */
