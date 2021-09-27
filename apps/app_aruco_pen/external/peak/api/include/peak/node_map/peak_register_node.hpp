/*!
 * \file    peak_register_node.hpp
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
#include <peak/error_handling/peak_error_handling.hpp>
#include <peak/node_map/peak_common_node_enums.hpp>
#include <peak/node_map/peak_node.hpp>

#include <cstddef>
#include <cstdint>
#include <vector>


namespace peak
{
namespace core
{
namespace nodes
{

/*!
 * \brief Represents a GenAPI register node.
 *
 */
class RegisterNode : public Node
{
public:
    RegisterNode() = delete;
    ~RegisterNode() override = default;
    RegisterNode(const RegisterNode& other) = delete;
    RegisterNode& operator=(const RegisterNode& other) = delete;
    RegisterNode(RegisterNode&& other) = delete;
    RegisterNode& operator=(RegisterNode&& other) = delete;

    /*!
     * \brief Returns the address.
     *
     * \return Address
     *
     * \since 1.0
     *
     * \throws InternalErrorException An internal error has occurred.
     */
    uint64_t Address() const;
    /*!
     * \brief Returns the length.
     *
     * \return Length
     *
     * \since 1.0
     *
     * \throws InternalErrorException An internal error has occurred.
     */
    size_t Length() const;

    /*!
     * \brief Reads a given amount of bytes.
     *
     * \param[in] numBytes The amount of bytes to read.
     * \param[in] cacheUsePolicy A flag telling whether the value should be read using the internal cache.
     *
     * \return Read bytes
     *
     * \since 1.0
     *
     * \throws InternalErrorException An internal error has occurred.
     */
    std::vector<uint8_t> Read(size_t numBytes, NodeCacheUsePolicy cacheUsePolicy = NodeCacheUsePolicy::UseCache) const;
    /*!
     * \brief Writes a given amount of bytes.
     *
     * \param[in] bytes The bytes to write.
     *
     * \since 1.0
     *
     * \throws InternalErrorException An internal error has occurred.
     */
    void Write(const std::vector<uint8_t>& bytes);

private:
    friend ClassCreator<RegisterNode>;
    RegisterNode(PEAK_REGISTER_NODE_HANDLE registerNodeHandle, const std::weak_ptr<NodeMap>& parentNodeMap);
    PEAK_REGISTER_NODE_HANDLE m_backendHandle;
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

inline RegisterNode::RegisterNode(
    PEAK_REGISTER_NODE_HANDLE registerNodeHandle, const std::weak_ptr<NodeMap>& parentNodeMap)
    : Node(QueryNumericFromCInterfaceFunction<PEAK_NODE_HANDLE>([&](PEAK_NODE_HANDLE* nodeHandle) {
        return PEAK_C_ABI_PREFIX PEAK_RegisterNode_ToNode(registerNodeHandle, nodeHandle);
    }),
        parentNodeMap)
    , m_backendHandle(registerNodeHandle)
{}

inline uint64_t RegisterNode::Address() const
{
    return QueryNumericFromCInterfaceFunction<uint64_t>([&](uint64_t* address) {
        return PEAK_C_ABI_PREFIX PEAK_RegisterNode_GetAddress(m_backendHandle, address);
    });
}

inline size_t RegisterNode::Length() const
{
    return QueryNumericFromCInterfaceFunction<size_t>(
        [&](size_t* length) { return PEAK_C_ABI_PREFIX PEAK_RegisterNode_GetLength(m_backendHandle, length); });
}

inline std::vector<uint8_t> RegisterNode::Read(
    size_t numBytes, NodeCacheUsePolicy cacheUsePolicy /* = NodeCacheUsePolicy::UseCache */) const
{
    std::vector<uint8_t> bytes(numBytes);
    ExecuteAndMapReturnCodes([&] {
        return PEAK_C_ABI_PREFIX PEAK_RegisterNode_Read(
            m_backendHandle, static_cast<PEAK_NODE_CACHE_USE_POLICY>(cacheUsePolicy), bytes.data(), bytes.size());
    });

    return bytes;
}

inline void RegisterNode::Write(const std::vector<uint8_t>& bytes)
{
    ExecuteAndMapReturnCodes(
        [&] { return PEAK_C_ABI_PREFIX PEAK_RegisterNode_Write(m_backendHandle, bytes.data(), bytes.size()); });
}

} /* namespace nodes */
} /* namespace core */
} /* namespace peak */
