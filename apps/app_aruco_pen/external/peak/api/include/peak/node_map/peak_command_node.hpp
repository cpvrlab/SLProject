/*!
 * \file    peak_command_node.hpp
 *
 * \author  IDS Imaging Development Systems GmbH
 * \date    2019-05-01
 * \since   1.0
 *
 * Copyright (c) 2019, IDS Imaging Development Systems GmbH. All rights reserved.
 */

#pragma once


#include <peak/backend/peak_backend.h>
#include <peak/common/peak_timeout.hpp>
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
 * \brief Represents a GenAPI command node.
 *
 */
class CommandNode : public Node
{
public:
    CommandNode() = delete;
    ~CommandNode() override = default;
    CommandNode(const CommandNode& other) = delete;
    CommandNode& operator=(const CommandNode& other) = delete;
    CommandNode(CommandNode&& other) = delete;
    CommandNode& operator=(CommandNode&& other) = delete;

    /*!
     * \brief Checks whether the command is done.
     *
     * \return True, if the command is done
     * \return False otherwise.
     *
     * \since 1.0
     *
     * \throws InternalErrorException An internal error has occurred.
     */
    bool IsDone() const;

    /*!
     * \brief Executes the command associated with this node
     *
     * \since 1.0
     *
     * \throws InternalErrorException An internal error has occurred.
     */
    void Execute();

    /*!
     * \brief (Blocking) Wait for the executed command to be finished, i.e. until IsDone() is true.
     *
     * \param[in] waitTimeout_ms The maximum waiting time in milliseconds.
     *
     * \since 1.0
     *
     * \throws InternalErrorException An internal error has occurred.
     * \throws TimeoutException       The specified timeout expired without the executed command being finished.
     */
    void WaitUntilDone(Timeout waitTimeout_ms = 500);


private:
    friend ClassCreator<CommandNode>;
    CommandNode(PEAK_COMMAND_NODE_HANDLE commandNodeHandle, const std::weak_ptr<NodeMap>& parentNodeMap);
    PEAK_COMMAND_NODE_HANDLE m_backendHandle;
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

inline CommandNode::CommandNode(
    PEAK_COMMAND_NODE_HANDLE commandNodeHandle, const std::weak_ptr<NodeMap>& parentNodeMap)
    : Node(QueryNumericFromCInterfaceFunction<PEAK_NODE_HANDLE>([&](PEAK_NODE_HANDLE* nodeHandle) {
        return PEAK_C_ABI_PREFIX PEAK_CommandNode_ToNode(commandNodeHandle, nodeHandle);
    }),
        parentNodeMap)
    , m_backendHandle(commandNodeHandle)
{}

inline bool CommandNode::IsDone() const
{
    return QueryNumericFromCInterfaceFunction<PEAK_BOOL8>([&](PEAK_BOOL8* isDone) {
        return PEAK_C_ABI_PREFIX PEAK_CommandNode_GetIsDone(m_backendHandle, isDone);
    }) > 0;
}
inline void CommandNode::Execute()
{
    CallAndCheckCInterfaceFunction([&] { return PEAK_C_ABI_PREFIX PEAK_CommandNode_Execute(m_backendHandle); });
}

inline void CommandNode::WaitUntilDone(Timeout waitTimeout_ms /*= 500*/)
{
    CallAndCheckCInterfaceFunction(
        [&] { return PEAK_C_ABI_PREFIX PEAK_CommandNode_WaitUntilDone(m_backendHandle, waitTimeout_ms); });
}

} /* namespace nodes */
} /* namespace core */
} /* namespace peak */
