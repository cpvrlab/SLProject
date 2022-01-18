//#############################################################################
//  File:      AppPenTrackingROSNode.cpp
//  Date:      October 2021
//  Codestyle: https://github.com/cpvrlab/SLProject/wiki/SLProject-Coding-Style
//  Authors:   Marino von Wattenwyl
//  License:   This software is provided under the GNU General Public License
//             Please visit: http://opensource.org/licenses/GPL-3.0
//#############################################################################

#include <app/AppPenTrackingROSNode.h>

#include <geometry_msgs/Pose.h>

//-----------------------------------------------------------------------------
AppPenTrackingROSNode::AppPenTrackingROSNode()
{
    int    argc = 0;
    char** argv = nullptr;

    ros::init(argc, argv, "aruco_pen");
    ros::NodeHandle node;
    _posePublisher      = node.advertise<geometry_msgs::Pose>("aruco_pen/pose", 1000);
    _keyEventsPublisher = node.advertise<geometry_msgs::Pose>("aruco_pen/key_events", 1000);
}
//-----------------------------------------------------------------------------
void AppPenTrackingROSNode::publishPose(const SLVec3f& position,
                                     SLQuat4f       orientation) const
{
    geometry_msgs::Pose msg;
    msg.position.x    = (double)position.x;
    msg.position.y    = (double)position.y;
    msg.position.z    = (double)position.z;
    msg.orientation.x = (double)orientation.x();
    msg.orientation.y = (double)orientation.y();
    msg.orientation.z = (double)orientation.z();
    msg.orientation.w = (double)orientation.w();
    _posePublisher.publish(msg);
}
//-----------------------------------------------------------------------------
void AppPenTrackingROSNode::publishKeyEvent(const SLVec3f& position,
                                         SLQuat4f       orientation) const
{
    geometry_msgs::Pose msg;
    msg.position.x    = (double)position.x;
    msg.position.y    = (double)position.y;
    msg.position.z    = (double)position.z;
    msg.orientation.x = (double)orientation.x();
    msg.orientation.y = (double)orientation.y();
    msg.orientation.z = (double)orientation.z();
    msg.orientation.w = (double)orientation.w();
    _keyEventsPublisher.publish(msg);
}
//-----------------------------------------------------------------------------