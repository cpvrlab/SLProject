//#############################################################################
//  File:      AppPenTrackingROSNode.h
//  Date:      October 2021
//  Codestyle: https://github.com/cpvrlab/SLProject/wiki/SLProject-Coding-Style
//  Authors:   Marino von Wattenwyl
//  License:   This software is provided under the GNU General Public License
//             Please visit: http://opensource.org/licenses/GPL-3.0
//#############################################################################

#ifndef SRC_APPPENTRACKINGROSNODE_H
#define SRC_APPPENTRACKINGROSNODE_H

#define BOOST_BIND_GLOBAL_PLACEHOLDERS // A Boost header which ROS includes is complaining without this
#include <ros/ros.h>

#include <SLVec3.h>
#include <SLQuat4.h>

//-----------------------------------------------------------------------------
//! AppPenTrackingROSNode provides methods for publishing data to the ROS network
/*! Currently the only method is "publish" that publishes the ArUco pen tip
 * position to the "aruco_pen/tip" topic
 */
class AppPenTrackingROSNode
{

public:
    static AppPenTrackingROSNode& instance()
    {
        static AppPenTrackingROSNode instance;
        return instance;
    }

    AppPenTrackingROSNode();
    void publishPose(const SLVec3f& position,
                     SLQuat4f       orientation) const;
    void publishKeyEvent(const SLVec3f& position,
                         SLQuat4f       orientation) const;

    ros::Publisher _posePublisher;
    ros::Publisher _keyEventsPublisher;
};
//-----------------------------------------------------------------------------

#endif // SRC_APPPENTRACKINGROSNODE_H
