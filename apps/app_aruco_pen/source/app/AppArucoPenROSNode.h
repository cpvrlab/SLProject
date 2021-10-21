//#############################################################################
//  File:      AppArucoPenROSNode.h
//  Date:      October 2021
//  Codestyle: https://github.com/cpvrlab/SLProject/wiki/SLProject-Coding-Style
//  Authors:   Marino von Wattenwyl
//  License:   This software is provided under the GNU General Public License
//             Please visit: http://opensource.org/licenses/GPL-3.0
//#############################################################################

#ifndef SRC_APPARUCOPENROSNODE_H
#define SRC_APPARUCOPENROSNODE_H

#define BOOST_BIND_GLOBAL_PLACEHOLDERS      // A Boost header which ROS includes is complaining without this
#include <ros/ros.h>

//-----------------------------------------------------------------------------
//! AppArucoPenROSNode provides methods for publishing data to the ROS network
/*! Currently the only method is "publish" that publishes the ArUco pen tip
 * position to the "aruco_pen/tip" topic
 */
class AppArucoPenROSNode {

public:
    static AppArucoPenROSNode& instance()
    {
        static AppArucoPenROSNode instance;
        return instance;
    }

    AppArucoPenROSNode();
    void publish(float x, float y, float z);

private:
    ros::Publisher _publisher;

};
//-----------------------------------------------------------------------------

#endif //SRC_APPARUCOPENROSNODE_H
