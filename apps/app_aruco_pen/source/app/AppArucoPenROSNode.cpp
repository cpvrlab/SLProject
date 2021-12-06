//#############################################################################
//  File:      AppArucoPenROSNode.cpp
//  Date:      October 2021
//  Codestyle: https://github.com/cpvrlab/SLProject/wiki/SLProject-Coding-Style
//  Authors:   Marino von Wattenwyl
//  License:   This software is provided under the GNU General Public License
//             Please visit: http://opensource.org/licenses/GPL-3.0
//#############################################################################

#include <app/AppArucoPenROSNode.h>

#include <std_msgs/String.h>

//-----------------------------------------------------------------------------
AppArucoPenROSNode::AppArucoPenROSNode()
{
    int    argc = 0;
    char** argv = nullptr;

    ros::init(argc, argv, "aruco_pen");
    ros::NodeHandle node;
    _publisher = node.advertise<std_msgs::String>("aruco_pen/tip", 1000);
}
//-----------------------------------------------------------------------------
void AppArucoPenROSNode::publish(float x, float y, float z)
{
    std::stringstream stream;
    stream << x << ":" << y << ":" << z;

    std_msgs::String msg;
    msg.data = stream.str();
    _publisher.publish(msg);
}
//-----------------------------------------------------------------------------