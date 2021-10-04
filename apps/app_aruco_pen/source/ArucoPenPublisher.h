//#############################################################################
//  File:      ArucoPenPublisher.cpp
//  Date:      October 2021
//  Codestyle: https://github.com/cpvrlab/SLProject/wiki/SLProject-Coding-Style
//  Authors:   Marino von Wattenwyl
//  License:   This software is provided under the GNU General Public License
//             Please visit: http://opensource.org/licenses/GPL-3.0
//#############################################################################

#ifndef SLPROJECT_ARUCOPENPUBLISHER_H
#define SLPROJECT_ARUCOPENPUBLISHER_H

// Functions for interfacing with the Banjo "aruco_pen_publisher" library
extern "C" {
    void aruco_pen_listen();
    void aruco_pen_publish(void* data, int len);
    void aruco_pen_close();
}

#endif // SLPROJECT_ARUCOPENPUBLISHER_H
