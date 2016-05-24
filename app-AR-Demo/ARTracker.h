//#############################################################################
//  File:      ARTracker.cpp
//  Author:    Michael Göttlicher
//  Date:      Spring 2016
//  Codestyle: https://github.com/cpvrlab/SLProject/wiki/Coding-Style-Guidelines
//  Copyright: Marcus Hudritsch
//             This software is provide under the GNU General Public License
//             Please visit: http://opensource.org/licenses/GPL-3.0
//#############################################################################

#ifndef ARTracker_H
#define ARTracker_H

#include <stdafx.h>
#include <opencv/cv.h>
#include <opencv2/aruco.hpp>

class ARSceneView;
//-----------------------------------------------------------------------------
//! ARTracker is the central class for tracking features in video stream.
/*!   
A instance of this class is hold by the SLScene instance.
*/
class ARTracker
{
public:
    ARTracker(cv::Mat intrinsics, cv::Mat distoriton);
    ~ARTracker();

    //new functions
    virtual bool init           (string paramsFileDir) = 0;
    virtual bool track          () = 0;
    virtual void updateSceneView( ARSceneView* sv ) = 0;
    virtual void unloadSGObjects() = 0;

    void         setImage        (cv::Mat image)     { _image = image; }

    //raw rotation vector from opencvs solvePNP function
    cv::Mat _rVec;
    //raw translation vector from opencvs solvePNP function
    cv::Mat _tVec;
    //rotation matrix after Rodrigues transformation
    cv::Mat _rMat;

    //camera color image
    cv::Mat _image;
    //gray image
    cv::Mat _grayImg;

    //view transformation matrix
    SLMat4f _viewMat;

    //camera intrinsic parameter
    cv::Mat _intrinsics;
    //camera distortion parameter
    cv::Mat _distortion;

    bool _showUndistorted;
};

//-----------------------------------------------------------------------------
#endif
