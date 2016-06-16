//#############################################################################
//  File:      ARTracker.cpp
//  Author:    Michael Göttlicher
//  Date:      Spring 2016
//  Codestyle: https://github.com/cpvrlab/SLProject/wiki/Coding-Style-Guidelines
//  Copyright: Marcus Hudritsch, Michael Göttlicher
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
//! ARTracker is the pure virtual base class for tracking features in video.
/*!   
A instance of this class is hold by the SLScene instance.
*/
class ARTracker
{
    public:
                     ARTracker      (cv::Mat intrinsics,
                                     cv::Mat distoriton);
                    ~ARTracker      ();

        //new functions
        virtual bool init           (string paramsFileDir) = 0;
        virtual bool track          () = 0;
        virtual void updateSceneView(ARSceneView* sv) = 0;
        virtual void unloadSGObjects() = 0;

        void         image          (cv::Mat image)  {_image = image;}
        SLMat4f      cvMatToGLMat   (cv::Mat& tVec, cv::Mat& rMat);

        cv::Mat _rVec;              //!< raw rotation vector from opencvs solvePNP
        cv::Mat _tVec;              //!< raw translation vector from opencvs solvePNP
        cv::Mat _rMat;              //!< rotation matrix after Rodrigues transformation
        cv::Mat _image;             //!< camera color image
        cv::Mat _grayImg;           //!< gray image
        SLMat4f _viewMat;           //!< view transformation matrix
        cv::Mat _intrinsics;        //!< camera intrinsic parameter
        cv::Mat _distortion;        //!< camera distortion parameter
        bool    _showUndistorted;   //!< flag to show image undistorted
};

//-----------------------------------------------------------------------------
#endif
