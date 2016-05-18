//#############################################################################
//  File:      SLTracker.cpp
//  Author:    Michael Göttlicher
//  Date:      Spring 2016
//  Codestyle: https://github.com/cpvrlab/SLProject/wiki/Coding-Style-Guidelines
//  Copyright: Marcus Hudritsch
//             This software is provide under the GNU General Public License
//             Please visit: http://opensource.org/licenses/GPL-3.0
//#############################################################################

#ifndef SLTRACKER_H
#define SLTRACKER_H

#include <stdafx.h>
#include <opencv/cv.h>

//-----------------------------------------------------------------------------
//! SLTracker is the central class for tracking features in video stream.
/*!   
A instance of this class is hold by the SLScene instance.
*/
class SLTracker
{
public:
    ~SLTracker();
    void            initChessboard      (int boardWidth, int boardHeight, float edgeLengthM);
    void            loadCamParams       (string filename);
    bool            trackChessboard     ();
    void            setImage            (cv::Mat grayImage) { _image = grayImage; }
    float           getCameraFov ()                  { return _cameraFovDeg; }
    SLMat4f         getViewMatrix       ()                  { return _viewMat; }

private:
    void            calculateCameraFieldOfView();

    //camera intrinsic parameter
    cv::Mat _intrinsics;
    //camera distortion parameter
    cv::Mat _distortion;

    //chessboard size (number of inner squares)
    cv::Size _cbSize;
    //chessboard corners in world coordinate system
    vector<cv::Point3d> _boardPoints;

    //calculated image points in findChessboardCorners
    vector<cv::Point2d> _imagePoints;

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

    // camera field of view
    float   _cameraFovDeg;
    //view transformation matrix
    SLMat4f _viewMat;
};

//-----------------------------------------------------------------------------
#endif
