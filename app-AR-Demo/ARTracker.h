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

//-----------------------------------------------------------------------------
//! ARTracker is the central class for tracking features in video stream.
/*!   
A instance of this class is hold by the SLScene instance.
*/
class ARTracker
{
public:
    enum TrackingTypes { CHESSBOARD, ARUCO };

    ARTracker();
    ~ARTracker();
    void            initChessboard      (int boardWidth, int boardHeight, float edgeLengthM);
    void            loadCamParams       (string filename);
    bool            trackChessboard     ();
    void            setImage            (cv::Mat image)     { _image = image; }
    float           getCameraFov        ()                  { return _cameraFovDeg; }
    SLMat4f         getViewMatrix       ()                  { return _viewMat; }
    TrackingTypes   getType             ()                  { return _type; }
    void            setType             (TrackingTypes type){ _type = type; }
    float           getCBEdgeLengthM    ()                  { return _cbEdgeLengthM; }
    std::map<int,SLMat4f>& getArucoVMs  ()                  { return _arucoVMs; }
    float           getArucoMargerLength()                  { return _arucoMarkerLength; }

    void drawArucoMarkerBoard(int numMarkersX, int numMarkersY, int markerEdgeLengthPix, int markerSepaPix,
                                         int dictionaryId, string imgName, bool showImage = false, int borderBits = 1, int marginsSize = 0 );

    bool            initArucoMarkerDetection( int dictionaryId, float markerLength,
                                              string paramFileName = "detector_params.yml" );
    bool            trackArucoMarkers   ();

private:
    void            calculateCameraFieldOfView();

    //camera intrinsic parameter
    cv::Mat _intrinsics;
    //camera distortion parameter
    cv::Mat _distortion;

    //chessboard size (number of inner squares)
    cv::Size _cbSize;
    //chessboard square edge length
    float _cbEdgeLengthM;
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

    //aruco marker detection
    cv::Ptr<cv::aruco::DetectorParameters> _detectorParams;
    //predefined dictionary
    cv::Ptr<cv::aruco::Dictionary> _dictionary;
    //marker length
    float _arucoMarkerLength;
    //Transformations of aruco markers with respect to camera
    std::map<int,SLMat4f> _arucoVMs;

    //active tracking type
    TrackingTypes _type;
};

//-----------------------------------------------------------------------------
#endif
