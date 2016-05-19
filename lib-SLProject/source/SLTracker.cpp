//#############################################################################
//  File:      SLTracker.cpp
//  Author:    Michael Göttlicher
//  Date:      Spring 2016
//  Codestyle: https://github.com/cpvrlab/SLProject/wiki/Coding-Style-Guidelines
//  Copyright: Marcus Hudritsch
//             This software is provide under the GNU General Public License
//             Please visit: http://opensource.org/licenses/GPL-3.0
//#############################################################################

#include <stdafx.h>
#ifdef SL_MEMLEAKDETECT       // set in SL.h for debug config only
#include <debug_new.h>        // memory leak detector
#endif
#include <SLScene.h>
#include <SLTracker.h>

// OpenCV
#include <opencv2/core/core.hpp>
#include <opencv2/core/utility.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/video/tracking.hpp>
#include <opencv2/aruco.hpp>

using namespace cv;

//-----------------------------------------------------------------------------
//! constructor
SLTracker::SLTracker() :
    _cbEdgeLengthM(0.035f),
    _cameraFovDeg(0.0f),
    _arucoMarkerLength(100.0),
    _type(TrackingTypes::CHESSBOARD)
{
}
//-----------------------------------------------------------------------------
//! destructor
SLTracker::~SLTracker()
{
}

//-----------------------------------------------------------------------------
void SLTracker::initChessboard(int boardWidth, int boardHeight, float edgeLengthM)
{
    _cbSize = Size(boardHeight, boardWidth);
    _cbEdgeLengthM = edgeLengthM;
    //set up matrices for storage of translation and rotation vector
    _rVec = Mat(Size(3, 1), CV_64F);
    _tVec = Mat(Size(3, 1), CV_64F);
    //set up matrix for rotation matrix after rodrigues transformation
    _rMat = Mat(3,3,CV_64F);

    //generate vectors for the points on the chessboard
    for (int i = 0; i < boardWidth; i++)
    {
        for (int j = 0; j < boardHeight; j++)
        {
            _boardPoints.push_back(Point3d(double(i * edgeLengthM), double(j * edgeLengthM), 0.0));
        }
    }
}

//-----------------------------------------------------------------------------
void SLTracker::loadCamParams(string filename)
{
    //load camera parameter
    //set up a FileStorage object to read camera params from file
    FileStorage fs;
    fs.open(filename, FileStorage::READ);
    // read camera matrix and distortion coefficients from file

    if (!fs.isOpened())
    {
        cout << "Could not open the calibration file" << endl;
        return;
    }

    fs["camera_matrix"] >> _intrinsics;
    fs["distortion_coefficients"] >> _distortion;
    // close the input file
    fs.release();

    //calculate projection matrix
    calculateCameraFieldOfView();
}

//-----------------------------------------------------------------------------
void SLTracker::calculateCameraFieldOfView()
{
    //calculate vertical field of view
    float fy = _intrinsics.at<double>(1,1);
    float cy = _intrinsics.at<double>(1,2);
    float fovRad = 2 * atan2( cy, fy );
    _cameraFovDeg = fovRad * SL_RAD2DEG;
}

//-----------------------------------------------------------------------------
bool SLTracker::trackChessboard()
{
    bool found = false;

    if(!_image.empty() && !_intrinsics.empty())
    {
        //make a gray copy of the webcam image
        cvtColor(_image, _grayImg, CV_RGB2GRAY);

        //detect chessboard corners
        found = findChessboardCorners(_grayImg, _cbSize, _imagePoints );

        if(found)
        {
            //find the camera extrinsic parameters
            solvePnP(Mat(_boardPoints), Mat(_imagePoints), _intrinsics, _distortion, _rVec, _tVec, false);

            //Transform calculated position (rotation and translation vector) from openCV to SLProject form
            //as discribed in this post:
            //http://www.morethantechnical.com/2015/02/17/augmented-reality-on-libqglviewer-and-opencv-opengl-tips-wcode/
            //attention: We dont have to transpose the resulting matrix, because SLProject uses row-major matrices.
            //For direct openGL use you have to transpose the resulting matrix additionally.

            //convert vector to rotation matrix
            Rodrigues(_rVec, _rMat);

            //convert to SLMat4f:
            //y- and z- axis have to be inverted
            /*
                  |  r00   r01   r02   t0 |
                  | -r10  -r11  -r12  -t1 |
              m = | -r20  -r21  -r22  -t2 |
                  |    0     0     0    1 |
            */
            //1st row
            _viewMat(0,0) = _rMat.at<double>(0,0);
            _viewMat(0,1) = _rMat.at<double>(0,1);
            _viewMat(0,2) = _rMat.at<double>(0,2);
            _viewMat(0,3) = _tVec.at<double>(0,0);
            //2nd row
            _viewMat(1,0) = -_rMat.at<double>(1,0);
            _viewMat(1,1) = -_rMat.at<double>(1,1);
            _viewMat(1,2) = -_rMat.at<double>(1,2);
            _viewMat(1,3) = -_tVec.at<double>(1,0);
            //3rd row
            _viewMat(2,0) = -_rMat.at<double>(2,0);
            _viewMat(2,1) = -_rMat.at<double>(2,1);
            _viewMat(2,2) = -_rMat.at<double>(2,2);
            _viewMat(2,3) = -_tVec.at<double>(2,0);
            //4th row
            _viewMat(3,0) = 0.0f;
            _viewMat(3,1) = 0.0f;
            _viewMat(3,2) = 0.0f;
            _viewMat(3,3) = 1.0f;

            //_viewMat.print();
        }
    }

    return found;
}
//-----------------------------------------------------------------------------
void SLTracker::drawArucoMarkerBoard(int numMarkersX, int numMarkersY, int markerEdgeLengthPix, int markerSepaPix,
                                     int dictionaryId, string imgName, bool showImage, int borderBits, int marginsSize )
{
    if(marginsSize == 0)
        marginsSize = markerSepaPix;

    Size imageSize;
    imageSize.width = numMarkersX * (markerEdgeLengthPix + markerSepaPix) - markerSepaPix + 2 * marginsSize;
    imageSize.height =
     numMarkersY * (markerEdgeLengthPix + markerSepaPix) - markerSepaPix + 2 * marginsSize;

    Ptr<aruco::Dictionary> dictionary =
     aruco::getPredefinedDictionary(aruco::PREDEFINED_DICTIONARY_NAME(dictionaryId));

    Ptr<aruco::GridBoard> board = aruco::GridBoard::create(numMarkersX, numMarkersY, float(markerEdgeLengthPix),
                                                   float(markerSepaPix), dictionary);

    // show created board
    Mat boardImage;
    board->draw(imageSize, boardImage, marginsSize, borderBits);

    if(showImage) {
     imshow("board", boardImage);
     waitKey(0);
    }

    imwrite(imgName, boardImage);
}
//-----------------------------------------------------------------------------
static bool readDetectorParameters(string filename, Ptr<aruco::DetectorParameters> &params) {
    FileStorage fs(filename, FileStorage::READ);
    if(!fs.isOpened())
        return false;
    fs["adaptiveThreshWinSizeMin"] >> params->adaptiveThreshWinSizeMin;
    fs["adaptiveThreshWinSizeMax"] >> params->adaptiveThreshWinSizeMax;
    fs["adaptiveThreshWinSizeStep"] >> params->adaptiveThreshWinSizeStep;
    fs["adaptiveThreshConstant"] >> params->adaptiveThreshConstant;
    fs["minMarkerPerimeterRate"] >> params->minMarkerPerimeterRate;
    fs["maxMarkerPerimeterRate"] >> params->maxMarkerPerimeterRate;
    fs["polygonalApproxAccuracyRate"] >> params->polygonalApproxAccuracyRate;
    fs["minCornerDistanceRate"] >> params->minCornerDistanceRate;
    fs["minDistanceToBorder"] >> params->minDistanceToBorder;
    fs["minMarkerDistanceRate"] >> params->minMarkerDistanceRate;
    fs["doCornerRefinement"] >> params->doCornerRefinement;
    fs["cornerRefinementWinSize"] >> params->cornerRefinementWinSize;
    fs["cornerRefinementMaxIterations"] >> params->cornerRefinementMaxIterations;
    fs["cornerRefinementMinAccuracy"] >> params->cornerRefinementMinAccuracy;
    fs["markerBorderBits"] >> params->markerBorderBits;
    fs["perspectiveRemovePixelPerCell"] >> params->perspectiveRemovePixelPerCell;
    fs["perspectiveRemoveIgnoredMarginPerCell"] >> params->perspectiveRemoveIgnoredMarginPerCell;
    fs["maxErroneousBitsInBorderRate"] >> params->maxErroneousBitsInBorderRate;
    fs["minOtsuStdDev"] >> params->minOtsuStdDev;
    fs["errorCorrectionRate"] >> params->errorCorrectionRate;
    return true;
}
//-----------------------------------------------------------------------------
bool SLTracker::initArucoMarkerDetection( int dictionaryId, float markerLength, string paramFileName )
{
    _detectorParams = aruco::DetectorParameters::create();

    bool readOk = readDetectorParameters(paramFileName, _detectorParams);
    if(!readOk) {
        cerr << "Invalid detector parameters file" << endl;
        return false;
    }
    // do corner refinement in markers
    _detectorParams->doCornerRefinement = true;
    _detectorParams->adaptiveThreshWinSizeMin = 4;
    _detectorParams->adaptiveThreshWinSizeMax = 7;
    _detectorParams->adaptiveThreshWinSizeStep = 1;

    _dictionary =  aruco::getPredefinedDictionary(aruco::PREDEFINED_DICTIONARY_NAME(dictionaryId));
    _arucoMarkerLength = markerLength;

    return true;
}
//-----------------------------------------------------------------------------
bool SLTracker::trackArucoMarkers()
{
    vector< int > ids;
    vector< vector< Point2f > > corners, rejected;
    vector< Vec3d > rvecs, tvecs;
    //clear detected Objects from last frame
    _arucoVMs.clear();

    if(!_image.empty() && !_intrinsics.empty() && !_detectorParams.empty() && !_dictionary.empty())
    {
        aruco::detectMarkers(_image, _dictionary, corners, ids, _detectorParams, rejected);

        if( ids.size() > 0)
        {
            aruco::estimatePoseSingleMarkers(corners, _arucoMarkerLength, _intrinsics, _distortion, rvecs,
                                                         tvecs);

            for( size_t i=0; i < rvecs.size(); ++i)
            {
                //Transform calculated position (rotation and translation vector) from openCV to SLProject form
                //as discribed in this post:
                //http://www.morethantechnical.com/2015/02/17/augmented-reality-on-libqglviewer-and-opencv-opengl-tips-wcode/
                //attention: We dont have to transpose the resulting matrix, because SLProject uses row-major matrices.
                //For direct openGL use you have to transpose the resulting matrix additionally.

                //convert vector to rotation matrix
                Rodrigues(rvecs[i], _rMat);
                _tVec = Mat(tvecs[i]);

                //convert to SLMat4f:
                //y- and z- axis have to be inverted
                /*
                      |  r00   r01   r02   t0 |
                      | -r10  -r11  -r12  -t1 |
                  m = | -r20  -r21  -r22  -t2 |
                      |    0     0     0    1 |
                */

                SLMat4f vm;
                //1st row
                vm(0,0) = _rMat.at<double>(0,0);
                vm(0,1) = _rMat.at<double>(0,1);
                vm(0,2) = _rMat.at<double>(0,2);
                vm(0,3) = _tVec.at<double>(0,0);
                //2nd row
                vm(1,0) = -_rMat.at<double>(1,0);
                vm(1,1) = -_rMat.at<double>(1,1);
                vm(1,2) = -_rMat.at<double>(1,2);
                vm(1,3) = -_tVec.at<double>(1,0);
                //3rd row
                vm(2,0) = -_rMat.at<double>(2,0);
                vm(2,1) = -_rMat.at<double>(2,1);
                vm(2,2) = -_rMat.at<double>(2,2);
                vm(2,3) = -_tVec.at<double>(2,0);
                //4th row
                vm(3,0) = 0.0f;
                vm(3,1) = 0.0f;
                vm(3,2) = 0.0f;
                vm(3,3) = 1.0f;

                _arucoVMs.insert( pair<int,SLMat4f>(ids[i], vm));
            }
        }
        return true;
    }

    return false;
}

//-----------------------------------------------------------------------------
