//#############################################################################
//  File:      SLCVCalibration.cpp
//  Author:    Michael Göttlicher
//  Date:      Spring 2016
//  Codestyle: https://github.com/cpvrlab/SLProject/wiki/Coding-Style-Guidelines
//  Copyright: Marcus Hudritsch, Michael Göttlicher
//             This software is provide under the GNU General Public License
//             Please visit: http://opensource.org/licenses/GPL-3.0
//#############################################################################

#include <stdafx.h>         // precompiled headers
#include <SLCVCalibration.h>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/calib3d.hpp>

using namespace cv;
using namespace std;

//-----------------------------------------------------------------------------
//! Default path for calibration files
SLstring SLCVCalibration::defaultPath = "../_data/calibrations/";
//-----------------------------------------------------------------------------
SLCVCalibration::SLCVCalibration() :
    _cameraFovDeg(1.0f),
    _state(CS_uncalibrated),
    _calibFileName("cam_calibration.xml"),
    _calibParamsFileName("calib_in_params.yml"),
    _boardSize(0, 0),
    _boardSquareMM(0.0f),
    _numOfImgsToCapture(0),
    _numCaptured(0),
    _reprojectionError(-1.0f),
    _showUndistorted(true)
{
}
//-----------------------------------------------------------------------------
void SLCVCalibration::clear()
{
    _numCaptured = 0;
    _reprojectionError = -1.0f;
    _imagePoints.clear();
}
//-----------------------------------------------------------------------------
bool SLCVCalibration::loadCamParams()
{
    //load camera parameter
    FileStorage fs;
    fs.open(defaultPath + _calibFileName, FileStorage::READ);
    if (!fs.isOpened())
    {
        cout << "Could not open the calibration file: "
             << (defaultPath + _calibFileName) << endl;
        _state = CS_uncalibrated;
        return false;
    }

    fs["camera_matrix"] >> _intrinsics;
    fs["distortion_coefficients"] >> _distortion;
    fs["avg_reprojection_error"] >> _reprojectionError;


    // close the input file
    fs.release();

    //calculate projection matrix
    calculateCameraFOV();

    _state = CS_calibrated;

    return true;
}
//-----------------------------------------------------------------------------
bool SLCVCalibration::loadCalibParams()
{
    //load camera parameter
    FileStorage fs;
    fs.open(defaultPath + _calibParamsFileName, FileStorage::READ);
    if (!fs.isOpened())
    {   cout << "Could not open the calibration parameter file: "
             << (defaultPath + _calibParamsFileName) << endl;
        _state = CS_uncalibrated;
        return false;
    }

    //assign paramters
    fs["numInnerCornersWidth"] >> _boardSize.width;
    fs["numInnerCornersHeight"] >> _boardSize.height;
    fs["squareSizeMM"] >> _boardSquareMM;
    fs["numOfImgsToCapture"] >> _numOfImgsToCapture;

    return true;
}
//-----------------------------------------------------------------------------
void SLCVCalibration::calculateCameraFOV()
{
    //calculate vertical field of view
    float fy = (float)_intrinsics.at<double>(1,1);
    float cy = (float)_intrinsics.at<double>(1,2);
    float fovRad = 2 * (float)atan2(cy, fy);
    _cameraFovDeg = fovRad * SL_RAD2DEG;
}
//-----------------------------------------------------------------------------
static void calcBoardCornerPositions(Size boardSize, 
                                     float squareSize, 
                                     vector<Point3f>& corners)
{
    corners.clear();
    for(int i = 0; i < boardSize.height; ++i)
        for(int j = 0; j < boardSize.width; ++j)
            corners.push_back(Point3f(j*squareSize, i*squareSize, 0));
}
//-----------------------------------------------------------------------------
static double computeReprojectionErrors(const vector<vector<Point3f>>& objectPoints,
                                        const vector<vector<Point2f>>& imagePoints,
                                        const vector<Mat>& rvecs,
                                        const vector<Mat>& tvecs,
                                        const Mat& cameraMatrix ,
                                        const Mat& distCoeffs,
                                        vector<float>& perViewErrors)
{
    vector<Point2f> imagePoints2;
    size_t totalPoints = 0;
    double totalErr = 0, err;
    perViewErrors.resize(objectPoints.size());

    for(size_t i = 0; i < objectPoints.size(); ++i)
    {
        projectPoints(objectPoints[i], rvecs[i], tvecs[i], cameraMatrix, distCoeffs, imagePoints2);
        err = norm(imagePoints[i], imagePoints2, NORM_L2);

        size_t n = objectPoints[i].size();
        perViewErrors[i] = (float) std::sqrt(err*err/n);
        totalErr        += err*err;
        totalPoints     += n;
    }

    return std::sqrt(totalErr/totalPoints);
}
//-----------------------------------------------------------------------------
static bool runCalibration(Size& imageSize,
                           Mat& cameraMatrix,
                           Mat& distCoeffs,
                           vector<vector<Point2f>> imagePoints,
                           vector<Mat>& rvecs, vector<Mat>& tvecs,
                           vector<float>& reprojErrs,
                           double& totalAvgErr,
                           Size& boardSize,
                           float squareSize,
                           int flag)
{
    // [fixed_aspect]
    cameraMatrix = Mat::eye(3, 3, CV_64F);

    //if 1, only fy is considered as a free parameter, the ratio fx/fy stays the same as in the input cameraMatrix.
    cameraMatrix.at<double>(0,0) = 1.0;

    // [fixed_aspect]
    distCoeffs = Mat::zeros(8, 1, CV_64F);

    vector<vector<Point3f> > objectPoints(1);
    calcBoardCornerPositions(boardSize, squareSize, objectPoints[0]);

    objectPoints.resize(imagePoints.size(),objectPoints[0]);

    //Find intrinsic and extrinsic camera parameters
    double rms = calibrateCamera(objectPoints,
                                 imagePoints,
                                 imageSize,
                                 cameraMatrix,
                                 distCoeffs,
                                 rvecs,
                                 tvecs,
                                 flag);

    cout << "Re-projection error reported by calibrateCamera: "<< rms << endl;

    bool ok = checkRange(cameraMatrix) && checkRange(distCoeffs);

    totalAvgErr = computeReprojectionErrors(objectPoints,
                                            imagePoints,
                                            rvecs,
                                            tvecs,
                                            cameraMatrix,
                                            distCoeffs,
                                            reprojErrs);
    return ok;
}
//-----------------------------------------------------------------------------
// Print camera parameters to the output file
static void saveCameraParams(Size& imageSize,
                             Mat& cameraMatrix, Mat& distCoeffs,
                             const vector<Mat>& rvecs,
                             const vector<Mat>& tvecs,
                             const vector<float>& reprojErrs,
                             const vector<vector<Point2f>>& imagePoints,
                             double totalAvgErr,
                             int flag,
                             string filename,
                             Size& boardSize,
                             float squareSize)
{
    cv::FileStorage fs(SLCVCalibration::defaultPath + filename, FileStorage::WRITE);

    time_t tm;
    time(&tm);
    struct tm *t2 = localtime(&tm);
    char buf[1024];
    strftime(buf, sizeof(buf), "%c", t2);

    fs << "calibration_time" << buf;

    if(!rvecs.empty() || !reprojErrs.empty())
        fs << "nr_of_frames" << (int)std::max(rvecs.size(), reprojErrs.size());
    fs << "image_width" << imageSize.width;
    fs << "image_height" << imageSize.height;
    fs << "board_width" << boardSize.width;
    fs << "board_height" << boardSize.height;
    fs << "square_size" << squareSize;

    fs << "fix_aspect_ratio" << 1;

    if (flag)
    {   sprintf(buf, "flags:%s%s%s%s",
                flag & CALIB_USE_INTRINSIC_GUESS ? " +use_intrinsic_guess" : "",
                flag & CALIB_FIX_ASPECT_RATIO ? " +fix_aspectRatio" : "",
                flag & CALIB_FIX_PRINCIPAL_POINT ? " +fix_principal_point" : "",
                flag & CALIB_ZERO_TANGENT_DIST ? " +zero_tangent_dist" : "");
        cvWriteComment(*fs, buf, 0);
    }

    fs << "flags" << flag;
    fs << "camera_matrix" << cameraMatrix;
    fs << "distortion_coefficients" << distCoeffs;
    fs << "avg_reprojection_error" << totalAvgErr;
}
//-----------------------------------------------------------------------------
//!< Find the chessboard
bool SLCVCalibration::findChessboard(cv::Mat image,
                                     bool drawCorners)
{   
    _imageSize = image.size();
    vector<cv::Point2f> corners;
    int flags = CALIB_CB_ADAPTIVE_THRESH | 
                CALIB_CB_NORMALIZE_IMAGE | 
                CALIB_CB_FAST_CHECK;

    bool found = cv::findChessboardCorners(image, _boardSize, corners, flags);

    if(found && drawCorners)
        drawChessboardCorners(image, _boardSize, Mat(corners), found);

    if (found && _state == CS_calibrateGrab)
    {
        Mat imageGray;
        cv::cvtColor(image, imageGray, COLOR_BGR2GRAY);
        
        cornerSubPix(imageGray, 
                     corners, Size(11,11),
                     Size(-1,-1), 
                     TermCriteria(TermCriteria::EPS+TermCriteria::COUNT, 
                     30, 
                     0.1));

        //debug save image
        //stringstream ss;
        //ss << "imageIn_" << _numCaptured << ".png";
        //cv::imwrite(ss.str(), image);

        //add detected points
        _imagePoints.push_back(corners);
        _numCaptured++;

        //simulate a snapshot
        cv::bitwise_not(image, image);

        _state = CS_calibrateStream;

        // make beep sound
        cout << '\a';
    }
    return found;
}
//-----------------------------------------------------------------------------
void SLCVCalibration::calibrate()
{
    clear();
    _state = CS_calibrateStream;
}
//-----------------------------------------------------------------------------
void SLCVCalibration::calculate()
{
    _state = CS_starcCalculating;

    vector<Mat> rvecs, tvecs;
    vector<float> reprojErrs;
    double totalAvgErr = 0;

    int flag = 0;
    flag |= CALIB_FIX_PRINCIPAL_POINT;
    flag |= CALIB_ZERO_TANGENT_DIST;
    flag |= CALIB_FIX_ASPECT_RATIO;

    bool ok = runCalibration(_imageSize, 
                             _intrinsics, 
                             _distortion, 
                             _imagePoints, 
                             rvecs, 
                             tvecs, 
                             reprojErrs,
                             totalAvgErr, 
                             _boardSize, 
                             _boardSquareMM, 
                             flag);

    cout << (ok ? "Calibration succeeded" : "Calibration failed")
         << ". avg re projection error = " << totalAvgErr << endl;

    if (ok)
    {   cout << "intrinsics" << _intrinsics << endl;
        cout << "distortion" << _distortion << endl;

        saveCameraParams(_imageSize, 
                         _intrinsics, 
                         _distortion, 
                         rvecs, 
                         tvecs, 
                         reprojErrs, 
                         _imagePoints,
                         totalAvgErr, 
                         flag, 
                         _calibFileName, 
                         _boardSize, 
                         _boardSquareMM);

        calculateCameraFOV();
        _reprojectionError = (float)totalAvgErr;
        _state = CS_calibrated;
    }
}
//-----------------------------------------------------------------------------
//! Create an OpenGL 4x4 matrix from a translation & rotation vector
SLMat4f SLCVCalibration::createGLMatrix(const Mat& tVec, const Mat& rVec)
{
    // 1) convert the passed rotation vector to a rotation matrix
    cv::Mat rMat;
    Rodrigues(rVec, rMat);

    // 2) Create an OpenGL 4x4 column major matrix from the rotation matrix and 
    // translation vector from openCV as discribed in this post:
    // www.morethantechnical.com/2015/02/17/
    // augmented-reality-on-libqglviewer-and-opencv-opengl-tips-wcode
      
    // The y- and z- axis have to be inverted:
    /*
    tVec = |  t0,  t1,  t2 |
                                        |  r00   r01   r02   t0 |
           | r00, r10, r20 |            | -r10  -r11  -r12  -t1 |
    rMat = | r01, r11, r21 |    slMat = | -r20  -r21  -r22  -t2 |
           | r02, r12, r22 |            |    0     0     0    1 |
    */

    SLMat4f slMat((SLfloat) rMat.at<double>(0, 0), (SLfloat) rMat.at<double>(0, 1), (SLfloat) rMat.at<double>(0, 2), (SLfloat) tVec.at<double>(0, 0),
                  (SLfloat)-rMat.at<double>(1, 0), (SLfloat)-rMat.at<double>(1, 1), (SLfloat)-rMat.at<double>(1, 2), (SLfloat)-tVec.at<double>(1, 0),
                  (SLfloat)-rMat.at<double>(2, 0), (SLfloat)-rMat.at<double>(2, 1), (SLfloat)-rMat.at<double>(2, 2), (SLfloat)-tVec.at<double>(2, 0),
                                             0.0f,                            0.0f,                            0.0f,                            1.0f);
    return slMat;
}
//-----------------------------------------------------------------------------
