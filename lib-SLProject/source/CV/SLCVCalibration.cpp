//#############################################################################
//  File:      SLCVCalibration.cpp
//  Author:    Michael G�ttlicher
//  Date:      Spring 2016
//  Codestyle: https://github.com/cpvrlab/SLProject/wiki/Coding-Style-Guidelines
//  Copyright: Marcus Hudritsch, Michael G�ttlicher
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
SLCVCalibration::SLCVCalibration() :
    _cameraFovDeg(1.0f),
    _state(IDLE),
    _calibFileName("cam_calibration.xml"),
    _calibParamsFileName("calib_in_params.yml"),
    _numInnerCornersWidth(0),
    _numInnerCornersHeight(0),
    _squareSizeMM(0.0f),
    _captureDelayMS(0),
    _numOfImgsToCapture(0),
    _numCaptured(0),
    _reprojectionError(-1.0f),
    _prevTimestamp(0),
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
bool SLCVCalibration::loadCamParams(string dir)
{
    //load camera parameter
    FileStorage fs;
    fs.open(dir + _calibFileName, FileStorage::READ);
    if (!fs.isOpened())
    {
        cout << "Could not open the calibration file: "
             << (dir + _calibFileName) << endl;
        _state = IDLE;
        return false;
    }

    fs["camera_matrix"] >> _intrinsics;
    fs["distortion_coefficients"] >> _distortion;
    // close the input file
    fs.release();

    //calculate projection matrix
    calculateCameraFOV();

    _state = CALIBRATED;

    return true;
}
//-----------------------------------------------------------------------------
bool SLCVCalibration::loadCalibParams(std::string calibFilesDir)
{
    //load camera parameter
    FileStorage fs;
    fs.open(calibFilesDir + _calibParamsFileName, FileStorage::READ);
    if (!fs.isOpened())
    {
        cout << "Could not open the calibration parameter file: "
             << (calibFilesDir + _calibParamsFileName) << endl;
        _state = IDLE;
        return false;
    }

    //assign paramters
    fs["numInnerCornersWidth"] >> _numInnerCornersWidth;
    fs["numInnerCornersHeight"] >> _numInnerCornersHeight;
    fs["squareSizeMM"] >> _squareSizeMM;
    fs["captureDelayMS"] >> _captureDelayMS;
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
    FileStorage fs(filename, FileStorage::WRITE);

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
//!< Find the chessboard corners with CV C-function interface
bool SLCVCalibration::findChessboard(cv::Mat& frame,
                                            cv::Size& size,
                                            vector<cv::Point2f>& corners,
                                            int flags)
{
    // C++ version with STL vector crashes in visual studio
    //bool ok = cv::findChessboardCorners(frame, size, corners, flags);

    int count = 0;
    CvMat image = frame;
    cv::Mat tmpCorners;
    tmpCorners.create(size.area(), 1, CV_32FC2);
    bool ok = cvFindChessboardCorners(&image,
        size,
        reinterpret_cast<CvPoint2D32f*>(tmpCorners.data),
        &count,
        flags) > 0;
    corners.assign((Point2f*)tmpCorners.datastart, (Point2f*)tmpCorners.dataend);
    return ok;
}
//-----------------------------------------------------------------------------
void SLCVCalibration::calibrate()
{
    clear();
    _state = CAPTURING;
}
//-----------------------------------------------------------------------------
void SLCVCalibration::calculate(string saveDir)
{
    _state = CALCULATING;

    vector<Mat> rvecs, tvecs;
    vector<float> reprojErrs;
    double totalAvgErr = 0;
    Size boardSize;
    boardSize.width = _numInnerCornersWidth;
    boardSize.height = _numInnerCornersHeight;

    int flag = 0;
    flag |= CALIB_FIX_PRINCIPAL_POINT;
    flag |= CALIB_ZERO_TANGENT_DIST;
    flag |= CALIB_FIX_ASPECT_RATIO;

    bool ok = runCalibration(_imageSize, _intrinsics, _distortion, _imagePoints, rvecs, tvecs, reprojErrs,
                             totalAvgErr, boardSize, _squareSizeMM, flag);
    cout << (ok ? "Calibration succeeded" : "Calibration failed")
         << ". avg re projection error = " << totalAvgErr << endl;

    if (ok)
    {   cout << "intrinsics" << _intrinsics << endl;
        cout << "distortion" << _distortion << endl;

        saveCameraParams(_imageSize, _intrinsics, _distortion, rvecs, tvecs, reprojErrs, _imagePoints,
                         totalAvgErr, flag, saveDir + _calibFileName, boardSize, _squareSizeMM);

        //calculation successful
        calculateCameraFOV();
        _reprojectionError = (float)totalAvgErr;
        _state = CALIBRATED;
    }
}
//-----------------------------------------------------------------------------
void SLCVCalibration::addImage(cv::Mat image)
{
    //set image size
    _imageSize = image.size();

    //check if we have a timeout
    bool timeOut = clock() - _prevTimestamp > _captureDelayMS * 1e-3 * CLOCKS_PER_SEC ? true : false;

    //try to detect chessboard
    vector<cv::Point2f> corners;
    Size boardSize = cv::Size(_numInnerCornersWidth, _numInnerCornersHeight);
    int flags = CALIB_CB_ADAPTIVE_THRESH | CALIB_CB_NORMALIZE_IMAGE | CALIB_CB_FAST_CHECK;

    bool found = findChessboard(image, boardSize, corners, flags);

    //draw colored points
    //if chessboard was not found reset timer
    if(!found)
    {
        //reset timer
        _prevTimestamp = clock();
    }
    //if chessboard was found and timer is down, add detected points to container
    else if(timeOut)
    {
        Mat imageGray;
        cv::cvtColor(image, imageGray, COLOR_BGR2GRAY);
        cornerSubPix(imageGray, corners, Size(11,11),
            Size(-1,-1), TermCriteria(TermCriteria::EPS+TermCriteria::COUNT, 30, 0.1));

        //debug save image
        stringstream ss;
        ss << "imageIn_" << _numCaptured << ".png";
        cv::imwrite(ss.str(), image);

        //add detected points
        _imagePoints.push_back(corners);
        _numCaptured++;

        //reset timer
        _prevTimestamp = clock();

        //simulate a snapshot
        cv::bitwise_not(image, image);
    }

    // Draw the corners
    if(found)
        drawChessboardCorners(image, boardSize, Mat(corners), found);
}
//-----------------------------------------------------------------------------