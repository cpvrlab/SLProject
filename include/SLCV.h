//#############################################################################
//  File:      SLCV.h
//  Author:    Marcus Hudritsch
//  Date:      Autumn 2016
//  Codestyle: https://github.com/cpvrlab/SLProject/wiki/Coding-Style-Guidelines
//  Copyright: Marcus Hudritsch
//             This software is provide under the GNU General Public License
//             Please visit: http://opensource.org/licenses/GPL-3.0
//#############################################################################

#ifndef SLCV_H
#define SLCV_H

/*
The OpenCV library version 3.4 with extra module must be present.
If the application captures the live video stream with OpenCV you have
to define in addition the constant SL_USES_CVCAPTURE.
All classes that use OpenCV begin with SLCV.
See also the class docs for SLCVCapture, SLCVCalibration and SLCVTracked
for a good top down information.
*/

#include <stdafx.h>
#include <opencv2/opencv.hpp>
#include <opencv2/core/ocl.hpp>
#include <opencv2/xfeatures2d.hpp>
#include <opencv2/face.hpp>

//-----------------------------------------------------------------------------
typedef cv::Mat                     SLCVMat;
typedef cv::Rect                    SLCVRect;
typedef cv::Rect2f                  SLCVRect2f;
typedef cv::Point                   SLCVPoint;
typedef cv::Point2i                 SLCVPoint2i;
typedef cv::Point2f                 SLCVPoint2f;
typedef cv::Point2d                 SLCVPoint2d;
typedef cv::Point3f                 SLCVPoint3f;
typedef cv::Point3d                 SLCVPoint3d;
typedef cv::Size                    SLCVSize;
typedef cv::Size2f                  SLCVSize2f;
typedef cv::KeyPoint                SLCVKeyPoint;
typedef cv::FileStorage             SLCVFileStorage;
typedef cv::DMatch                  SLCVDMatch;
typedef cv::InputArray              SLCVInputArray;
typedef cv::OutputArray             SLCVOutputArray;
typedef cv::Feature2D               SLCVFeature2D;
typedef cv::CascadeClassifier       SLCVCascadeClassifier;
typedef cv::face::Facemark          SLCVFacemark;

// 1D STL vectors
typedef vector<cv::Mat>             SLCVVMat;
typedef vector<cv::Rect>            SLCVVRect;
typedef vector<cv::Point>           SLCVVPoint;
typedef vector<cv::Point2i>         SLCVVPoint2i;
typedef vector<cv::Point2f>         SLCVVPoint2f;
typedef vector<cv::Point2d>         SLCVVPoint2d;
typedef vector<cv::Point3f>         SLCVVPoint3f;
typedef vector<cv::Point3d>         SLCVVPoint3d;
typedef vector<cv::KeyPoint>        SLCVVKeyPoint;
typedef vector<cv::DMatch>          SLCVVDMatch;

// 2D STL vectors
typedef vector<vector<cv::Point>>    SLCVVVPoint;
typedef vector<vector<cv::Point2i>>  SLCVVVPoint2i;
typedef vector<vector<cv::Point2f>>  SLCVVVPoint2f;
typedef vector<vector<cv::Point2d>>  SLCVVVPoint2d;
typedef vector<vector<cv::Point3i>>  SLCVVVPoint3i;
typedef vector<vector<cv::Point3f>>  SLCVVVPoint3f;
typedef vector<vector<cv::Point3d>>  SLCVVVPoint3d;
typedef vector<vector<cv::DMatch>>   SLCVVVDMatch;
typedef vector<vector<cv::KeyPoint>> SLCVVVKeyPoint;
//-----------------------------------------------------------------------------
#endif // SL_CV_H
