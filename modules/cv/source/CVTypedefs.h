//#############################################################################
//  File:      CVTypedefs.h
//  Date:      Autumn 2016
//  Codestyle: https://github.com/cpvrlab/SLProject/wiki/SLProject-Coding-Style
//  Authors:   Marcus Hudritsch
//  License:   This software is provided under the GNU General Public License
//             Please visit: http://opensource.org/licenses/GPL-3.0
//#############################################################################

#ifndef CVTYPEDEFS_H
#define CVTYPEDEFS_H

/*
The OpenCV library version 3.4 with extra module must be present.
If the application captures the live video stream with OpenCV you have
to define in addition the constant APP_USES_CVCAPTURE.
All classes that use OpenCV begin with CV.
See also the class docs for CVCapture, CVCalibration and CVTracked
for a good top down information.
*/

#include <vector>
#include <opencv2/core/core.hpp>
#include <opencv2/core/ocl.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#ifdef __EMSCRIPTEN__
#    include <list>
#endif

#include <opencv2/face.hpp>
#include <opencv2/xfeatures2d.hpp>

using std::vector;

//-----------------------------------------------------------------------------
typedef cv::Mat               CVMat;
typedef cv::Rect              CVRect;
typedef cv::Rect2f            CVRect2f;
typedef cv::Point             CVPoint;
typedef cv::Point2i           CVPoint2i;
typedef cv::Point2f           CVPoint2f;
typedef cv::Point2d           CVPoint2d;
typedef cv::Point3f           CVPoint3f;
typedef cv::Point3d           CVPoint3d;
typedef cv::Vec2b             CVVec2b;
typedef cv::Vec3b             CVVec3b;
typedef cv::Vec4b             CVVec4b;
typedef cv::Vec2f             CVVec2f;
typedef cv::Vec2d             CVVec2d;
typedef cv::Vec3f             CVVec3f;
typedef cv::Vec3d             CVVec3d;
typedef cv::Vec4f             CVVec4f;
typedef cv::Size              CVSize;
typedef cv::Size2i            CVSize2i;
typedef cv::Size2f            CVSize2f;
typedef cv::Matx33f           CVMatx33f;
typedef cv::Matx44f           CVMatx44f;
typedef cv::KeyPoint          CVKeyPoint;
typedef cv::FileStorage       CVFileStorage;
typedef cv::DMatch            CVDMatch;
typedef cv::InputArray        CVInputArray;
typedef cv::OutputArray       CVOutputArray;
typedef cv::Feature2D         CVFeature2D;
typedef cv::CascadeClassifier CVCascadeClassifier;
#ifndef __EMSCRIPTEN__
typedef cv::VideoCapture CVVideoCapture;
#endif
typedef cv::face::Facemark CVFacemark;

// 1D STL vectors
typedef vector<cv::Mat>      CVVMat;
typedef vector<cv::Rect>     CVVRect;
typedef vector<cv::Point>    CVVPoint;
typedef vector<cv::Point2i>  CVVPoint2i;
typedef vector<cv::Point2f>  CVVPoint2f;
typedef vector<cv::Point2d>  CVVPoint2d;
typedef vector<cv::Point3f>  CVVPoint3f;
typedef vector<cv::Point3d>  CVVPoint3d;
typedef vector<cv::Vec2b>    CVVVec2b;
typedef vector<cv::Vec3b>    CVVVec3b;
typedef vector<cv::Vec4b>    CVVVec4b;
typedef vector<cv::Vec2f>    CVVVec2f;
typedef vector<cv::Vec3f>    CVVVec3f;
typedef vector<cv::Vec4f>    CVVVec4f;
typedef vector<cv::Size>     CVVSize;
typedef vector<cv::KeyPoint> CVVKeyPoint;
typedef vector<cv::DMatch>   CVVDMatch;
typedef vector<cv::Matx44f>  CVVMatx44f;
typedef vector<cv::Matx33f>  CVVMatx33f;

// 2D STL vectors
typedef vector<vector<cv::Point>>    CVVVPoint;
typedef vector<vector<cv::Point2i>>  CVVVPoint2i;
typedef vector<vector<cv::Point2f>>  CVVVPoint2f;
typedef vector<vector<cv::Point2d>>  CVVVPoint2d;
typedef vector<vector<cv::Point3i>>  CVVVPoint3i;
typedef vector<vector<cv::Point3f>>  CVVVPoint3f;
typedef vector<vector<cv::Point3d>>  CVVVPoint3d;
typedef vector<vector<cv::DMatch>>   CVVVDMatch;
typedef vector<vector<cv::KeyPoint>> CVVVKeyPoint;
//-----------------------------------------------------------------------------
#endif // CVTYPEDEFS_H
