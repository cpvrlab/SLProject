#include "WAIImageStabilizedOrientation.h"

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <Utils.h>
#include <F2FTransform.h>

bool WAIImageStabilizedOrientation::findCameraOrientationDifferenceF2FHorizon(const SLVec3f& horizon,
                                                                              cv::Mat        imageGray, //for corner extraction
                                                                              cv::Mat&       imageRgb,
                                                                              const cv::Mat& intrinsic,
                                                                              float          scaleToGray,
                                                                              bool           decorate)
{
    //initialization
    if (_lastImageGray.empty())
    {
        _lastImageGray = imageGray.clone(); //todo: maybe clone not needed
        return false;
    }

    //extract fast corners on current image
    _lastPts.clear();
    _lastKeyPts.clear();
    cv::FAST(_lastImageGray, _lastKeyPts, _fIniThFAST, true);
    _lastPts.reserve(_lastKeyPts.size());
    for (int i = 0; i < _lastKeyPts.size(); i++)
        _lastPts.push_back(_lastKeyPts[i].pt);
    //Utils::log("WAI", "num features extracted: %d", _lastPts.size());

    if (_lastPts.size() < 10)
    {
        _lastImageGray = imageGray.clone();
        return false;
    }

    F2FTransform::opticalFlowMatch(_lastImageGray,
                                   imageGray,
                                   _lastPts,
                                   _currPts,
                                   _inliers,
                                   _err);

    F2FTransform::filterPoints(_lastPts,
                               _currPts,
                               _lastGoodPts,
                               _currGoodPts,
                               _inliers,
                               _err);

    float xAngRAD = 0.0f, yAngRAD = 0.0f, zAngRAD = 0.0f;

    //estimate z from horizon
    bool success = F2FTransform::estimateRotXY(intrinsic,
                                               _lastGoodPts,
                                               _currGoodPts,
                                               xAngRAD,
                                               yAngRAD,
                                               zAngRAD,
                                               _inliers);

    if (success)
    {
        if (decorate)
        {
            std::vector<cv::Point2f> _lastRealGoodPts;
            std::vector<cv::Point2f> _currRealGoodPts;
            F2FTransform::filterPoints(_lastGoodPts,
                                       _currGoodPts,
                                       _lastRealGoodPts,
                                       _currRealGoodPts,
                                       _inliers,
                                       _err);

            cv::Point2f r(2.f, 2.f);
            for (unsigned int i = 0; i < _lastRealGoodPts.size(); i++)
            {
                cv::Point2f p1 = _lastRealGoodPts[i] * scaleToGray;
                cv::Point2f p2 = _currRealGoodPts[i] * scaleToGray;
                cv::line(imageRgb, p1, p2, cv::Scalar(0, 255, 0));
                cv::rectangle(imageRgb, (p1 - r) * scaleToGray, (p1 + r) * scaleToGray, CV_RGB(255, 0, 0));
            }
        }

        cv::Mat Rx, Ry, Rz, Tcw;
        _xAngRAD += xAngRAD;
        _yAngRAD += yAngRAD;
        _zAngRAD += zAngRAD;
        //std::cout << "_xAngRAD: " << _xAngRAD * RAD2DEG << std::endl;
        //std::cout << "_yAngRAD: " << _yAngRAD * RAD2DEG << std::endl;
        //std::cout << "_zAngRAD: " << _zAngRAD * RAD2DEG << std::endl;
        Utils::log("WAI track", "x: %.0f y: %.0f z: %.0f", _xAngRAD * Utils::RAD2DEG, _yAngRAD * Utils::RAD2DEG, _zAngRAD * Utils::RAD2DEG);
        /*
        F2FTransform::eulerToMat(_xAngRAD, _yAngRAD, _zAngRAD, Rx, Ry, Rz);
        Tcw = Rx * Ry * Rz;
        cv::Mat pose = cv::Mat::eye(4, 4, CV_32F);
        //?????????????????
        //pose.at<float>(2, 3) = -1.5; //?????????????????
        pose = Tcw * pose;
        //pos.copyTo(_objectViewMat);
         */
    }

    _lastImageGray = imageGray.clone();

    return success;
}

bool WAIImageStabilizedOrientation::findCameraOrientationDifferenceF2F(cv::Mat        imageGray,
                                                                       cv::Mat&       imageRgb,
                                                                       const cv::Mat& intrinsic,
                                                                       float          scaleToGray,
                                                                       bool           decorate)
{
    //initialization
    if (_lastImageGray.empty())
    {
        _lastImageGray = imageGray.clone(); //todo: maybe clone not needed
        return false;
    }

    //extract fast corners on current image
    _lastPts.clear();
    _lastKeyPts.clear();
    cv::FAST(_lastImageGray, _lastKeyPts, _fIniThFAST, true);
    _lastPts.reserve(_lastKeyPts.size());
    for (int i = 0; i < _lastKeyPts.size(); i++)
        _lastPts.push_back(_lastKeyPts[i].pt);
    //Utils::log("WAI", "num features extracted: %d", _lastPts.size());

    if (_lastPts.size() < 10)
    {
        _lastImageGray = imageGray.clone();
        return false;
    }

    F2FTransform::opticalFlowMatch(_lastImageGray,
                                   imageGray,
                                   _lastPts,
                                   _currPts,
                                   _inliers,
                                   _err);

    F2FTransform::filterPoints(_lastPts,
                               _currPts,
                               _lastGoodPts,
                               _currGoodPts,
                               _inliers,
                               _err);

    float xAngRAD, yAngRAD, zAngRAD;
    bool  success = F2FTransform::estimateRotXYZ(intrinsic, _lastGoodPts, _currGoodPts, xAngRAD, yAngRAD, zAngRAD, _inliers);

    if (success)
    {
        if (decorate)
        {
            std::vector<cv::Point2f> _lastRealGoodPts;
            std::vector<cv::Point2f> _currRealGoodPts;
            F2FTransform::filterPoints(_lastGoodPts,
                                       _currGoodPts,
                                       _lastRealGoodPts,
                                       _currRealGoodPts,
                                       _inliers,
                                       _err);

            cv::Point2f r(2.f, 2.f);
            for (unsigned int i = 0; i < _lastRealGoodPts.size(); i++)
            {
                cv::Point2f p1 = _lastRealGoodPts[i] * scaleToGray;
                cv::Point2f p2 = _currRealGoodPts[i] * scaleToGray;
                cv::line(imageRgb, p1, p2, cv::Scalar(0, 255, 0));
                cv::rectangle(imageRgb, (p1 - r) * scaleToGray, (p1 + r) * scaleToGray, CV_RGB(255, 0, 0));
            }
        }

        cv::Mat Rx, Ry, Rz, Tcw;
        _xAngRAD += xAngRAD;
        _yAngRAD += yAngRAD;
        _zAngRAD += zAngRAD;
        //std::cout << "_xAngRAD: " << _xAngRAD * RAD2DEG << std::endl;
        //std::cout << "_yAngRAD: " << _yAngRAD * RAD2DEG << std::endl;
        //std::cout << "_zAngRAD: " << _zAngRAD * RAD2DEG << std::endl;
        Utils::log("WAI track", "x: %.0f y: %.0f z: %.0f", _xAngRAD * Utils::RAD2DEG, _yAngRAD * Utils::RAD2DEG, _zAngRAD * Utils::RAD2DEG);
        /*
        F2FTransform::eulerToMat(_xAngRAD, _yAngRAD, _zAngRAD, Rx, Ry, Rz);
        Tcw = Rx * Ry * Rz;
        cv::Mat pose = cv::Mat::eye(4, 4, CV_32F);
        //?????????????????
        //pose.at<float>(2, 3) = -1.5; //?????????????????
        pose = Tcw * pose;
        //pos.copyTo(_objectViewMat);
         */
    }

    _lastImageGray = imageGray.clone();

    return success;
}
