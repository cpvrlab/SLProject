//#############################################################################
//  File:      CVTrackedAruco.cpp
//  Date:      Winter 2016
//  Codestyle: https://github.com/cpvrlab/SLProject/wiki/SLProject-Coding-Style
//  Authors:   Marcus Hudritsch, Michael Goettlicher, Marino von Wattenwyl
//  License:   This software is provided under the GNU General Public License
//             Please visit: http://opensource.org/licenses/GPL-3.0
//#############################################################################

/*
The OpenCV library version 3.4 or above with extra module must be present.
If the application captures the live video stream with OpenCV you have
to define in addition the constant APP_USES_CVCAPTURE.
All classes that use OpenCV begin with CV.
See also the class docs for CVCapture, CVCalibration and CVTracked
for a good top down information.
*/
#include <CVTrackedAruco.h>
#include <Utils.h>
#include <Profiler.h>

//-----------------------------------------------------------------------------
// Initialize static variables
bool          CVTrackedAruco::paramsLoaded = false;
CVArucoParams CVTrackedAruco::params;
//-----------------------------------------------------------------------------
CVTrackedAruco::CVTrackedAruco(int arucoID, string calibIniPath)
  : _calibIniPath(calibIniPath),
    _arucoID(arucoID)
{
}
//-----------------------------------------------------------------------------
//! Tracks the all Aruco markers in the given image for the first sceneview
bool CVTrackedAruco::track(CVMat          imageGray,
                           CVMat          imageRgb,
                           CVCalibration* calib)
{
    if (!trackAll(imageGray, imageRgb, calib))
    {
        return false;
    }

    if (!arucoIDs.empty())
    {
        // Find the marker with the matching id
        for (size_t i = 0; i < arucoIDs.size(); ++i)
        {
            if (arucoIDs[i] == _arucoID)
            {
                _objectViewMat = objectViewMats[i];
                return true;
            }
        }
    }

    return false;
}
//-----------------------------------------------------------------------------
bool CVTrackedAruco::trackAll(CVMat          imageGray,
                              CVMat          imageRgb,
                              CVCalibration* calib,
                              CVRect         roi)
{
    PROFILE_FUNCTION();

    assert(!imageGray.empty() && "ImageGray is empty");
    assert(!imageRgb.empty() && "ImageRGB is empty");
    assert(!calib->cameraMat().empty() && "Calibration is empty");

    // Load aruco parameter once
    if (!paramsLoaded)
    {
        paramsLoaded = params.loadFromFile(_calibIniPath);
        if (!paramsLoaded)
            Utils::exitMsg("SLProject",
                           "CVTrackedAruco::track: Failed to load Aruco parameters.",
                           __LINE__,
                           __FILE__);
    }

#if CV_MAJOR_VERSION < 4 || CV_MINOR_VERSION < 7
    if (params.arucoParams.empty() || params.dictionary.empty())
    {
        Utils::warnMsg("SLProject",
                       "CVTrackedAruco::track: Aruco paramters are empty.",
                       __LINE__,
                       __FILE__);
        return false;
    }
#endif

    ////////////
    // Detect //
    ////////////

    CVMat croppedImageGray = roi.empty() ? imageGray : imageGray(roi);

    float startMS = _timer.elapsedTimeInMilliSec();

    arucoIDs.clear();
    objectViewMats.clear();
    CVVVPoint2f corners, rejected;

#if CV_MAJOR_VERSION < 4 || CV_MINOR_VERSION < 7
    cv::aruco::detectMarkers(croppedImageGray,
                             params.dictionary,
                             corners,
                             arucoIDs,
                             params.arucoParams,
                             rejected);
#else
    cv::aruco::ArucoDetector detector(params.dictionary, params.arucoParams);
    detector.detectMarkers(croppedImageGray,
                           corners,
                           arucoIDs,
                           rejected);
#endif

    for (auto& corner : corners)
    {
        for (auto& j : corner)
        {
            j.x += (float)roi.x;
            j.y += (float)roi.y;
        }
    }

    CVTracked::detectTimesMS.set(_timer.elapsedTimeInMilliSec() - startMS);

    if (!arucoIDs.empty())
    {
        if (_drawDetection)
            cv::aruco::drawDetectedMarkers(imageRgb,
                                           corners,
                                           arucoIDs,
                                           cv::Scalar(0, 0, 255));

        /////////////////////
        // Pose Estimation //
        /////////////////////

        startMS = _timer.elapsedTimeInMilliSec();

        // find the camera extrinsic parameters (rVec & tVec)
        CVVPoint3d rVecs, tVecs;
        cv::aruco::estimatePoseSingleMarkers(corners,
                                             params.edgeLength,
                                             calib->cameraMat(),
                                             calib->distortion(),
                                             rVecs,
                                             tVecs);

        CVTracked::poseTimesMS.set(_timer.elapsedTimeInMilliSec() - startMS);

        // Get the object view matrix for all aruco markers
        for (size_t i = 0; i < arucoIDs.size(); ++i)
        {
            CVMatx44f ovm = createGLMatrix(cv::Mat(tVecs[i]), cv::Mat(rVecs[i]));
            objectViewMats.push_back(ovm);

            if (_drawDetection)
            {
#if CV_MAJOR_VERSION < 4 || CV_MINOR_VERSION < 6
#else
                cv::drawFrameAxes(imageRgb,
                                  calib->cameraMat(),
                                  calib->distortion(),
                                  cv::Mat(rVecs[i]),
                                  cv::Mat(tVecs[i]),
                                  0.01f);
#endif
            }
        }
    }

    return true;
}
//-----------------------------------------------------------------------------
/*! CVTrackedAruco::drawArucoMarkerBoard draws and saves an aruco board
into an image.
@param dictionaryId integer id of the dictionary
@param numMarkersX NO. of markers in x-direction
@param numMarkersY NO. of markers in y-direction
@param markerEdgeM Length of one marker in meters
@param markerSepaM Separation between markers in meters
@param imgName Image filename inklusive format extension
@param dpi Dots per inch (default 256)
@param showImage Shows image in window (default false)
*/
void CVTrackedAruco::drawArucoMarkerBoard(int           dictionaryId,
                                          int           numMarkersX,
                                          int           numMarkersY,
                                          float         markerEdgeM,
                                          float         markerSepaM,
                                          const string& imgName,
                                          float         dpi,
                                          bool          showImage)
{
#if CV_MAJOR_VERSION < 4 || CV_MINOR_VERSION < 7
    cv::Ptr<cv::aruco::Dictionary> dictionary = cv::aruco::getPredefinedDictionary(cv::aruco::PREDEFINED_DICTIONARY_NAME(dictionaryId));
    cv::Ptr<cv::aruco::GridBoard>  board      = cv::aruco::GridBoard::create(numMarkersX,
                                                                       numMarkersY,
                                                                       markerEdgeM,
                                                                       markerSepaM,
                                                                       dictionary);
    CVSize                         imageSize;
    imageSize.width  = (int)((markerEdgeM + markerSepaM) * 100.0f / 2.54f * dpi * (float)numMarkersX);
    imageSize.height = (int)((markerEdgeM + markerSepaM) * 100.0f / 2.54f * dpi * (float)numMarkersY);

    imageSize.width -= (imageSize.width % 4);
    imageSize.height -= (imageSize.height % 4);

    // show created board
    CVMat boardImage;
    board->draw(imageSize, boardImage, 0, 1);

    if (showImage)
    {
        imshow("board", boardImage);
        cv::waitKey(0);
    }
#else
    cv::aruco::Dictionary dictionary = cv::aruco::getPredefinedDictionary(cv::aruco::PredefinedDictionaryType(dictionaryId));
    cv::aruco::GridBoard  board      = cv::aruco::GridBoard(cv::Size(numMarkersX, numMarkersY),
                                                      markerEdgeM,
                                                      markerSepaM,
                                                      dictionary);

    CVSize imageSize;
    imageSize.width  = (int)((markerEdgeM + markerSepaM) * 100.0f / 2.54f * dpi * (float)numMarkersX);
    imageSize.height = (int)((markerEdgeM + markerSepaM) * 100.0f / 2.54f * dpi * (float)numMarkersY);

    imageSize.width -= (imageSize.width % 4);
    imageSize.height -= (imageSize.height % 4);

    // show created board
    CVMat boardImage;
    cv::aruco::drawPlanarBoard(&board,
                               imageSize,
                               boardImage,
                               0,
                               1);
#    ifndef __EMSCRIPTEN__
    if (showImage)
    {
        imshow("board", boardImage);
        cv::waitKey(0);
    }
#    endif
#endif

#ifndef __EMSCRIPTEN__
    imwrite(imgName, boardImage);
#endif
}
//-----------------------------------------------------------------------------
void CVTrackedAruco::drawArucoMarker(int dictionaryId,
                                     int minMarkerId,
                                     int maxMarkerId,
                                     int markerSizePX)
{
    assert(dictionaryId > 0);
    assert(minMarkerId > 0);
    assert(minMarkerId < maxMarkerId);

#if CV_MAJOR_VERSION < 4 || CV_MINOR_VERSION < 7
    cv::Ptr<cv::aruco::Dictionary> dict = getPredefinedDictionary(cv::aruco::PREDEFINED_DICTIONARY_NAME(dictionaryId));
    if (maxMarkerId > dict->bytesList.rows)
        maxMarkerId = dict->bytesList.rows;

    CVMat markerImg;

    for (int i = minMarkerId; i < maxMarkerId; ++i)
    {
        cv::aruco::drawMarker(dict, i, markerSizePX, markerImg, 1);
#    ifndef __EMSCRIPTEN__
        imwrite(Utils::formatString("ArucoMarker_Dict%d_%dpx_Id%d.png",
                                    dictionaryId,
                                    markerSizePX,
                                    i),
                markerImg);
#    endif
    }
#else
    cv::aruco::Dictionary dict = getPredefinedDictionary(cv::aruco::PredefinedDictionaryType(dictionaryId));
    if (maxMarkerId > dict.bytesList.rows)
        maxMarkerId = dict.bytesList.rows;

    CVMat markerImg;

    for (int i = minMarkerId; i < maxMarkerId; ++i)
    {
        cv::aruco::generateImageMarker(dict,
                                       i,
                                       markerSizePX,
                                       markerImg,
                                       1);
#    ifndef __EMSCRIPTEN__
        imwrite(Utils::formatString("ArucoMarker_Dict%d_%dpx_Id%d.png",
                                    dictionaryId,
                                    markerSizePX,
                                    i),
                markerImg);
#    endif
    }
#endif
}
//-----------------------------------------------------------------------------
