//#############################################################################
//  File:      CVTrackedAruco.cpp
//  Author:    Michael Goettlicher, Marcus Hudritsch
//  Date:      Winter 2016
//  Codestyle: https://github.com/cpvrlab/SLProject/wiki/SLProject-Coding-Style
//  Copyright: Marcus Hudritsch, Michael Goettlicher
//             This software is provide under the GNU General Public License
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

using namespace cv;
//-----------------------------------------------------------------------------
// Initialize static variables
bool          CVTrackedAruco::paramsLoaded = false;
vector<int>   CVTrackedAruco::arucoIDs;
CVVMatx44f    CVTrackedAruco::objectViewMats;
CVArucoParams CVTrackedAruco::params;
//-----------------------------------------------------------------------------
CVTrackedAruco::CVTrackedAruco(int arucoID)
{
    _arucoID = arucoID;
}
//-----------------------------------------------------------------------------
//! Tracks the all Aruco markers in the given image for the first sceneview
bool CVTrackedAruco::track(CVMat          imageGray,
                           CVMat          imageRgb,
                           CVCalibration* calib)
{

    assert(!imageGray.empty() && "ImageGray is empty");
    assert(!imageRgb.empty() && "ImageRGB is empty");
    assert(!calib->cameraMat().empty() && "Calibration is empty");

    // Load aruco parameter once
    if (!paramsLoaded)
    {
        paramsLoaded = params.loadFromFile();
        if (!paramsLoaded)
            Utils::exitMsg("SLProject",
                           "CVTrackedAruco::track: Failed to load Aruco parameters.",
                           __LINE__,
                           __FILE__);
    }
    if (params.arucoParams.empty() || params.dictionary.empty())
    {
        Utils::warnMsg("SLProject",
                       "CVTrackedAruco::track: Aruco paramters are empty.",
                       __LINE__,
                       __FILE__);
        return false;
    }

    ////////////
    // Detect //
    ////////////

    float startMS = _timer.elapsedTimeInMilliSec();

    arucoIDs.clear();
    objectViewMats.clear();
    CVVVPoint2f corners, rejected;

    aruco::detectMarkers(imageGray,
                         params.dictionary,
                         corners,
                         arucoIDs,
                         params.arucoParams,
                         rejected);

    CVTracked::detectTimesMS.set(_timer.elapsedTimeInMilliSec() - startMS);

    if (!arucoIDs.empty())
    {
        if (_drawDetection)
            aruco::drawDetectedMarkers(imageRgb,
                                       corners,
                                       arucoIDs,
                                       Scalar(0, 0, 255));

        /////////////////////
        // Pose Estimation //
        /////////////////////

        startMS = _timer.elapsedTimeInMilliSec();

        cout << "Aruco IdS: " << arucoIDs.size() << " : ";

        //find the camera extrinsic parameters (rVec & tVec)
        CVVPoint3d rVecs, tVecs;
        aruco::estimatePoseSingleMarkers(corners,
                                         params.edgeLength,
                                         calib->cameraMat(),
                                         calib->distortion(),
                                         rVecs,
                                         tVecs);

        CVTracked::poseTimesMS.set(_timer.elapsedTimeInMilliSec() - startMS);

        // Get the object view matrix for all aruco markers
        for (size_t i = 0; i < arucoIDs.size(); ++i)
        {
            cout << arucoIDs[i] << ",";
            CVMatx44f ovm = createGLMatrix(cv::Mat(tVecs[i]), cv::Mat(rVecs[i]));
            objectViewMats.push_back(ovm);
        }
        cout << endl;
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
/*! CVTrackedAruco::drawArucoMarkerBoard draws and saves an aruco board
into an image.
\param dictionaryId integer id of the dictionary
\param numMarkersX NO. of markers in x-direction
\param numMarkersY NO. of markers in y-direction
\param markerEdgeM Length of one marker in meters
\param markerSepaM Separation between markers in meters
\param imgName Image filename inklusive format extension
\param dpi Dots per inch (default 256)
\param showImage Shows image in window (default false)
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
    cv::Ptr<aruco::Dictionary> dictionary =
      aruco::getPredefinedDictionary(aruco::PREDEFINED_DICTIONARY_NAME(dictionaryId));

    cv::Ptr<aruco::GridBoard> board = aruco::GridBoard::create(numMarkersX,
                                                               numMarkersY,
                                                               markerEdgeM,
                                                               markerSepaM,
                                                               dictionary);
    CVSize                    imageSize;
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
        waitKey(0);
    }

    imwrite(imgName, boardImage);
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

    using namespace aruco;

    cv::Ptr<Dictionary> dict = getPredefinedDictionary(PREDEFINED_DICTIONARY_NAME(dictionaryId));
    if (maxMarkerId > dict->bytesList.rows)
        maxMarkerId = dict->bytesList.rows;

    CVMat markerImg;

    for (int i = minMarkerId; i < maxMarkerId; ++i)
    {
        drawMarker(dict, i, markerSizePX, markerImg, 1);
        imwrite(Utils::formatString("ArucoMarker_Dict%d_%dpx_Id%d.png",
                                    dictionaryId,
                                    markerSizePX,
                                    i),
                markerImg);
    }
}
//-----------------------------------------------------------------------------
