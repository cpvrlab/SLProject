//#############################################################################
//  File:      CVTrackedAruco.h
//  Date:      Winter 2016
//  Codestyle: https://github.com/cpvrlab/SLProject/wiki/SLProject-Coding-Style
//  Authors:   Marcus Hudritsch, Michael Goettlicher
//  License:   This software is provided under the GNU General Public License
//             Please visit: http://opensource.org/licenses/GPL-3.0
//#############################################################################

#ifndef CVTrackedAruco_H
#define CVTrackedAruco_H

/*
The OpenCV library version 3.4 or above with extra module must be present.
If the application captures the live video stream with OpenCV you have
to define in addition the constant APP_USES_CVCAPTURE.
All classes that use OpenCV begin with CV.
See also the class docs for CVCapture, CVCalibration and CVTracked
for a good top down information.
*/

#include <CVTypedefs.h>
#include <CVTracked.h>
#include <SLFileStorage.h>
#include <opencv2/aruco.hpp>

//-----------------------------------------------------------------------------
//! ArUco Parameters loaded from configuration file.
class CVArucoParams
{
public:
    CVArucoParams() : edgeLength(0.06f),
                      arucoDictionaryId(0),
                      filename("aruco_detector_params.yml")
    {
#if CV_MAJOR_VERSION < 4 || CV_MINOR_VERSION < 7
        arucoParams = cv::aruco::DetectorParameters::create();
#else
        arucoParams = cv::aruco::DetectorParameters();
#endif
    }

    bool loadFromFile(string calibIniPath)
    {
        string path = calibIniPath + filename;

        SLstring      paramString = SLFileStorage::readIntoString(path, IOK_config);
        CVFileStorage fs(paramString, CVFileStorage::READ | CVFileStorage::MEMORY);
        if (!fs.isOpened())
        {
            cout << "Could not find parameter file for ArUco tracking!" << endl;
            cout << "Tried " << path << endl;
            return false;
        }

#if CV_MAJOR_VERSION < 4 || CV_MINOR_VERSION < 7
        fs["adaptiveThreshWinSizeMin"] >> arucoParams->adaptiveThreshWinSizeMin;
        fs["adaptiveThreshWinSizeMax"] >> arucoParams->adaptiveThreshWinSizeMax;
        fs["adaptiveThreshWinSizeStep"] >> arucoParams->adaptiveThreshWinSizeStep;
        fs["adaptiveThreshConstant"] >> arucoParams->adaptiveThreshConstant;
        fs["minMarkerPerimeterRate"] >> arucoParams->minMarkerPerimeterRate;
        fs["maxMarkerPerimeterRate"] >> arucoParams->maxMarkerPerimeterRate;
        fs["polygonalApproxAccuracyRate"] >> arucoParams->polygonalApproxAccuracyRate;
        fs["minCornerDistanceRate"] >> arucoParams->minCornerDistanceRate;
        fs["minDistanceToBorder"] >> arucoParams->minDistanceToBorder;
        fs["cornerRefinementMethod"] >> arucoParams->cornerRefinementMethod; // cv::aruco::CornerRefineMethod
        fs["cornerRefinementWinSize"] >> arucoParams->cornerRefinementWinSize;
        fs["cornerRefinementMaxIterations"] >> arucoParams->cornerRefinementMaxIterations;
        fs["cornerRefinementMinAccuracy"] >> arucoParams->cornerRefinementMinAccuracy;
        fs["markerBorderBits"] >> arucoParams->markerBorderBits;
        fs["perspectiveRemovePixelPerCell"] >> arucoParams->perspectiveRemovePixelPerCell;
        fs["perspectiveRemoveIgnoredMarginPerCell"] >> arucoParams->perspectiveRemoveIgnoredMarginPerCell;
        fs["maxErroneousBitsInBorderRate"] >> arucoParams->maxErroneousBitsInBorderRate;
        fs["edgeLength"] >> edgeLength;
        fs["arucoDictionaryId"] >> arucoDictionaryId;
#else
        fs["adaptiveThreshWinSizeMin"] >> arucoParams.adaptiveThreshWinSizeMin;
        fs["adaptiveThreshWinSizeMax"] >> arucoParams.adaptiveThreshWinSizeMax;
        fs["adaptiveThreshWinSizeStep"] >> arucoParams.adaptiveThreshWinSizeStep;
        fs["adaptiveThreshConstant"] >> arucoParams.adaptiveThreshConstant;
        fs["minMarkerPerimeterRate"] >> arucoParams.minMarkerPerimeterRate;
        fs["maxMarkerPerimeterRate"] >> arucoParams.maxMarkerPerimeterRate;
        fs["polygonalApproxAccuracyRate"] >> arucoParams.polygonalApproxAccuracyRate;
        fs["minCornerDistanceRate"] >> arucoParams.minCornerDistanceRate;
        fs["minDistanceToBorder"] >> arucoParams.minDistanceToBorder;
        fs["cornerRefinementMethod"] >> arucoParams.cornerRefinementMethod; // cv::aruco::CornerRefineMethod
        fs["cornerRefinementWinSize"] >> arucoParams.cornerRefinementWinSize;
        fs["cornerRefinementMaxIterations"] >> arucoParams.cornerRefinementMaxIterations;
        fs["cornerRefinementMinAccuracy"] >> arucoParams.cornerRefinementMinAccuracy;
        fs["markerBorderBits"] >> arucoParams.markerBorderBits;
        fs["perspectiveRemovePixelPerCell"] >> arucoParams.perspectiveRemovePixelPerCell;
        fs["perspectiveRemoveIgnoredMarginPerCell"] >> arucoParams.perspectiveRemoveIgnoredMarginPerCell;
        fs["maxErroneousBitsInBorderRate"] >> arucoParams.maxErroneousBitsInBorderRate;
        fs["edgeLength"] >> edgeLength;
        fs["arucoDictionaryId"] >> arucoDictionaryId;
#endif

#if CV_MAJOR_VERSION < 4 || CV_MINOR_VERSION < 7
        dictionary = cv::aruco::getPredefinedDictionary(cv::aruco::PREDEFINED_DICTIONARY_NAME(arucoDictionaryId));
#else
        dictionary = cv::aruco::getPredefinedDictionary(cv::aruco::PredefinedDictionaryType(arucoDictionaryId));
#endif

        return true;
    }

#if CV_MAJOR_VERSION < 4 || CV_MINOR_VERSION < 7
    cv::Ptr<cv::aruco::Dictionary>         dictionary;  //!< predefined dictionary
    cv::Ptr<cv::aruco::DetectorParameters> arucoParams; //!< detector parameter structure for aruco detection function
#else
    cv::aruco::DetectorParameters arucoParams; //!< detector parameter structure for aruco detection function
    cv::aruco::Dictionary         dictionary;  //!< predefined dictionary
#endif

    float  edgeLength;          //!< marker edge length
    int    arucoDictionaryId;   //!< id of aruco dictionary
    string arucoDetectorParams; //!< todo: put in one file
    string filename;            //!< parameter filename
};

//-----------------------------------------------------------------------------
//! OpenCV ArUco marker tracker class derived from CVTracked
/*! Tracking class for ArUco markers tracking. See the official OpenCV docs on
ArUco markers: http://docs.opencv.org/3.1.0/d5/dae/tutorial_aruco_detection.html
The aruco marker used in the SLProject are printed in a PDF stored in the
data/Calibration folder. They use the dictionary 0 and where generated with the
functions CVTrackedAruco::drawArucoMarkerBoard and
CVTrackedAruco::drawArucoMarker.
*/
class CVTrackedAruco : public CVTracked
{
public:
    explicit CVTrackedAruco(int arucoID, string calibIniPath);

    bool track(CVMat          imageGray,
               CVMat          imageRgb,
               CVCalibration* calib);

    //! Helper function to draw and save an aruco marker board image
    static void drawArucoMarkerBoard(int           dictionaryId,
                                     int           numMarkersX,
                                     int           numMarkersY,
                                     float         markerEdgeLengthM,
                                     float         markerSepaM,
                                     const string& imgName,
                                     float         dpi       = 254.0f,
                                     bool          showImage = false);

    //! Helper function to draw and save an aruco marker set
    static void drawArucoMarker(int dictionaryId,
                                int minMarkerId,
                                int maxMarkerId,
                                int markerSizePX = 200);

    static CVArucoParams params; //!< Parameter class instance

protected:
    bool trackAll(CVMat          imageGray,
                  CVMat          imageRgb,
                  CVCalibration* calib,
                  CVRect         roi = CVRect(0, 0, 0, 0));

    vector<int> arucoIDs;       //!< detected Aruco marker IDs
    CVVMatx44f  objectViewMats; //!< object view matrices for all found markers

private:
    static bool paramsLoaded; //!< Flag for loaded parameters

    int    _arucoID;          //!< Aruco Marker ID for this node
    string _calibIniPath;
};
//-----------------------------------------------------------------------------
#endif // CVTrackedAruco_H