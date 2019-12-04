//#############################################################################
//  File:      CVTrackedAruco.h
//  Author:    Michael Goettlicher, Marcus Hudritsch
//  Date:      Winter 2016
//  Codestyle: https://github.com/cpvrlab/SLProject/wiki/SLProject-Coding-Style
//  Copyright: Marcus Hudritsch, Michael Goettlicher
//             This software is provide under the GNU General Public License
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
#include <opencv2/aruco.hpp>
#include <SLApplication.h>
//-----------------------------------------------------------------------------
//! ArUco Paramters loaded from configuration file.
class CVArucoParams
{
public:
    CVArucoParams() : edgeLength(0.06f),
                      arucoDictionaryId(0),
                      filename("aruco_detector_params.yml")
    {
        arucoParams = cv::aruco::DetectorParameters::create();
    }

    bool loadFromFile()
    {
        string        path = SLApplication::calibIniPath + filename;
        CVFileStorage fs(path, cv::FileStorage::READ);
        if (!fs.isOpened())
        {
            cout << "Could not find parameter file for ArUco tracking!" << endl;
            cout << "Tried " << SLApplication::calibIniPath + filename << endl;
            return false;
        }

        fs["adaptiveThreshWinSizeMin"] >> arucoParams->adaptiveThreshWinSizeMin;
        fs["adaptiveThreshWinSizeMax"] >> arucoParams->adaptiveThreshWinSizeMax;
        fs["adaptiveThreshWinSizeStep"] >> arucoParams->adaptiveThreshWinSizeStep;
        fs["adaptiveThreshConstant"] >> arucoParams->adaptiveThreshConstant;
        fs["minMarkerPerimeterRate"] >> arucoParams->minMarkerPerimeterRate;
        fs["maxMarkerPerimeterRate"] >> arucoParams->maxMarkerPerimeterRate;
        fs["polygonalApproxAccuracyRate"] >> arucoParams->polygonalApproxAccuracyRate;
        fs["minCornerDistanceRate"] >> arucoParams->minCornerDistanceRate;
        fs["minDistanceToBorder"] >> arucoParams->minDistanceToBorder;
        //fs["minMarkerDistanceRate"] >> arucoParams->minMarkerDistanceRate; //achtung minMarkerDistance -> minMarkerDistanceRate
        //fs["doCornerRefinement"] >> arucoParams->doCornerRefinement; //does not exist anymore in opencv 3.4.0
        fs["cornerRefinementMethod"] >> arucoParams->cornerRefinementMethod; //cv::aruco::CornerRefineMethod
        fs["cornerRefinementWinSize"] >> arucoParams->cornerRefinementWinSize;
        fs["cornerRefinementMaxIterations"] >> arucoParams->cornerRefinementMaxIterations;
        fs["cornerRefinementMinAccuracy"] >> arucoParams->cornerRefinementMinAccuracy;
        fs["markerBorderBits"] >> arucoParams->markerBorderBits;
        fs["perspectiveRemovePixelPerCell"] >> arucoParams->perspectiveRemovePixelPerCell;
        fs["perspectiveRemoveIgnoredMarginPerCell"] >> arucoParams->perspectiveRemoveIgnoredMarginPerCell;
        fs["maxErroneousBitsInBorderRate"] >> arucoParams->maxErroneousBitsInBorderRate;
        fs["edgeLength"] >> edgeLength;
        fs["arucoDictionaryId"] >> arucoDictionaryId;
        dictionary = cv::aruco::getPredefinedDictionary(cv::aruco::PREDEFINED_DICTIONARY_NAME(arucoDictionaryId));

        return true;
    }

    cv::Ptr<cv::aruco::DetectorParameters> arucoParams; //!< detector parameter structure for aruco detection function
    cv::Ptr<cv::aruco::Dictionary>         dictionary;  //!< predefined dictionary

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
    explicit CVTrackedAruco(int arucoID);

    bool track(CVMat          imageGray,
               CVMat          imageRgb,
               CVCalibration* calib) final;

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

private:
    static bool        paramsLoaded;   //!< Flag for loaded parameters
    static vector<int> arucoIDs;       //!< detected Aruco marker IDs
    static CVVMatx44f  objectViewMats; //!< object view matrices for all found markers

    int _arucoID; //!< Aruco Marker ID for this node
};
//-----------------------------------------------------------------------------
#endif // CVTrackedAruco_H
