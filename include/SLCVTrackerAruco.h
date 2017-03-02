//#############################################################################
//  File:      SLCVTrackerAruco.cpp
//  Author:    Michael Goettlicher, Marcus Hudritsch
//  Date:      Winter 2016
//  Codestyle: https://github.com/cpvrlab/SLProject/wiki/Coding-Style-Guidelines
//  Copyright: Marcus Hudritsch, Michael Goettlicher
//             This software is provide under the GNU General Public License
//             Please visit: http://opensource.org/licenses/GPL-3.0
//#############################################################################

#ifndef SLCVTrackerAruco_H
#define SLCVTrackerAruco_H

/*
The OpenCV library version 3.1 with extra module must be present.
If the application captures the live video stream with OpenCV you have
to define in addition the constant SL_USES_CVCAPTURE.
All classes that use OpenCV begin with SLCV.
See also the class docs for SLCVCapture, SLCVCalibration and SLCVTracker
for a good top down information.
*/

#include <SLCV.h>
#include <SLCVTracker.h>
#include <SLNode.h>
#include <opencv2/aruco.hpp>

//-----------------------------------------------------------------------------
//! ArUco Paramters loaded from configuration file.
class SLCVArucoParams
{
public:
    SLCVArucoParams() :
        edgeLength(0.06f),
        arucoDictionaryId(0),
        filename("aruco_detector_params.yml")
        {
            arucoParams = cv::aruco::DetectorParameters::create();
        }

    bool loadFromFile()
    {
        string path = SLCVCalibration::calibIniPath + filename;
        cv::FileStorage fs(path, cv::FileStorage::READ);
        if(!fs.isOpened())
        {
            cout << "Could not find parameter file for ArUco tracking!" << endl;
            cout << "Tried " << SLCVCalibration::calibIniPath + filename << endl;
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
        fs["doCornerRefinement"] >> arucoParams->doCornerRefinement;
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

    cv::Ptr<cv::aruco::DetectorParameters>  arucoParams;    //!< detector parameter structure for aruco detection function
    cv::Ptr<cv::aruco::Dictionary>          dictionary;     //!< predefined dictionary

    SLfloat     edgeLength;             //!< marker edge length
    SLint       arucoDictionaryId;      //!< id of aruco dictionary
    SLstring    arucoDetectorParams;    //!< todo: put in one file
    SLstring    filename;               //!< parameter filename
};

//-----------------------------------------------------------------------------
//! OpenCV ArUco marker tracker class derived from SLCVTracker
/*! Tracking class for ArUco markers tracking. See the official OpenCV docs on
ArUco markers: http://docs.opencv.org/3.1.0/d5/dae/tutorial_aruco_detection.html
The aruco marker used in the SLProject are printed in a PDF stored in the
_data/Calibration folder. They use the dictionary 0 and where generated with the
functions SLCVTrackerAruco::drawArucoMarkerBoard and 
SLCVTrackerAruco::drawArucoMarker.
*/
class SLCVTrackerAruco : public SLCVTracker
{
    public:
                SLCVTrackerAruco    (SLNode* node, SLint arucoID);
               ~SLCVTrackerAruco    () {;}

        SLbool  track               (SLCVMat imageGray,
                                     SLCVMat imageRgb,
                                     SLCVCalibration* calib,
                                     SLSceneView* sv);

        //! Helper function to draw and save an aruco marker board image
        static void drawArucoMarkerBoard(SLint numMarkersX,
                                         SLint numMarkersY,
                                         SLint markerEdgeLengthPix,
                                         SLint markerSepaPix,
                                         SLint dictionaryId,
                                         SLstring imgName,
                                         SLbool showImage = false,
                                         SLint borderBits = 1,
                                         SLint marginsSize = 0);
 
        //! Helper function to draw and save an aruco marker set
        static void drawArucoMarker(SLint dictionaryId,
                                    SLint minMarkerId,
                                    SLint maxMarkerId,
                                    SLint markerSizePX=200);
                                    
        static SLbool           trackAllOnce;   //!< Flag for tracking all markers once per frame
        static SLCVArucoParams  params;         //!< Parameter class instance

    private:
        static SLbool           paramsLoaded;   //!< Flag for loaded parameters
        static SLVint           arucoIDs;       //!< detected Aruco marker IDs
        static SLVMat4f         objectViewMats; //!< object view matrices

               SLint            _arucoID;       //!< Aruco Marker ID for this node
};
//-----------------------------------------------------------------------------
#endif // SLCVTrackerAruco_H
