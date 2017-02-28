//#############################################################################
//  File:      ARTracker.cpp
//  Author:    Michael Goettlicher
//  Date:      Spring 2016
//  Codestyle: https://github.com/cpvrlab/SLProject/wiki/Coding-Style-Guidelines
//  Copyright: Marcus Hudritsch, Michael Goettlicher
//             This software is provide under the GNU General Public License
//             Please visit: http://opensource.org/licenses/GPL-3.0
//#############################################################################

#ifndef ARARUCOTRACKER_H
#define ARARUCOTRACKER_H

#include <ARTracker.h>
#include <SLNode.h>
#include <opencv2/aruco.hpp>

using namespace std;


//-----------------------------------------------------------------------------
/*!
Parameter class for aruco tracking
*/
class ARArucoParams
{
public:
    ARArucoParams() :
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

    float   edgeLength;             //!< marker edge length
    int     arucoDictionaryId;      //!< id of aruco dictionary
    string  arucoDetectorParams;    //!< todo: put in one file
    string  filename;               //!< parameter filename
};

//-----------------------------------------------------------------------------

/*!
Tracking class for ArUco markers tracking
*/
class ARArucoTracker : public ARTracker
{
    public:
                ARArucoTracker  (){;}

        bool    init            () override;
        bool    track           (cv::Mat image, 
                                 SLCVCalibration* calib) override;
        void    updateSceneView (ARSceneView* sv) override;
        void    unloadSGObjects () override;

        std::map<int,SLMat4f>& getArucoVMs () {return _arucoOVMs;}

        //! Helper function to draw and save an aruco marker board image
        static void drawArucoMarkerBoard(int numMarkersX,
                                    int numMarkersY,
                                    int markerEdgeLengthPix,
                                    int markerSepaPix,
                                    int dictionaryId,
                                    string imgName,
                                    bool showImage = false,
                                    int borderBits = 1,
                                    int marginsSize = 0);

        //! Helper function to draw and save an aruco marker set
        static void drawArucoMarker(int dictionaryId,
                                    int minMarkerId,
                                    int maxMarkerId,
                                    int markerSizePX=200);

    private:
        map<int,SLMat4f>    _arucoOVMs;     //!< Transformations of aruco markers with respect to camera
        map<int,SLNode*>    _arucoNodes;    //!< Map from aruco marker id to SLNode
        ARArucoParams       _params;        //!< Parameter class instance
};

#endif // ARARUCOTRACKER_H
