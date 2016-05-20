//#############################################################################
//  File:      ARTracker.cpp
//  Author:    Michael Göttlicher
//  Date:      Spring 2016
//  Codestyle: https://github.com/cpvrlab/SLProject/wiki/Coding-Style-Guidelines
//  Copyright: Marcus Hudritsch
//             This software is provide under the GNU General Public License
//             Please visit: http://opensource.org/licenses/GPL-3.0
//#############################################################################

#ifndef ARARUCOTRACKER_H
#define ARARUCOTRACKER_H

#include <ARTracker.h>
#include <SLNode.h>
#include <opencv2/aruco.hpp>

class ARArucoParams
{
public:
    ARArucoParams() :
        arucoDetectorParams("aruco_detector_params.yml"),
        edgeLength(0.06f),
        arucoDictionaryId(0),
        filename("todo")
    {
        arucoParams = cv::aruco::DetectorParameters::create();
        dictionary =  cv::aruco::getPredefinedDictionary(cv::aruco::PREDEFINED_DICTIONARY_NAME(arucoDictionaryId));

        // do corner refinement in markers
        //todo: put in param file
        arucoParams->doCornerRefinement = true;
        arucoParams->adaptiveThreshWinSizeMin = 4;
        arucoParams->adaptiveThreshWinSizeMax = 7;
        arucoParams->adaptiveThreshWinSizeStep = 1;
    }

    //-----------------------------------------------------------------------------
    bool readDetectorParameters(string filename, cv::Ptr<cv::aruco::DetectorParameters> &params) {
        cv::FileStorage fs(filename, cv::FileStorage::READ);
        if(!fs.isOpened())
            return false;
        fs["adaptiveThreshWinSizeMin"] >> params->adaptiveThreshWinSizeMin;
        fs["adaptiveThreshWinSizeMax"] >> params->adaptiveThreshWinSizeMax;
        fs["adaptiveThreshWinSizeStep"] >> params->adaptiveThreshWinSizeStep;
        fs["adaptiveThreshConstant"] >> params->adaptiveThreshConstant;
        fs["minMarkerPerimeterRate"] >> params->minMarkerPerimeterRate;
        fs["maxMarkerPerimeterRate"] >> params->maxMarkerPerimeterRate;
        fs["polygonalApproxAccuracyRate"] >> params->polygonalApproxAccuracyRate;
        fs["minCornerDistanceRate"] >> params->minCornerDistanceRate;
        fs["minDistanceToBorder"] >> params->minDistanceToBorder;
        fs["minMarkerDistanceRate"] >> params->minMarkerDistanceRate;
        fs["doCornerRefinement"] >> params->doCornerRefinement;
        fs["cornerRefinementWinSize"] >> params->cornerRefinementWinSize;
        fs["cornerRefinementMaxIterations"] >> params->cornerRefinementMaxIterations;
        fs["cornerRefinementMinAccuracy"] >> params->cornerRefinementMinAccuracy;
        fs["markerBorderBits"] >> params->markerBorderBits;
        fs["perspectiveRemovePixelPerCell"] >> params->perspectiveRemovePixelPerCell;
        fs["perspectiveRemoveIgnoredMarginPerCell"] >> params->perspectiveRemoveIgnoredMarginPerCell;
        fs["maxErroneousBitsInBorderRate"] >> params->maxErroneousBitsInBorderRate;
        fs["minOtsuStdDev"] >> params->minOtsuStdDev;
        fs["errorCorrectionRate"] >> params->errorCorrectionRate;
        return true;
    }

    bool loadFromFile(string fileName)
    {
        bool readOk = readDetectorParameters(fileName + arucoDetectorParams, arucoParams);
        if(!readOk) {
            cerr << "Invalid detector parameters file" << endl;
            return false;
        }

        // do corner refinement in markers
        //todo: put in param file
        arucoParams->doCornerRefinement = true;
        arucoParams->adaptiveThreshWinSizeMin = 4;
        arucoParams->adaptiveThreshWinSizeMax = 7;
        arucoParams->adaptiveThreshWinSizeStep = 1;

        //if not loading
        {
            cout << "Could not find parameter file for ArUco tracking!" << endl;
            cout << "Tried ..." << endl;
        }
    }

    //detector parameter structure for aruco detection function
    cv::Ptr<cv::aruco::DetectorParameters> arucoParams;
    //todo: put in one file
    string arucoDetectorParams;
    //marker edge length
    float edgeLength;
    //id of aruco dictionary
    int arucoDictionaryId;
    //parameter filename
    string filename;
    //predefined dictionary
    cv::Ptr<cv::aruco::Dictionary> dictionary;
};

class ARArucoTracker : public ARTracker
{
public:
    ARArucoTracker(cv::Mat intrinsics, cv::Mat distoriton);
    bool init(string paramsFileDir) override;
    bool track() override;
    void updateSceneView( ARSceneView* sv ) override;
    void unloadSGObjects() override;

    std::map<int,SLMat4f>& getArucoVMs () { return _arucoVMs; }

private:
    bool            trackArucoMarkers   ();
    void            drawArucoMarkerBoard(int numMarkersX, int numMarkersY, int markerEdgeLengthPix,
                    int markerSepaPix,  int dictionaryId, string imgName, bool showImage = false,
                    int borderBits = 1, int marginsSize = 0 );


    //Transformations of aruco markers with respect to camera
    std::map<int,SLMat4f> _arucoVMs;
    std::map<int,SLNode*> _arucoNodes;

    //Parameter class instance
    ARArucoParams _p;
};

#endif // ARARUCOTRACKER_H
