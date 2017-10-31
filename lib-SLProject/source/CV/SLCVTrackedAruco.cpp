//#############################################################################
//  File:      SLCVTrackedAruco.cpp
//  Author:    Michael Goettlicher, Marcus Hudritsch
//  Date:      Winter 2016
//  Codestyle: https://github.com/cpvrlab/SLProject/wiki/Coding-Style-Guidelines
//  Copyright: Marcus Hudritsch, Michael Goettlicher
//             This software is provide under the GNU General Public License
//             Please visit: http://opensource.org/licenses/GPL-3.0
//#############################################################################

#include <stdafx.h>         // precompiled headers

/*
The OpenCV library version 3.1 with extra module must be present.
If the application captures the live video stream with OpenCV you have
to define in addition the constant SL_USES_CVCAPTURE.
All classes that use OpenCV begin with SLCV.
See also the class docs for SLCVCapture, SLCVCalibration and SLCVTracked
for a good top down information.
*/
#include <SLSceneView.h>
#include <SLCVTrackedAruco.h>

#include <opencv2/core/core.hpp>
#include <opencv2/core/utility.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/video/tracking.hpp>

using namespace cv;
//-----------------------------------------------------------------------------
// Initialize static variables
bool            SLCVTrackedAruco::trackAllOnce = true;
bool            SLCVTrackedAruco::paramsLoaded = false;
SLVint          SLCVTrackedAruco::arucoIDs;
SLVMat4f        SLCVTrackedAruco::objectViewMats;
SLCVArucoParams SLCVTrackedAruco::params;
//-----------------------------------------------------------------------------
SLCVTrackedAruco::SLCVTrackedAruco(SLNode* node, SLint arucoID) : 
                  SLCVTracked(node) 
{
    _arucoID = arucoID;
}
//-----------------------------------------------------------------------------
//! Tracks the all ArUco markers in the given image for the first sceneview
/* The tracking of all aruco markers is done only once even if multiple aruco 
markers are used for different SLNode.
*/
SLbool SLCVTrackedAruco::track(SLCVMat imageGray,
                               SLCVMat imageRgb,
                               SLCVCalibration* calib,
                               SLbool drawDetection,
                               SLSceneView* sv)
{
    assert(!imageGray.empty() && "ImageGray is empty");
    assert(!imageRgb.empty() && "ImageRGB is empty");
    assert(!calib->cameraMat().empty() && "Calibration is empty");
    assert(_node && "Node pointer is null");
    assert(sv && "No sceneview pointer passed");
    assert(sv->camera() && "No active camera in sceneview");
   
    // Load aruco parameter once
    if (!paramsLoaded)
    {   paramsLoaded = params.loadFromFile();
        if (!paramsLoaded)
            SL_EXIT_MSG("SLCVTrackedAruco::track: Failed to load Aruco parameters.");
    }
    if(params.arucoParams.empty() || params.dictionary.empty())
    {   SL_WARN_MSG("SLCVTrackedAruco::track: Aruco paramters are empty.");
        return false;
    }

    // Track all Aruco markers only once per frame
    if (trackAllOnce)
    {
        ////////////
        // Detect //
        ////////////

        SLScene* s = SLScene::current;
        SLfloat startMS = s->timeMilliSec();

        arucoIDs.clear();
        objectViewMats.clear();
        SLCVVVPoint2f corners, rejected;

        aruco::detectMarkers(imageGray,
                             params.dictionary, 
                             corners, 
                             arucoIDs, 
                             params.arucoParams, 
                             rejected);

        s->detectTimesMS().set(s->timeMilliSec()-startMS);

        if(arucoIDs.size() > 0)
        {
            if (drawDetection)
            {
                aruco::drawDetectedMarkers(imageRgb, corners, arucoIDs, Scalar(0,0,255));
            }

            /////////////////////
            // Pose Estimation //
            /////////////////////

            startMS = s->timeMilliSec();

            //cout << "Aruco IdS: " << arucoIDs.size() << " : ";

            //find the camera extrinsic parameters (rVec & tVec)
            SLCVVPoint3d rVecs, tVecs;
            aruco::estimatePoseSingleMarkers(corners, 
                                             params.edgeLength, 
                                             calib->cameraMat(),
                                             calib->distortion(),
                                             rVecs,
                                             tVecs);

            s->poseTimesMS().set(s->timeMilliSec() - startMS);

            // Get the object view matrix for all aruco markers
            for(size_t i=0; i < arucoIDs.size(); ++i)
            {   //cout << arucoIDs[i] << ",";
                SLMat4f ovm = createGLMatrix(cv::Mat(tVecs[i]), cv::Mat(rVecs[i]));
                objectViewMats.push_back(ovm);
            }
            //cout << endl;
        }
        trackAllOnce = false;
    }

    if(arucoIDs.size() > 0)
    {   
        // Find the marker with the matching id
        for(size_t i=0; i < arucoIDs.size(); ++i)
        {   if (arucoIDs[i] == _arucoID)
            {
                // set the object matrix depending if the
                // tracked node is attached to a camera or not
                if (typeid(*_node)==typeid(SLCamera))
                    _node->om(objectViewMats[i].inverted());
                else
                {   _node->om(calcObjectMatrix(sv->camera()->om(),
                                               objectViewMats[i]));
                    _node->setDrawBitsRec(SL_DB_HIDDEN, false);
                }
            }
        }
        return true;
    } else
    {
        // Hide tracked node if not visible
        //if (_node != sv->camera())
            //_node->setDrawBitsRec(SL_DB_HIDDEN, true);
    }

    return false;
}
//-----------------------------------------------------------------------------
/*! SLCVTrackedAruco::drawArucoMarkerBoard draws and saves an aruco board
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
void SLCVTrackedAruco::drawArucoMarkerBoard(SLint dictionaryId,
                                            SLint numMarkersX,
                                            SLint numMarkersY, 
                                            SLfloat markerEdgeM,
                                            SLfloat markerSepaM,
                                            SLstring imgName,
                                            SLfloat dpi,
                                            SLbool showImage)
{
    Ptr<aruco::Dictionary> dictionary =
        aruco::getPredefinedDictionary(aruco::PREDEFINED_DICTIONARY_NAME(dictionaryId));

    Ptr<aruco::GridBoard> board = aruco::GridBoard::create(numMarkersX, 
                                                           numMarkersY, 
                                                           markerEdgeM,
                                                           markerSepaM,
                                                           dictionary);
    SLCVSize imageSize;
    imageSize.width  = (SLint)((markerEdgeM + markerSepaM) * 100.0f / 2.54f * dpi * numMarkersX);
    imageSize.height = (SLint)((markerEdgeM + markerSepaM) * 100.0f / 2.54f * dpi * numMarkersY);

    imageSize.width  -= (imageSize.width%4);  
    imageSize.height -= (imageSize.height%4);  

    // show created board
    SLCVMat boardImage;
    board->draw(imageSize, boardImage, 0, 1);

    if(showImage) 
    {   imshow("board", boardImage);
        waitKey(0);
    }

    imwrite(imgName, boardImage);
}
//-----------------------------------------------------------------------------
void SLCVTrackedAruco::drawArucoMarker(SLint dictionaryId,
                                       SLint minMarkerId,
                                       SLint maxMarkerId,
                                       SLint markerSizePX)
{
    assert(dictionaryId > 0);
    assert(minMarkerId > 0);
    assert(minMarkerId < maxMarkerId);

    using namespace aruco;

    Ptr<Dictionary> dict = getPredefinedDictionary(PREDEFINED_DICTIONARY_NAME(dictionaryId));
    if (maxMarkerId > dict->bytesList.rows)
        maxMarkerId = dict->bytesList.rows;

    SLCVMat markerImg;

    for (SLint i=minMarkerId; i<maxMarkerId; ++i)
    {   drawMarker(dict, i, markerSizePX, markerImg, 1);
        imwrite(SLUtils::formatString("ArucoMarker_Dict%d_%dpx_Id%d.png", 
                                      dictionaryId, markerSizePX, i), markerImg);
    }
}
//-----------------------------------------------------------------------------
