//#############################################################################
//  File:      SLCVTrackerAruco.cpp
//  Author:    Michael Göttlicher & Marcus Hudritsch
//  Date:      Spring 2016
//  Codestyle: https://github.com/cpvrlab/SLProject/wiki/Coding-Style-Guidelines
//  Copyright: Marcus Hudritsch, Michael Göttlicher
//             This software is provide under the GNU General Public License
//             Please visit: http://opensource.org/licenses/GPL-3.0
//#############################################################################

#include <stdafx.h>         // precompiled headers
#include <SLSceneView.h>
#include <SLCVTrackerAruco.h>

#include <opencv2/core/core.hpp>
#include <opencv2/core/utility.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/video/tracking.hpp>

using namespace cv;
//-----------------------------------------------------------------------------
// Initialize static variables
bool            SLCVTrackerAruco::trackAllOnce = true;
bool            SLCVTrackerAruco::paramsLoaded = false;
SLVint          SLCVTrackerAruco::arucoIDs;
SLVMat4f        SLCVTrackerAruco::objectViewMats;
SLCVArucoParams SLCVTrackerAruco::params;
//-----------------------------------------------------------------------------
SLCVTrackerAruco::SLCVTrackerAruco(SLNode* node, SLint arucoID) : 
                  SLCVTracker(node) 
{
    _arucoID = arucoID;
}
//-----------------------------------------------------------------------------
//! Tracks the all ArUco markers in the given image for the first sceneview
bool SLCVTrackerAruco::track(SLCVMat image, 
                             SLCVCalibration& calib,
                             SLSceneView* sv)
{
    assert(!image.empty() && "Image is empty");
    assert(!calib.intrinsics().empty() && "Calibration is empty");
    assert(_node && "Node pointer is null");
    assert(sv && "No sceneview pointer passed");
    assert(sv->camera() && "No active camera in sceneview");
   
    // Load aruco parameter once
    if (!paramsLoaded)
    {   paramsLoaded = params.loadFromFile();
        if (!paramsLoaded)
            SL_EXIT_MSG("SLCVTrackerAruco::track: Failed to load Aruco parameters.");
    }
    if(params.arucoParams.empty() || params.dictionary.empty())
    {   SL_WARN_MSG("SLCVTrackerAruco::track: Aruco paramters are empty.");
        return false;
    }

    // Track all Aruco markers only once per frame
    if (trackAllOnce)
    {   arucoIDs.clear();
        objectViewMats.clear();
        SLCVVVPoint2f corners, rejected;

        aruco::detectMarkers(image, 
                             params.dictionary, 
                             corners, 
                             arucoIDs, 
                             params.arucoParams, 
                             rejected);

        if(arucoIDs.size() > 0)
        {   cout << "Aruco IdS: " << arucoIDs.size() << " : ";
            SLCVVPoint3d rVecs, tVecs;

            aruco::estimatePoseSingleMarkers(corners, 
                                             params.edgeLength, 
                                             calib.intrinsics(), 
                                             calib.distortion(), 
                                             rVecs, tVecs);

            for(size_t i=0; i < arucoIDs.size(); ++i)
            {   cout << arucoIDs[i] << ",";
                SLMat4f ovm = createGLMatrix(cv::Mat(tVecs[i]), cv::Mat(rVecs[i]));
                objectViewMats.push_back(ovm);
            }
            cout << endl;
        }
        trackAllOnce = false;
    }

    if(arucoIDs.size() > 0)
    {   
        // Find the marker with the matching id
        for(size_t i=0; i < arucoIDs.size(); ++i)
        {   if (arucoIDs[i] == _arucoID)
            {   if (_node == sv->camera())
                    _node->om(objectViewMats[i].inverse());
                else
                {   _node->om(calcObjectMatrix(sv->camera()->om(),_viewMat));
                    _node->setDrawBitsRec(SL_DB_HIDDEN, false);
                }
            }
        }
        return true;
    } else
    {
        // Hide tracked node if not visible
        if (_node != sv->camera())
            _node->setDrawBitsRec(SL_DB_HIDDEN, true);
    }

    return false;
}
//-----------------------------------------------------------------------------
void SLCVTrackerAruco::drawArucoMarkerBoard(int dictionaryId,
                                            int numMarkersX,
                                            int numMarkersY, 
                                            int markerEdgePX, 
                                            int markerSepaPX,
                                            string imgName, 
                                            bool showImage, 
                                            int borderBits, 
                                            int marginsSize)
{
    if(marginsSize == 0)
        marginsSize = markerSepaPX;

    SLCVSize imageSize;
    imageSize.width  = numMarkersX * (markerEdgePX + markerSepaPX) - markerSepaPX + 2 * marginsSize;
    imageSize.height = numMarkersY * (markerEdgePX + markerSepaPX) - markerSepaPX + 2 * marginsSize;

    Ptr<aruco::Dictionary> dictionary =
        aruco::getPredefinedDictionary(aruco::PREDEFINED_DICTIONARY_NAME(dictionaryId));

    Ptr<aruco::GridBoard> board = aruco::GridBoard::create(numMarkersX, 
                                                           numMarkersY, 
                                                           float(markerEdgePX),
                                                           float(markerSepaPX), 
                                                           dictionary);

    // show created board
    Mat boardImage;
    board->draw(imageSize, boardImage, marginsSize, borderBits);

    if(showImage) 
    {   imshow("board", boardImage);
        waitKey(0);
    }

    imwrite(imgName, boardImage);
}
//-----------------------------------------------------------------------------
void SLCVTrackerAruco::drawArucoMarker(int dictionaryId,
                                       int minMarkerId,
                                       int maxMarkerId,
                                       int markerSizePX)
{
    assert(dictionaryId > 0);
    assert(minMarkerId > 0);
    assert(minMarkerId < maxMarkerId);

    using namespace aruco;

    Ptr<Dictionary> dict = getPredefinedDictionary(PREDEFINED_DICTIONARY_NAME(dictionaryId));
    if (maxMarkerId > dict->bytesList.rows)
        maxMarkerId = dict->bytesList.rows;

    Mat markerImg;

    for (int i=minMarkerId; i<maxMarkerId; ++i)
    {   drawMarker(dict, i, markerSizePX, markerImg, 1);
        char name[255];
        sprintf(name, 
                "ArucoMarker_Dict%d_%dpx_Id%d.png", 
                dictionaryId, 
                markerSizePX, 
                i);

        imwrite(name, markerImg);
    }
}
//-----------------------------------------------------------------------------
