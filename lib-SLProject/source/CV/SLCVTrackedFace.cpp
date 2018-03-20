//#############################################################################
//  File:      SLCVTrackedFace.cpp
//  Author:    Marcus Hudritsch
//  Date:      Spring 2018
//  Codestyle: https://github.com/cpvrlab/SLProject/wiki/Coding-Style-Guidelines
//  Copyright: Marcus Hudritsch
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
#include <SLApplication.h>
#include <SLSceneView.h>
#include <SLCVTrackedFace.h>

#include <opencv2/core/core.hpp>
#include <opencv2/core/utility.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/video/tracking.hpp>

using namespace cv;
//-----------------------------------------------------------------------------
SLCVTrackedFace::SLCVTrackedFace(SLNode* node) :
                 SLCVTracked(node)
{
}
//-----------------------------------------------------------------------------
//! Tracks the ...
/* The tracking ...
*/
SLbool SLCVTrackedFace::track(SLCVMat imageGray,
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
   
    /*
    // Load aruco parameter once
    if (!paramsLoaded)
    {
        paramsLoaded = params.loadFromFile();
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

        SLScene* s = SLApplication::scene;
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
    */

    return false;
}
//-----------------------------------------------------------------------------
