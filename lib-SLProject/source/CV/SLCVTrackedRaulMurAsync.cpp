//#############################################################################
//  File:      SLCVTrackedRaulMurAsync.cpp
//  Author:    Michael Goettlicher
//  Date:      Apr 2018
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
#include <SLCVCapture.h>
#include <SLCVTrackedRaulMurAsync.h>
#include <SLCVFrame.h>
#include <SLPoints.h>
#include <OrbSlam/ORBmatcher.h>
#include <OrbSlam/PnPsolver.h>
#include <OrbSlam/Optimizer.h>
#include <SLAverageTiming.h>
#include <SLCVMapNode.h>
#include <SLCVStateEstimator.h>

using namespace cv;
using namespace ORB_SLAM2;

//-----------------------------------------------------------------------------
SLCVTrackedRaulMurAsync::SLCVTrackedRaulMurAsync(SLNode *node, SLCVMapNode* mapNode)
  : SLCVTracked(node), _orbTracking(&_stateEstimator, mapNode, false)
{
    //the map is rotated w.r.t world because ORB-SLAM uses x-axis right, 
    //y-axis down and z-forward
    mapNode->rotate(180, 1, 0, 0);
    //the tracking camera has to be a child of the slam map, 
    //because we manipulate its position (object matrix) in the maps coordinate system
    mapNode->addChild(node);
}

SLbool SLCVTrackedRaulMurAsync::track(SLCVMat imageGray,
                                      SLCVMat image,
                                      SLCVCalibration* calib,
                                      SLbool drawDetection,
                                      SLSceneView* sv)
{
    if (_frameCount == 0) {
        _orbTracking.calib(calib);
    }
    _frameCount++;

    if (_orbTracking.serial())
        _orbTracking.track();
      
    SLMat4f slMat = _stateEstimator.getPose();
    _node->om(slMat);
    
    return false;
}

SLCVOrbTracking* SLCVTrackedRaulMurAsync::orbTracking()
{
    return &_orbTracking;
}

SLCVStateEstimator* SLCVTrackedRaulMurAsync::stateEstimator()
{
    return &_stateEstimator;
}
