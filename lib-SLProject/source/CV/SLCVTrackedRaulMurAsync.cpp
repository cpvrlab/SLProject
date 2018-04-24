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
SLCVTrackedRaulMurAsync::SLCVTrackedRaulMurAsync(SLNode *node, ORBVocabulary* vocabulary,
    SLCVKeyFrameDB* keyFrameDB, SLCVMap* map, SLCVMapNode* mapNode)
  : SLCVTracked(node), _orbTracking(&_stateEstimator, keyFrameDB, map, mapNode, vocabulary, true)
{
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
    _orbTracking.trackOrbs();
      
  SLMat4f slMat = _stateEstimator.getPose();
  _node->om(slMat);
    
  return false;
}

SLCVOrbTracking* SLCVTrackedRaulMurAsync::orbTracking()
{
  return &_orbTracking;
}
