//#############################################################################
//  File:      SLCVTracked.h
//  Author:    Michael Goettlicher, Marcus Hudritsch
//  Date:      Winter 2016
//  Codestyle: https://github.com/cpvrlab/SLProject/wiki/Coding-Style-Guidelines
//  Copyright: Marcus Hudritsch, Michael Goettlicher
//             This software is provide under the GNU General Public License
//             Please visit: http://opensource.org/licenses/GPL-3.0
//#############################################################################

#ifndef SLCVTRACKER_H
#define SLCVTRACKER_H

/*
The OpenCV library version 3.4 or above with extra module must be present.
If the application captures the live video stream with OpenCV you have
to define in addition the constant SL_USES_CVCAPTURE.
All classes that use OpenCV begin with SLCV.
See also the class docs for SLCVCapture, SLCVCalibration and SLCVTracked
for a good top down information.
*/

#include <SLNode.h>
#include <SLSceneView.h>
#include <SLCV.h>
#include <SLCVCalibration.h>
#include <opencv2/aruco.hpp>
#include <opencv2/xfeatures2d.hpp>

//-----------------------------------------------------------------------------
//! SLCVTracked is the pure virtual base class for tracking features in video.
/*! The SLScene instance holds a vector of SLCVTrackeds that are tracked in 
scenes that require a live video image from the device camera. A tracker is
bound to a scene node. If the node is the camera node the tracker calculates
the relative position of the camera to the tracker. This is the standard 
aumented reality case. If the camera is a normal scene node, the tracker 
calculates the object matrix relative to the scene camera.
See also the derived classes SLCVTrackedAruco and SLCVTrackedChessboard for
example implementations.
*/
class SLCVTracked
{
    public:
                     SLCVTracked        (SLNode* node = nullptr):
                                         _node(node), _isVisible(false){;}
        virtual     ~SLCVTracked        (){;}

        virtual SLbool track            (SLCVMat imageGray,
                                         SLCVMat imageRgb,
                                         SLCVCalibration* calib,
                                         SLbool drawDetection,
                                         SLSceneView* sv) = 0;

        SLMat4f     createGLMatrix      (const SLCVMat& tVec,
                                         const SLCVMat& rVec);
        void        createRvecTvec      (const SLMat4f glMat,
                                         SLCVMat& tVec,
                                         SLCVMat& rVec);
        SLMat4f     calcObjectMatrix    (const SLMat4f& cameraObjectMat,
                                         const SLMat4f& objectViewMat);

    protected:
        SLNode*     _node;              //!< Connected node
        SLbool      _isVisible;         //!< Flag if marker is visible
        SLMat4f     _objectViewMat;     //!< view transformation matrix
};
//-----------------------------------------------------------------------------
#endif
