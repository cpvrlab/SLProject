//#############################################################################
//  File:      ARTracker.cpp
//  Author:    Michael Göttlicher
//  Date:      Spring 2016
//  Codestyle: https://github.com/cpvrlab/SLProject/wiki/Coding-Style-Guidelines
//  Copyright: Marcus Hudritsch, Michael Göttlicher
//             This software is provide under the GNU General Public License
//             Please visit: http://opensource.org/licenses/GPL-3.0
//#############################################################################

#ifndef ARTracker_H
#define ARTracker_H

#include <stdafx.h>
#include <SLNode.h>
#include <SLCVCalibration.h>
#include <opencv/cv.h>
#include <opencv2/aruco.hpp>

class ARSceneView;
//-----------------------------------------------------------------------------
//! ARTracker is the pure virtual base class for tracking features in video.
/*!   
A instance of this class is hold by the SLScene instance.
*/
class ARTracker
{
    public:
                     ARTracker      (): _node(nullptr), _isVisible(false){;}
                    ~ARTracker      (){;}

        //new functions
        virtual bool init           () = 0;
        virtual bool track          (cv::Mat image, 
                                     SLCVCalibration& calib) = 0;
        virtual void updateSceneView(ARSceneView* sv) = 0;
        virtual void unloadSGObjects() = 0;

         SLMat4f     createGLMatrix  (const SLCVMat& tVec,
                                      const SLCVMat& rVec);
         SLMat4f     calcObjectMatrix(const SLMat4f& cameraObjectMat,
                                      const SLMat4f& objectViewMat);
    
    protected:
        SLNode*     _node;          //<! Connected node
        bool        _isVisible;     //<! Flag if marker is visible
        SLMat4f     _viewMat;       //!< view transformation matrix
};
//-----------------------------------------------------------------------------
#endif
