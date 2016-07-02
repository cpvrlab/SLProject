//#############################################################################
//  File:      SLCVTracker.cpp
//  Author:    Michael Göttlicher
//  Date:      Spring 2016
//  Codestyle: https://github.com/cpvrlab/SLProject/wiki/Coding-Style-Guidelines
//  Copyright: Marcus Hudritsch, Michael Göttlicher
//             This software is provide under the GNU General Public License
//             Please visit: http://opensource.org/licenses/GPL-3.0
//#############################################################################

#ifndef SLCVTRACKER_H
#define SLCVTRACKER_H

#include <stdafx.h>
#include <SLNode.h>
#include <SLCVCalibration.h>
#include <opencv/cv.h>
#include <opencv2/aruco.hpp>

class ARSceneView;
//-----------------------------------------------------------------------------
//! SLCVTracker is the pure virtual base class for tracking features in video.
/*!   
A instance of this class is hold by the SLScene instance.
*/
class SLCVTracker
{
    public:
                     SLCVTracker      (SLNode* node = nullptr): 
                                      _node(node), _isVisible(false){;}
        virtual     ~SLCVTracker      () = 0;

        //new functions
        virtual bool init           (string paramsFileDir) = 0;
        virtual bool track          (cv::Mat image, 
                                     SLCVCalibration& calib,
                                     SLSceneView* sv) = 0;
    
    protected:
        SLNode*     _node;          //<! Connected node
        bool        _isVisible;     //<! Flag if marker is visible
        SLMat4f     _viewMat;       //!< view transformation matrix
};
//-----------------------------------------------------------------------------
#endif
