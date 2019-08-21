//#############################################################################
//  File:      SLKeyframeCamera.h
//  Author:    Michael Goettlicher
//  Date:      Dezember 2017
//  Codestyle: https://github.com/cpvrlab/SLProject/wiki/SLProject-Coding-Style
//  Copyright: Marcus Hudritsch
//             This software is provide under the GNU General Public License
//             Please visit: http://opensource.org/licenses/GPL-3.0
//#############################################################################

#ifndef SLCVCAMERA_H
#define SLCVCAMERA_H

#include <SLCamera.h>

//-----------------------------------------------------------------------------
/*! Special camera for ORB-SLAM keyframes that allows the video image display
on the near clippling plane.
*/
class SLKeyframeCamera : public SLCamera
{
    public:
    SLKeyframeCamera(SLstring name = "Camera");
    virtual ~SLKeyframeCamera() { ; }
    virtual void drawMeshes(SLSceneView* sv);

    // Getters
    bool renderBackground() { return _renderBackground; }
    bool allowAsActiveCam() { return _allowAsActiveCam; }

    private:
    bool _allowAsActiveCam = false;
    bool _renderBackground = false;
};
//-----------------------------------------------------------------------------
#endif //SLCVCAMERA_H
