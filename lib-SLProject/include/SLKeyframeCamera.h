//#############################################################################
//  File:      SLKeyframeCamera.h
//  Author:    Michael Goettlicher
//  Date:      Dezember 2017
//  Codestyle: https://github.com/cpvrlab/SLProject/wiki/SLProject-Coding-Style
//  Copyright: Marcus Hudritsch
//             This software is provide under the GNU General Public License
//             Please visit: http://opensource.org/licenses/GPL-3.0
//#############################################################################

#ifndef SLKEYFRAMECAMERA_H
#define SLKEYFRAMECAMERA_H

#include <SLCamera.h>
#include <SLKeyframeCamera.h>

//-----------------------------------------------------------------------------
/*! Special camera for ORB-SLAM keyframes that allows the video image display
on the near clippling plane.
*/
class SLKeyframeCamera : public SLCamera
{
public:
    explicit SLKeyframeCamera(SLstring name = "Camera");
    virtual void drawMeshes(SLSceneView* sv);

    // Getters
    bool renderBackground() { return _renderBackground; }
    bool allowAsActiveCam() { return _allowAsActiveCam; }

private:
    bool _allowAsActiveCam = false;
    bool _renderBackground = false;
};
//-----------------------------------------------------------------------------
#endif //SLKEYFRAMECAMERA_H
