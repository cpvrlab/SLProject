//#############################################################################
//  File:      SLKeyframeCamera.h
//  Date:      Dezember 2017
//  Codestyle: https://github.com/cpvrlab/SLProject/wiki/SLProject-Coding-Style
//  Authors:   Marcus Hudritsch, Michael Goettlicher
//  License:   This software is provided under the GNU General Public License
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
    virtual void drawMesh(SLSceneView* sv);
    void         setDrawColor(SLCol4f color = SLCol4f::WHITE * 0.7f);

    // Getters
    bool renderBackground() { return _renderBackground; }
    bool allowAsActiveCam() { return _allowAsActiveCam; }

private:
    bool _allowAsActiveCam = false;
    bool _renderBackground = false;

    SLCol4f _color;
};
//-----------------------------------------------------------------------------
#endif // SLKEYFRAMECAMERA_H
