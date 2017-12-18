//#############################################################################
//  File:      SLCamera.cpp
//  Author:    Michael Göttlicher
//  Date:      Dezember 2017
//  Codestyle: https://github.com/cpvrlab/SLProject/wiki/Coding-Style-Guidelines
//  Copyright: Marcus Hudritsch
//             This software is provide under the GNU General Public License
//             Please visit: http://opensource.org/licenses/GPL-3.0
//#############################################################################

#ifndef SLCVCAMERA_H
#define SLCVCAMERA_H

#include <SLCamera.h>

class SLCVCamera : public SLCamera
{
public:
    SLCVCamera(SLstring name = "Camera");
    virtual void drawMeshes(SLSceneView* sv);

private:
};

#endif //SLCVCAMERA_H
