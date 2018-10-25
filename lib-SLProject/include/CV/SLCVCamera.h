//#############################################################################
//  File:      SLCamera.cpp
//  Author:    Michael Goettlicher
//  Date:      Dezember 2017
//  Codestyle: https://github.com/cpvrlab/SLProject/wiki/Coding-Style-Guidelines
//  Copyright: Marcus Hudritsch
//             This software is provide under the GNU General Public License
//             Please visit: http://opensource.org/licenses/GPL-3.0
//#############################################################################

#ifndef SLCVCAMERA_H
#define SLCVCAMERA_H

#include <SLCamera.h>

class SLCVKeyFrame;
class SLCVMapNode;

class SLCVCamera : public SLCamera
{
public:
    SLCVCamera(SLCVMapNode* mapNode, SLstring name = "Camera");
    virtual void drawMeshes(SLSceneView* sv);

    bool renderBackground();
    bool allowAsActiveCam();

private:
    //bool _renderBackground=false;
    //pointer to corresponding keyframe
    //SLCVKeyFrame* _kf = NULL;
    SLCVMapNode* _mapNode = NULL;
};

#endif //SLCVCAMERA_H
