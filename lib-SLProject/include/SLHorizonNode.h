//#############################################################################
//  File:      SLHorizonNode.h
//  Author:    Michael Göttlicher
//  Date:      November 2020
//  Codestyle: https://github.com/cpvrlab/SLProject/wiki/SLProject-Coding-Style
//  Copyright: Marcus Hudritsch
//             This software is provide under the GNU General Public License
//             Please visit: http://opensource.org/licenses/GPL-3.0
//#############################################################################

#ifndef SL_HORIZON_NODE_H
#define SL_HORIZON_NODE_H

#include <SLSceneView.h>
#include <SLTexFont.h>
#include <SLDeviceRotation.h>
#include <SLPolyline.h>

bool estimateHorizon(const SLMat3f& enuRs, const SLMat3f& sRc, SLVec3f& horizon)
{
    SLMat3f cRenu = (enuRs * sRc).transposed();
    //estimate horizon in camera frame:
    //-normal vector of camera x-y-plane in enu frame definition: this is the camera z-axis epressed in enu frame
    SLVec3f normalCamXYPlane = SLVec3f(0, 0, 1);
    //-normal vector of enu x-y-plane in camera frame: this is the enu z-axis rotated into camera coord. frame
    SLVec3f normalEnuXYPlane = cRenu * SLVec3f(0, 0, 1);
    //-Estimation of intersetion line (horizon):
    //Then the crossproduct of both vectors defines the direction of the intersection line. In our special case we know that the origin is a point that lies on both planes.
    //Then origin together with the direction vector define the horizon.
    horizon.cross(normalCamXYPlane, normalEnuXYPlane);

    //check that vectors are not parallel
    float l = horizon.length();
    if(l < 0.01f)
    {
        horizon = {1.f, 0.f, 0.f};
        return false;
    }
    else
    {
        horizon /= l;
        return true;
    }
}

class SLHorizonNode : public SLNode
{
public:
    SLHorizonNode(SLstring name, SLDeviceRotation* devRot, SLTexFont* font, SLstring shaderDir, int scrW, int scrH);
    ~SLHorizonNode();

    void doUpdate() override;

private:
    SLDeviceRotation* _devRot = nullptr;
    SLTexFont*        _font   = nullptr;
    SLstring          _shaderDir;

    SLGLProgram* _prog = nullptr;
    SLMaterial*  _matR = nullptr;
    SLPolyline*  _line = nullptr;

    SLMat3f _sRc;
};
//-----------------------------------------------------------------------------
#endif
