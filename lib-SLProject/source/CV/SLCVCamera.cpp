//#############################################################################
//  File:      SLCamera.cpp
//  Author:    Michael Göttlicher
//  Date:      Dezember 2017
//  Codestyle: https://github.com/cpvrlab/SLProject/wiki/Coding-Style-Guidelines
//  Copyright: Marcus Hudritsch
//             This software is provide under the GNU General Public License
//             Please visit: http://opensource.org/licenses/GPL-3.0
//#############################################################################

#include <stdafx.h>
#include <SLCVCamera.h>
#include <SLSceneView.h>

SLCVCamera::SLCVCamera(SLstring name)
    : SLCamera(name)
{

}
//-----------------------------------------------------------------------------
//! SLCamera::drawMeshes draws the cameras frustum lines
/*!
Only draws the frustum lines without lighting when the camera is not the
active one. This means that it can be seen from the active view point.
*/
void SLCVCamera::drawMeshes(SLSceneView* sv)
{
    if (sv->camera() != this)
    {
        // Return if hidden
        if (sv->drawBit(SL_DB_HIDDEN) || this->drawBit(SL_DB_HIDDEN))
            return;

        // Vertices of the far plane
        SLVec3f farRT, farRB, farLT, farLB;

        if (_projection == P_monoOrthographic)
        {
            const SLMat4f& vm = updateAndGetWMI();
            SLVVec3f P;
            SLVec3f pos(vm.translation());
            SLfloat t = tan(SL_DEG2RAD*_fov*0.5f) * pos.length();
            SLfloat b = -t;
            SLfloat l = -sv->scrWdivH() * t;
            SLfloat r = -l;

            // small line in view direction
            P.push_back(SLVec3f(0, 0, 0)); P.push_back(SLVec3f(0, 0, _clipNear));

            // frustum pyramid lines
            farRT.set(r, t, -_clipFar); farRB.set(r, b, -_clipFar);
            farLT.set(l, t, -_clipFar); farLB.set(l, b, -_clipFar);
            P.push_back(SLVec3f(r, t, _clipNear)); P.push_back(farRT);
            P.push_back(SLVec3f(l, t, _clipNear)); P.push_back(farLT);
            P.push_back(SLVec3f(l, b, _clipNear)); P.push_back(farLB);
            P.push_back(SLVec3f(r, b, _clipNear)); P.push_back(farRB);

            //// around far clipping plane
            //P.push_back(farRT); P.push_back(farRB);
            //P.push_back(farRB); P.push_back(farLB);
            //P.push_back(farLB); P.push_back(farLT);
            //P.push_back(farLT); P.push_back(farRT);

            //// around projection plane at focal distance
            //P.push_back(SLVec3f(r, t, -_focalDist)); P.push_back(SLVec3f(r, b, -_focalDist));
            //P.push_back(SLVec3f(r, b, -_focalDist)); P.push_back(SLVec3f(l, b, -_focalDist));
            //P.push_back(SLVec3f(l, b, -_focalDist)); P.push_back(SLVec3f(l, t, -_focalDist));
            //P.push_back(SLVec3f(l, t, -_focalDist)); P.push_back(SLVec3f(r, t, -_focalDist));

            // around near clipping plane
            P.push_back(SLVec3f(r, t, _clipNear)); P.push_back(SLVec3f(r, b, _clipNear));
            P.push_back(SLVec3f(r, b, _clipNear)); P.push_back(SLVec3f(l, b, _clipNear));
            P.push_back(SLVec3f(l, b, _clipNear)); P.push_back(SLVec3f(l, t, _clipNear));
            P.push_back(SLVec3f(l, t, _clipNear)); P.push_back(SLVec3f(r, t, _clipNear));

            _vao.generateVertexPos(&P);
        }
        else
        {
            SLVVec3f P;
            SLfloat aspect = sv->scrWdivH();
            SLfloat tanFov = tan(_fov*SL_DEG2RAD*0.5f);
            SLfloat tF = tanFov * _clipFar;    //top far
            SLfloat rF = tF * aspect;          //right far
            SLfloat lF = -rF;                   //left far
            SLfloat tP = tanFov * _focalDist;  //top projection at focal distance
            SLfloat rP = tP * aspect;          //right projection at focal distance
            SLfloat lP = -tP * aspect;          //left projection at focal distance
            SLfloat tN = tanFov * _clipNear;   //top near
            SLfloat rN = tN * aspect;          //right near
            SLfloat lN = -tN * aspect;          //left near

                                                // small line in view direction
            P.push_back(SLVec3f(0, 0, 0)); P.push_back(SLVec3f(0, 0, _clipNear));

            // frustum pyramid lines
            farRT.set(rF, tF, -_clipFar); farRB.set(rF, -tF, -_clipFar);
            farLT.set(lF, tF, -_clipFar); farLB.set(lF, -tF, -_clipFar);
            P.push_back(SLVec3f(0, 0, 0)); P.push_back(farRT);
            P.push_back(SLVec3f(0, 0, 0)); P.push_back(farLT);
            P.push_back(SLVec3f(0, 0, 0)); P.push_back(farLB);
            P.push_back(SLVec3f(0, 0, 0)); P.push_back(farRB);

            //// around far clipping plane
            //P.push_back(farRT); P.push_back(farRB);
            //P.push_back(farRB); P.push_back(farLB);
            //P.push_back(farLB); P.push_back(farLT);
            //P.push_back(farLT); P.push_back(farRT);

            //// around projection plane at focal distance
            //P.push_back(SLVec3f(rP, tP, -_focalDist)); P.push_back(SLVec3f(rP, -tP, -_focalDist));
            //P.push_back(SLVec3f(rP, -tP, -_focalDist)); P.push_back(SLVec3f(lP, -tP, -_focalDist));
            //P.push_back(SLVec3f(lP, -tP, -_focalDist)); P.push_back(SLVec3f(lP, tP, -_focalDist));
            //P.push_back(SLVec3f(lP, tP, -_focalDist)); P.push_back(SLVec3f(rP, tP, -_focalDist));

            // around near clipping plane
            P.push_back(SLVec3f(rN, tN, -_clipNear)); P.push_back(SLVec3f(rN, -tN, -_clipNear));
            P.push_back(SLVec3f(rN, -tN, -_clipNear)); P.push_back(SLVec3f(lN, -tN, -_clipNear));
            P.push_back(SLVec3f(lN, -tN, -_clipNear)); P.push_back(SLVec3f(lN, tN, -_clipNear));
            P.push_back(SLVec3f(lN, tN, -_clipNear)); P.push_back(SLVec3f(rN, tN, -_clipNear));

            _vao.generateVertexPos(&P);
        }

        _vao.drawArrayAsColored(PT_lines, SLCol4f::WHITE*0.7f);
        //_background.renderInScene(farLT, farLB, farRT, farRB);
    }
}