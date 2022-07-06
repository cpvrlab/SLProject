//#############################################################################
//  File:      SLKeyframeCamera.cpp
//  Authors:   Michael Goettlicher, Marcus Hudritsch
//  Date:      Dezember 2017
//  Codestyle: https://github.com/cpvrlab/SLProject/wiki/SLProject-Coding-Style
//  Authors:   Marcus Hudritsch
//  License:   This software is provided under the GNU General Public License
//             Please visit: http://opensource.org/licenses/GPL-3.0
//#############################################################################

#include <SLKeyframeCamera.h>
#include <SLSceneView.h>
#include <SLGLTexture.h>

SLKeyframeCamera::SLKeyframeCamera(SLstring name)
  : SLCamera(name)
{
    setDrawColor();
}
//-----------------------------------------------------------------------------
//! SLKeyframeCamera::drawMeshes draws the cameras frustum lines
/*!
Only draws the frustum lines without lighting when the camera is not the
active one. This means that it can be seen from the active view point.
*/
void SLKeyframeCamera::drawMesh(SLSceneView* sv)
{
    if (sv->camera() != this)
    {
        // Return if hidden
        if (sv->drawBit(SL_DB_HIDDEN) || this->drawBit(SL_DB_HIDDEN))
            return;

        // Vertices of the near plane
        SLVec3f nearRT, nearRB, nearLT, nearLB;

        if (_projType == P_monoOrthographic)
        {
            const SLMat4f& vm = updateAndGetWMI();
            SLVVec3f       P;
            SLVec3f        pos(vm.translation());
            SLfloat        t = tan(Utils::DEG2RAD * _fovV * 0.5f) * pos.length();
            SLfloat        b = -t;
            SLfloat        l = -sv->scrWdivH() * t;
            SLfloat        r = -l;

            // small line in view direction
            P.push_back(SLVec3f(0, 0, 0));
            P.push_back(SLVec3f(0, 0, _clipNear));

            // frustum pyramid lines
            nearRT.set(r, t, -_clipNear);
            nearRB.set(r, b, -_clipNear);
            nearLT.set(l, t, -_clipNear);
            nearLB.set(l, b, -_clipNear);

            // around near clipping plane
            P.push_back(SLVec3f(r, t, _clipNear));
            P.push_back(SLVec3f(r, b, _clipNear));
            P.push_back(SLVec3f(r, b, _clipNear));
            P.push_back(SLVec3f(l, b, _clipNear));
            P.push_back(SLVec3f(l, b, _clipNear));
            P.push_back(SLVec3f(l, t, _clipNear));
            P.push_back(SLVec3f(l, t, _clipNear));
            P.push_back(SLVec3f(r, t, _clipNear));

            _vao.generateVertexPos(&P);
        }
        else
        {
            SLVVec3f P;
            SLfloat  aspect = sv->scrWdivH();
            SLfloat  tanFov = tan(_fovV * Utils::DEG2RAD * 0.5f);
            SLfloat  tN     = tanFov * _clipNear; // top near
            SLfloat  rN     = tN * aspect;        // right near
            SLfloat  lN     = -tN * aspect;       // left near

            // small line in view direction
            P.push_back(SLVec3f(0, 0, 0));
            P.push_back(SLVec3f(0, 0, _clipNear));

            // frustum pyramid lines
            nearRT.set(rN, tN, -_clipNear);
            nearRB.set(rN, -tN, -_clipNear);
            nearLT.set(lN, tN, -_clipNear);
            nearLB.set(lN, -tN, -_clipNear);

            P.push_back(SLVec3f(0, 0, 0));
            P.push_back(nearRT);
            P.push_back(SLVec3f(0, 0, 0));
            P.push_back(nearLT);
            P.push_back(SLVec3f(0, 0, 0));
            P.push_back(nearLB);
            P.push_back(SLVec3f(0, 0, 0));
            P.push_back(nearRB);

            // around near clipping plane
            P.push_back(SLVec3f(rN, tN, -_clipNear));
            P.push_back(SLVec3f(rN, -tN, -_clipNear));
            P.push_back(SLVec3f(rN, -tN, -_clipNear));
            P.push_back(SLVec3f(lN, -tN, -_clipNear));
            P.push_back(SLVec3f(lN, -tN, -_clipNear));
            P.push_back(SLVec3f(lN, tN, -_clipNear));
            P.push_back(SLVec3f(lN, tN, -_clipNear));
            P.push_back(SLVec3f(rN, tN, -_clipNear));

            _vao.generateVertexPos(&P);
        }

        SLCol4f color = sv->s()->singleNodeSelected() == this ? SLCol4f::YELLOW : _color;
        _vao.drawArrayAsColored(PT_lines, color);

        if (renderBackground())
            _background.renderInScene(updateAndGetWM(), nearLT, nearLB, nearRT, nearRB);
    }
}

//-----------------------------------------------------------------------------

//! SLKeyframeCamera::setDrawColor specify which color should be use.

void SLKeyframeCamera::setDrawColor(SLCol4f color)
{
    _color = color;
}
