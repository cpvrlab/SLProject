//#############################################################################
//  File:      SLKeyframeCamera.cpp
//  Author:    Michael Goettlicher, Marcus Hudritsch
//  Date:      Dezember 2017
//  Codestyle: https://github.com/cpvrlab/SLProject/wiki/Coding-Style-Guidelines
//  Copyright: Marcus Hudritsch
//             This software is provide under the GNU General Public License
//             Please visit: http://opensource.org/licenses/GPL-3.0
//#############################################################################

#include <stdafx.h>
#include <SLKeyframeCamera.h>
#include <SLSceneView.h>
#include <SLGLTexture.h>

SLKeyframeCamera::SLKeyframeCamera(SLstring name)
  : SLCamera(name)
{
}
//-----------------------------------------------------------------------------
//! SLKeyframeCamera::drawMeshes draws the cameras frustum lines
/*!
Only draws the frustum lines without lighting when the camera is not the
active one. This means that it can be seen from the active view point.
*/
void SLKeyframeCamera::drawMeshes(SLSceneView* sv)
{
    if (sv->camera() != this)
    {
        // Return if hidden
        if (sv->drawBit(SL_DB_HIDDEN) || this->drawBit(SL_DB_HIDDEN))
            return;

        // Vertices of the near plane
        SLVec3f nearRT, nearRB, nearLT, nearLB;

        if (_projection == P_monoOrthographic)
        {
            const SLMat4f& vm = updateAndGetWMI();
            SLVVec3f       P;
            SLVec3f        pos(vm.translation());
            SLfloat        t = tan(SL_DEG2RAD * _fov * 0.5f) * pos.length();
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
            SLfloat  tanFov = tan(_fov * SL_DEG2RAD * 0.5f);
            SLfloat  tN     = tanFov * _clipNear; //top near
            SLfloat  rN     = tN * aspect;        //right near
            SLfloat  lN     = -tN * aspect;       //left near

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

        _vao.drawArrayAsColored(PT_lines, SLCol4f::WHITE * 0.7f);

        if (renderBackground())
            _background.renderInScene(nearLT, nearLB, nearRT, nearRB);

        //if (_background.texture()->images().size()) {
        //    auto& imgs = _background.texture()->images();
        //    SLCVImage* img = imgs[0];
        //    auto mat = img->cvMat();
        //    cv::imwrite("D:/Development/SLProject/_data/calibrations/imgs/kf0-test.jpg", mat);
        //}
    }
}
//-----------------------------------------------------------------------------
