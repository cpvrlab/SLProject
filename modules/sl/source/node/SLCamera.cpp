//#############################################################################
//   File:      SLCamera.cpp
//   Date:      July 2014
//   Codestyle: https://github.com/cpvrlab/SLProject/wiki/SLProject-Coding-Style
//   Authors:   Marc Wacker, Marcus Hudritsch
//   License:   This software is provided under the GNU General Public License
//              Please visit: http://opensource.org/licenses/GPL-3.0
//#############################################################################

#include <SLSceneView.h>
#include <SLDeviceLocation.h>
#include <SLDeviceRotation.h>
#include <SLGLProgramManager.h>
#include <SLFrustum.h>

//-----------------------------------------------------------------------------
// Static global default parameters for new cameras
SLCamAnim  SLCamera::currentAnimation   = CA_turntableYUp;
SLProjType SLCamera::currentProjection  = P_monoPerspective;
SLfloat    SLCamera::currentFOV         = 45.0f;
SLint      SLCamera::currentDevRotation = 0;
//-----------------------------------------------------------------------------
SLCamera::SLCamera(const SLstring& name,
                   SLStdShaderProg textureOnlyProgramId,
                   SLStdShaderProg colorAttributeProgramId)
  : SLNode(name),
    _movedLastFrame(false),
    _oldTouchPos1(0, 0),
    _oldTouchPos2(0, 0),
    _trackballSize(0.8f),
    _moveDir(0, 0, 0),
    _drag(0.05f),
    _maxSpeed(0.0f),
    _velocity(0.0f, 0.0f, 0.0f),
    _acceleration(0, 0, 0),
    _brakeAccel(16.0f),
    _moveAccel(16.0f),
    _unitScaling(1.0f),
    _fogIsOn(false),
    _fogMode(FM_linear),
    _fogDensity(0.1f),
    _fogStart(1.0f),
    _fogEnd(6.0f),
    _fogColor(SLCol4f::GRAY),
    _fogColorIsBack(true),
    _viewport(0, 0, 640, 480),
    _background(SLGLProgramManager::get(textureOnlyProgramId),
                SLGLProgramManager::get(colorAttributeProgramId)),
    _onCamUpdateCB(nullptr)
{
    _fovInit       = 0;
    _viewportRatio = 640.0f / 480.0f; // will be overwritten in setProjection
    _clipNear      = 0.1f;
    _clipFar       = 300.0f;
    _fovV          = 45.0;
    _projType      = P_monoPerspective;
    _camAnim       = CA_turntableYUp;
    _castsShadows  = false;

    // depth of field parameters
    _lensDiameter = 0.3f;
    _lensSamples.samples(1, 1); // e.g. 10,10 > 10x10=100 lens samples
    _focalDist           = 5;
    _stereoEyeSeparation = _focalDist / 30.0f;

    _background.colors(SLCol4f(0.6f, 0.6f, 0.6f), SLCol4f(0.3f, 0.3f, 0.3f));

    _enucorrRenu.identity();
}
//-----------------------------------------------------------------------------
SLCamera::~SLCamera()
{
}
//-----------------------------------------------------------------------------
/*! SLCamera::camUpdate does the smooth transition for the walk animation. It
is called in every frame. It moves the camera after the key was released and
smoothly stops the motion by decreasing the speed every frame.
*/
SLbool SLCamera::camUpdate(SLSceneView* sv, SLfloat elapsedTimeMS)
{
    // call option update callback
    if (_onCamUpdateCB)
        _onCamUpdateCB(sv);

    if (_velocity == SLVec3f::ZERO && _moveDir == SLVec3f::ZERO)
    {
        return false;
    }

    if (!_movedLastFrame)
    {
        _movedLastFrame = true;
        return true;
    }

    SLfloat dtS = elapsedTimeMS * 0.001f;

    SLbool braking = false;
    if (_moveDir != SLVec3f::ZERO)
    {
        // x and z movement direction vector should be projected on the x,z plane while
        // but still in local space
        // the y movement direction should always be in world space
        SLVec3f f = forwardOS();
        f.y       = 0;
        f.normalize();

        SLVec3f r = rightOS();
        r.y       = 0;
        r.normalize();

        _acceleration   = f * -_moveDir.z + r * _moveDir.x;
        _acceleration.y = _moveDir.y;
        _acceleration.normalize();
        _acceleration *= _moveAccel;

    } // accelerate in the opposite velocity to brake
    else
    {
        _acceleration = -_velocity.normalized() * _brakeAccel;
        braking       = true;
    }

    // accelerate
    SLVec3f increment = _acceleration * dtS; // all units in m/s, convert MS to S

    // early out if we're braking and the velocity would fall < 0
    if (braking && increment.lengthSqr() > _velocity.lengthSqr())
    {
        _velocity.set(SLVec3f::ZERO);
        _movedLastFrame = false;
        return false;
    }

    _velocity += increment - _drag * _velocity * dtS;
    SLfloat velMag = _velocity.length();

    // don't go over max speed
    SLfloat maxSpeed = _maxSpeed > 0.0f ? _maxSpeed : _clipFar / 50.0f;
    if (velMag > maxSpeed)
        _velocity = _velocity.normalized() * maxSpeed;

    // final delta movement vector
    SLVec3f delta = _velocity * dtS;

    // adjust for scaling (if the character is shrinked or enlarged)
    delta *= _unitScaling;

    translate(delta, TS_world);

    _movedLastFrame = true;
    return true;
}

//-----------------------------------------------------------------------------
//! SLCamera::drawMeshes draws the cameras frustum lines
/*!
Only draws the frustum lines without lighting when the camera is not the
active one. This means that it can be seen from the active view point.
*/
void SLCamera::drawMesh(SLSceneView* sv)
{
    if (sv->camera() != this)
    {
        // Vertices of the far plane
        SLVec3f farRT, farRB, farLT, farLB;

        if (_projType == P_monoOrthographic)
        {
            const SLMat4f& vm = updateAndGetWMI();
            SLVVec3f       P;
            SLVec3f        pos(vm.translation());
            SLfloat        t = tan(Utils::DEG2RAD * _fovV * 0.5f) * pos.length(); // top
            SLfloat        b = -t;                                                // bottom
            SLfloat        l = -sv->scrWdivH() * t;                               // left
            SLfloat        r = -l;                                                // right
            SLfloat        c = std::min(l, r) * 0.05f;                            // size of cross at focal point

            // small line in view direction
            P.push_back(SLVec3f(0, 0, 0));
            P.push_back(SLVec3f(0, 0, _clipNear * 4));

            // frustum pyramid lines
            farRT.set(r, t, -_clipFar);
            farRB.set(r, b, -_clipFar);
            farLT.set(l, t, -_clipFar);
            farLB.set(l, b, -_clipFar);
            P.push_back(SLVec3f(r, t, _clipNear));
            P.push_back(farRT);
            P.push_back(SLVec3f(l, t, _clipNear));
            P.push_back(farLT);
            P.push_back(SLVec3f(l, b, _clipNear));
            P.push_back(farLB);
            P.push_back(SLVec3f(r, b, _clipNear));
            P.push_back(farRB);

            // around far clipping plane
            P.push_back(farRT);
            P.push_back(farRB);
            P.push_back(farRB);
            P.push_back(farLB);
            P.push_back(farLB);
            P.push_back(farLT);
            P.push_back(farLT);
            P.push_back(farRT);

            // around projection plane at focal distance
            P.push_back(SLVec3f(r, t, -_focalDist));
            P.push_back(SLVec3f(r, b, -_focalDist));
            P.push_back(SLVec3f(r, b, -_focalDist));
            P.push_back(SLVec3f(l, b, -_focalDist));
            P.push_back(SLVec3f(l, b, -_focalDist));
            P.push_back(SLVec3f(l, t, -_focalDist));
            P.push_back(SLVec3f(l, t, -_focalDist));
            P.push_back(SLVec3f(r, t, -_focalDist));

            // cross at focal point in focal distance
            P.push_back(SLVec3f(-c, 0, -_focalDist));
            P.push_back(SLVec3f(c, 0, -_focalDist));
            P.push_back(SLVec3f(0, -c, -_focalDist));
            P.push_back(SLVec3f(0, c, -_focalDist));
            P.push_back(SLVec3f(0, 0, -_focalDist - c));
            P.push_back(SLVec3f(0, 0, -_focalDist + c));

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
        else if (_projType == P_monoPerspective || _projType == P_monoIntrinsic)
        {
            SLVVec3f P;
            SLfloat  aspect = sv->scrWdivH();
            SLfloat  tanFov = tan(_fovV * Utils::DEG2RAD * 0.5f);
            SLfloat  tF     = tanFov * _clipFar;        // top far
            SLfloat  rF     = tF * aspect;              // right far
            SLfloat  lF     = -rF;                      // left far
            SLfloat  tP     = tanFov * _focalDist;      // top projection at focal distance
            SLfloat  rP     = tP * aspect;              // right projection at focal distance
            SLfloat  lP     = -tP * aspect;             // left projection at focal distance
            SLfloat  cP     = std::min(lP, rP) * 0.05f; // size of cross at focal point
            SLfloat  tN     = tanFov * _clipNear;       // top near
            SLfloat  rN     = tN * aspect;              // right near
            SLfloat  lN     = -tN * aspect;             // left near

            // small line in view direction
            P.push_back(SLVec3f(0, 0, 0));
            P.push_back(SLVec3f(0, 0, _clipNear * 4));

            // frustum pyramid lines
            farRT.set(rF, tF, -_clipFar);
            farRB.set(rF, -tF, -_clipFar);
            farLT.set(lF, tF, -_clipFar);
            farLB.set(lF, -tF, -_clipFar);
            P.push_back(SLVec3f(0, 0, 0));
            P.push_back(farRT);
            P.push_back(SLVec3f(0, 0, 0));
            P.push_back(farLT);
            P.push_back(SLVec3f(0, 0, 0));
            P.push_back(farLB);
            P.push_back(SLVec3f(0, 0, 0));
            P.push_back(farRB);

            // around far clipping plane
            P.push_back(farRT);
            P.push_back(farRB);
            P.push_back(farRB);
            P.push_back(farLB);
            P.push_back(farLB);
            P.push_back(farLT);
            P.push_back(farLT);
            P.push_back(farRT);

            // around projection plane at focal distance
            P.push_back(SLVec3f(rP, tP, -_focalDist));
            P.push_back(SLVec3f(rP, -tP, -_focalDist));
            P.push_back(SLVec3f(rP, -tP, -_focalDist));
            P.push_back(SLVec3f(lP, -tP, -_focalDist));
            P.push_back(SLVec3f(lP, -tP, -_focalDist));
            P.push_back(SLVec3f(lP, tP, -_focalDist));
            P.push_back(SLVec3f(lP, tP, -_focalDist));
            P.push_back(SLVec3f(rP, tP, -_focalDist));

            // cross at focal point in focal distance
            P.push_back(SLVec3f(-cP, 0, -_focalDist));
            P.push_back(SLVec3f(cP, 0, -_focalDist));
            P.push_back(SLVec3f(0, -cP, -_focalDist));
            P.push_back(SLVec3f(0, cP, -_focalDist));
            P.push_back(SLVec3f(0, 0, -_focalDist - cP));
            P.push_back(SLVec3f(0, 0, -_focalDist + cP));

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

        SLCol4f color = sv->s()->singleNodeSelected() == this ? SLCol4f::YELLOW : SLCol4f::WHITE * 0.7f;
        _vao.drawArrayAsColored(PT_lines, color);

        if (!sv->s()->skybox())
            _background.renderInScene(updateAndGetWM(), farLT, farLB, farRT, farRB);
    }
}
//-----------------------------------------------------------------------------
//! SLCamera::statsRec updates the statistic parameters
void SLCamera::statsRec(SLNodeStats& stats)
{
    stats.numTriangles += 12;
    stats.numBytes += sizeof(SLCamera);
    SLNode::statsRec(stats);
}
//-----------------------------------------------------------------------------
/*!
SLCamera::calcMinMax calculates the axis aligned minimum and maximum point of
the camera position and the 4 near clipping plane points in object space (OS).
*/
void SLCamera::calcMinMax(SLVec3f& minV, SLVec3f& maxV) const
{
    SLVec3f P[5];
    SLfloat tanFov = tan(_fovV * Utils::DEG2RAD * 0.5f);
    SLfloat tN     = tanFov * _clipNear;  // top near
    SLfloat rN     = tN * _viewportRatio; // right near

    // The camera center
    P[0].set(0, 0, 0);

    // around near clipping plane
    P[1].set(rN, tN, -_clipNear);
    P[2].set(rN, -tN, -_clipNear);
    P[3].set(-rN, -tN, -_clipNear);
    P[4].set(-rN, tN, -_clipNear);

    // init min & max points
    minV.set(FLT_MAX, FLT_MAX, FLT_MAX);
    maxV.set(-FLT_MAX, -FLT_MAX, -FLT_MAX);

    // calc min and max point of all vertices
    for (auto& i : P)
    {
        if (i.x < minV.x) minV.x = i.x;
        if (i.x > maxV.x) maxV.x = i.x;
        if (i.y < minV.y) minV.y = i.y;
        if (i.y > maxV.y) maxV.y = i.y;
        if (i.z < minV.z) minV.z = i.z;
        if (i.z > maxV.z) maxV.z = i.z;
    }
}
//-----------------------------------------------------------------------------
/*!
 SLCamera::buildAABB builds the passed axis-aligned bounding box in OS and
 updates the min & max points in WS with the passed WM of the node. The camera
 node has no mesh associated, so we have to calculate the min and max point
 from the camera frustum.
 */
void SLCamera::buildAABB(SLAABBox& aabb, const SLMat4f& wmNode)
{
    SLVec3f minP, maxP;
    calcMinMax(minP, maxP);

    // Apply world matrix
    aabb.fromOStoWS(minP, maxP, wmNode);
}

//-----------------------------------------------------------------------------
//! Calculate and return frustum size at distance to camera center
SLVec2f SLCamera::frustumSizeAtDistance(SLfloat distance)
{
    SLVec2f frustumSize;

    frustumSize.y = 2.f * distance * std::tan(_fovV * 0.5f * DEG2RAD);
    frustumSize.x = frustumSize.y * _viewportRatio; // w / h

    return frustumSize;
}
//-----------------------------------------------------------------------------
//! Returns the projection type as string
SLstring SLCamera::projTypeToStr(SLProjType pt)
{
    switch (pt)
    {
        case P_monoPerspective: return "Perspective";
        case P_monoIntrinsic: return "Perspective Intrinsic";
        case P_monoOrthographic: return "Orthographic";
        case P_stereoSideBySide: return "Side by Side";
        case P_stereoSideBySideP: return "Side by Side proportional";
        case P_stereoSideBySideD: return "Side by Side distorted";
        case P_stereoLineByLine: return "Line by Line";
        case P_stereoColumnByColumn: return "Column by Column";
        case P_stereoPixelByPixel: return "Checkerboard";
        case P_stereoColorRC: return "Red-Cyan";
        case P_stereoColorRG: return "Red-Green";
        case P_stereoColorRB: return "Red-Blue";
        case P_stereoColorYB: return "Yellow-Blue";
        default: return "Unknown";
    }
}
//-----------------------------------------------------------------------------
/*!
Returns the height of the screen at focal distance. In stereo rendering this
should correspond to the height of the projection plane.
*/
SLfloat SLCamera::focalDistScrH() const
{
    return tan(_fovV * Utils::DEG2RAD / 2.0f) * _focalDist * 2.0f;
}
//-----------------------------------------------------------------------------
/*!
Returns the width of the screen at focal distance. In stereo rendering this
should correspond to the width of the projection plane.
*/
SLfloat SLCamera::focalDistScrW() const
{
    return focalDistScrH() * _viewportRatio;
}
//-----------------------------------------------------------------------------
//! Sets the viewport transform depending on the projection
void SLCamera::setViewport(SLSceneView* sv, const SLEyeType eye)
{
    SLGLState* stateGL = SLGLState::instance();
    SLRecti    vpRect  = sv->viewportRect();
    _viewportRatio     = (float)vpRect.width / (float)vpRect.height;

    //////////////////
    // Set viewport //
    //////////////////

    // calculate frame buffer size
    SLint fbX  = vpRect.x;
    SLint fbY  = vpRect.y;
    SLint fbW  = vpRect.width;
    SLint fbH  = vpRect.height;
    SLint fbW2 = fbW >> 1;  // fbW/2
    SLint fbH2 = fbH >> 1;  // fbH/2
    SLint fbH4 = fbH2 >> 1; // fbH2/2

    if (_projType == P_stereoSideBySideD)
    {
        SLint fbOcW2 = sv->oculusFB()->halfWidth();
        SLint fbOcH  = sv->oculusFB()->height();
        if (eye == ET_left)
            stateGL->viewport(0, 0, fbOcW2, fbOcH);
        else
            stateGL->viewport(fbOcW2, 0, fbOcW2, fbOcH);
    }
    else if (_projType == P_stereoSideBySide)
    {
        if (eye == ET_left)
            stateGL->viewport(0, 0, fbW2, fbH);
        else
            stateGL->viewport(fbW2, 0, fbW2, fbH);
    }
    else if (_projType == P_stereoSideBySideP)
    {
        if (eye == ET_left)
            stateGL->viewport(0, fbH4, fbW2, fbH2);
        else
            stateGL->viewport(fbW2, fbH4, fbW2, fbH2);
    }
    else
        stateGL->viewport(fbX, fbY, fbW, fbH);

    _viewport.set(fbX, fbY, fbW, fbH);
}
//-----------------------------------------------------------------------------
/*!
Sets the projection transformation matrix and the drawing buffer.
In case of a stereographic projection it additionally sets the
stereo splitting parameters such as the color masks and the color filter matrix
for stereo color anaglyph.
*/
void SLCamera::setProjection(SLSceneView* sv, const SLEyeType eye)
{
    ////////////////////
    // Set Projection //
    ////////////////////

    const SLMat4f& vm      = updateAndGetWMI();
    SLGLState*     stateGL = SLGLState::instance();

    _stereoEye = eye;

    SLVec3f pos(vm.translation());
    SLfloat top, bottom, left, right, d; // frustum parameters

    switch (_projType)
    {
        case P_monoPerspective:
            stateGL->projectionMatrix.perspective(_fovV, _viewportRatio, _clipNear, _clipFar);
            break;

        case P_monoIntrinsic:
            stateGL->projectionMatrix.perspectiveCenteredPP((float)_viewport.width,
                                                            (float)_viewport.height,
                                                            _fx,
                                                            _fy,
                                                            _cx,
                                                            _cy,
                                                            _clipNear,
                                                            _clipFar);
            break;

        case P_monoOrthographic:
            top    = tan(Utils::DEG2RAD * _fovV * 0.5f) * pos.length();
            bottom = -top;
            left   = -_viewportRatio * top;
            right  = -left;

            // The orthographic projection should have its near clip plane behind the camera
            // rather than slightly in front of it. Else we will see cross sections of scenes if
            // we zoom in close
            stateGL->projectionMatrix.ortho(left, right, bottom, top, -_clipNear, _clipFar);
            break;

        case P_stereoSideBySideD:
            stateGL->projectionMatrix = sv->s()->oculus()->projection(eye);

            break;
        // all other stereo projections
        default:
            // asymmetric frustum shift d (see chapter stereo projection)
            d      = (SLfloat)eye * 0.5f * _stereoEyeSeparation * _clipNear / _focalDist;
            top    = tan(Utils::DEG2RAD * _fovV / 2) * _clipNear;
            bottom = -top;
            left   = -_viewportRatio * top - d;
            right  = _viewportRatio * top - d;
            stateGL->projectionMatrix.frustum(left, right, bottom, top, _clipNear, _clipFar);
    }

    ///////////////////
    // Clear Buffers //
    ///////////////////

    if (eye == ET_right)
        // Do not clear color on right eye because it contains the color of the
        // left eye. The right eye must be drawn after the left into the same buffer
        stateGL->clearDepthBuffer();

    /////////////////////////////////
    //  Set Color Mask and Filter  //
    /////////////////////////////////

    if (_projType >= P_stereoColorRC)
    {
        if (eye == ET_left)
        {
            switch (_projType)
            {
                case P_stereoColorRC:
                case P_stereoColorRB:
                case P_stereoColorRG: stateGL->colorMask(1, 0, 0, 1); break;
                case P_stereoColorYB: stateGL->colorMask(1, 1, 0, 1); break;
                default: break;
            }
        }
        else
        {
            switch (_projType)
            {
                case P_stereoColorRC: stateGL->colorMask(0, 1, 1, 1); break;
                case P_stereoColorRB: stateGL->colorMask(0, 0, 1, 1); break;
                case P_stereoColorRG: stateGL->colorMask(0, 1, 0, 1); break;
                case P_stereoColorYB: stateGL->colorMask(0, 0, 1, 1); break;
                default: break;
            }
        }

        // Set color filter matrix for red-cyan and yello-blue (ColorCode3D)
        switch (_projType)
        {
            case P_stereoColorRC:
                _stereoColorFilter.setMatrix(0.29f,
                                             0.59f,
                                             0.12f,
                                             0.00f,
                                             1.00f,
                                             0.00f,
                                             0.00f,
                                             0.00f,
                                             1.00f);
                break;
            case P_stereoColorYB:
                _stereoColorFilter.setMatrix(1.00f,
                                             0.00f,
                                             0.00f,
                                             0.00f,
                                             1.00f,
                                             0.00f,
                                             0.15f,
                                             0.15f,
                                             0.70f);
                break;
            default: break;
        }
    }
    GET_GL_ERROR;
}
//-----------------------------------------------------------------------------
//! Calculate and apply correction from finger x-y-rotation
void SLCamera::updateEnuCorrRenu(SLSceneView*   sv,
                                 const SLMat3f& enuRc,
                                 float&         f,
                                 SLVec3f&       enuOffsetPix)
{
    /* 1. Estimate horizon in enu-frame: intersection between camera x-y-plane
    defined in enu-frame and enu x-y-plane: normal vector of camera x-y-plane
    in enu frame definition: this is the camera z-axis expressed in enu frame.*/
    SLVec3f normalCamXYPlane = enuRc * SLVec3f(0, 0, 1);

    // enu x-y-plane definition:  this is just the z-axis
    SLVec3f normalEnuXYPlane = SLVec3f(0, 0, 1);

    /* 2. Estimation of intersection line (horizon): Then the crossproduct of
    both vectors defines the direction of the intersection line. In our special
    case we know that the origin is a point the lies on both planes. Then
    origin plus direction vector defines the horizon.*/
    SLVec3f enuHorizon;
    enuHorizon.cross(normalEnuXYPlane, normalCamXYPlane);
    enuHorizon.normalize();

    /* 3. Use horizon angle to express screen (camera plane) finger movement
    in enuUp-horizon plane express horizon in camera coordinate system.*/
    SLMat3f cRenu    = enuRc.transposed();
    SLVec3f cHorizon = cRenu * enuHorizon;
    cHorizon.normalize();

    // angle between x-axis and horizon
    float horizAngDEG = atan2f((float)cHorizon.y, (float)cHorizon.x) * RAD2DEG;

    // rotate display x- and y-offsets to enuUp - horizon plane
    SLVec3f cOffsetPix((SLfloat)_xOffsetPix, (SLfloat)_yOffsetPix, 0.f);
    SLMat3f rot(horizAngDEG, 0, 0, 1);
    enuOffsetPix = rot * cOffsetPix;

    /* 4. Apply rotation angles defined in camera plane onto vertical enu axis
    and horizon axis estimate focal length (todo: calculate once when fov is set)*/
    f = (SLfloat)sv->scrH() / (2.0f * tan(0.5f * fovV() * DEG2RAD));

    if (_devRot->offsetMode() == ROM_oneFingerXY)
    {
        if (_xOffsetPix != 0 && _yOffsetPix != 0)
        {
            float   yawOffsetRAD   = atanf((float)enuOffsetPix.x / f);
            float   pitchOffsetRAD = atanf((float)enuOffsetPix.y / f);
            SLMat3f rotVertical(yawOffsetRAD * RAD2DEG,
                                SLVec3f(0, 0, 1));
            SLMat3f rotHorizon(pitchOffsetRAD * RAD2DEG,
                               enuHorizon.x,
                               enuHorizon.y,
                               enuHorizon.z);

            // we have to right multiply new rotation because new rotations
            // are estimated w.r.t. enu coordinate frame
            _enucorrRenu = _enucorrRenu * rotHorizon * rotVertical;
        }
    }
    else if (_devRot->offsetMode() == ROM_oneFingerX) //||_devRot->offsetMode() == OM_fingerXRotYTrans)
    {
        if (_xOffsetPix != 0)
        {
            float   yawOffsetRAD = atanf((float)enuOffsetPix.x / f);
            SLMat3f rotVertical(yawOffsetRAD * RAD2DEG,
                                SLVec3f(0, 0, 1));

            // we have to right multiply new rotation because new rotations are estimated w.r.t. enu coordinate frame
            _enucorrRenu = _enucorrRenu * rotVertical;
        }
    }
}
//-----------------------------------------------------------------------------
/*!
Applies the view transform to the modelview matrix depending on the eye:
eye=-1 for the left and eye=1 for the right eye.
The view matrix that projects all points from the world coordinate system to
the camera coordinate system (that means relative to the camera) is the camera
nodes inverse world matrix.
*/
void SLCamera::setView(SLSceneView* sv, const SLEyeType eye)
{
    SLGLState* stateGL = SLGLState::instance();

    if (_camAnim == CA_deviceRotYUp)
    {
        if (!_devRot)
        {
            SL_WARN_MSG("SLCamera: _devRot is invalid");
            return;
        }

        // camera focal length
        float f = 1.f;

        // finger x-y-movement expressed in enu frame
        SLVec3f enuOffsetPix;

        ///////////////////////////////////////////////////////////////////////
        // Build pose of camera in world frame (scene) using device rotation //
        ///////////////////////////////////////////////////////////////////////

        // camera rotation with respect to (w.r.t.) sensor
        SLMat3f sRc;
        sRc.rotation(-90, 0, 0, 1);

        // sensor rotation w.r.t. east-north-down
        SLMat3f enuRs;
        enuRs.setMatrix(_devRot->rotationAveraged());

        // define camera to enu rotation matrix
        SLMat3f enuRc = enuRs * sRc;

        // Calculate and apply correction from finger x-y-rotation
        updateEnuCorrRenu(sv, enuRc, f, enuOffsetPix);

        SLMat3f wyRenucorr;
        if (_devRot->zeroYawAtStart())
        {
            // east-north-down w.r.t. world-yaw
            SLfloat rotYawOffsetDEG = -1 * _devRot->startYawRAD() * Utils::RAD2DEG + 90;
            if (rotYawOffsetDEG > 180)
                rotYawOffsetDEG -= 360;
            wyRenucorr.rotation(rotYawOffsetDEG, 0, 0, 1);
        }

        // world-yaw rotation w.r.t. world
        SLMat3f wRwy;
        wRwy.rotation(-90, 1, 0, 0);

        // combination of partial rotations to orientation of camera w.r.t world
        SLMat3f wRc = wRwy * wyRenucorr * _enucorrRenu * enuRc;

        // camera translations w.r.t world:
        // SLVec3f wtc = updateAndGetWM().translation();
        // SLVec3f wtc = _om.translation();

        // combination of rotation and translation:
        // SLMat4f wTc;
        // wTc.setRotation(wRc);
        // wTc.setTranslation(wtc);

        // set camera pose to the object matrix
        // om(wTc);
        _om.setRotation(wRc);
        needUpdate();

        /*
        //alternative concatenation of single transformations
        SLMat4f wTc_2;
        wTc_2.translate(updateAndGetWM().translation());
        wTc_2.rotate(-90, 1, 0, 0);
        wTc_2.rotate(rotYawOffsetDEG, 0, 0, 1);
        SLMat4f enuTs;
        enuTs.setRotation(s->deviceRotation());
        wTc_2 *= enuTs;
        wTc_2.rotate(-90, 0, 0, 1);
        */
    }

    // location sensor is turned on and the scene has a global reference position
    else if (_camAnim == CA_deviceRotLocYUp)
    {
        // camera focal length
        float f = 1.f;

        // finger x-y-movement expressed in enu frame
        SLVec3f enuOffsetPix;

        if (!_devRot) SL_EXIT_MSG("SLCamera::setView: _devRot not set!");
        if (!_devLoc) SL_EXIT_MSG("SLCamera::setView: _devLoc not set!");

        // The device rotation sensor (IMU) is turned on and sends rotation angles
        if (_devRot->isUsed())
        {
            /* Camera rotation w.r.t. to sensor
            This is how the camera coordinate system is aligned relative to the sensor coordinate
            system: we are in the camera coordinate system if we rotate the sensor coordinate
            system -90 degrees about the z-axis)*/
            SLMat3f sRc;
            sRc.rotation(-90, 0, 0, 1);

            /* Sensor rotation w.r.t. east-north-up (gliding averaged)
            This is how the sensor is aligned relative to the east-north-down coordinate system:
            this rotation is what we get from the device rotation sensor api.*/
            SLMat3f enuRs;
            enuRs.setMatrix(_devRot->rotationAveraged());

            // define camera to enu rotation matrix
            SLMat3f enuRc = enuRs * sRc;

            // Calculate and apply correction from finger x-y-rotation
            updateEnuCorrRenu(sv, enuRc, f, enuOffsetPix);

            /* enu rotation (after correction) w.r.t. world
            "world" is our scene coordinate system! This rotation matrix defines how the scene
            coordinate system is aligned relative to the enu coordinate system */
            SLMat3f wRenucorr;
            wRenucorr.rotation(-90, 1, 0, 0);

            /* camera rotation w.r.t world
            "world" is our scene coordinate system! We are searching for the camera pose in the
            scene (world) coordinate system! Combination of partial rotations to orientation of
            camera w.r.t world */
            SLMat3f wRc = wRenucorr * _enucorrRenu * enuRc;

            _om.setRotation(wRc);

            needUpdate();
        }

        // The device location sensor (GPS) is turned on and the scene has a global reference position
        if (_devLoc && _devLoc->hasOrigin())
        {
            if (_devLoc->isUsed())
            {
                // Direction vector from camera to world origin
                SLVec3d wtc = _devLoc->locENU() - _devLoc->originENU() + _devLoc->offsetENU();

                // Reset to default if device is too far away
                if (wtc.length() > _devLoc->locMaxDistanceM())
                    wtc = _devLoc->defaultENU() - _devLoc->originENU() + _devLoc->offsetENU();

                // Set the camera position
                SLVec3f wtc_f((SLfloat)wtc.x, (SLfloat)wtc.y, (SLfloat)wtc.z);

                _om.setTranslation(wtc_f);

                needUpdate();
            }
            else // with disabled GPS use the default location
            {
                // Direction vector from camera to world origin with default location in ENU
                SLVec3d wtc = _devLoc->defaultENU() - _devLoc->originENU() + _devLoc->offsetENU();

                // Set the camera position
                SLVec3f wtc_f((SLfloat)wtc.x, (SLfloat)wtc.y, (SLfloat)wtc.z);

                _om.setTranslation(wtc_f);

                needUpdate();
            }
        }

        /*Calculate and apply finger y-translation
        if (_devRot->offsetMode() == OM_fingerYTrans || _devRot->offsetMode() == OM_fingerXRotYTrans)
        {
            //_enucorrTRenu += enuOffsetPix.y / f * _distanceToObjectM;
            // Set the camera position
            const SLVec3f& wtc = _om.translation();
            SLVec3f        wtc_f((SLfloat)wtc.x,
                          (SLfloat)wtc.y + enuOffsetPix.y / f * _distanceToObjectM,
                          (SLfloat)wtc.z);
            _om.setTranslation(wtc_f);
            needUpdate();
        }*/
    }
    else if (_camAnim == CA_off)
    {
        // nothing
    }

    // clear stored finger rotation
    _xOffsetPix = 0;
    _yOffsetPix = 0;

    // The view matrix is the camera nodes inverse world matrix
    SLMat4f vm = updateAndGetWMI();

    // Single eye projection
    if (eye == ET_center)
    {
        // Standard case: Just overwrite the view matrix
        stateGL->viewMatrix.setMatrix(vm);
    }
    else // stereo viewing
    {
        if (_projType == P_stereoSideBySideD)
        {
            // half inter-pupilar distance
            SLfloat halfIPD = (SLfloat)eye * _stereoEyeSeparation * -0.5f;

            SLMat4f trackingPos;
            if (_camAnim == CA_deviceRotYUp)
            {
                // get the oculus or mobile device orientation
                SLQuat4f rotation;
                if (sv->s()->oculus()->isConnected())
                {
                    rotation = sv->s()->oculus()->orientation(eye);
                    trackingPos.translate(-1 * sv->s()->oculus()->position(eye));
                }
                // todo else rotation = s->deviceRotation();

                SLfloat rotX = 0.0f, rotY = 0.0f, rotZ = 0.0f;
                rotation.toMat4().toEulerAnglesZYX(rotZ, rotY, rotX);
                /*
                SL_LOG("rotx : %3.1f, roty: %3.1f, rotz: %3.1f",
                       rotX * SL_RAD2DEG,
                       rotY * SL_RAD2DEG,
                       rotZ * SL_RAD2DEG);
                */

                SLVec3f viewAdjust = sv->s()->oculus()->viewAdjust(eye) * _unitScaling;

                SLMat4f vmEye(SLMat4f(viewAdjust.x,
                                      viewAdjust.y,
                                      viewAdjust.z) *
                              rotation.inverted().toMat4() * trackingPos * vm);
                stateGL->viewMatrix = vmEye;
            }
            else
            {
                SLMat4f vmEye(SLMat4f(halfIPD, 0.0f, 0.f) * vm);
                stateGL->viewMatrix = vmEye;
            }
        }
        else
        {
            // Get central camera vectors eye, lookAt, lookUp out of the view matrix vm
            SLVec3f EYE, LA, LU, LR;
            vm.lookAt(&EYE, &LA, &LU, &LR);

            // Shorten LR to half of the eye dist (eye=-1 for left, eye=1 for right)
            LR *= _stereoEyeSeparation * 0.5f * (SLfloat)eye;

            // Set the OpenGL view matrix for the left eye
            SLMat4f vmEye;
            vmEye.lookAt(EYE + LR, EYE + _focalDist * LA + LR, LU);
            stateGL->viewMatrix = vmEye;
        }
    }
}
//-----------------------------------------------------------------------------
//! Sets the view to look from a direction towards the current focal point
void SLCamera::lookFrom(const SLVec3f& fromDir,
                        const SLVec3f& upDir)
{
    SLVec3f lookAt = focalPointWS();
    translation(lookAt + _focalDist * fromDir);
    this->lookAt(lookAt, upDir);
}
//-----------------------------------------------------------------------------
//! SLCamera::animationStr() returns the animation enum as string
SLstring SLCamera::animationStr() const
{
    switch (_camAnim)
    {
        case CA_turntableYUp: return "Turntable Y up";
        case CA_turntableZUp: return "Turntable Z up";
        case CA_walkingYUp: return "Walking Y up";
        case CA_walkingZUp: return "Walking Z up";
        case CA_deviceRotYUp: return "Device Rotated Y up";
        case CA_deviceRotLocYUp: return "Device Rotated Y up and Gps Positioned";
        default: return "unknown";
    }
}

//-----------------------------------------------------------------------------
/*
Event Handlers: Because the SLNode class also inherits the SLEventHandler
class a node can also act as a event handler. The camera class uses this to
implement the camera animation.
*/
//-----------------------------------------------------------------------------
//! Gets called whenever a mouse button gets pressed.
SLbool SLCamera::onMouseDown(const SLMouseButton button,
                             const SLint         x,
                             const SLint         y,
                             const SLKey         mod)
{
    // Init both position in case that the second finger came with delay
    _oldTouchPos1.set((SLfloat)x, (SLfloat)y);
    _oldTouchPos2.set((SLfloat)x, (SLfloat)y);

    if (button == MB_left)
    {
        // Start selection rectangle. See also SLMesh::handleRectangleSelection
        if (mod & K_ctrl)
            _selectRect.tl(_oldTouchPos1);

        // Start deselection rectangle. See also SLMesh::handleRectangleSelection
        if (mod & K_alt)
            _deselectRect.tl(_oldTouchPos1);

        if (_camAnim == CA_trackball)
            _trackballStartVec = trackballVec(x, y);
    }
    return true;
}
//-----------------------------------------------------------------------------
//! Gets called whenever the mouse is moved.
SLbool SLCamera::onMouseMove(const SLMouseButton button,
                             const SLint         x,
                             const SLint         y,
                             const SLKey         mod)
{
    if (button == MB_left) //==================================================
    {
        // Set selection rectangle. See also SLMesh::handleRectangleSelection
        if (mod & K_ctrl)
        {
            _selectRect.setScnd(SLVec2f((SLfloat)x, (SLfloat)y));
            return true;
        }

        // Set deselection rectangle. See also SLMesh::handleRectangleSelection
        if (mod & K_alt)
        {
            _deselectRect.setScnd(SLVec2f((SLfloat)x, (SLfloat)y));
            return true;
        }

        // Position and directions in view space
        SLVec3f positionVS = this->translationOS();
        SLVec3f forwardVS  = this->forwardOS();
        SLVec3f rightVS    = this->rightOS();

        // The lookAt point
        SLVec3f lookAtPoint = positionVS + _focalDist * forwardVS;

        // Determine rot angles around x- & y-axis
        SLfloat dY = ((SLfloat)y - _oldTouchPos1.y) * _mouseRotationFactor;
        SLfloat dX = ((SLfloat)x - _oldTouchPos1.x) * _mouseRotationFactor;

        if (_camAnim == CA_turntableYUp) //....................................
        {
            SLMat4f rot;
            rot.translate(lookAtPoint);
            rot.rotate(-dX, SLVec3f(0, 1, 0));
            rot.rotate(-dY, rightVS);
            rot.translate(-lookAtPoint);

            _om.setMatrix(rot * _om);
            needUpdate();
        }
        else if (_camAnim == CA_turntableZUp) //...............................
        {
            SLMat4f rot;
            rot.translate(lookAtPoint);
            rot.rotate(dX, SLVec3f(0, 0, 1));
            rot.rotate(dY, rightVS);
            rot.translate(-lookAtPoint);

            _om.setMatrix(rot * _om);
            needUpdate();
        }
        else if (_camAnim == CA_trackball) //..................................
        {
            // Reference: https://en.wikibooks.org/wiki/OpenGL_Programming/Modern_OpenGL_Tutorial_Arcball
            // calculate current mouse vector at current mouse position
            SLVec3f curMouseVec = trackballVec(x, y);

            // calculate angle between the old and the current mouse vector
            // Take care that the dot product isn't greater than 1.0 otherwise
            // the acos will return undefined.
            SLfloat dot   = _trackballStartVec.dot(curMouseVec);
            SLfloat angle = acos(dot > 1 ? 1 : dot) * Utils::RAD2DEG;

            // calculate rotation axis with the cross product
            SLVec3f axisVS;
            axisVS.cross(_trackballStartVec, curMouseVec);

            // To stabilise the axis we average it with the last axis
            static SLVec3f lastAxisVS = SLVec3f::ZERO;
            if (lastAxisVS != SLVec3f::ZERO) axisVS = (axisVS + lastAxisVS) / 2.0f;

            // Because we calculate the mouse vectors from integer mouse positions
            // we can get some numerical instability from the dot product when the
            // mouse is on the silhouette of the virtual sphere.
            // We calculate therefore an alternative for the angle from the mouse
            // motion length.
            SLVec2f dMouse(_oldTouchPos1.x - (SLfloat)x,
                           _oldTouchPos1.y - (SLfloat)y);
            SLfloat dMouseLenght = dMouse.length();
            if (angle > dMouseLenght) angle = dMouseLenght * 0.2f;

            // Transform rotation axis into world space
            // Remember: The cameras om is the view matrix inversed
            SLVec3f axisWS = _om.mat3() * axisVS;

            // Create rotation from one rotation around one axis
            SLMat4f rot;
            rot.translate(lookAtPoint);          // undo camera translation
            rot.rotate((SLfloat)-angle, axisWS); // create incremental rotation
            rot.translate(-lookAtPoint);         // redo camera translation
            _om.setMatrix(rot * _om);            // accumulate rotation to the existing camera matrix

            // set current to last
            _trackballStartVec = curMouseVec;
            lastAxisVS         = axisVS;

            needUpdate();
        }
        else if (_camAnim == CA_walkingYUp) //.................................
        {
            dY *= 0.5f;
            dX *= 0.5f;

            SLMat4f rot;
            rot.rotate(-dX, SLVec3f(0, 1, 0));
            rot.rotate(-dY, rightVS);

            forwardVS.set(rot.multVec(forwardVS));
            lookAt(positionVS + forwardVS);
            needUpdate();
        }
        else if (_camAnim == CA_walkingZUp) //.................................
        {
            dY *= 0.5f;
            dX *= 0.5f;

            SLMat4f rot;
            rot.rotate(-dX, SLVec3f(0, 0, 1));
            rot.rotate(-dY, rightVS);

            forwardVS.set(rot.multVec(forwardVS));
            lookAt(positionVS + forwardVS, SLVec3f(0, 0, 1));
            needWMUpdate();
        }
        else if (_camAnim == CA_deviceRotLocYUp || _camAnim == CA_deviceRotYUp)
        {
            if (_devRot)
            {
                if (_devRot->offsetMode() == ROM_oneFingerX ||
                    _devRot->offsetMode() == ROM_oneFingerXY)
                {
                    _yOffsetPix += (SLint)((SLfloat)y - _oldTouchPos1.y);
                    _xOffsetPix += (SLint)((SLfloat)x - _oldTouchPos1.x);
                }
            }
        }

        _oldTouchPos1.set((SLfloat)x, (SLfloat)y);
    }
    else if (button == MB_middle) //===========================================
    {
        if (_camAnim == CA_turntableYUp ||
            _camAnim == CA_turntableZUp ||
            _camAnim == CA_trackball)
        {
            // Calculate the fraction delta of the mouse movement
            SLVec2f dMouse((SLfloat)x - _oldTouchPos1.x,
                           _oldTouchPos1.y - (SLfloat)y);
            dMouse.x /= (SLfloat)_viewport.width;
            dMouse.y /= (SLfloat)_viewport.height;

            // scale factor depending on the space size at focal dist
            SLfloat spaceH = tan(Utils::DEG2RAD * _fovV / 2) * _focalDist * 2.0f;
            SLfloat spaceW = spaceH * _viewportRatio;

            dMouse.x *= spaceW;
            dMouse.y *= spaceH;

            if (mod == K_ctrl)
                translate(SLVec3f(-dMouse.x, 0, dMouse.y), TS_object);
            else
                translate(SLVec3f(-dMouse.x, -dMouse.y, 0), TS_object);

            // todo anim : bleibt!!
            _oldTouchPos1.set((SLfloat)x, (SLfloat)y);
        }
    } //=======================================================================

    return true;
}
//-----------------------------------------------------------------------------
//! Gets called whenever the mouse button is released
SLbool SLCamera::onMouseUp(const SLMouseButton button,
                           const SLint         x,
                           const SLint         y,
                           const SLKey         mod)
{
    if (button == MB_left)
    {
        // End rectangle select. See also SLMesh::handleRectangleSelection
        if (mod & K_ctrl)
        {
            _selectRect.setZero();
            return true;
        }
        // End rectangle deselect. See also SLMesh::handleRectangleSelection
        if (mod & K_alt)
        {
            _deselectRect.setZero();
            return true;
        }

        // todo anim
        if (_camAnim == CA_turntableYUp)
            return true;
        else if (_camAnim == CA_walkingYUp)
            return true;
    }
    else if (button == MB_middle)
    {
        return true;
    }
    return false;
}
//-----------------------------------------------------------------------------
/*!
SLCamera::onMouseWheel event handler moves camera forwards or backwards
*/
SLbool SLCamera::onMouseWheel(const SLint delta,
                              const SLKey mod)
{
    SLfloat sign = (SLfloat)Utils::sign(delta);

    if (_camAnim == CA_turntableYUp ||
        _camAnim == CA_turntableZUp ||
        _camAnim == CA_trackball) //...........................................
    {
        if (mod == K_none)
        {
            translate(SLVec3f(0, 0, -sign * _focalDist * _keyboardDeltaPos), TS_object);
            _focalDist += -sign * _focalDist * _keyboardDeltaPos;

            needUpdate();
        }
        if (mod == K_ctrl)
        {
            _stereoEyeSeparation *= (1.0f + sign * 0.1f);
        }
        if (mod == K_alt)
        {
            _fovV += sign * 5.0f;
            currentFOV = _fovV;
        }
    }
    else if (_camAnim == CA_walkingYUp || _camAnim == CA_walkingZUp) //........
    {
        _maxSpeed *= (1.0f + sign * 0.1f);
    }
    return false;
}
//-----------------------------------------------------------------------------
/*!
SLCamera::onDoubleTouch gets called whenever two fingers touch a handheld
screen.
*/
SLbool SLCamera::onTouch2Down(const SLint x1,
                              const SLint y1,
                              const SLint x2,
                              const SLint y2)
{
    _oldTouchPos1.set((SLfloat)x1, (SLfloat)y1);
    _oldTouchPos2.set((SLfloat)x2, (SLfloat)y2);
    return true;
}

//-----------------------------------------------------------------------------
/*!
SLCamera::onTouch2Move gets called whenever two fingers move on a handheld
screen.
*/
SLbool SLCamera::onTouch2Move(const SLint x1,
                              const SLint y1,
                              const SLint x2,
                              const SLint y2)
{
    SLVec2f now1((SLfloat)x1, (SLfloat)y1);
    SLVec2f now2((SLfloat)x2, (SLfloat)y2);
    SLVec2f delta1(now1 - _oldTouchPos1);
    SLVec2f delta2(now2 - _oldTouchPos2);

    // Average out the deltas over the last 4 events for correct 1 pixel moves
    static SLuint  cnt = 0;
    static SLVec2f d1[4];
    static SLVec2f d2[4];
    d1[cnt % 4] = delta1;
    d2[cnt % 4] = delta2;
    SLVec2f avgDelta1(d1[0].x + d1[1].x + d1[2].x + d1[3].x,
                      d1[0].y + d1[1].y + d1[2].y + d1[3].y);
    SLVec2f avgDelta2(d2[0].x + d2[1].x + d2[2].x + d2[3].x,
                      d2[0].y + d2[1].y + d2[2].y + d2[3].y);
    avgDelta1 /= 4.0f;
    avgDelta2 /= 4.0f;
    cnt++;

    SLfloat r1, phi1, r2, phi2;
    avgDelta1.toPolar(r1, phi1);
    avgDelta2.toPolar(r2, phi2);

    // scale factor depending on the space sice at focal dist
    SLfloat spaceH = tan(Utils::DEG2RAD * _fovV / 2) * _focalDist * 2.0f;
    SLfloat spaceW = spaceH * _viewportRatio;

    // if fingers move parallel slide camera vertically or horizontally
    if (Utils::abs(phi1 - phi2) < 0.2f)
    {
        // Calculate center between finger points
        SLVec2f nowCenter((now1 + now2) * 0.5f);
        SLVec2f oldCenter((_oldTouchPos1 + _oldTouchPos2) * 0.5f);

        // For first move set oldCenter = nowCenter
        if (oldCenter == SLVec2f::ZERO) oldCenter = nowCenter;

        SLVec2f delta(nowCenter - oldCenter);

        // scale to 0-1
        delta.x /= (SLfloat)_viewport.width;
        delta.y /= (SLfloat)_viewport.height;

        // scale to space size
        delta.x *= spaceW;
        delta.y *= spaceH;

        if (_camAnim == CA_turntableYUp || _camAnim == CA_turntableZUp)
        {
            // apply delta to x- and y-position
            translate(SLVec3f(-delta.x, delta.y, 0), TS_object);
        }
        else if (_camAnim == CA_walkingYUp || _camAnim == CA_walkingZUp)
        {
            //_moveDir.x = delta.x * 100.0f,
            //_moveDir.z = delta.y * 100.0f;
        }
        else if (_camAnim == CA_deviceRotYUp || _camAnim == CA_deviceRotLocYUp)
        {
            if (_devLoc)
            {
                if (_devLoc->offsetMode() == LOM_twoFingerY)
                {
                    // string msg = "TwoFingerOffset: " + delta.toString();
                    // SL_LOG(msg.c_str());
                    SLVec3d offsetENU = _devLoc->offsetENU();
                    delta *= _mouseRotationFactor;
                    offsetENU.y += delta.y;
                    _devLoc->offsetENU(offsetENU);
                }
            }
        }
    }
    else // Two finger pinch
    {
        // Calculate vector between fingers
        SLVec2f nowDist(now2 - now1);
        SLVec2f oldDist(_oldTouchPos2 - _oldTouchPos1);

        // For first move set oldDist = nowDist
        if (oldDist == SLVec2f::ZERO) oldDist = nowDist;

        SLfloat delta = oldDist.length() - nowDist.length();

        if (_camAnim == CA_turntableYUp)
        { // scale to 0-1
            delta /= (SLfloat)_viewport.height;

            // scale to space height
            delta *= spaceH * 2;

            // apply delta to the z-position
            translate(SLVec3f(0, 0, delta), TS_object);
            _focalDist += delta;
        }
        else if (_camAnim == CA_walkingYUp)
        {
            // change field of view
            _fovV += Utils::sign(delta) * 0.5f;
            currentFOV = _fovV;
        }
    }

    _oldTouchPos1.set((SLfloat)x1, (SLfloat)y1);
    _oldTouchPos2.set((SLfloat)x2, (SLfloat)y2);
    return true;
}
//-----------------------------------------------------------------------------
/*!
SLCamera::onDoubleTouch gets called whenever two fingers touch a handheld
screen.
*/
SLbool SLCamera::onTouch2Up(const SLint x1,
                            const SLint y1,
                            const SLint x2,
                            const SLint y2)
{
    _velocity.set(0.0f, 0.0f, 0.0f);
    return true;
}
//-----------------------------------------------------------------------------
/*!
SLCamera::onKeyPress applies the keyboard view navigation to the view matrix.
The key code constants are defined in SL.h
*/
SLbool SLCamera::onKeyPress(const SLKey key, const SLKey mod)
{
    // Keep in sync with SLDemoGui::buildMenuBar
    switch ((SLchar)key)
    {
        case 'W': _moveDir.z -= 1.0f; return true;
        case 'A': _moveDir.x -= 1.0f; return true;
        case 'S': _moveDir.z += 1.0f; return true;
        case 'D': _moveDir.x += 1.0f; return true;
        case 'Q': _moveDir.y -= 1.0f; return true;
        case 'E': _moveDir.y += 1.0f; return true;
        case (SLchar)K_up: _moveDir.z -= 1.0f; return true;
        case (SLchar)K_down: _moveDir.z += 1.0f; return true;
        case (SLchar)K_right: _moveDir.x += 1.0f; return true;
        case (SLchar)K_left: _moveDir.x -= 1.0f; return true;

        // View setting as in standard Blender
        case '1':
            if (mod == K_ctrl)
                lookFrom(-SLVec3f::AXISZ);
            else
                lookFrom(SLVec3f::AXISZ);
            return true;
        case '3':
            if (mod == K_ctrl)
                lookFrom(-SLVec3f::AXISX);
            else
                lookFrom(SLVec3f::AXISX);
            return true;
        case '7':
            if (mod == K_ctrl)
                lookFrom(-SLVec3f::AXISY, SLVec3f::AXISZ);
            else
                lookFrom(SLVec3f::AXISY, -SLVec3f::AXISZ);
            return true;

        default: return false;
    }
}
//-----------------------------------------------------------------------------
/*!
SLCamera::onKeyRelease gets called when a key is released
*/
SLbool SLCamera::onKeyRelease(const SLKey key, const SLKey mod)
{
    switch ((SLchar)key)
    {
        case 'D': _moveDir.x -= 1.0f; return true;
        case 'A': _moveDir.x += 1.0f; return true;
        case 'Q': _moveDir.y += 1.0f; return true;
        case 'E': _moveDir.y -= 1.0f; return true;
        case 'S': _moveDir.z -= 1.0f; return true;
        case 'W': _moveDir.z += 1.0f; return true;
        case (SLchar)K_up: _moveDir.z += 1.0f; return true;
        case (SLchar)K_down: _moveDir.z -= 1.0f; return true;
        case (SLchar)K_right: _moveDir.x -= 1.0f; return true;
        case (SLchar)K_left: _moveDir.x += 1.0f; return true;
    }

    return false;
}
//-----------------------------------------------------------------------------
//! SLCamera::setFrustumPlanes set the 6 plane from the view frustum.
/*! SLCamera::setFrustumPlanes set the 6 frustum planes by extracting the plane
coefficients from the combined view and projection matrix.
*/
void SLCamera::setFrustumPlanes()
{
    SLGLState* stateGL = SLGLState::instance();
    SLFrustum::viewToFrustumPlanes(_plane,
                                   stateGL->projectionMatrix,
                                   stateGL->viewMatrix);
}
//-----------------------------------------------------------------------------
//!< Horizontal field of view
SLfloat SLCamera::fovH() const
{
    float f = (0.5f * (float)_viewport.height) / tanf(_fovV * 0.5f * Utils::DEG2RAD);
    return 2.f * atanf(0.5f * (float)_viewport.width / f) * Utils::RAD2DEG;
}
//-----------------------------------------------------------------------------
//! eyeToPixelRay returns the a ray from the eye to the center of a pixel.
/*! This method is used for object picking. The calculation is the same as for
primary rays in Ray Tracing.
*/
void SLCamera::eyeToPixelRay(SLfloat x, SLfloat y, SLRay* ray)
{
    SLVec3f EYE, LA, LU, LR;

    // get camera vectors eye, lookAt, lookUp from view matrix
    updateAndGetVM().lookAt(&EYE, &LA, &LU, &LR);

    if (_projType == P_monoOrthographic)
    { /*
        In orthographic projection the top-left vector (TL) points
        from the eye to the center of the TL-left pixel of a plane that
        parallel to the projection plan at zero distance from the eye.
        */
        SLVec3f pos(updateAndGetVM().translation());
        SLfloat hh = tan(Utils::DEG2RAD * _fovV * 0.5f) * pos.length();
        SLfloat hw = hh * _viewportRatio;

        // calculate the size of a pixel in world coords.
        SLfloat pixel = hw * 2 / (SLfloat)_viewport.width;

        SLVec3f TL  = EYE - hw * LR + hh * LU + pixel / 2 * LR - pixel / 2 * LU;
        SLVec3f dir = LA;
        dir.normalize();
        ray->setDir(dir);
        ray->origin.set(TL + pixel * (x * LR - y * LU));
    }
    else
    { /*
        In perspective projection the top-left vector (TL) points
        from the eye to the center of the top-left pixel on a projection
        plan in focal distance. See also the computer graphics script about
        primary ray calculation.
        */
        // calculate half window width & height in world coords
        SLfloat hh = tan(Utils::DEG2RAD * _fovV * 0.5f) * _focalDist;
        SLfloat hw = hh * _viewportRatio;

        // calculate the size of a pixel in world coords.
        SLfloat pixel = hw * 2 / (SLfloat)_viewport.width;

        // calculate a vector to the center (C) of the top left (TL) pixel
        SLVec3f C   = LA * _focalDist;
        SLVec3f TL  = C - hw * LR + hh * LU + pixel * 0.5f * (LR - LU);
        SLVec3f dir = TL + pixel * (x * LR - y * LU);

        dir.normalize();
        ray->setDir(dir);
        ray->origin.set(EYE);
    }

    ray->length      = FLT_MAX;
    ray->depth       = 1;
    ray->contrib     = 1.0f;
    ray->type        = PRIMARY;
    ray->x           = x;
    ray->y           = y;
    ray->hitTriangle = -1;
    ray->hitNormal.set(SLVec3f::ZERO);
    ray->hitPoint.set(SLVec3f::ZERO);
    ray->hitNode     = nullptr;
    ray->hitMesh     = nullptr;
    ray->srcTriangle = 0;
}

void SLCamera::UVWFrame(SLVec3f& EYE, SLVec3f& U, SLVec3f& V, SLVec3f& W)
{
    SLVec3f LA, LU, LR;

    // get camera vectors eye, lookAt, lookUp from view matrix
    updateAndGetVM().lookAt(&EYE, &LA, &LU, &LR);

    W = LA * _focalDist;

    U.cross(W, LU);
    U.normalize();

    V.cross(U, W);
    V.normalize();

    SLfloat hh = tan(Utils::DEG2RAD * _fovV * 0.5f) * _focalDist;
    SLfloat hw = hh * _viewportRatio;
    V *= hh;
    U *= hw;
}

//-----------------------------------------------------------------------------
//! Project a world position into screen coordinates
SLVec2f SLCamera::projectWorldToNDC(const SLVec4f& worldPos) const
{
    SLMat4f projectionMatrix;
    projectionMatrix.perspective(_fovV, _viewportRatio, _clipNear, _clipFar);
    SLMat4f viewMatrix = updateAndGetVM();

    SLVec4f eyePos  = viewMatrix * worldPos;
    SLVec4f clipPos = projectionMatrix * eyePos;

    SLVec4f ndcPos = clipPos / clipPos.w;

    SLVec2f result = SLVec2f(ndcPos.x,
                             ndcPos.y);

    return result;
}
//-----------------------------------------------------------------------------
//! SLCamera::isInFrustum does a simple and fast frustum culling test for AABBs
/*! SLCamera::isInFrustum checks if the bounding sphere of an AABB is within
the view frustum defined by its 6 planes by simply testing the distance of the
AABBs center minus its radius. This is faster than the AABB in frustum test but
not as precise. Please refer to the nice tutorial on frustum culling on:
https://cgvr.cs.uni-bremen.de/teaching/cg_literatur/lighthouse3d_view_frustum_culling/
*/
SLbool SLCamera::isInFrustum(SLAABBox* aabb)
{
    // check the 6 planes of the frustum
    for (auto& i : _plane)
    {
        SLfloat distance = i.distToPoint(aabb->centerWS());
        if (distance < -aabb->radiusWS())
        {
            aabb->isVisible(false);
            return false;
        }
    }
    aabb->isVisible(true);

    // Calculate squared dist. from AABB's center to viewer for blend sorting.
    SLVec3f viewToCenter(_wm.translation() - aabb->centerWS());
    aabb->sqrViewDist(viewToCenter.lengthSqr());
    return true;
}
//-----------------------------------------------------------------------------
//! SLCamera::to_string returns important camera parameter as a string
SLstring SLCamera::toString() const
{
    SLMat4f            vm = updateAndGetVM();
    std::ostringstream ss;
    ss << "Projection: " << projTypeStr() << endl;
    ss << "FOV: " << _fovV << endl;
    ss << "ClipNear: " << _clipNear << endl;
    ss << "ClipFar: " << _clipFar << endl;
    ss << "Animation: " << animationStr() << endl;
    ss << vm.toString() << endl;
    return ss.str();
}
//-----------------------------------------------------------------------------
//! Returns a vector from the window center to a virtual trackball at [x,y].
/*! The trackball vector is a vector from the window center to a hemisphere
over the window at the specified cursor position. With two trackball vectors
you can calculate a single rotation axis with the cross product. This routine
is used for the trackball camera animation.
 */
SLVec3f SLCamera::trackballVec(const SLint x, const SLint y) const
{
    // Calculate x & y component to the virtual unit sphere
    SLfloat r = (SLfloat)(_viewport.width < _viewport.height ? _viewport.width / 2 : _viewport.height / 2) * _trackballSize;

    SLVec3f vec((SLfloat)((SLfloat)x - (SLfloat)_viewport.width * 0.5f) / r,
                -(SLfloat)((SLfloat)y - (SLfloat)_viewport.height * 0.5f) / r);

    // d = length of vector x,y
    SLfloat d = sqrt(vec.x * vec.x + vec.y * vec.y);

    // z component with pytagoras
    if (d < 1.0f)
        vec.z = sqrt(1.0f - d * d);
    else
    {
        vec.z = 0.0f;
        vec.normalize(); // d >= 1, so normalize
    }
    return vec;
}
//-----------------------------------------------------------------------------
//! Pass camera parameters to the uniform variables
void SLCamera::passToUniforms(SLGLProgram* program)
{
    assert(program && "SLCamera::passToUniforms: No shader program set!");

    program->uniform1i("u_camProjType", _projType);
    program->uniform1i("u_camStereoEye", _stereoEye);
    program->uniformMatrix3fv("u_camStereoColors",
                              1,
                              (SLfloat*)&_stereoColorFilter);
    // Pass fog parameters
    if (_fogColorIsBack)
        _fogColor = _background.avgColor();
    _fogStart = _clipNear;
    _fogEnd   = _clipFar;

    program->uniform1i("u_camFogIsOn", _fogIsOn);
    program->uniform1i("u_camFogMode", _fogMode);
    program->uniform1f("u_camFogDensity", _fogDensity);
    program->uniform1f("u_camFogStart", _fogStart);
    program->uniform1f("u_camFogEnd", _fogEnd);
    program->uniform1f("u_camClipNear", _clipNear);
    program->uniform1f("u_camClipFar", _clipFar);
    program->uniform1f("u_camBkgdWidth", _background.rect().width);
    program->uniform1f("u_camBkgdHeight", _background.rect().height);
    program->uniform1f("u_camBkgdLeft", _background.rect().x);
    program->uniform1f("u_camBkgdBottom", _background.rect().y);
    program->uniform4fv("u_camFogColor", 1, (SLfloat*)&_fogColor);
}
//-----------------------------------------------------------------------------
