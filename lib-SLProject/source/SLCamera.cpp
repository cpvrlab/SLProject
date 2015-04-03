//#############################################################################
//  File:      SLCamera.cpp
//  Author:    Marc Wacker, Marcus Hudritsch
//  Date:      July 2014
//  Codestyle: https://github.com/cpvrlab/SLProject/wiki/Coding-Style-Guidelines
//  Copyright: Marcus Hudritsch
//             This software is provide under the GNU General Public License
//             Please visit: http://opensource.org/licenses/GPL-3.0
//#############################################################################

#include <stdafx.h>           // precompiled headers
#ifdef SL_MEMLEAKDETECT       // set in SL.h for debug config only
#include <debug_new.h>        // memory leak detector
#endif

#include <SLSceneView.h>
#include <SLCamera.h>
#include <SLRay.h>
#include <SLAABBox.h>

//-----------------------------------------------------------------------------
// Static global default parameters for new cameras
SLCamAnim    SLCamera::currentAnimation    = turntableYUp;
SLProjection SLCamera::currentProjection   = monoPerspective;
SLfloat      SLCamera::currentFOV          = 45.0f;
SLint        SLCamera::currentDevRotation  = 0;
//-----------------------------------------------------------------------------
SLCamera::SLCamera() 
    : SLNode("Camera"), 
      _maxSpeed(2.0f), 
      _velocity(0.0f, 0.0f, 0.0f),
      _drag(0.05f),
      _brakeAccel(16.0f),
      _moveAccel(16.0f),
      _moveDir(0, 0, 0),
      _acceleration(0, 0, 0),
      _unitScaling(1.0f)
{  
    _fovInit      = 0;
    _clipNear     = 0.1f;
    _clipFar      = 300.0f;
    _fov          = 45;               //currentFOV;
    _projection   = monoPerspective;  //currentProjection;
    _camAnim      = turntableYUp;     //currentAnimation
    _useDeviceRot = true;

    // depth of field parameters
    _lensDiameter = 0.3f;
    _lensSamples.samples(1,1); // e.g. 10,10 > 10x10=100 lenssamples
    _focalDist = 5;
   
    _eyeSeparation = _focalDist / 30.0f;
}
//-----------------------------------------------------------------------------
//! Destructor: Be sure to delete the OpenGL display list.
SLCamera::~SLCamera() 
{  
}
//-----------------------------------------------------------------------------
/*! SLCamera::camUpdate does the smooth transition for the walk animation. It
is called in every frame. It moves the camera after the key was released and
smoothly stops the motion by decreasing the speed every frame.
*/
SLbool SLCamera::camUpdate(SLfloat elapsedTimeMS)
{  
    if (_velocity == SLVec3f::ZERO && _moveDir == SLVec3f::ZERO)
        return false;

    SLfloat dtS = elapsedTimeMS * 0.001f;

    SLbool braking = false;
    if (_moveDir != SLVec3f::ZERO)
    {   
        // x and z movement direction vector should be projected on the x,z plane while
        // but still in local space
        // the y movement direction should alway be in world space
        SLVec3f f = forward();
        f.y = 0;
        f.normalize();

        SLVec3f r = right();
        r.y = 0;
        r.normalize();

        _acceleration = f * -_moveDir.z + r * _moveDir.x;
        _acceleration.y = _moveDir.y;
        _acceleration.normalize();
        _acceleration *= _moveAccel;

    } // accelerate in the opposite velocity to brake
    else
    {  
        _acceleration = -_velocity.normalized() * _brakeAccel;
        braking = true;
    }
    
    // accelerate
    SLfloat velMag = _velocity.length();
    SLVec3f increment = _acceleration * dtS; // all units in m/s, convert MS to S
    
    // early out if we're braking and the velocity would fall < 0
    if (braking && increment.lengthSqr() > _velocity.lengthSqr())
    {
        _velocity.set(SLVec3f::ZERO);
        return false;
    }

    _velocity += increment - _drag * _velocity * dtS; 
    velMag = _velocity.length();

    // don't go over max speed
    if (velMag > _maxSpeed)
        _velocity = _velocity.normalized() * _maxSpeed;

    // final delta movement vector
    SLVec3f delta = _velocity * dtS;

    // adjust for scaling (if the character is shrinked or enlarged)
    delta *= _unitScaling;

    translate(delta, TS_World);
    
    return true;
    
    //SL_LOG("cs: %3.2f | %3.2f, %3.2f, %3.2f\n", _velocity.length(), _acceleration.x, _acceleration.y, _acceleration.z);

    /* OLD CODE BELOW
    // ToDo: The recursive update traversal is not yet implemented
    if (_maxSpeed != SLVec3f::ZERO || _curSpeed != SLVec3f::ZERO)
    {  
        // delta speed during acceleration/slow down
        SLfloat ds = _speedLimit / 20.0f;
      
        // Accelerate
        if (_maxSpeed.x>0 && _curSpeed.x<_maxSpeed.x) _curSpeed.x += ds; else
        if (_maxSpeed.x<0 && _curSpeed.x>_maxSpeed.x) _curSpeed.x -= ds;
        if (_maxSpeed.y>0 && _curSpeed.y<_maxSpeed.y) _curSpeed.y += ds; else
        if (_maxSpeed.y<0 && _curSpeed.y>_maxSpeed.y) _curSpeed.y -= ds;      
        if (_maxSpeed.z>0 && _curSpeed.z<_maxSpeed.z) _curSpeed.z += ds; else
        if (_maxSpeed.z<0 && _curSpeed.z>_maxSpeed.z) _curSpeed.z -= ds;
      
        if (_curSpeed.z == 0.0f) {
            int i = 0;
        }

        // Slow down
        if (_maxSpeed.z == 0)
        {   if (_curSpeed.z > 0) 
            {  _curSpeed.z -= ds;
                if (_curSpeed.z < 0) _curSpeed.z = 0.0f;
            } else
            if (_curSpeed.z < 0) 
            {  _curSpeed.z += ds;
                if (_curSpeed.z > 0) _curSpeed.z = 0.0f;
            }
        }
        if (_maxSpeed.x == 0)
        {   if (_curSpeed.x < 0) 
            {  _curSpeed.x += ds;
                if (_curSpeed.x > 0) _curSpeed.x = 0.0f;
            } else 
            if (_curSpeed.x > 0) 
            {  _curSpeed.x -= ds;
                if (_curSpeed.x < 0) _curSpeed.x = 0.0f;
            }
        }
        if (_maxSpeed.y == 0)
        {   if (_curSpeed.y < 0) 
            {  _curSpeed.y += ds;
                if (_curSpeed.y > 0) _curSpeed.y = 0.0f;
            } else 
            if (_curSpeed.y > 0) 
            {  _curSpeed.y -= ds;
                if (_curSpeed.y < 0) _curSpeed.y = 0.0f;
            }
        }
      
        SL_LOG("cs: %3.1f, %3.1f, %3.1f\n", _curSpeed.x, _curSpeed.y, _curSpeed.z);
        SLfloat temp = _curSpeed.length();

        _curSpeed = updateAndGetWM().mat3() * _curSpeed;
        _curSpeed.y = 0;
        _curSpeed.normalize();
        _curSpeed *= temp;

        forward();

        SLVec3f delta(_curSpeed * elapsedTimeMS / 1000.0f);

        translate(delta, TS_World);
      
        return true;
    }
    return false;*/
}

//-----------------------------------------------------------------------------
//! SLCamera::drawMeshes draws the cameras frustum lines
/*!
Only draws the frustum lines without lighting when the camera is not the 
active one. This means that it can be seen from the active view point.
*/
void SLCamera::drawMeshes(SLSceneView* sv) 
{  
    if (sv->camera()!=this)
    {
        if (_projection == monoOrthographic)
        {
            const SLMat4f& vm = updateAndGetWMI();
            SLVec3f P[17*2];
            SLuint  i=0;
            SLVec3f pos(vm.translation());
            SLfloat t = tan(SL_DEG2RAD*_fov*0.5f) * pos.length();
            SLfloat b = -t;
            SLfloat l = -sv->scrWdivH() * t;
            SLfloat r = -l;

            // small line in view direction
            P[i++].set(0,0,0); P[i++].set(0,0,_clipNear*4);

            // frustum pyramid lines
            P[i++].set(r,t,_clipNear); P[i++].set(r,t,-_clipFar);
            P[i++].set(l,t,_clipNear); P[i++].set(l,t,-_clipFar);
            P[i++].set(l,b,_clipNear); P[i++].set(l,b,-_clipFar);
            P[i++].set(r,b,_clipNear); P[i++].set(r,b,-_clipFar);

            // around far clipping plane
            P[i++].set(r,t,-_clipFar); P[i++].set(r,b,-_clipFar);
            P[i++].set(r,b,-_clipFar); P[i++].set(l,b,-_clipFar);
            P[i++].set(l,b,-_clipFar); P[i++].set(l,t,-_clipFar);
            P[i++].set(l,t,-_clipFar); P[i++].set(r,t,-_clipFar);

            // around projection plane at focal distance
            P[i++].set(r,t,-_focalDist); P[i++].set(r,b,-_focalDist);
            P[i++].set(r,b,-_focalDist); P[i++].set(l,b,-_focalDist);
            P[i++].set(l,b,-_focalDist); P[i++].set(l,t,-_focalDist);
            P[i++].set(l,t,-_focalDist); P[i++].set(r,t,-_focalDist);

            // around near clipping plane
            P[i++].set(r,t,_clipNear); P[i++].set(r,b,_clipNear);
            P[i++].set(r,b,_clipNear); P[i++].set(l,b,_clipNear);
            P[i++].set(l,b,_clipNear); P[i++].set(l,t,_clipNear);
            P[i++].set(l,t,_clipNear); P[i++].set(r,t,_clipNear);

            _bufP.generate(P, i, 3);
        }
        else
        {
            SLVec3f P[17*2];
            SLuint  i=0;
            SLfloat aspect = sv->scrWdivH();
            SLfloat tanFov = tan(_fov*SL_DEG2RAD*0.5f);
            SLfloat tF =  tanFov * _clipFar;    //top far
            SLfloat rF =  tF * aspect;          //right far
            SLfloat lF = -rF;                   //left far
            SLfloat tP =  tanFov * _focalDist;  //top projection at focal distance
            SLfloat rP =  tP * aspect;          //right projection at focal distance
            SLfloat lP = -tP * aspect;          //left projection at focal distance
            SLfloat tN =  tanFov * _clipNear;   //top near
            SLfloat rN =  tN * aspect;          //right near
            SLfloat lN = -tN * aspect;          //left near

            // small line in view direction
            P[i++].set(0,0,0); P[i++].set(0,0,_clipNear*4);

            // frustum pyramid lines
            P[i++].set(0,0,0); P[i++].set(rF, tF,-_clipFar);
            P[i++].set(0,0,0); P[i++].set(lF, tF,-_clipFar);
            P[i++].set(0,0,0); P[i++].set(lF,-tF,-_clipFar);
            P[i++].set(0,0,0); P[i++].set(rF,-tF,-_clipFar);

            // around far clipping plane
            P[i++].set(rF, tF,-_clipFar); P[i++].set(rF,-tF,-_clipFar);
            P[i++].set(rF,-tF,-_clipFar); P[i++].set(lF,-tF,-_clipFar);
            P[i++].set(lF,-tF,-_clipFar); P[i++].set(lF, tF,-_clipFar);
            P[i++].set(lF, tF,-_clipFar); P[i++].set(rF, tF,-_clipFar);

            // around projection plane at focal distance
            P[i++].set(rP, tP,-_focalDist); P[i++].set(rP,-tP,-_focalDist);
            P[i++].set(rP,-tP,-_focalDist); P[i++].set(lP,-tP,-_focalDist);
            P[i++].set(lP,-tP,-_focalDist); P[i++].set(lP, tP,-_focalDist);
            P[i++].set(lP, tP,-_focalDist); P[i++].set(rP, tP,-_focalDist);

            // around near clipping plane
            P[i++].set(rN, tN,-_clipNear); P[i++].set(rN,-tN,-_clipNear);
            P[i++].set(rN,-tN,-_clipNear); P[i++].set(lN,-tN,-_clipNear);
            P[i++].set(lN,-tN,-_clipNear); P[i++].set(lN, tN,-_clipNear);
            P[i++].set(lN, tN,-_clipNear); P[i++].set(rN, tN,-_clipNear);

            _bufP.generate(P, i, 3);
        }
      
        _bufP.drawArrayAsConstantColorLines(SLCol3f::WHITE*0.7f);
    }
}
//-----------------------------------------------------------------------------
//! SLCamera::statsRec updates the statistic parameters
void SLCamera::statsRec(SLNodeStats &stats)
{  
    stats.numTriangles += 12;
    stats.numBytes += sizeof(SLCamera);
    SLNode::statsRec(stats);
}
//-----------------------------------------------------------------------------
/*!
SLCamera::updateAABBRec builds and returns the axis-aligned bounding box.
*/
SLAABBox& SLCamera::updateAABBRec()
{
    // calculate min & max in object space
    SLVec3f minOS, maxOS;
    calcMinMax(minOS, maxOS);

    // apply world matrix
    _aabb.fromOStoWS(minOS, maxOS, _wm);
    return _aabb;
}
//-----------------------------------------------------------------------------
/*!
SLCamera::calcMinMax calculates the axis alligned minimum and maximum point of
the camera position and the 4 near clipping plane points.
*/
void SLCamera::calcMinMax(SLVec3f &minV, SLVec3f &maxV)
{
    SLVec3f P[5];
    SLfloat tanFov = tan(_fov*SL_DEG2RAD*0.5f);
    SLfloat tN = tanFov * _clipNear; //top near
    SLfloat rN = tN * _aspect;       //right near

    // frustum pyramid lines
    P[0].set(0,0,0);

    // around near clipping plane
    P[1].set( rN, tN,-_clipNear);
    P[2].set( rN,-tN,-_clipNear);
    P[3].set(-rN,-tN,-_clipNear);
    P[4].set(-rN, tN,-_clipNear);

    // init min & max points
    minV.set( FLT_MAX,  FLT_MAX,  FLT_MAX);
    maxV.set(-FLT_MAX, -FLT_MAX, -FLT_MAX);

    // calc min and max point of all vertices
    for (SLuint i=0; i<5; ++i)
    {   if (P[i].x < minV.x) minV.x = P[i].x;
        if (P[i].x > maxV.x) maxV.x = P[i].x;
        if (P[i].y < minV.y) minV.y = P[i].y;
        if (P[i].y > maxV.y) maxV.y = P[i].y;
        if (P[i].z < minV.z) minV.z = P[i].z;
        if (P[i].z > maxV.z) maxV.z = P[i].z;
    }
}
//-----------------------------------------------------------------------------
//! Returns the projection type as string
SLstring SLCamera::projectionToStr(SLProjection p)
{  
    switch (p)
    {   case monoPerspective:      return "Perspective";
        case monoOrthographic:     return "Orthographic";
        case stereoSideBySide:     return "Side by Side";
        case stereoSideBySideP:    return "Side by Side proportional";
        case stereoSideBySideD:    return "Side by Side distorted";
        case stereoLineByLine:     return "Line by Line";
        case stereoColumnByColumn: return "Pixel by Pixel";
        case stereoPixelByPixel:   return "Checkerboard";
        case stereoColorRC:        return "Red-Cyan";
        case stereoColorRG:        return "Red-Green";
        case stereoColorRB:        return "Red-Blue";
        case stereoColorYB:        return "Yellow-Blue";
        default:                   return "Unknown";
    }
}
//-----------------------------------------------------------------------------
/*! 
Returns the height of the screen at focal distance. In stereo rendering this
shoud correspond to the height of the projection plane.
*/
SLfloat SLCamera::focalDistScrH() const
{  
   return tan(_fov*SL_DEG2RAD/2.0f) * _focalDist * 2.0f;
}
//-----------------------------------------------------------------------------
/*! 
Returns the width of the screen at focal distance. In stereo rendering this
shoud correspond to the width of the projection plane.
*/
SLfloat SLCamera::focalDistScrW() const
{  
    return focalDistScrH() * _aspect;
}
//-----------------------------------------------------------------------------
/*!
Sets the projection transformation matrix, the viewport transformation and the
drawing buffer. In case of a stereographic projection it additionally sets the
stereo splitting parameters such as the color masks and the color filter matrix
for stereo color anaglyphs. 
*/
void SLCamera::setProjection(SLSceneView* sv, const SLEye eye)
{  
    ////////////////////
    // Set Projection //
    ////////////////////

    const SLMat4f& vm = updateAndGetWMI();

    _stateGL->stereoEye  = eye;
    _stateGL->projection = _projection;
   
    SLVec3f pos(vm.translation());
    SLfloat top, bottom, left, right, d;   // frustum paramters
    _scrW = sv->scrW();
    _scrH = sv->scrH();
    _aspect = sv->scrWdivH();
   
    switch (_projection) 
    {  
        case monoPerspective:
            _stateGL->projectionMatrix.perspective(_fov, sv->scrWdivH(), _clipNear, _clipFar);
            break;

        case monoOrthographic:
            top    = tan(SL_DEG2RAD*_fov*0.5f) * pos.length();
            bottom = -top;
            left   = -sv->scrWdivH()*top;
            right  = -left;

            // The ortographic projection should have its near clip plane behind the camera
            // rather than slightly in front of it. Else we will see cross sections of scenes if
            // we zoom in close
            _stateGL->projectionMatrix.ortho(left,right,bottom,top, -_clipNear, _clipFar);
            break;

        case stereoSideBySideD:
            _stateGL->projectionMatrix = SLScene::current->oculus()->projection(eye);

            break;
        // all other stereo projections
        default: 
            // assymetric frustum shift d (see chapter stereo projection)
            d = (SLfloat)eye * 0.5f * _eyeSeparation * _clipNear / _focalDist;
            top    = tan(SL_DEG2RAD*_fov/2) * _clipNear;
            bottom = -top;
            left   = -sv->scrWdivH()*top - d;
            right  =  sv->scrWdivH()*top - d;
            _stateGL->projectionMatrix.frustum(left,right,bottom,top,_clipNear,_clipFar);
    }
   
    //////////////////
    // Set Viewport //
    //////////////////
   
    SLint w = sv->scrW();
    SLint h = sv->scrH();
    SLint w2 = sv->scrWdiv2();
    SLint h2 = sv->scrHdiv2();
    SLint h4 = h2 >> 1;
   
    if (_projection == stereoSideBySideD)
    {   SLint fbW2 = sv->oculusFB()->halfWidth();
        SLint fbH  = sv->oculusFB()->height();
        if (eye==leftEye) 
             _stateGL->viewport(   0, 0, fbW2, fbH);
        else _stateGL->viewport(fbW2, 0, fbW2, fbH);
    } else
    if (_projection == stereoSideBySide)
    {  
        if (eye==leftEye) 
             _stateGL->viewport( 0, 0, w2, h);
        else _stateGL->viewport(w2, 0, w2, h);
    } else
    if (_projection == stereoSideBySideP)
    {  
        if (eye==leftEye) 
             _stateGL->viewport( 0, h4, w2, h2);
        else _stateGL->viewport(w2, h4, w2, h2);
    } else  
        _stateGL->viewport(0, 0, w, h);
   

    ///////////////////
    // Clear Buffers //
    ///////////////////

    if (eye==rightEye) //&& _projection >= stereoColorRC) 
        // Do not clear color on right eye because it contains the color of the
        // left eye. The right eye must be drawn after the left into the same buffer
        _stateGL->clearDepthBuffer();
   
    //  Set Color Mask and Filter
    if (_projection >= stereoColorRC)
    {   if (eye==leftEye)
        {  switch (_projection) 
            {   case stereoColorRC: _stateGL->colorMask(1, 0, 0, 1); break;
                case stereoColorRB: _stateGL->colorMask(1, 0, 0, 1); break;
                case stereoColorRG: _stateGL->colorMask(1, 0, 0, 1); break;
                case stereoColorYB: _stateGL->colorMask(1, 1, 0, 1); break;
                default: break;
            }
        } else
        {   switch (_projection) 
            {   case stereoColorRC: _stateGL->colorMask(0, 1, 1, 1); break;
                case stereoColorRB: _stateGL->colorMask(0, 0, 1, 1); break;
                case stereoColorRG: _stateGL->colorMask(0, 1, 0, 1); break;
                case stereoColorYB: _stateGL->colorMask(0, 0, 1, 1); break;
                default: break;
            }
        }
      
        // Set color filter matrix for red-cyan and yello-blue (ColorCode3D)
        switch (_projection) 
        {   case stereoColorRC:
            _stateGL->stereoColorFilter.setMatrix(0.29f, 0.59f, 0.12f,
                                                  0.00f, 1.00f, 0.00f,
                                                  0.00f, 0.00f, 1.00f); break;
            case stereoColorYB:
            _stateGL->stereoColorFilter.setMatrix(1.00f, 0.00f, 0.00f,
                                                  0.00f, 1.00f, 0.00f,
                                                  0.15f, 0.15f, 0.70f); break;
            default: break;
        }
    }
    GET_GL_ERROR;
}
//-----------------------------------------------------------------------------
/*!
Applies the view transform to the modelview matrix depending on the eye:
eye=-1 for left, eye=1 for right
*/
void SLCamera::setView(SLSceneView* sv, const SLEye eye)
{  
    SLScene* s = SLScene::current;
   
    SLMat4f vm = updateAndGetWMI();

    if (eye == centerEye)
    {   _stateGL->modelViewMatrix.identity();
        _stateGL->viewMatrix.setMatrix(vm);
    } 
    else // stereo viewing
    {
        if (_projection == stereoSideBySideD)
        {  
            // half interpupilar disqtance
            //_eyeSeparation = s->oculus()->interpupillaryDistance(); update old rift code
            SLfloat halfIPD = (SLfloat)eye * _eyeSeparation * -0.5f;
            
            SLMat4f trackingPos;
            if (_useDeviceRot)
            {
                // get the oculus or mobile device orientation
                SLQuat4f rotation;
                if (s->oculus()->isConnected())
                {
                    rotation = s->oculus()->orientation(eye);
                    trackingPos.translate(-s->oculus()->position(eye));
                }
                else rotation = sv->deviceRotation();

                SLfloat rotX, rotY, rotZ;
                rotation.toMat4().toEulerAnglesZYX(rotZ, rotY, rotX);
                //SL_LOG("rotx : %3.1f, roty: %3.1f, rotz: %3.1f\n", rotX*SL_RAD2DEG, rotY*SL_RAD2DEG, rotZ*SL_RAD2DEG);
                SLVec3f viewAdjust = s->oculus()->viewAdjust(eye) * _unitScaling;
                SLMat4f vmEye(SLMat4f(viewAdjust.x, viewAdjust.y, viewAdjust.z) * rotation.inverted().toMat4() * trackingPos * vm);
                _stateGL->modelViewMatrix = vmEye;
                _stateGL->viewMatrix = vmEye;
            } 
            else
            {      
                SLMat4f vmEye(SLMat4f(halfIPD, 0.0f, 0.f) * vm);
                _stateGL->modelViewMatrix = vmEye;
                _stateGL->viewMatrix = vmEye;
            }
        }
        else
        {
            // Get central camera vectors eye, lookAt, lookUp out of the view matrix vm
            SLVec3f EYE, LA, LU, LR;
            vm.lookAt(&EYE, &LA, &LU, &LR);
   
            // Shorten LR to half of the eye dist (eye=-1 for left, eye=1 for right)
            LR *= _eyeSeparation * 0.5f * (SLfloat)eye;
      
            // Set the OpenGL view matrix for the left eye
            SLMat4f vmEye;
            vmEye.lookAt(EYE+LR, EYE + _focalDist*LA+LR, LU);
            _stateGL->modelViewMatrix = vmEye;
            _stateGL->viewMatrix = vmEye;
        }
    } 
}
//-----------------------------------------------------------------------------
//! SLCamera::animationStr() returns the animation enum as string
SLstring SLCamera::animationStr() const
{  
    switch (_camAnim)
    {   case turntableYUp: return "Turntable Y up";
        case turntableZUp: return "Turntable Z up";
        case walkingYUp:  return "Walking Y up";
        case walkingZUp:  return "Walking Z up";
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
                             const SLint x, const SLint y, const SLKey mod)
{  
    SLScene* s = SLScene::current;

    // Determine the lookAt point by ray cast
    eyeToPixelRay((SLfloat)(_scrW>>1), (SLfloat)(_scrH>>1), &_lookAtRay);
    //eyeToPixelRay(x, y, &_lookAtRay);
   
    if (s->root3D()) s->root3D()->hitRec(&_lookAtRay);
    
    // Init both position in case that the second finger came with delay
    _oldTouchPos1.set((SLfloat)x, (SLfloat)y);
    _oldTouchPos2.set((SLfloat)x, (SLfloat)y);
   
    return false;
}  
//-----------------------------------------------------------------------------
//! Gets called whenever the mouse is moved.
SLbool SLCamera::onMouseMove(const SLMouseButton button, 
                             const SLint x, const SLint y, const SLKey mod)
{  
    if (button == ButtonLeft) //================================================
    {   
        // new vars needed
        SLVec3f position = this->translation();
        SLVec3f forward =  this->forward();
        SLVec3f right =    this->right();
        SLVec3f up =       this->up();

        // The lookAt point
        SLVec3f laP = position + _focalDist * forward;

         
        // Determine rotation point as the center of the AABB of the hitNode
        SLVec3f rtP;
        if (_lookAtRay.length < FLT_MAX && _lookAtRay.hitNode)
             rtP = _lookAtRay.hitNode->aabb()->centerWS();
        else rtP = laP;
              
        // Determine rot angles around x- & y-axis
        SLfloat dY = (y-_oldTouchPos1.y) * _rotFactor;
        SLfloat dX = (x-_oldTouchPos1.x) * _rotFactor;

        if (_camAnim==turntableYUp) //.......................................
        {
            SLMat4f rot;
            rot.translate(rtP);
            rot.rotate(-dX, SLVec3f(0,1,0));
            rot.rotate(-dY, right);
            rot.translate(-rtP);
			
            _om.setMatrix(rot * _om);
            needWMUpdate();
        }
        else if (_camAnim==turntableZUp) //..................................
        {
            SLMat4f rot;
            rot.translate(rtP);
            rot.rotate(dX, SLVec3f(0,0,1));
            rot.rotate(dY, right);
            rot.translate(-rtP);
         
            _om.setMatrix(rot * _om);
            needWMUpdate();
        }
        else if (_camAnim==walkingYUp) //....................................
        {
            dY *= 0.5f; 
            dX *= 0.5f; 
            
            SLMat4f rot;
            rot.rotate(-dX, SLVec3f(0, 1, 0));
            rot.rotate(-dY, right);

            forward.set(rot.multVec(forward));
            lookAt(position + forward);
        }
        else if (_camAnim==walkingZUp) //....................................
        {
            dY *= 0.5f; 
            dX *= 0.5f; 
            
            SLMat4f rot;
            rot.rotate(-dX, SLVec3f(0, 0, 1));
            rot.rotate(-dY, right);

            forward.set(rot.multVec(forward));
            lookAt(position + forward, SLVec3f(0, 0, 1));
        }
        
        _oldTouchPos1.set((SLfloat)x,(SLfloat)y);
    } 
    else
    if (button == ButtonMiddle) //==============================================
    {   if (_camAnim==turntableYUp || _camAnim==turntableZUp)
        {  
            // Calculate the fraction delta of the mouse movement
            SLVec2f dMouse(x-_oldTouchPos1.x, _oldTouchPos1.y-y);
            dMouse.x /= (SLfloat)_scrW;
            dMouse.y /= (SLfloat)_scrH;
         
            // Scale the mouse delta by the lookAt distance
            SLfloat lookAtDist;
            if (_lookAtRay.length < FLT_MAX)
                lookAtDist = _lookAtRay.length;
            else lookAtDist = _focalDist;

            // scale factor depending on the space sice at focal dist
            SLfloat spaceH = tan(SL_DEG2RAD*_fov/2) * lookAtDist * 2.0f;
            SLfloat spaceW = spaceH * _aspect;

            dMouse.x *= spaceW;
            dMouse.y *= spaceH;
         
            if (mod==KeyCtrl)
            {
                translate(SLVec3f(-dMouse.x, 0, dMouse.y), TS_Object);

            } else
            {
                translate(SLVec3f(-dMouse.x, -dMouse.y, 0), TS_Object);
            }
            _oldTouchPos1.set((SLfloat)x,(SLfloat)y);
        }
    } //========================================================================
    return true;
}
//-----------------------------------------------------------------------------
//! Gets called whenever the mouse button is released
SLbool SLCamera::onMouseUp(const SLMouseButton button, 
                           const SLint x, const SLint y, const SLKey mod)
{
    // Stop any motion
    //_acceleration.set(0.0f, 0.0f, 0.0f);
   
    //SL_LOG("onMouseUp\n");
    if (button == ButtonLeft) //===============================================
    {   if (_camAnim==turntableYUp) //.........................................
        {  return true;
        } 
        else if (_camAnim==walkingYUp) //......................................
        {  return true;
        }
    } else if (button == ButtonMiddle) //======================================
    {   return true;
    }
    //=========================================================================
    return false;
}
//-----------------------------------------------------------------------------
/*! 
SLCamera::onMouseWheel event handler moves camera forwards or backwards
*/
SLbool SLCamera::onMouseWheel(const SLint delta, const SLKey mod)
{  
    SLScene* s = SLScene::current;
    SLfloat sign = (SLfloat)SL_sign(delta);
   
    if (_camAnim==turntableYUp || _camAnim==turntableZUp) //....................
    {   if (mod==KeyNone)
        {  
            // Determine the lookAt point by ray cast
            eyeToPixelRay((SLfloat)(_scrW>>1),
                        (SLfloat)(_scrH>>1), &_lookAtRay);

            if (s->root3D()) s->root3D()->hitRec(&_lookAtRay);

            if (_lookAtRay.length < FLT_MAX) 
                _lookAtRay.hitPoint = _lookAtRay.origin + 
                                      _lookAtRay.dir*_lookAtRay.length;
         
            // Scale the mouse delta by the lookAt distance
            SLfloat lookAtDist;
            if (_lookAtRay.length < FLT_MAX && _lookAtRay.hitNode)
                 lookAtDist = _lookAtRay.length;
            else lookAtDist = _focalDist;
                  
            translate(SLVec3f(0, 0, -sign*lookAtDist*_dPos), TS_Object);

            _lookAtRay.length = FLT_MAX;
        }
        if (mod==KeyCtrl)
        {   _eyeSeparation *= (1.0f + sign*0.1f);
        }
        if (mod==KeyAlt)
        {   _fov += sign*5.0f;
            currentFOV = _fov;
        }
        if (mod==KeyShift)
        {  _focalDist *= (1.0f + sign*0.05f);
        }
        return true;
    }
    else if (_camAnim==walkingYUp || _camAnim==walkingZUp) //...................
    {  
        _maxSpeed *= (1.0f + sign*0.1f);
    }
    return false;
}
//-----------------------------------------------------------------------------
/*! 
SLCamera::onDoubleTouch gets called whenever two fingers touch a handheld
screen.
*/
SLbool SLCamera::onTouch2Down(const SLint x1, const SLint y1,
                              const SLint x2, const SLint y2)
{
    SLScene* s = SLScene::current;
   
    // Determine the lookAt point by ray cast
    eyeToPixelRay((SLfloat)(_scrW>>1), 
                  (SLfloat)(_scrH>>1), &_lookAtRay);

    s->root3D()->hitRec(&_lookAtRay);
   
    _oldTouchPos1.set((SLfloat)x1, (SLfloat)y1);
    _oldTouchPos2.set((SLfloat)x2, (SLfloat)y2);
    return true;
}
//-----------------------------------------------------------------------------
/*! 
SLCamera::onTouch2Move gets called whenever two fingers move on a handheld 
screen.
*/
SLbool SLCamera::onTouch2Move(const SLint x1, const SLint y1,
                              const SLint x2, const SLint y2)
{
    SLScene* s = SLScene::current;

    SLVec2f now1((SLfloat)x1, (SLfloat)y1);
    SLVec2f now2((SLfloat)x2, (SLfloat)y2);
    SLVec2f delta1(now1-_oldTouchPos1);
    SLVec2f delta2(now2-_oldTouchPos2);
   
    // Average out the deltas over the last 4 events for correct 1 pixel moves
    static SLuint  cnt=0;
    static SLVec2f d1[4];
    static SLVec2f d2[4];
    d1[cnt%4] = delta1;
    d2[cnt%4] = delta2;
    SLVec2f avgDelta1(d1[0].x+d1[1].x+d1[2].x+d1[3].x, d1[0].y+d1[1].y+d1[2].y+d1[3].y);
    SLVec2f avgDelta2(d2[0].x+d2[1].x+d2[2].x+d2[3].x, d2[0].y+d2[1].y+d2[2].y+d2[3].y);
    avgDelta1 /= 4.0f;
    avgDelta2 /= 4.0f;
    cnt++;
      
    SLfloat r1, phi1, r2, phi2;
    avgDelta1.toPolar(r1, phi1);
    avgDelta2.toPolar(r2, phi2);
    
    // Scale the mouse delta by the lookAt distance
    SLfloat lookAtDist;
    if (_lookAtRay.length < FLT_MAX)
        lookAtDist = _lookAtRay.length;
    else lookAtDist = _focalDist;
         
    // scale factor depending on the space sice at focal dist
    SLfloat spaceH = tan(SL_DEG2RAD*_fov/2) * lookAtDist * 2.0f;
    SLfloat spaceW = spaceH * _aspect;
   
    //SL_LOG("avgDelta1: (%05.2f,%05.2f), dPhi=%05.2f\n", avgDelta1.x, avgDelta1.y, SL_abs(phi1-phi2));
   
    // if fingers move parallel slide camera vertically or horizontally
    if (SL_abs(phi1-phi2) < 0.2f)
    {  
        // Calculate center between finger points
        SLVec2f nowCenter((now1+now2)*0.5f);
        SLVec2f oldCenter((_oldTouchPos1+_oldTouchPos2)*0.5f);
      
        // For first move set oldCenter = nowCenter
        if (oldCenter == SLVec2f::ZERO) oldCenter = nowCenter;
      
        SLVec2f delta(nowCenter - oldCenter);

        // scale to 0-1
        delta.x /= _scrW;
        delta.y /= _scrH;

        // scale to space size
        delta.x *= spaceW;
        delta.y *= spaceH;
      
        if (_camAnim==turntableYUp || _camAnim==turntableZUp)
        {           
            // apply delta to x- and y-position
            translate(SLVec3f(-delta.x, delta.y, 0), TS_Object);
        } 
        else if (_camAnim == walkingYUp || _camAnim == walkingZUp)
        {
            //_moveDir.x = delta.x * 100.0f,
            //_moveDir.z = delta.y * 100.0f;
        }

    } else // Two finger pinch
    {  
        // Calculate vector between fingers
        SLVec2f nowDist(now2 - now1);
        SLVec2f oldDist(_oldTouchPos2-_oldTouchPos1);
      
        // For first move set oldDist = nowDist
        if (oldDist == SLVec2f::ZERO) oldDist = nowDist;
      
        SLfloat delta = oldDist.length() - nowDist.length();

        if (_camAnim==turntableYUp)
        {  // scale to 0-1
            delta /= (SLfloat)_scrH;

            // scale to space height
            delta *= spaceH*2;
         
            // apply delta to the z-position
            translate(SLVec3f(0, 0, delta), TS_Object);

        } 
        else if (_camAnim == walkingYUp)
        {  
            // change field of view
            _fov += SL_sign(delta) * 0.5f;
            currentFOV = _fov;
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
SLbool SLCamera::onTouch2Up(const SLint x1, const SLint y1,
                            const SLint x2, const SLint y2)
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
    switch ((SLchar)key)
    {   case 'W': _moveDir.z -= 1.0f; return true;
        case 'S': _moveDir.z += 1.0f; return true;
        case 'A': _moveDir.x -= 1.0f; return true;
        case 'D': _moveDir.x += 1.0f; return true;
        case 'Q': _moveDir.y += 1.0f; return true;
        case 'E': _moveDir.y -= 1.0f; return true;
        // @todo    I tried implementing 'sprint' on pressed down shift
        //          but modifier keys don't fire a normal key press event...
        //          fix that please. This is why speed control is on 1 and 2 for now
        case '1': _maxSpeed = 10.0f; return true;
        case '2': _maxSpeed = 20.0f; return true;

        case (SLchar)KeyDown: return onMouseWheel( 1, mod);
        case (SLchar)KeyUp:   return onMouseWheel(-1, mod);
        
        default:  return false;
    }
}
//-----------------------------------------------------------------------------
/*!
SLCamera::onKeyRelease gets called when a key is released
*/
SLbool SLCamera::onKeyRelease(const SLKey key, const SLKey mod)
{  
    switch ((SLchar)key)
    {   case 'W': _moveDir.z += 1.0f; return true;
        case 'S': _moveDir.z -= 1.0f; return true;
        case 'A': _moveDir.x += 1.0f; return true;
        case 'D': _moveDir.x -= 1.0f; return true;
        case 'Q': _moveDir.y -= 1.0f; return true;
        case 'E': _moveDir.y += 1.0f; return true;
    }

    return false;
}
//-----------------------------------------------------------------------------
//! SLCamera::setFrustumPlanes set the 6 plane from the view frustum.
/*! SLCamera::setFrustumPlanes set the 6 frustum planes by extracting the plane 
coefficients from the combined view and projection matrix.
See the paper from Gribb and Hartmann:
http://www2.ravensoft.com/users/ggribb/plane%20extraction.pdf
*/
void SLCamera::setFrustumPlanes()
{
    // build combined view projection matrix
    // SLCamera::setView should've been called before so viewMatrix contains the right value
    SLMat4f A(_stateGL->projectionMatrix*_stateGL->viewMatrix); 
	
    // set the A,B,C & D coeffitient for each plane
    _plane[T].setCoefficients(-A.m( 1) + A.m( 3),-A.m( 5) + A.m( 7),
				              -A.m( 9) + A.m(11),-A.m(13) + A.m(15));
    _plane[B].setCoefficients( A.m( 1) + A.m( 3), A.m( 5) + A.m( 7),
				               A.m( 9) + A.m(11), A.m(13) + A.m(15));
    _plane[L].setCoefficients( A.m( 0) + A.m( 3), A.m( 4) + A.m( 7),
				               A.m( 8) + A.m(11), A.m(12) + A.m(15));
    _plane[R].setCoefficients(-A.m( 0) + A.m( 3),-A.m( 4) + A.m( 7),
				              -A.m( 8) + A.m(11),-A.m(12) + A.m(15));
    _plane[N].setCoefficients( A.m( 2) + A.m( 3), A.m( 6) + A.m( 7),
				               A.m(10) + A.m(11), A.m(14) + A.m(15));
    _plane[F].setCoefficients(-A.m( 2) + A.m( 3),-A.m( 6) + A.m( 7),
				              -A.m(10) + A.m(11),-A.m(14) + A.m(15));
    _numRendered = 0;
}
//-----------------------------------------------------------------------------
//! eyeToPixelRay returns the a ray from the eye to the center of a pixel.
/*! This method is used for object picking. The calculation is the same as for
primary rays in Ray Tracing.
*/
void SLCamera::eyeToPixelRay(SLfloat x, SLfloat y, SLRay* ray)
{
    SLVec3f  EYE, LA, LU, LR;

    // get camera vectors eye, lookAt, lookUp from view matrix
    updateAndGetVM().lookAt(&EYE, &LA, &LU, &LR);

    if (_projection == monoOrthographic)
    {   /*
        In orthographic projection the top-left vector (TL) points
        from the eye to the center of the TL-left pixel of a plane that
        parallel to the projection plan at zero distance from the eye.
        */
        SLVec3f pos(updateAndGetVM().translation());
        SLfloat hh = tan(SL_DEG2RAD*_fov*0.5f) * pos.length();
        SLfloat hw = hh * _aspect;

        // calculate the size of a pixel in world coords.
        SLfloat pixel = hw * 2 / _scrW;

        SLVec3f TL = EYE - hw*LR + hh*LU  +  pixel/2*LR - pixel/2*LU;
        SLVec3f dir = LA;
        dir.normalize();
        ray->setDir(dir);
        ray->origin.set(TL + pixel*(x*LR - y*LU));
    }
    else
    {   /*
        In perspective projection the top-left vector (TL) points
        from the eye to the center of the top-left pixel on a projection
        plan in focal distance. See also the computergraphics script about
        primary ray calculation.
        */
        // calculate half window width & height in world coords
        SLfloat hh = tan(SL_DEG2RAD*_fov*0.5f) * _focalDist;
        SLfloat hw = hh * _aspect;

        // calculate the size of a pixel in world coords.
        SLfloat pixel = hw * 2 / _scrW;

        // calculate a vector to the center (C) of the top left (TL) pixel
        SLVec3f C  = LA * _focalDist;
        SLVec3f TL = C - hw*LR + hh*LU  +  pixel*0.5f*(LR - LU);
        SLVec3f dir = TL + pixel*(x*LR - y*LU);

        dir.normalize();
        ray->setDir(dir);
        ray->origin.set(EYE);
    }

    ray->length = FLT_MAX;
    ray->depth = 1;
    ray->contrib = 1.0f; 
    ray->type = PRIMARY;
    ray->x = x;
    ray->y = y;  
    ray->hitTriangle = -1;
    ray->hitMat = 0;
    ray->hitNormal.set(SLVec3f::ZERO);
    ray->hitPoint.set(SLVec3f::ZERO); 
    ray->originMat = 0;
    ray->originTria = 0;
}
//-----------------------------------------------------------------------------
//! SLCamera::isInFrustum does a simple and fast frustum culling test for AABBs
/*! SLCamera::isInFrustum checks if the bounding sphere of an AABB is within 
the view frustum defined by its 6 planes by simply testing the distance of the
AABBs center minus its radius. This is faster than the AABB in frustum test but
not as precise. Please refer to the nice tutorial on frustum culling on:
http://www.lighthouse3d.com/opengl/viewfrustum/
*/
SLbool SLCamera::isInFrustum(SLAABBox* aabb) 
{	
    // check the 6 planes of the frustum
    for(SLint i=0; i < 6; ++i) 
    {	SLfloat distance = _plane[i].distToPoint(aabb->centerWS());
	    if (distance < -aabb->radiusWS()) 
	    {   aabb->isVisible(false);
		    return false;
	    }
    }
    aabb->isVisible(true);
    _numRendered++;
	
    // Calculate squared dist. from AABB's center to viewer for blend sorting.
    SLVec3f viewToCenter(_wm.translation()-aabb->centerWS());
    aabb->sqrViewDist(viewToCenter.lengthSqr());    	   
    return true;
}
//-----------------------------------------------------------------------------
//! SLCamera::to_string returns important camera parameter as a string
SLstring SLCamera::toString() const
{
    SLMat4f vm = updateAndGetVM();
    std::ostringstream ss;
    ss << "Projection: " << projectionStr() << endl;
    ss << "FOV: " << _fov << endl;
    ss << "ClipNear: " << _clipNear << endl;
    ss << "ClipFar: " << _clipFar << endl;
    ss << "Animation: " << animationStr()  << endl;
    ss << vm.toString() << endl;
    return ss.str();
}
//-----------------------------------------------------------------------------

