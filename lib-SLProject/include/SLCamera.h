//#############################################################################
//  File:      SLCamera.h
//  Author:    Marc Wacker, Marcus Hudritsch
//  Date:      July 2014
//  Codestyle: https://github.com/cpvrlab/SLProject/wiki/SLProject-Coding-Style
//  Copyright: Marcus Hudritsch
//             This software is provide under the GNU General Public License
//             Please visit: http://opensource.org/licenses/GPL-3.0
//#############################################################################

#ifndef SLCAMERA_H
#define SLCAMERA_H

#include <SLBackground.h>
#include <SLEnums.h>
#include <SLGLState.h>
#include <SLGLVertexArrayExt.h>
#include <SLNode.h>
#include <SLPlane.h>
#include <SLRay.h>
#include <SLSamples2D.h>
#include <SLRect.h>

class SLSceneView;
class SLDeviceRotation;
class SLDeviceLocation;

class SLCameraAnimation;

//-----------------------------------------------------------------------------
//! Active or visible camera node class
/*! An instance of this SLNode derived class serves as an active camera with
all its view and projection parameters or if inactive as a visible scene graph
node with camera body and its view frustum. The position and orientation
of the active camera is set in the setView method by loading the view matrix _vm
into the OpenGL modelview matrix. The view matrix _vm is simply the inverse of
the shapes world matrix _wm. Every SLSceneView instance has a pointer _camera
to its active camera.\n
Because the SLNode class is inherited from the abstract SLEventHandler class a
camera can handle mouse & keyboard event. All camera animations are handled in
these event handlers.\n
The following camera animations are available:\n
   - SLCamAnim::CA_turntableYUp
   - SLCamAnim::CA_turntableZUp
   - SLCamAnim::CA_trackball
   - SLCamAnim::CA_walkingYUp
   - SLCamAnim::CA_walkingZUp
   - SLCamAnim::CA_deviceRotYUp
\n
Every camera has an instance of SLBackground that defines the views background
if the camera is the active one. If the camera is inactive the background gets
drawn on the far clipping plane of the visualized view frustum.
*/
class SLCamera : public SLNode
{
public:
    explicit SLCamera(const SLstring& name = "Camera");
    ~SLCamera() override;

    void           statsRec(SLNodeStats& stats) override;
    void           drawMesh(SLSceneView* sv) override;
    virtual SLbool camUpdate(SLfloat timeMS);
    void           preShade(SLRay* ray) { (void)ray; }
    void           calcMinMax(SLVec3f& minV, SLVec3f& maxV) const;
    void           buildAABB(SLAABBox& aabb, const SLMat4f& wmNode);
    SLVec2i        frustumSizeAtDistance(SLfloat distance);

    // Event handlers for camera animation
    SLbool onMouseDown(SLMouseButton button, SLint x, SLint y, SLKey mod) override;
    SLbool onMouseMove(SLMouseButton button, SLint x, SLint y, SLKey mod) override;
    SLbool onMouseUp(SLMouseButton button, SLint x, SLint y, SLKey mod) override;
    SLbool onMouseWheel(SLint delta, SLKey mod) override;
    SLbool onTouch2Down(SLint x1, SLint y1, SLint x2, SLint y2) override;
    SLbool onTouch2Move(SLint x1, SLint y1, SLint x2, SLint y2) override;
    SLbool onTouch2Up(SLint x1, SLint y1, SLint x2, SLint y2) override;
    SLbool onKeyPress(SLKey key, SLKey mod) override;
    SLbool onKeyRelease(SLKey key, SLKey mod) override;

    void    eyeToPixelRay(SLfloat x, SLfloat y, SLRay* ray);
    void    UVWFrame(SLVec3f& EYE, SLVec3f& U, SLVec3f& V, SLVec3f& W);
    SLVec2f projectWorldToNDC(const SLVec4f& worldPos) const;
    SLVec3f trackballVec(SLint x, SLint y) const;
    SLbool  isInFrustum(SLAABBox* aabb);
    void    passToUniforms(SLGLProgram* program);

    // Apply projection, viewport and view transformations
    void setViewport(SLSceneView* sv, SLEyeType eye);
    void setProjection(SLSceneView* sv, SLEyeType eye);
    void setView(SLSceneView* sv, SLEyeType eye);
    void setFrustumPlanes();

    // Setters
    void projection(SLProjection p)
    {
        _projection       = p;
        currentProjection = p;
    }
    //! vertical field of view
    void fov(const SLfloat fov)
    {
        _fovV      = fov;
        currentFOV = fov;
    }
    void camAnim(SLCamAnim ca)
    {
        _camAnim         = ca;
        currentAnimation = ca;
    }
    /*
    void intrinsics(const SLfloat fx, const SLfloat fy, const SLfloat cx, const SLfloat cy)
    {
        _fx = fx;
        _fy = fy;
        _cx = cx;
        _cy = cy;
    }
     */
    void clipNear(const SLfloat cNear) { _clipNear = cNear; }
    void clipFar(const SLfloat cFar) { _clipFar = cFar; }
    void lookFrom(const SLVec3f& fromDir,
                  const SLVec3f& upDir = SLVec3f::AXISY);
    void maxSpeed(const SLfloat ms) { _maxSpeed = ms; }
    void moveAccel(const SLfloat accel) { _moveAccel = accel; }
    void brakeAccel(const SLfloat accel) { _brakeAccel = accel; }
    void drag(const SLfloat drag) { _drag = drag; }
    void focalDist(const SLfloat f) { _focalDist = f; }
    void lensDiameter(const SLfloat d) { _lensDiameter = d; }
    void lensSamples(SLuint x, SLuint y) { _lensSamples.samples(x, y); }
    void stereoEyeSeparation(const SLfloat es) { _stereoEyeSeparation = es; }
    void devRotLoc(SLDeviceRotation* devRot, SLDeviceLocation* devLoc)
    {
        _devRot = devRot;
        _devLoc = devLoc;
    }
    void fogIsOn(const bool isOn) { _fogIsOn = isOn; }
    void fogMode(const int mode) { _fogMode = mode; }
    void fogDensity(const float density) { _fogDensity = density; }

    // Getters
    const SLMat4f& updateAndGetVM() const { return updateAndGetWMI(); }
    SLProjection   projection() const { return _projection; }
    SLstring       projectionStr() const { return projectionToStr(_projection); }
    SLfloat        unitScaling() const { return _unitScaling; }
    SLfloat        fovV() const { return _fovV; }                  //!< Vertical field of view
    //todo: fovH calculation is wrong
    SLfloat        fovH() const { return _viewportRatio * _fovV; } //!< Horizontal field of view
    SLint          viewportW() const { return _viewportW; }
    SLint          viewportH() const { return _viewportH; }
    SLfloat        aspect() const { return _viewportRatio; }
    SLfloat        clipNear() const { return _clipNear; }
    SLfloat        clipFar() const { return _clipFar; }
    SLCamAnim      camAnim() const { return _camAnim; }
    SLstring       animationStr() const;

    SLfloat stereoEyeSeparation() const { return _stereoEyeSeparation; }
    SLint   stereoEye() const { return _stereoEye; }
    SLMat3f stereoColorFilter() const { return _stereoColorFilter; }

    SLfloat      lensDiameter() const { return _lensDiameter; }
    SLSamples2D* lensSamples() { return &_lensSamples; }
    SLfloat      focalDist() const { return _focalDist; }
    SLfloat      focalDistScrW() const;
    SLfloat      focalDistScrH() const;
    SLVec3f      focalPointWS() const { return translationWS() + _focalDist * forwardWS(); }
    SLVec3f      focalPointOS() const { return translationOS() + _focalDist * forwardOS(); }

    SLbool  fogIsOn() const { return _fogIsOn; }
    SLint   fogMode() const { return _fogMode; }
    SLfloat fogDensity() const { return _fogDensity; }
    SLfloat fogDistStart() const { return _fogStart; }
    SLfloat fogDistEnd() const { return _fogEnd; }
    SLCol4f fogColor() const { return _fogColor; }

    SLfloat       trackballSize() const { return _trackballSize; }
    SLBackground& background() { return _background; }
    SLfloat       maxSpeed() const { return _maxSpeed; }
    SLfloat       moveAccel() const { return _moveAccel; }
    SLfloat       brakeAccel() const { return _brakeAccel; }
    SLfloat       drag() const { return _drag; }
    SLstring      toString() const;
    SLRectf&      selectRect() { return _selectRect; }
    SLRectf&      deselectRect() { return _deselectRect; }
    
    //update rotation matrix _enucorrRenu
    void updateEnucorrRenu(SLSceneView* sv, const SLMat3f& enuRc, float& f, SLVec3f& enuOffsetPix);

    // Static global default parameters for new cameras
    static SLCamAnim    currentAnimation;
    static SLProjection currentProjection;
    static SLfloat      currentFOV;
    static SLint        currentDevRotation;
    static SLstring     projectionToStr(SLProjection p);

protected:
    // projection parameters
    SLProjection _projection;    //!< projection type
    SLfloat      _fovV;          //!< Current vertical field of view (view angle) in degrees
    SLfloat      _fovInit;       //!< Initial vertical field of view (view angle) in degrees
    SLfloat      _clipNear;      //!< Dist. to the near clipping plane
    SLfloat      _clipFar;       //!< Dist. to the far clipping plane
    SLPlane      _plane[6];      //!< 6 frustum planes (t, b, l, r, n, f)
    SLint        _viewportW;     //!< screen width in screen coordinates (the framebuffer may be bigger)
    SLint        _viewportH;     //!< screen height in screen coordinates (the framebuffer may be bigger)
    SLfloat      _viewportRatio; //!< _scrW /_srcH = screen ratio
    SLfloat      _fx;            //!< horizontal focal length
    SLfloat      _fy;            //!< vertical focal length
    SLfloat      _cx;            //!< sensor center in x direction
    SLfloat      _cy;            //!< sensor center in y direction
    SLRecti      _fbRect;        //!< framebuffer rectangle (this could be different than the viewport on high dpi displays such as on MacOS)

    enum
    {
        T = 0,
        B,
        L,
        R,
        N,
        F
    };                        //!< enumeration for frustum planes
    SLBackground _background; //!< Colors or texture displayed in the background

    SLGLVertexArrayExt _vao; //!< OpenGL Vertex array for rendering

    // animation parameters (SLCamera is owner and calls delete if valid)
    SLCameraAnimation* _camAnimation = nullptr;
    
    SLbool    _movedLastFrame;    //! did the camera updateRec in the last frame?
    SLCamAnim _camAnim;           //!< Type of camera animation
    SLVec2f   _oldTouchPos1;      //!< Old mouse/touch position in pixels
    SLVec2f   _oldTouchPos2;      //!< Old 2nd finger touch position in pixels
    SLVec3f   _trackballStartVec; //!< Trackball vector at mouse down
    SLfloat   _trackballSize;     //!< Size of trackball (0.8 = 80% of window size)

    SLVec3f _moveDir;      //!< accumulated movement directions based on pressed buttons
    SLfloat _drag;         //!< simple constant drag that affects velocity
    SLfloat _maxSpeed;     //!< maximum speed in m/s, with high drag values this speed might not be achievable at all
    SLVec3f _velocity;     //!< current velocity vector
    SLVec3f _acceleration; //!< current acceleration vector
    SLfloat _brakeAccel;   //!< brake acceleration
    SLfloat _moveAccel;    //!< move acceleration

    // ray tracing parameters
    SLfloat     _focalDist;    //!< distance to lookAt point on the focal plane from lens
    SLfloat     _lensDiameter; //!< Lens diameter
    SLSamples2D _lensSamples;  //!< sample points for lens sampling (DOF)

    // Stereo rendering
    SLfloat _stereoEyeSeparation; //!< eye separation for stereo mode
    SLfloat _unitScaling;         //!< indicate what the current unit scale is
    SLint   _stereoEye;           //!< -1=left, 0=center, 1=right
    SLMat3f _stereoColorFilter;   //!< color filter matrix for anaglyphling is to adjust movement and stereo rendering correctly

    // fog
    SLbool  _fogIsOn;        //!< Flag if fog blending is enabled
    SLint   _fogMode;        //!< 0=LINEAR, 1=EXP, 2=EXP2
    SLfloat _fogDensity;     //!< Fog density for exponential modes
    SLfloat _fogStart;       //!< Fog start distance for linear mode
    SLfloat _fogEnd;         //!< Fog end distance for linear mode
    SLCol4f _fogColor;       //!< fog color blended to the final color
    SLbool  _fogColorIsBack; //!< fog color blended to the final color

    SLDeviceRotation* _devRot = nullptr;
    SLDeviceLocation* _devLoc = nullptr;

    SLRectf _selectRect;   //!< Mouse selection rectangle. See SLMesh::handleRectangleSelection
    SLRectf _deselectRect; //!< Mouse deselection rectangle. See SLMesh::handleRectangleSelection
    
    //!parameter for manual finger rotation and translation
    SLint _xOffsetPix = 0;
    SLint _yOffsetPix = 0;
    float _distanceToObjectM = 1.0f; //!< distance to object in meter that should be shifted relative to camera
    float _enucorrTRenu = 0.f;        //!< manual camera shift in y direction
    SLMat3f _enucorrRenu;
};
//-----------------------------------------------------------------------------

class SLCameraAnimation
{
public:
    SLCameraAnimation(const SLstring& name, const SLstring& description)
     : _name(name),
       _description(description)
    {
        _rotFactor = 0.5f;
        _dPos      = 0.1f;
    }
    virtual ~SLCameraAnimation() {}
    
    virtual SLbool onMouseDown(const SLint x, const SLint y) { return false; }
    virtual SLbool onMouseUpDispatched() { return false; }
    virtual void onMouseMove(const SLMouseButton button, const SLKey mod, SLfloat x, SLfloat y, SLfloat xOld, SLfloat yOld) {}
    virtual SLbool onMouseWheel(const SLint delta, const SLKey mod) { return false; }
    virtual void onTouch2Move(const SLint x1,
                              const SLint y1,
                              const SLint x2,
                              const SLint y2,
                              const SLVec2f& oldTouchPos1,
                              const SLVec2f& oldTouchPos2) {}
  
    //setters:
    //set the camera that will be manipulated (is done by SLCamera when set)
    void camera(SLCamera* camera) { _camera = camera; }
    void rotFactor(SLfloat rf) { _rotFactor = rf; }
    void dPos(SLfloat dp) { _dPos = dp; }
    
    //getters:
    SLstring description() { return _description; }
    SLstring name() { return _name; }
    SLfloat rotFactor() { return _rotFactor; }
    SLfloat dPos() { return _dPos; }
    
protected:
    //! camera node that will be manipulated
    SLCamera* _camera;
    
    SLfloat _rotFactor; //!< Mouse rotation sensibility
    SLfloat _dPos;      //!< Delta dist. for keyb. transl.
    //SLfloat _dRot = 15.f;      //!< Delta angle for keyb. rot.
    
private:
    //! animation type name and description
    SLstring _name;
    SLstring _description;
};

class SLCAOff : public SLCameraAnimation
{
public:
    SLCAOff()
    : SLCameraAnimation("CA_off", "Animation Disabled")
    {}
};

class SLCATurntableYUp : public SLCameraAnimation
{
public:
    SLCATurntableYUp()
    : SLCameraAnimation("CA_turntableYUp", "Turntable Y up")
    {}
    
    void onMouseMove(const SLMouseButton button, const SLKey mod, SLfloat x, SLfloat y, SLfloat xOld, SLfloat yOld) override
    {
        assert(_camera);
        
        if (button == MB_left) //==================================================
        {
            // Position and directions in view space
            SLVec3f positionVS = _camera->translationOS();
            SLVec3f forwardVS  = _camera->forwardOS();
            SLVec3f rightVS    = _camera->rightOS();
            
            // The lookAt point
            SLVec3f lookAtPoint = positionVS + _camera->focalDist() * forwardVS;

            // Determine rot angles around x- & y-axis
            SLfloat dY = (y - yOld) * _rotFactor;
            SLfloat dX = (x - xOld) * _rotFactor;
            
            SLMat4f rot;
            rot.translate(lookAtPoint);
            rot.rotate(-dX, SLVec3f(0, 1, 0));
            rot.rotate(-dY, rightVS);
            rot.translate(-lookAtPoint);

            _camera->om(rot * _camera->om());
        }
        else if (button == MB_middle) //===========================================
        {
            // Calculate the fraction delta of the mouse movement
            SLVec2f dMouse(x - xOld, yOld - y);
            dMouse.x /= (SLfloat)_camera->viewportW();
            dMouse.y /= (SLfloat)_camera->viewportH();

            // scale factor depending on the space size at focal dist
            SLfloat spaceH = tan(Utils::DEG2RAD * _camera->fovV() / 2) * _camera->focalDist() * 2.0f;
            SLfloat spaceW = spaceH * _camera->aspect();

            dMouse.x *= spaceW;
            dMouse.y *= spaceH;

            if (mod == K_ctrl)
                _camera->translate(SLVec3f(-dMouse.x, 0, dMouse.y), TS_object);
            else
                _camera->translate(SLVec3f(-dMouse.x, -dMouse.y, 0), TS_object);
        }
    
    }
    
    SLbool onMouseUpDispatched() override
    {
        return true;
    }
    
    SLbool onMouseWheel(const SLint delta, const SLKey mod) override
    {
        SLfloat sign = (SLfloat)Utils::sign(delta);
        
        if (mod == K_none)
        {
            const SLfloat& focalDist = _camera->focalDist();
            SLfloat update = -sign * focalDist * _dPos;
            _camera->translate(SLVec3f(0, 0, update), TS_object);
            _camera->focalDist(focalDist + update);
        }
        if (mod == K_ctrl)
        {
            _camera->stereoEyeSeparation( _camera->stereoEyeSeparation() * (1.0f + sign * 0.1f));
        }
        if (mod == K_alt)
        {
            _camera->fov(_camera->fovV() + sign * 5.0f);
            _camera->currentFOV = _camera->fovV();
        }
        //todo anim: why always false?
        return false;
    }
    
    void onTouch2Move(const SLint x1,
                      const SLint y1,
                      const SLint x2,
                      const SLint y2,
                      const SLVec2f& oldTouchPos1,
                      const SLVec2f& oldTouchPos2) override
    {
        SLVec2f now1((SLfloat)x1, (SLfloat)y1);
        SLVec2f now2((SLfloat)x2, (SLfloat)y2);
        SLVec2f delta1(now1 - oldTouchPos1);
        SLVec2f delta2(now2 - oldTouchPos2);

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
        SLfloat spaceH = tan(Utils::DEG2RAD * _camera->fovV() / 2) * _camera->focalDist() * 2.0f;
        SLfloat spaceW = spaceH * _camera->aspect();

        // if fingers move parallel slide camera vertically or horizontally
        if (Utils::abs(phi1 - phi2) < 0.2f)
        {
            // Calculate center between finger points
            SLVec2f nowCenter((now1 + now2) * 0.5f);
            SLVec2f oldCenter((oldTouchPos1 + oldTouchPos2) * 0.5f);

            // For first move set oldCenter = nowCenter
            if (oldCenter == SLVec2f::ZERO) oldCenter = nowCenter;

            SLVec2f delta(nowCenter - oldCenter);

            // scale to 0-1
            delta.x /= _camera->viewportW();
            delta.y /= _camera->viewportH();

            // scale to space size
            delta.x *= spaceW;
            delta.y *= spaceH;

            // apply delta to x- and y-position
            _camera->translate(SLVec3f(-delta.x, delta.y, 0), TS_object);
        }
        else // Two finger pinch
        {
            // Calculate vector between fingers
            SLVec2f nowDist(now2 - now1);
            SLVec2f oldDist(oldTouchPos2 - oldTouchPos1);

            // For first move set oldDist = nowDist
            if (oldDist == SLVec2f::ZERO) oldDist = nowDist;

            SLfloat delta = oldDist.length() - nowDist.length();

            // scale to 0-1
            delta /= (SLfloat)_camera->viewportH();

            // scale to space height
            delta *= spaceH * 2;

            // apply delta to the z-position
            _camera->translate(SLVec3f(0, 0, delta), TS_object);
            _camera->focalDist(_camera->focalDist() + delta);
        }
    }
};

//the functionality is not as complete as for y-up (no two finger scale)
class SLCATurntableZUp : public SLCameraAnimation
{
public:
    SLCATurntableZUp()
    : SLCameraAnimation("CA_turntableZUp", "Turntable Z up")
    {}
    
    void onMouseMove(const SLMouseButton button, const SLKey mod, SLfloat x, SLfloat y, SLfloat xOld, SLfloat yOld) override
    {
        assert(_camera);
        
        if (button == MB_left) //==================================================
        {
            // Position and directions in view space
            SLVec3f positionVS = _camera->translationOS();
            SLVec3f forwardVS  = _camera->forwardOS();
            SLVec3f rightVS    = _camera->rightOS();
            
            // The lookAt point
            SLVec3f lookAtPoint = positionVS + _camera->focalDist() * forwardVS;

            // Determine rot angles around x- & y-axis
            SLfloat dY = (y - yOld) * _rotFactor;
            SLfloat dX = (x - xOld) * _rotFactor;
            
            SLMat4f rot;
            rot.translate(lookAtPoint);
            rot.rotate(dX, SLVec3f(0, 0, 1));
            rot.rotate(dY, rightVS);
            rot.translate(-lookAtPoint);

            _camera->om(rot * _camera->om());
        }
        else if (button == MB_middle) //===========================================
        {
            // Calculate the fraction delta of the mouse movement
            SLVec2f dMouse(x - xOld, yOld - y);
            dMouse.x /= (SLfloat)_camera->viewportW();
            dMouse.y /= (SLfloat)_camera->viewportH();

            // scale factor depending on the space size at focal dist
            SLfloat spaceH = tan(Utils::DEG2RAD * _camera->fovV() / 2) * _camera->focalDist() * 2.0f;
            SLfloat spaceW = spaceH * _camera->aspect();

            dMouse.x *= spaceW;
            dMouse.y *= spaceH;

            if (mod == K_ctrl)
                _camera->translate(SLVec3f(-dMouse.x, 0, dMouse.y), TS_object);
            else
                _camera->translate(SLVec3f(-dMouse.x, -dMouse.y, 0), TS_object);
        }
    }
    
    SLbool onMouseUpDispatched() override
    {
        return true;
    }
    
    SLbool onMouseWheel(const SLint delta, const SLKey mod) override
    {
        SLfloat sign = (SLfloat)Utils::sign(delta);
        
        if (mod == K_none)
        {
            const SLfloat& focalDist = _camera->focalDist();
            SLfloat update = -sign * focalDist * _dPos;
            _camera->translate(SLVec3f(0, 0, update), TS_object);
            _camera->focalDist(focalDist + update);
        }
        if (mod == K_ctrl)
        {
            _camera->stereoEyeSeparation( _camera->stereoEyeSeparation() * (1.0f + sign * 0.1f));
        }
        if (mod == K_alt)
        {
            _camera->fov(_camera->fovV() + sign * 5.0f);
            _camera->currentFOV = _camera->fovV();
        }
        
        return false;
    }
    
    void onTouch2Move(const SLint x1,
                      const SLint y1,
                      const SLint x2,
                      const SLint y2,
                      const SLVec2f& oldTouchPos1,
                      const SLVec2f& oldTouchPos2) override
    {
        SLVec2f now1((SLfloat)x1, (SLfloat)y1);
        SLVec2f now2((SLfloat)x2, (SLfloat)y2);
        SLVec2f delta1(now1 - oldTouchPos1);
        SLVec2f delta2(now2 - oldTouchPos2);

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
        SLfloat spaceH = tan(Utils::DEG2RAD * _camera->fovV() / 2) * _camera->focalDist() * 2.0f;
        SLfloat spaceW = spaceH * _camera->aspect();

        // if fingers move parallel slide camera vertically or horizontally
        if (Utils::abs(phi1 - phi2) < 0.2f)
        {
            // Calculate center between finger points
            SLVec2f nowCenter((now1 + now2) * 0.5f);
            SLVec2f oldCenter((oldTouchPos1 + oldTouchPos2) * 0.5f);

            // For first move set oldCenter = nowCenter
            if (oldCenter == SLVec2f::ZERO) oldCenter = nowCenter;

            SLVec2f delta(nowCenter - oldCenter);

            // scale to 0-1
            delta.x /= _camera->viewportW();
            delta.y /= _camera->viewportH();

            // scale to space size
            delta.x *= spaceW;
            delta.y *= spaceH;

            // apply delta to x- and y-position
            _camera->translate(SLVec3f(-delta.x, delta.y, 0), TS_object);
        }
        else // Two finger pinch
        {
            // Calculate vector between fingers
            SLVec2f nowDist(now2 - now1);
            SLVec2f oldDist(oldTouchPos2 - oldTouchPos1);

            // For first move set oldDist = nowDist
            if (oldDist == SLVec2f::ZERO) oldDist = nowDist;

            SLfloat delta = oldDist.length() - nowDist.length();

            // scale to 0-1
            delta /= (SLfloat)_camera->viewportH();

            // scale to space height
            delta *= spaceH * 2;

            // apply delta to the z-position
            _camera->translate(SLVec3f(0, 0, delta), TS_object);
            _camera->focalDist(_camera->focalDist() + delta);
        }
    }
};

class SLCATrackball : public SLCameraAnimation
{
public:
    SLCATrackball()
     : SLCameraAnimation("CA_trackball", "Trackball"),
       _trackballSize(0.8f)
    {}
    
    SLbool onMouseDown(const SLint x, const SLint y) override
    {
        _trackballStartVec = trackballVec(x, y);
        return true;
    }
    
    void onMouseMove(const SLMouseButton button, const SLKey mod, SLfloat x, SLfloat y, SLfloat xOld, SLfloat yOld) override
    {
        assert(_camera);
        
        if (button == MB_left) //==================================================
        {
            // Position and directions in view space
            SLVec3f positionVS = _camera->translationOS();
            SLVec3f forwardVS  = _camera->forwardOS();
            SLVec3f rightVS    = _camera->rightOS();
            
            // The lookAt point
            SLVec3f lookAtPoint = positionVS + _camera->focalDist() * forwardVS;

            // Determine rot angles around x- & y-axis
            SLfloat dY = (y - yOld) * _rotFactor;
            SLfloat dX = (x - xOld) * _rotFactor;
            
            // Reference: https://en.wikibooks.org/wiki/OpenGL_Programming/Modern_OpenGL_Tutorial_Arcball
            // calculate current mouse vector at currenct mouse position
            SLVec3f curMouseVec = trackballVec(x, y);

            // calculate angle between the old and the current mouse vector
            // Take care that the dot product isn't greater than 1.0 otherwise
            // the acos will return indefined.
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
            SLVec2f dMouse(xOld - x, yOld - y);
            SLfloat dMouseLenght = dMouse.length();
            if (angle > dMouseLenght) angle = dMouseLenght * 0.2f;

            // Transform rotation axis into world space
            // Remember: The cameras om is the view matrix inversed
            SLVec3f axisWS = _camera->om().mat3() * axisVS;

            // Create rotation from one rotation around one axis
            SLMat4f rot;
            rot.translate(lookAtPoint);          // undo camera translation
            rot.rotate((SLfloat)-angle, axisWS); // create incremental rotation
            rot.translate(-lookAtPoint);         // redo camera translation
            _camera->om(rot * _camera->om());    // accumulate rotation to the existing camera matrix

            // set current to last
            _trackballStartVec = curMouseVec;
            lastAxisVS         = axisVS;
        }
        else if (button == MB_middle) //===========================================
        {
            // Calculate the fraction delta of the mouse movement
            SLVec2f dMouse(x - xOld, yOld - y);
            dMouse.x /= (SLfloat)_camera->viewportW();
            dMouse.y /= (SLfloat)_camera->viewportH();

            // scale factor depending on the space size at focal dist
            SLfloat spaceH = tan(Utils::DEG2RAD * _camera->fovV() / 2) * _camera->focalDist() * 2.0f;
            SLfloat spaceW = spaceH * _camera->aspect();

            dMouse.x *= spaceW;
            dMouse.y *= spaceH;

            if (mod == K_ctrl)
                _camera->translate(SLVec3f(-dMouse.x, 0, dMouse.y), TS_object);
            else
                _camera->translate(SLVec3f(-dMouse.x, -dMouse.y, 0), TS_object);
        }
    }
    
    SLbool onMouseWheel(const SLint delta, const SLKey mod) override
    {
        SLfloat sign = (SLfloat)Utils::sign(delta);
        
        if (mod == K_none)
        {
            const SLfloat& focalDist = _camera->focalDist();
            SLfloat update = -sign * focalDist * _dPos;
            _camera->translate(SLVec3f(0, 0, update), TS_object);
            _camera->focalDist(focalDist + update);
        }
        if (mod == K_ctrl)
        {
            _camera->stereoEyeSeparation( _camera->stereoEyeSeparation() * (1.0f + sign * 0.1f));
        }
        if (mod == K_alt)
        {
            _camera->fov(_camera->fovV() + sign * 5.0f);
            _camera->currentFOV = _camera->fovV();
        }
        return false;
    }
    
private:
    //-----------------------------------------------------------------------------
    //! Returns a vector from the window center to a virtual trackball at [x,y].
    /*! The trackball vector is a vector from the window center to a hemisphere
    over the window at the specified cursor position. With two trackball vectors
    you can calculate a single rotation axis with the cross product. This routine
    is used for the trackball camera animation. */
    SLVec3f trackballVec(const SLint x, const SLint y) const
    {
        //Calculate x & y component to the virtual unit sphere
        SLfloat r = (SLfloat)(_camera->viewportW() < _camera->viewportH() ?
                              _camera->viewportW() / 2 :
                              _camera->viewportH() / 2) * _trackballSize;

        SLVec3f vec((SLfloat)(x - _camera->viewportW() * 0.5f) / r,
                    -(SLfloat)(y - _camera->viewportH() * 0.5f) / r);

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
    
    SLVec3f   _trackballStartVec; //!< Trackball vector at mouse down
    SLfloat   _trackballSize;     //!< Size of trackball (0.8 = 80% of window size)
};

class SLCAWalkingYUp : public SLCameraAnimation
{
public:
    SLCAWalkingYUp()
    : SLCameraAnimation("CA_walkingYUp", "Walking Y up")
    {}
};

class SLCAWalkingZUp : public SLCameraAnimation
{
public:
    SLCAWalkingZUp()
    : SLCameraAnimation("CA_walkingZUp", "Walking Z up")
    {}
};

class SLCADeviceRotYUp : public SLCameraAnimation
{
public:
    SLCADeviceRotYUp()
    : SLCameraAnimation("CA_deviceRotYUp", "Device Rotated Y up")
    {}
};

class SLCADeviceRotLocYUp : public SLCameraAnimation
{
public:
    SLCADeviceRotLocYUp()
    : SLCameraAnimation("CA_deviceRotLocYUp", "Device Rotated Y up and Gps Positioned")
    {}
};

#endif
