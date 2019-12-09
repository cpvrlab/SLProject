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

class SLSceneView;

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
    SLCamera(const SLstring& name = "Camera");
    ~SLCamera() { ; }

    void statsRec(SLNodeStats& stats);

    virtual void   drawMeshes(SLSceneView* sv);
    virtual SLbool camUpdate(SLfloat timeMS);
    void           preShade(SLRay* ray) { (void)ray; }
    void           calcMinMax(SLVec3f& minV, SLVec3f& maxV);
    void           buildAABB(SLAABBox& aabb, const SLMat4f& wmNode);

    // Event handlers for camera animation
    virtual SLbool onMouseDown(SLMouseButton button,
                               SLint         x,
                               SLint         y,
                               SLKey         mod);
    virtual SLbool onMouseMove(SLMouseButton button,
                               SLint         x,
                               SLint         y,
                               SLKey         mod);
    virtual SLbool onMouseUp(SLMouseButton button,
                             SLint         x,
                             SLint         y,
                             SLKey         mod);
    virtual SLbool onMouseWheel(SLint delta, SLKey mod);
    virtual SLbool onTouch2Down(SLint x1, SLint y1, SLint x2, SLint y2);
    virtual SLbool onTouch2Move(SLint x1, SLint y1, SLint x2, SLint y2);
    virtual SLbool onTouch2Up(SLint x1, SLint y1, SLint x2, SLint y2);
    virtual SLbool onKeyPress(SLKey key, SLKey mod);
    virtual SLbool onKeyRelease(SLKey key, SLKey mod);

    void    eyeToPixelRay(SLfloat x, SLfloat y, SLRay* ray);
    SLVec3f trackballVec(SLint x, SLint y);
    SLbool  isInFrustum(SLAABBox* aabb);

    // Apply projection, viewport and view transformations
    void setViewport(SLSceneView* sv, SLEyeType eye);
    void setProjection(SLSceneView* sv, SLEyeType eye);
    void setView(SLSceneView* sv, SLEyeType eye);
    void setFrustumPlanes();

    // Setters
    void unitScaling(SLfloat s) { _unitScaling = s; }

    void projection(SLProjection p)
    {
        _projection       = p;
        currentProjection = p;
    }
    void fov(const SLfloat fov)
    {
        _fov       = fov;
        currentFOV = fov;
    }
    void camAnim(SLCamAnim ca)
    {
        _camAnim         = ca;
        currentAnimation = ca;
    }
    void intrinsics(const SLfloat fx, const SLfloat fy, const SLfloat cx, const SLfloat cy)
    {
        _fx = fx;
        _fy = fy;
        _cx = cx;
        _cy = cy;
    }
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
    void eyeSeparation(const SLfloat es) { _eyeSeparation = es; }

    // Getters
    const SLMat4f& updateAndGetVM() const { return updateAndGetWMI(); }
    SLProjection   projection() const { return _projection; }
    SLstring       projectionStr() const { return projectionToStr(_projection); }
    SLfloat        unitScaling() { return _unitScaling; }
    SLfloat        fov() const { return _fov; }
    SLfloat        aspect() const { return _viewportRatio; }
    SLfloat        clipNear() const { return _clipNear; }
    SLfloat        clipFar() const { return _clipFar; }
    SLCamAnim      camAnim() const { return _camAnim; }
    SLstring       animationStr() const;
    SLfloat        lensDiameter() const { return _lensDiameter; }
    SLSamples2D*   lensSamples() { return &_lensSamples; }
    SLfloat        eyeSeparation() const { return _eyeSeparation; }
    SLfloat        focalDist() const { return _focalDist; }
    SLfloat        focalDistScrW() const;
    SLfloat        focalDistScrH() const;
    SLVec3f        focalPointWS() const { return translationWS() + _focalDist * forwardWS(); }
    SLVec3f        focalPointOS() const { return translationOS() + _focalDist * forwardOS(); }
    SLfloat        trackballSize() const { return _trackballSize; }
    SLBackground&  background() { return _background; }
    SLfloat        maxSpeed() const { return _maxSpeed; }
    SLfloat        moveAccel() const { return _moveAccel; }
    SLfloat        brakeAccel() const { return _brakeAccel; }
    SLfloat        drag() const { return _drag; }
    SLstring       toString() const;

    // Static global default parameters for new cameras
    static SLCamAnim    currentAnimation;
    static SLProjection currentProjection;
    static SLfloat      currentFOV;
    static SLint        currentDevRotation;
    static SLstring     projectionToStr(SLProjection p);

protected:
    // projection parameters
    SLProjection _projection;    //!< projection type
    SLfloat      _fov;           //!< Current field of view (view angle)
    SLfloat      _fovInit;       //!< Initial field of view (view angle)
    SLfloat      _clipNear;      //!< Dist. to the near clipping plane
    SLfloat      _clipFar;       //!< Dist. to the far clipping plane
    SLPlane      _plane[6];      //!< 6 frustum planes (t, b, l, r, n, f)
    SLint        _viewportW;     //!< screen width in pixels
    SLint        _viewportH;     //!< screen height in pixels
    SLfloat      _viewportRatio; //!< _scrW /_srcH = screen ratio
    SLfloat      _fx;
    SLfloat      _fy;
    SLfloat      _cx;
    SLfloat      _cy;

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

    // animation parameters
    SLbool    _movedLastFrame;    //! did the camera update in the last frame?
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
    SLfloat _eyeSeparation; //!< eye separation for stereo mode
    SLfloat _unitScaling;   //!< indicate what the current unit scaling is to adjust movement and stereo rendering correctly
};
//-----------------------------------------------------------------------------
#endif
