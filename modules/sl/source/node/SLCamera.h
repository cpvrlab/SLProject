//#############################################################################
//  File:      SLCamera.h
//  Date:      July 2014
//  Codestyle: https://github.com/cpvrlab/SLProject/wiki/SLProject-Coding-Style
//  Authors:   Marc Wacker, Marcus Hudritsch
//  License:   This software is provided under the GNU General Public License
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
#include <SLRaySamples2D.h>
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
    explicit SLCamera(const SLstring& name                    = "Camera",
                      SLStdShaderProg textureOnlyProgramId    = SP_TextureOnly,
                      SLStdShaderProg colorAttributeProgramId = SP_colorAttribute);
    ~SLCamera() override;

    void           statsRec(SLNodeStats& stats) override;
    void           drawMesh(SLSceneView* sv) override;
    virtual SLbool camUpdate(SLSceneView* sv, SLfloat timeMS);
    void           preShade(SLRay* ray) { (void)ray; }
    void           calcMinMax(SLVec3f& minV, SLVec3f& maxV) const;
    void           buildAABB(SLAABBox& aabb, const SLMat4f& wmNode);
    SLVec2f        frustumSizeAtDistance(SLfloat distance);

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
    void projType(SLProjType p)
    {
        _projType         = p;
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
    void fogMode(const SLFogMode mode) { _fogMode = mode; }
    void fogDensity(const float density) { _fogDensity = density; }
    void onCamUpdateCB(function<void(SLSceneView* sv)> callbackFunc) { _onCamUpdateCB = callbackFunc; }

    // Getters
    const SLMat4f& updateAndGetVM() const { return updateAndGetWMI(); }
    SLProjType     projType() const { return _projType; }
    SLstring       projTypeStr() const { return projTypeToStr(_projType); }
    SLfloat        unitScaling() const { return _unitScaling; }
    SLfloat        fovV() const { return _fovV; } //!< Vertical field of view
    SLfloat        fovH() const;                  //!< Horizontal field of view

    SLRecti   viewport() const { return _viewport; }
    SLfloat   aspect() const { return _viewportRatio; }
    SLfloat   clipNear() const { return _clipNear; }
    SLfloat   clipFar() const { return _clipFar; }
    SLCamAnim camAnim() const { return _camAnim; }
    SLstring  animationStr() const;

    SLfloat stereoEyeSeparation() const { return _stereoEyeSeparation; }
    SLint   stereoEye() const { return _stereoEye; }
    SLMat3f stereoColorFilter() const { return _stereoColorFilter; }

    SLfloat         lensDiameter() const { return _lensDiameter; }
    SLRaySamples2D* lensSamples() { return &_lensSamples; }
    SLfloat         focalDist() const { return _focalDist; }
    SLfloat         focalDistScrW() const;
    SLfloat         focalDistScrH() const;
    SLVec3f         focalPointWS() const { return translationWS() + _focalDist * forwardWS(); }
    SLVec3f         focalPointOS() const { return translationOS() + _focalDist * forwardOS(); }

    SLbool    fogIsOn() const { return _fogIsOn; }
    SLFogMode fogMode() const { return _fogMode; }
    SLfloat   fogDensity() const { return _fogDensity; }
    SLfloat   fogDistStart() const { return _fogStart; }
    SLfloat   fogDistEnd() const { return _fogEnd; }
    SLCol4f   fogColor() const { return _fogColor; }

    SLfloat       trackballSize() const { return _trackballSize; }
    SLBackground& background() { return _background; }
    SLfloat       maxSpeed() const { return _maxSpeed; }
    SLfloat       moveAccel() const { return _moveAccel; }
    SLfloat       brakeAccel() const { return _brakeAccel; }
    SLfloat       drag() const { return _drag; }
    SLstring      toString() const;
    SLRectf&      selectRect() { return _selectRect; }
    SLRectf&      deselectRect() { return _deselectRect; }

    // update rotation matrix _enucorrRenu
    void updateEnuCorrRenu(SLSceneView* sv, const SLMat3f& enuRc, float& f, SLVec3f& enuOffsetPix);

    // Static global default parameters for new cameras
    static SLCamAnim  currentAnimation;
    static SLProjType currentProjection;
    static SLfloat    currentFOV;
    static SLint      currentDevRotation;
    static SLstring   projTypeToStr(SLProjType pt);

protected:
    // projection parameters
    SLProjType   _projType;       //!< Projection type
    SLfloat      _fovV;           //!< Current vertical field of view (view angle) in degrees
    SLfloat      _fovInit;        //!< Initial vertical field of view (view angle) in degrees
    SLfloat      _clipNear;       //!< Dist. to the near clipping plane
    SLfloat      _clipFar;        //!< Dist. to the far clipping plane
    SLPlane      _plane[6];       //!< 6 frustum planes (l, r, t, b, n, f)
    SLRecti      _viewport;       //!< framebuffer rectangle
    SLfloat      _viewportRatio;  //!< viewport.width / viewport.height = screen ratio
    SLfloat      _fx;             //!< horizontal focal length
    SLfloat      _fy;             //!< vertical focal length
    SLfloat      _cx;             //!< sensor center in x direction
    SLfloat      _cy;             //!< sensor center in y direction
    SLBackground _background;     //!< Colors or texture displayed in the background

    SLGLVertexArrayExt _vao;      //!< OpenGL Vertex array for rendering

    SLbool    _movedLastFrame;    //! did the camera updateRec in the last frame?
    SLCamAnim _camAnim;           //!< Type of camera animation
    SLVec2f   _oldTouchPos1;      //!< Old mouse/touch position in pixels
    SLVec2f   _oldTouchPos2;      //!< Old 2nd finger touch position in pixels
    SLVec3f   _trackballStartVec; //!< Trackball vector at mouse down
    SLfloat   _trackballSize;     //!< Size of trackball (0.8 = 80% of window size)

    SLVec3f _moveDir;             //!< accumulated movement directions based on pressed buttons
    SLfloat _drag;                //!< simple constant drag that affects velocity
    SLfloat _maxSpeed;            //!< maximum speed in m/s, with high drag values this speed might not be achievable at all
    SLVec3f _velocity;            //!< current velocity vector
    SLVec3f _acceleration;        //!< current acceleration vector
    SLfloat _brakeAccel;          //!< brake acceleration
    SLfloat _moveAccel;           //!< move acceleration

    // ray tracing parameters
    SLfloat        _focalDist;    //!< distance to lookAt point on the focal plane from lens
    SLfloat        _lensDiameter; //!< Lens diameter
    SLRaySamples2D _lensSamples;  //!< sample points for lens sampling (DOF)

    // Stereo rendering
    SLfloat _stereoEyeSeparation; //!< eye separation for stereo mode
    SLfloat _unitScaling;         //!< indicate what the current unit scale is
    SLint   _stereoEye;           //!< -1=left, 0=center, 1=right
    SLMat3f _stereoColorFilter;   //!< color filter matrix for anaglyphling is to adjust movement and stereo rendering correctly

    // fog
    SLbool    _fogIsOn;        //!< Flag if fog blending is enabled
    SLFogMode _fogMode;        //!< 0=LINEAR, 1=EXP, 2=EXP2
    SLfloat   _fogDensity;     //!< Fog density for exponential modes
    SLfloat   _fogStart;       //!< Fog start distance for linear mode
    SLfloat   _fogEnd;         //!< Fog end distance for linear mode
    SLCol4f   _fogColor;       //!< fog color blended to the final color
    SLbool    _fogColorIsBack; //!< fog color blended to the final color

    SLDeviceRotation* _devRot = nullptr;
    SLDeviceLocation* _devLoc = nullptr;

    SLRectf _selectRect;   //!< Mouse selection rectangle. See SLMesh::handleRectangleSelection
    SLRectf _deselectRect; //!< Mouse deselection rectangle. See SLMesh::handleRectangleSelection

    //! parameter for manual finger rotation and translation
    SLint   _xOffsetPix = 0;
    SLint   _yOffsetPix = 0;
    SLMat3f _enucorrRenu;

    function<void(SLSceneView* sv)> _onCamUpdateCB;
};
//-----------------------------------------------------------------------------
#endif
