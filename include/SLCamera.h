//#############################################################################
//  File:      SLCamera.h
//  Author:    Marc Wacker, Marcus Hudritsch
//  Date:      July 2014
//  Codestyle: https://github.com/cpvrlab/SLProject/wiki/Coding-Style-Guidelines
//  Copyright: Marcus Hudritsch
//             This software is provide under the GNU General Public License
//             Please visit: http://opensource.org/licenses/GPL-3.0
//#############################################################################

#ifndef SLCAMERA_H
#define SLCAMERA_H

#include <stdafx.h>
#include <SLEnums.h>
#include <SLNode.h>
#include <SLGLState.h>
#include <SLGLBuffer.h>
#include <SLSamples2D.h>
#include <SLRay.h>

class SLSceneView;

//-----------------------------------------------------------------------------
//! Active or visible camera node class
/*! An instance of this SLNode derived class serves as an active camera with
all its view and projection parameters or if inactive as a visible scene graph
node with camera body and its view frustum. The position and orientation
of the active camera is set in the setView method by loading the viewmatrix _vm
into the OpenGL modelview matrix. The view matrix _vm is simply the inverse of
the shapes world matrix _wm. Every SLSceneView instance has a pointer _camera
to its active camera.
Because the SLNode class is inherited from the abstract SLEventHandler class a
camera can handle mouse & keyboard event. All camera animations are handled in
these eventhandlers.
*/ 
class SLCamera: public SLNode
{  
    public:
                            SLCamera        ();
                           ~SLCamera        ();

            SLbool          hitRec          (SLRay* ray){(void)ray; return false;}
            void            statsRec        (SLNodeStats &stats);
            SLAABBox&       updateAABBRec    ();

            void            drawMeshes      (SLSceneView* sv);
    virtual SLbool          camUpdate       (SLfloat timeMS);
            void            preShade        (SLRay* ray){(void)ray;}
            void            calcMinMax      (SLVec3f &minV, SLVec3f &maxV);

            // Event handlers for camera animation
    virtual SLbool          onMouseDown     (const SLMouseButton button, 
                                             const SLint x, const SLint y, 
                                             const SLKey mod); 
    virtual SLbool          onMouseMove     (const SLMouseButton button, 
                                             const SLint x, const SLint y,
                                             const SLKey mod);
    virtual SLbool          onMouseUp       (const SLMouseButton button, 
                                             const SLint x, const SLint y, 
                                             const SLKey mod);
    virtual SLbool          onMouseWheel    (const SLint delta, const SLKey mod);
    virtual SLbool          onTouch2Down    (const SLint x1, const SLint y1,
                                             const SLint x2, const SLint y2);
    virtual SLbool          onTouch2Move    (const SLint x1, const SLint y1,
                                             const SLint x2, const SLint y2);
    virtual SLbool          onTouch2Up      (const SLint x1, const SLint y1,
                                             const SLint x2, const SLint y2);
    virtual SLbool          onKeyPress      (const SLKey key, const SLKey mod);
    virtual SLbool          onKeyRelease    (const SLKey key, const SLKey mod);
                            
            void            eyeToPixelRay   (SLfloat x, SLfloat y, SLRay* ray);  
            SLbool          isInFrustum     (SLAABBox* aabb);
                            
            // Apply projection, viewport and view transformations
            void            setProjection   (SLSceneView* sv, const SLEye eye);
            void            setView         (SLSceneView* sv, const SLEye eye);
            void            setFrustumPlanes();

            // Setters
            void            unitScaling     (SLfloat s)          {_unitScaling = s; }

            void            projection      (SLProjection p)     {_projection = p;
                                                                  currentProjection = p;}
            void            fov             (const SLfloat fov)  {_fov = fov;
                                                                  currentFOV = fov;}
            void            camAnim         (SLCamAnim ca)       {_camAnim = ca;
                                                                  currentAnimation = ca;}
            void            clipNear        (const SLfloat cNear){_clipNear = cNear;}
            void            clipFar         (const SLfloat cFar) {_clipFar = cFar;}
            void            numRendered     (const SLuint numR)  {_numRendered = numR;}
            void            maxSpeed        (const SLfloat ms)   {_maxSpeed = ms;}
            void            focalDist       (const SLfloat f)    {_focalDist = f;}
            void            lensDiameter    (const SLfloat d)    {_lensDiameter = d;}
            void            lensSamples     (SLint x, SLint y)   {_lensSamples.samples(x, y);}
            void            eyeSeparation   (const SLfloat es)   {_eyeSeparation = es;}
            void            useDeviceRot    (const SLbool use)   {_useDeviceRot = use;}
               
            // Getters
            const SLMat4f&  updateAndGetVM  () const {return updateAndGetWMI();}
            SLProjection    projection      () const {return _projection;}
            SLstring        projectionStr   () const {return projectionToStr(_projection);}
            
            SLfloat         unitScaling     () {return _unitScaling;}     

            SLfloat         fov             () const {return _fov;}
            SLfloat         aspect          () const {return _aspect;}
            SLfloat         clipNear        () const {return _clipNear;}
            SLfloat         clipFar         () const {return _clipFar;}
            SLCamAnim       camAnim         () const {return _camAnim;}
            SLstring        animationStr    () const;
            SLuint          numRendered     () const {return _numRendered;}
            SLfloat         focalDist       () const {return _focalDist;} 
            SLfloat         lensDiameter    () const {return _lensDiameter;}
            SLSamples2D*    lensSamples     () {return &_lensSamples;} 
            SLfloat         eyeSeparation   () const {return _eyeSeparation;}
            SLfloat         focalDistScrW   () const;
            SLfloat         focalDistScrH   () const;
            SLRay*          lookAtRay       () {return &_lookAtRay;}
            SLfloat         maxSpeed        () const {return _maxSpeed;}
            SLbool          useDeviceRot    () const {return _useDeviceRot;}
            SLstring        toString        () const;
   

    // Static global default parameters for new cameras
    static  SLCamAnim       currentAnimation;
    static  SLProjection    currentProjection;
    static  SLfloat         currentFOV;
    static  SLint           currentDevRotation;
    static  SLstring        projectionToStr(SLProjection p);

   protected:
            // projection parameters
            SLProjection    _projection;            //!< projection type
            SLint           _scrW;                  //!< screen width in pixels
            SLint           _scrH;                  //!< screen height in pixels
            SLfloat         _aspect;                //!< _scrW /_srcH = screen ratio
            SLfloat         _fov;                   //!< Current field of view (view angle)
            SLfloat         _fovInit;               //!< Initial field of view (view angle)
            SLfloat         _clipNear;              //!< Dist. to the near clipping plane
            SLfloat         _clipFar;               //!< Dist. to the far clipping plane
            SLPlane         _plane[6];              //!< 6 frustum planes (t, b, l, r, n, f)
            SLuint          _numRendered;           //!< num. of shapes in frustum
            enum {T=0,B,L,R,N,F};                   // enumeration for frustum planes
               
            SLGLBuffer      _bufP;                  //!< Buffer object for visualization
               
            // animation parameters
            SLCamAnim       _camAnim;               //!< Type of camera animation
            SLVec2f         _oldTouchPos1;          //!< Old mouse/thouch position in pixels
            SLVec2f         _oldTouchPos2;          //!< Old 2nd finger touch position in pixels

            SLVec3f         _moveDir;               //!< accumulated movement directions based on pressed buttons
            SLfloat         _drag;                  //!< simple constant drag that affects velocity
            SLfloat         _maxSpeed;              //!< maximum speed in m/s, with high drag values this speed might not be achievable at all
            SLVec3f         _velocity;              //!< current velocity vector
            SLVec3f         _acceleration;          //!< current acceleration vector
            SLfloat         _brakeAccel;            //!< brake acceleration
            SLfloat         _moveAccel;             //!< move acceleration
            
            SLbool          _useDeviceRot;          //!< Flag if mobile device or oculus Rift rotation should be used 
               
            // ray tracing parameters
            SLRay           _lookAtRay;             //!< Ray through the center of screen
            SLfloat         _focalDist;             //!< distance of focal plane from lens
            SLfloat         _lensDiameter;          //!< Lens diameter
            SLSamples2D     _lensSamples;           //!< samplepoints for lens sampling (dof)

            // Stereo rendering
            SLfloat         _eyeSeparation;         //!< eye separation for stereo mode
            SLfloat         _unitScaling;           //!< indicate what the current unit scaling is to ajust movement and stereo rendering correctly
};
//-----------------------------------------------------------------------------
#endif
