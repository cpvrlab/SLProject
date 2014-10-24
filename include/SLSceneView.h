//#############################################################################
//  File:      SLSceneView.h
//  Author:    Marc Wacker, Marcus Hudritsch
//  Date:      July 2014
//  Codestyle: https://code.google.com/p/slproject/wiki/CodingStyleGuidelines
//  Copyright: 2002-2014 Marcus Hudritsch
//             This software is provide under the GNU General Public License
//             Please visit: http://opensource.org/licenses/GPL-3.0
//#############################################################################

#ifndef SLSCENEVIEW_H
#define SLSCENEVIEW_H

#include <stdafx.h>

#include <SLScene.h>
#include <SLNode.h>
#include <SLEventHandler.h>
#include <SLRaytracer.h>
#include <SLPathtracer.h>
#include <SLAABBox.h>
#include <SLDrawBits.h>
#include <SLAverage.h>
#include <SLGLOculusFB.h>

//-----------------------------------------------------------------------------
class SLCamera;
class SLLight;
//-----------------------------------------------------------------------------
/*
There are only a very few callbacks from the SLProject library up to the GUI
framework. All other function calls are donw downwards from the GUI framework
into the SLProject library.
*/
//! Callback function typedef for custom SLSceneView derived creator function
typedef int (*cbOnNewSceneView)();

//! Callback function typedef for gui window update
typedef SLbool (SL_STDCALL *cbOnWndUpdate)(void);

//! Callback function typedef for select node 
typedef void (SL_STDCALL *cbOnSelectNodeMesh)(SLNode*, SLMesh*);

//! Callback function typedef for showing and hiding the system cursor
typedef void(SL_STDCALL *cbOnShowSysCursor)(bool);

//-----------------------------------------------------------------------------
//! SceneView class represents a dynamic real time 3D view onto the scene.
/*!      
The SLSceneView class has a pointer to an active camera that is used to 
generate the 3D view into a window of the clients GUI system. 
OpenGL ES2.0 is used the default renderer for framebuffer rendering.
Alternatively the sceneview can be rendered with a software raytracing or
pathtracing renderer. 
All mouse, touch, keyboard, resize and paint events of the GUI system are 
handled in this class by the appropriate event handler methods.
If the scene contains itself no camera node the sceneview provides its own
camere object.
*/
class SLSceneView: public SLObject
{   friend class SLNode;
    friend class SLRaytracer;
    friend class SLPathtracer;
   
    public:           
                            SLSceneView     ();
                           ~SLSceneView     ();

            void            init            (SLstring name,
                                             SLint screenWidth,
                                             SLint screenHeight,
                                             SLint dotsPerInch,
                                             SLVstring& cmdLineArgs,
                                             void* onWndUpdateCallback,
                                             void* onSelectNodeMeshCallback,
                                             void* onToggleSystemCursorCallback);

		   // virtual hooks for subclasses of SLSceneView
   virtual  void            onStartup       (SLVstring& cmdLineArgs) { }
   virtual  void            preDraw         () { }
   virtual  void            postDraw        () { }
   virtual  void            postSceneLoad   () { }

            // Main event handlers
            void            onInitialize    ();
            SLbool          onPaint         ();
            void            onResize        (const SLint width, const SLint height);
            SLbool          onMouseDown     (const SLMouseButton button, 
                                             SLint x, SLint y,
                                             const SLKey mod);  
            SLbool          onMouseUp       (const SLMouseButton button, 
                                             SLint x, SLint y,
                                             const SLKey mod); 
            SLbool          onMouseMove     (SLint x, SLint y);
            SLbool          onMouseWheelPos (const SLint wheelPos, const SLKey mod);
            SLbool          onMouseWheel    (const SLint delta, const SLKey mod); 
            SLbool          onTouch2Down    (SLint x1, SLint y1,
                                             SLint x2, SLint y2);
            SLbool          onTouch2Move    (SLint x1, SLint y1,
                                             SLint x2, SLint y2);
            SLbool          onTouch2Up      (SLint x1, SLint y1,
                                             SLint x2, SLint y2);
            SLbool          onDoubleClick   (const SLMouseButton button, 
                                             SLint x, SLint y,
                                             const SLKey mod);
    virtual SLbool          onKeyPress      (const SLKey key, const SLKey mod);
    virtual SLbool          onKeyRelease    (const SLKey key, const SLKey mod);
            void            onRotationPYR   (const SLfloat pitchRAD, 
                                             const SLfloat yawRAD, 
                                             const SLfloat rollRAD,
                                             const SLfloat zeroYawAfterSec);
            void            onRotationQUAT  (const SLfloat quatX, 
                                             const SLfloat quatY, 
                                             const SLfloat quatZ, 
                                             const SLfloat quatW);
            SLbool          onCommand       (const SLCmd cmd);
            
            // Drawing subroutines
            SLbool          updateAndDrawGL3D(SLfloat elapsedTimeSec);
            SLbool          updateAndDrawRT3D(SLfloat elapsedTimeSec);
            SLbool          updateAndDrawPT3D(SLfloat elapsedTimeSec);
            void            draw3D          ();
            void            draw3DNodeLines (SLVNode &nodes);
            void            draw3DNodes     (SLVNode &nodes);
            SLbool          updateAndDraw2D (SLfloat elapsedTimeSec);
            void            draw2D          ();
            
            // SceneView camera
            void            initSceneViewCamera(const SLVec3f& dir = -SLVec3f::AXISZ, 
                                                SLProjection proj = monoPerspective);
            void            switchToSceneViewCamera();
            SLbool          isSceneViewCameraActive() {return _camera == &_sceneViewCamera;}

            // Misc.
            void            rebuild2DMenus      ();
            void            build2DMenus        ();
            void            build2DInfoGL       ();
            void            build2DInfoRT       ();
            void            build2DInfoLoading  ();
            void            build2DMsgBoxes     ();
            SLfloat         calcFPS             (SLfloat deltaTimeSec); 
            SLstring        windowTitle         ();
            void            startRaytracing     (SLint maxDepth);
            void            startPathtracing    (SLint maxDepth, SLint samples);
            void            printStats          () {_stats.print();}

            // Callback routines
            cbOnWndUpdate   onWndUpdate;            //!< Callback for intermediate window repaint
            cbOnSelectNodeMesh onSelectedNodeMesh;  //!< Callback on node selection
            cbOnShowSysCursor onShowSysCursor;      //!< Callback for hiding and showing system cursor
   
            // Setters
            void            camera          (SLCamera* camera) {_camera = camera;}
            void            scrW            (const SLint  scrW){_scrW = scrW;}
            void            scrH            (const SLint  scrH){_scrH = scrH;} 
            void            waitEvents      (const SLbool wait){_waitEvents = wait;}
            void            dpi             (SLint newDPI) {_dpi = newDPI;}
            void            showLoading     (SLbool showLoading);
            void            showMenu        (SLbool show) {_showMenu = show;
                                                           SLScene::current->_menu2D = SLScene::current->_menuGL;}
            void            showInfo        (SLbool show) {_showInfo = show;}

            // Getters
            SLuint          index           () {return _index;}
    inline  SLCamera*       camera          () {return _camera;}
    inline  SLCamera*       sceneViewCamera    () {return &_sceneViewCamera;}
    inline  SLint           scrW            () {return _scrW;}
    inline  SLint           scrH            () {return _scrH;}
    inline  SLint           scrWdiv2        () {return _scrWdiv2;}
    inline  SLint           scrHdiv2        () {return _scrHdiv2;}
    inline  SLfloat         scrWdivH        () {return _scrWdivH;}
    inline  SLint           dpi             () {return _dpi;}
    inline  SLfloat         dpmm            () {return (float)_dpi/25.4f;}
    inline  SLQuat4f        deviceRotation  () {return _deviceRotation;}
            SLbool          doFrustumCulling() {return _doFrustumCulling;}
            SLbool          doMultiSampling () {return _doMultiSampling;}
            SLbool          hasMultiSampling() {return _hasMultiSampling;}
            SLbool          doDepthTest     () {return _doDepthTest;}
            SLbool          waitEvents      () {return _waitEvents;}
            SLbool          showStats       () {return _showStats;}
            SLbool          showInfo        () {return _showInfo;}
            SLbool          showMenu        () {return _showMenu;}
            SLfloat         fps             () {return _fps;}
            SLVNode*        blendNodes      () {return &_blendNodes;}
            SLVNode*        opaqueNodes     () {return &_opaqueNodes;}
            SLRaytracer*    raytracer       () {return &_raytracer;}
            SLPathtracer*   pathtracer      () {return &_pathtracer;}
            SLRenderer      renderType      () {return _renderType;}
            SLGLOculusFB*   oculusFB        () {return &_oculusFB;}
            SLDrawBits*     drawBits        () {return &_drawBits;}
            SLbool          drawBit         (SLuint bit) {return _drawBits.get(bit);}

   protected:
            SLuint          _index;             //!< index of this pointer in SLScene::sceneView vector
            SLGLState*      _stateGL;           //!< Pointer to the global SLGLState instance
            SLCamera*       _camera;            //!< Pointer to the _active camera
            SLCamera        _sceneViewCamera;      //!< Default editor camera for this SceneView (default cam not in scenegraph)
            SLNodeStats     _stats;             //!< Statistic numbers

            SLRenderer      _renderType;        //!< rendering type (GL,RT,PT)
            
            SLbool          _doDepthTest;       //!< Flag if depth test is turned on
            SLbool          _doMultiSampling;   //!< Flag if multisampling is on
            SLbool          _hasMultiSampling;  //!< Flag if multisampling is possible
            SLbool          _doFrustumCulling;  //!< Flag if view frustum culling is on
            SLbool          _waitEvents;        //!< Flag for Event waiting
            SLDrawBits      _drawBits;          //!< Sceneview level drawing flags

            SLbool          _showMenu;          //!< Flag if menu should be displayed
            SLbool          _showStats;         //!< Flag if stats should be displayed
            SLbool          _showInfo;          //!< Flag if help should be displayed
            SLbool          _showLoading;       //!< Flag if loading should be displayed
            
     static SLfloat         _lastFrameMS;       //!< Last frame time (this is static to work around a current bad design (see todo in onPaint))
            SLfloat         _fps;               //!< Averaged no. of frames per second
            SLAvgFloat      _frameTimeMS;       //!< Averaged time per frame in ms
            SLAvgFloat      _updateTimeMS;      //!< Averaged time for update in ms
            SLAvgFloat      _cullTimeMS;        //!< Averaged time for culling in ms
            SLAvgFloat      _draw3DTimeMS;      //!< Averaged time for 3D drawing in ms
            SLAvgFloat      _draw2DTimeMS;      //!< Averaged time for 2D drawing in ms
            SLuint          _totalBufferCount;  //!< Total NO. of VBOs in last frame
            SLuint          _totalBufferSize;   //!< Total size of buffer memory in last frame
            SLuint          _totalDrawCalls;    //!< Total NO. of drawCalls in last frame

            SLbool          _mouseDownL;        //!< Flag if left mouse button is pressed
            SLbool          _mouseDownR;        //!< Flag if right mouse button is pressed
            SLbool          _mouseDownM;        //!< Flag if middle mouse button is pressed
            SLKey           _mouseMod;          //!< mouse modifier key on key down
            SLint           _touchDowns;        //!< finger touche down count
            SLVec2i         _touch[3];          //!< up to 3 finger touch coordinates
            SLGLBuffer      _bufTouch;          //!< Buffer for touch pos. rendering
            SLGLBuffer      _bufCursor;         //!< Virtual cursor for stereo rendering
            
            SLVec2i         _posCursor;         //!< Cursor position as reported by the os
            SLint           _scrW;              //!< Screen width in pixels
            SLint           _scrH;              //!< Screen height in pixels
            SLint           _scrWdiv2;          //!< Screen half width in pixels
            SLint           _scrHdiv2;          //!< Screen half height in pixels
            SLfloat         _scrWdivH;          //!< Screen side aspect ratio
            SLint           _dpi;               //!< Screen resolution in dots per inch
            SLQuat4f        _deviceRotation;    //!< Mobile device rotation as quaternion

            SLGLOculusFB    _oculusFB;          //!< Oculus framebuffer
			SLbool			_vrMode;			//!< Flag if we're in VR mode (forces camera to stereoD)

            SLVNode         _blendNodes;        //!< Vector of blended nodes
            SLVNode         _opaqueNodes;       //!< Vector of opaque nodes
            
            SLRaytracer     _raytracer;         //!< Whitted style raytracer
            SLbool          _stopRT;            //!< Flag to stop the RT

            SLPathtracer    _pathtracer;        //!< Pathtracer
            SLbool          _stopPT;            //!< Flag to stop the PT 
};
//-----------------------------------------------------------------------------
#endif
