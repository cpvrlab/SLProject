//#############################################################################
//  File:      SLSceneView.h
//  Author:    Marc Wacker, Marcus Hudritsch
//  Date:      July 2014
//  Codestyle: https://github.com/cpvrlab/SLProject/wiki/Coding-Style-Guidelines
//  Copyright: Marcus Hudritsch
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
#include <SLGLOculusFB.h>
#include <SLGLVertexArrayExt.h>

//-----------------------------------------------------------------------------
class SLCamera;
class SLLight;
//-----------------------------------------------------------------------------
/*
There are only a very few callbacks from the SLProject library up to the GUI
framework. All other function calls are downwards from the GUI framework
into the SLProject library.
*/
//! Callback function typedef for custom SLSceneView derived creator function
typedef int (*cbOnNewSceneView)();

//! Callback function typedef for GUI window update
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
OpenGL ES2.0 or newer is used the default renderer for framebuffer rendering.
Alternatively the sceneview can be rendered with a software ray tracing or
path tracing renderer. 
All mouse, touch, keyboard, resize and paint events of the GUI system are 
handled in this class by the appropriate event handler methods.
If the scene contains itself no camera node the sceneview provides its own
camera object.
*/
class SLSceneView: public SLObject
{   friend class SLNode;
    friend class SLRaytracer;
    friend class SLPathtracer;
   
    public:           
                            SLSceneView     ();
    virtual                ~SLSceneView     ();

            void            init            (SLstring name,
                                             SLint screenWidth,
                                             SLint screenHeight,
                                             void* onWndUpdateCallback,
                                             void* onSelectNodeMeshCallback,
                                             void* onToggleSystemCursorCallback);

		      // virtual hooks for subclasses of SLSceneView
   virtual  void            onStartup       () { }
   virtual  void            preDraw         () { }
   virtual  void            postDraw        () { }
   virtual  void            postSceneLoad   () { }

            // Main event handlers
            void            onInitialize    ();
            SLbool          onPaint         ();
            void            onResize        (SLint width, SLint height);
            SLbool          onMouseDown     (SLMouseButton button, 
                                             SLint x, SLint y, SLKey mod);  
            SLbool          onMouseUp       (SLMouseButton button, SLint x, SLint y,
                                             SLKey mod); 
            SLbool          onMouseMove     (SLint x, SLint y);
            SLbool          onMouseWheelPos (SLint wheelPos, SLKey mod);
            SLbool          onMouseWheel    (SLint delta, SLKey mod); 
            SLbool          onTouch2Down    (SLint x1, SLint y1, SLint x2, SLint y2);
            SLbool          onTouch2Move    (SLint x1, SLint y1, SLint x2, SLint y2);
            SLbool          onTouch2Up      (SLint x1, SLint y1, SLint x2, SLint y2);
            SLbool          onDoubleClick   (SLMouseButton button, 
                                             SLint x, SLint y,
                                             SLKey mod);
            SLbool          onLongTouch     (SLint x, SLint y);
    virtual SLbool          onKeyPress      (SLKey key, SLKey mod);
    virtual SLbool          onKeyRelease    (SLKey key, SLKey mod);
            void            onRotationPYR   (SLfloat pitchRAD, 
                                             SLfloat yawRAD, 
                                             SLfloat rollRAD,
                                             SLfloat zeroYawAfterSec);
            void            onRotationQUAT  (SLfloat quatX, 
                                             SLfloat quatY, 
                                             SLfloat quatZ, 
                                             SLfloat quatW);
            SLbool          onCommand       (SLCommand cmd);
            
            // Drawing subroutines
            SLbool          draw3DGL            (SLfloat elapsedTimeSec);
            void            draw3DGLAll         ();
            void            draw3DGLNodes       (SLVNode &nodes,
                                                 SLbool alphaBlended,
                                                 SLbool depthSorted);
            void            draw3DGLLines       (SLVNode &nodes);
            void            draw3DGLLinesOverlay(SLVNode &nodes);
            void            draw2DGL            ();
            void            draw2DGLAll         ();
            SLbool          draw3DRT            ();
            SLbool          draw3DPT            ();
            
            // SceneView camera
            void            initSceneViewCamera (const SLVec3f& dir = -SLVec3f::AXISZ, 
                                                SLProjection proj = P_monoPerspective);
            void            switchToSceneViewCamera();
            SLbool          isSceneViewCameraActive() {return _camera == &_sceneViewCamera;}

            // Misc.
            void            rebuild2DMenus      (SLbool showAboutFirst=false);
            void            build2DMenus        ();
            void            build2DInfoGL       ();
            void            build2DInfoRT       ();
            void            build2DInfoLoading  ();
            void            build2DMsgBoxes     ();
            SLstring        windowTitle         ();
            void            startRaytracing     (SLint maxDepth);
            void            startPathtracing    (SLint maxDepth, SLint samples);
            void            printStats          () {_stats3D.print();}
            SLbool          testRunIsFinished   ();

            // Callback routines
            cbOnWndUpdate       onWndUpdate;        //!< Callback for intermediate window repaint
            cbOnSelectNodeMesh  onSelectedNodeMesh; //!< Callback on node selection
            cbOnShowSysCursor   onShowSysCursor;    //!< Callback for hiding and showing system cursor
   
            // Setters
            void            camera              (SLCamera* camera) {_camera = camera;}
            void            scrW                (SLint  scrW){_scrW = scrW;}
            void            scrH                (SLint  scrH){_scrH = scrH;}
            void            waitEvents          (SLbool wait){_waitEvents = wait;}
            void            usesRotation        (SLbool use){_usesRotation = use;}
            void            showLoading         (SLbool showLoading);
            void            showMenu            (SLbool show){_showMenu = show;
                                                              SLScene::current->menu2D(SLScene::current->menuGL());}
            void            showInfo            (SLbool show) {_showInfo = show;}
            void            showStatsTiming     (SLbool show) {_showStatsTiming = show;}
            void            showStatsRenderer   (SLbool show) {_showStatsRenderer = show;}
            void            showStatsScene      (SLbool show) {_showStatsScene = show;}
            void            showStatsCamera     (SLbool show) {_showStatsCamera = show;}
            void            showStatsVideo      (SLbool show) {_showStatsVideo = show;}
            void            gotPainted          (SLbool val) {_gotPainted = val;}

            // Getters
            SLuint          index               () const {return _index;}
    inline  SLCamera*       camera              () {return _camera;}
    inline  SLCamera*       sceneViewCamera     () {return &_sceneViewCamera;}
    inline  SLint           scrW                () const {return _scrW;}
    inline  SLint           scrH                () const {return _scrH;}
    inline  SLint           scrWdiv2            () const {return _scrWdiv2;}
    inline  SLint           scrHdiv2            () const {return _scrHdiv2;}
    inline  SLfloat         scrWdivH            () const {return _scrWdivH;}
    inline  SLQuat4f        deviceRotation      () const {return _deviceRotation;}
            SLbool          gotPainted          () const {return _gotPainted;}
            SLbool          doFrustumCulling    () const {return _doFrustumCulling;}
            SLbool          hasMultiSampling    () const {return _stateGL->hasMultiSampling();}
            SLbool          doMultiSampling     () const {return _doMultiSampling;}
            SLbool          doDepthTest         () const {return _doDepthTest;}
            SLbool          usesRotation        () const {return _usesRotation;}
            SLbool          waitEvents          () const {return _waitEvents;}
            SLbool          showStatsTiming     () const {return _showStatsTiming;}
            SLbool          showStatsRenderer   () const {return _showStatsRenderer;}
            SLbool          showStatsScene      () const {return _showStatsScene;}
            SLbool          showStatsCamera     () const {return _showStatsCamera;}
            SLbool          showStatsVideo      () const {return _showStatsVideo;}
            SLbool          showInfo            () const {return _showInfo;}
            SLbool          showMenu            () const {return _showMenu;}
            SLVNode*        visibleNodes        () {return &_visibleNodes;}
            SLVNode*        blendNodes          () {return &_blendNodes;}
            SLRaytracer*    raytracer           () {return &_raytracer;}
            SLPathtracer*   pathtracer          () {return &_pathtracer;}
            SLRenderType    renderType          () const {return _renderType;}
            SLGLOculusFB*   oculusFB            () {return &_oculusFB;}
            SLDrawBits*     drawBits            () {return &_drawBits;}
            SLbool          drawBit             (SLuint bit) {return _drawBits.get(bit);}
            SLfloat         cullTimeMS          () const {return _cullTimeMS;}
            SLfloat         draw3DTimeMS        () const {return _draw3DTimeMS;}
            SLfloat         draw2DTimeMS        () const {return _draw2DTimeMS;}

    static const SLint      LONGTOUCH_MS;       //!< Milliseconds duration of a long touch event

   protected:
            SLuint          _index;             //!< index of this pointer in SLScene::sceneView vector
            SLGLState*      _stateGL;           //!< Pointer to the global SLGLState instance
            SLCamera*       _camera;            //!< Pointer to the _active camera
            SLCamera        _sceneViewCamera;   //!< Default camera for this SceneView (default cam not in scenegraph)
            SLNodeStats     _stats2D;           //!< Statistic numbers for 2D nodes
            SLNodeStats     _stats3D;           //!< Statistic numbers for 3D nodes
            SLbool          _gotPainted;        //!< flag if this sceneview got painted

            SLRenderType    _renderType;        //!< rendering type (GL,RT,PT)
            
            SLbool          _doDepthTest;       //!< Flag if depth test is turned on
            SLbool          _doMultiSampling;   //!< Flag if multisampling is on
            SLbool          _doFrustumCulling;  //!< Flag if view frustum culling is on
            SLbool          _waitEvents;        //!< Flag for Event waiting
            SLbool          _isFirstFrame;      //!< Flag if it is the first frame rendering
            SLbool          _usesRotation;      //!< Flag if device rotation is used
            SLDrawBits      _drawBits;          //!< Sceneview level drawing flags

            SLbool          _showStatsTiming;   //!< Flag if timing stats should be displayed
            SLbool          _showStatsRenderer; //!< Flag if GL or RT stats should be displayed
            SLbool          _showStatsCamera;   //!< Flag if camera stats should be displayed
            SLbool          _showStatsScene;   //!< Flag if memory stats should be displayed
            SLbool          _showStatsVideo;    //!< Flag if video stats should be displayed
            SLbool          _showMenu;          //!< Flag if menu should be displayed
            SLbool          _showInfo;          //!< Flag if help should be displayed
            SLbool          _showLoading;       //!< Flag if loading should be displayed

            SLfloat         _cullTimeMS;        //!< time for culling in ms
            SLfloat         _draw3DTimeMS;      //!< time for 3D drawing in ms
            SLfloat         _draw2DTimeMS;      //!< time for 2D drawing in ms 

            SLbool          _mouseDownL;        //!< Flag if left mouse button is pressed
            SLbool          _mouseDownR;        //!< Flag if right mouse button is pressed
            SLbool          _mouseDownM;        //!< Flag if middle mouse button is pressed
            SLKey           _mouseMod;          //!< mouse modifier key on key down
            SLint           _touchDowns;        //!< finger touch down count
            SLVec2i         _touch[3];          //!< up to 3 finger touch coordinates
            SLGLVertexArrayExt _vaoTouch;       //!< Buffer for touch pos. rendering
            SLGLVertexArrayExt _vaoCursor;      //!< Virtual cursor for stereo rendering
            
            SLVec2i         _posCursor;         //!< Cursor position as reported by the os
            SLint           _scrW;              //!< Screen width in pixels
            SLint           _scrH;              //!< Screen height in pixels
            SLint           _scrWdiv2;          //!< Screen half width in pixels
            SLint           _scrHdiv2;          //!< Screen half height in pixels
            SLfloat         _scrWdivH;          //!< Screen side aspect ratio
            SLQuat4f        _deviceRotation;    //!< Mobile device rotation as quaternion

            SLGLOculusFB    _oculusFB;          //!< Oculus framebuffer
			SLbool			_vrMode;			//!< Flag if we're in VR mode (forces camera to stereoD)

            SLVNode         _blendNodes;        //!< Vector of visible and blended nodes
            SLVNode         _visibleNodes;      //!< Vector of all visible nodes
            
            SLRaytracer     _raytracer;         //!< Whitted style raytracer
            SLbool          _stopRT;            //!< Flag to stop the RT

            SLPathtracer    _pathtracer;        //!< Pathtracer
            SLbool          _stopPT;            //!< Flag to stop the PT
};
//-----------------------------------------------------------------------------
#endif
