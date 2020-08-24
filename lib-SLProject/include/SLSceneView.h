//#############################################################################
//  File:      SLSceneView.h
//  Author:    Marc Wacker, Marcus Hudritsch
//  Date:      July 2014
//  Codestyle: https://github.com/cpvrlab/SLProject/wiki/SLProject-Coding-Style
//  Copyright: Marcus Hudritsch
//             This software is provide under the GNU General Public License
//             Please visit: http://opensource.org/licenses/GPL-3.0
//#############################################################################

#ifndef SLSCENEVIEW_H
#define SLSCENEVIEW_H

#include <SLAABBox.h>
#include <SLDrawBits.h>
#include <SLEventHandler.h>
#include <SLGLOculusFB.h>
#include <SLGLVertexArrayExt.h>
#include <SLNode.h>
#include <SLPathtracer.h>
#include <SLRaytracer.h>
#include <SLGLConetracer.h>
#include <SLScene.h>
#include <SLSkybox.h>
#include <SLOptixRaytracer.h>
#include <SLOptixPathtracer.h>
#include <SLRect.h>
#include <SLUiInterface.h>

//-----------------------------------------------------------------------------
class SLCamera;
class SLLight;
class SLScene;
class SLInputManager;

//-----------------------------------------------------------------------------
/*
There are only a very few callbacks from the SLInterface up to the GUI
framework. All other function calls are downwards from the GUI framework
*/
//! Callback function typedef for custom SLSceneView derived creator function
typedef SLSceneView*(SL_STDCALL* cbOnNewSceneView)(SLScene* s, int dotsPerInch, SLInputManager& inputManager);

//! Callback function typedef for GUI window updateRec
typedef SLbool(SL_STDCALL* cbOnWndUpdate)();

//! Callback function typedef for select node
typedef void(SL_STDCALL* cbOnSelectNodeMesh)(SLNode*, SLMesh*);

//-----------------------------------------------------------------------------
//! SceneView class represents a dynamic real time 3D view onto the scene.
/*!
 The SLSceneView class has a pointer to an active camera that is used to
 generate the 3D view into a window of the clients GUI system.
 OpenGL ES3.0 or newer is used the default renderer for framebuffer rendering.
 Alternatively the sceneview can be rendered with a software ray tracing or
 path tracing renderer.
 All mouse, touch, keyboard, resize and paint events of the GUI system are
 handled in this class by the appropriate event handler methods. If your
 app need special event handling you can subclass this class and override the
 virtual function.
 If the scene contains itself no camera node the sceneview provides its own
 camera object.
*/
class SLSceneView : public SLObject
{
    friend class SLNode;
    friend class SLRaytracer;
    friend class SLOptixRaytracer;
    friend class SLPathtracer;

public:
    SLSceneView(SLScene* s, int dpi, SLInputManager& inputManager);
    ~SLSceneView() override;

    void init(SLstring           name,
              SLint              screenWidth,
              SLint              screenHeight,
              void*              onWndUpdateCallback,
              void*              onSelectNodeMeshCallback,
              SLUiInterface*     gui,
              const std::string& configPath);
    void unInit();

    // Not overridable event handlers
    void   onInitialize();
    SLbool onPaint();
    void   onResize(SLint width, SLint height);

    // overridable for subclasses of SLSceneView
    virtual void   onStartup() {}
    virtual void   preDraw() {}
    virtual void   postDraw() {}
    virtual void   postSceneLoad() {}
    virtual SLbool onMouseDown(SLMouseButton button, SLint scrX, SLint scrY, SLKey mod);
    virtual SLbool onMouseUp(SLMouseButton button, SLint scrX, SLint scrY, SLKey mod);
    virtual SLbool onMouseMove(SLint x, SLint y);
    virtual SLbool onMouseWheelPos(SLint wheelPos, SLKey mod);
    virtual SLbool onMouseWheel(SLint delta, SLKey mod);
    virtual SLbool onTouch2Down(SLint scrX1, SLint scrY1, SLint scrX2, SLint scrY2);
    virtual SLbool onTouch2Move(SLint scrX1, SLint scrY1, SLint scrX2, SLint scrY2);
    virtual SLbool onTouch2Up(SLint scrX1, SLint scrY1, SLint scrX2, SLint scrY2);
    virtual SLbool onDoubleClick(SLMouseButton button, SLint x, SLint y, SLKey mod);
    virtual SLbool onLongTouch(SLint x, SLint y);
    virtual SLbool onKeyPress(SLKey key, SLKey mod);
    virtual SLbool onKeyRelease(SLKey key, SLKey mod);
    virtual SLbool onCharInput(SLuint c);

    // Drawing subroutines
    SLbool draw3DGL(SLfloat elapsedTimeSec);
    void   draw3DGLAll();
    void   draw3DGLNodes(SLVNode& nodes, SLbool alphaBlended, SLbool depthSorted);
    void   draw3DGLLines(SLVNode& nodes);
    void   draw3DGLLinesOverlay(SLVNode& nodes);
    void   draw2DGL();
    void   draw2DGLNodes();
    SLbool draw3DRT();
    SLbool draw3DPT();
    SLbool draw3DCT();

    // SceneView camera
    void   initSceneViewCamera(const SLVec3f& dir  = -SLVec3f::AXISZ,
                               SLProjection   proj = P_monoPerspective);
    void   switchToSceneViewCamera();
    void   switchToNextCameraInScene();
    SLbool isSceneViewCameraActive() { return _camera == &_sceneViewCamera; }

    // Misc.
    SLstring windowTitle();
    void     printStats() { _stats3D.print(); }
    void     startRaytracing(SLint maxDepth);
    void     startPathtracing(SLint maxDepth, SLint samples);
    void     startConetracing();
    void     setViewportFromRatio(const SLVec2i&  vpRatio,
                                  SLViewportAlign vpAlignment,
                                  SLbool          vpSameAsVideo);
    void     initConeTracer(SLstring shaderDir);

    // Callback routines
    cbOnWndUpdate      onWndUpdate;        //!< C-Callback for app for intermediate window repaint
    cbOnSelectNodeMesh onSelectedNodeMesh; //!< C-Callback for app on node selection

    // Setters
    void camera(SLCamera* camera) { _camera = camera; }
    void scene(SLScene* scene) { _s = scene; }
    void skybox(SLSkybox* skybox) { _skybox = skybox; }
    void scrW(SLint scrW) { _scrW = scrW; }
    void scrH(SLint scrH) { _scrH = scrH; }
    void doWaitOnIdle(SLbool doWI) { _doWaitOnIdle = doWI; }
    void doMultiSampling(SLbool doMS) { _doMultiSampling = doMS; }
    void doDepthTest(SLbool doDT) { _doDepthTest = doDT; }
    void doFrustumCulling(SLbool doFC) { _doFrustumCulling = doFC; }
    void renderType(SLRenderType rt) { _renderType = rt; }
    void viewportSameAsVideo(bool sameAsVideo) { _viewportSameAsVideo = sameAsVideo; }
    void scr2fb(float scr2fbX, float scr2fbY)
    {
        _scr2fbX = scr2fbX;
        _scr2fbY = scr2fbY;

        if (_gui)
            _gui->onResize(_viewportRect.width,
                           _viewportRect.height,
                           _scr2fbX,
                           _scr2fbY);
    }

    // Getters
    SLScene*        s() { return _s; }
    SLCamera*       camera() { return _camera; }
    SLCamera*       sceneViewCamera() { return &_sceneViewCamera; }
    SLSkybox*       skybox() { return _skybox; }
    SLint           scrW() const { return _scrW; }
    SLint           scrH() const { return _scrH; }
    SLint           scrWdiv2() const { return _scrWdiv2; }
    SLint           scrHdiv2() const { return _scrHdiv2; }
    SLfloat         scrWdivH() const { return _scrWdivH; }
    SLfloat         scr2fbX() const { return _scr2fbX; }
    SLfloat         scr2fbY() const { return _scr2fbY; }
    SLint           dpi() const { return _dpi; }
    SLfloat         dpmm() const { return (float)_dpi / 25.4f; }
    SLRecti         viewportRect() const { return _viewportRect; }
    SLVec2i         viewportRatio() const { return _viewportRatio; }
    SLfloat         viewportWdivH() const { return (float)_viewportRect.width / (float)_viewportRect.height; }
    SLint           viewportW() const { return _viewportRect.width; }
    SLint           viewportH() const { return _viewportRect.height; }
    SLViewportAlign viewportAlign() const { return _viewportAlign; }
    SLbool          viewportSameAsVideo() const { return _viewportSameAsVideo; }
    SLUiInterface*  gui() { return _gui; }
    SLbool          doFrustumCulling() const { return _doFrustumCulling; }
    SLbool          doMultiSampling() const { return _doMultiSampling; }
    SLbool          doDepthTest() const { return _doDepthTest; }
    SLbool          doWaitOnIdle() const { return _doWaitOnIdle; }
#ifdef SL_RENDER_BY_MATERIAL
    SLVNode&        nodesText2D() { return _nodesText2D; }
    SLVNode&        nodesText3D() { return _nodesText3D; }
#else
    SLVNode&        nodesVisible() { return _nodesVisible; }
    SLVNode&        nodesVisible2D() { return _nodesVisible2D; }
    SLVNode&        nodesBlended() { return _nodesBlended; }
#endif
    SLVNode&        nodesOverdrawn() { return _nodesOverdrawn; }
    SLRaytracer*    raytracer() { return &_raytracer; }
    SLPathtracer*   pathtracer() { return &_pathtracer; }
    SLGLConetracer* conetracer() { return _conetracer.get(); }
    SLRenderType    renderType() const { return _renderType; }
    SLGLOculusFB*   oculusFB() { return &_oculusFB; }
    SLDrawBits*     drawBits() { return &_drawBits; }
    SLbool          drawBit(SLuint bit) { return _drawBits.get(bit); }
    AvgFloat&       shadowMapTimeMS() { return _shadowMapTimesMS; }
    AvgFloat&       cullTimesMS() { return _cullTimesMS; }
    AvgFloat&       draw2DTimesMS() { return _draw2DTimesMS; }
    AvgFloat&       draw3DTimesMS() { return _draw3DTimesMS; }
    SLNodeStats&    stats2D() { return _stats2D; }
    SLNodeStats&    stats3D() { return _stats3D; }

#ifdef SL_HAS_OPTIX
    SLOptixRaytracer* optixRaytracer()
    {
        return &_optixRaytracer;
    }
    SLOptixPathtracer* optixPathtracer() { return &_optixPathtracer; }
    SLbool             draw3DOptixRT();
    SLbool             draw3DOptixPT();
    void               startOptixRaytracing(SLint maxDepth);
    void               startOptixPathtracing(SLint maxDepth, SLint samples);
#endif

    static const SLint LONGTOUCH_MS; //!< Milliseconds duration of a long touch event

protected:
    SLScene*       _s;               //!< Pointer scene observed by this scene view
    SLCamera*      _camera;          //!< Pointer to the _active camera
    SLCamera       _sceneViewCamera; //!< Default camera for this SceneView (default cam not in scenegraph)
    SLUiInterface* _gui = nullptr;   //!< ImGui instance
    SLSkybox*      _skybox;          //!< pointer to skybox
    SLNodeStats    _stats2D;         //!< Statistic numbers for 2D nodes
    SLNodeStats    _stats3D;         //!< Statistic numbers for 3D nodes
    SLbool         _gotPainted;      //!< flag if this sceneview got painted
    SLRenderType   _renderType;      //!< rendering type (GL,RT,PT)

    SLbool     _doDepthTest;      //!< Flag if depth test is turned on
    SLbool     _doMultiSampling;  //!< Flag if multisampling is on
    SLbool     _doFrustumCulling; //!< Flag if view frustum culling is on
    SLbool     _doWaitOnIdle;     //!< Flag for Event waiting
    SLbool     _isFirstFrame;     //!< Flag if it is the first frame rendering
    SLDrawBits _drawBits;         //!< Sceneview level drawing flags

    SLfloat _shadowMapTimeMS; //!< time for drawing the shadow maps in ms
    SLfloat _cullTimeMS;      //!< time for culling in ms
    SLfloat _draw3DTimeMS;    //!< time for 3D drawing in ms
    SLfloat _draw2DTimeMS;    //!< time for 2D drawing in ms

    SLbool  _mouseDownL; //!< Flag if left mouse button is pressed
    SLbool  _mouseDownR; //!< Flag if right mouse button is pressed
    SLbool  _mouseDownM; //!< Flag if middle mouse button is pressed
    SLKey   _mouseMod;   //!< mouse modifier key on key down
    SLint   _touchDowns; //!< finger touch down count
    SLVec2i _touch[3];   //!< up to 3 finger touch coordinates

    SLGLVertexArrayExt _vaoTouch;  //!< Buffer for touch pos. rendering
    SLGLVertexArrayExt _vaoCursor; //!< Virtual cursor for stereo rendering

    SLint           _scrW;                //!< Screen width in pixels
    SLint           _scrH;                //!< Screen height in pixels
    SLint           _scrWdiv2;            //!< Screen half width in pixels
    SLint           _scrHdiv2;            //!< Screen half height in pixels
    SLfloat         _scrWdivH;            //!< Screen side aspect ratio
    SLfloat         _scr2fbX{1.0f};       //!< Horizontal screen to framebuffer ratio
    SLfloat         _scr2fbY{1.0f};       //!< Vertical screen to framebuffer ratio
    int             _dpi;                 //!< dots per inch of screen
    SLVec2i         _viewportRatio;       //!< ratio of viewport
    SLViewportAlign _viewportAlign;       //!< alignment of viewport
    SLRecti         _viewportRect;        //!< rectangle of viewport
    SLbool          _viewportSameAsVideo; //!< Adapt viewport aspect to the input video

    SLGLOculusFB _oculusFB; //!< Oculus framebuffer

#ifdef SL_RENDER_BY_MATERIAL
    SLVNode _nodesText2D;      //!< Vector of 2D text nodes that get blended (needs redesign)
    SLVNode _nodesText3D;      //!< Vector of 3D text nodes that get blended (needs redesign)
#else
    SLVNode _nodesVisible;   //!< Vector of all visible 3D nodes
    SLVNode _nodesVisible2D; //!< Vector of all visible 2D nodes drawn in orthographic projection
    SLVNode _nodesBlended;   //!< Vector of visible and blended nodes
#endif
    SLVNode _nodesOverdrawn; //!< Vector of helper nodes drawn over all others

    SLRaytracer                     _raytracer;  //!< Whitted style raytracer
    SLbool                          _stopRT;     //!< Flag to stop the RT
    SLPathtracer                    _pathtracer; //!< Pathtracer
    SLbool                          _stopPT;     //!< Flag to stop the PT
    std::unique_ptr<SLGLConetracer> _conetracer; //!< Conetracer CT

#ifdef SL_HAS_OPTIX
    SLOptixRaytracer  _optixRaytracer;  //!< Whitted style raytracer with Optix
    SLbool            _stopOptixRT;     //!< Flag to stop the Optix RT
    SLOptixPathtracer _optixPathtracer; //!< Path tracer with Optix
    SLbool            _stopOptixPT;     //!< Flag to stop the Optix PT
#endif

    SLInputManager& _inputManager;

    AvgFloat _shadowMapTimesMS; //!< Averaged time for drawing the shadow maps in ms
    AvgFloat _cullTimesMS;      //!< Averaged time for culling in ms
    AvgFloat _draw3DTimesMS;    //!< Averaged time for 3D drawing in ms
    AvgFloat _draw2DTimesMS;    //!< Averaged time for 2D drawing in ms
};
//-----------------------------------------------------------------------------
#endif
