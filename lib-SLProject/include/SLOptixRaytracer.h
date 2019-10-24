//#############################################################################
//  File:      SLOptixRaytracer.h
//  Author:    Nic Dorner
//  Date:      October 2019
//  Copyright: Nic Dorner
//             This software is provide under the GNU General Public License
//             Please visit: http://opensource.org/licenses/GPL-3.0
//#############################################################################

#ifndef SLPROJECT_SLOPTIXRAYTRACER_H
#define SLPROJECT_SLOPTIXRAYTRACER_H
#include <SLEventHandler.h>
#include <SLGLTexture.h>

class SLScene;
class SLSceneView;
class SLRay;
class SLMaterial;
class SLCamera;

//-----------------------------------------------------------------------------
//! SLOptixRaytracer hold all the methods for Whitted style Ray Tracing.
class SLOptixRaytracer : public SLGLTexture
        , public SLEventHandler
{
public:
    SLOptixRaytracer();
    ~SLOptixRaytracer() override;

    // setup raytracer
    void loadPrograms();
    void setupScene(SLSceneView* sv);

    // ray tracer functions
    SLbool  renderClassic();
    void    finishBeforeUpdate();

    // Setters
    void state(SLRTState state)
    {
        if (_state != rtBusy) _state = state;
    }
    void maxDepth(SLint depth)
    {
        _maxDepth = depth;
        state(rtReady);
    }

    // Getters
    SLRTState state() const { return _state; }
    SLint     maxDepth() const { return _maxDepth; }

    // Render target image
    void prepareImage();
    void renderImage();
    void saveImage();

protected:
    SLSceneView* _sv;            //!< Parent sceneview
    SLRTState    _state;         //!< RT state;
    SLCamera*    _cam;           //!< shortcut to the camera
    SLint        _maxDepth;      //!< Max. allowed recursion depth
    SLfloat      _renderSec;     //!< Rendering time in seconds

    SLfloat     _pxSize;       //!< Pixel size
    SLVec3f     _EYE;          //!< Camera position
    SLVec3f     _LA, _LU, _LR; //!< Camera lookat, lookup, lookright
    SLVec3f     _BL;           //!< Bottom left vector
    atomic<int> _next;         //!< next index to render RT
};
#endif //SLPROJECT_SLOPTIXRAYTRACER_H
