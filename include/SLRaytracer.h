//#############################################################################
//  File:      SLRaytracer.h
//  Author:    Marcus Hudritsch
//  Date:      July 2014
//  Copyright: Marcus Hudritsch
//             This software is provide under the GNU General Public License
//             Please visit: http://opensource.org/licenses/GPL-3.0
//#############################################################################

#ifndef SLRAYTRACER_H
#define SLRAYTRACER_H

#include <stdafx.h>
#include <SLGLTexture.h>
#include <SLGLBuffer.h>
#include <SLEventHandler.h>

class SLScene;
class SLSceneView;
class SLRay;
class SLMaterial;
class SLCamera;

//-----------------------------------------------------------------------------
//! Ray tracing state
typedef enum
{
    rtReady,       // RT is ready to start
    rtBusy,        // RT is running
    rtFinished,    // RT is finished
    rtMoveGL       // RT is finished and GL camera is moving
} SLRTState;
//-----------------------------------------------------------------------------
//! Pixel index struct used in anti aliasing in ray tracing
struct SLRTAAPixel
{
    SLRTAAPixel(SLushort X=0, SLushort Y=0) {x=X;y=Y;}
    SLushort x;      //!< Unsigned short x-pixel index
    SLushort y;      //!< Unsigned short x-pixel index
};
typedef vector<SLRTAAPixel> SLVPixel;
//-----------------------------------------------------------------------------
//! SLRaytracer hold all the methods for Whitted style Ray Tracing.
/*!      
SLRaytracer implements the methods render, eyeToPixel, trace and shade for
classic Whitted style Ray Tracing. This class is a friend class of SLScene and
can access via the pointer _s all members of SLScene. The scene traversal for
the ray intersection tests is done within the intersection method of all nodes. 
*/
class SLRaytracer: public SLGLTexture, public SLEventHandler
{
    public:
                        SLRaytracer ();
            virtual    ~SLRaytracer ();
            
            // ray tracer functions
            SLbool      renderClassic   (SLSceneView* sv);
            SLbool      renderDistrib   (SLSceneView* sv);
            void        renderSlices    (const bool isMainThread);
            void        renderSlicesMS  (const bool isMainThread);
            SLCol4f     traceClassic    (SLRay* ray);
            SLCol4f     traceDistrib    (SLRay* ray);
            SLCol4f     shade           (SLRay* ray);
            void        sampleAAPixels  (const bool isMainThread);
            
            // additional ray tracer functions
            void        setPrimaryRay   (SLfloat x, SLfloat y, SLRay* primaryRay); 
            void        getAAPixels     ();
            SLCol4f     fogBlend        (SLfloat z, SLCol4f color);
            void        printStats      (SLfloat sec);
            void        initStats       (SLint depth);
            
            // Setters
            void        state           (SLRTState state) {if (_state!=rtBusy) _state=state;}
            void        maxDepth        (SLint depth)     {_maxDepth = depth; state(rtReady);}
            void        distributed     (SLbool distrib)  {_distributed = distrib;}
            void        continuous      (SLbool cont)     {_continuous = cont; state(rtReady);}
            void        aaSamples       (SLint samples)   {_aaSamples = samples; state(rtReady);}
            
            // Getters
            SLRTState   state           () const {return _state;}
            SLint       maxDepth        () const {return _maxDepth;}
            SLbool      distributed     () const {return _distributed;}
            SLbool      continuous      () const {return _continuous;}
            SLint       aaSamples       () const {return _aaSamples;}
            SLint       numThreads      () const {return _numThreads;}
            SLint       pcRendered      () const {return _pcRendered;}
            SLfloat     aaThreshold     () const {return _aaThreshold;}
            SLfloat     renderSec       () const {return _renderSec;}
            
            // Render target image
            void        prepareImage    ();
            void        renderImage     ();
            void        saveImage       ();
                        
   protected:
            SLSceneView* _sv;           //!< Parent sceneview
            SLGLState*  _stateGL;       //!< Pointer to the global state
            SLRTState   _state;         //!< RT state;
            SLCamera*   _cam;           //!< shortcut to the camera
            SLint       _maxDepth;      //!< Max. allowed recursion depth
            SLbool      _continuous;    //!< if true state goes into ready again
            SLbool      _distributed;   //!< Flag for parallel distributed RT
            SLint       _pcRendered;    //!< % rendered
            SLfloat     _renderSec;     //!< Rendering time in seconds
            SLstring    _infoText;      //!< Original info string
            SLCol4f     _infoColor;     //!< Original info string color

            SLfloat     _pxSize;        //!< Pixel size
            SLVec3f     _EYE;           //!< Camera position
            SLVec3f     _LA, _LU, _LR;  //!< Camera lookat, lookup, lookright
            SLVec3f     _BL;            //!< Bottom left vector
            SLuint      _next;          //!< next index to render RT
            SLVPixel    _aaPixels;      //!< Vector for antialiasing pixels

            // variables for distributed ray tracing
            SLfloat     _aaThreshold;   //!< threshold for anti aliasing
            SLint       _aaSamples;     //!< SQRT of uneven num. of AA samples
            SLint       _numThreads;    //!< Num. of threads used for RT
};
//-----------------------------------------------------------------------------
#endif
