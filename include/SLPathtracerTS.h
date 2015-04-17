//#############################################################################
//  File:      SLPhotonMapper.h
//  Author:    Thomas Schneiter, Marcus Hudritsch
//  Date:      September 2011 (HS11)
//  Copyright (c): 2002-2013 Marcus Hudritsch
//             This software is provide under the GNU General Public License
//             Please visit: http://opensource.org/licenses/GPL-3.0
//#############################################################################

#ifndef SLPATHTRACER_H
#define SLPATHTRACER_H

#include <stdafx.h>
#include <SLRaytracer.h>

class SLScene;
class SLSceneView;
class SLRay;
class SLMaterial;
class SLCamera;

//-----------------------------------------------------------------------------
//! Ray tracing state
typedef enum
{  ptReady,    // PT is ready to start
   ptBusy,     // PT is running
   ptFinished, // PT is finished
   ptMoveGL,   // PT is finished and GL camera is moving
} SLStatePT;

//-----------------------------------------------------------------------------
// callback function typedef for ray tracing gui window update
typedef SLbool (SL_STDCALL *cbPTWndUpdate)(void);

class SLPathtracer : public SLRaytracer
{  public:           
                            SLPathtracer ();
                            ~SLPathtracer ();
            
            // classic ray tracer functions
            SLbool          render         ();
            void            renderSlices   (const bool isMainThread, SLint currentSample);
            SLCol4f         trace          (SLRay* ray, SLbool em);
            SLCol4f         sampleLights   (SLRay* ray);

            SLCol4f         shade         (SLRay* ray, SLCol4f* mat);

            // Setters
            void            state          (SLStatePT state) {if (_state!=ptBusy) _state=state;}
            void            continuous     (SLbool cont) {_continuous = cont; state(ptReady);}
            void            maxDepth       (SLint depth) {_maxDepth = depth; state(ptReady);}
            void            samples        (SLint samples) {_samples = samples; state(ptReady);}
            void            volumeRendering(SLbool volumeRendering) { _volumeRendering = volumeRendering; state(ptReady); }

            // Getters
            SLStatePT       state          () {return _state;}
            SLbool          continuous     () {return _continuous;}
            SLint           maxDepth       () {return _maxDepth;}
            SLint           samples        () {return _samples;}
            SLbool          volumeRendering() {return _volumeRendering;}

            // Render target image
            void            createImage    (SLint width, SLint height);
            void            prepareImage   ();
            void            renderImage    ();
            void            saveImage      ();

            // Callback routine
            cbPTWndUpdate   guiPTWndUpdate;

   private:
            SLGLState*      _stateGL;      //!< Pointer to the global state
            SLStatePT       _state;        //!< state of PT
            SLCamera*       _cam;          //!< shortcut to the camera
            SLfloat         _renderSec;    //!< Rendering time in seconds
            SLbool          _continuous;   //!< if true state goes into ready again
            SLint           _pcRendered;   //!< % rendered
            SLstring        _infoText;     //!< Original info string
            SLCol4f         _infoColor;    //!< Original info string color
            SLint           _maxDepth;     //!< Max. allowed recursion depth
            SLint           _samples;      //!< Samples per pixel
            SLbool          _volumeRendering; //!< Samples per pixel

            SLfloat         _pxSize;       //!< Pixel size
            SLVec3f         _EYE;          //!< Camera position
            SLVec3f         _LA, _LU, _LR; //!< Camera lookat, lookup, lookright
            SLVec3f         _BL;           //!< Bottom left vector
            SLuint          _next;         //!< next index to render RT

            SLGLBuffer      _bufP;         //!< Buffer object for vertex positions
            SLGLBuffer      _bufT;         //!< Buffer object for vertex texcoords
            SLGLBuffer      _bufI;         //!< Buffer object for vertex indexes

            SLint           _numThreads;  //!< Num. of threads used for PT

            default_random_engine               _generator;
            uniform_real_distribution<double>   _uniformRandom;

            // variables for pathtracing
            SLfloat         _gamma;        //!< gamma correction
};
//-----------------------------------------------------------------------------
#endif
