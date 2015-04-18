//#############################################################################
//  File:      SLRaytracer.cpp
//  Author:    Marcus Hudritsch
//  Date:      July 2014
//  Copyright: Marcus Hudritsch
//             This software is provide under the GNU General Public License
//             Please visit: http://opensource.org/licenses/GPL-3.0
//#############################################################################

#include <stdafx.h>           // precompiled headers
#ifdef SL_MEMLEAKDETECT       // set in SL.h for debug config only
#include <debug_new.h>        // memory leak detector
#endif

using namespace std::placeholders;
using namespace std::chrono;

#include <SLRay.h>
#include <SLRaytracer.h>
#include <SLCamera.h>
#include <SLSceneView.h>
#include <SLLightSphere.h>
#include <SLLightRect.h>
#include <SLLight.h>
#include <SLNode.h>
#include <SLMesh.h>
#include <SLText.h>
#include <SLGLTexture.h>
#include <SLSamples2D.h>
#include <SLGLProgram.h>

//-----------------------------------------------------------------------------
SLRaytracer::SLRaytracer()
{  
    name("myCoolRaytracer");
   
    _state = rtReady;
    _distributed = true;
    _maxDepth = 5;
    _aaThreshold = 0.3f; // = 10% color difference
    _aaSamples = 3;
   
    // set texture properties
    _min_filter   = GL_NEAREST;
    _mag_filter   = GL_NEAREST;
    _wrap_s       = GL_CLAMP_TO_EDGE;
    _wrap_t       = GL_CLAMP_TO_EDGE;
    _resizeToPow2 = false;
   
    _numThreads = 1;
    _continuous = false;
    _distributed = true;
}
//-----------------------------------------------------------------------------
SLRaytracer::~SLRaytracer()
{  
    SL_LOG("~SLRaytracer\n");
}
//-----------------------------------------------------------------------------
/*!
This is the main rendering method for the classic ray tracing. It loops over all 
lines and pixels and determines for each pixel a color with a partly global 
illumination calculation.
*/
SLbool SLRaytracer::renderClassic(SLSceneView* sv)
{
    _sv = sv;
    _state = rtBusy;                    // From here we state the RT as busy
    _stateGL = SLGLState::getInstance();// OpenGL state shortcut
    _numThreads = 1;                    // No. of threads
    _pcRendered = 0;                    // % rendered
    _renderSec = 0.0f;                  // reset time
    _infoText  = SLScene::current->info(_sv)->text();  // keep original info string
    _infoColor = SLScene::current->info(_sv)->color(); // keep original info color

    initStats(_maxDepth);               // init statistics
    prepareImage();                     // Setup image & precalculations

    // Measure time 
    double t1 = SLScene::current->timeSec();
    double tStart = t1;

    for (SLuint x=0; x<_images[0]->width(); ++x)
    {   for (SLuint y=0; y<_images[0]->height(); ++y)
        {
            SLRay primaryRay;
            setPrimaryRay((SLfloat)x, (SLfloat)y, &primaryRay);

            //////////////////////////////////////////
            SLCol4f color = traceClassic(&primaryRay);
            //////////////////////////////////////////

            _images[0]->setPixeliRGB(x, y, color);

            SLRay::avgDepth += SLRay::depthReached;
            SLRay::maxDepthReached = SL_max(SLRay::depthReached, SLRay::maxDepthReached);
        }

        // Update image after 500 ms
        double t2 = SLScene::current->timeSec();
        if (t2-t1 > 0.5)
        {   _pcRendered = (SLint)((SLfloat)x/(SLfloat)_images[0]->width()*100);
            _sv->onWndUpdate();
            t1 = SLScene::current->timeSec();
        }
    }

    _renderSec = (SLfloat)(SLScene::current->timeSec() - tStart);
    _pcRendered = 100;

    if (_continuous)
        _state = rtReady;
    else
    {   _state = rtFinished;
        printStats(_renderSec);
    }
    return true;
}
//-----------------------------------------------------------------------------
/*!
This is the main rendering method for paralllel and distributed ray tracing.
*/
SLbool SLRaytracer::renderDistrib(SLSceneView* sv)
{
    _sv = sv;
    _state = rtBusy;                    // From here we state the RT as busy
    _stateGL = SLGLState::getInstance();// OpenGL state shortcut
    _numThreads = 1;                    // No. of threads
    _pcRendered = 0;                    // % rendered
    _renderSec = 0.0f;                  // reset time
    _infoText  = SLScene::current->info(_sv)->text();  // keep original info string
    _infoColor = SLScene::current->info(_sv)->color(); // keep original info color

    initStats(_maxDepth);               // init statistics
    prepareImage();                     // Setup image & precalculations
   
    // Measure time 
    double t1 = SLScene::current->timeSec();
   
    // Bind render functions to be called multithreaded
    auto sampleAAPixelsFunction = bind(&SLRaytracer::sampleAAPixels, this, _1);
    auto renderSlicesFunction   = _cam->lensSamples()->samples() == 1 ? 
                                  bind(&SLRaytracer::renderSlices, this, _1) : 
                                  bind(&SLRaytracer::renderSlicesMS, this, _1);


    // Do multithreading only in release config
    #ifdef _DEBUG
    _numThreads = 1;
    #else
    _numThreads = thread::hardware_concurrency();
    #endif

    // Render image without antialiasing
    {   vector<thread> threads; // vector for additional threads  
        _next = 0;              // init _next=0. _next should be atomic

        // Start additional threads on the renderSlices function
        for (int t=0; t<_numThreads-1; t++)
            threads.push_back(thread(renderSlicesFunction, false));

        // Do the same work in the main thread
        renderSlicesFunction(true);

        // Wait for the other threads to finish
        for(auto& thread : threads) thread.join();
    }

    // Do anti-aliasing w. contrast compare in a 2nd. pass
    if (!_continuous && _aaSamples > 1 && _cam->lensSamples()->samples() == 1)
    {
        getAAPixels();          // Fills in the AA pixels by contrast
        vector<thread> threads; // vector for additional threads
        _next = 0;              // init _next=0. _next should be atomic

        // Start additional threads on the sampleAAPixelFunction function
        for (int t=0; t<_numThreads-1; t++)
            threads.push_back(thread(sampleAAPixelsFunction, false));

        // Do the same work in the main thread
        sampleAAPixelsFunction(true);

        // Wait for the other threads to finish
        for(auto& thread : threads) thread.join();
    }
   
    _renderSec = (SLfloat)(SLScene::current->timeSec() - t1);
    _pcRendered = 100;

    if (_continuous)
        _state = rtReady;
    else
    {   _state = rtFinished;
        printStats(_renderSec);
    }
    return true;
}
//-----------------------------------------------------------------------------
/*!
Renders slices of 4 columns until the full width of the image is rendered. This
method can be called as a function by multiple threads.
The _next index is used and incremented by every thread. So it should be locked
or an atomic index. I prefer not protecting it because it's faster. If the
increment is not done proberly some pixels may get raytraced twice. Only the
main thread is allowed to call a repaint of the image.
*/
void SLRaytracer::renderSlices(const bool isMainThread)
{
    // Time points
    double t1 = 0;

    while (_next < _images[0]->width())
    {
        const SLint minX = _next;

        // The next line should be theoretically atomic
        _next += 4;

        for (SLint x=minX; x<minX+4; ++x)
        {
            for (SLuint y=0; y<_images[0]->height(); ++y)
            {
                SLRay primaryRay;
                setPrimaryRay((SLfloat)x, (SLfloat)y, &primaryRay);

                //////////////////////////////////////////
                SLCol4f color = traceDistrib(&primaryRay);
                //////////////////////////////////////////

                _images[0]->setPixeliRGB(x, y, color);

                #if _DEBUG
                SLRay::avgDepth += SLRay::depthReached;
                SLRay::maxDepthReached = SL_max(SLRay::depthReached, SLRay::maxDepthReached);
                #endif
            }

            // Update image after 500 ms
            if (isMainThread && !_continuous)
            {   if (SLScene::current->timeSec() - t1 > 0.5)
                {   _pcRendered = (SLint)((SLfloat)x/(SLfloat)_images[0]->width()*100);
                    if (_aaSamples > 0) _pcRendered /= 2;
                    _sv->onWndUpdate();
                    t1 = SLScene::current->timeSec();
                }
            }
        }
    }
}
//-----------------------------------------------------------------------------
/*!
Renders slices of 4 columns multisampled until the full width of the image is 
rendered. Every pixel is multisampled for depth of field lens sampling. This
method can be called as a function by multiple threads.
The _next index is used and incremented by every thread. So it should be locked
or an atomic index. I prefer not protecting it because it's faster. If the
increment is not done proberly some pixels may get raytraced twice. Only the
main thread is allowed to call a repaint of the image.
*/
void SLRaytracer::renderSlicesMS(const bool isMainThread)
{
    // Time points
    double t1 = 0;

    // lens sampling constants
    SLVec3f lensRadiusX = _LR*(_cam->lensDiameter()*0.5f);
    SLVec3f lensRadiusY = _LU*(_cam->lensDiameter()*0.5f);

    while (_next < _images[0]->width())
    {
        const SLint minX = _next;
        _next += 4;

        for (SLint x=minX; x<minX+4; ++x)
        {
            for (SLuint y=0; y<_images[0]->height(); ++y)
            {
                // focal point is single shot primary dir
                SLVec3f primaryDir(_BL + _pxSize*((SLfloat)x*_LR + (SLfloat)y*_LU));
                SLVec3f FP = _EYE + primaryDir;
                SLCol4f color(SLCol4f::BLACK);
            
                // Loop over radius r and angle phi of lens
                for (SLint iR=_cam->lensSamples()->samplesX()-1; iR>=0; --iR)
                {   for (SLint iPhi=_cam->lensSamples()->samplesY()-1; iPhi>=0; --iPhi)
                    {   
                        SLVec2f discPos(_cam->lensSamples()->point(iR,iPhi));
                  
                        // calculate lensposition out of disc position
                        SLVec3f lensPos(_EYE + discPos.x*lensRadiusX + discPos.y*lensRadiusY);
                        SLVec3f lensToFP(FP-lensPos);
                        lensToFP.normalize();
                        SLRay primaryRay(lensPos, lensToFP, (SLfloat)x, (SLfloat)y);
                  
                        ///////////////////////////////////
                        color += traceDistrib(&primaryRay);
                        ///////////////////////////////////
                  
                        SLRay::avgDepth += SLRay::depthReached;
                        SLRay::maxDepthReached = SL_max(SLRay::depthReached, SLRay::maxDepthReached);   
                    }
                }
                color /= (SLfloat)_cam->lensSamples()->samples();
                _images[0]->setPixeliRGB(x, y, color);
         

                #if _DEBUG
                SLRay::avgDepth += SLRay::depthReached;
                SLRay::maxDepthReached = SL_max(SLRay::depthReached, SLRay::maxDepthReached);
                #endif
            }

            if (isMainThread && !_continuous)
            {   if (SLScene::current->timeSec() - t1 > 0.5)
                {   _sv->onWndUpdate();
                    t1 = SLScene::current->timeSec();
                }
            }
        }
    }
}
//-----------------------------------------------------------------------------
/*!
This method is the classic recursive ray tracing method that checks the scene
for intersection. If the ray hits an object the local color is calculated and
if the material is reflective and/or transparent new rays are created and
passed to this trace method again. If no object got intersected the
background color is return.
*/
SLCol4f SLRaytracer::traceClassic(SLRay* ray)
{
    SLScene* s = SLScene::current;
    SLCol4f color(s->backColor());

    s->root3D()->hitRec(ray);

    if (ray->length < FLT_MAX)
    {
        color = shade(ray);

        if (ray->depth < SLRay::maxDepth && ray->contrib > SLRay::minContrib)
        {
            if (ray->hitMat->kt())
            {   SLRay refracted;
                ray->refract(&refracted);
                color += ray->hitMat->kt() * traceClassic(&refracted);
            }
            if (ray->hitMat->kr())
            {   SLRay reflected;
                ray->reflect(&reflected);
                color += ray->hitMat->kr() * traceClassic(&reflected);
            }
        }
    }

    color.clampMinMax(0,1);
    return color;
}
//-----------------------------------------------------------------------------
//! Set the parameters of a primary ray for a pixel position at x, y.
void SLRaytracer::setPrimaryRay(SLfloat x, SLfloat y, SLRay* primaryRay)
{   
    primaryRay->x = x;
    primaryRay->y = y;

    // calculate ray from eye to pixel (See also prepareImage())
    if (_cam->projection() == monoOrthographic)
    {   primaryRay->setDir(_LA);
        primaryRay->origin = _BL + _pxSize*((SLfloat)x*_LR + (SLfloat)y*_LU);
    } else
    {   SLVec3f primaryDir(_BL + _pxSize*((SLfloat)x*_LR + (SLfloat)y*_LU));
        primaryDir.normalize();
        primaryRay->setDir(primaryDir);
        primaryRay->origin = _EYE;
    }
}
//-----------------------------------------------------------------------------
/*!
This method is the recursive ray tracing method that checks the scene
for intersection. If the ray hits an object the local color is calculated and 
if the material is reflective and/or transparent new rays are created and 
passed to this trace method again. If no object got intersected the 
background color is return. The distributed extension includes the Fresnel
appoximation.
*/
SLCol4f SLRaytracer::traceDistrib(SLRay* ray)
{  
    SLScene* s = SLScene::current;
    SLCol4f color(s->backColor());
   
    s->root3D()->hitRec(ray);
   
    if (ray->length < FLT_MAX)
    {  
        color = shade(ray);
      
        if (ray->depth < SLRay::maxDepth && ray->contrib > SLRay::minContrib)
        {  
            if (ray->hitMat->kt())
            {   SLRay refracted, reflected;
                ray->refract(&refracted);
                ray->reflect(&reflected);
                SLCol4f refrCol = traceDistrib(&refracted);
                SLCol4f reflCol = traceDistrib(&reflected);
            
                // Mix refr. & refl. color w. Schlick's Fresnel aproximation
                SLfloat F0 = ray->hitMat->kr();
                SLfloat theta = -(ray->dir * ray->hitNormal);
                SLfloat F_theta = F0 + (1-F0) * pow(1-theta, 5);
                color += refrCol*(1-F_theta) + reflCol*F_theta;
            } else
            {   if (ray->hitMat->kr())
                {   SLRay reflected;
                    ray->reflect(&reflected);
                    color += ray->hitMat->kr() * traceDistrib(&reflected);
                }
            }
        }
    }
   
    if (_stateGL->fogIsOn) 
        color = fogBlend(ray->length,color);
   
    color.clampMinMax(0,1);
    return color;
}
//-----------------------------------------------------------------------------
/*!
This method calculates the local illumination at the rays intersection point. 
It uses the OpenGL local light model where the color is calculated as 
follows:
color = material emission + 
        global ambient light scaled by the material's ambient color + 
        ambient, diffuse, and specular contributions from all lights, 
        properly attenuated
*/
SLCol4f SLRaytracer::shade(SLRay* ray)
{  
    SLScene*    s = SLScene::current;
    SLCol4f     localColor = SLCol4f::BLACK;
    SLMaterial* mat = ray->hitMat;
    SLVGLTexture& texture = mat->textures();
    SLVec3f     L,N,H;
    SLfloat     lightDist, LdN, NdH, df, sf, spotEffect, att, lighted = 0.0f;
    SLCol4f     amdi, spec;
    SLCol4f     localSpec(0,0,0,1);
   
    // Don't shade lights. Only take emissive color as material 
    if (typeid(*ray->hitNode)==typeid(SLLightSphere) || 
        typeid(*ray->hitNode)==typeid(SLLightRect))
    {   localColor = mat->emission();
        return localColor;
    } 

    localColor = mat->emission() + (mat->ambient()&s->globalAmbiLight());
  
    ray->hitMesh->preShade(ray);
      
    for (SLint i=0; i<s->lights().size(); ++i) 
    {  SLLight* light = s->lights()[i];
   
        if (light && light->on())
        {              
            // calculate light vector L and distance to light
            N.set(ray->hitNormal);
            L.sub(light->positionWS(), ray->hitPoint);
            lightDist = L.length();
            L/=lightDist; 
            LdN = L.dot(N);

            // check shadow ray if hit point is towards the light
            lighted = (LdN>0) ? light->shadowTest(ray, L, lightDist) : 0;
         
            // calculate the ambient part
            amdi = light->ambient() & mat->ambient();
            spec.set(0,0,0);
      
            // calculate spot effect if light is a spotlight
            if (lighted > 0.0f && light->spotCutoff() < 180.0f)
            {  SLfloat LdS = SL_max(-L.dot(light->spotDirWS()), 0.0f);
         
            // check if point is in spot cone
            if (LdS > light->spotCosCut())
            {  spotEffect = pow(LdS, (SLfloat)light->spotExponent());
            } else 
            {   lighted = 0.0f;
                spotEffect = 0.0f;
            }
            } else spotEffect = 1.0f;
         
            // calculate local illumination only if point is not shaded
            if (lighted > 0.0f) 
            {   H.sub(L,ray->dir); // half vector between light & eye
                H.normalize();
                df   = SL_max(LdN     , 0.0f);           // diffuse factor
                NdH  = SL_max(N.dot(H), 0.0f);
                sf = pow(NdH, (SLfloat)mat->shininess()); // specular factor
         
                amdi += lighted * df * light->diffuse() & mat->diffuse();
                spec  = lighted * sf * light->specular()& mat->specular();
            }
      
            // apply attenuation and spot effect
            att = light->attenuation(lightDist) * spotEffect;
            localColor += att * amdi;
            localSpec  += att * spec;
        }
    }

    if (texture.size()) 
    {   localColor &= ray->hitTexCol;    // componentwise multiply
        localColor += localSpec;         // add afterwards the specular component
    } else localColor += localSpec; 
         
    localColor.clampMinMax(0, 1); 
    return localColor;  
}
//-----------------------------------------------------------------------------
/*!
This method fills the pixels into the vector pix that need to be supsampled
because the contrast to its left and/or above neighbour is above a threshold.
*/
void SLRaytracer::getAAPixels()
{
    SLCol4f  color, colorLeft, colorUp;    // pixel colors to be compared
    SLbool*  gotSampled = new SLbool[_images[0]->width()];// Flags if above pixel got sampled
    SLbool   isSubsampled = false;         // Flag if pixel got subsampled

    // Nothing got sampled at beginning
    for (SLuint x=0; x<_images[0]->width(); ++x) gotSampled[x] = false;

    // Loop through all pixels & add the pixel that have to be subsampled
    _aaPixels.clear();
    for (SLuint y=0; y<_images[0]->height(); ++y)
    {  for (SLuint x=0; x<_images[0]->width(); ++x)
        {
            color = _images[0]->getPixeli(x, y);
            isSubsampled = false;
            if (x>0)
            {  colorLeft = _images[0]->getPixeli(x-1, y);
                if (color.diffRGB(colorLeft) > _aaThreshold)
                {   if (!gotSampled[x-1])
                    {   _aaPixels.push_back(SLRTAAPixel(x-1,y));
                        gotSampled[x-1] = true;
                    }
                    _aaPixels.push_back(SLRTAAPixel(x,y));
                    isSubsampled = true;
                }
            }
            if (y>0)
            {   colorUp = _images[0]->getPixeli(x, y-1);
                if(color.diffRGB(colorUp) > _aaThreshold)
                {   if (!gotSampled[x]) _aaPixels.push_back(SLRTAAPixel(x,y-1));
                    if (!isSubsampled)
                    {   _aaPixels.push_back(SLRTAAPixel(x,y));
                        isSubsampled = true;
                    }
                }
            }
            gotSampled[x] = isSubsampled;
        }
    }
    delete[] gotSampled;
    SLRay::subsampledPixels = (SLint)_aaPixels.size();
}
//-----------------------------------------------------------------------------
/*!
SLRaytracer::sampleAAPixels does the subsampling of the pixels that need to be
antialiased. See also getAAPixels. This routine can be called by multiple
threads.
The _next index is used and incremented by every thread. So it should be locked
or an atomic index. I prefer not protecting it because it's faster. If the
increment is not done proberly some pixels may get raytraced twice. Only the
main thread is allowed to call a repaint of the image.
*/
void SLRaytracer::sampleAAPixels(const bool isMainThread)
{  
    assert(_aaSamples%2==1 && "subSample: maskSize must be uneven");
    double t1 = 0, t2 = 0;

    while (_next < _aaPixels.size())
    {
        SLuint mini = _next;
        _next += 4;
      
        for (SLuint i = mini; i< mini+4 && i<_aaPixels.size(); ++i)
        {   SLuint x = _aaPixels[i].x;
            SLuint y = _aaPixels[i].y;
            SLCol4f centerColor = _images[0]->getPixeli(x, y);
            SLint   centerIndex = _aaSamples>>1;
            SLfloat f = 1.0f/(SLfloat)_aaSamples;
            SLCol4f color(0,0,0);
            SLfloat xpos = x - centerIndex*f;
            SLfloat ypos = y - centerIndex*f;
            SLfloat samples = (SLfloat)_aaSamples*_aaSamples;

            // Loop regularly over the float pixel
            for (SLint j=0; j<_aaSamples; ++j)
            {   for (SLint i=0; i<_aaSamples; ++i)
                {   if (i==centerIndex && j==centerIndex) 
                        color += centerColor; // don't shoot for center position
                    else 
                    {   
                        SLRay primaryRay;
                        setPrimaryRay(xpos+i*f, ypos+i*f, &primaryRay);

                        color += traceDistrib(&primaryRay);
                    }
                }
                ypos += f;
            }
            #if _DEBUG
            SLRay::subsampledRays += (SLuint)samples;
            #endif
            color /= samples;
            _images[0]->setPixeliRGB(x, y, color);
        }

        if (isMainThread && !_continuous)
        {   t2 = SLScene::current->timeSec();
            if (t2-t1 > 0.5)
            {   _pcRendered = 50 + (SLint)((SLfloat)_next/(SLfloat)_aaPixels.size()*50);
                _sv->onWndUpdate();
                t1 = SLScene::current->timeSec();
            }
        }
    }
}
//-----------------------------------------------------------------------------
/*! 
fogBlend: Blends the a fog color to the passed color according to to OpenGL fog 
calculation. See OpenGL docs for more information on fog properties.
*/
SLCol4f SLRaytracer::fogBlend(SLfloat z, SLCol4f color)
{  
    SLfloat f=0.0f;
    if (z > _sv->_camera->clipFar()) z = _sv->_camera->clipFar();
    switch (_stateGL->fogMode)
    {   case 0:  f = (_stateGL->fogDistEnd-z)/
                     (_stateGL->fogDistEnd-_stateGL->fogDistStart); break;
        case 1:  f = exp(-_stateGL->fogDensity*z); break;
        default: f = exp(-_stateGL->fogDensity*z*_stateGL->fogDensity*z); break;
    }
    color = f*color + (1-f)*_stateGL->fogColor;
    color.clampMinMax(0, 1);
    return color;   
}
//-----------------------------------------------------------------------------
/*!
Initialises the statistic variables in SLRay to zero
*/
void SLRaytracer::initStats(SLint depth)
{  
    SLRay::maxDepth = (depth) ? depth : SL_MAXTRACE;
    SLRay::reflectedRays = 0;
    SLRay::refractedRays = 0;
    SLRay::shadowRays = 0;
    SLRay::subsampledRays = 0;
    SLRay::subsampledPixels = 0;
    SLRay::tests = 0;
    SLRay::intersections = 0;
    SLRay::maxDepthReached = 0;
    SLRay::avgDepth = 0.0f;
}
//-----------------------------------------------------------------------------
/*! 
Prints some statistics after the rendering
*/
void SLRaytracer::printStats(SLfloat sec)
{
    SL_LOG("\nRender time  : %10.2f sec.", sec);
    SL_LOG("\nImage size   : %10d x %d",_images[0]->width(), _images[0]->height());
    SL_LOG("\nNum. Threads : %10d", _numThreads);
    SL_LOG("\nAllowed depth: %10d", SLRay::maxDepth);

    #if _DEBUG
    SLint  primarys = _sv->scrW()*_sv->scrH();
    SLuint total = primarys +
                   SLRay::reflectedRays +
                   SLRay::subsampledRays +
                   SLRay::refractedRays +
                   SLRay::shadowRays;

    SL_LOG("\nMaximum depth     : %10d", SLRay::maxDepthReached);
    SL_LOG("\nAverage depth     : %10.6f", SLRay::avgDepth/primarys);
    SL_LOG("\nAA threshold      : %10.1f", _aaThreshold);
    SL_LOG("\nAA subsampling    : %8dx%d\n", _aaSamples, _aaSamples);
    SL_LOG("\nSubsampled pixels : %10u, %4.1f%% of total", SLRay::subsampledPixels,  
            (SLfloat)SLRay::subsampledPixels/primarys*100.0f);   
    SL_LOG("\nPrimary rays      : %10u, %4.1f%% of total", primarys,               
            (SLfloat)primarys/total*100.0f);
    SL_LOG("\nReflected rays    : %10u, %4.1f%% of total", SLRay::reflectedRays,   
            (SLfloat)SLRay::reflectedRays/total*100.0f);
    SL_LOG("\nTransmitted rays  : %10u, %4.1f%% of total", SLRay::refractedRays, 
            (SLfloat)SLRay::refractedRays/total*100.0f);
    SL_LOG("\nTIR rays          : %10u, %4.1f%% of total", SLRay::tirRays,         
            (SLfloat)SLRay::tirRays/total*100.0f);
    SL_LOG("\nShadow rays       : %10u, %4.1f%% of total", SLRay::shadowRays,      
            (SLfloat)SLRay::shadowRays/total*100.0f);
    SL_LOG("\nAA subsampled rays: %10u, %4.1f%% of total", SLRay::subsampledRays,  
            (SLfloat)SLRay::subsampledRays/total*100.0f);
    SL_LOG("\nTotal rays        : %10u,100.0%%\n", total);
   
    SL_LOG("\nRays per second   : %10u", (SLuint)(total / sec));
    SL_LOG("\nIntersection tests: %10u", SLRay::tests);
    SL_LOG("\nIntersections     : %10u, %4.1f%%", SLRay::intersections, 
            SLRay::intersections/(SLfloat)SLRay::tests*100.0f);
    #endif
    SL_LOG("\n\n");
}
//-----------------------------------------------------------------------------
/*!
Creates the inherited image in the texture class. The RT is drawn into
a texture map that is displayed with OpenGL in 2D-orthographic projection.
Also precalculate as much as possible.
*/
void SLRaytracer::prepareImage()
{
    ///////////////////////
    //  PRECALCULATIONS  //
    ///////////////////////

    _cam = _sv->_camera;                // camera shortcut

    // get camera vectors eye, lookAt, lookUp
    _cam->updateAndGetVM().lookAt(&_EYE, &_LA, &_LU, &_LR);

    if (_cam->projection() == monoOrthographic)
    {   /*
        In orthographic projection the bottom-left vector (_BL) points
        from the eye to the center of the bottom-left pixel of a plane that
        parallel to the projection plan at zero distance from the eye.
        */
        SLVec3f pos(_cam->updateAndGetVM().translation());
        SLfloat hh = tan(SL_DEG2RAD*_cam->fov()*0.5f) * pos.length();
        SLfloat hw = hh * _sv->scrWdivH();

        // calculate the size of a pixel in world coords.
        _pxSize = hw * 2 / _sv->scrW();

        _BL = _EYE - hw*_LR - hh*_LU  +  _pxSize/2*_LR - _pxSize/2*_LU;
    }
    else
    {   /* 
        In perspective projection the bottom-left vector (_BL) points
        from the eye to the center of the bottom-left pixel on a projection
        plan in focal distance. See also the computer graphics script about
        primary ray calculation.
        */
        // calculate half window width & height in world coords
        SLfloat hh = tan(SL_DEG2RAD*_cam->fov()*0.5f) * _cam->focalDist();
        SLfloat hw = hh * _sv->scrWdivH();

        // calculate the size of a pixel in world coords.
        _pxSize = hw * 2 / _sv->scrW();

        // calculate a vector to the center (C) of the bottom left (BL) pixel
        SLVec3f C  = _LA * _cam->focalDist();
        _BL = C - hw*_LR - hh*_LU  +  _pxSize/2*_LR + _pxSize/2*_LU;
    }

    // Create the image for the first time
    if (_images.size()==0)
        _images.push_back(new SLImage(_sv->scrW(), _sv->scrH(), GL_RGB));

    // Allocate image of the inherited texture class 
    if (_sv->scrW() != _images[0]->width() || _sv->scrH() != _images[0]->height())
    {  
        // Delete the OpenGL Texture if it already exists
        if (_texName) 
        {   glDeleteTextures(1, &_texName);
            _texName = 0;
        }

        // Dispose VBO is they already exist
        _bufP.dispose();
        _bufT.dispose();
        _bufI.dispose();

        _images[0]->allocate(_sv->scrW(), _sv->scrH(), GL_RGB);
    }
   
    // Fill image black for single RT
    if (!_continuous) _images[0]->fill();
}
//-----------------------------------------------------------------------------
/*! 
Draw the RT-Image as a textured quad in 2D-Orthographic projection
*/
void SLRaytracer::renderImage()
{
    SLfloat w = (SLfloat)_sv->scrW();
    SLfloat h = (SLfloat)_sv->scrH();
    if (w != _images[0]->width()) return;
    if (h != _images[0]->height()) return;
      
    // Set orthographic projection with the size of the window
    _stateGL->projectionMatrix.ortho(0.0f, w, 0.0f, h, -1.0f, 0.0f);
    _stateGL->modelViewMatrix.identity();
    _stateGL->clearColorBuffer();
    _stateGL->depthTest(false);
    _stateGL->multiSample(false);
    _stateGL->polygonLine(false);

    drawSprite(true);
   
    // Write progress into info text
    if (_pcRendered < 100)
    {  SLchar str[255];  
        sprintf(str,"%s Tracing: Threads: %d, Progress: %d%%", 
                _infoText.c_str(), _numThreads, _pcRendered);
        SLScene::current->info(_sv, str, _infoColor);
    } else SLScene::current->info(_sv, _infoText.c_str(), _infoColor);

    _stateGL->depthTest(true);
    GET_GL_ERROR;
}
//-----------------------------------------------------------------------------
//! Saves the current RT image as PNG image
void SLRaytracer::saveImage()
{
    static SLint no = 0;
    SLchar filename[255];  
    sprintf(filename,"Raytraced_%d_%d.png", _maxDepth, no++);
    _images[0]->savePNG(filename);
}
//-----------------------------------------------------------------------------
