//#############################################################################
//  File:      SLRaytracer.cpp
//  Author:    Marcus Hudritsch
//  Date:      July 2014
//  Copyright: Marcus Hudritsch
//             This software is provide under the GNU General Public License
//             Please visit: http://opensource.org/licenses/GPL-3.0
//#############################################################################

#include <stdafx.h> // Must be the 1st include followed by  an empty line

#ifdef SL_MEMLEAKDETECT    // set in SL.h for debug config only
#    include <debug_new.h> // memory leak detector
#endif

using namespace std::placeholders;
using namespace std::chrono;

#include <SLApplication.h>
#include <SLLightRect.h>
#include <SLRay.h>
#include <SLRaytracer.h>
#include <SLSceneView.h>

//-----------------------------------------------------------------------------
SLRaytracer::SLRaytracer()
{
    name("myCoolRaytracer");

    _state         = rtReady;
    _doDistributed = true;
    _doContinuous  = false;
    _doFresnel     = false;
    _maxDepth      = 5;
    _aaThreshold   = 0.3f; // = 10% color difference
    _aaSamples     = 3;
    gamma(1.0f);

    // set texture properties
    _min_filter   = GL_NEAREST;
    _mag_filter   = GL_NEAREST;
    _wrap_s       = GL_CLAMP_TO_EDGE;
    _wrap_t       = GL_CLAMP_TO_EDGE;
    _resizeToPow2 = false;
}
//-----------------------------------------------------------------------------
SLRaytracer::~SLRaytracer()
{
    SL_LOG("Destructor      : ~SLRaytracer");
}
//-----------------------------------------------------------------------------
/*!
This is the main rendering method for the classic ray tracing. It loops over all 
lines and pixels and determines for each pixel a color with a partly global 
illumination calculation.
*/
SLbool SLRaytracer::renderClassic(SLSceneView* sv)
{
    _sv         = sv;
    _state      = rtBusy; // From here we state the RT as busy
    _pcRendered = 0;      // % rendered
    _renderSec  = 0.0f;   // reset time

    initStats(_maxDepth); // init statistics
    prepareImage();       // Setup image & precalculations

    // Measure time
    float t1     = SLApplication::timeS();
    float tStart = t1;

    for (SLuint y = 0; y < _images[0]->height(); ++y)
    {
        for (SLuint x = 0; x < _images[0]->width(); ++x)
        {
            SLRay primaryRay(_sv);
            setPrimaryRay((SLfloat)x, (SLfloat)y, &primaryRay);

#ifdef DEBUG_RAY
            cout << "\nRay(" << x << "," << y << "):" << endl;
#endif

            ///////////////////////////////////
            SLCol4f color = trace(&primaryRay);
            ///////////////////////////////////

            color.gammaCorrect(_oneOverGamma);

            _images[0]->setPixeliRGB((SLint)x, (SLint)y, CVVec4f(color.r, color.g, color.b, color.a));

            SLRay::avgDepth += SLRay::depthReached;
            SLRay::maxDepthReached = std::max(SLRay::depthReached,
                                              SLRay::maxDepthReached);
        }

        // Update image after 500 ms
        double t2 = SLApplication::timeS();
        if (t2 - t1 > 0.5)
        {
            _pcRendered = (SLint)((SLfloat)y / (SLfloat)_images[0]->height() * 100);
            renderUIBeforeUpdate();
            _sv->onWndUpdate();
            t1 = SLApplication::timeS();
        }
    }

    _renderSec  = SLApplication::timeS() - tStart;
    _pcRendered = 100;

    if (_doContinuous)
        _state = rtReady;
    else
    {
        _state = rtFinished;
        printStats(_renderSec);
    }
    return true;
}
//-----------------------------------------------------------------------------
/*!
This is the main rendering method for parallel and distributed ray tracing.
*/
SLbool SLRaytracer::renderDistrib(SLSceneView* sv)
{
    _sv         = sv;
    _state      = rtBusy; // From here we state the RT as busy
    _pcRendered = 0;      // % rendered
    _renderSec  = 0.0f;   // reset time

    initStats(_maxDepth); // init statistics
    prepareImage();       // Setup image & precalculations

    // Measure time
    float t1 = SLApplication::timeS();

    // Bind render functions to be called multithreaded
    auto sampleAAPixelsFunction = bind(&SLRaytracer::sampleAAPixels, this, _1);
    auto renderSlicesFunction   = _cam->lensSamples()->samples() == 1
                                  ? bind(&SLRaytracer::renderSlices, this, _1)
                                  : bind(&SLRaytracer::renderSlicesMS, this, _1);

    // Do multithreading only in release config
    // Render image without antialiasing
    vector<thread> threads; // vector for additional threads
    _next = 0;              // init _next=0. _next should be atomic

    // Start additional threads on the renderSlices function
    for (SLuint t = 0; t < Utils::maxThreads() - 1; t++)
        threads.emplace_back(renderSlicesFunction, false);

    // Do the same work in the main thread
    renderSlicesFunction(true);

    // Wait for the other threads to finish
    for (auto& thread : threads)
        thread.join();

    // Do anti-aliasing w. contrast compare in a 2nd. pass
    if (!_doContinuous && _aaSamples > 1)
    {
        getAAPixels();          // Fills in the AA pixels by contrast
        vector<thread> threads; // vector for additional threads
        _next = 0;              // init _next=0. _next should be atomic

        // Start additional threads on the sampleAAPixelFunction function
        for (SLuint t = 0; t < Utils::maxThreads() - 1; t++)
            threads.emplace_back(sampleAAPixelsFunction, false);

        // Do the same work in the main thread
        sampleAAPixelsFunction(true);

        // Wait for the other threads to finish
        for (auto& thread : threads)
            thread.join();
    }

    _renderSec  = SLApplication::timeS() - t1;
    _pcRendered = 100;

    if (_doContinuous)
        _state = rtReady;
    else
    {
        _state = rtFinished;
        printStats(_renderSec);
    }
    return true;
}
//-----------------------------------------------------------------------------
/*!
Renders slices of 4 rows until the full width of the image is rendered. This
method can be called as a function by multiple threads.
The _next index is used and incremented by every thread. So it should be locked
or an atomic index. I prefer not protecting it because it's faster. If the
increment is not done properly some pixels may get ray traced twice. Only the
main thread is allowed to call a repaint of the image.
*/
void SLRaytracer::renderSlices(const bool isMainThread)
{
    // Time points
    double t1 = 0;

    while (_next < (SLint)_images[0]->height())
    {
        const SLint minY = _next;

        // The next line should be atomic
        _next += 4;

        for (SLint y = minY; y < minY + 4; ++y)
        {
            for (SLuint x = 0; x < _images[0]->width(); ++x)
            {
                SLRay primaryRay(_sv);
                setPrimaryRay((SLfloat)x, (SLfloat)y, &primaryRay);

                ///////////////////////////////////
                SLCol4f color = trace(&primaryRay);
                ///////////////////////////////////

                color.gammaCorrect(_oneOverGamma);

                _images[0]->setPixeliRGB((SLint)x, (SLint)y, CVVec4f(color.r, color.g, color.b, color.a));

                SLRay::avgDepth += SLRay::depthReached;
                SLRay::maxDepthReached = std::max(SLRay::depthReached,
                                                  SLRay::maxDepthReached);
            }

            // Update image after 500 ms
            if (isMainThread && !_doContinuous)
            {
                if (SLApplication::timeS() - t1 > 0.5)
                {
                    _pcRendered = (SLint)((SLfloat)y /
                                          (SLfloat)_images[0]->height() * 100);
                    if (_aaSamples > 0) _pcRendered /= 2;
                    renderUIBeforeUpdate();
                    _sv->onWndUpdate();
                    t1 = SLApplication::timeS();
                }
            }
        }
    }
}
//-----------------------------------------------------------------------------
/*!
Renders slices of 4 rows multisampled until the full width of the image is
rendered. Every pixel is multisampled for depth of field lens sampling. This
method can be called as a function by multiple threads.
The _next index is used and incremented by every thread. So it should be locked
or an atomic index. I prefer not protecting it because it's faster. If the
increment is not done properly some pixels may get ray traced twice. Only the
main thread is allowed to call a repaint of the image.
*/
void SLRaytracer::renderSlicesMS(const bool isMainThread)
{
    // Time points
    double t1 = 0;

    // lens sampling constants
    SLVec3f lensRadiusX = _LR * (_cam->lensDiameter() * 0.5f);
    SLVec3f lensRadiusY = _LU * (_cam->lensDiameter() * 0.5f);

    while (_next < (SLint)_images[0]->width())
    {
        const SLint minY = _next;

        // The next line should be atomic
        _next += 4;

        for (SLint y = minY; y < minY + 4; ++y)
        {
            for (SLuint x = 0; x < _images[0]->width(); ++x)
            {
                // focal point is single shot primary dir
                SLVec3f primaryDir(_BL + _pxSize * ((SLfloat)x * _LR + (SLfloat)y * _LU));
                SLVec3f FP = _EYE + primaryDir;
                SLCol4f color(SLCol4f::BLACK);

                // Loop over radius r and angle phi of lens
                for (SLint iR = (SLint)_cam->lensSamples()->samplesX() - 1; iR >= 0; --iR)
                {
                    for (SLint iPhi = (SLint)_cam->lensSamples()->samplesY() - 1; iPhi >= 0; --iPhi)
                    {
                        SLVec2f discPos(_cam->lensSamples()->point((SLuint)iR, (SLuint)iPhi));

                        // calculate lens position out of disc position
                        SLVec3f lensPos(_EYE + discPos.x * lensRadiusX + discPos.y * lensRadiusY);
                        SLVec3f lensToFP(FP - lensPos);
                        lensToFP.normalize();

                        SLCol4f backColor;
                        if (_sv->skybox())
                            backColor = _sv->skybox()->colorAtDir(lensToFP);
                        else
                            backColor = _sv->camera()->background().colorAtPos((SLfloat)x, (SLfloat)y);

                        SLRay primaryRay(lensPos, lensToFP, (SLfloat)x, (SLfloat)y, backColor, _sv);

                        ////////////////////////////
                        color += trace(&primaryRay);
                        ////////////////////////////

                        SLRay::avgDepth += SLRay::depthReached;
                        SLRay::maxDepthReached = std::max(SLRay::depthReached,
                                                          SLRay::maxDepthReached);
                    }
                }
                color /= (SLfloat)_cam->lensSamples()->samples();

                color.gammaCorrect(_oneOverGamma);

                _images[0]->setPixeliRGB((SLint)x, y, CVVec4f(color.r, color.g, color.b, color.a));

                SLRay::avgDepth += SLRay::depthReached;
                SLRay::maxDepthReached = std::max(SLRay::depthReached, SLRay::maxDepthReached);
            }

            if (isMainThread && !_doContinuous)
            {
                if (SLApplication::timeS() - t1 > 0.5)
                {
                    renderUIBeforeUpdate();
                    _sv->onWndUpdate();
                    t1 = SLApplication::timeS();
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
SLCol4f SLRaytracer::trace(SLRay* ray)
{
    SLScene* s = SLApplication::scene;
    SLCol4f  color(ray->backgroundColor);

    s->root3D()->hitRec(ray);

    if (ray->length < FLT_MAX)
    {
        color = shade(ray);

        SLfloat kt = ray->hitMesh->mat()->kt();
        SLfloat kr = ray->hitMesh->mat()->kr();

        if (ray->depth < SLRay::maxDepth && ray->contrib > SLRay::minContrib)
        {
            if (!_doFresnel)
            { // Do classic refraction and/or reflection
                if (kt > 0.0f)
                {
                    SLRay refracted(_sv);
                    ray->refract(&refracted);
                    color += kt * trace(&refracted);
                }
                if (kr > 0.0f)
                {
                    SLRay reflected(_sv);
                    ray->reflect(&reflected);
                    color += kr * trace(&reflected);
                }
            }
            else
            { // Mix refr. & refl. color w. Fresnel aproximation
                if (kt > 0.0f)
                {
                    SLRay refracted(_sv), reflected(_sv);
                    ray->refract(&refracted);
                    ray->reflect(&reflected);
                    SLCol4f refrCol = trace(&refracted);
                    SLCol4f reflCol = trace(&reflected);

                    // Apply Schlick's Fresnel aproximation
                    SLfloat F0      = kr;
                    SLfloat theta   = -(ray->dir * ray->hitNormal);
                    SLfloat F_theta = F0 + (1 - F0) * (SLfloat)pow(1 - theta, 5);
                    color += refrCol * (1 - F_theta) + reflCol * F_theta;
                }
                else
                {
                    if (kr > 0.0f)
                    {
                        SLRay reflected(_sv);
                        ray->reflect(&reflected);
                        color += kr * trace(&reflected);
                    }
                }
            }
        }
    }

    if (SLGLState::instance()->fogIsOn)
        color = fogBlend(ray->length, color);

    color.clampMinMax(0, 1);
    return color;
}
//-----------------------------------------------------------------------------
//! Set the parameters of a primary ray for a pixel position at x, y.
void SLRaytracer::setPrimaryRay(SLfloat x, SLfloat y, SLRay* primaryRay)
{
    primaryRay->x  = x;
    primaryRay->y  = y;
    primaryRay->sv = _sv;

    // calculate ray from eye to pixel (See also prepareImage())
    if (_cam->projection() == P_monoOrthographic)
    {
        primaryRay->setDir(_LA);
        primaryRay->origin = _BL + _pxSize * ((SLfloat)x * _LR + (SLfloat)y * _LU);
    }
    else
    {
        SLVec3f primaryDir(_BL + _pxSize * ((SLfloat)x * _LR + (SLfloat)y * _LU));
        primaryDir.normalize();
        primaryRay->setDir(primaryDir);
        primaryRay->origin = _EYE;
    }

    if (_sv->skybox())
        primaryRay->backgroundColor = _sv->skybox()->colorAtDir(primaryRay->dir);
    else
        primaryRay->backgroundColor = _sv->camera()->background().colorAtPos(x, y);
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
    SLScene*      s          = SLApplication::scene;
    SLCol4f       localColor = SLCol4f::BLACK;
    SLMaterial*   mat        = ray->hitMesh->mat();
    SLVGLTexture& texture    = mat->textures();
    SLVec3f       L, N, H;
    SLfloat       lightDist, LdN, NdH, df, sf, spotEffect, att, lighted;
    SLCol4f       amdi, spec;
    SLCol4f       localSpec(0, 0, 0, 1);

    localColor = mat->emissive() + (mat->ambient() & s->globalAmbiLight());

    ray->hitMesh->preShade(ray);

    for (auto light : s->lights())
    {
        if (light && light->isOn())
        {
            // calculate light vector L and distance to light
            N.set(ray->hitNormal);

            // Distinguish between point and directional lights
            SLVec4f lightPos = light->positionWS();
            if (lightPos.w == 0.0f)
            { // directional light
                L         = lightPos.vec3().normalized();
                lightDist = FLT_MAX; // = infinity
            }
            else
            { // Point light
                L.sub(lightPos.vec3(), ray->hitPoint);
                lightDist = L.length();
                L /= lightDist;
            }

            // Cosine between L and N
            LdN = L.dot(N);

            // check shadow ray if hit point is towards the light
            lighted = (LdN > 0) ? light->shadowTest(ray, L, lightDist) : 0;

            // calculate the ambient part
            amdi = light->ambient() & mat->ambient();
            spec.set(0, 0, 0);

            // calculate spot effect if light is a spotlight
            if (lighted > 0.0f && light->spotCutOffDEG() < 180.0f)
            {
                SLfloat LdS = std::max(-L.dot(light->spotDirWS()), 0.0f);

                // check if point is in spot cone
                if (LdS > light->spotCosCut())
                {
                    spotEffect = pow(LdS, (SLfloat)light->spotExponent());
                }
                else
                {
                    lighted    = 0.0f;
                    spotEffect = 0.0f;
                }
            }
            else
                spotEffect = 1.0f;

            // calculate local illumination only if point is not shaded
            if (lighted > 0.0f)
            {
                H.sub(L, ray->dir); // half vector between light & eye
                H.normalize();
                df  = std::max(LdN, 0.0f); // diffuse factor
                NdH = std::max(N.dot(H), 0.0f);
                sf  = pow(NdH, (SLfloat)mat->shininess()); // specular factor

                amdi += lighted * df * light->diffuse() & mat->diffuse();
                spec = lighted * sf * light->specular() & mat->specular();
            }

            // apply attenuation and spot effect
            att = light->attenuation(lightDist) * spotEffect;
            localColor += att * amdi;
            localSpec += att * spec;
        }
    }

    if (!texture.empty() || !ray->hitMesh->C.empty())
    {
        localColor &= ray->hitColor; // component wise multiply
        localColor += localSpec;     // add afterwards the specular component
    }
    else
        localColor += localSpec;

    localColor.clampMinMax(0, 1);
    return localColor;
}
//-----------------------------------------------------------------------------
/*!
This method fills the pixels into the vector pix that need to be subsampled
because the contrast to its left and/or above neighbor is above a threshold.
*/
void SLRaytracer::getAAPixels()
{
    SLCol4f color, colorLeft, colorUp; // pixel colors to be compared
    SLVbool gotSampled;
    gotSampled.resize(_images[0]->width()); // Flags if above pixel got sampled
    SLbool isSubsampled;                    // Flag if pixel got subsampled

    // Nothing got sampled at beginning
    for (SLuint x = 0; x < _images[0]->width(); ++x)
        gotSampled[x] = false;

    // Loop through all pixels & add the pixel that have to be subsampled
    _aaPixels.clear();
    for (SLuint y = 0; y < _images[0]->height(); ++y)
    {
        for (SLuint x = 0; x < _images[0]->width(); ++x)
        {
            CVVec4f c4f = _images[0]->getPixeli((SLint)x, (SLint)y);
            color.set(c4f[0], c4f[1], c4f[2], c4f[3]);

            isSubsampled = false;
            if (x > 0)
            {
                CVVec4f colL = _images[0]->getPixeli((SLint)x - 1, (SLint)y);
                colorLeft.set(colL[0], colL[1], colL[2], colL[3]);
                if (color.diffRGB(colorLeft) > _aaThreshold)
                {
                    if (!gotSampled[x - 1])
                    {
                        _aaPixels.push_back(SLRTAAPixel((SLushort)x - 1, (SLushort)y));
                        gotSampled[x - 1] = true;
                    }
                    _aaPixels.push_back(SLRTAAPixel((SLushort)x, (SLushort)y));
                    isSubsampled = true;
                }
            }
            if (y > 0)
            {
                CVVec4f colU = _images[0]->getPixeli((SLint)x, (SLint)y - 1);
                colorUp.set(colU[0], colU[1], colU[2], colU[3]);
                if (color.diffRGB(colorUp) > _aaThreshold)
                {
                    if (!gotSampled[x]) _aaPixels.push_back(SLRTAAPixel((SLushort)x, (SLushort)y - 1));
                    if (!isSubsampled)
                    {
                        _aaPixels.push_back(SLRTAAPixel((SLushort)x, (SLushort)y));
                        isSubsampled = true;
                    }
                }
            }
            gotSampled[x] = isSubsampled;
        }
    }
    SLRay::subsampledPixels = (SLuint)_aaPixels.size();
}
//-----------------------------------------------------------------------------
/*!
SLRaytracer::sampleAAPixels does the subsampling of the pixels that need to be
antialiased. See also getAAPixels. This routine can be called by multiple
threads.
The _next index is used and incremented by every thread. So it should be locked
or an atomic index. I prefer not protecting it because it's faster. If the
increment is not done properly some pixels may get ray traced twice. Only the
main thread is allowed to call a repaint of the image.
*/
void SLRaytracer::sampleAAPixels(const bool isMainThread)
{
    assert(_aaSamples % 2 == 1 && "subSample: maskSize must be uneven");
    double t1 = 0, t2 = 0;

    while (_next < (SLint)_aaPixels.size())
    {
        SLuint mini = (SLuint)_next;
        _next += 4;

        for (SLuint i = mini; i < mini + 4 && i < _aaPixels.size(); ++i)
        {
            SLuint  x   = _aaPixels[i].x;
            SLuint  y   = _aaPixels[i].y;
            CVVec4f c4f = _images[0]->getPixeli((SLint)x, (SLint)y);
            SLCol4f centerColor(c4f[0], c4f[1], c4f[2], c4f[3]);
            SLint   centerIndex = _aaSamples >> 1;
            SLfloat f           = 1.0f / (SLfloat)_aaSamples;
            SLfloat xpos        = x - centerIndex * f;
            SLfloat ypos        = y - centerIndex * f;
            SLfloat samples     = (SLfloat)_aaSamples * _aaSamples;
            SLCol4f color(0, 0, 0);

            // Loop regularly over the float pixel
            for (SLint j = 0; j < _aaSamples; ++j)
            {
                for (SLint i = 0; i < _aaSamples; ++i)
                {
                    if (i == centerIndex && j == centerIndex)
                        color += centerColor; // don't shoot for center position
                    else
                    {
                        SLRay primaryRay(_sv);
                        setPrimaryRay(xpos + i * f, ypos + i * f, &primaryRay);
                        color += trace(&primaryRay);
                    }
                }
                ypos += f;
            }
            SLRay::subsampledRays += (SLuint)samples;
            color /= samples;

            color.gammaCorrect(_oneOverGamma);

            _images[0]->setPixeliRGB((SLint)x, (SLint)y, CVVec4f(color.r, color.g, color.b, color.a));
        }

        if (isMainThread && !_doContinuous)
        {
            t2 = SLApplication::timeS();
            if (t2 - t1 > 0.5)
            {
                _pcRendered = 50 + (SLint)((SLfloat)_next / (SLfloat)_aaPixels.size() * 50);
                renderUIBeforeUpdate();
                _sv->onWndUpdate();
                t1 = SLApplication::timeS();
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
    SLfloat    f;
    SLGLState* stateGL = SLGLState::instance();

    if (z > _sv->_camera->clipFar())
        z = _sv->_camera->clipFar();

    switch (stateGL->fogMode)
    {
        case 0:
            f = (stateGL->fogDistEnd - z) /
                (stateGL->fogDistEnd - stateGL->fogDistStart);
            break;
        case 1: f = exp(-stateGL->fogDensity * z); break;
        default: f = exp(-stateGL->fogDensity * z * stateGL->fogDensity * z); break;
    }
    color = f * color + (1 - f) * stateGL->fogColor;
    color.clampMinMax(0, 1);
    return color;
}
//-----------------------------------------------------------------------------
/*!
Initialises the statistic variables in SLRay to zero
*/
void SLRaytracer::initStats(SLint depth)
{
    SLRay::maxDepth         = (depth) ? depth : SL_MAXTRACE;
    SLRay::reflectedRays    = 0;
    SLRay::refractedRays    = 0;
    SLRay::tirRays          = 0;
    SLRay::shadowRays       = 0;
    SLRay::subsampledRays   = 0;
    SLRay::subsampledPixels = 0;
    SLRay::tests            = 0;
    SLRay::intersections    = 0;
    SLRay::maxDepthReached  = 0;
    SLRay::avgDepth         = 0.0f;
}
//-----------------------------------------------------------------------------
/*! 
Prints some statistics after the rendering
*/
void SLRaytracer::printStats(SLfloat sec)
{
    SL_LOG("\nRender time  : %10.2f sec.", sec);
    SL_LOG("Image size   : %10d x %d", _images[0]->width(), _images[0]->height());
    SL_LOG("Num. Threads : %10d", Utils::maxThreads());
    SL_LOG("Allowed depth: %10d", SLRay::maxDepth);

    SLuint primarys = (SLuint)(_sv->viewportRect().width * _sv->viewportRect().height);
    SLuint total    = primarys +
                   SLRay::reflectedRays +
                   SLRay::subsampledRays +
                   SLRay::refractedRays +
                   SLRay::shadowRays;

    SL_LOG("Maximum depth     : %10d", SLRay::maxDepthReached);
    SL_LOG("Average depth     : %10.6f", SLRay::avgDepth / primarys);
    SL_LOG("AA threshold      : %10.1f", _aaThreshold);
    SL_LOG("AA subsampling    : %8dx%d\n", _aaSamples, _aaSamples);
    SL_LOG("Subsampled pixels : %10u, %4.1f%% of total", SLRay::subsampledPixels, (SLfloat)SLRay::subsampledPixels / primarys * 100.0f);
    SL_LOG("Primary rays      : %10u, %4.1f%% of total", primarys, (SLfloat)primarys / total * 100.0f);
    SL_LOG("Reflected rays    : %10u, %4.1f%% of total", SLRay::reflectedRays, (SLfloat)SLRay::reflectedRays / total * 100.0f);
    SL_LOG("Refracted rays    : %10u, %4.1f%% of total", SLRay::refractedRays, (SLfloat)SLRay::refractedRays / total * 100.0f);
    SL_LOG("Ignored rays      : %10u, %4.1f%% of total", SLRay::ignoredRays, (SLfloat)SLRay::ignoredRays / total * 100.0f);
    SL_LOG("TIR rays          : %10u, %4.1f%% of total", SLRay::tirRays, (SLfloat)SLRay::tirRays / total * 100.0f);
    SL_LOG("Shadow rays       : %10u, %4.1f%% of total", SLRay::shadowRays, (SLfloat)SLRay::shadowRays / total * 100.0f);
    SL_LOG("AA subsampled rays: %10u, %4.1f%% of total", SLRay::subsampledRays, (SLfloat)SLRay::subsampledRays / total * 100.0f);
    SL_LOG("Total rays        : %10u,100.0%%\n", total);

    SL_LOG("Rays per second   : %10u", (SLuint)(total / sec));
    SL_LOG("Intersection tests: %10u", SLRay::tests);
    SL_LOG("Intersections     : %10u, %4.1f%%\n", SLRay::intersections, SLRay::intersections / (SLfloat)SLRay::tests * 100.0f);
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

    _cam = _sv->_camera; // camera shortcut

    // get camera vectors eye, lookAt, lookUp
    _cam->updateAndGetVM().lookAt(&_EYE, &_LA, &_LU, &_LR);

    if (_cam->projection() == P_monoOrthographic)
    { /*
        In orthographic projection the bottom-left vector (_BL) points
        from the eye to the center of the bottom-left pixel of a plane that
        parallel to the projection plan at zero distance from the eye.
        */
        SLVec3f pos(_cam->updateAndGetVM().translation());
        SLfloat hh = tan(Utils::DEG2RAD * _cam->fov() * 0.5f) * pos.length();
        SLfloat hw = hh * _sv->viewportWdivH();

        // calculate the size of a pixel in world coords.
        _pxSize = hw * 2 / _sv->viewportW();

        _BL = _EYE - hw * _LR - hh * _LU + _pxSize / 2 * _LR - _pxSize / 2 * _LU;
    }
    else
    { /*
        In perspective projection the bottom-left vector (_BL) points
        from the eye to the center of the bottom-left pixel on a projection
        plan in focal distance. See also the computer graphics script about
        primary ray calculation.
        */
        // calculate half window width & height in world coords
        SLfloat hh = tan(Utils::DEG2RAD * _cam->fov() * 0.5f) * _cam->focalDist();
        SLfloat hw = hh * _sv->viewportWdivH();

        // calculate the size of a pixel in world coords.
        _pxSize = hw * 2 / _sv->viewportW();

        // calculate a vector to the center (C) of the bottom left (BL) pixel
        SLVec3f C = _LA * _cam->focalDist();
        _BL       = C - hw * _LR - hh * _LU + _pxSize / 2 * _LR + _pxSize / 2 * _LU;
    }

    // Create the image for the first time
    if (_images.empty())
        _images.push_back(new CVImage(_sv->viewportW(),
                                      _sv->viewportH(),
                                      PF_rgb,
                                      "Raytracer"));

    // Allocate image of the inherited texture class
    if (_sv->viewportW() != (SLint)_images[0]->width() ||
        _sv->viewportH() != (SLint)_images[0]->height())
    {
        // Delete the OpenGL Texture if it already exists
        if (_texID)
        {
            glDeleteTextures(1, &_texID);
            _texID = 0;
        }

        _vaoSprite.clearAttribs();
        _images[0]->allocate(_sv->viewportW(),
                             _sv->viewportH(),
                             PF_rgb);
    }

    // Fill image black for single RT
    if (!_doContinuous) _images[0]->fill(0, 0, 0);
}
//-----------------------------------------------------------------------------
/*! 
Draw the RT-Image as a textured quad in 2D-Orthographic projection
*/
void SLRaytracer::renderImage()
{
    SLRecti vpRect = _sv->viewportRect();
    SLfloat w      = (SLfloat)vpRect.width;
    SLfloat h      = (SLfloat)vpRect.height;
    if (Utils::abs(_images[0]->width() - w) > 0.0001f) return;
    if (Utils::abs(_images[0]->height() - h) > 0.0001f) return;

    // Set orthographic projection with the size of the window
    SLGLState* stateGL = SLGLState::instance();
    stateGL->viewport(vpRect.x, vpRect.y, (size_t)w, (size_t)h);
    stateGL->projectionMatrix.ortho(0.0f, w, 0.0f, h, -1.0f, 0.0f);
    stateGL->modelViewMatrix.identity();
    stateGL->clearColorBuffer();
    stateGL->depthTest(false);
    stateGL->multiSample(false);
    stateGL->polygonLine(false);

    drawSprite(true);

    stateGL->depthTest(true);
    GET_GL_ERROR;
}
//-----------------------------------------------------------------------------
//! Saves the current RT image as PNG image
void SLRaytracer::saveImage()
{
    static SLint no = 0;
    SLchar       filename[255];
    sprintf(filename, "Raytraced_%d_%d.png", _maxDepth, no++);
    _images[0]->savePNG(filename, 9, true, true);
}
//-----------------------------------------------------------------------------
//! Must be called before an inbetween frame update
/* Ray and path tracing usually take much more time to render one frame.
We therefore call every half second _sv->onWndUpdate() that initiates another
paint message from the top-level UI system of the OS. We therefore have to
finish our UI and end OpenGL rendering properly.
*/
void SLRaytracer::renderUIBeforeUpdate()
{
    _sv->gui()->onPaint(_sv->viewportRect());
    SLGLState::instance()->unbindAnythingAndFlush();
}
//-----------------------------------------------------------------------------
