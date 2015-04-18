//#############################################################################
//  File:      SLPathtracer.cpp
//  Author:    Thomas Schneiter
//  Date:      July 2014
//  Copyright: Marcus Hudritsch
//             This software is provide under the GNU General Public License
//             Please visit: http://opensource.org/licenses/GPL-3.0
//#############################################################################

#include <stdafx.h>           // precompiled headers
#include <algorithm>
#ifdef SL_MEMLEAKDETECT
#include <nvwa/debug_new.h>   // memory leak detector
#endif

using namespace std::placeholders;
using namespace std::chrono;

#include <SLPathtracer.h>
#include <SLCamera.h>
#include <SLSceneView.h>
#include <SLLightSphere.h>
#include <SLLightRect.h>
#include <SLLight.h>
#include <SLVolume.h>
#include <SLNode.h>
#include <SLText.h>
#include <SLMesh.h>
#include <SLGLTexture.h>
#include <SLSamples2D.h>
#include <SLGLProgram.h>
#include <SLRay.h>

extern SLfloat rnd01();

//-----------------------------------------------------------------------------
SLPathtracer::SLPathtracer()
{  
    name("PathTracer");
    _gamma = 2.2f;
}

//-----------------------------------------------------------------------------
/*!
Main render function. The Path Tracing algorithm starts from here
*/
SLbool SLPathtracer::render(SLSceneView* sv)
{
    _sv = sv;
    _state = rtBusy;                    // From here we state the PT as busy
    _stateGL = SLGLState::getInstance();// OpenGL state shortcut
    _numThreads = 1;
    _renderSec = 0.0f;                  // reset time
    _pcRendered = 0;                    // % rendered
    _infoText  = SLScene::current->info(_sv)->text();  // keep original info string
    _infoColor = SLScene::current->info(_sv)->color(); // keep original info color

    prepareImage();

    // Set second image for render update to the same size
    _images.push_back(new SLImage(_sv->scrW(), _sv->scrH(), GL_RGB));

    // Measure time 
    double t1 = SLScene::current->timeSec();

    auto renderSlicesFunction   = bind(&SLPathtracer::renderSlices, this, _1, _2);

    // Do multi threading only in release config
    #ifdef _DEBUG
    _numThreads = 1;
    #else
    _numThreads = thread::hardware_concurrency();
    #endif
    {
        SL_LOG("\n\nRendering with %d samples", _aaSamples);
        SL_LOG("\nCurrent Sample:       ");
        for (int currentSample = 1; currentSample <= _aaSamples; currentSample++)
        {
            SL_LOG("\b\b\b\b\b\b%6d", currentSample);
            vector<thread> threads; // vector for additional threads  
            _next = 0;              // init _next=0. _next should be atomic

            // Start additional threads on the renderSlices function
            for (int t = 0; t < _numThreads - 1; t++)
                threads.push_back(thread(renderSlicesFunction, false, currentSample));

            // Do the same work in the main thread
            renderSlicesFunction(true, currentSample);

            for (auto& thread : threads) thread.join();

            _pcRendered = (SLint)((SLfloat)currentSample/(SLfloat)_aaSamples*100.0f);
        }
    }

    ////////////////////////////////////////////////////////////////////////////
    _renderSec = SLScene::current->timeSec() - (SLfloat)t1;

    SL_LOG("\nTime to render image: %6.3fsec", _renderSec);

    _state = rtFinished;
    return true;
}

//-----------------------------------------------------------------------------
/*!
Renders a slice of 4px width.
*/
void SLPathtracer::renderSlices(const bool isMainThread, SLint currentSample)
{
    // Time points
    SLScene* s = SLScene::current;
    double t1 = 0;
    const SLfloat oneOverGamma = 1.0f / _gamma;

    while (_next < _images[0]->width())
    {
        const SLint minX = _next;

        // The next line should be theoretically atomic
        _next += 4;

        for (SLint x=minX; x<minX+4; ++x)
        {
            for (SLuint y=0; y<_images[0]->height(); ++y)
            {
                SLCol4f color(SLCol4f::BLACK);

                // calculate direction for primary ray - scatter with random variables for anti aliasing
                SLRay primaryRay;
                setPrimaryRay((SLfloat)(x - rnd01() + 0.5f), (SLfloat)(y - rnd01() + 0.5f), &primaryRay);

                ///////////////////////////////
                color += trace(&primaryRay, 0);
                ///////////////////////////////

                // weight old and new color for continuous rendering
                SLCol4f oldColor;
                if (currentSample > 1)
                {
                    oldColor = _images[1]->getPixeli(x, y);

                    // weight old color ( examp. 3/4, 4/5, 5/6 )
                    oldColor /= (SLfloat)currentSample;
                    oldColor *= (SLfloat)(currentSample-1);

                    // weight new color ( examp. 1/4, 1/5, 1/6 )
                    color /= (SLfloat)currentSample;

                    // bring them together ( examp. 4/4, 5/5, 6/6)
                    color += oldColor;
                }

                color.clampMinMax(0.0f, 1.0f);

                // save image without gamma
                _images[1]->setPixeliRGB(x, y, color);

                // gamma correction
                color.x = pow((color.x), oneOverGamma);
                color.y = pow((color.y), oneOverGamma);
                color.z = pow((color.z), oneOverGamma);

                // image to render
                _images[0]->setPixeliRGB(x, y, color);
            }

            // update image after 500 ms
            if (isMainThread)
            {  
                if (SLScene::current->timeSec()-t1 > 0.5f)
                {  
                    _sv->onWndUpdate(); // update window
                    t1 = SLScene::current->timeSec();
                }
            }
        }
    }
}

//-----------------------------------------------------------------------------
/*!
Recursively traces Ray in Scene.
*/
SLCol4f SLPathtracer::trace(SLRay* ray, SLbool em)
{
    SLScene* s = SLScene::current;
    SLCol4f finalColor(SLCol4f::BLACK);

    // Participating Media init
    SLfloat  absorbtion = 1.0f;    // used to calculate absorbtion along the ray
    SLfloat  scaleBy = 1.0f;       // used to scale surface reflectance at the end of random walk

    s->root3D()->hitRec(ray);

    // end of recursion - no object hit OR max depth reached
    if (ray->length >= FLT_MAX || ray->depth > maxDepth())
        return SLCol4f::BLACK;

    // hit material
    SLMaterial* mat = ray->hitMat;
    ray->hitMesh->preShade(ray);

    SLCol4f objectEmission = mat->emission();
    SLCol4f objectColor    = SLCol4f::BLACK;

    // get base color of object
    if (ray->nodeDiffuse())            objectColor = mat->diffuse();
    else if (ray->nodeReflectance())   objectColor = mat->specular();
    else if (ray->nodeTransparency())  objectColor = mat->transmission();

    SLfloat maxEmission = objectEmission.r > objectEmission.g && 
                          objectEmission.r > objectEmission.b ? objectEmission.r : 
                          objectEmission.g > objectEmission.b ? objectEmission.g : 
                                                                objectEmission.b;

    // stop recursion if light source is hit
    if (maxEmission > 0)
    {   if (ray->depth == 1)
             return mat->emission() * absorbtion;
        else return mat->emission() * absorbtion * em;
    }

    // add absorbtion to base color from Participating Media
    objectColor = objectColor * absorbtion;

    // diffuse reflection
    if (ray->nodeDiffuse())
    {
        // Add component wise the texture color
        if (mat->textures().size()) 
        {  objectColor &= ray->hitTexCol;
        }

        // calculate direct illumination
        finalColor += shade(ray, &objectColor) * scaleBy;

        SLRay scatter;
        ray->diffuseMC(&scatter);

        // material emission, material diffuse and recursive indirect illumination
        finalColor += (trace(&scatter, 0) & objectColor) * scaleBy;
    }
    else 
    if (ray->nodeReflectance())
    {
        //scatter toward perfect specular direction
        SLRay reflected;
        ray->reflect(&reflected);

        //scatter around perfect reflected direction only if material not perfect
        if (mat->shininess() < SLMaterial::PERFECT)
        {
            //rotation matrix for glossy
            SLMat3f rotMat;
            SLVec3f rotAxis((SLVec3f(0.0f, 0.0f, 1.0f)^reflected.dir).normalize());
            SLfloat rotAngle = acos(reflected.dir.z);
            rotMat.rotation(rotAngle*180.0f / SL_PI, rotAxis);
            ray->reflectMC(&reflected, rotMat);
        }

        // shininess contribution * recursive indirect illumination and matrial base color
        finalColor += ((mat->shininess() + 2.0f) / (mat->shininess() + 1.0f) * (trace(&reflected, 1) & objectColor)) * scaleBy;
    }
    else 
    if (ray->nodeTransparency())
    {
        //scatter toward perfect transmissive direction
        SLRay refracted;
        ray->refract(&refracted);

        // init Schlick's approx.
        SLVec3f rayDir = ray->dir; rayDir.normalize();
        SLVec3f refrDir = refracted.dir; refrDir.normalize();
        SLfloat n, nt;
        SLVec3f hitNormal = ray->hitNormal; hitNormal.normalize();

        // ray from outside in
        if (ray->isOutside)
        {   n = 1.0f;
            nt = mat->kn();
        }
        else // ray from inside out
        {   n = mat->kn();
            nt = 1.0f;
        }

        // calculate Schlick's approx.
        SLfloat nbig, nsmall;
        nbig = n > nt ? n : nt;
        nsmall = n < nt ? n : nt;
        SLfloat R0 = ((nbig - nsmall) / (nbig + nsmall)); R0 = R0*R0;
        SLbool into = (rayDir * hitNormal) < 0;
        SLfloat c = 1.0f - (into ? (-rayDir * hitNormal) : (refrDir * hitNormal));
        SLfloat schlick = R0 + (1 - R0)* c * c * c * c * c;

        SLfloat P = 0.25f + 0.5f * schlick; // probability of reflectance
        SLfloat reflectionProbability = schlick / P;
        SLfloat refractionProbability = (1.0f - schlick) / (1.0f - P);

        //scatter around perfect transmissive direction only if material not perfect
        if (mat->translucency() < SLMaterial::PERFECT)
        {
            //rotation matrix for translucency
            SLMat3f rotMat;
            SLVec3f rotAxis((SLVec3f(0.0f, 0.0f, 1.0f)^refracted.dir).normalize());
            SLfloat rotAngle = acos(refracted.dir.z);
            rotMat.rotation((SLfloat)(rotAngle*180.0f / SL_PI), rotAxis);
            ray->refractMC(&refracted, rotMat);
        }

        // probability of reflection
        if (rnd01() > (0.25f + 0.5f * schlick))
            // scatter toward transmissive direction
            finalColor += ((mat->translucency() + 2.0f) / (mat->translucency() + 1.0f) * (trace(&refracted, 1) & objectColor) * refractionProbability) * scaleBy;
        else
        {
            //scatter toward perfect specular direction
            SLRay scattered;
            ray->reflect(&scattered);
            // shininess contribution * recursive indirect illumination and matrial basecolor
            finalColor += ((mat->shininess() + 2.0f) / (mat->shininess() + 1.0f) * (trace(&scattered, 1) & objectColor) * reflectionProbability) *scaleBy;
        }
    }

    return finalColor;
}

//-----------------------------------------------------------------------------
/*!
Calculates direct illumination for intersection point of ray
*/
SLCol4f SLPathtracer::shade(SLRay* ray, SLCol4f* objectColor)
{
    SLScene* s = SLScene::current;
    SLCol4f  color = SLCol4f::BLACK;
    SLCol4f  diffuseColor = SLCol4f::BLACK;
    SLVec3f  L, N;
    SLfloat  lightDist, LdN, df, spotEffect, lighted = 0.0f;

    // loop over light sources in scene
    for (SLint i = 0; i < s->lights().size(); ++i)
    {
        SLLight* light = s->lights()[i];

        if (light && light->on())
        {
            N.set(ray->hitNormal);
            L.sub(light->positionWS(), ray->hitPoint);
            lightDist = L.length();
            L /= lightDist;
            LdN = L.dot(N);

            // check shadow ray if hit point is towards the light
            lighted = (SLfloat)((LdN > 0) ? light->shadowTestMC(ray, L, lightDist) : 0);

            // calculate spot effect if light is a spotlight
            if (lighted > 0.0f && light->spotCutoff() < 180.0f)
            {
                SLfloat LdS = SL_max(-L.dot(light->spotDirWS()), 0.0f);

                // check if point is in spot cone
                if (LdS > light->spotCosCut())
                {
                    spotEffect = pow(LdS, (SLfloat)light->spotExponent());
                }
                else
                {
                    lighted = 0.0f;
                    spotEffect = 0.0f;
                }
            }
            else spotEffect = 1.0f;

            if (lighted > 0.0f)
            {
                df = SL_max(LdN, 0.0f); // diffuse factor

                // material color * light emission * LdN * brdf(1/pi) * lighted(for soft shadows)
                diffuseColor = (*objectColor & (light->diffuse() * df) * (1 / SL_PI) * lighted);
            }

            color += light->attenuation(lightDist) * spotEffect * diffuseColor;
        }
    }

    return color;
}
//-----------------------------------------------------------------------------
//! Saves the current PT image as PNG image
void SLPathtracer::saveImage()
{   static SLint no = 0;
    SLchar filename[255];  
    sprintf(filename,"Pathtraced_%d_%d.png", _aaSamples, no++);
    _images[0]->savePNG(filename);
}
//-----------------------------------------------------------------------------
