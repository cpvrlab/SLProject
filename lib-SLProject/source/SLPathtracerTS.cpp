//#############################################################################
//  File:      SLPathtracer.cpp
//  Author:    Thomas Schneiter
//  Date:      Dezember 2013
//  Copyright (c): 2002-2013 Marcus Hudritsch
//             This software is provide under the GNU General Public License
//             Please visit: http://opensource.org/licenses/GPL-3.0
//#############################################################################

#include <stdafx.h>           // precompiled headers
#include <algorithm>
#ifdef SL_MEMLEAKDETECT
#include <nvwa/debug_new.h>   // memory leak detector
#endif

#ifdef SL_CPP11
using namespace std::placeholders;
using namespace std::chrono;
#endif

#include "SLPathtracer.h"
#include "SLCamera.h"
#include "SLSceneView.h"
#include "SLLightSphere.h"
#include "SLLightRect.h"
#include "SLLight.h"
#include "SLVolume.h"
#include "SLGroup.h"
#include "SLMesh.h"
#include "SLGLTexture.h"
#include "SLSamples2D.h"
#include "SLGLShaderProg.h"

//-----------------------------------------------------------------------------
SLPathtracer::SLPathtracer() : _uniformRandom(0, 1)
{  
   name("PathTracer");
   _gamma = 2.2f;
}

//-----------------------------------------------------------------------------
SLPathtracer::~SLPathtracer()
{
}

//-----------------------------------------------------------------------------
/*!
Main render function.
Path Tracing Algorithm starts from here
*/
SLbool SLPathtracer::render()
{
   _state = ptBusy;                    // From here we state the PT as busy
   _stateGL = SLGLState::getInstance();// OpenGL state shortcut
   _numThreads = 1;
   _renderSec = 0.0f;                  // reset time
   _infoText  = SLScene::current->info()->text();  // keep original info string
   _infoColor = SLScene::current->info()->color(); // keep original info color

   prepareImage();

   // Measure time 
   double t1 = SLScene::current->timeSec();

   #ifdef SL_CPP11
   auto renderSlicesFunction   = bind(&SLPathtracer::renderSlices, this, _1, _2);

   // Do multithreading only in release config
   #ifdef _DEBUG
   _numThreads = 1;
   #else
   _numThreads = thread::hardware_concurrency();
   #endif
   {
      SL_LOG("\n\nRendering with %d samples", samples())
      SL_LOG("\nCurrent Sample:    ");
      for (int currentSample = 1; currentSample <= samples(); currentSample++)
      {
         SL_LOG("\b\b\b\b%4d%", currentSample);
         vector<thread> threads; // vector for additional threads  
         _next = 0;              // init _next=0. _next should be atomic

         // Start additional threads on the renderSlices function
         for (int t = 0; t < _numThreads - 1; t++)
            threads.push_back(thread(renderSlicesFunction, false, currentSample));

         // Do the same work in the main thread
         renderSlicesFunction(true, currentSample);

         for (auto& thread : threads) thread.join();
      }
   }
   #else
   // Single threaded
   _next = 0;
   renderSlices(true);
   #endif

   ////////////////////////////////////////////////////////////////////////////
   _renderSec = SLScene::current->timeSec() - t1;

   SL_LOG("\nTime to render image: %6.3fsec", _renderSec)

   _state = ptFinished;
   return true;
}

//-----------------------------------------------------------------------------
/*!
Renders a slice of 4px width.
*/
void SLPathtracer::renderSlices(const bool isMainThread, SLint currentSample)
{
   // Time points
   double t1 = 0, t2 = 0;

   while (_next < _img[0].width())
   {
      const SLint minX = _next;

      // The next line should be theoretically atomic
      _next += 4;

      for (SLint x=minX; x<minX+4; ++x)
      {
         for (SLint y=0; y<_img[0].height(); ++y)
         {
            SLCol4f color(SLCol4f::BLACK);

            SLfloat primaryRand     = _uniformRandom(_generator);
            SLfloat secondaryRand   = _uniformRandom(_generator);

            // calculate direction for primary ray - scatter with random variables for antialiasing
            SLVec3f primaryDir(_BL + _pxSize*((SLfloat)(x-primaryRand+0.5f)*_LR + (SLfloat)(y-secondaryRand+0.5f)*_LU));
            primaryDir.normalize();

            // trace primary Ray
            SLRay primaryRay(_EYE, primaryDir, (SLfloat)x, (SLfloat)y);
            color += trace(&primaryRay, 0);

            // weight old and new color for continous rendering
            SLCol4f oldColor;
            if (currentSample > 1)
            {
               oldColor = _img[1].getPixeli(x, y);

               // weight old color ( examp. 3/4, 4/5, 5/6 )
               oldColor /= (SLfloat)currentSample;
               oldColor *= (SLfloat)(currentSample-1);

               // weight new color ( examp. 1/4, 1/5, 1/6 )
               color /= (SLfloat)currentSample;

               // bring them together ( examp. 4/4, 5/5, 6/6)
               color += oldColor;
            }

            color.clampMinMax(0, 1);

            // save image without gamma
            _img[1].setPixeliRGB(x, y, color);

            // gamma correctur
            color.x = pow((color.x), 1.0f / _gamma);
            color.y = pow((color.y), 1.0f / _gamma);
            color.z = pow((color.z), 1.0f / _gamma);

            // image to render
            _img[0].setPixeliRGB(x, y, color);
         }

         // update image after 500 ms
         if (isMainThread)
         {  
            t2 = SLScene::current->timeSec();
            if (t2-t1 > 0.5)
            {
               guiPTWndUpdate(); // update window
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
   SLfloat const sigma_s = 0.1f; // scattering coefficient
   SLfloat absorbtion = 1.0f;    // used to calculate absorbtion along the ray
   SLfloat scaleBy = 1.0f;       // used to scale surface reflectancy at the end of random walk
   SLRay scatterRay;
   SLfloat epsilon = _uniformRandom(_generator); // random value for ray lenght

   // Henyey Greenstein init
   SLfloat e1 = _uniformRandom(_generator);
   SLfloat e2 = _uniformRandom(_generator);
   SLfloat g = 0.5f; // scatter direction according to henyey greenstein 0.5 = uniform

   s->root3D()->hit(ray);

   // end of recursion - no object hit OR max depth reached
   if (ray->length >= SL_FLOAT_MAX || ray->depth > maxDepth())
      return SLCol4f::BLACK;

   // hit material
   SLMaterial* mat = ray->hitMat;
   ray->hitShape->preShade(ray);

   // Participating Media
   if (ray->isOutside && volumeRendering())
   {
      SLfloat rayLenght = -log(1.0 - epsilon * (1.0 - exp(-sigma_s * ray->length))) / sigma_s; // lenght of ray according to adaptive Ray march algorithm
      SLVec3f x = ray->origin + ray->dir * rayLenght; // vector to scatter point

      // Henry Greenstein phase function
      SLfloat s = 1.0 - 2.0 * e1;
      SLfloat cost = (s + 2.0f * g * g * g * (-1.0 + e1) * e1 + g * g * s + 2.0f * g * (1.0 - e1 + e1 * e1)) / ((1.0f + g * s) * (1.0f + g * s));
      SLfloat sint = sqrt(1.0f - cost * cost);
      SLVec3f dir = SLVec3f(cos(2.0f * SL_PI * e2) * sint, sin(2.0f * SL_PI * e2) * sint, cost);
      
      // uniform phase function
      //SLfloat z = 1.0 - 2.0 * e1, sint = sqrt(1.0 - z * z);
      //SLVec3f dir = SLVec3f(cos(2.0 * SL_PI * e2) * sint, sin(2.0 * SL_PI * e2) * sint, z);

      // generate orthonormal basis
      SLVec3f u, v;

      SLVec3f coVec = ray->dir;
      if(fabs(ray->dir.x) <= fabs(ray->dir.y))
      {
         if(fabs(ray->dir.x) <= fabs(ray->dir.z))
            coVec = SLVec3f(0, -ray->dir.z, ray->dir.y);
         else
            coVec = SLVec3f(-ray->dir.y, ray->dir.x, 0);
      }
      else if(fabs(ray->dir.y) <= fabs(ray->dir.z))
         coVec = SLVec3f(-ray->dir.z, 0, ray->dir.x);
      else
         coVec = SLVec3f(-ray->dir.y, ray->dir.x, 0);
      coVec.normalize();

      u.cross(ray->dir, coVec);
      v.cross(ray->dir, u);

      // direction of scatter ray
      dir = u * dir.x + v * dir.y + ray->dir * dir.z;

      // create new ray in scatter direction
      SLRay scatterRay;
      scatterRay.setDir(dir);
      scatterRay.origin.set(x);
      scatterRay.depth++;
      scatterRay.length = SL_FLOAT_MAX;
      scatterRay.type = REFLECTED;
      scatterRay.isOutside = true;
      scatterRay.x = ray->x;
      scatterRay.y = ray->y;

      // average distance to scatter point
      SLfloat ms = (1.0 - exp(-sigma_s * ray->length));

      scaleBy = 1.0f / (1.0f - ms); // scale surface reflectancy dependent on average distance

      // trace another ray in random walk or shade surface?
      if(_uniformRandom(_generator) <= ms)
      {
         return trace(&scatterRay, 1);
      }

      // calculate absorbtion along the ray
      absorbtion = exp(-sigma_s * ray->length);
   }

   SLCol4f objectEmission = mat->emission();
   SLCol4f objectColor    = SLCol4f::BLACK;

   // get base color of object
   if (ray->nodeDiffuse())            objectColor = mat->diffuse();
   else if (ray->nodeReflectance())   objectColor = mat->specular();
   else if (ray->nodeTransparency())  objectColor = mat->transmission();

   SLfloat maxEmission     = objectEmission.r > objectEmission.g && objectEmission.r > objectEmission.b ? objectEmission.r : objectEmission.g > objectEmission.b ? objectEmission.g : objectEmission.b;

   // stop recursion if lightsource is hit
   if (maxEmission > 0)
   {
      if (ray->depth == 1)
         return mat->emission() * absorbtion;
      else
         return mat->emission() * em * absorbtion;
   }

   // add absorbtion to basecolor from Participating Media
   objectColor = objectColor * absorbtion;

   // diffuse reflection
   if (ray->nodeDiffuse())
   {
      // calculate direct illumination
      finalColor += shade(ray, &objectColor) * scaleBy;

      SLRay scatter;
      ray->diffuseMC(&scatter);

      // material emission, material diffuse and recursive indirect illumination
      finalColor += (trace(&scatter, 0) & objectColor) * scaleBy;
   }

   else if (ray->nodeReflectance())
   {
      //scatter toward perfect specular direction
      SLRay reflected;
      ray->reflect(&reflected);

      //scatter around perfect reflected direction only if material not perfect
      if (mat->shininess() < SLMaterial::PERFECT)
      {
         //rotation matrix for glossy
         SLMat3f rotMat;
         SLVec3f rotAxis((SLVec3f(0.0, 0.0, 1.0)^reflected.dir).normalize());
         SLfloat rotAngle = acos(reflected.dir.z);
         rotMat.rotation(rotAngle*180.0 / SL_PI, rotAxis);
         ray->reflectMC(&reflected, rotMat);
      }

      // shininess contribution * recursive indirect illumination and matrial base color
      finalColor += ((mat->shininess() + 2.0f) / (mat->shininess() + 1.0f) * (trace(&reflected, 1) & objectColor)) * scaleBy;
   }

   else if (ray->nodeTransparency())
   {
      //scatter toward perfect transmissive direction
      SLRay refracted;
      ray->refract(&refracted);

      // init schlick's approx.
      SLVec3f rayDir = ray->dir; rayDir.normalize();
      SLVec3f refrDir = refracted.dir; refrDir.normalize();
      SLfloat n, nt;
      SLVec3f hitNormal = ray->hitNormal; hitNormal.normalize();

      // ray from outside in
      if (ray->isOutside)
      {
         n = 1.0f;
         nt = mat->kn();
      }
      // ray from inside out
      else
      {
         n = mat->kn();
         nt = 1.0f;
      }

      // calculate schlick's approx.
      SLfloat nbig, nsmall;
      nbig = n > nt ? n : nt;
      nsmall = n < nt ? n : nt;
      SLfloat R0 = ((nbig - nsmall) / (nbig + nsmall)); R0 = R0*R0;
      SLbool into = (rayDir * hitNormal) < 0;
      SLfloat c = 1 - (into ? (-rayDir * hitNormal) : (refrDir * hitNormal));
      SLfloat schlick = R0 + (1 - R0)* c * c * c * c * c;

      SLfloat P = 0.25 + 0.5 * schlick; // probability of reflectance
      SLfloat reflectionProbability = schlick / P;
      SLfloat refractionProbability = (1 - schlick) / (1 - P);

      //scatter around perfect transmissive direction only if material not perfect
      if (mat->translucency() < SLMaterial::PERFECT)
      {
         //rotation matrix for transluency
         SLMat3f rotMat;
         SLVec3f rotAxis((SLVec3f(0.0, 0.0, 1.0)^refracted.dir).normalize());
         SLfloat rotAngle = acos(refracted.dir.z);
         rotMat.rotation(rotAngle*180.0 / SL_PI, rotAxis);
         ray->refractMC(&refracted, rotMat);
      }

      // probability of reflection
      if (_uniformRandom(_generator) > (0.25 + 0.5 * schlick))
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
   SLCol4f        color = SLCol4f::BLACK;
   SLCol4f        diffuseColor = SLCol4f::BLACK;
   SLVec3f        L, N;
   SLfloat        lightDist, LdN, att, df, spotEffect, lighted = 0.0f;

   // loop over lightsources in scene
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
         lighted = (SLfloat)((LdN > 0) ? light->randomShadowTest(ray, L, lightDist) : 0);

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
/*!
Creates the inherited image in the texture class. The RT is drawn into
a texture map that is displayed with OpenGL in 2D-orthographic projection.
If
*/
void SLPathtracer::createImage(SLint width, SLint height)
{
   // Allocate image of the inherited texture class 
   if (width != _img[0].width() || height != _img[0].height())
   {  
      // Delete the OpenGL Texture if it allready exists
      if (_texName) 
      {  glDeleteTextures(1, &_texName);
         //SL_LOG("glDeleteTextures id: %u   ", _texName);
         _texName = 0;
      }

      // Dispose VBO is they allready exist
      if (_bufP.id()) _bufP.dispose();
      if (_bufT.id()) _bufT.dispose();
      if (_bufI.id()) _bufI.dispose();

      _img[0].allocate(width, height, GL_RGB);
   }
   
   // Fill image black for single PT
   if (!_continuous) _img[0].fill();
}

//-----------------------------------------------------------------------------
/*!
Creates the inherited image in the texture class. The RT is drawn into
a texture map that is displayed with OpenGL in 2D-orthographic projection.
If
*/
void SLPathtracer::prepareImage()
{
   ///////////////////////
   //  PRECALCULATIONS  //
   ///////////////////////

   SLScene* s = SLScene::current;      // scene shortcut
   SLSceneView* sv = s->activeSV();    // sceneview shortcut
   _cam = sv->_camera;        // camera shortcut

   // calculate half window width & height in world coords
   SLfloat hh = tan(SL_DEG2RAD*_cam->fov()*0.5f) * _cam->focalDist();
   SLfloat hw = hh * (SLfloat)sv->scrW() / (SLfloat)sv->scrH();

   // calculate the size of a pixel in world coords.
   _pxSize = hw * 2 / sv->scrW();

   // get camera vectors eye, lookAt, lookUp
   _cam->vm().lookAt(&_EYE, &_LA, &_LU, &_LR);

   // calculate a vector to the center (C) of the bottom left (BL) pixel
   SLVec3f C  = _LA * _cam->focalDist();
   _BL = C - hw*_LR - hh*_LU  +  _pxSize/2*_LR - _pxSize/2*_LU;

   // Allocate image of the inherited texture class 
   if (sv->scrW() != _img[0].width() || sv->scrH() != _img[0].height())
   {  
      // Delete the OpenGL Texture if it allready exists
      if (_texName) 
      {  glDeleteTextures(1, &_texName);
         //SL_LOG("glDeleteTextures id: %u   ", _texName);
         _texName = 0;
      }

      // Dispose VBO is they allready exist
      if (_bufP.id()) _bufP.dispose();
      if (_bufT.id()) _bufT.dispose();
      if (_bufI.id()) _bufI.dispose();

      _img[0].allocate(sv->scrW(), sv->scrH(), GL_RGB);
      _img[1].allocate(sv->scrW(), sv->scrH(), GL_RGB);
   }
   
   // Fill image black for single RT
   if (!_continuous) _img[0].fill();
}

//-----------------------------------------------------------------------------
/*! 
Draw the RT-Image as a textured quad in 2D-Orthographic projection
*/
void SLPathtracer::renderImage()
{
   SLScene* s = SLScene::current;
   SLSceneView* sv = s->activeSV();
   SLfloat w = (SLfloat)sv->scrW();
   SLfloat h = (SLfloat)sv->scrH();
   if (w != _img[0].width()) return;
   if (h != _img[0].height()) return;
      
   // Set orthographic projection with the size of the window
   _stateGL->projectionMatrix.ortho(0.0f, w, 0.0f, h, -1.0f, 1.0f);
   
   glClear(GL_COLOR_BUFFER_BIT);
   _stateGL->depthTest(false);
   _stateGL->multiSample(false);
   _stateGL->polygonLine(false);
   
   // build buffer object once
   if (!_bufP.id() && !_bufT.id() && !_bufI.id())
   {
      // Vertex X & Y of wnd. corners
      SLfloat P[8] = {0.0f,h, 0.0f,0.0f, w,h, w,0.0f};
      
      // Texture coords of wnd. corners
      SLfloat T[8] = {0.0f,1.0f, 0.0f,0.0f, 1.0f,1.0f, 1.0f,0.0f};
      
      // Indexes for a triangle strip
      SLushort I[4] = {0,1,2,3};
    
      _bufP.generate(P, 4, 2);
      _bufT.generate(T, 4, 2);
      _bufI.generate(I, 4, 1, SL_UNSIGNED_SHORT, SL_ELEMENT_ARRAY_BUFFER);
   }
   
   SLGLShaderProg* sp = SLScene::current->shaderProgs(TextureOnly);
   
   SLGLTexture::bindActive(0); // Enable & build texture with the ray tracing image
   SLGLTexture::fullUpdate();  // Update the OpenGL texture on each draw
   
   // Draw the character triangles                       
   sp->useProgram();
   sp->uniformMatrix4fv("u_mvpMatrix", 1,
                        (SLfloat*)&_stateGL->projectionMatrix);
   sp->uniform1i("u_texture0", 0);
   
   // bind buffers and draw 
   _bufP.bindAndEnableAttrib(sp->getAttribLocation("a_position"));
   _bufT.bindAndEnableAttrib(sp->getAttribLocation("a_texCoord"));
   
   _bufI.bindAndDrawElementsAs(SL_TRIANGLE_STRIP);
   
   _bufP.disableAttribArray();
   _bufT.disableAttribArray();
   
   GET_GL_ERROR;
}

//-----------------------------------------------------------------------------
/*! Saves the current RT image as PNG image*/
void SLPathtracer::saveImage()
{  static SLint no = 0;
   SLchar filename[255];
   sprintf(filename,"Pathtrace_%d_%d.png", no++, _maxDepth);
   _img[0].savePNG(filename);
}