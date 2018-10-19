//#############################################################################
//  File:      SLPhotonMapper.cpp
//  Author:    Michael Strub, Stefan Traud, Marcus Hudritsch
//  Date:      September 2011 (HS11)
//  Copyright (c): 2002-2013 Marcus Hudritsch
//             This software is provide under the GNU General Public License
//             Please visit: http://opensource.org/licenses/GPL-3.0
//#############################################################################

#include <stdafx.h>           // precompiled headers
#ifdef SL_MEMLEAKDETECT
#include <nvwa/debug_new.h>   // memory leak detector
#endif
#ifdef SL_OMP
#include <omp.h>              // OpenMP
#endif

#include "SLPhotonMap.h"
#include "SLPhotonMapper.h"
#include "SLCamera.h"
#include "SLSceneView.h"
#include "SLLightSphere.h"
#include "SLLightRect.h"
#include "SLLight.h"
#include "SLGroup.h"
#include "SLMesh.h"
#include "SLGLTexture.h"
#include "SLSamples2D.h"
#include "SLGLShaderProg.h"

//-----------------------------------------------------------------------------
SLPhotonMapper::SLPhotonMapper()
{  
   name("myCoolPhotonMapper");
   
   // random vars are initialized with current time
   _random = new TRanrotBGenerator((unsigned)time(NULL));
   _russianRandom = new TRanrotBGenerator((unsigned)time(NULL)+1);

   // photonmapping
   _photonsToEmit=0;
   _mapCaustic = new SLPhotonMap();
   _mapGlobal = new SLPhotonMap();
   _mapCausticGotFull = false;
   _mapGlobalGotFull = false;
   _gamma = 2.2f;
}
//-----------------------------------------------------------------------------
SLPhotonMapper::~SLPhotonMapper()
{  
   delete _mapCaustic;
   delete _mapGlobal;
   delete _random;
   delete _russianRandom;
   delete SLRay::random;
}
//-----------------------------------------------------------------------------
/*!
*/
SLbool SLPhotonMapper::render()
{  
   return true;
}
//-----------------------------------------------------------------------------
/*!
*/
SLCol4f SLPhotonMapper::trace(SLRay* ray)
{  
   SLScene* s = SLScene::current;
   SLCol4f color(s->backColor());
   
   return color;
}
//-----------------------------------------------------------------------------
/*!
*/
SLCol4f SLPhotonMapper::shade(SLRay* ray)
{  
   //SLScene*    s = SLScene::current;
   SLCol4f     localColor = SLCol4f::BLACK;
   
   return localColor;  
}
//-----------------------------------------------------------------------------
/*!
Inits the photonmap members with the photonmap settings
*/
void SLPhotonMapper::setPhotonmaps(SLlong photonsToEmit,
                                   SLlong maxCausticStoredPhotons, 
                                   SLuint maxCausticEstimationPhotons, 
                                   SLfloat maxCausticEstimationRadius,
                                   SLlong maxGlobalStoredPhotons, 
                                   SLuint maxGlobalEstimationPhotons, 
                                   SLfloat maxGlobalEstimationRadius)
{
   _mapCaustic->setPhotonMapParams(maxCausticStoredPhotons, 
                                   maxCausticEstimationPhotons, 
                                   maxCausticEstimationRadius);
   _mapGlobal->setPhotonMapParams(maxGlobalStoredPhotons, 
                                  maxGlobalEstimationPhotons, 
                                  maxGlobalEstimationRadius);
   _photonsToEmit = photonsToEmit;
}
//-----------------------------------------------------------------------------
/*!
Photons are scattered (or absorbed) according to surface properties. 
This is done by russian roulette.
Photons are stored on diffuse surfaces only.
*/
void SLPhotonMapper::photonScatter(SLRay* photon, 
                                   SLVec3f power, 
                                   SLPhotonType photonType)//, SLint RGB)
{
   SLScene* s = SLScene::current;      // scene shortcut
   s->root3D()->hit(photon);
   
   if (photon->length < SL_FLOAT_MAX)
   {  
      //photons "absorbed" by luminaire
      if (typeid(*photon->hitShape)==typeid(SLLightSphere) ||
          typeid(*photon->hitShape)==typeid(SLLightRect)) 
         return;

      //abort if maps are full or depth>100
      if((_mapCaustic->isFull() && _mapGlobal->isFull()) || 
         photon->depth>100) //physically plausible ;-)
         return;
      
      photon->normalizeNormal();
      SLMaterial* mat = photon->hitMat;

      //store photon if diffuse surface and not from light
      if (photon->nodeDiffuse() && photonType!=LIGHT)//photon->type()!=PRIMARY)
      {
         if(photonType!=CAUSTIC)
            _mapGlobal->store(photon->hitPoint,photon->dir,power);
         else
         {
            _mapCaustic->store(photon->hitPoint,photon->dir,power);
            return; // caustic photons "die" on diffuse surfaces
         }
      }

      //calculate average of materials
      SLfloat avgDiffuse     = (mat->diffuse().x+mat->diffuse().y+mat->diffuse().z)/3.0f;
      SLfloat avgSpecular    = (mat->specular().x+mat->specular().y+mat->specular().z)/3.0f;
      SLfloat avgTransmission= (mat->transmission().x+mat->transmission().y+mat->transmission().z)/3.0f;
      SLfloat eta            = _russianRandom->Random();

      //Decide type of photon (Global or Caustic) if from light
      if (photonType == LIGHT)
      {
         if ((eta*(avgDiffuse+avgSpecular+avgTransmission))<=avgDiffuse)
            photonType=GLOBAL;
         else
            photonType=CAUSTIC;
      }

      //Russian Roulette
      if (eta <= avgDiffuse) 
      {
         //scattered diffuse (cosine distribution around normal)
         SLRay scattered;
         photon->diffuseMC(&scattered);
         //adjust power
         power.x*=(mat->diffuse().x/avgDiffuse);
         power.y*=(mat->diffuse().y/avgDiffuse);
         power.z*=(mat->diffuse().z/avgDiffuse);

         ++SLRay::diffusePhotons;
         photonScatter(&scattered, power, photonType);

         
      }
      else if (eta <= avgDiffuse+avgSpecular)
      {
         //scatter toward perfect specular direction
         SLRay scattered;
         photon->reflect(&scattered);
         
         //scatter around perfect reflected direction only if material not perfect
         if(photon->hitMat->shininess() < SLMaterial::PERFECT)
         {
            //rotation matrix
            SLMat3f rotMat;
            SLVec3f rotAxis((SLVec3f(0.0,0.0,1.0) ^ scattered.dir).normalize());
            SLfloat rotAngle=acos(scattered.dir.z);//z*scattered.dir()
            rotMat.rotation(rotAngle*180.0/SL_PI,rotAxis);
            photon->reflectMC(&scattered,rotMat);
         }
         
         //avoid scattering into surface
         if (scattered.dir*photon->hitNormal >= 0.0f)
         {
            //adjust power
            power.x*=(mat->specular().x/avgSpecular);
            power.y*=(mat->specular().y/avgSpecular);
            power.z*=(mat->specular().z/avgSpecular);

            ++SLRay::reflectedPhotons;
            photonScatter(&scattered,power,photonType);
         }
      }
      else if (eta <= avgDiffuse+avgSpecular+avgTransmission) //scattered refracted
      {
         //scatter toward perfect transmissive direction
         SLRay scattered;
         photon->refract(&scattered);
         
         //scatter around perfect transmissive direction only if material not perfect
         if(photon->hitMat->translucency() < SLMaterial::PERFECT)
         {
            //rotation matrix
            SLMat3f rotMat;
            SLVec3f rotAxis((SLVec3f(0.0,0.0,1.0) ^ scattered.dir).normalize());
            SLfloat rotAngle=acos(scattered.dir.z);//z*scattered.dir()
            rotMat.rotation(rotAngle*180.0/SL_PI,rotAxis);
            photon->refractMC(&scattered,rotMat);
         }
         SLVec3f N = -photon->hitNormal;
         if(scattered.type==REFLECTED) N*=-1.0;  //In case of total reflection invert the Normal
         
         if (scattered.dir*N >= 0.0f)
         {
            //adjust power
            power.x*=(mat->transmission().x/avgTransmission);
            power.y*=(mat->transmission().y/avgTransmission);
            power.z*=(mat->transmission().z/avgTransmission);
            if(scattered.type==TRANSMITTED) ++SLRay::refractedPhotons; else ++SLRay::tirPhotons;
            photonScatter(&scattered,power,photonType);
         }
      }
      else //absorbed [rest in peace]
      {
      }
   }
}
//-----------------------------------------------------------------------------
/*!Creates photons (ray with power) from a point light source and sends 
it into the scene. 
Note that a point light has no cosine distribution and no surface area
*/
void SLPhotonMapper::photonEmission(SLLight* light)
{
   SLfloat eta1,eta2,f1,f2;
   SLVec3f C,N;
   SLVec3f power;
      
   C = light->positionWS();

   //because point lights have no surface use only radiance for power of emitted photons
   power.set(light->diffuse().r,light->diffuse().g,light->diffuse().b);

   //progress
   SLlong maxPhotons=_mapCaustic->maxStoredPhotons()+_mapGlobal->maxStoredPhotons();
   SLlong curPhotons=_mapCaustic->storedPhotons()+_mapGlobal->storedPhotons();;
   
   SLlong emitted=0;

   //shoot the photons as long as maps are not full
   while(emitted<light->photons() && !(_mapCaustic->isFull()&&(_mapGlobal->isFull())))
   {
      //progress
      if(emitted%1000==0)
      {  curPhotons=_mapCaustic->storedPhotons()+_mapGlobal->storedPhotons();
         printf("\b\b\b\b%3.0f%%",(SLfloat)curPhotons/(SLfloat)maxPhotons*100.0f);
      }

      //create spherical random direction
      eta1=_random->Random();eta2=_random->Random();
      f1 = SL_2PI*eta2;
      f2 = 2.0f * sqrt(eta1 * (1-eta1));

      //direction in cartesian coordinates
      N.set(cos(f1)*f2, sin(f1)*f2, (1.0-2.0*eta1));
      
      //create and emit photon
      SLRay scattered(C,N,PRIMARY, (SLShape*)light, SL_FLOAT_MAX, 1);
      photonScatter(&scattered,power,LIGHT);

      emitted++;
      //scaling of stored photons is necessary if one of the maps was filled by this photon
      //(because emission of photons continues in order to fill the other map)
      if(_mapCaustic->isFull() && !_mapCausticGotFull)
      {  _mapCausticGotFull = true;
         _mapCaustic->scalePhotonPower(1.0f/SLfloat(emitted));
      }
      if(_mapGlobal->isFull() && !_mapGlobalGotFull)
      {  _mapGlobalGotFull = true;
         _mapGlobal->scalePhotonPower(1.0f/SLfloat(emitted));
      }

   }

   //scale all stored photons of this light source
   if(emitted)
   {  SLRay::emittedPhotons+=emitted;
      _mapCaustic->scalePhotonPower(1.0f/SLfloat(emitted));
      _mapGlobal->scalePhotonPower(1.0f/SLfloat(emitted));
   }
}
//-----------------------------------------------------------------------------
