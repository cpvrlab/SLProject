//#############################################################################
//  File:      SLRay.cpp
//  Author:    Marcus Hudritsch
//  Date:      February 2013
//  Copyright (c): 2002-2013 Marcus Hudritsch
//             This software is provide under the GNU General Public License
//             Please visit: http://opensource.org/licenses/GPL-3.0
//#############################################################################

#include <stdafx.h>           // precompiled headers
#ifdef SL_MEMLEAKDETECT
#include <nvwa/debug_new.h>   // memory leak detector
#endif

#include "SLRay.h"

// init static variables
SLint   SLRay::maxDepth = 0;
SLfloat SLRay::minContrib = 1.0 / 256.0;     
SLuint  SLRay::reflectedRays = 0;
SLuint  SLRay::refractedRays = 0;
SLuint  SLRay::shadowRays = 0;
SLuint  SLRay::subsampledRays = 0;
SLuint  SLRay::subsampledPixels = 0;
SLuint  SLRay::tirRays = 0;
SLuint  SLRay::tests = 0;
SLuint  SLRay::intersections = 0;
SLint   SLRay::depthReached = 1;
SLint   SLRay::maxDepthReached = 0;
SLfloat SLRay::avgDepth = 0;

SLlong   SLRay::emittedPhotons = 0;      
SLlong   SLRay::diffusePhotons = 0;      
SLlong   SLRay::refractedPhotons = 0;  
SLlong   SLRay::reflectedPhotons = 0; 
SLlong   SLRay::tirPhotons = 0;

SLbool   SLRay::ignoreLights = true; 

//init only once
TRanrotBGenerator* SLRay::random = new TRanrotBGenerator((unsigned)time(NULL));  

//-----------------------------------------------------------------------------
/*! 
SLRay::SLRay default constructor
*/
SLRay::SLRay()  
{  
   origin      = SLVec3f::ZERO;
   setDir(SLVec3f::ZERO);
   type        = PRIMARY;
   length      = SL_FLOAT_MAX;
   depth       = 1;
   hitTriangle = 0;
   hitPoint    = SLVec3f::ZERO;
   hitNormal   = SLVec3f::ZERO;
   hitTexCol   = SLCol4f::BLACK;
   hitShape    = 0;
   hitMat      = 0;
   originTria  = 0;
   originMat   = 0;
   x           = -1;
   y           = -1;
   contrib     = 1.0f;
   isOutside   = true;
}
//-----------------------------------------------------------------------------
/*! 
SLRay::SLRay constructor for primary rays
*/
SLRay::SLRay(SLVec3f Origin, SLVec3f Dir, SLint X, SLint Y)  
{  
   origin      = Origin;
   setDir(Dir);
   type        = PRIMARY;
   length      = SL_FLOAT_MAX;
   depth       = 1;
   hitTriangle = 0;
   hitPoint    = SLVec3f::ZERO;
   hitNormal   = SLVec3f::ZERO;
   hitTexCol   = SLCol4f::BLACK;
   hitShape    = 0;
   hitMat      = 0;
   originTria  = 0;
   originMat   = 0;
   x           = (SLfloat)X;
   y           = (SLfloat)Y;
   contrib     = 1.0f;
   isOutside   = true;
}
//-----------------------------------------------------------------------------
/*! 
SLRay::SLRay constructor for shadow rays
*/
SLRay::SLRay(SLfloat distToLight,
             SLVec3f dirToLight,
             SLRay*  rayFromHitPoint)  
{  origin      = rayFromHitPoint->hitPoint;
   setDir(dirToLight);
   type        = SHADOW;
   length      = distToLight;
   lightDist   = distToLight;
   depth       = rayFromHitPoint->depth;
   hitPoint    = SLVec3f::ZERO;
   hitNormal   = SLVec3f::ZERO;
   hitTexCol   = SLCol4f::BLACK;
   hitTriangle = 0;
   hitShape    = 0;
   hitMat      = 0;
   originTria  = rayFromHitPoint->hitTriangle;
   originMat   = rayFromHitPoint->hitMat;
   x           = rayFromHitPoint->x;
   y           = rayFromHitPoint->y;
   contrib     = 0.0f;
   isOutside   = rayFromHitPoint->isOutside;
   shadowRays++;
}
//-----------------------------------------------------------------------------
SLRay::SLRay(SLVec3f origin, 
             SLVec3f dir, 
             SLRayType type, 
             SLShape* originShape,
             SLfloat length,
             SLint depth)  
{  origin = origin;
   setDir(dir);
   type = type;
   length = length;
   depth = depth;
   hitShape = 0;
   hitPoint = SLVec3f::ZERO;
   hitNormal = SLVec3f::ZERO;
   hitMat = 0;
   originShape = originShape;
   originMat = 0;
   x = y = -1;

   if (type==SHADOW) 
   {  ++shadowRays;
      lightDist = length;
   }
}
//-----------------------------------------------------------------------------
/*!
SLRay::prints prints the rays origin (O), direction (D) and the length to the 
intersection (L) 
*/
void SLRay::print()
{  SL_LOG("Ray: O(%.2f, %.2f, %.2f), D(%.2f, %.2f, %.2f), L: %.2f\n",
          origin.x,origin.y,origin.z, dir.x,dir.y,dir.z, length);
}
//-----------------------------------------------------------------------------
/*!
SLRay::normalizeNormal does a careful normalization of the normal only when the
squared length is > 1.0+SL_EPSILON or < 1.0-SL_EPSILON.
*/
void SLRay::normalizeNormal()
{  SLfloat nLenSqr = hitNormal.lengthSqr();
   if (nLenSqr > 1.0f+SL_EPSILON || nLenSqr < 1.0f-SL_EPSILON)
   {  SLfloat len = sqrt(nLenSqr);
      hitNormal /= len;
   }
}
//-----------------------------------------------------------------------------
/*!
SLRay::reflect calculates a secondary ray reflected at the normal, starting at 
the intersection point. All vectors must be normalized vectors.
R = 2(-I·N) N + I
*/
void SLRay::reflect(SLRay* reflected)
{  SLVec3f R(dir - 2.0f*(dir*hitNormal)*hitNormal);

   reflected->setDir(R);
   reflected->origin.set(hitPoint);
   reflected->depth = depth + 1;
   reflected->length = SL_FLOAT_MAX;
   reflected->contrib = contrib * hitMat->kr();
   reflected->originMat = hitMat;
   reflected->originShape = hitShape;
   reflected->originTria = hitTriangle;
   reflected->type = REFLECTED;
   reflected->isOutside = isOutside;
   reflected->x = x;
   reflected->y = y;
   depthReached = reflected->depth;
   ++reflectedRays;
}
//-----------------------------------------------------------------------------
/*!
SLRay::refract calculates a secondary refracted ray, starting at the 
intersection point. All vectors must be normalized vectors, so the refracted 
vector T will be a unit vector too. If total internal refraction occurs a 
reflected ray is calculated instead.
Index of refraction eta = Kn_Source/Kn_Destination (Kn_Air = 1.0)
*/
void SLRay::refract(SLRay* refracted)
{  
   SLVec3f T;   // refracted direction
   SLfloat eta; // refraction coefficient
      
   // Calculate index of refraction eta = Kn_Source/Kn_Destination
   if (isOutside)
   {  if (originMat==0) // from air (outside) into a material
         eta = 1 / hitMat->kn();
      else // from another material into another one
         eta = originMat->kn() / hitMat->kn();
   } else
   {  if (originMat==hitMat) // from the inside a material into air
         eta = hitMat->kn(); // hitMat / 1
      else // from inside a material into another material
         eta = originMat->kn() / hitMat->kn();
   }

   // Bec's formula is a little faster (from Ray Tracing News) 
   SLfloat c1 = hitNormal * -dir;
   SLfloat w  = eta * c1;
   SLfloat c2 = 1.0f + (w - eta) * (w + eta);

   if (c2 >= 0.0f) 
   {  T = eta * dir + (w - sqrt(c2)) * hitNormal;
      refracted->contrib = contrib * hitMat->kt();
      refracted->type = TRANSMITTED;
      refracted->isOutside = !isOutside;
      ++refractedRays;
   } 
   else // total internal refraction results in a internal reflected ray
   {  T = 2.0f * (-dir*hitNormal) * hitNormal + dir;
      refracted->contrib = 1.0f;
      refracted->type = REFLECTED;
      refracted->isOutside = isOutside;
      ++tirRays;
   }
   
   refracted->setDir(T);
   refracted->origin.set(hitPoint);
   refracted->originMat = hitMat;
   refracted->length = SL_FLOAT_MAX;
   refracted->originShape = hitShape;
   refracted->originTria = hitTriangle;
   refracted->depth = depth + 1;
   refracted->x = x;
   refracted->y = y;
   depthReached = refracted->depth;
}
//-----------------------------------------------------------------------------
/*!
SLRay::reflectMC scatters a ray around perfect specular direction according to 
shininess (for higher shininess the ray is less scattered). This is used for 
path tracing and distributed ray tracing as well as for photon scattering. 
The direction is calculated according to MCCABE. The created direction is 
along z-axis and then transformed to lie along specular direction with 
rotationMatrix rotMat. The rotation matrix must be precalculated (stays the 
same for each ray sample, needs to be be calculated only once)
*/
bool SLRay::reflectMC(SLRay* reflected,SLMat3f rotMat)
{  SLfloat eta1, eta2;
   SLVec3f randVec;
   SLfloat shininess = hitMat->shininess();

   //scatter within specular lobe
   eta1 = (SLfloat)random->Random();
   eta2 = SL_2PI*(SLfloat)random->Random();
   SLfloat f1 = sqrt(1.0f-pow(eta1, 2.0f/(shininess+1.0f)));

   //tranform to cartesian
   randVec.set(f1 * cos(eta2),
               f1 * sin(eta2),
               pow(eta1, 1.0f/(shininess+1.0f)));

   //ray needs to be reset if already hit a shape
   if(reflected->hitShape)
   {  reflected->length = SL_FLOAT_MAX;
      reflected->hitShape = 0;
      reflected->hitPoint = SLVec3f::ZERO;
      reflected->hitNormal = SLVec3f::ZERO;
   }
   
   //apply rotation
   reflected->setDir(rotMat*randVec);
   
   //true if in direction of normal
   return (hitNormal * reflected->dir >= 0.0f);
}
//-----------------------------------------------------------------------------
/*!
SLRay::refractMC scatters a ray around perfect transmissive direction according 
to translucency (for higher translucency the ray is less scattered).
This is used for path tracing and distributed ray tracing as well as for photon 
scattering. The direction is calculated the same as with specular scattering
(see reflectMC). The created direction is along z-axis and then transformed to 
lie along transmissive direction with rotationMatrix rotMat. The rotation 
matrix must be precalculated (stays the same for each ray sample, needs to be 
be calculated only once)
*/
void SLRay::refractMC(SLRay* refracted,SLMat3f rotMat)
{  SLfloat eta1, eta2;
   SLVec3f randVec;
   SLfloat translucency = hitMat->translucency();

   //scatter within transmissive lobe
   eta1 = (SLfloat)random->Random(); 
   eta2 = SL_2PI*(SLfloat)random->Random();
   SLfloat f1=sqrt(1.0f-pow(eta1,2.0f/(translucency+1.0f)));

   //transform to cartesian
   randVec.set( f1*cos(eta2),
                f1*sin(eta2),
                pow(eta1,1.0f/(translucency+1.0f)));

   //ray needs to be reset if already hit a shape
   if(refracted->hitShape)
   {  refracted->length = SL_FLOAT_MAX;
      refracted->hitShape = 0;
      refracted->hitPoint = SLVec3f::ZERO;
      refracted->hitNormal = SLVec3f::ZERO;
   }
   
   refracted->setDir(rotMat*randVec);
}
//-----------------------------------------------------------------------------
/*!
SLRay::diffuseMC scatters a ray around hit normal (cosine distribution).
This is only used for photonmapping(russian roulette).
The random direction lies around z-Axis and is then transformed by a rotation 
matrix to lie along the normal. The direction is calculated according to MCCABE
*/
void SLRay::diffuseMC(SLRay* scattered)
{
   SLVec3f randVec;
   SLfloat eta1,eta2,eta1sqrt;

   scattered->setDir(hitNormal);
   scattered->origin = hitPoint;
   scattered->depth = depth+1;
   depthReached = scattered->depth;
   
   // for reflectance the start material stays the same
   scattered->originMat = hitMat;
   scattered->originShape = hitShape;
   scattered->type = REFLECTED;

   //calculate rotation matrix
   SLMat3f rotMat;
   SLVec3f rotAxis((SLVec3f(0.0,0.0,1.0) ^ scattered->dir).normalize());
   SLfloat rotAngle=acos(scattered->dir.z); //z*scattered.dir()
   rotMat.rotation(rotAngle*180.0f/SL_PI, rotAxis);

   //cosine distribution
   eta1 = (SLfloat)random->Random(); 
   eta2 = SL_2PI*(SLfloat)random->Random();
   eta1sqrt = sqrt(1-eta1);
   //transform to cartesian
   randVec.set(eta1sqrt * cos(eta2),
               eta1sqrt * sin(eta2),
               sqrt(eta1));

   scattered->setDir(rotMat*randVec);
}
//-----------------------------------------------------------------------------
