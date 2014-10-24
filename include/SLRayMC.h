//#############################################################################
//  File:      SLRay.h
//  Author:    Marcus Hudritsch
//  Date:      February 2013
//  Copyright (c): 2002-2013 Marcus Hudritsch
//             This software is provide under the GNU General Public License
//             Please visit: http://opensource.org/licenses/GPL-3.0
//#############################################################################

#ifndef SLRAY_H
#define SLRAY_H

#include <stdafx.h>     
#include <randomc.h>    // high qualtiy random generators
#include <SLMaterial.h>

struct SLFace;
class  SLShape;

//-----------------------------------------------------------------------------
enum SLRayType {PRIMARY=0, REFLECTED=1, TRANSMITTED=2, SHADOW=3};
#define SL_MAXTRACE    15

//-----------------------------------------------------------------------------
//! Ray class with ray and intersection properties
/*!
Ray class for Ray Tracing. It not only holds informations about the ray itself
but also about the node hit by the ray. With that information the method 
reflect calculates a reflected ray and the method transmit calculates a 
transmitted ray.
*/
class SLRay
{  public:  
                        //! default ctor
                        SLRay       ();
                        
                        //! ctor for primary rays
                        SLRay       (SLVec3f Origin,
                                     SLVec3f Dir,
                                     SLint X,
                                     SLint Y);
                        
                        //! ctor for shadow rays
                        SLRay       (SLfloat distToLight,
                                     SLVec3f dirToLight,
                                     SLRay*  rayFromHitPoint);                      
                        
                        //! ctor for all parameters              
                        SLRay       (SLVec3f origin,
                                     SLVec3f dir,
                                     SLRayType type,
                                     SLShape* originShape,
                                     SLfloat length = SL_FLOAT_MAX,
                                     SLint depth = 1); 
                                     
            void        reflect     (SLRay* reflected);
            void        refract     (SLRay* refracted);
            bool        reflectMC   (SLRay* reflected, SLMat3f rotMat);
            void        refractMC   (SLRay* refracted, SLMat3f rotMat);
            void        diffuseMC   (SLRay* scattered);
            
            // Helper methods
     inline void        setDir      (SLVec3f Dir)
                                    {  dir = Dir; 
                                       invDir.x=(SLfloat)(1/dir.x); 
                                       invDir.y=(SLfloat)(1/dir.y); 
                                       invDir.z=(SLfloat)(1/dir.z);
                                       sign[0]=(invDir.x<0);
                                       sign[1]=(invDir.y<0);
                                       sign[2]=(invDir.z<0);
                                    }    
     inline void        setDirOS    (SLVec3f Dir)
                                    {  dirOS = Dir;
                                       invDirOS.x=(SLfloat)(1/dirOS.x);
                                       invDirOS.y=(SLfloat)(1/dirOS.y);
                                       invDirOS.z=(SLfloat)(1/dirOS.z);
                                       signOS[0]=(invDirOS.x<0);
                                       signOS[1]=(invDirOS.y<0);
                                       signOS[2]=(invDirOS.z<0);
                                    }
            SLbool      isShaded    () {return type==SHADOW && length<lightDist;}
            void        print       ();
            void        normalizeNormal();
            
            // Classic ray members
            SLVec3f     origin;        //!< Vector to the origin of ray in WS
            SLVec3f     dir;           //!< Direction vector of ray in WS
            SLVec3f     originOS;      //!< Vector to the origin of ray in OS
            SLVec3f     dirOS;         //!< Direction vector of ray in OS
            SLfloat     length;        //!< length from origin to an intersection
            SLint       depth;         //!< Recursion depth for ray tracing
            SLfloat     contrib;       //!< Current contibution of ray to color
            
            // Additional info for intersection 
            SLRayType   type;          //!< PRIMARY, REFLECTED, TRANSMITTED, SHADOW
            SLfloat     lightDist;     //!< Distance to light for shadow rays
            SLfloat     x, y;          //!< Pixel position for primary rays
            SLbool      isOutside;     //!< Flag if ray is inside of a material
            SLShape*    originShape;   //!< Points to the shape at ray origin
            SLFace*     originTria;    //!< Points to the triangle at ray origin
            SLMaterial* originMat;     //!< Points to appearance at ray origin

            // Members set after at intersection
            SLfloat     hitU, hitV;    //!< barycentric coords in hit triangle
            SLShape*    hitShape;      //!< Points to the intersected shape
            SLFace*     hitTriangle;   //!< Points to the intersected triangle
            SLMaterial* hitMat;        //!< Points to material of intersected node
            
            // Members set before shading
            SLVec3f     hitPoint;      //!< Point of intersection
            SLVec3f     hitNormal;     //!< Surface normal at intersection point
            SLCol4f     hitTexCol;     //!< Texture color at intersection point
            
            // Helpers for fast AABB intersection
            SLVec3f     invDir;        //!< Inverse ray dir for fast AABB hit in WS
            SLVec3f     invDirOS;      //!< Inverse ray dir for fast AABB hit in OS
            SLint       sign[3];       //!< Sign of invDir for fast AABB hit in WS
            SLint       signOS[3];     //!< Sign of invDir for fast AABB hit in OS
            SLfloat     tmin;          //!< min. dist. of last AABB intersection
            SLfloat     tmax;          //!< max. dist. of last AABB intersection

     // static variables for statistics
     static SLint       maxDepth;         //!< Max. recursion depth
     static SLfloat     minContrib;       //!< Min. contibution to color (1/256)
     static SLuint      reflectedRays;    //!< NO. of reflected rays
     static SLuint      refractedRays;    //!< NO. of transmitted rays
     static SLuint      shadowRays;       //!< NO. of shadow rays
     static SLuint      tirRays;          //!< NO. of TIR refraction rays
     static SLuint      tests;            //!< NO. of intersection tests
     static SLuint      intersections;    //!< NO. of intersection
     static SLint       depthReached;     //!< depth reached for a primary ray
     static SLint       maxDepthReached;  //!< max. depth reached for all rays
     static SLfloat     avgDepth;         //!< average depth reached
     static SLuint      subsampledRays;   //!< NO. of of subsampled rays
     static SLuint      subsampledPixels; //!< NO. of of subsampled pixels

     //statistics for photonmapping
     static SLlong      emittedPhotons;   //!< NO. of emitted photons from all lightsources
     static SLlong      diffusePhotons;   //!< NO. of diffusely scattered photons on surfaces
     static SLlong      reflectedPhotons; //!< NO. of reflected photons;
     static SLlong      refractedPhotons; //!< NO. of refracted photons;
     static SLlong      tirPhotons;       //!< NO. of total internal refraction photons
     static SLbool      ignoreLights;     //!< flag for gloss sampling
     static TRanrotBGenerator* random;    //!< Random generator
     
     ////////////////////
     // Photon Mapping //
     ////////////////////

            SLbool      nodeReflectance   () {return ((hitMat->specular().r >0.0f)||
                                                      (hitMat->specular().g >0.0f)||  
                                                      (hitMat->specular().b >0.0f));}
            SLbool      nodeTransparency  () {return ((hitMat->transmission().r >0.0f)||
                                                      (hitMat->transmission().g >0.0f)||  
                                                      (hitMat->transmission().b >0.0f));}
            SLbool      nodeDiffuse       () {return ((hitMat->diffuse().r >0.0f)||
                                                      (hitMat->diffuse().g >0.0f)||  
                                                      (hitMat->diffuse().b >0.0f));}
};
//-----------------------------------------------------------------------------
#endif















