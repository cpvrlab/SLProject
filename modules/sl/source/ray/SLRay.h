//#############################################################################
//  File:      SLRay.h
//  Date:      July 2014
//  Authors:   Marcus Hudritsch
//  License:   This software is provided under the GNU General Public License
//             Please visit: http://opensource.org/licenses/GPL-3.0
//#############################################################################

#ifndef SLRAY_H
#define SLRAY_H

#include <SLMaterial.h>
#include <SLMesh.h>

class SLNode;
class SLSceneView;

//-----------------------------------------------------------------------------
//! SLRayType enumeration for specifying ray type in ray tracing
enum SLRayType
{
    PRIMARY   = 0,
    REFLECTED = 1,
    REFRACTED = 2,
    SHADOW    = 3
};

//! Ray tracing constant for max. allowed recursion depth
#define SL_MAXTRACE 15
//-----------------------------------------------------------------------------
//! Ray class with ray and intersection properties
/*!
Ray class for Ray Tracing. It not only holds informations about the ray itself
but also about the node hit by the ray. With that information the method
reflect calculates a reflected ray and the method transmit calculates a
REFRACTED ray.
*/
class SLRay
{
public:
    //! default ctor
    explicit SLRay(SLSceneView* sv = nullptr);

    //! ctor for primary rays
    SLRay(const SLVec3f& Origin,
          const SLVec3f& Dir,
          SLfloat        X,
          SLfloat        Y,
          const SLCol4f& backColor,
          SLSceneView*   sv);

    //! ctor for shadow rays
    SLRay(SLfloat        distToLight,
          const SLVec3f& dirToLight,
          SLRay*         rayFromHitPoint);

    void reflect(SLRay* reflected) const;
    void refract(SLRay* refracted);
    bool reflectMC(SLRay* reflected, const SLMat3f& rotMat) const;
    void refractMC(SLRay* refracted, const SLMat3f& rotMat) const;
    void diffuseMC(SLRay* scattered) const;
    void print() const;

    // Helper methods
    inline void   setDir(const SLVec3f& Dir);
    inline void   setDirOS(const SLVec3f& Dir);
    inline void   normalizeNormal();
    inline SLbool isShaded() const;
    inline SLbool hitMatIsReflective() const;
    inline SLbool hitMatIsTransparent() const;
    inline SLbool hitMatIsDiffuse() const;

    // Classic ray members
    SLVec3f origin;   //!< Vector to the origin of ray in WS
    SLVec3f dir;      //!< Direction vector of ray in WS
    SLfloat length;   //!< length from origin to an intersection
    SLint   depth;    //!< Recursion depth for ray tracing
    SLfloat contrib;  //!< Current contribution of ray to color
    SLVec3f originOS; //!< Vector to the origin of ray in OS
    SLVec3f dirOS;    //!< Direction vector of ray in OS

    //! Total NO. of rays shot during RT
    static SLuint totalNumRays() { return SLRay::primaryRays +
                                          SLRay::reflectedRays +
                                          SLRay::refractedRays +
                                          SLRay::tirRays +
                                          SLRay::subsampledRays +
                                          SLRay::shadowRays; }

    // Additional info for intersection
    SLRayType    type;            //!< PRIMARY, REFLECTED, REFRACTED, SHADOW
    SLfloat      lightDist;       //!< Distance to light for shadow rays
    SLfloat      x, y;            //!< Pixel position for primary rays
    SLbool       isOutside;       //!< Flag if ray is inside of a material
    SLbool       isInsideVolume;  //!< Flag if ray is in Volume
    SLNode*      srcNode;         //!< Points to the node at ray origin
    SLMesh*      srcMesh;         //!< Points to the mesh at ray origin
    SLint        srcTriangle;     //!< Points to the triangle at ray origin
    SLCol4f      backgroundColor; //!< Background color at pixel x,y
    SLSceneView* sv;              //!< Pointer to the sceneview

    // Members set after at intersection
    SLfloat hitU, hitV;  //!< barycentric coords in hit triangle
    SLNode* hitNode;     //!< Points to the intersected node
    SLMesh* hitMesh;     //!< Points to the intersected mesh
    SLint   hitTriangle; //!< Points to the intersected triangle

    // Members set before shading
    SLVec3f hitPoint;    //!< Point of intersection
    SLVec3f hitNormal;   //!< Surface normal at intersection point
    SLCol4f hitTexColor; //!< Color at intersection for texture or color attributes
    SLfloat hitAO;       //!< Ambient occlusion factor at intersection point

    // Helpers for fast AABB intersection
    SLVec3f invDir;    //!< Inverse ray dir for fast AABB hit in WS
    SLVec3f invDirOS;  //!< Inverse ray dir for fast AABB hit in OS
    SLint   sign[3];   //!< Sign of invDir for fast AABB hit in WS
    SLint   signOS[3]; //!< Sign of invDir for fast AABB hit in OS
    SLfloat tmin;      //!< min. dist. of last AABB intersection
    SLfloat tmax;      //!< max. dist. of last AABB intersection

    // static variables for statistics
    static SLint   maxDepth;         //!< Max. recursion depth
    static SLfloat minContrib;       //!< Min. contibution to color (1/256)
    static SLuint  primaryRays;      //!< NO. of primary rays shot
    static SLuint  reflectedRays;    //!< NO. of reflected rays
    static SLuint  refractedRays;    //!< NO. of refracted rays
    static SLuint  ignoredRays;      //!< NO. of ignore refraction rays
    static SLuint  shadowRays;       //!< NO. of shadow rays
    static SLuint  tirRays;          //!< NO. of TIR refraction rays
    static SLuint  tests;            //!< NO. of intersection tests
    static SLuint  intersections;    //!< NO. of intersection
    static SLint   depthReached;     //!< depth reached for a primary ray
    static SLint   maxDepthReached;  //!< max. depth reached for all rays
    static SLfloat avgDepth;         //!< average depth reached
    static SLuint  subsampledRays;   //!< NO. of of subsampled rays
    static SLuint  subsampledPixels; //!< NO. of of subsampled pixels
};

//-----------------------------------------------------------------------------
// inline functions
//-----------------------------------------------------------------------------
//! Setter for the rays direction in world space also setting the inverse direction
inline void
SLRay::setDir(const SLVec3f& Dir)
{
    dir      = Dir;
    invDir.x = (SLfloat)(1 / dir.x);
    invDir.y = (SLfloat)(1 / dir.y);
    invDir.z = (SLfloat)(1 / dir.z);
    sign[0]  = (invDir.x < 0);
    sign[1]  = (invDir.y < 0);
    sign[2]  = (invDir.z < 0);
}
//-----------------------------------------------------------------------------
//! Setter for the rays direction in object space also setting the inverse direction
inline void
SLRay::setDirOS(const SLVec3f& Dir)
{
    dirOS      = Dir;
    invDirOS.x = (SLfloat)(1 / dirOS.x);
    invDirOS.y = (SLfloat)(1 / dirOS.y);
    invDirOS.z = (SLfloat)(1 / dirOS.z);
    signOS[0]  = (invDirOS.x < 0);
    signOS[1]  = (invDirOS.y < 0);
    signOS[2]  = (invDirOS.z < 0);
}
//-----------------------------------------------------------------------------
/*!
SLRay::normalizeNormal does a careful normalization of the normal only when the
squared length is > 1.0+FLT_EPSILON or < 1.0-FLT_EPSILON.
*/
inline void
SLRay::normalizeNormal()
{
    SLfloat nLenSqr = hitNormal.lengthSqr();
    if (nLenSqr > 1.0f + FLT_EPSILON || nLenSqr < 1.0f - FLT_EPSILON)
    {
        SLfloat len = sqrt(nLenSqr);
        hitNormal /= len;
    }
}
//-----------------------------------------------------------------------------
//! Returns true if a shadow ray hits an object on the ray to the light
inline SLbool
SLRay::isShaded() const
{
    return type == SHADOW && length < lightDist;
}
//-----------------------------------------------------------------------------
//! Returns true if the hit material specular color is not black
inline SLbool
SLRay::hitMatIsReflective() const
{
    if (!hitMesh) return false;
    SLMaterial* mat = hitMesh->mat();
    return ((mat->specular().r > 0.0f) ||
            (mat->specular().g > 0.0f) ||
            (mat->specular().b > 0.0f));
}
//-----------------------------------------------------------------------------
//! Returns true if the hit material transmission color is not black
inline SLbool
SLRay::hitMatIsTransparent() const
{
    if (!hitMesh) return false;
    SLMaterial* mat = hitMesh->mat();
    return ((mat->transmissive().r > 0.0f) ||
            (mat->transmissive().g > 0.0f) ||
            (mat->transmissive().b > 0.0f));
}
//-----------------------------------------------------------------------------
//! Returns true if the hit material diffuse color is not black
inline SLbool
SLRay::hitMatIsDiffuse() const
{
    if (!hitMesh) return false;
    SLMaterial* mat = hitMesh->mat();
    return ((mat->diffuse().r > 0.0f) ||
            (mat->diffuse().g > 0.0f) ||
            (mat->diffuse().b > 0.0f));
}
//-----------------------------------------------------------------------------
#endif
