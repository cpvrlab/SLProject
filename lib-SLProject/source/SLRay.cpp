//#############################################################################
//  File:      SLRay.cpp
//  Author:    Marcus Hudritsch
//  Date:      July 2014
//  Copyright: Marcus Hudritsch
//             This software is provide under the GNU General Public License
//             Please visit: http://opensource.org/licenses/GPL-3.0
//#############################################################################

#include <stdafx.h> // Must be the 1st include followed by  an empty line

#ifdef SL_MEMLEAKDETECT
#    include <nvwa/debug_new.h> // memory leak detector
#endif

#include <SLRay.h>
#include <SLSceneView.h>

// init static variables
SLint   SLRay::maxDepth         = 0;
SLfloat SLRay::minContrib       = 1.0 / 256.0;
SLuint  SLRay::reflectedRays    = 0;
SLuint  SLRay::refractedRays    = 0;
SLuint  SLRay::ignoredRays      = 0;
SLuint  SLRay::shadowRays       = 0;
SLuint  SLRay::subsampledRays   = 0;
SLuint  SLRay::subsampledPixels = 0;
SLuint  SLRay::tirRays          = 0;
SLuint  SLRay::tests            = 0;
SLuint  SLRay::intersections    = 0;
SLint   SLRay::depthReached     = 1;
SLint   SLRay::maxDepthReached  = 0;
SLfloat SLRay::avgDepth         = 0;

//-----------------------------------------------------------------------------
/*! Global uniform random number generator for numbers between 0 and 1 that are
used in SLRay, SLLightRect and SLPathtracer. So far they work perfectly with 
CPP11 multithreading.
*/
auto    random01 = bind(uniform_real_distribution<SLfloat>(0.0, 1.0),
                     mt19937((SLuint)time(nullptr)));
SLfloat rnd01();
SLfloat rnd01()
{
    return random01();
}
//-----------------------------------------------------------------------------
/*! 
SLRay::SLRay default constructor
*/
SLRay::SLRay(SLSceneView* sceneView)
{
    origin = SLVec3f::ZERO;
    setDir(SLVec3f::ZERO);
    type           = PRIMARY;
    length         = FLT_MAX;
    depth          = 1;
    hitTriangle    = -1;
    hitPoint       = SLVec3f::ZERO;
    hitNormal      = SLVec3f::ZERO;
    hitColor       = SLCol4f::BLACK;
    hitNode        = nullptr;
    hitMesh        = nullptr;
    srcNode        = nullptr;
    srcMesh        = nullptr;
    srcTriangle    = -1;
    x              = -1;
    y              = -1;
    contrib        = 1.0f;
    isOutside      = true;
    isInsideVolume = false;
    sv             = sceneView;
}
//-----------------------------------------------------------------------------
/*! 
SLRay::SLRay constructor for primary rays
*/
SLRay::SLRay(SLVec3f      Origin,
             SLVec3f      Dir,
             SLfloat      X,
             SLfloat      Y,
             SLCol4f      backColor,
             SLSceneView* sceneView)
{
    origin = Origin;
    setDir(Dir);
    type            = PRIMARY;
    length          = FLT_MAX;
    depth           = 1;
    hitTriangle     = -1;
    hitPoint        = SLVec3f::ZERO;
    hitNormal       = SLVec3f::ZERO;
    hitColor        = SLCol4f::BLACK;
    hitNode         = nullptr;
    hitMesh         = nullptr;
    srcNode         = nullptr;
    srcMesh         = nullptr;
    srcTriangle     = -1;
    x               = (SLfloat)X;
    y               = (SLfloat)Y;
    contrib         = 1.0f;
    isOutside       = true;
    isInsideVolume  = false;
    backgroundColor = backColor;
    sv              = sceneView;
}
//-----------------------------------------------------------------------------
/*! 
SLRay::SLRay constructor for shadow rays
*/
SLRay::SLRay(SLfloat distToLight,
             SLVec3f dirToLight,
             SLRay*  rayFromHitPoint)
{
    origin = rayFromHitPoint->hitPoint;
    setDir(dirToLight);
    type            = SHADOW;
    length          = distToLight;
    lightDist       = distToLight;
    depth           = rayFromHitPoint->depth;
    hitPoint        = SLVec3f::ZERO;
    hitNormal       = SLVec3f::ZERO;
    hitColor        = SLCol4f::BLACK;
    hitTriangle     = -1;
    hitNode         = nullptr;
    hitMesh         = nullptr;
    srcNode         = rayFromHitPoint->hitNode;
    srcMesh         = rayFromHitPoint->hitMesh;
    srcTriangle     = rayFromHitPoint->hitTriangle;
    x               = rayFromHitPoint->x;
    y               = rayFromHitPoint->y;
    backgroundColor = rayFromHitPoint->backgroundColor;
    sv              = rayFromHitPoint->sv;
    contrib         = 0.0f;
    isOutside       = rayFromHitPoint->isOutside;
    shadowRays++;
}
//-----------------------------------------------------------------------------
/*!
SLRay::prints prints the rays origin (O), direction (D) and the length to the 
intersection (L) 
*/
void SLRay::print() const
{
    SL_LOG("Ray: O(%.2f, %.2f, %.2f), D(%.2f, %.2f, %.2f), L: %.2f",
           origin.x,
           origin.y,
           origin.z,
           dir.x,
           dir.y,
           dir.z,
           length);
}
//-----------------------------------------------------------------------------
/*!
SLRay::reflect calculates a secondary ray reflected at the normal, starting at 
the intersection point. All vectors must be normalized vectors.
R = 2(-I*N) N + I
*/
void SLRay::reflect(SLRay* reflected)
{
#ifdef DEBUG_RAY
    for (SLint i = 0; i < depth; ++i)
        cout << " ";
    cout << "Reflect: " << hitMesh->name() << endl;
#endif

    SLVec3f R(dir - 2.0f * (dir * hitNormal) * hitNormal);

    reflected->setDir(R);
    reflected->origin.set(hitPoint);
    reflected->depth       = depth + 1;
    reflected->length      = FLT_MAX;
    reflected->contrib     = contrib * hitMesh->mat()->kr();
    reflected->srcNode     = hitNode;
    reflected->srcMesh     = hitMesh;
    reflected->srcTriangle = hitTriangle;
    reflected->type        = REFLECTED;
    reflected->isOutside   = isOutside;
    reflected->x           = x;
    reflected->y           = y;
    reflected->sv          = sv;
    if (sv->skybox())
        reflected->backgroundColor = sv->skybox()->colorAtDir(reflected->dir);
    else
        reflected->backgroundColor = backgroundColor;

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
We are using a formula by Xavier Bec that is a little faster:
http://www.realtimerendering.com/resources/RTNews/html/rtnv10n1.html#art3
*/
void SLRay::refract(SLRay* refracted)
{
    assert(hitMesh && "hitMesh is null");

    SLVec3f T;   // refracted direction
    SLfloat eta; // refraction coefficient

    SLfloat c1           = hitNormal.dot(-dir);
    SLbool  hitFrontSide = c1 > 0.0f;

    SLMaterial* srcMat    = srcMesh ? srcMesh->mat() : nullptr;
    SLMaterial* hitMat    = hitMesh ? hitMesh->mat() : nullptr;
    SLMaterial* hitMatOut = hitMesh ? hitMesh->matOut() : nullptr;

#ifdef DEBUG_RAY
    for (SLint i = 0; i < depth; ++i)
        cout << " ";
    cout << "Refract: ";
#endif

    // Calculate index of refraction eta = Kn_Source/Kn_Destination
    // Case 1: From air into a mesh
    if (isOutside)
    {
        eta = 1.0f / hitMat->kn();
    }
    else
    { // Case 2: From inside the same mesh
        if (hitMesh == srcMesh)
        {
            if (hitMatOut) // Case 2a: into another material
                eta = hitMat->kn() / hitMatOut->kn();
            else                    // Case 2b: into air
                eta = hitMat->kn(); // = hitMat / 1.0
        }
        else
        { // Case 3: We hit inside another material from the front
            if (hitFrontSide)
            {
                if (hitMatOut)
                    eta = hitMatOut->kn() / hitMat->kn();
                else
                { // Mesh hit without outside material before leaving another mesh.
                    // This should not happen, but can due to float inaccuracies
                    eta = srcMat->kn() / hitMat->kn();
                }
            }
            else // Case 4: We hit inside another material from behind
            {
                if (hitMatOut) // Case 4a: into another material
                    eta = hitMat->kn() / hitMatOut->kn();
                else                    // Case 4b: into air
                    eta = hitMat->kn(); // = hitMat / 1.0
            }
        }
    }

    // Invert the hit normal if ray hit backside for correct refraction
    if (!hitFrontSide)
    {
        c1 *= -1.0f;
        hitNormal *= -1.0f;
    }

    SLfloat w  = eta * c1;
    SLfloat c2 = 1.0f + (w - eta) * (w + eta);

    if (c2 >= 0.0f)
    {
        T                  = eta * dir + (w - sqrt(c2)) * hitNormal;
        refracted->contrib = contrib * hitMat->kt();
        refracted->type    = REFRACTED;

        if (isOutside)
            refracted->isOutside = false;
        else // inside
        {
            if (srcMesh == hitMesh)
            {
                if (hitMatOut)
                    refracted->isOutside = false;
                else
                    refracted->isOutside = true;
            }
            else
            {
                if (hitFrontSide)
                    refracted->isOutside = false; // hit from front
                else
                    refracted->isOutside = true; // hit from back
            }
        }

        ++refractedRays;
    }
    else // total internal refraction results in a internal reflected ray
    {
        T                    = 2.0f * (-dir * hitNormal) * hitNormal + dir;
        refracted->contrib   = 1.0f;
        refracted->type      = REFLECTED;
        refracted->isOutside = isOutside; // remain inside
        ++tirRays;
    }

    refracted->setDir(T);
    refracted->origin.set(hitPoint);
    refracted->length      = FLT_MAX;
    refracted->srcNode     = hitNode;
    refracted->srcMesh     = hitMesh;
    refracted->srcTriangle = hitTriangle;
    refracted->depth       = depth + 1;
    refracted->x           = x;
    refracted->y           = y;
    refracted->sv          = sv;
    if (sv->skybox())
        refracted->backgroundColor = sv->skybox()->colorAtDir(refracted->dir);
    else
        refracted->backgroundColor = backgroundColor;
    depthReached = refracted->depth;

#ifdef DEBUG_RAY
    cout << hitMesh->name();
    if (isOutside)
        cout << ",out";
    else
        cout << ",in";
    if (refracted->isOutside)
        cout << ">out";
    else
        cout << ">in";
    cout << ", dir: " << refracted->dir.toString();
    cout << ", contrib: " << Utils::toString(refracted->contrib, 2);
    cout << endl;
#endif
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
bool SLRay::reflectMC(SLRay* reflected, SLMat3f rotMat)
{
    SLfloat eta1, eta2;
    SLVec3f randVec;
    SLfloat shininess = hitMesh->mat()->shininess();

    //scatter within specular lobe
    eta1       = rnd01();
    eta2       = Utils::TWOPI * rnd01();
    SLfloat f1 = sqrt(1.0f - pow(eta1, 2.0f / (shininess + 1.0f)));

    //tranform to cartesian
    randVec.set(f1 * cos(eta2),
                f1 * sin(eta2),
                pow(eta1, 1.0f / (shininess + 1.0f)));

    //ray needs to be reset if already hit a scene node
    if (reflected->hitNode)
    {
        reflected->length    = FLT_MAX;
        reflected->hitNode   = nullptr;
        reflected->hitMesh   = nullptr;
        reflected->hitPoint  = SLVec3f::ZERO;
        reflected->hitNormal = SLVec3f::ZERO;
    }

    //apply rotation
    reflected->setDir(rotMat * randVec);

    // Set pixel and background
    reflected->x  = x;
    reflected->y  = y;
    reflected->sv = sv;
    if (sv->skybox())
        reflected->backgroundColor = sv->skybox()->colorAtDir(reflected->dir);
    else
        reflected->backgroundColor = backgroundColor;

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
void SLRay::refractMC(SLRay* refracted, SLMat3f rotMat)
{
    SLfloat eta1, eta2;
    SLVec3f randVec;
    SLfloat translucency = hitMesh->mat()->translucency();

    //scatter within transmissive lobe
    eta1       = rnd01();
    eta2       = Utils::TWOPI * rnd01();
    SLfloat f1 = sqrt(1.0f - pow(eta1, 2.0f / (translucency + 1.0f)));

    //transform to cartesian
    randVec.set(f1 * cos(eta2),
                f1 * sin(eta2),
                pow(eta1, 1.0f / (translucency + 1.0f)));

    //ray needs to be reset if already hit a scene node
    if (refracted->hitNode)
    {
        refracted->length    = FLT_MAX;
        refracted->hitNode   = nullptr;
        refracted->hitMesh   = nullptr;
        refracted->hitPoint  = SLVec3f::ZERO;
        refracted->hitNormal = SLVec3f::ZERO;
    }

    // Apply rotation
    refracted->setDir(rotMat * randVec);

    // Set pixel and background
    refracted->x  = x;
    refracted->y  = y;
    refracted->sv = sv;
    if (sv->skybox())
        refracted->backgroundColor = sv->skybox()->colorAtDir(refracted->dir);
    else
        refracted->backgroundColor = backgroundColor;
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
    SLfloat eta1, eta2, eta1sqrt;

    scattered->setDir(hitNormal);
    scattered->origin = hitPoint;
    scattered->depth  = depth + 1;
    depthReached      = scattered->depth;

    // for reflectance the start material stays the same
    scattered->srcNode = hitNode;
    scattered->srcMesh = hitMesh;
    scattered->type    = REFLECTED;

    //calculate rotation matrix
    SLMat3f rotMat;
    SLVec3f rotAxis((SLVec3f(0.0, 0.0, 1.0) ^ scattered->dir).normalize());
    SLfloat rotAngle = acos(scattered->dir.z); //z*scattered.dir()
    rotMat.rotation(rotAngle * 180.0f * Utils::ONEOVERPI, rotAxis);

    //cosine distribution
    eta1     = rnd01();
    eta2     = Utils::TWOPI * rnd01();
    eta1sqrt = sqrt(1 - eta1);

    //transform to cartesian
    randVec.set(eta1sqrt * cos(eta2),
                eta1sqrt * sin(eta2),
                sqrt(eta1));

    // Apply rotation
    scattered->setDir(rotMat * randVec);

    // Set pixel and background
    scattered->x  = x;
    scattered->y  = y;
    scattered->sv = sv;
    if (sv->skybox())
        scattered->backgroundColor = sv->skybox()->colorAtDir(scattered->dir);
    else
        scattered->backgroundColor = backgroundColor;
}
//-----------------------------------------------------------------------------
