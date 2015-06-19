//#############################################################################
//  File:      SLRay.cpp
//  Author:    Marcus Hudritsch
//  Date:      July 2014
//  Copyright: Marcus Hudritsch
//             This software is provide under the GNU General Public License
//             Please visit: http://opensource.org/licenses/GPL-3.0
//#############################################################################

#include <stdafx.h>           // precompiled headers
#ifdef SL_MEMLEAKDETECT
#include <nvwa/debug_new.h>   // memory leak detector
#endif

#include <SLRay.h>
#include <SLMesh.h>

// init static variables
SLint   SLRay::maxDepth = 0;
SLfloat SLRay::minContrib = 1.0 / 256.0;     
SLuint  SLRay::reflectedRays = 0;
SLuint  SLRay::refractedRays = 0;
SLuint  SLRay::ignoredRays = 0;
SLuint  SLRay::shadowRays = 0;
SLuint  SLRay::subsampledRays = 0;
SLuint  SLRay::subsampledPixels = 0;
SLuint  SLRay::tirRays = 0;
SLuint  SLRay::tests = 0;
SLuint  SLRay::intersections = 0;
SLint   SLRay::depthReached = 1;
SLint   SLRay::maxDepthReached = 0;
SLfloat SLRay::avgDepth = 0;

//-----------------------------------------------------------------------------
/*! Global uniform random number generator for numbers between 0 and 1 that are
used in SLRay, SLLightRect and SLPathtracer. So far they work perfectly with 
CPP11 multithreading.
*/
auto random01 = bind(uniform_real_distribution<SLfloat>(0.0, 1.0),
                    mt19937((SLuint)time(nullptr)));
SLfloat rnd01();
SLfloat rnd01(){return random01();}
//-----------------------------------------------------------------------------
/*! 
SLRay::SLRay default constructor
*/
SLRay::SLRay()
{  
    origin          = SLVec3f::ZERO;
    setDir(SLVec3f::ZERO);
    type            = PRIMARY;
    length          = FLT_MAX;
    depth           = 1;
    hitTriangle     = -1;
    hitPoint        = SLVec3f::ZERO;
    hitNormal       = SLVec3f::ZERO;
    hitTexCol       = SLCol4f::BLACK;
    hitNode         = nullptr;
    hitMesh         = nullptr;
    originNode      = nullptr;
    originMesh      = nullptr;
    originTriangle  = -1;
    ignoreMesh      = nullptr;
    x               = -1;
    y               = -1;
    contrib         = 1.0f;
    isOutside       = true;
    isInsideVolume  = false;
}
//-----------------------------------------------------------------------------
/*! 
SLRay::SLRay constructor for primary rays
*/
SLRay::SLRay(SLVec3f Origin, SLVec3f Dir, SLfloat X, SLfloat Y)  
{  
    origin          = Origin;
    setDir(Dir);
    type            = PRIMARY;
    length          = FLT_MAX;
    depth           = 1;
    hitTriangle     = -1;
    hitPoint        = SLVec3f::ZERO;
    hitNormal       = SLVec3f::ZERO;
    hitTexCol       = SLCol4f::BLACK;
    hitNode         = nullptr;
    hitMesh         = nullptr;
    originNode      = nullptr;
    originMesh      = nullptr;
    originTriangle  = -1;
    ignoreMesh      = nullptr;
    x               = (SLfloat)X;
    y               = (SLfloat)Y;
    contrib         = 1.0f;
    isOutside       = true;
    isInsideVolume  = false;
}
//-----------------------------------------------------------------------------
/*! 
SLRay::SLRay constructor for shadow rays
*/
SLRay::SLRay(SLfloat distToLight,
             SLVec3f dirToLight,
             SLRay*  rayFromHitPoint)  
{   
    origin          = rayFromHitPoint->hitPoint;
    setDir(dirToLight);
    type            = SHADOW;
    length          = distToLight;
    lightDist       = distToLight;
    depth           = rayFromHitPoint->depth;
    hitPoint        = SLVec3f::ZERO;
    hitNormal       = SLVec3f::ZERO;
    hitTexCol       = SLCol4f::BLACK;
    hitTriangle     = -1;
    hitNode         = nullptr;
    hitMesh         = nullptr;
    originNode      = rayFromHitPoint->hitNode;
    originMesh      = rayFromHitPoint->hitMesh;
    originTriangle  = rayFromHitPoint->hitTriangle;
    ignoreMesh      = nullptr;
    x               = rayFromHitPoint->x;
    y               = rayFromHitPoint->y;
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
    SL_LOG("Ray: O(%.2f, %.2f, %.2f), D(%.2f, %.2f, %.2f), L: %.2f\n",
           origin.x,origin.y,origin.z, dir.x,dir.y,dir.z, length);
}

//-----------------------------------------------------------------------------
/*!
SLRay::reflect calculates a secondary ray reflected at the normal, starting at 
the intersection point. All vectors must be normalized vectors.
R = 2(-I·N) N + I
*/
void SLRay::reflect(SLRay* reflected)
{
    #ifdef DEBUG_RAY
    for (SLint i = 0; i < depth; ++i) cout << " ";
    cout << "Reflect: " << hitMesh->name() << endl;
    #endif

    SLVec3f R(dir - 2.0f*(dir*hitNormal)*hitNormal);

    reflected->setDir(R);
    reflected->origin.set(hitPoint);
    reflected->depth = depth + 1;
    reflected->length = FLT_MAX;
    reflected->contrib = contrib * hitMesh->mat->kr();
    reflected->originNode = hitNode;
    reflected->originMesh = hitMesh;
    reflected->originTriangle = hitTriangle;
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
    assert(hitMesh && "hitMesh is null");

    SLVec3f T;   // refracted direction
    SLfloat eta; // refraction coefficient

    SLMaterial* originMat = originMesh ? originMesh->mat : nullptr;
    SLMaterial* hitMat = hitMesh ? hitMesh->mat : nullptr;

    #ifdef DEBUG_RAY
    for (SLint i=0; i<depth; ++i) cout << " ";
    cout << "Refract: ";
    #endif

    // Calculate index of refraction eta = Kn_Source/Kn_Destination
    // Case 1: From air into a material
    if (isOutside)
    {
        #ifdef DEBUG_RAY
        cout << "case 1:" << hitMesh->name() << endl;
        #endif
        eta = 1.0f / hitMat->kn();
    }
    else
    {   // Case 2: From inside a material back into air (outside)
        if (hitMesh==originMesh) 
        {
            #ifdef DEBUG_RAY
            cout << "case 2:" << hitMesh->name() << endl;
            #endif
            eta = hitMat->kn(); // = hitMat / 1.0
        }
        else 
        {   
            // Case 3a: From inside a material into another material 
            if (hitMesh!=ignoreMesh)
            {
                #ifdef DEBUG_RAY
                cout << "case 3a:" << hitMesh->name() << endl;
                #endif
                eta = originMat->kn() / hitMat->kn();
                refracted->ignoreMesh = originMesh;
            }
            else // Case 3b: Ignored refraction case is treated specially
            {
                #ifdef DEBUG_RAY
                cout << "case 3b::" << hitMesh->name() << endl;
                #endif
                
                // The new ray is not refracted an has the same direction
                refracted->setDir(dir);             // no direction change
                refracted->origin.set(hitPoint);    // new origin point
                refracted->contrib = contrib;       // no contribution change
                refracted->type = REFRACTED;        // remain refracted
                refracted->isOutside = isOutside;   // remain inside
                refracted->length = FLT_MAX;
                refracted->originNode = originNode; // origin remains
                refracted->originMesh = hitMesh; // mesh remains for next 3a case
                refracted->originTriangle = hitTriangle;
                refracted->ignoreMesh = nullptr;
                refracted->depth = depth + 1;
                refracted->x = x;
                refracted->y = y;
                ++ignoredRays;
                return;
            }    
        }
    }

    // Bec's formula is a little faster (from Ray Tracing News) 
    SLfloat c1 = hitNormal * -dir;
    SLfloat w  = eta * c1;
    SLfloat c2 = 1.0f + (w - eta) * (w + eta);

    if (c2 >= 0.0f) 
    {   T = eta * dir + (w - sqrt(c2)) * hitNormal;
        refracted->contrib = contrib * hitMat->kt();
        refracted->type = REFRACTED;
        // Set isOutside to false if the originMesh is the hitMesh
        refracted->isOutside = (hitMesh==originMesh || originMesh==nullptr) ? !isOutside : isOutside;
        ++refractedRays;
    } 
    else // total internal refraction results in a internal reflected ray
    {   T = 2.0f * (-dir*hitNormal) * hitNormal + dir;
        refracted->contrib = 1.0f;
        refracted->type = REFLECTED;
        refracted->isOutside = isOutside;   // remain inside
        ++tirRays;
    }
   
    refracted->setDir(T);
    refracted->origin.set(hitPoint);
    refracted->length = FLT_MAX;
    refracted->originNode = hitNode;
    refracted->originMesh = hitMesh;
    refracted->originTriangle = hitTriangle;
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
{
    SLfloat eta1, eta2;
    SLVec3f randVec;
    SLfloat shininess = hitMesh->mat->shininess();

    //scatter within specular lobe
    eta1 = rnd01();
    eta2 = SL_2PI*rnd01();
    SLfloat f1 = sqrt(1.0f-pow(eta1, 2.0f/(shininess+1.0f)));

    //tranform to cartesian
    randVec.set(f1 * cos(eta2),
                f1 * sin(eta2),
                pow(eta1, 1.0f/(shininess+1.0f)));

    //ray needs to be reset if already hit a scene node
    if(reflected->hitNode)
    {  reflected->length = FLT_MAX;
        reflected->hitNode = 0;
        reflected->hitMesh = 0;
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
{
    SLfloat eta1, eta2;
    SLVec3f randVec;
    SLfloat translucency = hitMesh->mat->translucency();

    //scatter within transmissive lobe
    eta1 = rnd01();
    eta2 = SL_2PI*rnd01();
    SLfloat f1=sqrt(1.0f-pow(eta1,2.0f/(translucency+1.0f)));

    //transform to cartesian
    randVec.set(f1*cos(eta2),
                f1*sin(eta2),
                pow(eta1,1.0f/(translucency+1.0f)));

    //ray needs to be reset if already hit a scene node
    if(refracted->hitNode)
    {   refracted->length = FLT_MAX;
        refracted->hitNode = 0;
        refracted->hitMesh = 0;
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
    scattered->originNode = hitNode;
    scattered->originMesh = hitMesh;
    scattered->type = REFLECTED;

    //calculate rotation matrix
    SLMat3f rotMat;
    SLVec3f rotAxis((SLVec3f(0.0,0.0,1.0) ^ scattered->dir).normalize());
    SLfloat rotAngle=acos(scattered->dir.z); //z*scattered.dir()
    rotMat.rotation(rotAngle*180.0f/SL_PI, rotAxis);

    //cosine distribution
    eta1 = rnd01();
    eta2 = SL_2PI*rnd01();
    eta1sqrt = sqrt(1-eta1);
    //transform to cartesian
    randVec.set(eta1sqrt * cos(eta2),
                eta1sqrt * sin(eta2),
                sqrt(eta1));

    scattered->setDir(rotMat*randVec);
}