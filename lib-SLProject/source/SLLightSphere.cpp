//#############################################################################
//  File:      SLLightSphere.cpp
//  Author:    Marcus Hudritsch
//  Date:      July 2014
//  Codestyle: https://github.com/cpvrlab/SLProject/wiki/Coding-Style-Guidelines
//  Copyright: Marcus Hudritsch
//             This software is provide under the GNU General Public License
//             Please visit: http://opensource.org/licenses/GPL-3.0
//#############################################################################

#include <stdafx.h>           // precompiled headers
#ifdef SL_MEMLEAKDETECT
#include <nvwa/debug_new.h>   // memory leak detector
#endif

#include <SLLightSphere.h>
#include <SLSphere.h>
#include <SLRay.h>
#include <SLScene.h>
#include <SLSceneView.h>
#include <SLMaterial.h>
#include <SLMesh.h>

//-----------------------------------------------------------------------------
SLLightSphere::SLLightSphere(SLfloat radius, SLbool hasMesh) :
               SLNode("LightSphere")
{  
    _radius = radius;
    _samples.samples(1,1,false);

    if (hasMesh)
    {   SLMaterial* mat = new SLMaterial("LightSphereMeshMat", 
                                         SLCol4f::BLACK, 
                                         SLCol4f::BLACK);
        addMesh(new SLSphere(radius, 16, 16, "LightSphereMesh", mat));
    }

    init();
}
//-----------------------------------------------------------------------------
SLLightSphere::SLLightSphere(SLfloat posx, 
                             SLfloat posy, 
                             SLfloat posz,
                             SLfloat radius,
                             SLfloat ambiPower,
                             SLfloat diffPower,
                             SLfloat specPower,
                             SLbool hasMesh) : 
                SLNode("Sphere Light"), 
                SLLight(ambiPower, diffPower, specPower)
{  
    _radius = radius;
    _samples.samples(1,1,false);
    translate(posx, posy, posz, TS_Object);

    if (hasMesh)
    {   SLMaterial* mat = new SLMaterial("LightSphereMeshMat", 
                                         SLCol4f::BLACK, 
                                         SLCol4f::BLACK);
        addMesh(new SLSphere(radius, 16, 16, "LightSphereMesh", mat));
    }
    init();
}
//-----------------------------------------------------------------------------
/*! 
SLLightSphere::init sets the light id, the light states & creates an 
emissive mat.
@todo properly remove this function and find a clean way to init lights in a scene
*/
void SLLightSphere::init()
{  
    // Check if OpenGL lights are available
    if (SLScene::current->lights().size() >= SL_MAX_LIGHTS) 
        SL_EXIT_MSG("Max. NO. of lights is exceeded!");

    // Add the light to the lights array of the scene
    if (_id==-1)
    {   _id = (SLint)SLScene::current->lights().size();
        SLScene::current->lights().push_back(this);
    }
   
    // Set the OpenGL light states
    setState();
    _stateGL->numLightsUsed = (SLint)SLScene::current->lights().size();
   
    // Set emissive light material to the lights diffuse color
    if (_meshes.size() > 0)
        if (_meshes[0]->mat)
            _meshes[0]->mat->emission(_on ? diffuse() : SLCol4f::BLACK);   
}
//-----------------------------------------------------------------------------
/*!
SLLightSphere::hitRec calls the recursive node intersection.
*/
SLbool SLLightSphere::hitRec(SLRay* ray)
{     
    // do not intersect shadow rays
    if (ray->type==SHADOW) return false;
   
    // only allow intersection with primary rays (no lights in reflections)
    if (ray->type!=PRIMARY) return false;
   
    // call the intersection routine of the node   
    return SLNode::hitRec(ray);
}
//-----------------------------------------------------------------------------
//! SLLightSphere::statsRec updates the statistic parameters
void SLLightSphere::statsRec(SLNodeStats &stats)
{  
    stats.numBytes += sizeof(SLLightSphere);
    stats.numBytes += _samples.sizeInBytes();
    SLNode::statsRec(stats);
}
//-----------------------------------------------------------------------------
/*!
SLLightSphere::drawMeshes sets the light states and calls then the drawMeshes 
method of its node.
*/
void SLLightSphere::drawMeshes(SLSceneView* sv)
{  
    if (_id!=-1) 
    {  
        // Set the OpenGL light states
        SLLightSphere::setState();
        _stateGL->numLightsUsed = (SLint)SLScene::current->lights().size();
   
        // Set emissive light material to the lights diffuse color
        if (_meshes.size() > 0)
            if (_meshes[0]->mat)
                _meshes[0]->mat->emission(_on ? diffuse() : SLCol4f::BLACK);   
   
        // now draw the inherited meshes
        SLNode::drawMeshes(sv);
   }
}
//-----------------------------------------------------------------------------
/*!
SLLightSphere::shadowTest returns 0.0 if the hit point is completely shaded and 
1.0 if it is 100% lighted. A return value inbetween is calculate by the ratio 
of the shadow rays not blocked to the total number of casted shadow rays.
*/
SLfloat SLLightSphere::shadowTest(SLRay* ray,         // ray of hit point
                                  const SLVec3f& L,   // vector from hit point to light
                                  SLfloat lightDist)  // distance to light
{  
    if (_samples.samples()==1)
    {  
        // define shadow ray and shoot 
        SLRay shadowRay(lightDist, L, ray);      
        SLScene::current->root3D()->hitRec(&shadowRay);
      
        if (shadowRay.length < lightDist)
        {  
            // Handle shadow value of transparent materials
            if (shadowRay.hitMat->hasAlpha())
            {  shadowRay.hitMesh->preShade(&shadowRay);
            SLfloat shadowTransp = SL_abs(shadowRay.dir.dot(shadowRay.hitNormal));
            return shadowTransp * shadowRay.hitMat->kt();
            }
            else return 0.0f;
        } 
        else return 1.0f;
    } 
    else // do light sampling for soft shadows
    {  
        SLVec3f C(updateAndGetWM().translation()); // Center of light
        SLVec3f LightX, LightY;       // main axis of sample plane
        SLfloat lighted = 0.0f;       // return value
        SLfloat invSamples = 1.0f/(_samples.samples());
        SLbool  outerCircleIsLighting = true;
        SLbool  innerCircleIsNotLighting = true;

        // Build normalized plain vectors X and Y that are perpendicular to L (=Z)
        if (fabs(L.x) >= fabs(L.y))
        {   SLfloat invLength = 1.0f/sqrt(L.x*L.x + L.z*L.z);
            LightX.set(L.z*invLength, 0, -L.x*invLength);  
        } else
        {   SLfloat invLength = 1.0f/sqrt(L.y*L.y + L.z*L.z);
            LightX.set(0, L.z*invLength, -L.y*invLength); 
        }
        LightY.cross(L,LightX);
        LightY*=_radius;
        LightX*=_radius;
      
        // Loop over radius r and angle phi of light circle
        for (SLint iR=_samples.samplesX()-1; iR>=0; --iR)
        {   for (SLint iPhi=_samples.samplesY()-1; iPhi>=0; --iPhi)
            {   SLVec2f discPos(_samples.point(iR,iPhi));

                // calculate disc position and vector LDisc to it
                SLVec3f conePos(C + discPos.x*LightX + discPos.y*LightY);
                SLVec3f LDisc(conePos - ray->hitPoint);
                LDisc.normalize();

                SLRay shadowRay(lightDist, LDisc, ray);
            
                SLScene::current->root3D()->hitRec(&shadowRay);

                if (shadowRay.length < lightDist) 
                    outerCircleIsLighting = false;               
                else 
                {   lighted += invSamples; // sum up the light
                    innerCircleIsNotLighting = false;
                }
            }
         
            // Early break 1: 
            // If the outer circle of shadow rays where not blocked return 1.0
            if (outerCircleIsLighting) return 1.0f;
         
            // Early break 2:
            // If a circle was completely shaded return lighted amount
            if (innerCircleIsNotLighting) return lighted;
            innerCircleIsNotLighting = true;
        }
        return lighted;
    }
}

//-----------------------------------------------------------------------------     
/*!
SLLightSphere::shadowTest returns 0.0 if the hit point is completely shaded and
1.0 if it is 100% lighted. A return value inbetween is calculate by the ratio
of the shadow rays not blocked to the total number of casted shadow rays.
*/
SLfloat SLLightSphere::shadowTestMC(SLRay* ray,         // ray of hit point
                                    const SLVec3f& L,   // vector from hit point to light
                                    SLfloat lightDist)  // distance to light
{
    if (_samples.samples() == 1)
    {
        // define shadow ray and shoot 
        SLRay shadowRay(lightDist, L, ray);
        SLScene::current->root3D()->hitRec(&shadowRay);

        if (shadowRay.length < lightDist)
        {
            // Handle shadow value of transparent materials
            if (shadowRay.hitMat->hasAlpha())
            {
            shadowRay.hitMesh->preShade(&shadowRay);
            SLfloat shadowTransp = SL_abs(shadowRay.dir.dot(shadowRay.hitNormal));
            return shadowTransp * shadowRay.hitMat->kt();
            }
            else return 0.0f;
        }
        else return 1.0f;
    }
    else // do light sampling for soft shadows
    {
        SLVec3f C(updateAndGetWM().translation()); // Center of light
        SLVec3f LightX, LightY;       // main axis of sample plane
        SLfloat lighted = 0.0f;       // return value
        SLfloat invSamples = 1.0f / (_samples.samples());
        SLbool  outerCircleIsLighting = true;
        SLbool  innerCircleIsNotLighting = true;

        // Build normalized plain vectors X and Y that are perpendicular to L (=Z)
        if (fabs(L.x) >= fabs(L.y))
        {
            SLfloat invLength = 1.0f / sqrt(L.x*L.x + L.z*L.z);
            LightX.set(L.z*invLength, 0, -L.x*invLength);
        }
        else
        {
            SLfloat invLength = 1.0f / sqrt(L.y*L.y + L.z*L.z);
            LightX.set(0, L.z*invLength, -L.y*invLength);
        }
        LightY.cross(L, LightX);
        LightY *= _radius;
        LightX *= _radius;

        // Loop over radius r and angle phi of light circle
        for (SLint iR = _samples.samplesX() - 1; iR >= 0; --iR)
        {
            for (SLint iPhi = _samples.samplesY() - 1; iPhi >= 0; --iPhi)
            {
                SLVec2f discPos(_samples.point(iR, iPhi));

                // calculate disc position and vector LDisc to it
                SLVec3f conePos(C + discPos.x*LightX + discPos.y*LightY);
                SLVec3f LDisc(conePos - ray->hitPoint);
                LDisc.normalize();

                SLRay shadowRay(lightDist, LDisc, ray);

                SLScene::current->root3D()->hitRec(&shadowRay);

                if (shadowRay.length < lightDist)
                    outerCircleIsLighting = false;
                else
                {
                    lighted += invSamples; // sum up the light
                    innerCircleIsNotLighting = false;
                }
            }

            // Early break 1: 
            // If the outer circle of shadow rays where not blocked return 1.0
            if (outerCircleIsLighting) return 1.0f;

            // Early break 2:
            // If a circle was completely shaded return lighted amount
            if (innerCircleIsNotLighting) return lighted;
            innerCircleIsNotLighting = true;
        }
        return 0.0f;
    }
}

//-----------------------------------------------------------------------------
/*! SLLightRect::setState sets the global rendering state
*/
void SLLightSphere::setState()
{  
    if (_id!=-1) 
    {   _stateGL->lightIsOn[_id]       = _on;
        _stateGL->lightPosWS[_id]      = positionWS();
        _stateGL->lightDirWS[_id]      = spotDirWS();           
        _stateGL->lightAmbient[_id]    = _ambient;              
        _stateGL->lightDiffuse[_id]    = _diffuse;              
        _stateGL->lightSpecular[_id]   = _specular;    
        _stateGL->lightSpotCutoff[_id] = _spotCutoff;           
        _stateGL->lightSpotCosCut[_id] = _spotCosCut;           
        _stateGL->lightSpotExp[_id]    = _spotExponent;         
        _stateGL->lightAtt[_id].x      = _kc;  
        _stateGL->lightAtt[_id].y      = _kl;    
        _stateGL->lightAtt[_id].z      = _kq; 
        _stateGL->lightDoAtt[_id]      = isAttenuated();
    }
}
//-----------------------------------------------------------------------------
