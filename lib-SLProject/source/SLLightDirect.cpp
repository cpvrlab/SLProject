//#############################################################################
//  File:      SLLightDirect.cpp
//  Author:    Marcus Hudritsch
//  Date:      July 2016
//  Codestyle: https://github.com/cpvrlab/SLProject/wiki/Coding-Style-Guidelines
//  Copyright: Marcus Hudritsch
//             This software is provide under the GNU General Public License
//             Please visit: http://opensource.org/licenses/GPL-3.0
//#############################################################################

#include <stdafx.h>           // precompiled headers
#ifdef SL_MEMLEAKDETECT
#include <nvwa/debug_new.h>   // memory leak detector
#endif

#include <SLLightDirect.h>
#include <SLSphere.h>
#include <SLRay.h>
#include <SLScene.h>
#include <SLSceneView.h>
#include <SLMaterial.h>
#include <SLMesh.h>
#include <SLPolyline.h>

//-----------------------------------------------------------------------------
SLLightDirect::SLLightDirect(SLfloat radius, SLbool hasMesh) :
               SLNode("LightSphere")
{  
    _radius = radius;
    _dirLength = 1.0;

    if (hasMesh)
    {   SLMaterial* mat = new SLMaterial("LightDirectMeshMat", 
                                         SLCol4f::BLACK, 
                                         SLCol4f::BLACK);
        addMesh(new SLSphere(radius, 16, 16, "LightDirectMesh", mat));
        addMesh(new SLPolyline({SLVec3f(0,0,-0.5f), SLVec3f(0,0,0.5f)}, 
                               false, "LightDirection", mat));
    }

    init();
}
//-----------------------------------------------------------------------------
SLLightDirect::SLLightDirect(SLfloat dirx, 
                             SLfloat diry, 
                             SLfloat dirz,
                             SLfloat radius,
                             SLfloat dirLength,
                             SLfloat ambiPower,
                             SLfloat diffPower,
                             SLfloat specPower,
                             SLbool hasMesh) : 
                SLNode("Directional Light"), 
                SLLight(ambiPower, diffPower, specPower)
{  
    _radius = radius;
    _dirLength = dirLength;
    translate(dirx, diry, dirz, TS_object);

    if (hasMesh)
    {   SLMaterial* mat = new SLMaterial("LightDirectMeshMat", 
                                         SLCol4f::BLACK, 
                                         SLCol4f::BLACK);
        addMesh(new SLSphere(radius, 16, 16, "LightDirectMesh", mat));
        addMesh(new SLPolyline({SLVec3f(0,0,-0.5f), SLVec3f(0,0,0.5f)}, 
                               false, "LightDirection", mat));
    }
    init();
}
//-----------------------------------------------------------------------------
/*! 
SLLightDirect::init sets the light id, the light states & creates an 
emissive mat.
@todo properly remove this function and find a clean way to init lights in a scene
*/
void SLLightDirect::init()
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
//! SLLightDirect::statsRec updates the statistic parameters
void SLLightDirect::statsRec(SLNodeStats &stats)
{  
    stats.numBytes += sizeof(SLLightDirect);
    SLNode::statsRec(stats);
}
//-----------------------------------------------------------------------------
/*!
SLLightDirect::drawMeshes sets the light states and calls then the drawMeshes 
method of its node.
*/
void SLLightDirect::drawMeshes(SLSceneView* sv)
{  
    if (_id!=-1) 
    {  
        // Set the OpenGL light states
        SLLightDirect::setState();
        _stateGL->numLightsUsed = (SLint)SLScene::current->lights().size();
   
        // Set emissive light material to the lights diffuse color
        if (_meshes.size() > 0)
        {   for (SLMesh* mesh : _meshes)
                if (mesh->mat)
                    mesh->mat->emission(_on ? diffuse() : SLCol4f::BLACK); 
        }  
   
        // now draw the meshes of the node
        SLNode::drawMeshes(sv);
   }
}
//-----------------------------------------------------------------------------
/*!
SLLightDirect::shadowTest returns 0.0 if the hit point is completely shaded and 
1.0 if it is 100% lighted. A directional light can not generate soft shadows.
*/
SLfloat SLLightDirect::shadowTest(SLRay* ray,         // ray of hit point
                                  const SLVec3f& L,   // vector from hit point to light
                                  SLfloat lightDist)  // distance to light
{  
    // define shadow ray and shoot 
    SLRay shadowRay(lightDist, L, ray);      
    SLScene::current->root3D()->hitRec(&shadowRay);
      
    if (shadowRay.length < lightDist)
    {  
        // Handle shadow value of transparent materials
        if (shadowRay.hitMesh->mat->hasAlpha())
        {  shadowRay.hitMesh->preShade(&shadowRay);
        SLfloat shadowTransp = SL_abs(shadowRay.dir.dot(shadowRay.hitNormal));
        return shadowTransp * shadowRay.hitMesh->mat->kt();
        }
        else return 0.0f;
    } 
    else return 1.0f;
}

//-----------------------------------------------------------------------------     
/*!
SLLightDirect::shadowTestMC returns 0.0 if the hit point is completely shaded 
and 1.0 if it is 100% lighted. A directional light can not generate soft shadows.
*/
SLfloat SLLightDirect::shadowTestMC(SLRay* ray,         // ray of hit point
                                    const SLVec3f& L,   // vector from hit point to light
                                    SLfloat lightDist)  // distance to light
{
    // define shadow ray and shoot 
    SLRay shadowRay(lightDist, L, ray);
    SLScene::current->root3D()->hitRec(&shadowRay);

    if (shadowRay.length < lightDist)
    {
        // Handle shadow value of transparent materials
        if (shadowRay.hitMesh->mat->hasAlpha())
        {
        shadowRay.hitMesh->preShade(&shadowRay);
        SLfloat shadowTransp = SL_abs(shadowRay.dir.dot(shadowRay.hitNormal));
        return shadowTransp * shadowRay.hitMesh->mat->kt();
        }
        else return 0.0f;
    }
    else return 1.0f;
}

//-----------------------------------------------------------------------------
/*! SLLightRect::setState sets the global rendering state
*/
void SLLightDirect::setState()
{  
    if (_id!=-1) 
    {   _stateGL->lightIsOn[_id]       = _on;

        // For directional lights the position vector is in infinite distance
        // We use its homogeneos component w as zero as the directional light flag.
        _stateGL->lightPosWS[_id]      = positionWS();
        
        // The spot direction is used in the shaders for the light direction
        _stateGL->lightSpotDirWS[_id]  = spotDirWS();
                  
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
