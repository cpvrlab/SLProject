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
#include <SLSpheric.h>
#include <SLSphere.h>
#include <SLRay.h>
#include <SLScene.h>
#include <SLSceneView.h>
#include <SLMaterial.h>
#include <SLMesh.h>
#include <SLArrow.h>

//-----------------------------------------------------------------------------
SLLightDirect::SLLightDirect(SLfloat arrowLength,
                             SLbool hasMesh) : SLNode("LightDirect Node")
{  
    _arrowRadius = arrowLength * 0.1f;
    _arrowLength = arrowLength;

    if (hasMesh)
    {   SLMaterial* mat = new SLMaterial("LightDirect Mesh Mat", 
                                         SLCol4f::BLACK, 
                                         SLCol4f::BLACK);
        addMesh(new SLArrow(_arrowRadius, 
                            _arrowLength, 
                            _arrowLength * 0.3f, 
                            _arrowRadius * 2.0f, 
                            16, "LightDirect Mesh", mat));
    }

    init();
}
//-----------------------------------------------------------------------------
SLLightDirect::SLLightDirect(SLfloat posx, 
                             SLfloat posy, 
                             SLfloat posz,
                             SLfloat arrowLength,
                             SLfloat ambiPower,
                             SLfloat diffPower,
                             SLfloat specPower,
                             SLbool hasMesh) : 
                SLNode("Directional Light"), 
                SLLight(ambiPower, diffPower, specPower)
{  
    _arrowRadius = arrowLength * 0.1f;
    _arrowLength = arrowLength;
    translate(posx, posy, posz, TS_object);

    if (hasMesh)
    {   SLMaterial* mat = new SLMaterial("LightDirect Mesh Mat", 
                                         SLCol4f::BLACK, 
                                         SLCol4f::BLACK);
        addMesh(new SLArrow(_arrowRadius, 
                            _arrowLength, 
                            _arrowLength * 0.3f, 
                            _arrowRadius * 2.0f, 
                            16, "LightDirect Mesh", mat));
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
            _meshes[0]->mat->emission(_isOn ? diffuse() : SLCol4f::BLACK);   
}
//-----------------------------------------------------------------------------
/*!
SLLightDirect::hitRec calls the recursive node intersection.
*/
SLbool SLLightDirect::hitRec(SLRay* ray)
{     
    // do not intersect shadow rays
    if (ray->type==SHADOW) return false;
   
    // only allow intersection with primary rays (no lights in reflections)
    if (ray->type!=PRIMARY) return false;
   
    // call the intersection routine of the node   
    return SLNode::hitRec(ray);
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
            if (_meshes[0]->mat)
                _meshes[0]->mat->emission(_isOn ? diffuse() : SLCol4f::BLACK);   
   
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
    {   _stateGL->lightIsOn[_id]       = _isOn;

        // For directional lights the position vector is in infinite distance
        // We use its homogeneos component w as zero as the directional light flag.
        _stateGL->lightPosWS[_id]      = positionWS();
        
        // The spot direction is used in the shaders for the light direction
        _stateGL->lightSpotDirWS[_id]  = spotDirWS();
                  
        _stateGL->lightAmbient[_id]    = _ambient;              
        _stateGL->lightDiffuse[_id]    = _diffuse;              
        _stateGL->lightSpecular[_id]   = _specular;    
        _stateGL->lightSpotCutoff[_id] = _spotCutOffDEG;           
        _stateGL->lightSpotCosCut[_id] = _spotCosCutOffRAD;           
        _stateGL->lightSpotExp[_id]    = _spotExponent;         
        _stateGL->lightAtt[_id].x      = _kc;  
        _stateGL->lightAtt[_id].y      = _kl;    
        _stateGL->lightAtt[_id].z      = _kq; 
        _stateGL->lightDoAtt[_id]      = isAttenuated();
    }
}
//-----------------------------------------------------------------------------