//#############################################################################
//  File:      SLLightDirect.cpp
//  Author:    Marcus Hudritsch
//  Date:      July 2016
//  Codestyle: https://github.com/cpvrlab/SLProject/wiki/SLProject-Coding-Style
//  Copyright: Marcus Hudritsch
//             This software is provide under the GNU General Public License
//             Please visit: http://opensource.org/licenses/GPL-3.0
//#############################################################################

#include <stdafx.h> // Must be the 1st include followed by  an empty line

#ifdef SL_MEMLEAKDETECT
#    include <nvwa/debug_new.h> // memory leak detector
#endif

#include <SLApplication.h>
#include <SLArrow.h>
#include <SLLightDirect.h>
#include <SLRay.h>
#include <SLScene.h>
#include <SLSceneView.h>
#include <SLSphere.h>
#include <SLSpheric.h>

//-----------------------------------------------------------------------------
SLLightDirect::SLLightDirect(SLfloat arrowLength,
                             SLbool  hasMesh)
  : SLNode("LightDirect Node")
{
    _arrowRadius = arrowLength * 0.1f;
    _arrowLength = arrowLength;

    if (hasMesh)
    {
        SLMaterial* mat = new SLMaterial("LightDirect Mesh Mat",
                                         SLCol4f::BLACK,
                                         SLCol4f::BLACK);
        addMesh(new SLArrow(_arrowRadius,
                            _arrowLength,
                            _arrowLength * 0.3f,
                            _arrowRadius * 2.0f,
                            16,
                            "LightDirect Mesh",
                            mat));
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
                             SLbool  hasMesh)
  : SLNode("Directional Light"),
    SLLight(ambiPower, diffPower, specPower)
{
    _arrowRadius = arrowLength * 0.1f;
    _arrowLength = arrowLength;
    translate(posx, posy, posz, TS_object);

    if (hasMesh)
    {
        SLMaterial* mat = new SLMaterial("LightDirect Mesh Mat",
                                         SLCol4f::BLACK,
                                         SLCol4f::BLACK);
        addMesh(new SLArrow(_arrowRadius,
                            _arrowLength,
                            _arrowLength * 0.3f,
                            _arrowRadius * 2.0f,
                            16,
                            "LightDirect Mesh",
                            mat));
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
    if (SLApplication::scene->lights().size() >= SL_MAX_LIGHTS)
        SL_EXIT_MSG("Max. NO. of lights is exceeded!");

    // Add the light to the lights array of the scene
    if (_id == -1)
    {
        _id = (SLint)SLApplication::scene->lights().size();
        SLApplication::scene->lights().push_back(this);
    }

    // Set the OpenGL light states
    setState();
    SLGLState::instance()->numLightsUsed = (SLint)SLApplication::scene->lights().size();

    // Set emissive light material to the lights diffuse color
    if (!_meshes.empty())
        if (_meshes[0]->mat())
            _meshes[0]->mat()->emissive(_isOn ? diffuse() : SLCol4f::BLACK);
}
//-----------------------------------------------------------------------------
/*!
SLLightDirect::hitRec calls the recursive node intersection.
*/
SLbool SLLightDirect::hitRec(SLRay* ray)
{
    // do not intersect shadow rays
    if (ray->type == SHADOW) return false;

    // only allow intersection with primary rays (no lights in reflections)
    if (ray->type != PRIMARY) return false;

    // call the intersection routine of the node
    return SLNode::hitRec(ray);
}
//-----------------------------------------------------------------------------
//! SLLightDirect::statsRec updates the statistic parameters
void SLLightDirect::statsRec(SLNodeStats& stats)
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
    if (_id != -1)
    {
        // Set the OpenGL light states
        SLLightDirect::setState();
        SLGLState::instance()->numLightsUsed = (SLint)SLApplication::scene->lights().size();

        // Set emissive light material to the lights diffuse color
        if (!_meshes.empty())
            if (_meshes[0]->mat())
                _meshes[0]->mat()->emissive(_isOn ? diffuse() : SLCol4f::BLACK);

        // now draw the meshes of the node
        SLNode::drawMeshes(sv);
    }
}
//-----------------------------------------------------------------------------
/*!
SLLightDirect::shadowTest returns 0.0 if the hit point is completely shaded and 
1.0 if it is 100% lighted. A directional light can not generate soft shadows.
*/
SLfloat SLLightDirect::shadowTest(SLRay*         ray, // ray of hit point
                                  const SLVec3f& L,   // vector from hit point to light
                                  SLfloat        lightDist)  // distance to light
{
    // define shadow ray and shoot
    SLRay shadowRay(lightDist, L, ray);
    SLApplication::scene->root3D()->hitRec(&shadowRay);

    if (shadowRay.length < lightDist)
    {
        // Handle shadow value of transparent materials
        if (shadowRay.hitMesh->mat()->hasAlpha())
        {
            shadowRay.hitMesh->preShade(&shadowRay);
            SLfloat shadowTransp = SL_abs(shadowRay.dir.dot(shadowRay.hitNormal));
            return shadowTransp * shadowRay.hitMesh->mat()->kt();
        }
        else
            return 0.0f;
    }
    else
        return 1.0f;
}
//-----------------------------------------------------------------------------
/*!
SLLightDirect::shadowTestMC returns 0.0 if the hit point is completely shaded 
and 1.0 if it is 100% lighted. A directional light can not generate soft shadows.
*/
SLfloat SLLightDirect::shadowTestMC(SLRay*         ray, // ray of hit point
                                    const SLVec3f& L,   // vector from hit point to light
                                    SLfloat        lightDist)  // distance to light
{
    // define shadow ray and shoot
    SLRay shadowRay(lightDist, L, ray);
    SLApplication::scene->root3D()->hitRec(&shadowRay);

    if (shadowRay.length < lightDist)
    {
        // Handle shadow value of transparent materials
        if (shadowRay.hitMesh->mat()->hasAlpha())
        {
            shadowRay.hitMesh->preShade(&shadowRay);
            SLfloat shadowTransp = SL_abs(shadowRay.dir.dot(shadowRay.hitNormal));
            return shadowTransp * shadowRay.hitMesh->mat()->kt();
        }
        else
            return 0.0f;
    }
    else
        return 1.0f;
}
//-----------------------------------------------------------------------------
/*! SLLightRect::setState sets the global rendering state
*/
void SLLightDirect::setState()
{
    if (_id != -1)
    {
        SLGLState* stateGL = SLGLState::instance();
        stateGL->lightIsOn[_id] = _isOn;

        // For directional lights the position vector is in infinite distance
        // We use its homogeneos component w as zero as the directional light flag.
        stateGL->lightPosWS[_id] = positionWS();

        // The spot direction is used in the shaders for the light direction
        stateGL->lightSpotDirWS[_id] = spotDirWS();

        stateGL->lightAmbient[_id]    = _ambient;
        stateGL->lightDiffuse[_id]    = _diffuse;
        stateGL->lightSpecular[_id]   = _specular;
        stateGL->lightSpotCutoff[_id] = _spotCutOffDEG;
        stateGL->lightSpotCosCut[_id] = _spotCosCutOffRAD;
        stateGL->lightSpotExp[_id]    = _spotExponent;
        stateGL->lightAtt[_id].x      = _kc;
        stateGL->lightAtt[_id].y      = _kl;
        stateGL->lightAtt[_id].z      = _kq;
        stateGL->lightDoAtt[_id]      = isAttenuated();
    }
}
//-----------------------------------------------------------------------------
