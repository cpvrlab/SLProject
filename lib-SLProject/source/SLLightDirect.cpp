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

#include <SLArrow.h>
#include <SLLightDirect.h>
#include <SLRay.h>
#include <SLScene.h>
#include <SLSceneView.h>
#include <SLShadowMap.h>
#include <SLSphere.h>
#include <SLSpheric.h>

//-----------------------------------------------------------------------------
SLLightDirect::SLLightDirect(SLAssetManager* assetMgr,
                             SLScene*        s,
                             SLfloat         arrowLength,
                             SLbool          hasMesh)
  : SLNode("LightDirect Node")
{
    _arrowRadius  = arrowLength * 0.1f;
    _arrowLength  = arrowLength;
    _castsShadows = false;

    if (hasMesh)
    {
        SLMaterial* mat = new SLMaterial(assetMgr,
                                         "LightDirect Mesh Mat",
                                         SLCol4f::BLACK,
                                         SLCol4f::BLACK);
        addMesh(new SLArrow(assetMgr,
                            _arrowRadius,
                            _arrowLength,
                            _arrowLength * 0.3f,
                            _arrowRadius * 2.0f,
                            16,
                            "LightDirect Mesh",
                            mat));
    }

    init(s);
}
//-----------------------------------------------------------------------------
SLLightDirect::SLLightDirect(SLAssetManager* assetMgr,
                             SLScene*        s,
                             SLfloat         posx,
                             SLfloat         posy,
                             SLfloat         posz,
                             SLfloat         arrowLength,
                             SLfloat         ambiPower,
                             SLfloat         diffPower,
                             SLfloat         specPower,
                             SLbool          hasMesh)
  : SLNode("Directional Light"),
    SLLight(ambiPower, diffPower, specPower)
{
    _arrowRadius = arrowLength * 0.1f;
    _arrowLength = arrowLength;
    translate(posx, posy, posz, TS_object);

    if (hasMesh)
    {
        SLMaterial* mat = new SLMaterial(assetMgr,
                                         "LightDirect Mesh Mat",
                                         SLCol4f::BLACK,
                                         SLCol4f::BLACK);
        addMesh(new SLArrow(assetMgr,
                            _arrowRadius,
                            _arrowLength,
                            _arrowLength * 0.3f,
                            _arrowRadius * 2.0f,
                            16,
                            "LightDirect Mesh",
                            mat));
    }
    init(s);
}
//-----------------------------------------------------------------------------
SLLightDirect::~SLLightDirect()
{
    delete _shadowMap;
}
//-----------------------------------------------------------------------------
/*!
SLLightDirect::init sets the light id, the light states & creates an
emissive mat.
@todo properly remove this function and find a clean way to init lights in a scene
*/
void SLLightDirect::init(SLScene* s)
{
    // Check if OpenGL lights are available
    if (s->lights().size() >= SL_MAX_LIGHTS)
        SL_EXIT_MSG("Max. NO. of lights is exceeded!");

    // Add the light to the lights array of the scene
    if (_id == -1)
    {
        _id = (SLint)s->lights().size();
        s->lights().push_back(this);
    }

    // Set the OpenGL light states
    //setState();
    //SLGLState::instance()->numLightsUsed = (SLint)s->lights().size();

    // Set emissive light material to the lights diffuse color
#ifdef SL_RENDER_BY_MATERIAL
    if (_mesh)
        if (_mesh->mat())
            _mesh->mat()->emissive(_isOn ? diffuseColor() : SLCol4f::BLACK);
#else
    if (!_meshes.empty())
        if (_meshes[0]->mat())
            _meshes[0]->mat()->emissive(_isOn ? diffuseColor() : SLCol4f::BLACK);
#endif
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

#ifdef SL_RENDER_BY_MATERIAL
void SLLightDirect::drawMesh(SLSceneView* sv)
{
    if (_id != -1)
    {
        // Set emissive light material to the lights diffuse color
        if (_mesh)
            if (_mesh->mat())
                _mesh->mat()->emissive(_isOn ? diffuse() : SLCol4f::BLACK);

        // now draw the meshes of the node
        SLNode::drawMesh(sv);

        // Draw the volume affected by the shadow map
        if (_createsShadows && _isOn && sv->s()->singleNodeSelected() == this)
        {
            _shadowMap->drawFrustum();
            _shadowMap->drawRays();
        }
    }
}
#else
void SLLightDirect::drawMeshes(SLSceneView* sv)
{
    if (_id != -1)
    {
        // Set emissive light material to the lights diffuse color
        if (!_meshes.empty())
            if (_meshes[0]->mat())
                _meshes[0]->mat()->emissive(_isOn ? diffuse() : SLCol4f::BLACK);

        // now draw the meshes of the node
        SLNode::drawMeshes(sv);

        // Draw the volume affected by the shadow map
        if (_createsShadows && _isOn && sv->s()->singleNodeSelected() == this)
        {
            _shadowMap->drawFrustum();
            _shadowMap->drawRays();
        }
    }
}
#endif
//-----------------------------------------------------------------------------
/*!
SLLightDirect::shadowTest returns 0.0 if the hit point is completely shaded and
1.0 if it is 100% lighted. A directional light can not generate soft shadows.
*/
SLfloat SLLightDirect::shadowTest(SLRay*         ray,       // ray of hit point
                                  const SLVec3f& L,         // vector from hit point to light
                                  SLfloat        lightDist, // distance to light
                                  SLNode*        root3D)
{
    // define shadow ray and shoot
    SLRay shadowRay(lightDist, L, ray);
    root3D->hitRec(&shadowRay);

    if (shadowRay.length < lightDist)
    {
        // Handle shadow value of transparent materials
        if (shadowRay.hitMesh->mat()->hasAlpha())
        {
            shadowRay.hitMesh->preShade(&shadowRay);
            SLfloat shadowTransp = Utils::abs(shadowRay.dir.dot(shadowRay.hitNormal));
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
SLfloat SLLightDirect::shadowTestMC(SLRay*         ray,       // ray of hit point
                                    const SLVec3f& L,         // vector from hit point to light
                                    SLfloat        lightDist, // distance to light
                                    SLNode*        root3D)
{
    // define shadow ray and shoot
    SLRay shadowRay(lightDist, L, ray);
    root3D->hitRec(&shadowRay);

    if (shadowRay.length < lightDist)
    {
        // Handle shadow value of transparent materials
        if (shadowRay.hitMesh->mat()->hasAlpha())
        {
            shadowRay.hitMesh->preShade(&shadowRay);
            SLfloat shadowTransp = Utils::abs(shadowRay.dir.dot(shadowRay.hitNormal));
            return shadowTransp * shadowRay.hitMesh->mat()->kt();
        }
        else
            return 0.0f;
    }
    else
        return 1.0f;
}
//-----------------------------------------------------------------------------
/*! SLLightDirect::renderShadowMap renders the shadow map of the light
*/
void SLLightDirect::renderShadowMap(SLSceneView* sv, SLNode* root)
{
    if (_shadowMap == nullptr)
        _shadowMap = new SLShadowMap(P_monoOrthographic, this);
    _shadowMap->render(sv, root);
}
//-----------------------------------------------------------------------------
