//#############################################################################
//  File:      SLLightDirect.cpp
//  Date:      July 2016
//  Codestyle: https://github.com/cpvrlab/SLProject/wiki/SLProject-Coding-Style
//  Authors:   Marcus Hudritsch
//  License:   This software is provided under the GNU General Public License
//             Please visit: http://opensource.org/licenses/GPL-3.0
//#############################################################################

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
                             SLbool          hasMesh,
                             SLbool          doCascadedShadows)

  : SLNode("LightDirect Node"),
    _arrowRadius(arrowLength * 0.1f),
    _arrowLength(arrowLength),
    _doSunPowerAdaptation(false),
    _sunLightPowerMin(0),
    _sunLightColorLUT(nullptr, CLUT_DAYLIGHT),
    _doCascadedShadows(doCascadedShadows)
{
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
        _castsShadows = false;
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
                             SLbool          hasMesh,
                             SLbool          doCascadedShadows)
  : SLNode("Directional Light"),
    SLLight(ambiPower, diffPower, specPower),
    _arrowRadius(arrowLength * 0.1f),
    _arrowLength(arrowLength),
    _doSunPowerAdaptation(false),
    _sunLightPowerMin(0),
    _sunLightColorLUT(nullptr, CLUT_DAYLIGHT),
    _doCascadedShadows(doCascadedShadows)
{
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

    // Clear the color LUT that is also an OpenGL texture
    _sunLightColorLUT.deleteData();
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

    // Set emissive light material to the lights diffuse color
    if (_mesh)
        if (_mesh->mat())
            _mesh->mat()->emissive(_isOn ? diffuseColor() : SLCol4f::BLACK);
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
            if (sv->camera() != _shadowMap->camera())
                _shadowMap->drawFrustum();
            _shadowMap->drawRays();
        }
    }
}
//-----------------------------------------------------------------------------
/*! Creates an fixed sized standard shadow map for a directional light.
 * @param clipNear The light frustums near clipping distance
 * @param clipFar The light frustums near clipping distance
 * @param size Width and height of the orthographic light frustum
 * @param texSize Shadow texture map size
 */
void SLLightDirect::createShadowMap(float   clipNear,
                                    float   clipFar,
                                    SLVec2f size,
                                    SLVec2i texSize)
{
    if (!_shadowMap)
        delete _shadowMap;

    _shadowMap = new SLShadowMap(this,
                                 clipNear,
                                 clipFar,
                                 size,
                                 texSize);
}
//-----------------------------------------------------------------------------
/*! Creates an automatic sized and cascaded shadow map for the directional light.
 * @param camera Pointer to the camera for witch the shadow map gets sized
 * @param texSize Shadow texture map size (equal for all cascades)
 * @param numCascades NO. of cascades shadow maps
 */
void SLLightDirect::createShadowMapAutoSize(SLCamera* camera,
                                            SLVec2i   texSize,
                                            int       numCascades)
{
    if (!_shadowMap)
        delete _shadowMap;

    _doCascadedShadows = true;
    _shadowMap         = new SLShadowMap(this,
                                 camera,
                                 texSize,
                                 numCascades);
}
//-----------------------------------------------------------------------------
/*! SLLightDirect::shadowTest returns 0.0 if the hit point is completely shaded
and 1.0 if it is 100% lighted. A directional light can not generate soft shadows.
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
//! Calculates the sunlight color depending on the zenith angle
/*! If the angle is 0 it return 1 and _sunLightPowerMin at 90 degrees or more.
 This can be used to the downscale the directional light to simulate the reduced
 power of the sun. The color is take from a color ramp that is white at 0 degree
 zenith angle.
 */
SLCol4f SLLightDirect::calculateSunLight(SLfloat standardPower)
{
    SLVec3f toSunDirWS = -forwardOS();

    // The sun power is equal to the cosine of the sun zenith angle
    SLfloat cosZenithAngle = std::max(toSunDirWS.dot(SLVec3f::AXISY),
                                      _sunLightPowerMin);

    // The color is take from a color ramp that is white at 0 degree zenith
    SLCol4f sunColor = _sunLightColorLUT.getTexelf(cosZenithAngle, 0);

    return sunColor * standardPower * cosZenithAngle;
}
//-----------------------------------------------------------------------------
//! Returns the product of the ambient light color and the ambient light power
SLCol4f SLLightDirect::ambient()
{
    return _ambientColor * _ambientPower;
}
//-----------------------------------------------------------------------------
//! Returns the product of the diffuse light color and the diffuse light power
/*! If the directional light is the sun the color and the power is depending
 from the zenith angle of the sun. At noon it is high and bright and at sunrise
 and sunset it is low and reddish.
 */
SLCol4f SLLightDirect::diffuse()
{
    if (_doSunPowerAdaptation)
        return calculateSunLight(_diffusePower);
    else
        return _diffuseColor * _diffusePower;
}
//-----------------------------------------------------------------------------
//! Returns the product of the specular light color and the specular light power
/*! If the directional light is the sun the color and the power is depending
 from the zenith angle of the sun. At noon it is high and bright and at sunrise
 and sunset it is low and reddish.
 */
SLCol4f SLLightDirect::specular()
{
    if (_doSunPowerAdaptation)
        return calculateSunLight(_specularPower);
    else
        return _specularColor * _specularPower;
}
//-----------------------------------------------------------------------------
void SLLightDirect::renderShadowMap(SLSceneView* sv, SLNode* root)
{
    // Check if no shadow map was created at load time
    if (!_shadowMap)
    {
        this->createShadowMap();
    }
    _shadowMap->renderShadows(sv, root);
}
//-----------------------------------------------------------------------------
