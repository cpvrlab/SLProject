//#############################################################################
//  File:      SLLightSpot.cpp
//  Authors:   Marcus Hudritsch
//  Date:      July 2014
//  Codestyle: https://github.com/cpvrlab/SLProject/wiki/SLProject-Coding-Style
//  Authors:   Marcus Hudritsch
//  License:   This software is provided under the GNU General Public License
//             Please visit: http://opensource.org/licenses/GPL-3.0
//#############################################################################

#include <SLLightSpot.h>
#include <SLRay.h>
#include <SLScene.h>
#include <SLSceneView.h>
#include <SLShadowMap.h>
#include <SLSphere.h>
#include <SLSpheric.h>

//-----------------------------------------------------------------------------
SLLightSpot::SLLightSpot(SLAssetManager* assetMgr,
                         SLScene*        s,
                         SLfloat         radius,
                         SLfloat         spotAngleDEG,
                         SLbool          hasMesh)
  : SLNode("LightSpot Node")
{
    _radius = radius;
    _samples.samples(1, 1, false);
    spotCutOffDEG(spotAngleDEG);

    if (hasMesh)
    {
        SLMaterial* mat = new SLMaterial(assetMgr,
                                         "LightSpot Mesh Mat",
                                         SLCol4f::BLACK,
                                         SLCol4f::BLACK);
        if (spotAngleDEG < 180.0f)
            addMesh(new SLSpheric(assetMgr,
                                  radius,
                                  0.0f,
                                  spotAngleDEG,
                                  16,
                                  16,
                                  "LightSpot Mesh",
                                  mat));
        else
            addMesh(new SLSphere(assetMgr,
                                 radius,
                                 16,
                                 16,
                                 "LightSpot Mesh",
                                 mat));
        _castsShadows = false;
    }

    init(s);
}
//-----------------------------------------------------------------------------
SLLightSpot::SLLightSpot(SLAssetManager* assetMgr,
                         SLScene*        s,
                         SLfloat         posx,
                         SLfloat         posy,
                         SLfloat         posz,
                         SLfloat         radius,
                         SLfloat         spotAngleDEG,
                         SLfloat         ambiPower,
                         SLfloat         diffPower,
                         SLfloat         specPower,
                         SLbool          hasMesh)
  : SLNode("LightSpot Node"),
    SLLight(ambiPower, diffPower, specPower)
{
    _radius = radius;
    _samples.samples(1, 1, false);
    _castsShadows = false;
    spotCutOffDEG(spotAngleDEG);

    translate(posx, posy, posz, TS_object);

    if (hasMesh)
    {
        SLMaterial* mat = new SLMaterial(assetMgr,
                                         "LightSpot Mesh Mat",
                                         SLCol4f::BLACK,
                                         SLCol4f::BLACK);
        if (spotAngleDEG < 180.0f)
            addMesh(new SLSpheric(assetMgr,
                                  radius,
                                  0.0f,
                                  spotAngleDEG,
                                  32,
                                  32,
                                  "LightSpot Mesh",
                                  mat));
        else
            addMesh(new SLSphere(assetMgr,
                                 radius,
                                 32,
                                 32,
                                 "LightSpot Mesh",
                                 mat));
    }
    init(s);
}
//-----------------------------------------------------------------------------
SLLightSpot::~SLLightSpot()
{
    delete _shadowMap;
}
//-----------------------------------------------------------------------------
/*!
SLLightSpot::init sets the light id, the light states & creates an emissive mat.
@todo properly remove this function and find a clean way to init lights in a scene
*/
void SLLightSpot::init(SLScene* s)
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
SLLightSpot::hitRec calls the recursive node intersection.
*/
SLbool SLLightSpot::hitRec(SLRay* ray)
{
    // do not intersect shadow rays
    if (ray->type == SHADOW) return false;

    // only allow intersection with primary rays (no lights in reflections)
    if (ray->type != PRIMARY) return false;

    // call the intersection routine of the node
    return SLNode::hitRec(ray);
}
//-----------------------------------------------------------------------------
//! SLLightSpot::statsRec updates the statistic parameters
void SLLightSpot::statsRec(SLNodeStats& stats)
{
    stats.numBytes += sizeof(SLLightSpot);
    stats.numBytes += _samples.sizeInBytes();
    SLNode::statsRec(stats);
}
//-----------------------------------------------------------------------------
/*!
SLLightSpot::drawMesh sets the light states and calls then the drawMesh
method of its node.
*/
void SLLightSpot::drawMesh(SLSceneView* sv)
{
    if (_id != -1)
    {
        // Set emissive light mesh material to the lights diffuse color
        if (_mesh)
        {
            if (_mesh->mat())
                _mesh->mat()->emissive(_isOn ? diffuseColor() : SLCol4f::BLACK);

            // now draw the single mesh of the node
            SLNode::drawMesh(sv);
        }

        // Draw the volume affected by the shadow map
        if (_createsShadows && _isOn && sv->s()->singleNodeSelected() == this)
        {
            _shadowMap->drawFrustum();
            _shadowMap->drawRays();
        }
    }
}
//-----------------------------------------------------------------------------
/*! Creates an fixed sized standard shadow map for the spotlight.
 * @param lightClipNear The light frustums near clipping distance
 * @param lightClipFar The light frustums near clipping distance
 * @param size Ignored for spot lights
 * @param texSize Shadow texture map size
 */
void SLLightSpot::createShadowMap(float   lightClipNear,
                                  float   lightClipFar,
                                  SLVec2f size,
                                  SLVec2i texSize)
{
    if (!_shadowMap)
        delete _shadowMap;

    _shadowMap = new SLShadowMap(this,
                                 lightClipNear,
                                 lightClipFar,
                                 size,
                                 texSize);
}
//-----------------------------------------------------------------------------
/*! Creates an automatic sized shadow map for the spot light.
 * @param camera Pointer to the camera for witch the shadow map gets sized
 * @param texSize Shadow texture map size
 * @param numCascades This value is ignored (default 0)
 */
void SLLightSpot::createShadowMapAutoSize(SLCamera* camera,
                                          SLVec2i   texSize,
                                          int       numCascades)
{
    (void)numCascades;
    if (!_shadowMap)
        delete _shadowMap;

    _shadowMap = new SLShadowMap(this,
                                 camera,
                                 texSize,
                                 0);
}
//-----------------------------------------------------------------------------
/*!
SLLightSpot::shadowTest returns 0.0 if the hit point is completely shaded and
1.0 if it is 100% lighted. A return value in between is calculate by the ratio
of the shadow rays not blocked to the total number of casted shadow rays.
*/
SLfloat SLLightSpot::shadowTest(SLRay*         ray,       // ray of hit point
                                const SLVec3f& L,         // vector from hit point to light
                                SLfloat        lightDist, // distance to light
                                SLNode*        root3D)
{
    if (_samples.samples() == 1)
    {
        // define shadow ray and shoot
        SLRay shadowRay(lightDist, L, ray);
        root3D->hitRec(&shadowRay);

        if (shadowRay.length < lightDist && shadowRay.hitMesh)
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
    else                                           // do light sampling for soft shadows
    {
        SLVec3f C(updateAndGetWM().translation()); // Center of light
        SLVec3f LightX, LightY;                    // main axis of sample plane
        SLfloat lighted                  = 0.0f;   // return value
        SLfloat invSamples               = 1.0f / (_samples.samples());
        SLbool  outerCircleIsLighting    = true;
        SLbool  innerCircleIsNotLighting = true;

        // Build normalized plain vectors X and Y that are perpendicular to L (=Z)
        if (fabs(L.x) >= fabs(L.y))
        {
            SLfloat invLength = 1.0f / sqrt(L.x * L.x + L.z * L.z);
            LightX.set(L.z * invLength, 0, -L.x * invLength);
        }
        else
        {
            SLfloat invLength = 1.0f / sqrt(L.y * L.y + L.z * L.z);
            LightX.set(0, L.z * invLength, -L.y * invLength);
        }
        LightY.cross(L, LightX);
        LightY *= _radius;
        LightX *= _radius;

        // Loop over radius r and angle phi of light circle
        for (SLint iR = (SLint)_samples.samplesX() - 1; iR >= 0; --iR)
        {
            for (SLint iPhi = (SLint)_samples.samplesY() - 1; iPhi >= 0; --iPhi)
            {
                SLVec2f discPos(_samples.point((SLuint)iR, (SLuint)iPhi));

                // calculate disc position and vector LDisc to it
                SLVec3f conePos(C + discPos.x * LightX + discPos.y * LightY);
                SLVec3f LDisc(conePos - ray->hitPoint);
                LDisc.normalize();

                SLRay shadowRay(lightDist, LDisc, ray);

                root3D->hitRec(&shadowRay);

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
        return lighted;
    }
}
//-----------------------------------------------------------------------------
/*!
SLLightSpot::shadowTest returns 0.0 if the hit point is completely shaded and
1.0 if it is 100% lighted. A return value inbetween is calculate by the ratio
of the shadow rays not blocked to the total number of casted shadow rays.
*/
SLfloat SLLightSpot::shadowTestMC(SLRay*         ray,       // ray of hit point
                                  const SLVec3f& L,         // vector from hit point to light
                                  SLfloat        lightDist, // distance to light
                                  SLNode*        root3D)
{
    if (_samples.samples() == 1)
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
    else                                           // do light sampling for soft shadows
    {
        SLVec3f C(updateAndGetWM().translation()); // Center of light
        SLVec3f LightX, LightY;                    // main axis of sample plane
        SLfloat lighted                  = 0.0f;   // return value
        SLfloat invSamples               = 1.0f / (_samples.samples());
        SLbool  outerCircleIsLighting    = true;
        SLbool  innerCircleIsNotLighting = true;

        // Build normalized plain vectors X and Y that are perpendicular to L (=Z)
        if (fabs(L.x) >= fabs(L.y))
        {
            SLfloat invLength = 1.0f / sqrt(L.x * L.x + L.z * L.z);
            LightX.set(L.z * invLength, 0, -L.x * invLength);
        }
        else
        {
            SLfloat invLength = 1.0f / sqrt(L.y * L.y + L.z * L.z);
            LightX.set(0, L.z * invLength, -L.y * invLength);
        }
        LightY.cross(L, LightX);
        LightY *= _radius;
        LightX *= _radius;

        // Loop over radius r and angle phi of light circle
        for (SLint iR = (SLint)_samples.samplesX() - 1; iR >= 0; --iR)
        {
            for (SLint iPhi = (SLint)_samples.samplesY() - 1; iPhi >= 0; --iPhi)
            {
                SLVec2f discPos(_samples.point((SLuint)iR, (SLuint)iPhi));

                // calculate disc position and vector LDisc to it
                SLVec3f conePos(C + discPos.x * LightX + discPos.y * LightY);
                SLVec3f LDisc(conePos - ray->hitPoint);
                LDisc.normalize();

                SLRay shadowRay(lightDist, LDisc, ray);

                root3D->hitRec(&shadowRay);

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
