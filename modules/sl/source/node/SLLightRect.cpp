//#############################################################################
//  File:      SLLightRect.cpp
//  Authors:   Marcus Hudritsch
//  Date:      July 2014
//  Codestyle: https://github.com/cpvrlab/SLProject/wiki/SLProject-Coding-Style
//  Authors:   Marcus Hudritsch
//  License:   This software is provided under the GNU General Public License
//             Please visit: http://opensource.org/licenses/GPL-3.0
//#############################################################################

#include <SLLightRect.h>
#include <SLPolygon.h>
#include <SLRay.h>
#include <SLScene.h>
#include <SLSceneView.h>
#include <SLShadowMap.h>

extern SLfloat rnd01();

//-----------------------------------------------------------------------------
SLLightRect::SLLightRect(SLAssetManager* assetMgr,
                         SLScene*        s,
                         SLfloat         w,
                         SLfloat         h,
                         SLbool          hasMesh) : SLNode("LightRect Node")
{
    width(w);
    height(h);
    _castsShadows = false;
    _samples.set(1, 1);

    // make sample number even
    if (_samples.x % 2 == 0) _samples.x++;
    if (_samples.y % 2 == 0) _samples.y++;

    spotCutOffDEG(90.0f);
    spotExponent(1.0);

    if (hasMesh)
    {
        SLMaterial* mat = new SLMaterial(assetMgr,
                                         "LightRect Mesh Mat",
                                         SLCol4f::BLACK,
                                         SLCol4f::BLACK);
        addMesh(new SLPolygon(assetMgr, w, h, "LightRect Mesh", mat));
        _castsShadows = false;
    }
    init(s);
}
//-----------------------------------------------------------------------------
SLLightRect::~SLLightRect()
{
    delete _shadowMap;
}
//-----------------------------------------------------------------------------
/*!
SLLightRect::init sets the light id, the light states & creates an
emissive mat.
@todo properly remove this function and find a clean way to init lights in a scene
*/
void SLLightRect::init(SLScene* s)
{
    // Check if OpenGL lights are available
    if (s->lights().size() >= SL_MAX_LIGHTS)
        SL_EXIT_MSG("Max. NO. of lights is exceeded!");

    // Add the light to the lights vector of the scene
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
SLLightRect::hitRec calls the nodes intersection code.
*/
SLbool
SLLightRect::hitRec(SLRay* ray)
{
    // do not intersect shadow rays
    if (ray->type == SHADOW) return false;

    // only allow intersection with primary rays (no lights in reflections)
    // if (ray->type!=PRIMARY) return false;

    // call the intersection routine of the node
    return SLNode::hitRec(ray);
}
//-----------------------------------------------------------------------------
//! SLLightSpot::statsRec updates the statistic parameters
void SLLightRect::statsRec(SLNodeStats& stats)
{
    stats.numBytes += sizeof(SLLightRect);
    SLNode::statsRec(stats);
}
//-----------------------------------------------------------------------------
/*!
SLLightRect::drawMeshes sets the light states and calls then the drawMeshes
method of its node.
*/
void SLLightRect::drawMesh(SLSceneView* sv)
{
    if (_id != -1)
    {
        // Set emissive light material to the lights diffuse color
        if (_mesh)
        {
            if (_mesh->mat())
                _mesh->mat()->emissive(_isOn ? diffuse() : SLCol4f::BLACK);
        }

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
//-----------------------------------------------------------------------------
/*!
SLLightRect::shadowTest returns 0.0 if the hit point is completely shaded and
1.0 if it is 100% lighted. A return value inbetween is calculate by the ratio
of the shadow rays not blocked to the total number of casted shadow rays.
*/
SLfloat SLLightRect::shadowTest(SLRay*         ray,       // ray of hit point
                                const SLVec3f& L,         // vector from hit point to light
                                const SLfloat  lightDist, // distance to light
                                SLNode*        root3D)
{
    if (_samples.x == 1 && _samples.y == 1)
    {
        // define shadow ray
        SLRay shadowRay(lightDist, L, ray);

        root3D->hitRec(&shadowRay);

        return (shadowRay.length < lightDist) ? 0.0f : 1.0f;
    }
    else                                                     // do light sampling for soft shadows
    {
        SLfloat dw = (SLfloat)_width / (SLfloat)_samples.x;  // width of a sample cell
        SLfloat dl = (SLfloat)_height / (SLfloat)_samples.y; // length of a sample cell
        SLint   x = 0, y = 0, hx = _samples.x / 2, hy = _samples.y / 2;
        SLint   samples = _samples.x * _samples.y;
        SLVbool isSampled;
        SLbool  importantPointsAreLighting = true;
        SLfloat lighted                    = 0.0f; // return value
        SLfloat invSamples                 = 1.0f / (SLfloat)(samples);
        SLVec3f SP;                                // vector hit point to sample point in world coords

        isSampled.resize((SLuint)samples);

        for (y = 0; y < _samples.y; ++y)
        {
            for (x = 0; x < _samples.x; ++x)
            {
                SLint iSP              = y * _samples.x + x;
                isSampled[(SLuint)iSP] = false;
            }
        }

        /*
        Important sample points (X) on a 7 by 5 rectangular light.
        If all of them are lighting the hit point the sample points
        in between (O) are not tested anymore.

             0   1   2   3   4   5   6
           +---+---+---+---+---+---+---+
        0  | X | . | . | X | . | . | X |
           +---+---+---+---+---+---+---+
        1  | . | . | . | . | . | . | . |
           +---+---+---+---+---+---+---+
        2  | X | . | . | X | . | . | X |
           +---+---+---+---+---+---+---+
        3  | . | . | . | . | . | . | . |
           +---+---+---+---+---+---+---+
        4  | X | . | . | X | . | . | X |
           +---+---+---+---+---+---+---+
        */

        // Double loop for the important sample points
        for (y = -hy; y <= hy; y += hy)
        {
            for (x = -hx; x <= hx; x += hx)
            {
                SLint iSP              = (y + hy) * _samples.x + x + hx;
                isSampled[(SLuint)iSP] = true;

                SP.set(updateAndGetWM().multVec(SLVec3f(x * dw, y * dl, 0)) - ray->hitPoint);
                SLfloat SPDist = SP.length();
                SP.normalize();
                SLRay shadowRay(SPDist, SP, ray);

                root3D->hitRec(&shadowRay);

                if (shadowRay.length >= SPDist - FLT_EPSILON)
                    lighted += invSamples; // sum up the light
                else
                    importantPointsAreLighting = false;
            }
        }

        if (importantPointsAreLighting)
            lighted = 1.0f;
        else
        { // Double loop for the sample points in between
            for (y = -hy; y <= hy; ++y)
            {
                for (x = -hx; x <= hx; ++x)
                {
                    SLint iSP = (y + hy) * _samples.x + x + hx;
                    if (!isSampled[(SLuint)iSP])
                    {
                        SP.set(updateAndGetWM().multVec(SLVec3f(x * dw, y * dl, 0)) - ray->hitPoint);
                        SLfloat SPDist = SP.length();
                        SP.normalize();
                        SLRay shadowRay(SPDist, SP, ray);

                        root3D->hitRec(&shadowRay);

                        // sum up the light
                        if (shadowRay.length >= SPDist - FLT_EPSILON)
                            lighted += invSamples;
                    }
                }
            }
        }
        return lighted;
    }
}
//-----------------------------------------------------------------------------
/*!
SLLightRect::shadowTestMC returns 0.0 if the hit point is shaded and 1.0 if it
lighted. Only one shadow sample is tested for path tracing.
*/
SLfloat SLLightRect::shadowTestMC(SLRay*         ray,       // ray of hit point
                                  const SLVec3f& L,         // vector from hit point to light
                                  const SLfloat  lightDist, // distance to light
                                  SLNode*        root3D)
{
    SLfloat rndX = rnd01();
    SLfloat rndY = rnd01();

    // Sample point in object space
    SLVec3f spOS(SLVec3f(rndX * _width - _width * 0.5f,
                         rndY * _height - _height * 0.5f,
                         0.0f));

    // Sample point in world space
    SLVec3f spWS(updateAndGetWM().multVec(spOS) - ray->hitPoint);

    SLfloat spDistWS = spWS.length();
    spWS.normalize();
    SLRay shadowRay(spDistWS, spWS, ray);

    root3D->hitRec(&shadowRay);

    return (shadowRay.length < spDistWS) ? 0.0f : 1.0f;
}
//-----------------------------------------------------------------------------
/*! Creates an fixed sized standard shadow map for a rectangular light.
 * @param lightClipNear The light frustums near clipping distance
 * @param lightClipFar The light frustums near clipping distance
 * @param size Ignored for rectangular lights
 * @param texSize Shadow texture map size
 */
void SLLightRect::createShadowMap(float   lightClipNear,
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
/*! Creates an automatic sized shadow map for the rectangular light.
 * @param camera Pointer to the camera for witch the shadow map gets sized
 * @param texSize Shadow texture map size (equal for all cascades)
 * @param numCascades This value is ignored (default 0)
 */
void SLLightRect::createShadowMapAutoSize(SLCamera* camera,
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
void SLLightRect::samples(const SLVec2i samples)
{
    assert(samples.x % 2 == 1 && samples.y % 2 == 1);
    _samples = samples;
}
//-----------------------------------------------------------------------------
void SLLightRect::samplesXY(const SLint x, const SLint y)
{
    assert(x % 2 == 1 && y % 2 == 1);
    _samples.set(x, y);
}
//-----------------------------------------------------------------------------
