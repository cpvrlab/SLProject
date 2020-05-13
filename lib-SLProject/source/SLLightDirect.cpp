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

#include <SLApplication.h>
#include <SLProjectScene.h>
#include <SLArrow.h>
#include <SLAssetManager.h>
#include <SLLightDirect.h>
#include <SLMaterial.h>
#include <SLRay.h>
#include <SLScene.h>
#include <SLSceneView.h>
#include <SLSphere.h>
#include <SLSpheric.h>
#include <SLGLProgramManager.h>

//-----------------------------------------------------------------------------
SLLightDirect::SLLightDirect(SLAssetManager* assetMgr,
                             SLScene*        s,
                             SLfloat         arrowLength,
                             SLbool          hasMesh)
  : SLNode("LightDirect Node")
{
    _arrowRadius         = arrowLength * 0.1f;
    _arrowLength         = arrowLength;
    _shadowMap           = nullptr;
    _shadowMapFrustumVAO = nullptr;

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
    _arrowRadius         = arrowLength * 0.1f;
    _arrowLength         = arrowLength;
    _shadowMap           = nullptr;
    _shadowMapFrustumVAO = nullptr;
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
    setState();
    SLGLState::instance()->numLightsUsed = (SLint)s->lights().size();

    // Set emissive light material to the lights diffuse color
    if (!_meshes.empty())
        if (_meshes[0]->mat())
            _meshes[0]->mat()->emissive(_isOn ? diffuse() : SLCol4f::BLACK);
}
//-----------------------------------------------------------------------------
SLLightDirect::~SLLightDirect()
{
    if (_shadowMap != nullptr)
        delete _shadowMap;

    if (_shadowMapFrustumVAO != nullptr)
        delete _shadowMapFrustumVAO;
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
        SLGLState::instance()->numLightsUsed = (SLint)sv->s().lights().size();

        // Set emissive light material to the lights diffuse color
        if (!_meshes.empty())
            if (_meshes[0]->mat())
                _meshes[0]->mat()->emissive(_isOn ? diffuse() : SLCol4f::BLACK);

        // now draw the meshes of the node
        SLNode::drawMeshes(sv);

        // Draw the volume affected by the shadow-map
        if (_createsShadows && sv->s().selectedNode() == this)
        {
            SLGLState* stateGL = SLGLState::instance();
            stateGL->modelViewMatrix.setMatrix(
              stateGL->viewMatrix * stateGL->lightProjection[_id].inverted());
            drawShadowMapFrustum();
        }
    }
}
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
/*! SLLightDirect::drawShadowMapFrustum draws the volume affected by the shadow-map
*/
void SLLightDirect::drawShadowMapFrustum()
{
    // clang-format off
    static SLVVec3f P = {
        {-1,  1, -1}, { 1,  1, -1}, // lower rect
        {-1,  1, -1}, {-1,  1,  1},
        { 1,  1,  1}, {-1,  1,  1},
        { 1,  1,  1}, { 1,  1, -1},

        {-1, -1, -1}, { 1, -1, -1}, // upper rect
        {-1, -1, -1}, {-1, -1,  1},
        { 1, -1,  1}, {-1, -1,  1},
        { 1, -1,  1}, { 1, -1, -1},

        {-1, -1, -1}, {-1,  1, -1}, // vertical lines
        { 1, -1, -1}, { 1,  1, -1},
        {-1, -1,  1}, {-1,  1,  1},
        { 1, -1,  1}, { 1,  1,  1},
    };
    // clang-format on

    if (_shadowMapFrustumVAO == nullptr)
    {
        _shadowMapFrustumVAO = new SLGLVertexArrayExt();
        _shadowMapFrustumVAO->generateVertexPos(&P);
    }
    _shadowMapFrustumVAO->drawArrayAsColored(PT_lines, SLCol3f(0, 1, 0), 1.0f, 0, (SLuint)P.size());
}
//-----------------------------------------------------------------------------
/*!
SLLightDirect::drawNodesIntoShadowMap recursively renders all objects which
cast shadows
*/
void SLLightDirect::drawNodesIntoShadowMap(SLNode* node, SLSceneView* sv, SLMaterial* depthMat)
{
    SLGLState* stateGL = SLGLState::instance();

    stateGL->modelViewMatrix.setMatrix(stateGL->viewMatrix);
    stateGL->modelViewMatrix.multiply(node->updateAndGetWM().m());

    if (node->castsShadows())
        for (auto* mesh : node->meshes())
            mesh->draw(sv, node, depthMat);

    for (SLNode* child : node->children())
        drawNodesIntoShadowMap(child, sv, depthMat);
}
//-----------------------------------------------------------------------------
/*! SLLightDirect::renderShadowMap renders the shadow map of the light
*/
void SLLightDirect::renderShadowMap(SLSceneView* sv, SLNode* root)
{
    SLGLState* stateGL = SLGLState::instance();

    const static unsigned int SHADOW_MAP_WIDTH = 1024, SHADOW_MAP_HEIGHT = 1024;
    static float              borderColor[] = {1.0, 1.0, 1.0, 1.0};

    static SLMaterial* depthMaterial = nullptr; // TODO
    // static SLGLGenericProgram depthProgram(SLApplication::scene, "Depth.vert", "Depth.frag");
    // static SLMaterial         depthMaterial(SLApplication::scene,
    //                                 "depthMaterial",
    //                                 nullptr,
    //                                 nullptr,
    //                                 nullptr,
    //                                 nullptr,
    //                                 &depthProgram);

    if (_shadowMap == nullptr)
        _shadowMap = new SLGLDepthBuffer(
          SHADOW_MAP_WIDTH,
          SHADOW_MAP_HEIGHT,
          GL_NEAREST,
          GL_NEAREST,
          GL_CLAMP_TO_BORDER,
          borderColor);
    _shadowMap->bind();

    // Initialize lightspace matrix
    SLMat4f vm;
    vm.lookAt(positionWS().vec3(),
              positionWS().vec3() + spotDirWS(),
              upWS());

    // Set viewport
    SLRecti vpRect = SLRecti(SHADOW_MAP_WIDTH, SHADOW_MAP_HEIGHT);
    stateGL->viewport(vpRect.x, vpRect.y, vpRect.width, vpRect.height);

    // Set projection
    SLfloat clipNear = 0.1f;
    SLfloat clipFar  = 20.0f;
    SLfloat radius   = 3.0;

    stateGL->stereoEye  = ET_center;
    stateGL->projection = P_monoOrthographic;
    stateGL->projectionMatrix.ortho(-radius, radius, -radius, radius, -clipNear, clipFar);

    // Save the light projection matrix
    stateGL->lightProjection[_id] = stateGL->projectionMatrix * vm;

    // Set view
    stateGL->modelViewMatrix.identity();
    stateGL->viewMatrix.setMatrix(vm);

    // Clear color buffer
    stateGL->clearColor(SLCol4f::BLACK);
    stateGL->clearColorDepthBuffer();

    // Draw meshes
    stateGL->currentMaterial(nullptr);
    drawNodesIntoShadowMap(root, sv, depthMaterial);
    GET_GL_ERROR;

    glBindFramebuffer(GL_FRAMEBUFFER, 0);
}
//-----------------------------------------------------------------------------
/*! SLLightRect::setState sets the global rendering state
*/
void SLLightDirect::setState()
{
    if (_id != -1)
    {
        SLGLState* stateGL      = SLGLState::instance();
        stateGL->lightIsOn[_id] = _isOn;

        // For directional lights the position vector is in infinite distance
        // We use its homogeneos component w as zero as the directional light flag.
        stateGL->lightPosWS[_id] = positionWS();

        // The spot direction is used in the shaders for the light direction
        stateGL->lightSpotDirWS[_id] = spotDirWS();

        stateGL->lightAmbient[_id]        = _ambient;
        stateGL->lightDiffuse[_id]        = _diffuse;
        stateGL->lightSpecular[_id]       = _specular;
        stateGL->lightSpotCutoff[_id]     = _spotCutOffDEG;
        stateGL->lightSpotCosCut[_id]     = _spotCosCutOffRAD;
        stateGL->lightSpotExp[_id]        = _spotExponent;
        stateGL->lightAtt[_id].x          = _kc;
        stateGL->lightAtt[_id].y          = _kl;
        stateGL->lightAtt[_id].z          = _kq;
        stateGL->lightDoAtt[_id]          = isAttenuated();
        stateGL->lightCreatesShadows[_id] = _createsShadows;
        stateGL->shadowMaps[_id]          = _shadowMap;
    }
}
//-----------------------------------------------------------------------------
