//#############################################################################
//  File:      SLShadowMap.cpp
//  Author:    Michael Schertenleib
//  Date:      May 2020
//  Codestyle: https://github.com/cpvrlab/SLProject/wiki/SLProject-Coding-Style
//  Copyright: Michael Schertenleib
//             This software is provide under the GNU General Public License
//             Please visit: http://opensource.org/licenses/GPL-3.0
//#############################################################################

#include <stdafx.h> // Must be the 1st include followed by  an empty line

#include <SLGLDepthBuffer.h>
#include <SLGLProgramManager.h>
#include <SLGLState.h>
#include <SLGLVertexArrayExt.h>
#include <SLLight.h>
#include <SLLightDirect.h>
#include <SLMaterial.h>
#include <SLNode.h>
#include <SLShadowMap.h>

//-----------------------------------------------------------------------------
SLShadowMap::SLShadowMap()
{
    _depthBuffer = nullptr;
    _frustumVAO  = nullptr;
    _mat         = nullptr;
    _clipNear    = 0.1f;
    _clipFar     = 20.0f;
    _size.set(8.0f, 8.0f);
    _textureSize.set(512, 512);
}
//-----------------------------------------------------------------------------
SLShadowMap::~SLShadowMap()
{
    if (_depthBuffer != nullptr) delete _depthBuffer;
    if (_frustumVAO != nullptr) delete _frustumVAO;
    if (_mat != nullptr) delete _mat;
}

//-----------------------------------------------------------------------------
/*! SLShadowMap::drawShadowMapFrustum draws the volume affected by the shadow-map
*/
void SLShadowMap::drawFrustum()
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

    if (_frustumVAO == nullptr)
    {
        _frustumVAO = new SLGLVertexArrayExt();
        _frustumVAO->generateVertexPos(&P);
    }

    SLGLState* stateGL = SLGLState::instance();
    stateGL->modelViewMatrix.setMatrix(
      stateGL->viewMatrix * _mvp.inverted());

    _frustumVAO->drawArrayAsColored(PT_lines,
                                    SLCol3f(0, 1, 0),
                                    1.0f,
                                    0,
                                    (SLuint)P.size());
}
//-----------------------------------------------------------------------------
/*!
SLShadowMap::updateLightSpaceMatrix creates a lightSpace matrix for a directional
light.
*/
void SLShadowMap::updateLightSpaceMatrix(SLLightDirect* light)
{
    // Set view-model matrix
    _vm.lookAt(light->positionWS().vec3(),
               light->positionWS().vec3() + light->spotDirWS(),
               light->upWS());

    // Set projection matrix
    SLVec2f halfSize = _size / 2;
    _p.ortho(-halfSize.x, halfSize.x, -halfSize.y, halfSize.y, -_clipNear, _clipFar);

    // Set the model-view-projection matrix
    _mvp = _p * _vm;
}

//-----------------------------------------------------------------------------
/*!
SLShadowMap::drawNodesIntoDepthBuffer recursively renders all objects which
cast shadows
*/
void SLShadowMap::drawNodesIntoDepthBuffer(SLNode* node, SLSceneView* sv)
{
    SLGLState* stateGL = SLGLState::instance();

    stateGL->modelViewMatrix.setMatrix(stateGL->viewMatrix);
    stateGL->modelViewMatrix.multiply(node->updateAndGetWM().m());

    if (node->castsShadows())
        for (auto* mesh : node->meshes())
            mesh->draw(sv, node, _mat, true);

    for (SLNode* child : node->children())
        drawNodesIntoDepthBuffer(child, sv);
}
//-----------------------------------------------------------------------------
/*! SLShadowMap::render renders the shadow map of the light
*/
void SLShadowMap::render(SLSceneView* sv, SLNode* root)
{
    SLGLState* stateGL = SLGLState::instance();

    if (_mat == nullptr)
        _mat = new SLMaterial(
          nullptr,
          "shadowMapMaterial",
          nullptr,
          nullptr,
          nullptr,
          nullptr,
          SLGLProgramManager::get(SP_depth));

    static float borderColor[] = {1.0, 1.0, 1.0, 1.0};

    if (_depthBuffer == nullptr || _depthBuffer->dimensions() != _textureSize)
    {
        delete _depthBuffer;
        _depthBuffer = new SLGLDepthBuffer(
          _textureSize,
          GL_NEAREST,
          GL_NEAREST,
          GL_CLAMP_TO_BORDER,
          borderColor);
    }
    _depthBuffer->bind();

    // Set viewport
    stateGL->viewport(0, 0, _textureSize.x, _textureSize.y);

    // Set projection
    stateGL->stereoEye        = ET_center;
    stateGL->projection       = P_monoOrthographic;
    stateGL->projectionMatrix = _p;

    // Set view
    stateGL->modelViewMatrix.identity();
    stateGL->viewMatrix.setMatrix(_vm);

    // Clear color buffer
    stateGL->clearColor(SLCol4f::BLACK);
    stateGL->clearColorDepthBuffer();

    // Draw meshes
    drawNodesIntoDepthBuffer(root, sv);
    GET_GL_ERROR;

    glBindFramebuffer(GL_FRAMEBUFFER, 0);
}
//-----------------------------------------------------------------------------
