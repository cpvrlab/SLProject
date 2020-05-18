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
#include <SLLightSpot.h>
#include <SLMaterial.h>
#include <SLNode.h>
#include <SLShadowMap.h>

//-----------------------------------------------------------------------------
SLShadowMap::SLShadowMap()
{
    _depthBuffer = nullptr;
    _frustumVAO  = nullptr;
    _rayCount.set(16, 16);
    _mat      = nullptr;
    _clipNear = 0.1f;
    _clipFar  = 20.0f;
    _size.set(8.0f, 8.0f);
    _halfSize = _size / 2;
    _textureSize.set(512, 512);
}
//-----------------------------------------------------------------------------
SLShadowMap::~SLShadowMap()
{
    delete _depthBuffer;
    delete _frustumVAO;
    delete _mat;
}
//-----------------------------------------------------------------------------
/*! SLShadowMap::drawFrustum draws the volume affected by the shadow map
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

    SLGLState* stateGL       = SLGLState::instance();
    stateGL->modelViewMatrix = stateGL->viewMatrix * _mvp.inverted();

    _frustumVAO->drawArrayAsColored(PT_lines,
                                    SLCol3f(0, 1, 0),
                                    1.0f,
                                    0,
                                    (SLuint)P.size());
}
//-----------------------------------------------------------------------------
/*! SLShadowMap::drawRays draws sample rays of the light.
*/
void SLShadowMap::drawRays()
{
    SLGLState* stateGL = SLGLState::instance();
    SLVVec3f   P;

    _depthBuffer->bind();

    SLint w = _rayCount.x;
    SLint h = _rayCount.y;

    SLfloat pixelWidth  = (SLfloat)_textureSize.x / w;
    SLfloat pixelHeight = (SLfloat)_textureSize.y / h;

    for (SLint x = 0; x < w; ++x)
    {
        for (SLint y = 0; y < h; ++y)
        {
            SLint pixelX = (SLint)(pixelWidth * (x + 0.5f));
            SLint pixelY = (SLint)(pixelHeight * (y + 0.5f));

            SLfloat viewSpaceX = Utils::lerp((x + 0.5f) / w, -1.0f, 1.0f);
            SLfloat viewSpaceY = Utils::lerp((y + 0.5f) / w, -1.0f, 1.0f);

            SLfloat depth = _depthBuffer->depth(pixelX, pixelY) * 2 - 1;

            if (depth == 1.0f) continue;

            P.push_back(SLVec3f(viewSpaceX, viewSpaceY, -1.0f));
            P.push_back(SLVec3f(viewSpaceX, viewSpaceY, depth));
        }
    }

    glBindFramebuffer(GL_FRAMEBUFFER, 0);

    if (P.size() == 0) return;

    SLGLVertexArrayExt vao;
    vao.generateVertexPos(&P);

    stateGL->modelViewMatrix = stateGL->viewMatrix * _mvp.inverted();

    vao.drawArrayAsColored(PT_lines,
                           SLCol3f(1, 1, 0),
                           1.0f,
                           0,
                           (SLuint)P.size());
}
//-----------------------------------------------------------------------------
/*!
SLShadowMap::updateMVP creates a lightSpace matrix for a directional
light.
*/
void SLShadowMap::updateMVP(SLLight* light, SLProjection projection)
{
    // Set view matrix
    SLNode* node = dynamic_cast<SLNode*>(light);
    _v.lookAt(light->positionWS().vec3(),
              light->positionWS().vec3() + node->forwardOS(),
              node->upWS());

    // Set projection matrix
    switch (projection)
    {
        case P_monoOrthographic:
            _p.ortho(-_halfSize.x, _halfSize.x, -_halfSize.y, _halfSize.y, _clipNear, _clipFar);
            break;

        case P_monoPerspective:
            _p.perspective(light->spotCutOffDEG() * 2, 1.0f, _clipNear, _clipFar);
            break;

        default:
            SL_EXIT_MSG("Unsupported light projection");
    }

    // Set the model-view-projection matrix
    _mvp = _p * _v;
}
//-----------------------------------------------------------------------------
/*!
SLShadowMap::drawNodesIntoDepthBuffer recursively renders all objects which
cast shadows
*/
void SLShadowMap::drawNodesIntoDepthBuffer(SLNode* node, SLSceneView* sv)
{
    SLGLState* stateGL       = SLGLState::instance();
    stateGL->modelViewMatrix = _v * node->updateAndGetWM();

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

    // Create Material
    if (_mat == nullptr)
        _mat = new SLMaterial(
          nullptr,
          "shadowMapMaterial",
          nullptr,
          nullptr,
          nullptr,
          nullptr,
          SLGLProgramManager::get(SP_depth));

    // Create depthbuffer
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

    // Set matrices
    stateGL->viewMatrix       = _v;
    stateGL->projectionMatrix = _p;

    // Clear color buffer
    stateGL->clearColor(SLCol4f::BLACK);
    stateGL->clearColorDepthBuffer();

    // Draw meshes
    drawNodesIntoDepthBuffer(root, sv);
    GET_GL_ERROR;

    glBindFramebuffer(GL_FRAMEBUFFER, 0);
}
//-----------------------------------------------------------------------------
