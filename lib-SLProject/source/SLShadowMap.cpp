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
#include <Instrumentor.h>

//-----------------------------------------------------------------------------
SLShadowMap::SLShadowMap(SLProjection projection, SLLight* light)
{
    PROFILE_FUNCTION();

    _light       = light;
    _projection  = projection;
    _useCubemap  = false;
    _depthBuffer = nullptr;
    _frustumVAO  = nullptr;
    _rayCount.set(16, 16);
    _mat      = nullptr;
    _clipNear = 0.1f;
    _clipFar  = 20.0f;
    _size.set(8.0f, 8.0f);
    _halfSize = _size / 2;
    _textureSize.set(1024, 1024);
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

    SLGLState* stateGL = SLGLState::instance();

    for (SLint i = 0; i < (_useCubemap ? 6 : 1); ++i)
    {
        stateGL->modelViewMatrix = stateGL->viewMatrix * _mvp[i].inverted();

        _frustumVAO->drawArrayAsColored(PT_lines,
                                        SLCol3f(0, 1, 0),
                                        1.0f,
                                        0,
                                        (SLuint)P.size());
    }
}
//-----------------------------------------------------------------------------
/*! SLShadowMap::drawRays draws sample rays of the light.
*/
void SLShadowMap::drawRays()
{
#ifndef SL_GLES // Reading the depth-buffer with GLES is non-trivial

    if (_useCubemap) return; // Not implemented for cubemaps

    SLint w = _rayCount.x;
    SLint h = _rayCount.y;

    if (w == 0 || h == 0)
        return;

    SLGLState* stateGL = SLGLState::instance();
    SLVVec3f   P;

    _depthBuffer->bind();

    SLfloat pixelWidth  = (SLfloat)_textureSize.x / w;
    SLfloat pixelHeight = (SLfloat)_textureSize.y / h;

    SLfloat* depths = _depthBuffer->readPixels();

    for (SLint x = 0; x < w; ++x)
    {
        for (SLint y = 0; y < h; ++y)
        {
            SLint pixelX = (SLint)(pixelWidth * (x + 0.5f));
            SLint pixelY = (SLint)(pixelHeight * (y + 0.5f));

            SLfloat viewSpaceX = Utils::lerp((x + 0.5f) / w, -1.0f, 1.0f);
            SLfloat viewSpaceY = Utils::lerp((y + 0.5f) / h, -1.0f, 1.0f);

            SLfloat depth = depths[pixelY * _depthBuffer->dimensions().x + pixelX] * 2 - 1;

            if (depth == 1.0f) continue;

            P.push_back(SLVec3f(viewSpaceX, viewSpaceY, -1.0f));
            P.push_back(SLVec3f(viewSpaceX, viewSpaceY, depth));
        }
    }

    delete depths;

    _depthBuffer->unbind();

    if (P.size() == 0) return;

    SLGLVertexArrayExt vao;
    vao.generateVertexPos(&P);

    stateGL->modelViewMatrix = stateGL->viewMatrix * _mvp[0].inverted();

    vao.drawArrayAsColored(PT_lines,
                           SLCol3f(1, 1, 0),
                           1.0f,
                           0,
                           (SLuint)P.size());

#endif
}
//-----------------------------------------------------------------------------
/*!
SLShadowMap::updateMVP creates a lightSpace matrix for a directional light.
*/
void SLShadowMap::updateMVP()
{
    // Calculate FOV
    SLfloat fov;

    if (_projection == P_monoPerspective)
    {
        fov = _light->spotCutOffDEG() * 2;

        // Automatically use cubemap when perspective projection makes no sense
        if (fov >= 180.0f)
            _useCubemap = true;

        if (_useCubemap)
            fov = 90.0f;
    }

    // Set view matrix
    SLVec3f positionWS = _light->positionWS().vec3();

    if (_useCubemap)
    {
        _v[0].lookAt(positionWS, positionWS + SLVec3f::AXISX, -SLVec3f::AXISY);
        _v[1].lookAt(positionWS, positionWS - SLVec3f::AXISX, -SLVec3f::AXISY);
        _v[2].lookAt(positionWS, positionWS + SLVec3f::AXISY, SLVec3f::AXISZ);
        _v[3].lookAt(positionWS, positionWS - SLVec3f::AXISY, -SLVec3f::AXISZ);
        _v[4].lookAt(positionWS, positionWS + SLVec3f::AXISZ, -SLVec3f::AXISY);
        _v[5].lookAt(positionWS, positionWS - SLVec3f::AXISZ, -SLVec3f::AXISY);
    }
    else
    {
        SLNode* node = dynamic_cast<SLNode*>(_light);
        _v[0].lookAt(positionWS, positionWS + node->forwardOS(), node->upWS());
    }

    // Set projection matrix
    switch (_projection)
    {
        case P_monoOrthographic:
            _p.ortho(-_halfSize.x, _halfSize.x, -_halfSize.y, _halfSize.y, _clipNear, _clipFar);
            break;

        case P_monoPerspective:
            _p.perspective(fov, 1.0f, _clipNear, _clipFar);
            break;

        default:
            SL_EXIT_MSG("Unsupported light projection");
    }

    // Set the model-view-projection matrix
    for (SLint i = 0; i < (_useCubemap ? 6 : 1); ++i)
        _mvp[i] = _p * _v[i];
}
//-----------------------------------------------------------------------------
/*!
SLShadowMap::drawNodesIntoDepthBuffer recursively renders all objects which
cast shadows
*/
void SLShadowMap::drawNodesIntoDepthBuffer(SLNode*      node,
                                           SLSceneView* sv,
                                           SLMat4f      v)
{
    SLGLState* stateGL       = SLGLState::instance();
    stateGL->modelViewMatrix = v * node->updateAndGetWM();

    if (node->castsShadows())
        for (auto* mesh : node->meshes())
            mesh->draw(sv, node, _mat, true);

    for (SLNode* child : node->children())
        drawNodesIntoDepthBuffer(child, sv, v);
}
//-----------------------------------------------------------------------------
/*! SLShadowMap::render renders the shadow map of the light
*/
void SLShadowMap::render(SLSceneView* sv, SLNode* root)
{
    PROFILE_FUNCTION();

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

    // Create depth buffer
    static SLfloat borderColor[] = {1.0, 1.0, 1.0, 1.0};

    updateMVP();

    if (this->_useCubemap)
        this->_textureSize.y = this->_textureSize.x;

#ifdef SL_GLES
    SLint wrapMode = GL_CLAMP_TO_EDGE;
#else
    SLint wrapMode = GL_CLAMP_TO_BORDER;
#endif

    if (_depthBuffer == nullptr ||
        _depthBuffer->dimensions() != _textureSize ||
        (_depthBuffer->target() == GL_TEXTURE_CUBE_MAP) != _useCubemap)
    {
        delete _depthBuffer;
        _depthBuffer = new SLGLDepthBuffer(_textureSize,
                                           GL_NEAREST,
                                           GL_NEAREST,
                                           wrapMode,
                                           borderColor,
                                           this->_useCubemap
                                             ? GL_TEXTURE_CUBE_MAP
                                             : GL_TEXTURE_2D);
    }
    _depthBuffer->bind();

    for (SLint i = 0; i < (_useCubemap ? 6 : 1); ++i)
    {
        if (_useCubemap)
            _depthBuffer->bindFace(GL_TEXTURE_CUBE_MAP_POSITIVE_X + i);

        // Set viewport
        stateGL->viewport(0, 0, _textureSize.x, _textureSize.y);

        // Set matrices
        stateGL->viewMatrix       = _v[i];
        stateGL->projectionMatrix = _p;

        // Clear color buffer
        stateGL->clearColor(SLCol4f::BLACK);
        stateGL->clearColorDepthBuffer();

        // Draw meshes
        drawNodesIntoDepthBuffer(root, sv, _v[i]);
    }

    _depthBuffer->unbind();
}
//-----------------------------------------------------------------------------
