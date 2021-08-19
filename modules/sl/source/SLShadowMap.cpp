//#############################################################################
//  File:      SLShadowMap.cpp
//  Author:    Michael Schertenleib
//  Date:      May 2020
//  Codestyle: https://github.com/cpvrlab/SLProject/wiki/SLProject-Coding-Style
//  Copyright: Michael Schertenleib
//             This software is provide under the GNU General Public License
//             Please visit: http://opensource.org/licenses/GPL-3.0
//#############################################################################

#include <algorithm>

#include <SLGLDepthBuffer.h>
#include <SLGLProgramManager.h>
#include <SLGLState.h>
#include <SLGLVertexArrayExt.h>
#include <SLLight.h>
#include <SLMaterial.h>
#include <SLShadowMap.h>
#include <Instrumentor.h>
#include <SLSceneView.h>
#include <SLCamera.h>
#include <SLFrustum.h>
#include <SLNodeLOD.h>

//-----------------------------------------------------------------------------
SLShadowMap::SLShadowMap(SLProjection   projection,
                         SLLight*       light,
                         float          clipNear,
                         float          clipFar,
                         const SLVec2f& size,
                         const SLVec2i& texSize)
{
    PROFILE_FUNCTION();

    _light        = light;
    _projection   = projection;
    _useCubemap   = false;
    _useCascaded  = false;
    _depthBuffers = SLGLVDepthBuffer();
    _frustumVAO   = nullptr;
    _rayCount     = SLVec2i(0, 0);
    _mat          = nullptr;
    _clipNear     = clipNear;
    _clipFar      = clipFar;
    _size         = size;
    _halfSize     = _size / 2;
    _textureSize  = texSize;
    _camera       = nullptr;
    _numCascades  = 0;
}
//-----------------------------------------------------------------------------
SLShadowMap::SLShadowMap(SLProjection   projection,
                         SLLight*       light,
                         SLCamera*      camera,
                         const SLVec2f& size,
                         const SLVec2i& texSize,
                         int            numCascades)
{
    PROFILE_FUNCTION();

    _light        = light;
    _projection   = projection;
    _useCubemap   = false;
    _useCascaded  = false;
    _depthBuffers = SLGLVDepthBuffer();
    _frustumVAO   = nullptr;
    _rayCount     = SLVec2i(0, 0);
    _mat          = nullptr;
    _camera       = camera;
    _numCascades  = numCascades;
    _size         = size;
    _halfSize     = _size / 2;
    _textureSize  = texSize;
    _clipNear     = 0.1f;
    _clipFar      = 20.f;

    _cascadesFactor = 30.f;
}
//-----------------------------------------------------------------------------
SLShadowMap::~SLShadowMap()
{
    _depthBuffers.erase(_depthBuffers.begin(), _depthBuffers.end());
    delete _frustumVAO;
    delete _mat;
}
//-----------------------------------------------------------------------------
SLfloat SLShadowMap::clipNear()
{
    return _clipNear;
}
//-----------------------------------------------------------------------------
SLfloat SLShadowMap::clipFar()
{
    return _clipFar;
}
//-----------------------------------------------------------------------------
//! SLShadowMap::drawFrustum draws the volume affected by the shadow map
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

    if (_useCascaded)
    {
        for (SLint i = 0; i < _numCascades; ++i)
        {
            // Inverse matrix in a way to avoid precision error
            SLMat4f s, t;
            s.scale(1.0f / _lightProj[i].m(0), 1.0f / _lightProj[i].m(5), 1.0f / _lightProj[i].m(10));
            t.translate((-_lightProj[i].m(12)), (-_lightProj[i].m(13)), (-_lightProj[i].m(14)));

            stateGL->modelViewMatrix = stateGL->viewMatrix * _lightView[i].inverted() * s * t;

            _frustumVAO->drawArrayAsColored(PT_lines,
                                            SLCol4f::GREEN,
                                            1.0f,
                                            0,
                                            (SLuint)P.size());
        }
    }
    else
    {
        for (SLint i = 0; i < (_useCubemap ? 6 : 1); ++i)
        {
            stateGL->modelViewMatrix = stateGL->viewMatrix * _lightSpace[i].inverted();

            _frustumVAO->drawArrayAsColored(PT_lines,
                                            SLCol4f::GREEN,
                                            1.0f,
                                            0,
                                            (SLuint)P.size());
        }
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

    _depthBuffers[0]->bind();

    SLfloat pixelWidth  = (SLfloat)_textureSize.x / w;
    SLfloat pixelHeight = (SLfloat)_textureSize.y / h;

    SLfloat* depths = _depthBuffers[0]->readPixels();

    for (SLint x = 0; x < w; ++x)
    {
        for (SLint y = 0; y < h; ++y)
        {
            SLint pixelX = (SLint)(pixelWidth * (x + 0.5f));
            SLint pixelY = (SLint)(pixelHeight * (y + 0.5f));

            SLfloat viewSpaceX = Utils::lerp((x + 0.5f) / w, -1.0f, 1.0f);
            SLfloat viewSpaceY = Utils::lerp((y + 0.5f) / h, -1.0f, 1.0f);

            SLfloat depth = depths[pixelY * _depthBuffers[0]->dimensions().x + pixelX] * 2 - 1;

            if (depth == 1.0f) continue;

            P.push_back(SLVec3f(viewSpaceX, viewSpaceY, -1.0f));
            P.push_back(SLVec3f(viewSpaceX, viewSpaceY, depth));
        }
    }

    delete depths;

    _depthBuffers[0]->unbind();

    if (P.empty()) return;

    SLGLVertexArrayExt vao;
    vao.generateVertexPos(&P);

    stateGL->modelViewMatrix = stateGL->viewMatrix * _lightSpace[0].inverted();

    vao.drawArrayAsColored(PT_lines,
                           SLCol4f::YELLOW,
                           1.0f,
                           0,
                           (SLuint)P.size());

#endif
}
//-----------------------------------------------------------------------------
//! SLShadowMap::updateLightViewProj updates a light view projection matrix
void SLShadowMap::updateLightViewProj()
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
        _lightView[0].lookAt(positionWS, positionWS + SLVec3f::AXISX, -SLVec3f::AXISY);
        _lightView[1].lookAt(positionWS, positionWS - SLVec3f::AXISX, -SLVec3f::AXISY);
        _lightView[2].lookAt(positionWS, positionWS + SLVec3f::AXISY, SLVec3f::AXISZ);
        _lightView[3].lookAt(positionWS, positionWS - SLVec3f::AXISY, -SLVec3f::AXISZ);
        _lightView[4].lookAt(positionWS, positionWS + SLVec3f::AXISZ, -SLVec3f::AXISY);
        _lightView[5].lookAt(positionWS, positionWS - SLVec3f::AXISZ, -SLVec3f::AXISY);
    }
    else
    {
        SLNode* node = dynamic_cast<SLNode*>(_light);
        _lightView[0].lookAt(positionWS, positionWS + node->forwardWS(), node->upWS());
    }

    // Set projection matrix
    switch (_projection)
    {
        case P_monoOrthographic:
            _lightProj[0].ortho(-_halfSize.x,
                                _halfSize.x,
                                -_halfSize.y,
                                _halfSize.y,
                                _clipNear,
                                _clipFar);
            break;

        case P_monoPerspective:
            _lightProj[0].perspective(fov, 1.0f, _clipNear, _clipFar);
            break;

        default:
            SL_EXIT_MSG("Unsupported light projection");
    }

    // Set the model-view-projection matrix
    for (SLint i = 0; i < (_useCubemap ? 6 : 1); ++i)
        _lightSpace[i] = _lightProj[0] * _lightView[i];
}
//-----------------------------------------------------------------------------
//! Returns the visible nodes inside the light frustum
/*!
 * Check if the passed node is inside the light frustum and adds if so to the
 * visibleNodes vector.
 * @param node Node to cull or add to to visibleNodes vector
 * @param lightProj The cascades light projection matrix that gets adapted
 * @param lightView The cascades light view matrix
 * @param frustumPlanes
 * @param visibleNodes
 */
void SLShadowMap::lightCullingAdaptiveRec(SLNode*  node,
                                          SLMat4f& lightProj,
                                          SLMat4f& lightView,
                                          SLPlane* frustumPlanes,
                                          SLVNode& visibleNodes)
{

    if (typeid(*node->parent()) == typeid(SLNodeLOD))
    {
        int levelForSM = node->levelForSM();
        if (levelForSM == 0 && node->drawBit(SL_DB_HIDDEN))
            return;
        else // levelForSM > 0
        {
            if (node->parent()->children().size() < levelForSM)
            {
                SL_EXIT_MSG("SLShadowMap::lightCullingAdaptiveRec: levelForSM > num. LOD Nodes.");
            }
            SLNode* nodeForSM = node->parent()->children()[levelForSM - 1];
            if (nodeForSM != node)
                return;
        }
    }
    else
    {
        if (node->drawBit(SL_DB_HIDDEN))
            return;
    }

    if (!node->castsShadows())
        return;

    // We don't need to increase far plane distance
    float distance = frustumPlanes[5].distToPoint(node->aabb()->centerWS());
    if (distance < -node->aabb()->radiusWS())
        return;

    for (int i = 0; i < 4; i++)
    {
        float distance = frustumPlanes[i].distToPoint(node->aabb()->centerWS());
        if (distance < -node->aabb()->radiusWS())
            return;
    }

    // If object is behind the light's near plane, move the near plane back
    distance = frustumPlanes[4].distToPoint(node->aabb()->centerWS());
    if (distance < node->aabb()->radiusWS())
    {
        float a = lightProj.m(10);
        float b = lightProj.m(14);
        float n = (b + 1.f) / a;
        float f = (b - 1.f) / a;
        n       = n + (distance - node->aabb()->radiusWS());
        lightProj.m(10, -2.f / (f - n));
        lightProj.m(14, -(f + n) / (f - n));
        SLFrustum::viewToFrustumPlanes(frustumPlanes, lightProj, lightView);
    }

    visibleNodes.push_back(node);

    for (SLNode* child : node->children())
        lightCullingAdaptiveRec(child, lightProj, lightView, frustumPlanes, visibleNodes);
}
//-----------------------------------------------------------------------------
/*!
SLShadowMap::drawNodesIntoDepthBuffer recursively renders all objects which
cast shadows
*/
void SLShadowMap::drawNodesDirectionalCulling(SLVNode      visibleNodes,
                                              SLSceneView* sv,
                                              SLMat4f&     lightProj,
                                              SLMat4f&     lightView,
                                              SLPlane*     planes)
{
    SLGLState* stateGL        = SLGLState::instance();

    for (SLNode* node : visibleNodes)
    {
        stateGL->modelViewMatrix  = lightView * node->updateAndGetWM();

        if (node->castsShadows() &&
            node->mesh() &&
            node->mesh()->primitive() >= GL_TRIANGLES)
            node->mesh()->drawIntoDepthBuffer(sv, node, _mat);
    }
}
//-----------------------------------------------------------------------------
/*
void SLShadowMap::drawNodesDirectional(SLNode*      node,
                                       SLSceneView* sv,
                                       SLMat4f&     P,
                                       SLMat4f&     lv)
{
    SLGLState* stateGL        = SLGLState::instance();
    stateGL->modelViewMatrix  = lv * node->updateAndGetWM();
    stateGL->projectionMatrix = P;

    if (node->castsShadows() &&
        node->mesh() &&
        node->mesh()->primitive() >= GL_TRIANGLES)
        node->mesh()->drawIntoDepthBuffer(sv, node, _mat);

    for (SLNode* child : node->children())
        drawNodesDirectional(child, sv, P, lv);
}
//-----------------------------------------------------------------------------
void SLShadowMap::drawNodesIntoDepthBufferCulling(SLNode*      node,
                                                  SLSceneView* sv,
                                                  SLMat4f&     P,
                                                  SLMat4f&     lv,
                                                  SLPlane*     planes)
{
    if (node->drawBit(SL_DB_HIDDEN))
        return;

    for (int i = 0; i < 6; i++)
    {
        SLfloat distance = planes[i].distToPoint(node->aabb()->centerWS());
        if (distance < -node->aabb()->radiusWS())
            return;
    }

    SLGLState* stateGL        = SLGLState::instance();
    stateGL->modelViewMatrix  = lv * node->updateAndGetWM();
    stateGL->projectionMatrix = P;

    if (node->castsShadows() &&
        node->mesh() &&
        node->mesh()->primitive() >= GL_TRIANGLES)
        node->mesh()->drawIntoDepthBuffer(sv, node, _mat);

    for (SLNode* child : node->children())
        drawNodesIntoDepthBufferCulling(child, sv, P, lv, planes);
}
 */
//-----------------------------------------------------------------------------
void SLShadowMap::drawNodesIntoDepthBuffer(SLNode*      node,
                                           SLSceneView* sv,
                                           SLMat4f&     P,
                                           SLMat4f&     v)
{
    if (node->drawBit(SL_DB_HIDDEN))
        return;
    SLGLState* stateGL        = SLGLState::instance();
    stateGL->modelViewMatrix  = v * node->updateAndGetWM();
    stateGL->projectionMatrix = P;

    if (node->castsShadows() &&
        node->mesh() &&
        node->mesh()->primitive() >= GL_TRIANGLES)
        node->mesh()->drawIntoDepthBuffer(sv, node, _mat);

    for (SLNode* child : node->children())
        drawNodesIntoDepthBuffer(child, sv, P, v);
}
//-----------------------------------------------------------------------------
/*! SLShadowMap::render renders the shadow map of the light
 */
void SLShadowMap::render(SLSceneView* sv, SLNode* root)
{
    PROFILE_FUNCTION();

    if (_projection == P_monoOrthographic && _camera != nullptr)
    {
        renderDirectionalLightCascaded(sv, root);
        return;
    }

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

    updateLightViewProj();

    if (this->_useCubemap)
        this->_textureSize.y = this->_textureSize.x;

#ifdef SL_GLES
    SLint wrapMode = GL_CLAMP_TO_EDGE;
#else
    SLint wrapMode = GL_CLAMP_TO_BORDER;
#endif

    if (_depthBuffers.size() != 0 &&
        (_depthBuffers[0]->dimensions() != _textureSize ||
         (_depthBuffers[0]->target() == GL_TEXTURE_CUBE_MAP) != _useCubemap))
    {
        _depthBuffers.erase(_depthBuffers.begin(), _depthBuffers.end());
    }

    if (_depthBuffers.size() == 0)
    {
        _depthBuffers.push_back(new SLGLDepthBuffer(_textureSize,
                                                    GL_NEAREST,
                                                    GL_NEAREST,
                                                    wrapMode,
                                                    borderColor,
                                                    this->_useCubemap
                                                      ? GL_TEXTURE_CUBE_MAP
                                                      : GL_TEXTURE_2D));
    }

    if (_depthBuffers.size() != 1 ||
        _depthBuffers[0]->dimensions() != _textureSize ||
        (_depthBuffers[0]->target() == GL_TEXTURE_CUBE_MAP) != _useCubemap)
    {
    }
    _depthBuffers[0]->bind();

    for (SLint i = 0; i < (_useCubemap ? 6 : 1); ++i)
    {
        if (_useCubemap)
            _depthBuffers[0]->bindFace(GL_TEXTURE_CUBE_MAP_POSITIVE_X + i);

        // Set viewport
        stateGL->viewportFB(0, 0, _textureSize.x, _textureSize.y);

        // Set matrices
        stateGL->viewMatrix       = _lightView[i];
        stateGL->projectionMatrix = _lightProj[0];

        // Clear color buffer
        stateGL->clearColor(SLCol4f::BLACK);
        stateGL->clearColorDepthBuffer();

        // Draw meshes
        drawNodesIntoDepthBuffer(root, sv, _lightProj[0], _lightView[i]);
    }

    _depthBuffers[0]->unbind();
}
//-----------------------------------------------------------------------------
//! Returns a vector of near and far clip distances for all shadow cascades
SLVVec2f SLShadowMap::getShadowMapCascades(int   numCascades,
                                           float camClipNear,
                                           float camClipFar)
{
    SLVVec2f cascades;

    float ni, fi = camClipNear;

    for (int i = 0; i < numCascades; i++)
    {
        ni = fi;
        fi = _cascadesFactor *
             camClipNear *
             pow((camClipFar / (_cascadesFactor * camClipNear)),
                 (float)(i + 1) / (float)numCascades);

        cascades.push_back(SLVec2f(ni, fi));
    }
    return cascades;
}
//-----------------------------------------------------------------------------
//! Renders the nodes into cascaded shadow maps for directional lights
void SLShadowMap::renderDirectionalLightCascaded(SLSceneView* sv,
                                                 SLNode*      root)
{
    _useCascaded       = true;
    SLGLState* stateGL = SLGLState::instance();

    SLint wrapMode = GL_CLAMP_TO_BORDER;

    // Create depth buffer
    static SLfloat borderColor[] = {1.0, 1.0, 1.0, 1.0};

    // Create material
    if (_mat == nullptr)
        _mat = new SLMaterial(
          nullptr,
          "shadowMapMaterial",
          nullptr,
          nullptr,
          nullptr,
          nullptr,
          SLGLProgramManager::get(SP_depth));

    // Erase depth buffer textures if the number or size changed
    if (_depthBuffers.size() != 0 &&
        (_depthBuffers[0]->dimensions() != _textureSize ||
         _depthBuffers[0]->target() != GL_TEXTURE_2D ||
         _depthBuffers.size() != _numCascades))
    {
        _depthBuffers.erase(_depthBuffers.begin(), _depthBuffers.end());
    }

    // Get the vector of cascades with near and far distances
    SLVVec2f cascades = getShadowMapCascades(_numCascades,
                                             _camera->clipNear(),
                                             _camera->clipFar());

    SLMat4f camWM     = _camera->updateAndGetWM(); // camera space in world space
    SLNode* lightNode = dynamic_cast<SLNode*>(_light);

    // Create the depth buffer if they don't exist
    if (_depthBuffers.size() == 0)
    {
        for (int i = 0; i < cascades.size(); i++)
            _depthBuffers.push_back(new SLGLDepthBuffer(_textureSize,
                                                        GL_NEAREST,
                                                        GL_NEAREST,
                                                        wrapMode,
                                                        borderColor,
                                                        GL_TEXTURE_2D));
    }

    // for all subdivision of frustum
    for (int i = 0; i < cascades.size(); i++)
    {
        // The cascades near and far distance on the view direction in WS
        float ni = cascades[i].x;
        float fi = cascades[i].y;

        // The cascades middle point on the view direction in WS
        SLVec3f v = camWM.translation() - camWM.axisZ().normalized() * (ni + fi) * 0.5f;

        // Build the view matrix with lookAt method
        SLMat4f lightViewMat; // world space to light space
        lightViewMat.lookAt(v, v + lightNode->forwardWS(), lightNode->upWS());

        // Get the 8 camera frustum points in view space
        SLVec3f camFrustumPoints[8];
        SLFrustum::getPointsInViewSpace(camFrustumPoints,
                                        _camera->fovV(),
                                        sv->scrWdivH(),
                                        ni,
                                        fi);

        // Build min & max point of the cascades light frustum around the view frustum
        float minx = FLT_MAX, maxx = FLT_MIN;
        float miny = FLT_MAX, maxy = FLT_MIN;
        float minz = FLT_MAX, maxz = FLT_MIN;
        for (int j = 0; j < 8; j++)
        {
            SLVec3f fp = lightViewMat * camWM * camFrustumPoints[j];
            if (fp.x < minx) minx = fp.x;
            if (fp.y < miny) miny = fp.y;
            if (fp.x > maxx) maxx = fp.x;
            if (fp.y > maxy) maxy = fp.y;
            if (fp.z < minz) minz = fp.z;
            if (fp.z > maxz) maxz = fp.z;
        }
        float   sx = 2.f / (maxx - minx);
        float   sy = 2.f / (maxy - miny);
        float   sz = -2.f / (maxz - minz);
        SLVec3f t  = SLVec3f(-0.5f * (maxx + minx),
                             -0.5f * (maxy + miny),
                             -0.5f * (maxz + minz));

        // Build orthographic light projection matrix
        SLMat4f lightProjMat;
        lightProjMat.scale(sx, sy, sz);
        lightProjMat.translate(t);

        // Do light culling recursively with light frustum adaptation
        SLPlane frustumPlanes[6];
        SLVNode visibleNodes;
        SLFrustum::viewToFrustumPlanes(frustumPlanes, lightProjMat, lightViewMat);
        for (SLNode* child : root->children())
        {
            lightCullingAdaptiveRec(child,
                                    lightProjMat,
                                    lightViewMat,
                                    frustumPlanes,
                                    visibleNodes);
        }

        _lightView[i]  = lightViewMat;
        _lightProj[i]  = lightProjMat;
        _lightSpace[i] = lightProjMat * lightViewMat;

        _depthBuffers[i]->bind();

        // Set OpenGL states for depth buffer rendering
        stateGL->viewportFB(0, 0, _textureSize.x, _textureSize.y);
        stateGL->clearColor(SLCol4f::BLACK);
        stateGL->clearColorDepthBuffer();
        stateGL->projectionMatrix = lightProjMat;
        stateGL->viewMatrix       = lightViewMat;

        drawNodesDirectionalCulling(visibleNodes,
                                    sv,
                                    lightProjMat,
                                    lightViewMat,
                                    frustumPlanes);

        _depthBuffers[i]->unbind();
    }
}
//-----------------------------------------------------------------------------
