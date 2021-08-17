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
    return _camera ? _camera->clipNear() : _clipNear;
}
//-----------------------------------------------------------------------------
SLfloat SLShadowMap::clipFar()
{
    return _camera ? _camera->clipFar() : _clipFar;
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

    if (_useCascaded)
    {
        for (SLint i = 0; i < _numCascades; ++i)
        {
            // Inverse matrix in a way to avoid precision error
            SLMat4f s, t;
            s.scale(1.0f / _p[i].m(0), 1.0f / _p[i].m(5), 1.0f / _p[i].m(10));
            t.translate((-_p[i].m(12)), (-_p[i].m(13)), (-_p[i].m(14)));

            stateGL->modelViewMatrix = stateGL->viewMatrix * _v[i].inverted() * s * t;

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
            stateGL->modelViewMatrix = stateGL->viewMatrix * _mvp[i].inverted();

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

    stateGL->modelViewMatrix = stateGL->viewMatrix * _mvp[0].inverted();

    vao.drawArrayAsColored(PT_lines,
                           SLCol4f::YELLOW,
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
        _v[0].lookAt(positionWS, positionWS + node->forwardWS(), node->upWS());
    }

    // Set projection matrix
    switch (_projection)
    {
        case P_monoOrthographic:
            _p[0].ortho(-_halfSize.x,
                        _halfSize.x,
                        -_halfSize.y,
                        _halfSize.y,
                        _clipNear,
                        _clipFar);
            break;

        case P_monoPerspective:
            _p[0].perspective(fov, 1.0f, _clipNear, _clipFar);
            break;

        default:
            SL_EXIT_MSG("Unsupported light projection");
    }

    // Set the model-view-projection matrix
    for (SLint i = 0; i < (_useCubemap ? 6 : 1); ++i)
        _mvp[i] = _p[0] * _v[i];
}
//-----------------------------------------------------------------------------
void SLShadowMap::findOptimalNearPlane(SLNode*      node,
                                       SLSceneView* sv,
                                       SLMat4f&     P,
                                       SLMat4f&     lv,
                                       SLPlane*     planes,
                                       SLVNode&     visibleNodes)
{
    if (node->drawBit(SL_DB_HIDDEN))
        return;

    if (node->castsShadows() == false)
        return;

    // We don't need to increase far plane distance
    float distance = planes[5].distToPoint(node->aabb()->centerWS());
    if (distance < -node->aabb()->radiusWS())
        return;
 
    for (int i = 0; i < 4; i++)
    {
        float distance = planes[i].distToPoint(node->aabb()->centerWS());
        if (distance < -node->aabb()->radiusWS())
            return;
    }

    // If object is behind the light's near plane, move the near plane back
    distance = planes[4].distToPoint(node->aabb()->centerWS());
    if (distance < node->aabb()->radiusWS())
    {
        float a = P.m(10);
        float b = P.m(14);
        float n = (b + 1.f) / a;
        float f = (b - 1.f) / a;
        n       = n + (distance - node->aabb()->radiusWS());
        P.m(10, -2.f / (f - n));
        P.m(14, -(f + n) / (f - n));
        SLFrustum::viewToFrustumPlanes(planes, P, lv);
    }

    visibleNodes.push_back(node);

    for (SLNode* child : node->children())
        findOptimalNearPlane(child, sv, P, lv, planes, visibleNodes);
}

//-----------------------------------------------------------------------------
/*!
SLShadowMap::drawNodesIntoDepthBuffer recursively renders all objects which
cast shadows
*/
void SLShadowMap::drawNodesDirectionalCulling(SLVNode      visibleNodes,
                                              SLSceneView* sv,
                                              SLMat4f&     P,
                                              SLMat4f&     lv,
                                              SLPlane*     planes)
{
    for (SLNode* node : visibleNodes)
    {
        SLGLState* stateGL        = SLGLState::instance();
        stateGL->modelViewMatrix  = lv * node->updateAndGetWM();
        stateGL->projectionMatrix = P;

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

    updateMVP();

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
        stateGL->viewMatrix       = _v[i];
        stateGL->projectionMatrix = _p[0];

        // Clear color buffer
        stateGL->clearColor(SLCol4f::BLACK);
        stateGL->clearColorDepthBuffer();

        // Draw meshes
        drawNodesIntoDepthBuffer(root, sv, _p[0], _v[i]);
    }

    _depthBuffers[0]->unbind();
}
//-----------------------------------------------------------------------------
SLVVec2f SLShadowMap::getShadowMapCascades(int   numCascades,
                                           float n,
                                           float f)
{
    SLVVec2f cascades;

    float ni = n;
    float fi = n;

    float factor = 30.0f;

    for (int i = 0; i < numCascades; i++)
    {
        ni = fi;
        fi = factor * n * pow((f / (factor * n)), (float)(i + 1) / (float)numCascades);
        cascades.push_back(SLVec2f(ni, fi));
    }
    return cascades;
}
//-----------------------------------------------------------------------------
void SLShadowMap::renderDirectionalLightCascaded(SLSceneView* sv,
                                                 SLNode*      root)
{
    _useCascaded       = true;
    SLGLState* stateGL = SLGLState::instance();

    SLint wrapMode = GL_CLAMP_TO_BORDER;

    // Create depth buffer
    static SLfloat borderColor[] = {1.0, 1.0, 1.0, 1.0};

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

    if (_depthBuffers.size() != 0 &&
        (_depthBuffers[0]->dimensions() != _textureSize ||
         _depthBuffers[0]->target() != GL_TEXTURE_2D ||
         _depthBuffers.size() != _numCascades))
    {
        _depthBuffers.erase(_depthBuffers.begin(), _depthBuffers.end());
    }

    SLVVec2f cascades  = getShadowMapCascades(_numCascades, _camera->clipNear(), _camera->clipFar());
    SLMat4f  camWN     = _camera->updateAndGetWM(); // camera space to world space
    SLNode*  lightNode = dynamic_cast<SLNode*>(_light);

    if (_depthBuffers.size() == 0)
    {
        for (int i = 0; i < cascades.size(); i++)
        {
            _depthBuffers.push_back(new SLGLDepthBuffer(_textureSize,
                                                        GL_NEAREST,
                                                        GL_NEAREST,
                                                        wrapMode,
                                                        borderColor,
                                                        GL_TEXTURE_2D));
        }
    }

    // for all subdivision of frustum
    for (int i = 0; i < cascades.size(); i++)
    {
        float   ni = cascades[i].x;
        float   fi = cascades[i].y;
        SLVec3f v  = camWN.translation() - camWN.axisZ().normalized() * (ni + fi) * 0.5f;

        SLMat4f lightViewMat; // world space to light space
        lightViewMat.lookAt(v, v + lightNode->forwardWS(), lightNode->upWS());

        SLVec3f frustumPoints[8];
        SLFrustum::getPointsEyeSpace(frustumPoints, _camera->fovV(), sv->scrWdivH(), ni, fi);

        float minx = FLT_MAX, maxx = FLT_MIN;
        float miny = FLT_MAX, maxy = FLT_MIN;
        float minz = FLT_MAX, maxz = FLT_MIN;
        for (int j = 0; j < 8; j++)
        {
            SLVec3f fp = lightViewMat * camWN * frustumPoints[j];
            if (fp.x < minx) minx = fp.x;
            if (fp.y < miny) miny = fp.y;
            if (fp.x > maxx) maxx = fp.x;
            if (fp.y > maxy) maxy = fp.y;
            if (fp.z < minz) minz = fp.z;
            if (fp.z > maxz) maxz = fp.z;
        }

        float sx = 2.f / (maxx - minx);
        float sy = 2.f / (maxy - miny);
        float sz = -2.f / (maxz - minz);

        SLVec3f t = SLVec3f(-0.5f * (maxx + minx),
                            -0.5f * (maxy + miny),
                            -0.5f * (maxz + minz));
        SLMat4f lightProjMat;
        lightProjMat.scale(sx, sy, sz);
        lightProjMat.translate(t);

        _depthBuffers[i]->bind();

        // Set matrices
        stateGL->viewMatrix       = lightViewMat;
        stateGL->projectionMatrix = lightProjMat;

        stateGL->viewportFB(0, 0, _textureSize.x, _textureSize.y);

        // Clear color buffer
        stateGL->clearColor(SLCol4f::BLACK);
        stateGL->clearColorDepthBuffer();

        // Draw meshes
        SLPlane planes[6];
        SLVNode visibleNodes;
        SLFrustum::viewToFrustumPlanes(planes, lightProjMat, lightViewMat);

        for (SLNode* child : root->children())
        {
            findOptimalNearPlane(child, sv, lightProjMat, lightViewMat, planes, visibleNodes);
        }

        _v[i]   = lightViewMat;
        _p[i]   = lightProjMat;
        _mvp[i] = lightProjMat * lightViewMat;

        drawNodesDirectionalCulling(visibleNodes,
                                    sv,
                                    lightProjMat,
                                    lightViewMat,
                                    planes);

        _depthBuffers[i]->unbind();
    }
}
//-----------------------------------------------------------------------------
