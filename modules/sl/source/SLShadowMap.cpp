//#############################################################################
//  File:      SLShadowMap.cpp
//  Date:      May 2020
//  Codestyle: https://github.com/cpvrlab/SLProject/wiki/SLProject-Coding-Style
//  Authors:   Michael Schertenleib, Marcus Hudritsch
//  License:   This software is provided under the GNU General Public License
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
#include <Profiler.h>
#include <SLSceneView.h>
#include <SLCamera.h>
#include <SLFrustum.h>
#include <SLNodeLOD.h>
#include <SLLightSpot.h>
#include <SLLightRect.h>
#include <SLLightDirect.h>

//-----------------------------------------------------------------------------
SLuint SLShadowMap::drawCalls = 0; //!< NO. of draw calls for shadow mapping
//-----------------------------------------------------------------------------
/*! Ctor for standard fixed size shadow map for any type of light
 * @param light Pointer to the light for which the shadow is created
 * @param lightClipNear The light frustums near clipping distance
 * @param lightClipFar The light frustums near clipping distance
 * @param size Ignored for rectangular lights
 * @param texSize Shadow texture map size
 */
SLShadowMap::SLShadowMap(SLLight*       light,
                         float          lightClipNear,
                         float          lightClipFar,
                         const SLVec2f& size,
                         const SLVec2i& texSize)
{
    assert(light && "SLShadowMap::SLShadowMap: No light passed");
    assert(lightClipFar > lightClipNear &&
           "SLShadowMap::SLShadowMap: Invalid clip distances passed");

    PROFILE_FUNCTION();

    if (typeid(*light) == typeid(SLLightDirect))
        _projection = P_monoOrthographic;
    else if (typeid(*light) == typeid(SLLightSpot))
        _projection = P_monoPerspective;
    else if (typeid(*light) == typeid(SLLightRect))
        _projection = P_monoPerspective;
    else
        SL_EXIT_MSG("SLShadowMap::SLShadowMap: Unknown light type");

    _light         = light;
    _useCubemap    = false;
    _useCascaded   = false;
    _depthBuffers  = SLGLVDepthBuffer();
    _frustumVAO    = nullptr;
    _rayCount      = SLVec2i(0, 0);
    _material      = nullptr;
    _lightClipNear = lightClipNear;
    _lightClipFar  = lightClipFar;
    _size          = size;
    _halfSize      = _size / 2;
    _textureSize   = texSize;
    _camera        = nullptr;
    _numCascades   = 0;
}
//-----------------------------------------------------------------------------
/*! Ctor for auto sized cascaded shadow mapping
 * @param light Pointer to the light for which the shadow is created
 * @param camera Pointer to the camera for witch the shadow map gets sized
 * @param texSize Shadow texture map size (equal for all cascades)
 * @param numCascades NO. of cascades for cascaded shadow mapping
 */
SLShadowMap::SLShadowMap(SLLight*       light,
                         SLCamera*      camera,
                         const SLVec2i& texSize,
                         int            numCascades)
{
    assert(light && "SLShadowMap::SLShadowMap: No light passed");
    assert(camera && "SLShadowMap::SLShadowMap: No camera passed");
    assert(numCascades >= 0 && numCascades <= 5 &&
           "SLShadowMap::SLShadowMap: Invalid NO.of cascades (0-5)");

    PROFILE_FUNCTION();

    if (typeid(*light) == typeid(SLLightDirect))
        _projection = P_monoOrthographic;
    else
        SL_EXIT_MSG("Auto sized shadow maps only exist for directional lights yet.");

    _light          = light;
    _useCubemap     = false;
    _useCascaded    = true;
    _depthBuffers   = SLGLVDepthBuffer();
    _frustumVAO     = nullptr;
    _rayCount       = SLVec2i(0, 0);
    _material       = nullptr;
    _camera         = camera;
    _numCascades    = numCascades;
    _maxCascades    = numCascades;
    _textureSize    = texSize;
    _size           = SLVec2f(0, 0); // will be ignored and automatically calculated
    _halfSize       = SLVec2f(0, 0); // will be ignored and automatically calculated
    _lightClipNear  = 0.1f;          // will be ignored and automatically calculated
    _lightClipFar   = 20.f;          // will be ignored and automatically calculated
    _cascadesFactor = 30.f;
}
//-----------------------------------------------------------------------------
SLShadowMap::~SLShadowMap()
{
    _depthBuffers.erase(_depthBuffers.begin(), _depthBuffers.end());
    delete _frustumVAO;
    delete _material;
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

            stateGL->modelMatrix = _lightView[i].inverted() * s * t;
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
            stateGL->modelMatrix = _lightSpace[i].inverted();
            _frustumVAO->drawArrayAsColored(PT_lines,
                                            SLCol4f::GREEN,
                                            1.0f,
                                            0,
                                            (SLuint)P.size());
        }
    }
}

//-----------------------------------------------------------------------------
/*! SLShadowMap::drawRays draws sample rays of the light for visualization
 * purpose only. Gets turned on when the light node is selected.
 */
void SLShadowMap::drawRays()
{
#ifndef SL_GLES              // Reading the depth-buffer with GLES is non-trivial

    if (_useCubemap) return; // Not implemented for cubemaps
    SLint w = _rayCount.x;
    SLint h = _rayCount.y;
    if (w == 0 || h == 0) return;

    SLGLState* stateGL = SLGLState::instance();
    SLVVec3f   P;

    _depthBuffers[0]->bind();

    SLfloat  pixelWidth  = (SLfloat)_textureSize.x / w;
    SLfloat  pixelHeight = (SLfloat)_textureSize.y / h;
    SLfloat* depths      = _depthBuffers[0]->readPixels();

    for (SLint x = 0; x < w; ++x)
    {
        for (SLint y = 0; y < h; ++y)
        {
            SLint   pixelX     = (SLint)(pixelWidth * (x + 0.5f));
            SLint   pixelY     = (SLint)(pixelHeight * (y + 0.5f));
            SLfloat viewSpaceX = Utils::lerp((x + 0.5f) / w, -1.0f, 1.0f);
            SLfloat viewSpaceY = Utils::lerp((y + 0.5f) / h, -1.0f, 1.0f);
            SLfloat depth      = depths[pixelY * _depthBuffers[0]->dimensions().x + pixelX] * 2 - 1;
            if (depth == 1.0f)
                continue;
            P.push_back(SLVec3f(viewSpaceX, viewSpaceY, -1.0f));
            P.push_back(SLVec3f(viewSpaceX, viewSpaceY, depth));
        }
    }

    delete depths;
    _depthBuffers[0]->unbind();
    if (P.empty()) return;

    SLGLVertexArrayExt vao;
    vao.generateVertexPos(&P);

    stateGL->modelMatrix = _lightSpace[0].inverted();
    vao.drawArrayAsColored(PT_lines,
                           SLCol4f::YELLOW,
                           1.0f,
                           0,
                           (SLuint)P.size());

#endif
}
//-----------------------------------------------------------------------------
//! SLShadowMap::updateLightSpaces updates a light view projection matrix
void SLShadowMap::updateLightSpaces()
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
                                _lightClipNear,
                                _lightClipFar);
            break;

        case P_monoPerspective:
            _lightProj[0].perspective(fov, 1.0f, _lightClipNear, _lightClipFar);
            break;

        default:
            SL_EXIT_MSG("Unsupported light projection");
    }

    // Set the model-view-projection matrix
    for (SLint i = 0; i < (_useCubemap ? 6 : 1); ++i)
        _lightSpace[i] = _lightProj[0] * _lightView[i];
}
//-----------------------------------------------------------------------------
/*! Returns the visible nodes inside the light frustum
 * Check if the passed node is inside the light frustum and add if so to the
 * visibleNodes vector. The goal is to exit this function as fast as possible
 * if the node is not visible from the light hence gets not lighted.
 * @param node Node to cull or add to to visibleNodes vector
 * @param lightProj The cascades light projection matrix that gets adapted
 * @param lightView The cascades light view matrix
 * @param lightFrustumPlanes The six light frustum planes
 * @param visibleNodes Vector to push the lighted nodes
 */
void SLShadowMap::lightCullingAdaptiveRec(SLNode*  node,
                                          SLMat4f& lightProj,
                                          SLMat4f& lightView,
                                          SLPlane* lightFrustumPlanes,
                                          SLVNode& visibleNodes)
{
    assert(node &&
           "SLShadowMap::lightCullingAdaptiveRec: No node passed.");
    assert(node &&
           "SLShadowMap::lightCullingAdaptiveRec: No lightFrustumPlanes passed.");

    // Exclude LOD level nodes
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
    float distance = lightFrustumPlanes[5].distToPoint(node->aabb()->centerWS());
    if (distance < -node->aabb()->radiusWS())
        return;

    // Check the 4 side planes of the frustum
    for (int i = 0; i < 4; i++)
    {
        distance = lightFrustumPlanes[i].distToPoint(node->aabb()->centerWS());
        if (distance < -node->aabb()->radiusWS())
            return;
    }

    // If object is behind the light's near plane, move the near plane back
    if (node->mesh()) // Don't add empty group nodes
    {
        distance = lightFrustumPlanes[4].distToPoint(node->aabb()->centerWS());
        if (distance < node->aabb()->radiusWS())
        {
            float a = lightProj.m(10);
            float b = lightProj.m(14);
            float n = (b + 1.f) / a;
            float f = (b - 1.f) / a;
            n       = n + (distance - node->aabb()->radiusWS());
            lightProj.m(10, -2.f / (f - n));
            lightProj.m(14, -(f + n) / (f - n));
            SLFrustum::viewToFrustumPlanes(lightFrustumPlanes,
                                           lightProj,
                                           lightView);
        }

        // If the node survived until now it can cast a shadow in this cascade
        visibleNodes.push_back(node);
    }

    // Now recursively cull the children nodes
    for (SLNode* child : node->children())
        lightCullingAdaptiveRec(child,
                                lightProj,
                                lightView,
                                lightFrustumPlanes,
                                visibleNodes);
}
//-----------------------------------------------------------------------------
/*! SLShadowMap::drawNodesDirectionalCulling draw all nodes in the vector
 * visibleNodes.
 * @param visibleNodes Vector of visible nodes
 * @param sv Pointer to the sceneview
 * @param lightView The light view matrix
 */
void SLShadowMap::drawNodesDirectionalCulling(SLVNode      visibleNodes,
                                              SLSceneView* sv,
                                              SLMat4f&     lightView)
{
    SLGLState* stateGL = SLGLState::instance();

    for (SLNode* node : visibleNodes)
    {
        if (node->castsShadows() &&
            node->mesh() &&
            node->mesh()->primitive() >= GL_TRIANGLES)
        {
            stateGL->viewMatrix  = lightView;
            stateGL->modelMatrix = node->updateAndGetWM();
            node->mesh()->drawIntoDepthBuffer(sv, node, _material);
            SLShadowMap::drawCalls++;
        }
    }
}
//-----------------------------------------------------------------------------
/*! Recursive node drawing function for standard shadow map drawing.
 * @param node The node do draw
 * @param sv Pointer to the sceneview
 * @param lightView The light view matrix
 */
void SLShadowMap::drawNodesIntoDepthBufferRec(SLNode*      node,
                                              SLSceneView* sv,
                                              SLMat4f&     lightView)
{
    assert(node && "SLShadowMap::drawNodesIntoDepthBufferRec: No node passed.");

    if (node->drawBit(SL_DB_HIDDEN))
        return;
    SLGLState* stateGL   = SLGLState::instance();
    stateGL->viewMatrix  = lightView;
    stateGL->modelMatrix = node->updateAndGetWM();

    if (node->castsShadows() &&
        node->mesh() &&
        node->mesh()->primitive() >= GL_TRIANGLES)
    {
        node->mesh()->drawIntoDepthBuffer(sv, node, _material);
        SLShadowMap::drawCalls++;
    }

    for (SLNode* child : node->children())
        drawNodesIntoDepthBufferRec(child, sv, lightView);
}
//-----------------------------------------------------------------------------
/*! SLShadowMap::render Toplevel entry function for shadow map rendering.
 * @param sv Pointer of the sceneview
 * @param root Pointer to the root node of the scene
 */
void SLShadowMap::renderShadows(SLSceneView* sv, SLNode* root)
{
    assert(root && "SLShadowMap::render: No root node passed.");

    PROFILE_FUNCTION();

    if (_projection == P_monoOrthographic && _camera != nullptr)
    {
        renderDirectionalLightCascaded(sv, root);
        return;
    }

    // Create Material
    if (_material == nullptr)
        _material = new SLMaterial(
          nullptr,
          "shadowMapMaterial",
          nullptr,
          nullptr,
          nullptr,
          nullptr,
          SLGLProgramManager::get(SP_depth));

    // Create depth buffer
    static SLfloat borderColor[] = {1.0, 1.0, 1.0, 1.0};

    updateLightSpaces();

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

    _depthBuffers[0]->bind();

    for (SLint i = 0; i < (_useCubemap ? 6 : 1); ++i)
    {
        if (_useCubemap)
            _depthBuffers[0]->bindFace(GL_TEXTURE_CUBE_MAP_POSITIVE_X + i);

        // Set OpenGL states
        SLGLState* stateGL = SLGLState::instance();
        stateGL->viewport(0, 0, _textureSize.x, _textureSize.y);
        stateGL->viewMatrix       = _lightView[i];
        stateGL->projectionMatrix = _lightProj[0];
        stateGL->clearColor(SLCol4f::BLACK);
        stateGL->clearColorDepthBuffer();

        /////////////////////////////////////////////////////
        drawNodesIntoDepthBufferRec(root, sv, _lightView[i]);
        /////////////////////////////////////////////////////
    }

    _depthBuffers[0]->unbind();
}
//-----------------------------------------------------------------------------
/*! Returns a vector of near and far clip distances for all shadow cascades
 * along the cameras view direction.
 * @param numCascades NO. of cascades
 * @param camClipNear The cameras near clipping distance
 * @param camClipFar The cameras far clipping distance
 * @return A SLVVec2f vector with the near and far clip distances
 */
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
/*! Renders the nodes into cascaded shadow maps for directional lights
 * @param sv Pointer of the sceneview
 * @param root Pointer to the root node of the scene
 */
void SLShadowMap::renderDirectionalLightCascaded(SLSceneView* sv,
                                                 SLNode*      root)
{
    assert(root && "LShadowMap::renderDirectionalLightCascaded: no root node");

    // Create depth buffer
    static SLfloat borderColor[] = {1.0, 1.0, 1.0, 1.0};

    // Create material
    if (_material == nullptr)
        _material = new SLMaterial(
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

#ifdef SL_GLES
    SLint wrapMode = GL_CLAMP_TO_EDGE;
#else
    SLint wrapMode = GL_CLAMP_TO_BORDER;
#endif

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
        float cn = cascades[i].x;
        float cf = cascades[i].y;

        // The cascades middle point on the view direction in WS
        SLVec3f cm = camWM.translation() - camWM.axisZ().normalized() * (cn + cf) * 0.5f;

        // Build the view matrix with lookAt method
        SLMat4f lightViewMat; // world space to light space
        lightViewMat.lookAt(cm, cm + lightNode->forwardWS(), lightNode->upWS());

        // Get the 8 camera frustum points in view space
        SLVec3f camFrustumPoints[8];
        SLFrustum::getPointsInViewSpace(camFrustumPoints,
                                        _camera->fovV(),
                                        sv->scrWdivH(),
                                        cn,
                                        cf);

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
        SLPlane lightFrustumPlanes[6];
        SLVNode visibleNodes;
        SLFrustum::viewToFrustumPlanes(lightFrustumPlanes,
                                       lightProjMat,
                                       lightViewMat);
        for (SLNode* child : root->children())
        {
            lightCullingAdaptiveRec(child,
                                    lightProjMat,
                                    lightViewMat,
                                    lightFrustumPlanes,
                                    visibleNodes);
        }

        _lightView[i]  = lightViewMat;
        _lightProj[i]  = lightProjMat;
        _lightSpace[i] = lightProjMat * lightViewMat;

        _depthBuffers[i]->bind();

        // Set OpenGL states for depth buffer rendering
        SLGLState* stateGL = SLGLState::instance();
        stateGL->viewport(0, 0, _textureSize.x, _textureSize.y);
        stateGL->clearColor(SLCol4f::BLACK);
        stateGL->clearColorDepthBuffer();
        stateGL->projectionMatrix = lightProjMat;
        stateGL->viewMatrix       = lightViewMat;

        ////////////////////////////////////////////////////////////
        drawNodesDirectionalCulling(visibleNodes, sv, lightViewMat);
        ////////////////////////////////////////////////////////////

        _depthBuffers[i]->unbind();
    }
}
//-----------------------------------------------------------------------------
