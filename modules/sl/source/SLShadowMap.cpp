//#############################################################################
//  File:      SLShadowMap.cpp
//  Author:    Michael Schertenleib
//  Date:      May 2020
//  Codestyle: https://github.com/cpvrlab/SLProject/wiki/SLProject-Coding-Style
//  Copyright: Michael Schertenleib
//             This software is provide under the GNU General Public License
//             Please visit: http://opensource.org/licenses/GPL-3.0
//#############################################################################

#include <SLGLDepthBuffer.h>
#include <SLGLProgramManager.h>
#include <SLGLState.h>
#include <SLGLVertexArrayExt.h>
#include <SLLight.h>
#include <SLMaterial.h>
#include <SLNode.h>
#include <SLShadowMap.h>
#include <Instrumentor.h>
#include <SLSceneView.h>

//-----------------------------------------------------------------------------
SLShadowMap::SLShadowMap(SLProjection   projection,
                         SLLight*       light,
                         float          clipNear,
                         float          clipFar,
                         const SLVec2f& size,
                         const SLVec2i& texSize)
{
    PROFILE_FUNCTION();

    _light       = light;
    _projection  = projection;
    _useCubemap  = false;
    _useCascaded = false;
    _depthBuffers = std::vector<SLGLDepthBuffer*>();
    _frustumVAO  = nullptr;
    _rayCount    = SLVec2i(0, 0);
    _mat         = nullptr;
    _clipNear    = clipNear;
    _clipFar     = clipFar;
    _size        = size;
    _halfSize    = _size / 2;
    _textureSize = texSize;
}
//-----------------------------------------------------------------------------
SLShadowMap::~SLShadowMap()
{
    _depthBuffers.erase(_depthBuffers.begin(), _depthBuffers.end());
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
                                        SLCol4f::GREEN,
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
void SLShadowMap::drawNodesIntoDepthBuffer(SLNode*        node,
                                           SLSceneView*   sv,
                                           const SLMat4f& v)
{
    if (node->drawBit(SL_DB_HIDDEN))
        return;

    SLGLState* stateGL       = SLGLState::instance();
    stateGL->modelViewMatrix = v * node->updateAndGetWM();

    if (node->castsShadows() &&
        node->mesh() &&
        node->mesh()->primitive() >= GL_TRIANGLES)
        node->mesh()->drawIntoDepthBuffer(sv, node, _mat);

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

    _depthBuffers[0]->unbind();
}
//-----------------------------------------------------------------------------

#include <algorithm>

void frustumGetPoints(std::vector<SLVec3f> &pts, SLVec3f pos, float fovV, float ratio, float clip)
{
    SLfloat t = tan(Utils::DEG2RAD * fovV * 0.5f) * clip;          // top
    SLfloat b = -t;                                                // bottom
    SLfloat r = ratio * t;                                         // right
    SLfloat l = -r;                                                // left
    pts.push_back(SLVec3f(r, t, -clip));
    pts.push_back(SLVec3f(r, b, -clip));
    pts.push_back(SLVec3f(l, t, -clip));
    pts.push_back(SLVec3f(l, b, -clip));
}

void SLShadowMap::renderDirectionalLightCascaded(SLSceneView* sv, SLNode* root)
{
    _useCascaded = true;
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
        _depthBuffers[0]->target() != GL_TEXTURE_2D))
    {
        _depthBuffers.erase(_depthBuffers.begin(), _depthBuffers.end());
    }

    if (_depthBuffers.size() == 0)
    {
        for (int i = 0; i < 6; i++)
        {
            _depthBuffers.push_back(new SLGLDepthBuffer(_textureSize,
                                                        GL_NEAREST,
                                                        GL_NEAREST,
                                                        wrapMode,
                                                        borderColor,
                                                        GL_TEXTURE_2D));
        }
    }

    SLCamera * camera = sv->camera();

    SLMat4f cm = camera->updateAndGetWM(); // camera to world space
    SLNode* node = dynamic_cast<SLNode*>(_light);

    float n = std::max(camera->clipNear(), 0.5f);
    float f = camera->clipFar();

    float ni = n;
    float fi = n;

    // for all subdivision of frustum
    for (int i = 0; i < 6; i++)
    {
        ni = fi;
        fi = n * pow((f/n), (i+1)/6.0f);

        SLVec3f v = cm.translation() - cm.axisZ().normalized() * (ni + fi) * 0.5f;

        SLMat4f lv; // world space to light space
        SLMat4f lp; // light space to light projected
        lv.lookAt(v, v + node->forwardOS(), node->upWS());
        lp.ortho(-_halfSize.x, _halfSize.x, -_halfSize.y, _halfSize.y, -50, 50); //TODO NEAR FAR CLIP LIGHT

        std::vector<SLVec3f> frustumPoints;
        frustumGetPoints(frustumPoints, v, camera->fovV(), sv->scrWdivH(), ni);
        frustumGetPoints(frustumPoints, v, camera->fovV(), sv->scrWdivH(), fi);

        float minx = 99999, miny = 99999;
        float maxx = -99999, maxy = -99999;
        float minz = 99999;
        for (int j = 0; j < 8; j++)
        {
            SLVec3f fp = lv * cm * frustumPoints[j];
            if (fp.x < minx)
                minx = fp.x;
            if (fp.y < miny)
                miny = fp.y;
            if (fp.x > maxx)
                maxx = fp.x;
            if (fp.y > maxy)
                maxy = fp.y;

            if (fp.z < minz)
                minz = fp.z;
        }

        float maxz = -minz + 100;

        float sx = 2.f/(maxx - minx);
        float sy = 2.f/(maxy - miny);
        float sz = -2.f/(maxz - minz);
        SLMat4f C;
        C.identity();
        C.scale(sx, sy, sz);
        C.translate(-0.5f * sx * (maxx + minx), -0.5f * sy * (maxy + miny), -0.5f * sz * (maxz - minz));

        _depthBuffers[i]->bind();

        // Set matrices
        stateGL->viewMatrix       = lv;
        stateGL->projectionMatrix = C;
        _v[i] = lv;
        _p = C;
        _mvp[i] = _p * lv;

        stateGL->viewport(0, 0, _textureSize.x, _textureSize.y);

        // Clear color buffer
        stateGL->clearColor(SLCol4f::BLACK);
        stateGL->clearColorDepthBuffer();

        // Draw meshes
        drawNodesIntoDepthBuffer(root, sv, lv);

        _depthBuffers[i]->unbind();
    }
}
