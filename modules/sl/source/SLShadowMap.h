//#############################################################################
//  File:      SLShadowMap.h
//  Date:      May 2020
//  Codestyle: https://github.com/cpvrlab/SLProject/wiki/SLProject-Coding-Style
//  Authors:   Michael Schertenleib, Marcus Hudritsch
//  License:   This software is provided under the GNU General Public License
//             Please visit: http://opensource.org/licenses/GPL-3.0
//#############################################################################

#ifndef SLSHADOWMAP_H
#define SLSHADOWMAP_H

#include <SL.h>
#include <SLEnums.h>
#include <SLPlane.h>
#include <SLMat4.h>
#include <SLNode.h>
#include <SLGLDepthBuffer.h>

class SLGLVertexArrayExt;
class SLLight;
class SLLightDirect;
class SLLightSpot;
class SLMaterial;
class SLMaterial;
class SLSceneView;
class SLCamera;
//-----------------------------------------------------------------------------
//! Class for standard and cascaded shadow mapping
/*! Shadow mapping is a technique to render shadows. The scene gets rendered
 * from the point of view of the lights which cast shadows. The resulting
 * depth-map of that render-pass can be used to determine which fragments are
 * affected by which lights. The standard fixed size shadow maps can be used
 * with all light types. The auto sized shadow maps get automatically sized
 * to a specified camera. At the moment only directional light get supported
 * with multiple cascaded shadow maps.
 */
class SLShadowMap
{
public:
    //! Ctor for standard fixed sized shadow mapping without cascades
    SLShadowMap(SLLight*       light,
                const SLfloat  lightClipNear = 0.1f,
                const SLfloat  lightClipFar  = 20.0f,
                const SLVec2f& size          = SLVec2f(8, 8),
                const SLVec2i& texSize       = SLVec2i(1024, 1024));

    //! Ctor for standard auto sized shadow mapping with cascades
    SLShadowMap(SLLight*       light,
                SLCamera*      camera,
                const SLVec2i& texSize     = SLVec2i(1024, 1024),
                const SLint    numCascades = 4);

    ~SLShadowMap();

    // Public methods
    void renderShadows(SLSceneView* sv, SLNode* root);
    void drawFrustum();
    void drawRays();

    // Setters
    void useCubemap(SLbool useCubemap) { _useCubemap = useCubemap; }
    void rayCount(const SLVec2i& rayCount) { _rayCount.set(rayCount); }
    void clipNear(SLfloat clipNear) { _lightClipNear = clipNear; }
    void clipFar(SLfloat clipFar) { _lightClipFar = clipFar; }
    void size(const SLVec2f& size)
    {
        _size.set(size);
        _halfSize.set(size / 2);
    }
    void textureSize(const SLVec2i& textureSize) { _textureSize.set(textureSize); }
    void numCascades(int numCascades) { _numCascades = numCascades; }
    void cascadesFactor(float factor) { _cascadesFactor = factor; }

    // Getters
    SLProjType       projection() { return _projection; }
    SLbool           useCubemap() const { return _useCubemap; }
    SLbool           useCascaded() const { return _useCascaded; }
    SLMat4f*         lightSpace() { return _lightSpace; }
    SLGLDepthBuffer* depthBuffer() { return _depthBuffers.at(0); }
    SLGLVDepthBuffer depthBuffers() { return _depthBuffers; }
    SLVec2i          rayCount() { return _rayCount; }
    SLfloat          lightClipNear() { return _lightClipNear; }
    SLfloat          lightClipFar() { return _lightClipFar; }
    SLVec2f          size() { return _size; }
    SLVec2i          textureSize() { return _textureSize; }
    int              numCascades() { return _numCascades; }
    int              maxCascades() { return _maxCascades; }
    float            cascadesFactor() { return _cascadesFactor; }
    SLCamera*        camera() { return _camera; }

    static SLuint drawCalls; //!< NO. of draw calls for shadow mapping

private:
    void     updateLightSpaces();
    void     renderDirectionalLightCascaded(SLSceneView* sv, SLNode* root);
    SLVVec2f getShadowMapCascades(int   numCascades,
                                  float camClipNear,
                                  float camClipFar);
    void     drawNodesIntoDepthBufferRec(SLNode*      node,
                                         SLSceneView* sv,
                                         SLMat4f&     lightView);
    void     lightCullingAdaptiveRec(SLNode*  node,
                                     SLMat4f& lightProj,
                                     SLMat4f& lightView,
                                     SLPlane* lightFrustumPlanes,
                                     SLVNode& visibleNodes);
    void     drawNodesDirectionalCulling(SLVNode      visibleNodes,
                                         SLSceneView* sv,
                                         SLMat4f&     lightView);

private:
    SLLight*            _light;          //!< The light which uses this shadow map
    SLProjType          _projection;     //!< Projection to use to create shadow map
    SLbool              _useCubemap;     //!< Flag if cubemap should be used for perspective projections
    SLbool              _useCascaded;    //!< Flag if cascaded shadow maps should be used
    SLint               _numCascades;    //!< Number of cascades for directional light shadow mapping
    SLint               _maxCascades;    //!< Max. number of cascades for for which the shader gets generated
    SLfloat             _cascadesFactor; //!< Factor that determines the cascades distribution
    SLMat4f             _lightView[6];   //!< Light view matrices
    SLMat4f             _lightProj[6];   //!< Light projection matrices
    SLMat4f             _lightSpace[6];  //!< Light space matrices (= _lightProj * _lightView)
    SLGLVDepthBuffer    _depthBuffers;   //!< Vector of framebuffers with texture
    SLGLVertexArrayExt* _frustumVAO;     //!< Visualization of light-space-frustum
    SLVec2i             _rayCount;       //!< Amount of rays drawn by drawRays()
    SLMaterial*         _material;       //!< Material used to render the shadow map
    SLfloat             _lightClipNear;  //!< Light frustum near clipping plane
    SLfloat             _lightClipFar;   //!< Light frustum far clipping plane
    SLVec2f             _size;           //!< Height and width of the frustum (only for SLLightDirect non cascaded)
    SLVec2f             _halfSize;       //!< _size divided by two (only for SLLightDirect non cascaded)
    SLVec2i             _textureSize;    //!< Size of the shadow map texture
    SLCamera*           _camera;         //!< Camera to witch the light frustums are adapted
};
//-----------------------------------------------------------------------------
#endif // SLSHADOWMAP_H
