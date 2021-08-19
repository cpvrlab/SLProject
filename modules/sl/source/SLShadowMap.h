//#############################################################################
//  File:      SLShadowMap.h
//  Author:    Michael Schertenleib
//  Date:      May 2020
//  Codestyle: https://github.com/cpvrlab/SLProject/wiki/SLProject-Coding-Style
//  Copyright: Michael Schertenleib
//             This software is provide under the GNU General Public License
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
//! Class for shadow mapping
/*!
Shadow mapping is a technique to render shadows. The scene gets rendered from
the point of view of the lights which cast shadows. The resulting depth-map of
that render-pass can be used to determine which fragments are affected by which
lights.
*/
class SLShadowMap
{
public:
    SLShadowMap(SLProjection   projection,
                SLLight*       light,
                float          clipNear = 0.1f,
                float          clipFar  = 20.0f,
                const SLVec2f& size     = SLVec2f(8, 8),
                const SLVec2i& texSize  = SLVec2i(1024, 1024));

    SLShadowMap(SLProjection   projection,
                SLLight*       light,
                SLCamera*      camera,
                const SLVec2f& size        = SLVec2f(8, 8),
                const SLVec2i& texSize     = SLVec2i(1024, 1024),
                int            numCascades = 4);

    ~SLShadowMap();

    // Setters
    void useCubemap(SLbool useCubemap) { _useCubemap = useCubemap; }
    void rayCount(const SLVec2i& rayCount) { _rayCount.set(rayCount); }
    void clipNear(SLfloat clipNear) { _clipNear = clipNear; }
    void clipFar(SLfloat clipFar) { _clipFar = clipFar; }
    void size(const SLVec2f& size)
    {
        _size.set(size);
        _halfSize.set(size / 2);
    }
    void textureSize(const SLVec2i& textureSize) { _textureSize.set(textureSize); }
    void numCascades(int numCascades) { _numCascades = numCascades; }
    void cascadesFactor(float factor) { _cascadesFactor = factor; }

    // Getters
    SLProjection     projection() { return _projection; }
    SLbool           useCubemap() const { return _useCubemap; }
    SLbool           useCascaded() const { return _useCascaded; }
    SLMat4f*         lightSpace() { return _lightSpace; }
    SLGLDepthBuffer* depthBuffer() { return _depthBuffers.at(0); }
    SLGLVDepthBuffer depthBuffers() { return _depthBuffers; }
    SLVec2i          rayCount() { return _rayCount; }
    SLfloat          clipNear();
    SLfloat          clipFar();
    SLVec2f          size() { return _size; }
    SLVec2i          textureSize() { return _textureSize; }
    int              numCascades() { return _numCascades; }
    float            cascadesFactor() { return _cascadesFactor; }
    SLCamera*        camera() { return _camera; }

    // Other methods
    void drawFrustum();
    void drawRays();
    void updateLightViewProj();
    void render(SLSceneView* sv, SLNode* root);
    void renderDirectionalLightCascaded(SLSceneView* sv, SLNode* root);

private:
    SLLight*            _light;         //!< The light which uses this shadow map
    SLProjection        _projection;    //!< Projection to use to create shadow map
    SLbool              _useCubemap;    //!< Flag if cubemap should be used for perspective projections
    SLbool              _useCascaded;   //!< Flag if cascaded shadow maps should be used
    SLint               _numCascades;   //!< Number of cascades
    SLMat4f             _lightView[6];  //!< Light view matrices
    SLMat4f             _lightProj[6];  //!< Light projection matrices
    SLMat4f             _lightSpace[6]; //!< Light space matrices
    SLGLVDepthBuffer    _depthBuffers;  //!< Framebuffer and texture
    SLGLVertexArrayExt* _frustumVAO;    //!< Visualization of light-space-frustum
    SLVec2i             _rayCount;      //!< Amount of rays drawn by drawRays()
    SLMaterial*         _mat;           //!< Material used to render the shadow map
    SLfloat             _clipNear;      //!< Light near clipping plane
    SLfloat             _clipFar;       //!< Light far clipping plane
    SLVec2f             _size;          //!< Height and width of the frustum (only for SLLightDirect)
    SLVec2f             _halfSize;      //!< _size divided by two
    SLVec2i             _textureSize;   //!< Size of the shadow map texture
    SLCamera*           _camera;        //!< Camera to witch the light frustums are adapted

    SLfloat _cascadesFactor;

    SLVVec2f getShadowMapCascades(int numCascades, float camClipNear, float camClipFar);
    void     drawNodesIntoDepthBuffer(SLNode*      node,
                                      SLSceneView* sv,
                                      SLMat4f&     p,
                                      SLMat4f&     v);
    void     lightCullingAdaptiveRec(SLNode*  node,
                                     SLMat4f& lightProj,
                                     SLMat4f& lightView,
                                     SLPlane* frustumPlanes,
                                     SLVNode& visibleNodes);
    void     drawNodesDirectionalCulling(SLVNode      visibleNodes,
                                         SLSceneView* sv,
                                         SLMat4f&     lightProj,
                                         SLMat4f&     lightView,
                                         SLPlane*     planes);
    /*
    void     drawNodesDirectional(SLNode*      node,
                                  SLSceneView* sv,
                                  SLMat4f&     P,
                                  SLMat4f&     lv);
    void     drawNodesIntoDepthBufferCulling(SLNode*      node,
                                             SLSceneView* sv,
                                             SLMat4f&     p,
                                             SLMat4f&     v,
                                             SLPlane*     planes);
                                             */
};
//-----------------------------------------------------------------------------
#endif // SLSHADOWMAP_H
