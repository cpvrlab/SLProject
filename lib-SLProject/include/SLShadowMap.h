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

class SLGLDepthBuffer;
class SLGLVertexArrayExt;
class SLLight;
class SLLightDirect;
class SLLightSpot;
class SLMaterial;
class SLMaterial;
class SLNode;
class SLSceneView;

//-----------------------------------------------------------------------------
//! Class for shadow mapping
/*!
shadow mapping is a technique to render shadows. The scene gets rendered from
the point of view of the lights which cast shadows. The resulting depth-map of
that render-pass can be used to determine which fragments are affected by which
lights.
*/
class SLShadowMap
{
public:
    SLShadowMap();
    ~SLShadowMap();

    // Setters
    void rayCount(SLVec2i rayCount) { _rayCount.set(rayCount); }
    void clipNear(float clipNear) { _clipNear = clipNear; }
    void clipFar(float clipFar) { _clipFar = clipFar; }
    void size(SLVec2f size)
    {
        _size.set(size);
        _halfSize.set(size / 2);
    }
    void textureSize(SLVec2i textureSize) { _textureSize.set(textureSize); }

    // Getters
    SLMat4f          mvp() { return _mvp; }
    SLGLDepthBuffer* depthBuffer() { return _depthBuffer; }
    SLVec2i          rayCount() { return _rayCount; }
    SLfloat          clipNear() { return _clipNear; }
    SLfloat          clipFar() { return _clipFar; }
    SLVec2f          size() { return _size; }
    SLVec2i          textureSize() { return _textureSize; }

    // Other methods
    void drawFrustum();
    void drawRays();
    void updateMVP(SLLight* light, SLProjection projection);
    void render(SLSceneView* sv, SLNode* root);

private:
    void drawNodesIntoDepthBuffer(SLNode* node, SLSceneView* sv);

    SLMat4f             _v;           //!< View matrix
    SLMat4f             _p;           //!< Projection matrix
    SLMat4f             _mvp;         //!< Model-view-projection matrix
    SLGLDepthBuffer*    _depthBuffer; //!< Framebuffer and texture
    SLGLVertexArrayExt* _frustumVAO;  //!< Visualization of light-space-furstum
    SLVec2i             _rayCount;    //!< Amount of rays drawn by drawRays()
    SLMaterial*         _mat;         //!< Material used to render the shadow map
    SLfloat             _clipNear;    //!< Near clipping plane
    SLfloat             _clipFar;     //!< Far clipping plane
    SLVec2f             _size;        //!< Height and width of the frustum (only for SLLightDirect)
    SLVec2f             _halfSize;    //!< _size divided by 2
    SLVec2i             _textureSize; //!< Size of the shadow map texture
};
//-----------------------------------------------------------------------------
#endif //SLSHADOWMAP_H
