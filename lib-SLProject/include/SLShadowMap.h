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

class SLGLDepthBuffer;
class SLGLVertexArrayExt;
class SLLight;
class SLLightDirect;
class SLMaterial;
class SLMaterial;
class SLNode;
class SLSceneView;

//-----------------------------------------------------------------------------
//! Class for shadow-mapping
/*!
Shadow-mapping is a technique to render shadows. The scene gets rendered from
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
    void size(SLVec2f size) { _size.set(size); }
    void size(SLfloat width, SLfloat height) { _size.set(width, height); }

    // Getters
    SLVec2f          size() { return _size; }
    SLMat4f          lightSpace() { return _mvp; }
    SLGLDepthBuffer* depthBuffer() { return _depthBuffer; }

    // Other methods
    void drawFrustum();
    void updateLightSpaceMatrix(SLLightDirect* light);
    void render(SLSceneView* sv, SLNode* root);

private:
    void drawNodesIntoDepthBuffer(SLNode* node, SLSceneView* sv);

    SLMat4f             _vm;          //!< Look-at matrix of the shadow-map
    SLMat4f             _p;           //!< Projection of the shadow-map
    SLMat4f             _mvp;         //!< Used to convert WorldSpace to LightSpace
    SLGLDepthBuffer*    _depthBuffer; //!< Framebuffer and texture
    SLGLVertexArrayExt* _frustumVAO;  //!< Visualization of light-space-furstum
    SLMaterial*         _mat;         //!< Material used to render the shadow-map
    SLfloat             _clipNear;    //!< Near clipping plane
    SLfloat             _clipFar;     //!< Far clipping plane
    SLVec2f             _size;        //!< Height and width of the frustum
    SLVec2i             _textureSize; //!< Size of the shadow-map-texture
};
//-----------------------------------------------------------------------------
#endif //SLSHADOWMAP_H
