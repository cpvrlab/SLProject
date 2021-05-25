//#############################################################################
//  File:      SLSkybox
//  Author:    Marcus Hudritsch
//  Date:      December 2017
//  Codestyle: https://github.com/cpvrlab/SLProject/wiki/SLProject-Coding-Style
//  Copyright: Marcus Hudritsch
//             This software is provide under the GNU General Public License
//             Please visit: http://opensource.org/licenses/GPL-3.0
//#############################################################################

#ifndef SLSKYBOX_H
#define SLSKYBOX_H

#include <SLEnums.h>
#include <SLNode.h>
#include <SLProjectScene.h>

class SLSceneView;
class SLAssetManager;

//-----------------------------------------------------------------------------
//! Skybox node class with a SLBox mesh
/*! The skybox instance is a node with a SLBox mesh with inwards pointing
normals. It gets drawn in SLSceneView::draw3DGL with frozen depth buffer and a
special cubemap shader. The box is always with the active camera in its
center. It has to be created in SLScene::onLoad and assigned to the skybox
pointer of SLSceneView. See the Skybox shader example.
*/
class SLSkybox : public SLNode
{
public:
    SLSkybox(SLAssetManager* assetMgr,
             SLstring        shaderPath,
             SLstring        cubeMapXPos,
             SLstring        cubeMapXNeg,
             SLstring        cubeMapYPos,
             SLstring        cubeMapYNeg,
             SLstring        cubeMapZPos,
             SLstring        cubeMapZNeg,
             SLstring        name = "Default Skybox");

    SLSkybox(SLProjectScene* projectScene,
             SLstring        shaderPath,
             SLstring        hdrImage,
             SLVec2i         resolution,
             SLstring        name            = "HDR Skybox",
             SLGLUniform1f*  exposureUniform = nullptr);

    ~SLSkybox() { ; }

    SLCol4f colorAtDir(const SLVec3f& dir);

    void drawAroundCamera(SLSceneView* sv);

private:
    void         build();
    SLGLTexture* _environmentCubemap;
    SLGLTexture* _irradianceCubemap;
    SLGLTexture* _roughnessCubemap;
    SLGLTexture* _brdfLUTTexture;
    SLGLTexture* _hdrTexture;
    SLbool       _isHDR;   //!< flag for HDR skyboxes
    SLbool       _isBuilt; //!< flag for late HDR skybox building
};
//-----------------------------------------------------------------------------
#endif // #define SLSKYBOX_H
