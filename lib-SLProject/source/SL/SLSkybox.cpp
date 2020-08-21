//#############################################################################
//  File:      SLSkybox
//  Author:    Marcus Hudritsch
//  Date:      December 2017
//  Codestyle: https://github.com/cpvrlab/SLProject/wiki/SLProject-Coding-Style
//  Copyright: Marcus Hudritsch
//             This software is provide under the GNU General Public License
//             Please visit: http://opensource.org/licenses/GPL-3.0
//#############################################################################

#include <stdafx.h> // Must be the 1st include followed by  an empty line

#include <SLBox.h>
#include <SLCamera.h>
#include <SLGLTexture.h>
#include <SLMaterial.h>
#include <SLSceneView.h>
#include <SLSkybox.h>
#include <SLScene.h>

//-----------------------------------------------------------------------------
//! Default constructor
SLSkybox::SLSkybox(SLstring shaderFilePath, SLstring name) : SLNode(name)
{
}
//-----------------------------------------------------------------------------
//! Cubemap Constructor with cubemap images
/*! All resources allocated are stored in the SLScene vectors for textures,
materials, programs and meshes and get deleted at scene destruction.
*/
SLSkybox::SLSkybox(SLAssetManager* assetMgr,
                   SLstring        shaderFilePath,
                   SLstring        cubeMapXPos,
                   SLstring        cubeMapXNeg,
                   SLstring        cubeMapYPos,
                   SLstring        cubeMapYNeg,
                   SLstring        cubeMapZPos,
                   SLstring        cubeMapZNeg,
                   SLstring        name) : SLNode(name)
{
    // Create texture, material and program
    SLGLTexture* cubeMap    = new SLGLTexture(assetMgr,
                                           cubeMapXPos,
                                           cubeMapXNeg,
                                           cubeMapYPos,
                                           cubeMapYNeg,
                                           cubeMapZPos,
                                           cubeMapZNeg);
    SLMaterial*  matCubeMap = new SLMaterial(assetMgr, "matCubeMap");
    matCubeMap->textures().push_back(cubeMap);
    SLGLProgram* sp = new SLGLGenericProgram(assetMgr,
                                             shaderFilePath + "SkyBox.vert",
                                             shaderFilePath + "SkyBox.frag");
    matCubeMap->program(sp);

    // Create a box with max. point at min. parameter and vice versa.
    // Like this the boxes normals will point to the inside.
    assert(assetMgr && "SLSkybox: asset manager is currently mandatory for sky-boxes! Alternatively the live-time of the box has to be managed in the sky-box!");
    this->addMesh(new SLBox(assetMgr,
                            10,
                            10,
                            10,
                            -10,
                            -10,
                            -10,
                            "box",
                            matCubeMap));
}
//-----------------------------------------------------------------------------
//! Draw the skybox with a cube map with the camera in its center.
void SLSkybox::drawAroundCamera(SLSceneView* sv)
{
    assert(sv && "No SceneView passed to SLSkybox::drawAroundCamera");

    SLGLState* stateGL = SLGLState::instance();

    // Set the view transform
    stateGL->modelViewMatrix.setMatrix(stateGL->viewMatrix);

    // Put skybox at the cameras position
    this->translation(sv->camera()->translationWS());

    // Apply world transform
    stateGL->modelViewMatrix.multiply(this->updateAndGetWM().m());

    // Freeze depth buffer
    stateGL->depthMask(false);

    // Draw the box
#ifdef SL_RENDER_BY_MATERIAL
    this->drawMesh(sv);
#else
    this->drawMeshes(sv);
#endif

    // Unlock depth buffer
    stateGL->depthMask(true);
}
//-----------------------------------------------------------------------------
//! Returns the color in the skybox at the the specified direction dir
SLCol4f SLSkybox::colorAtDir(const SLVec3f& dir)
{
#ifdef SL_RENDER_BY_MATERIAL
    assert(_mesh);
    assert(_mesh->mat()->textures().empty());
    SLGLTexture* tex = _mesh->mat()->textures()[0];
#else
    assert(!_meshes.empty());
    assert(!_meshes[0]->mat()->textures().empty());
    SLGLTexture* tex = _meshes[0]->mat()->textures()[0];
#endif

    return tex->getTexelf(dir);
}
//-----------------------------------------------------------------------------
