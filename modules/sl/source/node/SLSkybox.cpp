//#############################################################################
//  File:      SLSkybox
//  Authors:    Marcus Hudritsch
//  Date:      December 2017
//  Codestyle: https://github.com/cpvrlab/SLProject/wiki/SLProject-Coding-Style
//  Copyright: Marcus Hudritsch
//             This software is provide under the GNU General Public License
//             Please visit: http://opensource.org/licenses/GPL-3.0
//#############################################################################

#include <SLBox.h>
#include <SLGLProgramGeneric.h>
#include <SLGLFrameBuffer.h>
#include <SLGLTexture.h>
#include <SLGLTextureIBL.h>
#include <SLMaterial.h>
#include <SLSceneView.h>
#include <SLSkybox.h>
#include <SLProjectScene.h>

//-----------------------------------------------------------------------------
//! Cubemap Constructor with cubemap images
/*! All resources allocated are stored in the SLScene vectors for textures,
materials, programs and meshes and get deleted at scene destruction.
*/
SLSkybox::SLSkybox(SLAssetManager* assetMgr,
                   SLstring        shaderPath,
                   SLstring        cubeMapXPos,
                   SLstring        cubeMapXNeg,
                   SLstring        cubeMapYPos,
                   SLstring        cubeMapYNeg,
                   SLstring        cubeMapZPos,
                   SLstring        cubeMapZNeg,
                   SLstring        name) : SLNode(name)
{
    assert(assetMgr &&
           "SLSkybox: asset manager is currently mandatory for sky-boxes! "
           "Alternatively the live-time of the box has to be managed in the sky-box!");

    // Set HDR flag to false, this is a normal SkyBox
    _isHDR              = false;
    _isBuilt            = true;
    _environmentCubemap = nullptr;
    _irradianceCubemap  = nullptr;
    _roughnessCubemap   = nullptr;
    _brdfLUTTexture     = nullptr;
    _hdrTexture         = nullptr;

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
    SLGLProgram* sp = new SLGLProgramGeneric(assetMgr,
                                             shaderPath + "SkyBox.vert",
                                             shaderPath + "SkyBox.frag");
    matCubeMap->program(sp);

    // Create a box with max. point at min. parameter and vice versa.
    // Like this the boxes normals will point to the inside.
    addMesh(new SLBox(assetMgr,
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
/*! This constructor generates a cube map skybox from a HDR Image and also
all the textures needed for image based lighting and store them in the textures
of the material of this sky box.
*/
SLSkybox::SLSkybox(SLProjectScene* projectScene,
                   SLstring        shaderPath,
                   SLstring        hdrImage,
                   SLVec2i         resolution,
                   SLstring        name,
                   SLGLUniform1f*  exposureUniform) : SLNode(name)
{
    // Set HDR flag to true, this is a HDR SkyBox
    _isHDR   = true;
    _isBuilt = false;

    // Create shader program for the background
    SLGLProgram* backgroundShader = new SLGLProgramGeneric(projectScene,
                                                           shaderPath + "PBR_SkyboxHDR.vert",
                                                           shaderPath + "PBR_SkyboxHDR.frag");

    // if an exposure uniform is passed the initialize this exposure with it otherwise it is constant at 1.0
    SLGLUniform1f* exposure = exposureUniform ? exposureUniform : new SLGLUniform1f(UT_const, "u_exposure", 1.0f);
    projectScene->eventHandlers().push_back(exposure);
    backgroundShader->addUniform1f(exposure);

    // Create texture from the HDR Image
    _hdrTexture = new SLGLTexture(projectScene,
                                  hdrImage,
                                  GL_LINEAR,
                                  GL_LINEAR,
                                  TT_hdr,
                                  GL_CLAMP_TO_EDGE,
                                  GL_CLAMP_TO_EDGE);

    _environmentCubemap = new SLGLTextureIBL(projectScene,
                                             shaderPath,
                                             _hdrTexture,
                                             resolution,
                                             TT_environmentCubemap,
                                             GL_TEXTURE_CUBE_MAP,
                                             GL_LINEAR_MIPMAP_LINEAR);

    _irradianceCubemap = new SLGLTextureIBL(projectScene,
                                            shaderPath,
                                            _environmentCubemap,
                                            SLVec2i(32, 32),
                                            TT_irradianceCubemap,
                                            GL_TEXTURE_CUBE_MAP);

    _roughnessCubemap = new SLGLTextureIBL(projectScene,
                                           shaderPath,
                                           _environmentCubemap,
                                           SLVec2i(128, 128),
                                           TT_roughnessCubemap,
                                           GL_TEXTURE_CUBE_MAP);

    _brdfLUTTexture = new SLGLTextureIBL(projectScene,
                                         shaderPath,
                                         nullptr,
                                         SLVec2i(512, 512),
                                         TT_brdfLUT,
                                         GL_TEXTURE_2D);

    // Create the material of the sky box and store there the other texture to be used for other materials
    SLMaterial* hdrMaterial = new SLMaterial(projectScene, "matCubeMap");
    hdrMaterial->textures().push_back(_environmentCubemap);
    hdrMaterial->textures().push_back(_irradianceCubemap);
    hdrMaterial->textures().push_back(_roughnessCubemap);
    hdrMaterial->textures().push_back(_brdfLUTTexture);
    hdrMaterial->program(backgroundShader);

    // Create the box for the sky box
    addMesh(new SLBox(projectScene,
                      10,
                      10,
                      10,
                      -10,
                      -10,
                      -10,
                      "box",
                      hdrMaterial));
}

//-----------------------------------------------------------------------------
//! Builds all texture for HDR image based lighting
void SLSkybox::build()
{
    _hdrTexture->build(0);
    _environmentCubemap->build(0);
    _irradianceCubemap->build(2);
    _roughnessCubemap->build(3);
    _brdfLUTTexture->build(4);
    _isBuilt = true;
}
//-----------------------------------------------------------------------------
//! Draw the skybox with a cube map with the camera in its center.
void SLSkybox::drawAroundCamera(SLSceneView* sv)
{
    assert(sv && "No SceneView passed to SLSkybox::drawAroundCamera");

    SLGLState* stateGL = SLGLState::instance();

    if (_isHDR && !_isBuilt)
        build();

    // Set the view transform
    stateGL->modelViewMatrix.setMatrix(stateGL->viewMatrix);

    // Put skybox at the cameras position
    translation(sv->camera()->translationWS());

    // Apply world transform
    stateGL->modelViewMatrix.multiply(updateAndGetWM().m());

    // Freeze depth buffer
    stateGL->depthMask(false);

    // Change depth buffer comparisons for HDR SkyBoxes
    if (_isHDR)
        stateGL->depthFunc(GL_LEQUAL);

    // Draw the box
    drawMesh(sv);

    // Change back the depth buffer comparisons
    stateGL->depthFunc(GL_LESS);

    // Unlock depth buffer
    stateGL->depthMask(true);
}
//-----------------------------------------------------------------------------
//! Returns the color in the skybox at the the specified direction dir
SLCol4f SLSkybox::colorAtDir(const SLVec3f& dir)
{
    if (_mesh && !_mesh->mat()->textures().empty() &&
        _mesh->mat()->textures()[0]->images().size() == 6)
    {
        SLGLTexture* tex = _mesh->mat()->textures()[0];
        return tex->getTexelf(dir);
    }
    else
        return SLCol4f::BLACK; // Generated skybox texture do not exist in _image
}
//-----------------------------------------------------------------------------
