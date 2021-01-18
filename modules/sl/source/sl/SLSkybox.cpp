//#############################################################################
//  File:      SLSkybox
//  Authors:    Marcus Hudritsch
//  Date:      December 2017
//  Codestyle: https://github.com/cpvrlab/SLProject/wiki/SLProject-Coding-Style
//  Copyright: Marcus Hudritsch
//             This software is provide under the GNU General Public License
//             Please visit: http://opensource.org/licenses/GPL-3.0
//#############################################################################

#include <SLApplication.h>
#include <SLBox.h>
#include <SLGLProgramGeneric.h>
#include <SLGLFrameBuffer.h>
#include <SLGLTexture.h>
#include <SLGLTextureGenerated.h>
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
                   SLstring        shaderFilePath,
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
    _isHDR = false;

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
                                             shaderFilePath + "SkyBox.vert",
                                             shaderFilePath + "SkyBox.frag");
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
                   SLstring        hdrImage,
                   SLVec2i         resolution,
                   SLstring        name,
                   SLGLUniform1f*  exposureUniform) : SLNode(name)
{
    // Set HDR flag to true, this is a HDR SkyBox
    _isHDR = true;

    // Create shader program for the background
    SLGLProgram* backgroundShader = new SLGLProgramGeneric(projectScene,
                                                           SLApplication::shaderPath + "PBR_SkyboxHDR.vert",
                                                           SLApplication::shaderPath + "PBR_SkyboxHDR.frag");

    // if an exposure uniform is passed the initialize this exposure with it otherwise it is constant at 1.0
    SLGLUniform1f* exposure = exposureUniform ? exposureUniform : new SLGLUniform1f(UT_const, "u_exposure", 1.0f);
    projectScene->eventHandlers().push_back(exposure);
    backgroundShader->addUniform1f(exposure);

    // Create frame buffer for capturing the scene into a cube map
    SLGLFrameBuffer* captureBuffer = new SLGLFrameBuffer(true,
                                                         resolution.x,
                                                         resolution.y);
    captureBuffer->generate();

    // Create texture from the HDR Image
    SLGLTexture* hdrTexture = new SLGLTexture(projectScene,
                                              hdrImage,
                                              GL_LINEAR,
                                              GL_LINEAR,
                                              TT_hdr,
                                              GL_CLAMP_TO_EDGE,
                                              GL_CLAMP_TO_EDGE);

    // Generate cube map using the HDR texture
    SLGLTexture* envCubemap = new SLGLTextureGenerated(projectScene,
                                                       hdrTexture,
                                                       captureBuffer,
                                                       TT_environment,
                                                       GL_TEXTURE_CUBE_MAP,
                                                       GL_LINEAR_MIPMAP_LINEAR);

    // The buffer must be reduced, because we need to lower the resolution for the irradiance and prefilter map
    captureBuffer->bufferStorage(32, 32);
    SLGLTexture* irradiancemap  = new SLGLTextureGenerated(projectScene,
                                                          envCubemap,
                                                          captureBuffer,
                                                          TT_irradiance);
    SLGLTexture* prefilter      = new SLGLTextureGenerated(projectScene,
                                                      envCubemap,
                                                      captureBuffer,
                                                      TT_prefilter);
    SLGLTexture* brdfLUTTexture = new SLGLTextureGenerated(projectScene,
                                                           nullptr,
                                                           captureBuffer,
                                                           TT_lut,
                                                           GL_TEXTURE_2D);

    // Create the material of the sky box and store there the other texture to be used for other materials
    SLMaterial* hdrMaterial = new SLMaterial(projectScene, "matCubeMap");
    hdrMaterial->textures().push_back(envCubemap);
    hdrMaterial->textures().push_back(irradiancemap);
    hdrMaterial->textures().push_back(prefilter);
    hdrMaterial->textures().push_back(brdfLUTTexture);
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

    // We don't need the frame buffer any longer
    captureBuffer->unbind();
    delete captureBuffer;
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
