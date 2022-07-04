//#############################################################################
//  File:      SLGLConetracer.cpp
//  Date:      September 2018
//  Authors:   Stefan Thoeni
//  License:   This software is provided under the GNU General Public License
//             Please visit: http://opensource.org/licenses/GPL-3.0
//#############################################################################

#include <SLGLState.h>
#include <SLGLConetracer.h>
#include <SLGLConetracerTex3D.h>
#include <SLGLProgram.h>
#include <SLGLProgramGeneric.h>
#include <SLSceneView.h>

//-----------------------------------------------------------------------------
SLGLConetracer::SLGLConetracer(SLstring shaderFilePath)
  : _shaderFilePath(shaderFilePath)
{
    // SL_LOG("Constructor     : SLGLConetracer");
}
//-----------------------------------------------------------------------------
SLGLConetracer::~SLGLConetracer()
{
    // SL_LOG("Destructor      : ~SLGLConetracer");
    delete _voxelizeMat;
    delete _worldMat;
    delete _visualizeMat;
    delete _conetraceMat;
    delete _quadMesh;
    delete _cubeMesh;

    for (SLGLProgram* p : _programs)
        delete p;
}
//-----------------------------------------------------------------------------
void SLGLConetracer::init(SLint          scrW,
                          SLint          scrH,
                          const SLVec3f& minWs,
                          const SLVec3f& maxWs)
{
    // enable multisampling
#ifndef SL_GLES3
    glEnable(GL_MULTISAMPLE);
#endif

    // Initialize voxel 3D-Texture:
    const SLVfloat texture3D(4 * _voxelTexSize * _voxelTexSize * _voxelTexSize, 0.0f);
    _voxelTex = new SLGLConetracerTex3D(texture3D,
                                        _voxelTexSize,
                                        _voxelTexSize,
                                        _voxelTexSize,
                                        true);
    GET_GL_ERROR;

    // Initialize voxelization:
    SLGLProgram* voxelizeShader = new SLGLProgramGeneric(nullptr,
                                                         _shaderFilePath + "CTVoxelization.vert",
                                                         _shaderFilePath + "CTVoxelization.frag",
                                                         _shaderFilePath + "CTVoxelization.geom");
    voxelizeShader->initRaw();
    _programs.push_back(voxelizeShader);
    _voxelizeMat = new SLMaterial(nullptr, "Voxelization-Material", voxelizeShader);
    GET_GL_ERROR;

    // initialize voxel visualization:
    SLGLProgram* worldPosProg = new SLGLProgramGeneric(nullptr,
                                                       _shaderFilePath + "CTWorldpos.vert",
                                                       _shaderFilePath + "CTWorldpos.frag");
    worldPosProg->initRaw();
    _programs.push_back(worldPosProg);
    _worldMat = new SLMaterial(nullptr, "World-Material", worldPosProg);

    // initialize voxel visualization:
    SLGLProgram* visualizeShader = new SLGLProgramGeneric(nullptr,
                                                          _shaderFilePath + "CTVisualize.vert",
                                                          _shaderFilePath + "CTVisualize.frag");
    visualizeShader->initRaw();
    _programs.push_back(visualizeShader);
    _visualizeMat = new SLMaterial(nullptr, "World-Material", visualizeShader);

    // initialize voxel conetracing material:
    SLGLProgram* ctShader = new SLGLProgramGeneric(nullptr,
                                                   _shaderFilePath + "CT.vert",
                                                   _shaderFilePath + "CT.frag");
    ctShader->initRaw();
    _programs.push_back(ctShader);
    _conetraceMat = new SLMaterial(nullptr, "Conetrace Material", ctShader);

    // FBOs.
    // read current viewport
    GLint m_viewport[4];
    glGetIntegerv(GL_VIEWPORT, m_viewport);

    _visualizeBackfaceFBO  = new SLGLFbo(scrW, scrH);
    _visualizeFrontfaceFBO = new SLGLFbo(scrW, scrH);

    // SLGLConetracer manages the livetime of the following meshes
    _quadMesh = new SLRectangle(nullptr, SLVec2f(-1, -1), SLVec2f(1, 1), 1, 1);
    _cubeMesh = new SLBox(nullptr, -1, -1, -1);

    // The world's bounding box should not change during runtime.
    calcWS2VoxelSpaceTransform(minWs, maxWs);
}
//-----------------------------------------------------------------------------
void SLGLConetracer::visualizeVoxels()
{
    // store viewport
    GLint m_viewport[4];
    glGetIntegerv(GL_VIEWPORT, m_viewport);

    glViewport(0, 0, _voxelTexSize, _voxelTexSize);

    glUseProgram(_worldMat->program()->progID());
    GET_GL_ERROR;
    // Settings.
    SLGLState* stateGL = SLGLState::instance();
    stateGL->clearColor(_sv->camera()->background().colors()[0]);

    glEnable(GL_CULL_FACE);
    glEnable(GL_DEPTH_TEST);
    GET_GL_ERROR;

    // Back faces
    glCullFace(GL_FRONT);
    glBindFramebuffer(GL_FRAMEBUFFER, _visualizeBackfaceFBO->fboID);
    glViewport(0, 0, _visualizeBackfaceFBO->width, _visualizeBackfaceFBO->height);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    renderNode(new SLNode(_cubeMesh), _worldMat->program());
    GET_GL_ERROR;

    // Front faces
    glCullFace(GL_BACK);
    glBindFramebuffer(GL_FRAMEBUFFER, _visualizeFrontfaceFBO->fboID);
    glViewport(0, 0, _visualizeFrontfaceFBO->width, _visualizeFrontfaceFBO->height);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    renderNode(new SLNode(_cubeMesh), _worldMat->program());
    GET_GL_ERROR;

    // Render 3D Texture to screen
    glUseProgram(_visualizeMat->program()->progID());
    glBindRenderbuffer(GL_RENDERBUFFER, 0);
    glBindFramebuffer(GL_FRAMEBUFFER, 0);
    GET_GL_ERROR;

    _visualizeBackfaceFBO->activateAsTexture(_visualizeMat->program()->progID(),
                                             "textureBack",
                                             0);
    _visualizeFrontfaceFBO->activateAsTexture(_visualizeMat->program()->progID(),
                                              "textureFront",
                                              1);
    _voxelTex->activate(_visualizeMat->program()->progID(), "u_texture3D", 2);
    GET_GL_ERROR;

    glViewport(m_viewport[0], m_viewport[1], m_viewport[2], m_viewport[3]);

    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    GLint   loc = glGetUniformLocation(_visualizeMat->program()->progID(),
                                     "cameraPosition");
    SLVec3f pos = _sv->camera()->translationWS();
    glUniform3fv(loc, 1, (float*)&pos);

    renderNode(new SLNode(_quadMesh),
               _visualizeMat->program());
}
//-----------------------------------------------------------------------------
SLbool SLGLConetracer::render(SLSceneView* sv)
{
    _sv = sv;

    voxelize();

    if (_showVoxels)
        visualizeVoxels();
    else
    {
        glClearColor(0.0f, 0.0f, 0.0f, 1.0);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
        glEnable(GL_DEPTH_TEST);
        glEnable(GL_CULL_FACE);
        glCullFace(GL_BACK);
        glColorMask(GL_TRUE, GL_TRUE, GL_TRUE, GL_TRUE);
        glEnable(GL_BLEND);
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
        GET_GL_ERROR;

        glUseProgram(_conetraceMat->program()->progID());

        renderSceneGraph(_conetraceMat->program());
    }
    GET_GL_ERROR;
    return true;
}
//-----------------------------------------------------------------------------
/*
void SLGLConetracer::uploadLights(SLGLProgram* program)
{
    SLGLState* stateGL = SLGLState::instance();
    SLuint     progID  = program->progID();

    // no ambient color needed. :-)

    glUniform1i(glGetUniformLocation(progID, "u_numLightsUsed"), stateGL->numLightsUsed);

    SLVec4f lightsVoxelSpace[SL_MAX_LIGHTS];

    for (int i = 0; i < stateGL->numLightsUsed; i++)
        lightsVoxelSpace[i] = _wsToVoxelSpace->multVec(stateGL->lightPosWS[i]);

    if (stateGL->numLightsUsed > 0)
    {
        SLint nL = SL_MAX_LIGHTS;
        glUniform1iv(glGetUniformLocation(progID, "u_lightIsOn"), nL, (SLint*)stateGL->lightIsOn);
        glUniform4fv(glGetUniformLocation(progID, "u_lightPosVS"), nL, (SLfloat*)lightsVoxelSpace);
        glUniform4fv(glGetUniformLocation(progID, "u_lightPosWS"), nL, (SLfloat*)stateGL->lightPosWS);
        glUniform4fv(glGetUniformLocation(progID, "u_lightDiff"), nL, (SLfloat*)stateGL->lightDiffuse);
        glUniform4fv(glGetUniformLocation(progID, "u_lightSpec"), nL, (SLfloat*)stateGL->lightSpecular);
        glUniform3fv(glGetUniformLocation(progID, "u_lightSpotDirWS"), nL, (SLfloat*)stateGL->lightSpotDirWS);
        glUniform1fv(glGetUniformLocation(progID, "u_lightSpotDeg"), nL, (SLfloat*)stateGL->lightSpotCutoff);
        glUniform1fv(glGetUniformLocation(progID, "u_lightSpotCos"), nL, (SLfloat*)stateGL->lightSpotCosCut);
        glUniform1fv(glGetUniformLocation(progID, "u_lightSpotExp"), nL, (SLfloat*)stateGL->lightSpotExp);
        glUniform3fv(glGetUniformLocation(progID, "u_lightAtt"), nL, (SLfloat*)stateGL->lightAtt);
        glUniform1iv(glGetUniformLocation(progID, "u_lightDoAtt"), nL, (SLint*)stateGL->lightDoAtt);
    }
}
*/
//-----------------------------------------------------------------------------
void SLGLConetracer::uploadRenderSettings(SLGLProgram* program)
{
    SLuint progID = program->progID();
    glUniform1f(glGetUniformLocation(progID, "s_diffuseConeAngle"), _diffuseConeAngle);
    glUniform1f(glGetUniformLocation(progID, "s_specularConeAngle"), _specularConeAngle);
    glUniform1f(glGetUniformLocation(progID, "s_shadowConeAngle"), _shadowConeAngle);
    glUniform1i(glGetUniformLocation(progID, "s_directEnabled"), _doDirectIllum);
    glUniform1i(glGetUniformLocation(progID, "s_diffuseEnabled"), _doDiffuseIllum);
    glUniform1i(glGetUniformLocation(progID, "s_specularEnabled"), _doSpecularIllum);
    glUniform1i(glGetUniformLocation(progID, "s_shadowsEnabled"), _doShadows);
    glUniform1f(glGetUniformLocation(progID, "s_lightMeshSize"), _lightMeshSize);
    glUniform1f(glGetUniformLocation(progID, "u_oneOverGamma"), oneOverGamma());
}
//-----------------------------------------------------------------------------
void SLGLConetracer::voxelSpaceTransform(const SLfloat l,
                                         const SLfloat r,
                                         const SLfloat b,
                                         const SLfloat t,
                                         const SLfloat n,
                                         const SLfloat f)
{
    // clang-format off
    _wsToVoxelSpace->setMatrix(1/(r-l),      0,      0,-l/(r-l),
                                     0,1/(t-b),      0,-b/(t-b),
                                     0,      0,1/(f-n),-n/(f-n),
                                     0,      0,      0,      1);
    // clang-format on
}
//-----------------------------------------------------------------------------
void SLGLConetracer::calcWS2VoxelSpaceTransform(const SLVec3f& minWs,
                                                const SLVec3f& maxWs)
{
    // upload ws to vs settings:
    // figure out biggest component:
    SLVec3f p1 = maxWs - minWs;

    SLfloat maxComp = p1.comp[p1.maxComp()];

    voxelSpaceTransform(minWs.x,
                        minWs.x + maxComp,
                        minWs.y,
                        minWs.y + maxComp,
                        minWs.z,
                        minWs.z + maxComp);
}
//-----------------------------------------------------------------------------
//! Renders scene using a given Program
void SLGLConetracer::renderSceneGraph(SLGLProgram* program)
{
    glViewport(0, 0, _sv->viewportW(), _sv->viewportH());

    SLuint progID = program->progID();
    GLint  loc    = glGetUniformLocation(progID, "u_wsToVs");
    glUniformMatrix4fv(loc, 1, GL_FALSE, (SLfloat*)_wsToVoxelSpace->m());

    uploadRenderSettings(program);

    // uploadLights(program);
    program->passLightsToUniforms(&_sv->s()->lights(), 0);

    // upload camera position:
    SLVec3f camPosWS = _sv->camera()->translationWS();
    SLVec3f camPos   = _wsToVoxelSpace->multVec(camPosWS);
    glUniform3fv(glGetUniformLocation(progID, "u_EyePos"), 1, (SLfloat*)&camPos);
    glUniform3fv(glGetUniformLocation(progID, "u_EyePosWS"), 1, (SLfloat*)&camPosWS);
    GET_GL_ERROR;

    renderNode(_sv->s()->root3D(), program);
}
//-----------------------------------------------------------------------------
void SLGLConetracer::renderNode(SLNode* node, SLGLProgram* program)
{
    assert(node);

    SLGLState* stateGL = SLGLState::instance();

    stateGL->modelMatrix = node->updateAndGetWM();

    // pass the modelview projection matrix to the shader
    GLint locp = glGetUniformLocation(program->progID(), "u_pMatrix");
    GLint locv = glGetUniformLocation(program->progID(), "u_vMatrix");
    GLint locm = glGetUniformLocation(program->progID(), "u_mMatrix");
    glUniformMatrix4fv(locm, 1, GL_FALSE, (SLfloat*)&stateGL->modelMatrix);
    glUniformMatrix4fv(locv, 1, GL_FALSE, (SLfloat*)&stateGL->viewMatrix);
    glUniformMatrix4fv(locp, 1, GL_FALSE, (SLfloat*)&stateGL->projectionMatrix);

    // draw meshes of the node
    if (node->mesh())
    {
        SLMesh*     mesh = node->mesh();
        SLMaterial* mat  = mesh->mat();
        mat->passToUniforms(program, 0);

        // generate a VAO if it does not exist yet
        if (!mesh->vao().vaoID())
            mesh->generateVAO(mesh->vao());

        // bind the buffer
        glBindVertexArray(mesh->vao().vaoID());

        glDrawElements(GL_TRIANGLES,
                       mesh->vao().numIndicesElements(),
                       GL_UNSIGNED_SHORT,
                       nullptr);
    }
    // recursively draw the child nodes
    for (auto* child : node->children())
        renderNode(child, program);

    GET_GL_ERROR;
}
//-----------------------------------------------------------------------------
void SLGLConetracer::voxelize()
{
    _voxelTex->clear(SLVec4f(0.0f, 0.0f, 0.0f, 0.0f));
    SLGLProgram* program = _voxelizeMat->program();

    // store viewport
    GLint m_viewport[4];
    glGetIntegerv(GL_VIEWPORT, m_viewport);

    glUseProgram(program->progID());
    glBindFramebuffer(GL_FRAMEBUFFER, 0);

    glViewport(0, 0, _voxelTexSize, _voxelTexSize);

    glColorMask(GL_FALSE, GL_FALSE, GL_FALSE, GL_FALSE);
    glDisable(GL_CULL_FACE);
    glDisable(GL_DEPTH_TEST);
    glDisable(GL_BLEND);

    _voxelTex->activate(program->progID(), "u_texture3D", 0);

    // Bind texture where we want to write to:
#if defined(GL_VERSION_4_4)
    glBindImageTexture(0,
                       _voxelTex->textureID,
                       0,
                       GL_TRUE,
                       0,
                       GL_WRITE_ONLY,
                       GL_RGBA8);
#endif

    _sv->camera()->setProjection(_sv, ET_center);
    _sv->camera()->setView(_sv, ET_center);

    renderSceneGraph(program);

    // restore viewport:
    glViewport(m_viewport[0], m_viewport[1], m_viewport[2], m_viewport[3]);

    glGenerateMipmap(GL_TEXTURE_3D);

    // reset color mask
    glColorMask(GL_TRUE, GL_TRUE, GL_TRUE, GL_TRUE);
    GET_GL_ERROR;
}
//-----------------------------------------------------------------------------
