//#############################################################################
//  File:      SLVRCompositor.cpp
//  Author:    Marino von Wattenwyl
//  Date:      August 2021
//  Codestyle: https://github.com/cpvrlab/SLProject/wiki/SLProject-Coding-Style
//  Copyright: Marcus Hudritsch
//             This software is provide under the GNU General Public License
//             Please visit: http://opensource.org/licenses/GPL-3.0
//#############################################################################

#include <vr/SLVRCompositor.h>
#include <vr/SLVRSystem.h>
#include <vr/SLVR.h>

#include <SLGLTexture.h>

//-----------------------------------------------------------------------------
SLVRCompositor::SLVRCompositor()
{
}
//-----------------------------------------------------------------------------
SLVRCompositor::~SLVRCompositor()
{
    glDeleteRenderbuffers(1, &_leftDepthRenderBuffer);
    glDeleteRenderbuffers(1, &_rightDepthRenderBuffer);

    glDeleteTextures(1, &_leftTexture);
    glDeleteTextures(1, &_rightTexture);

    glDeleteFramebuffers(1, &_leftFBO);
    glDeleteFramebuffers(1, &_rightFBO);
}
//-----------------------------------------------------------------------------
/*! Prepares the SLVRCompositor for binding framebuffers and submitting textures
 * The method first gets the recommended framebuffer texture size from OpenVR
 * and then creates textures, depth renderbuffers and framebuffers for both eyes
 */
void SLVRCompositor::startup()
{
    vr::IVRSystem* system = SLVRSystem::instance().system();

    uint32_t width;
    uint32_t height;
    system->GetRecommendedRenderTargetSize(&width, &height);
    VR_LOG("Recommended frame buffer size: " << width << "x" << height)

    _frameBufferWidth  = (SLsizei)width;
    _frameBufferHeight = (SLsizei)height;

    initFBO(&_leftFBO, &_leftTexture, &_leftDepthRenderBuffer);
    initFBO(&_rightFBO, &_rightTexture, &_rightDepthRenderBuffer);
}
//-----------------------------------------------------------------------------
void SLVRCompositor::initFBO(unsigned int* fbo,
                             unsigned int* texture,
                             unsigned int* depthRenderBuffer) const
{
    glGenTextures(1, texture);
    glBindTexture(GL_TEXTURE_2D, *texture);
    glTexImage2D(GL_TEXTURE_2D,
                 0,
                 GL_RGBA8,
                 _frameBufferWidth,
                 _frameBufferHeight,
                 0,
                 GL_RGBA,
                 GL_UNSIGNED_BYTE,
                 nullptr);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glBindTexture(GL_TEXTURE_2D, 0);

    glGenRenderbuffers(1, depthRenderBuffer);
    glBindRenderbuffer(GL_RENDERBUFFER, *depthRenderBuffer);
    glRenderbufferStorage(GL_RENDERBUFFER,
                          GL_DEPTH_COMPONENT24,
                          _frameBufferWidth,
                          _frameBufferHeight);
    glBindRenderbuffer(GL_RENDERBUFFER, 0);

    glGenFramebuffers(1, fbo);
    glBindFramebuffer(GL_FRAMEBUFFER, *fbo);
    glFramebufferTexture2D(GL_FRAMEBUFFER,
                           GL_COLOR_ATTACHMENT0,
                           GL_TEXTURE_2D,
                           *texture,
                           0);
    glFramebufferRenderbuffer(GL_FRAMEBUFFER,
                              GL_DEPTH_ATTACHMENT,
                              GL_RENDERBUFFER,
                              *depthRenderBuffer);
    glBindFramebuffer(GL_FRAMEBUFFER, 0);
}
//-----------------------------------------------------------------------------
/*! Binds the framebuffer for the left eye
 * This method must be called before rendering the left eye image
 */
void SLVRCompositor::prepareLeftEye() const
{
    glBindFramebuffer(GL_FRAMEBUFFER, _leftFBO);
}
//-----------------------------------------------------------------------------
/*! Binds the framebuffer for the right eye
 * This method must be called before rendering the right eye image
 */
void SLVRCompositor::prepareRightEye() const
{
    glBindFramebuffer(GL_FRAMEBUFFER, _rightFBO);
}
//-----------------------------------------------------------------------------
/*! Unbinds the currently bound framebuffer
 * This methods must be called after rendering either eye image
 * if you wish to continue rendering to the default framebuffer
 */
void SLVRCompositor::finishEye()
{
    glBindFramebuffer(GL_FRAMEBUFFER, 0);
}
//-----------------------------------------------------------------------------
/*! Submits both framebuffer textures to the OpenVR compositor
 * OpenVR then applies the distortion to the textures and presents them in the HMD
 */
void SLVRCompositor::submit() const
{
    vr::Texture_t leftVRTexture = {(void*)(uintptr_t)_leftTexture,
                                   vr::ETextureType::TextureType_OpenGL,
                                   vr::EColorSpace::ColorSpace_Gamma};
    vr::VRCompositor()->Submit(vr::EVREye::Eye_Left, &leftVRTexture);

    vr::Texture_t rightVRTexture = {(void*)(uintptr_t)_rightTexture,
                                    vr::ETextureType::TextureType_OpenGL,
                                    vr::EColorSpace::ColorSpace_Gamma};
    vr::VRCompositor()->Submit(vr::EVREye::Eye_Right, &rightVRTexture);
}
//-----------------------------------------------------------------------------
/*! Fades to the color specified in the amount of time specified
 * @param seconds The number of seconds it takes the display to fade
 * @param color The color the display fades to
 */
void SLVRCompositor::fade(float seconds, const SLCol4f& color)
{
    vr::VRCompositor()->FadeToColor(seconds, color.x, color.y, color.z, color.w);
}
//-----------------------------------------------------------------------------