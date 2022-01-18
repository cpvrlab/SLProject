//#############################################################################
//  File:      SLVRCompositor.h
//  Author:    Marino von Wattenwyl
//  Date:      August 2021
//  Codestyle: https://github.com/cpvrlab/SLProject/wiki/SLProject-Coding-Style
//  Copyright: Marcus Hudritsch
//             This software is provide under the GNU General Public License
//             Please visit: http://opensource.org/licenses/GPL-3.0
//#############################################################################

#ifndef SLPROJECT_SLVRCOMPOSITOR_H
#define SLPROJECT_SLVRCOMPOSITOR_H

#include <SL.h>
#include <SLGLFrameBuffer.h>
#include <SLGLTexture.h>

//-----------------------------------------------------------------------------
//! SLVRCompositor is used for submitting frames to the HMD
/*! This class is responsible for handling the framebuffers the application renders to
 * and for submitting the textures of these framebuffers to the HMD
 */
class SLVRCompositor
{
    friend class SLVRSystem;

public:
    void prepareLeftEye();
    void prepareRightEye();
    void finishEye();
    void submit();
    void fade(float seconds, const SLCol4f& color);

    // Getters
    SLsizei frameBufferWidth() const { return _frameBufferWidth; }
    SLsizei frameBufferHeight() const { return _frameBufferHeight; }

private:
    SLVRCompositor();
    ~SLVRCompositor();

    void startup();
    void initFBO(unsigned int* fbo,
                 unsigned int* texture,
                 unsigned int* depthRenderBuffer);

    SLsizei _frameBufferWidth;
    SLsizei _frameBufferHeight;

    unsigned int _leftFBO;
    unsigned int _leftTexture;
    unsigned int _leftDepthRenderBuffer;

    unsigned int _rightFBO;
    unsigned int _rightTexture;
    unsigned int _rightDepthRenderBuffer;
};
//-----------------------------------------------------------------------------
#endif // SLPROJECT_SLVRCOMPOSITOR_H
