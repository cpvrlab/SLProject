//#############################################################################
//  File:      SLGLOculusFB.cpp
//  Purpose:   OpenGL Frame Buffer Object for the Oculus Rift Display  
//  Author:    Marc Wacker, Marcus Hudritsch
//  Date:      August 2014
//  Codestyle: https://github.com/cpvrlab/SLProject/wiki/Coding-Style-Guidelines
//  Copyright: Marcus Hudritsch
//             This software is provide under the GNU General Public License
//             Please visit: http://opensource.org/licenses/GPL-3.0
//#############################################################################

#include <stdafx.h>           // precompiled headers
#ifdef SL_MEMLEAKDETECT       // set in SL.h for debug config only
#include <debug_new.h>        // memory leak detector
#endif

#include <SLGLOculusFB.h>
#include <SLGLProgram.h>
#include <SLScene.h>

//-----------------------------------------------------------------------------
/*! Constructor initializing with default values
*/
SLGLOculusFB::SLGLOculusFB()
: _width(0),
_height(0),
_halfWidth(0),
_halfHeight(0),
_fbID(0),
_depthRbID(0),
_texID(0)
{ }
//-----------------------------------------------------------------------------
/*! Destructor calling dispose
*/
SLGLOculusFB::~SLGLOculusFB() 
{  
    dispose();
}
//-----------------------------------------------------------------------------
/*! Deletes the buffer object
*/
void SLGLOculusFB::dispose()
{  
    if (_fbID)        glDeleteFramebuffers(1, &_fbID);
    if (_texID)       glDeleteTextures(1, &_texID);
    if (_depthRbID)   glDeleteRenderbuffers(1, &_depthRbID);
}
//-----------------------------------------------------------------------------
/*! Activates the frame buffer. On the first time it calls the updateSize to
determine the size and then creates the FBO.
*/
void SLGLOculusFB::bindFramebuffer(SLint scrWidth, 
                                   SLint scrHeight)
{
    if (!_fbID) 
        updateSize(scrWidth, 
                   scrHeight);

    if ( _fbID) 
        glBindFramebuffer(GL_FRAMEBUFFER, _fbID);
}
//-----------------------------------------------------------------------------
/*! Frame Buffer generation. This is called from within updateSize because the
frame buffer size has to be calculated first
*/
void SLGLOculusFB::generateFBO()
{  
    // generate the intermediate screen texture
    glGenTextures(1, &_texID);
    glBindTexture(GL_TEXTURE_2D, _texID);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, _width, _height, 0, GL_RGBA, 
                 GL_UNSIGNED_BYTE, nullptr);
    glBindTexture(GL_TEXTURE_2D, 0);

    // generate the renderbuffer for the depth component
    glGenRenderbuffers(1, &_depthRbID);
    glBindRenderbuffer(GL_RENDERBUFFER, _depthRbID);
    #if defined(SL_OS_MACIOS) || defined(SL_OS_ANDROID)
    glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH_COMPONENT16, _width, _height);
    #else
    glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH_COMPONENT32, _width, _height);
    #endif
    glBindRenderbuffer(GL_RENDERBUFFER, 0);
   
    // finally generate the frame buffer and bind the targets
    glGenFramebuffers(1, &_fbID);
    glBindFramebuffer(GL_FRAMEBUFFER, _fbID);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, 
                           GL_TEXTURE_2D, _texID, 0);
    glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, 
                              GL_RENDERBUFFER, _depthRbID);

    // test if the generated fbo is valid
    if ((glCheckFramebufferStatus(GL_FRAMEBUFFER)) != GL_FRAMEBUFFER_COMPLETE) 
        SL_EXIT_MSG("Frame buffer creation failed!");

    glBindFramebuffer(GL_FRAMEBUFFER, 0);
    GET_GL_ERROR;
}
//-----------------------------------------------------------------------------      
/*! Updates everything when the screen gets resized: 
- Recalculates the stereo parameters
- Creates or updates the FBO
- Updates the shader uniforms 
*/
void SLGLOculusFB::updateSize(SLint scrWidth, 
                              SLint scrHeight)
{  
    _width      = scrWidth;
    _height     = scrHeight;
    _halfWidth  = scrWidth >> 1;
    _halfHeight = scrHeight >> 1;

    // Create FBO or resize it
    if (!_fbID) 
        generateFBO();
    else
    {  // Resize the intermediate render targets
        glBindTexture(GL_TEXTURE_2D, _texID);
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, _width, _height, 0, GL_RGBA, GL_UNSIGNED_BYTE, nullptr);
        glBindTexture(GL_TEXTURE_2D, 0);

        // Resize the depth render buffer
        glBindRenderbuffer(GL_RENDERBUFFER, _depthRbID);
        
        #if defined(SL_OS_MACIOS) || defined(SL_OS_ANDROID)
        glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH_COMPONENT16, _width, _height);
        #else
        glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH_COMPONENT32, _width, _height);
        #endif
        
        glBindRenderbuffer(GL_RENDERBUFFER, 0);
    }

    GET_GL_ERROR;
}
//-----------------------------------------------------------------------------
/*! Draws the intermediate render target (the texture) into the real
 * framebuffer.
 */
void SLGLOculusFB::drawFramebuffer()
{
    glViewport(0, 0, _width, _height);
    glBindFramebuffer(GL_FRAMEBUFFER, 0);
    glClear(GL_COLOR_BUFFER_BIT);
    glDisable(GL_DEPTH_TEST);
   
    // Create VBO for screen quad once
    if (!_vbo.id())
    {   SLfloat quadVerts[] = {-1,-1,  1,-1,  -1,1,  1,1};
        _vbo.generate(quadVerts, 4, 2); 
    }

    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, _texID);

    //bind the rift shader
    SLGLProgram* sp = SLScene::current->programs(StereoOculus);
    sp->useProgram();

    _vbo.bindAndEnableAttrib(0);
    _vbo.drawArrayAs(SL_TRIANGLE_STRIP);
    _vbo.disableAttribArray();

    glEnable(GL_DEPTH_TEST);
}
//-----------------------------------------------------------------------------
