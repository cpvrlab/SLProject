//#############################################################################
//  File:      SLGLOculusFB.h
//  Purpose:   OpenGL Frame Buffer Object for the Oculus Rift Display
//  Date:      August 2014
//  Codestyle: https://github.com/cpvrlab/SLProject/wiki/SLProject-Coding-Style
//  Authors:   Marc Wacker, Marcus Hudritsch
//  License:   This software is provided under the GNU General Public License
//             Please visit: http://opensource.org/licenses/GPL-3.0
//#############################################################################

#ifndef SLGLOCULUSFB_H
#define SLGLOCULUSFB_H

#include <SLCamera.h>
#include <SLGLVertexArray.h>

class SLGLProgram;
//-----------------------------------------------------------------------------
//! Wrapper around Oculus Rift and its Frame Buffer Objects
/*! This class is deprecated since the lib_ovr from Oculus completely changed
The lib_ovr that connects the Oculus hardware was removed since it only worked
for devkit2 under windows.
*/
class SLGLOculusFB
{
public:
    SLGLOculusFB();
    ~SLGLOculusFB();

    void bindFramebuffer(SLint scrWidth,
                         SLint scrHeight);
    void drawFramebuffer(SLGLProgram* stereoOculusProgram);
    void updateSize(SLint scrWidth,
                    SLint scrHeight);

    // Getters
    SLint  width() { return _width; }
    SLint  halfWidth() { return _halfWidth; }
    SLint  height() { return _height; }
    SLuint texID() { return _texID; }

private:
    void dispose();
    void generateFBO();

    SLint           _width;      //! Width of the buffer
    SLint           _height;     //! Height of the buffer
    SLint           _halfWidth;  //! Half the width of the buffer
    SLint           _halfHeight; //! Half the height of the buffer
    SLuint          _fbID;       //! OpenGL id of frame buffer
    SLuint          _depthRbID;  //! OpenGL id of depth render buffer
    SLuint          _texID;      //! OpenGL id of
    SLGLVertexArray _vao;        //! Vertex array object for the rendering
};
//-----------------------------------------------------------------------------

#endif
