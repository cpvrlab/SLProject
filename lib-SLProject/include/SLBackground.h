//#############################################################################
//  File:      SLBackground.h
//  Author:    Marcus Hudritsch
//  Date:      August 2015
//  Codestyle: https://github.com/cpvrlab/SLProject/wiki/SLProject-Coding-Style
//  Copyright: Marcus Hudritsch
//             This software is provide under the GNU General Public License
//             Please visit: http://opensource.org/licenses/GPL-3.0
//#############################################################################

#ifndef SLBACKGROUND_H
#define SLBACKGROUND_H

#include <SLGLVertexArray.h>
#include <SLObject.h>

class SLGLTexture;
class SLGLProgram;
//-----------------------------------------------------------------------------
//! Defines a 2D-Background for the OpenGL framebuffer background.
/*! The background can either be defined with a texture or with 4 colors for
the corners of the frame buffer. For a uniform background color the color at
index 0 of the _colors vector is taken. Every instance of SLCamera has a
background that is displayed on the far clipping plane if the camera is not the
active one. The OpenGL rendering is done in SLSceneView::draw3DGL for the active
camera or in SLCamera::drawMeshes for inactive ones.
*/
class SLBackground : public SLObject
{
public:
    //SLBackground();
    SLBackground(SLGLProgram* textureOnlyProgram, SLGLProgram* colorAttributeProgram);
    ~SLBackground();

    void    render(SLint widthPX, SLint heightPX);
    void    renderInScene(SLVec3f LT, SLVec3f LB, SLVec3f RT, SLVec3f RB);
    SLCol4f colorAtPos(SLfloat x, SLfloat y);
    void    rebuild() { _vao.clearAttribs(); }

    // Setters
    void colors(const SLCol4f& uniformColor);
    void colors(const SLCol4f& topColor, const SLCol4f& bottomColor);
    void colors(const SLCol4f& topLeftColor,
                const SLCol4f& bottomLeftColor,
                const SLCol4f& topRightColor,
                const SLCol4f& bottomRightColor);
    void texture(SLGLTexture* backgroundTexture);

    // Getters
    SLVCol4f     colors() { return _colors; }
    SLbool       isUniform() { return _isUniform; }
    SLGLTexture* texture() { return _texture; }

private:
    SLbool          _isUniform;    //!< Flag if background has uniform color
    SLVCol4f        _colors;       //!< Vector of 4 corner colors {TL,BL,TR,BR}
    SLGLTexture*    _texture;      //!< Pointer to a background texture
    SLGLTexture*    _textureError; //!< Pointer to a error texture if background texture is not available
    SLint           _resX;         //!< Background resolution in x-dir.
    SLint           _resY;         //!< Background resolution in y-dir.
    SLGLVertexArray _vao;          //!< OpenGL Vertex Array Object for drawing

    SLGLProgram* _textureOnlyProgram    = nullptr;
    SLGLProgram* _colorAttributeProgram = nullptr;
    bool         _deletePrograms        = false;
};
//-----------------------------------------------------------------------------
#endif
