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
    SLBackground(SLstring shaderDir);
    SLBackground(SLGLProgram* textureOnlyProgram, SLGLProgram* colorAttributeProgram);
    ~SLBackground();

    void    render(SLint widthPX, SLint heightPX);
    void    renderInScene(const SLVec3f& LT, const SLVec3f& LB, const SLVec3f& RT, const SLVec3f& RB);
    SLCol4f colorAtPos(SLfloat x, SLfloat y, SLfloat width, SLfloat height);
    void    rebuild() { _vao.clearAttribs(); }

    // Setters
    void colors(const SLCol4f& uniformColor);
    void colors(const SLCol4f& topColor, const SLCol4f& bottomColor);
    void colors(const SLCol4f& topLeftColor,
                const SLCol4f& bottomLeftColor,
                const SLCol4f& topRightColor,
                const SLCol4f& bottomRightColor);
    //! If flag _repeatBlurred is true the texture is not distorted if its size does not fit to screen aspect ratio. Instead it is repeated
    void texture(SLGLTexture* backgroundTexture, bool repeatBlurred = false);

    // Getters
    SLVCol4f     colors() { return _colors; }
    SLCol4f      avgColor() { return _avgColor; }
    SLbool       isUniform() const { return _isUniform; }
    SLGLTexture* texture() { return _texture; }

private:
    //! Define background with two additional triangles left and right for bars containing a small texture subregions
    void defineWithBars();
    
    SLbool          _isUniform;    //!< Flag if background has uniform color
    SLVCol4f        _colors;       //!< Vector of 4 corner colors {TL,BL,TR,BR}
    SLCol4f         _avgColor;     //!< Average color of all 4 corner colors
    SLGLTexture*    _texture;      //!< Pointer to a background texture
    SLGLTexture*    _textureError; //!< Pointer to a error texture if background texture is not available
    SLint           _resX;         //!< Background resolution in x-dir.
    SLint           _resY;         //!< Background resolution in y-dir.
    SLGLVertexArray _vao;          //!< OpenGL Vertex Array Object for drawing

    SLGLProgram* _textureOnlyProgram    = nullptr;
    SLGLProgram* _colorAttributeProgram = nullptr;
    bool         _deletePrograms        = false;
    //!if flag is true the texture is not distorted if its size does not fit to screen aspect ratio. Instead bars are
    //!added left and right and filled with a small subregion of the texture, that then looks blurred
    bool _repeatBlurred = false;
};
//-----------------------------------------------------------------------------
#endif
