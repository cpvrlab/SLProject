//#############################################################################
//  File:      SLBackground.cpp
//  Author:    Marcus Hudritsch
//  Date:      August 2015
//  Codestyle: https://github.com/cpvrlab/SLProject/wiki/Coding-Style-Guidelines
//  Copyright: Marcus Hudritsch
//             This software is provide under the GNU General Public License
//             Please visit: http://opensource.org/licenses/GPL-3.0
//#############################################################################

#include <stdafx.h>           // precompiled headers
#ifdef SL_MEMLEAKDETECT       // set in SL.h for debug config only
#include <debug_new.h>        // memory leak detector
#endif

#include <SLBackground.h>
#include <SLGLTexture.h>
#include <SLGLProgram.h>
#include <SLScene.h>

//-----------------------------------------------------------------------------
//! The constructor initializes to a uniform black background color
SLBackground::SLBackground() : SLObject("Background")
{
    _colors.push_back(SLCol4f::BLACK); // bottom left
    _colors.push_back(SLCol4f::BLACK); // bottom right
    _colors.push_back(SLCol4f::BLACK); // top right
    _colors.push_back(SLCol4f::BLACK); // top left
    _isUniform  = true;
    _texture = nullptr;
    _updateTex = false;
    _resX = -1;
    _resY = -1;
}
//-----------------------------------------------------------------------------
SLBackground::~SLBackground()
{
    _bufP.dispose();
    _bufC.dispose();
    _bufT.dispose();
    _bufI.dispose();
}
//-----------------------------------------------------------------------------
//! Sets a uniform background color
void SLBackground::colors(SLCol4f uniformColor)
{
    _colors[0].set(uniformColor);
    _colors[1].set(uniformColor);
    _colors[2].set(uniformColor);
    _colors[3].set(uniformColor);
    _isUniform = true;
    _texture = nullptr;
    _bufC.dispose();
}
//-----------------------------------------------------------------------------
//! Sets a gradient top-down background color
void SLBackground::colors(SLCol4f topColor, SLCol4f bottomColor)
{
    _colors[0].set(topColor);
    _colors[1].set(bottomColor);
    _colors[2].set(topColor);
    _colors[3].set(bottomColor);
    _isUniform = false;
    _texture = nullptr;
    _bufC.dispose();
}
//-----------------------------------------------------------------------------
//! Sets a gradient background color with a color per corner
void SLBackground::colors(SLCol4f topLeftColor,  SLCol4f bottomLeftColor,
                          SLCol4f topRightColor, SLCol4f bottomRightColor)
{
    _colors[0].set(topLeftColor);
    _colors[1].set(bottomLeftColor);
    _colors[2].set(topRightColor);
    _colors[3].set(bottomRightColor);
    _isUniform = false;
    _texture = nullptr;
    _bufC.dispose();
}
//-----------------------------------------------------------------------------
//! Sets the background texture
void SLBackground::texture(SLGLTexture* backgroundTexture,
                           SLbool updatePerFrame)
{
    _texture = backgroundTexture;
    _updateTex = updatePerFrame;
    _isUniform = false;
    _bufC.dispose();
}
//-----------------------------------------------------------------------------
//! Draws the background as 2D rectangle with OpenGL buffers
/*! Draws the background as a flat 2D rectangle with a height and a width on two
triangles with zero in the bottom left corner: <br>
          w
       +-----+
       |    /|
       |   / |
    h  |  /  |
       | /   |
       |/    |
     0 +-----+
       0

We render the quad as a triangle strip: <br>
     0         2
       +-----+
       |    /|
       |   / |
       |  /  |
       | /   |
       |/    |
       +-----+
     1         3
*/
void SLBackground::render(SLint widthPX, SLint heightPX)
{
    return;
    SLGLState* stateGL = SLGLState::getInstance();
    stateGL->projectionMatrix.ortho(0.0f, (SLfloat)widthPX, 0.0f, (SLfloat)heightPX, 0.0f, 1.0f);
    stateGL->modelViewMatrix.identity();
    stateGL->depthTest(false);
    stateGL->multiSample(false);

    // Create or update buffer for vertex position and indexes
    if (_resX != widthPX || _resY != heightPX)
    {
        _resX = widthPX;
        _resY = heightPX;

        // Vertex X & Y of corners
        SLfloat P[16] = {0.0f, (SLfloat)_resY, 0.0f, 1.0f,
                        0.0f, 0.0f, 0.0f, 1.0f,
                        (SLfloat)_resX, (SLfloat)_resY, 0.0f, 1.0f,
                        (SLfloat)_resX, 0.0f, 0.0f, 1.0f};

        // Indexes for a triangle strip
        SLushort I[4] = {0,1,2,3};

        _bufP.generate(P, 4, 4);
        _bufI.generate(I, 4, 1, SL_UNSIGNED_SHORT, SL_ELEMENT_ARRAY_BUFFER);
    }

    // draw a textured or colored quad
    if(_texture)
    {
        if (!_bufT.id())
        {
            // Float array of texture coords of corners
            SLfloat T[8] = {0.0f, 1.0f,
                            0.0f, 0.0f,
                            1.0f, 1.0f,
                            1.0f, 0.0f};
            _bufT.generate(T, 4, 2);
        }

        _texture->bindActive(0);    // Enable & build texture

        // Setup texture only shader
        SLMat4f mvp(stateGL->projectionMatrix * stateGL->modelViewMatrix);
        SLGLProgram* sp = SLScene::current->programs(TextureOnly);
        sp->useProgram();
        sp->uniformMatrix4fv("u_mvpMatrix", 1, (SLfloat*)&mvp);
        sp->uniform1i("u_texture0", 0);

        // bind buffers and draw
        _bufP.bindAndEnableAttrib(sp->getAttribLocation("a_position"));
        _bufT.bindAndEnableAttrib(sp->getAttribLocation("a_texCoord"));

        ///////////////////////////////////////////////
        _bufI.bindAndDrawElementsAs(SL_TRIANGLE_STRIP);
        ///////////////////////////////////////////////

        _bufP.disableAttribArray();
        _bufT.disableAttribArray();

    } else // draw a colored quad
    {
        if (!_bufC.id())
        {
            // Float array of colors of corners
            SLfloat C[16] = {_colors[0].r, _colors[0].g, _colors[0].b, 1.0f,
                             _colors[1].r, _colors[1].g, _colors[1].b, 1.0f,
                             _colors[2].r, _colors[2].g, _colors[2].b, 1.0f,
                             _colors[3].r, _colors[3].g, _colors[3].b, 1.0f};
            _bufC.generate(C, 4, 4);
        }

        // Setup color attribute shader
        SLMat4f mvp(stateGL->projectionMatrix * stateGL->modelViewMatrix);
        SLGLProgram* sp = SLScene::current->programs(ColorAttribute);
        sp->useProgram();
        sp->uniformMatrix4fv("u_mvpMatrix", 1, (SLfloat*)&mvp);

        // bind buffers and draw
        _bufP.bindAndEnableAttrib(sp->getAttribLocation("a_position"));
        _bufC.bindAndEnableAttrib(sp->getAttribLocation("a_color"));

        ///////////////////////////////////////////////
        _bufI.bindAndDrawElementsAs(SL_TRIANGLE_STRIP);
        ///////////////////////////////////////////////

        _bufP.disableAttribArray();
        _bufC.disableAttribArray();
    }
}
//-----------------------------------------------------------------------------
//! Returns the interpolated color at the pixel position p[x,y]
/*! Returns the interpolated color at the pixel position p[x,y] for ray tracing.
    x is expected to be between 0 and window width.
    y is expected to be between 0 and window height.

     C    w    B
       +-----+
       | p  /|
       | * / |
     h |  /  |
       | /   |
       |/    |
     0 +-----+
     A 0
*/
SLCol4f SLBackground::colorAtPos(SLfloat x, SLfloat y)
{
    if (_isUniform)
        return _colors[0];

    if (_texture)
        return _texture->getTexelf(x/_resX, y/_resY);

    // top-down gradient
    if (_colors[0]==_colors[2] && _colors[1]==_colors[3])
    {   SLfloat f = y/_resY;
        return f*_colors[0] + (1-f)*_colors[1];
    }
    // left-right gradient
    if (_colors[0]==_colors[1] && _colors[2]==_colors[3])
    {   SLfloat f = x/_resX;
        return f*_colors[0] + (1-f)*_colors[2];
    }

    // Quadrilateral interpolation
    // First check with barycentric coords if p is in the upper left triangle
    SLVec2f p(x,y);
    SLVec3f bc = p.barycentricCoords(SLVec2f(0,0),
                                     SLVec2f((SLfloat)_resX,(SLfloat)_resY),
                                     SLVec2f(0,(SLfloat)_resY));
    SLfloat u = bc.x;
    SLfloat v = bc.y;
    SLfloat w = 1 - bc.x - bc.y;

    SLCol4f color;

    if (u>0 && v>0 && u+v<=1)
        color = w*_colors[0] + u*_colors[1] + v*_colors[2]; // upper left triangle
    else
    {   u=1-u; v=1-v; w=1-u-v;
        color = w*_colors[3] + v*_colors[1] + u*_colors[2]; // lower right triangle
    }

    return color;
}
//-----------------------------------------------------------------------------
