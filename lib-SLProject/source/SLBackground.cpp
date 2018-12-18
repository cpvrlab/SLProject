//#############################################################################
//  File:      SLBackground.cpp
//  Author:    Marcus Hudritsch
//  Date:      August 2015
//  Codestyle: https://github.com/cpvrlab/SLProject/wiki/Coding-Style-Guidelines
//  Copyright: Marcus Hudritsch
//             This software is provide under the GNU General Public License
//             Please visit: http://opensource.org/licenses/GPL-3.0
//#############################################################################

#include <stdafx.h> // Must be the 1st include followed by  an empty line

#ifdef SL_MEMLEAKDETECT    // set in SL.h for debug config only
#    include <debug_new.h> // memory leak detector
#endif

#include <SLApplication.h>
#include <SLBackground.h>
#include <SLGLProgram.h>
#include <SLGLTexture.h>
#include <SLScene.h>

//-----------------------------------------------------------------------------
SLBackground::~SLBackground()
{
}
//-----------------------------------------------------------------------------
//! The constructor initializes to a uniform black background color
SLBackground::SLBackground() : SLObject("Background")
{
    _colors.push_back(SLCol4f::BLACK); // bottom left
    _colors.push_back(SLCol4f::BLACK); // bottom right
    _colors.push_back(SLCol4f::BLACK); // top right
    _colors.push_back(SLCol4f::BLACK); // top left
    _isUniform    = true;
    _texture      = nullptr;
    _textureError = SLApplication::scene->videoTextureErr(); // Fix for black video error
    _resX         = -1;
    _resY         = -1;
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
    _texture   = nullptr;
    _vao.clearAttribs();
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
    _texture   = nullptr;
    _vao.clearAttribs();
}
//-----------------------------------------------------------------------------
//! Sets a gradient background color with a color per corner
void SLBackground::colors(SLCol4f topLeftColor,
                          SLCol4f bottomLeftColor,
                          SLCol4f topRightColor,
                          SLCol4f bottomRightColor)
{
    _colors[0].set(topLeftColor);
    _colors[1].set(bottomLeftColor);
    _colors[2].set(topRightColor);
    _colors[3].set(bottomRightColor);
    _isUniform = false;
    _texture   = nullptr;
    _vao.clearAttribs();
}
//-----------------------------------------------------------------------------
//! Sets the background texture
void SLBackground::texture(SLGLTexture* backgroundTexture)
{
    _texture   = backgroundTexture;
    _isUniform = false;
    _vao.clearAttribs();
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
    SLGLState* stateGL = SLGLState::getInstance();
    SLScene*   s       = SLApplication::scene;

    // Set orthographic projection
    stateGL->projectionMatrix.ortho(0.0f, (SLfloat)widthPX, 0.0f, (SLfloat)heightPX, 0.0f, 1.0f);
    stateGL->modelViewMatrix.identity();

    // Combine modelview-projection matrix
    SLMat4f mvp(stateGL->projectionMatrix * stateGL->modelViewMatrix);

    stateGL->depthTest(false);
    stateGL->multiSample(false);

    // Get shader program
    SLGLProgram* sp = _texture ? s->programs(SP_TextureOnly) : s->programs(SP_colorAttribute);
    sp->useProgram();
    sp->uniformMatrix4fv("u_mvpMatrix", 1, (SLfloat*)&mvp);
    sp->uniform1f("u_oneOverGamma", stateGL->oneOverGamma);

    // Create or update buffer for vertex position and indices
    if (!_vao.id() || _resX != widthPX || _resY != heightPX)
    {
        _resX = widthPX;
        _resY = heightPX;
        _vao.clearAttribs();

        // Float array with vertex X & Y of corners
        SLVVec2f P = {{0.0f, (SLfloat)_resY},
                      {0.0f, 0.0f},
                      {(SLfloat)_resX, (SLfloat)_resY},
                      {(SLfloat)_resX, 0.0f}};

        _vao.setAttrib(AT_position, sp->getAttribLocation("a_position"), &P);

        // Indexes for a triangle strip
        SLVushort I = {0, 1, 2, 3};
        _vao.setIndices(&I);

        if (_texture)
        { // Float array of texture coordinates
            SLVVec2f T = {{0.0f, 1.0f}, {0.0f, 0.0f}, {1.0f, 1.0f}, {1.0f, 0.0f}};
            _vao.setAttrib(AT_texCoord, sp->getAttribLocation("a_texCoord"), &T);
            _vao.generate(4);
        }
        else
        { // Float array of colors of corners
            SLVVec3f C = {{_colors[0].r, _colors[0].g, _colors[0].b},
                          {_colors[1].r, _colors[1].g, _colors[1].b},
                          {_colors[2].r, _colors[2].g, _colors[2].b},
                          {_colors[3].r, _colors[3].g, _colors[3].b}};
            _vao.setAttrib(AT_color, sp->getAttribLocation("a_color"), &C);
            _vao.generate(4);
        }
    }

    // draw a textured or colored quad
    if (_texture)
    { // if video texture is not ready show error texture
        if (_texture->texName())
            _texture->bindActive(0);
        else
            _textureError->bindActive(0);
        sp->uniform1i("u_texture0", 0);
    }

    //////////////////////////////////////
    _vao.drawElementsAs(PT_triangleStrip);
    //////////////////////////////////////
}
//-----------------------------------------------------------------------------
//! Draws the background as a quad on the far clipping plane
/*! We render the quad as a triangle strip: <br>
     LT       RT
       +-----+
       |    /|
       |   / |
       |  /  |
       | /   |
       |/    |
       +-----+
     LB       RB
*/
void SLBackground::renderInScene(SLVec3f LT, SLVec3f LB, SLVec3f RT, SLVec3f RB)
{
    SLGLState* stateGL = SLGLState::getInstance();
    SLScene*   s       = SLApplication::scene;

    // Get shader program
    SLGLProgram* sp = _texture
                        ? s->programs(SP_TextureOnly)
                        : s->programs(SP_colorAttribute);
    sp->useProgram();
    sp->uniformMatrix4fv("u_mvpMatrix", 1, (const SLfloat*)stateGL->mvpMatrix());

    // Create or update buffer for vertex position and indices
    _vao.clearAttribs();

    // Float array with vertices
    SLVVec3f P = {LT, LB, RT, RB};
    _vao.setAttrib(AT_position, sp->getAttribLocation("a_position"), &P);

    // Indexes for a triangle strip
    SLVushort I = {0, 1, 2, 3};
    _vao.setIndices(&I);

    if (_texture)
    { // Float array of texture coordinates
        SLVVec2f T = {{0.0f, 1.0f}, {0.0f, 0.0f}, {1.0f, 1.0f}, {1.0f, 0.0f}};
        _vao.setAttrib(AT_texCoord, sp->getAttribLocation("a_texCoord"), &T);
        _vao.generate(4);
    }
    else
    { // Float array of colors of corners
        SLVVec3f C = {{_colors[0].r, _colors[0].g, _colors[0].b},
                      {_colors[1].r, _colors[1].g, _colors[1].b},
                      {_colors[2].r, _colors[2].g, _colors[2].b},
                      {_colors[3].r, _colors[3].g, _colors[3].b}};
        _vao.setAttrib(AT_color, sp->getAttribLocation("a_color"), &C);
        _vao.generate(4);
    }

    // draw a textured or colored quad
    if (_texture)
    { // if video texture is not ready show error texture
        if (_texture->texName())
            _texture->bindActive(0);
        else
            _textureError->bindActive(0);
        sp->uniform1i("u_texture0", 0);
    }

    ///////////////////////////////////////
    _vao.drawElementsAs(PT_triangleStrip);
    ///////////////////////////////////////
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
        return _texture->getTexelf(x / _resX, y / _resY);

    // top-down gradient
    if (_colors[0] == _colors[2] && _colors[1] == _colors[3])
    {
        SLfloat f = y / _resY;
        return f * _colors[0] + (1 - f) * _colors[1];
    }
    // left-right gradient
    if (_colors[0] == _colors[1] && _colors[2] == _colors[3])
    {
        SLfloat f = x / _resX;
        return f * _colors[0] + (1 - f) * _colors[2];
    }

    // Quadrilateral interpolation
    // First check with barycentric coords if p is in the upper left triangle
    SLVec2f p(x, y);
    SLVec3f bc = p.barycentricCoords(SLVec2f(0, 0),
                                     SLVec2f((SLfloat)_resX, (SLfloat)_resY),
                                     SLVec2f(0, (SLfloat)_resY));
    SLfloat u  = bc.x;
    SLfloat v  = bc.y;
    SLfloat w  = 1 - bc.x - bc.y;

    SLCol4f color;

    if (u > 0 && v > 0 && u + v <= 1)
        color = w * _colors[0] + u * _colors[1] + v * _colors[2]; // upper left triangle
    else
    {
        u     = 1 - u;
        v     = 1 - v;
        w     = 1 - u - v;
        color = w * _colors[3] + v * _colors[1] + u * _colors[2]; // lower right triangle
    }

    return color;
}
//-----------------------------------------------------------------------------
