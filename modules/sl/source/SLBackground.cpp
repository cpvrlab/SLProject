//#############################################################################
//  File:      SLBackground.cpp
//  Date:      August 2015
//  Codestyle: https://github.com/cpvrlab/SLProject/wiki/SLProject-Coding-Style
//  Authors:   Marcus Hudritsch
//  License:   This software is provided under the GNU General Public License
//             Please visit: http://opensource.org/licenses/GPL-3.0
//#############################################################################

#include <SLBackground.h>
#include <SLGLProgram.h>
#include <SLGLTexture.h>
#include <SLScene.h>

//-----------------------------------------------------------------------------
//! The constructor initializes to a uniform BLACK background color
SLBackground::SLBackground(SLstring shaderDir)
  : SLObject("Background")
{
    _colors.push_back(SLCol4f::BLACK); // bottom left
    _colors.push_back(SLCol4f::BLACK); // bottom right
    _colors.push_back(SLCol4f::BLACK); // top right
    _colors.push_back(SLCol4f::BLACK); // top left
    _avgColor     = SLCol4f::BLACK;
    _isUniform    = true;
    _texture      = nullptr;
    _textureError = nullptr;
    _resX         = -1;
    _resY         = -1;

    _textureOnlyProgram = new SLGLProgramGeneric(nullptr,
                                                 shaderDir + "TextureOnly.vert",
                                                 shaderDir + "TextureOnly.frag");

    _colorAttributeProgram = new SLGLProgramGeneric(nullptr,
                                                    shaderDir + "ColorAttribute.vert",
                                                    shaderDir + "Color.frag");
    _deletePrograms        = true;
}
//-----------------------------------------------------------------------------
//! The constructor initializes to a uniform gray background color
SLBackground::SLBackground(SLGLProgram* textureOnlyProgram,
                           SLGLProgram* colorAttributeProgram)
  : SLObject("Background"),
    _textureOnlyProgram(textureOnlyProgram),
    _colorAttributeProgram(colorAttributeProgram),
    _deletePrograms(false)
{
    _colors.push_back(SLCol4f::BLACK); // bottom left
    _colors.push_back(SLCol4f::BLACK); // bottom right
    _colors.push_back(SLCol4f::BLACK); // top right
    _colors.push_back(SLCol4f::BLACK); // top left
    _avgColor     = SLCol4f::BLACK;
    _isUniform    = true;
    _texture      = nullptr;
    _textureError = nullptr;
    _resX         = -1;
    _resY         = -1;
}
//-----------------------------------------------------------------------------
SLBackground::~SLBackground()
{
    if (_deletePrograms)
    {
        delete _textureOnlyProgram;
        delete _colorAttributeProgram;
    }
}
//-----------------------------------------------------------------------------
//! Sets a uniform background color
void SLBackground::colors(const SLCol4f& uniformColor)
{
    _colors[0].set(uniformColor);
    _colors[1].set(uniformColor);
    _colors[2].set(uniformColor);
    _colors[3].set(uniformColor);
    _avgColor  = uniformColor;
    _isUniform = true;
    _texture   = nullptr;
    _vao.clearAttribs();
}
//-----------------------------------------------------------------------------
//! Sets a gradient top-down background color
void SLBackground::colors(const SLCol4f& topColor,
                          const SLCol4f& bottomColor)
{
    _colors[0].set(topColor);
    _colors[1].set(bottomColor);
    _colors[2].set(topColor);
    _colors[3].set(bottomColor);
    _avgColor  = (topColor + bottomColor) / 2.0f;
    _isUniform = false;
    _texture   = nullptr;
    _vao.clearAttribs();
}
//-----------------------------------------------------------------------------
//! Sets a gradient background color with a color per corner
void SLBackground::colors(const SLCol4f& topLeftColor,
                          const SLCol4f& bottomLeftColor,
                          const SLCol4f& topRightColor,
                          const SLCol4f& bottomRightColor)
{
    _colors[0].set(topLeftColor);
    _colors[1].set(bottomLeftColor);
    _colors[2].set(topRightColor);
    _colors[3].set(bottomRightColor);
    _avgColor  = (_colors[0] + _colors[1] + _colors[2] + _colors[3]) / 4.0f;
    _isUniform = false;
    _texture   = nullptr;
    _vao.clearAttribs();
}
//-----------------------------------------------------------------------------
//! Sets the background texture
void SLBackground::texture(SLGLTexture* backgroundTexture, bool fixAspectRatio)
{
    _texture        = backgroundTexture;
    _fixAspectRatio = fixAspectRatio;
    _isUniform      = false;
    _avgColor       = SLCol4f::BLACK;

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
    SLGLState* stateGL = SLGLState::instance();

    // Set orthographic projection
    stateGL->projectionMatrix.ortho(0.0f, (SLfloat)widthPX, 0.0f, (SLfloat)heightPX, 0.0f, 1.0f);
    stateGL->modelMatrix.identity();
    stateGL->viewMatrix.identity();
    stateGL->depthTest(false);
    stateGL->multiSample(false);

    // Get shader program
    SLGLProgram* sp = _texture ? _textureOnlyProgram : _colorAttributeProgram;
    sp->useProgram();
    sp->uniformMatrix4fv("u_mMatrix", 1, (SLfloat*)&stateGL->modelMatrix);
    sp->uniformMatrix4fv("u_vMatrix", 1, (SLfloat*)&stateGL->viewMatrix);
    sp->uniformMatrix4fv("u_pMatrix", 1, (SLfloat*)&stateGL->projectionMatrix);
    sp->uniform1f("u_oneOverGamma", SLLight::oneOverGamma());

    // Create or update buffer for vertex position and indices
    if (!_vao.vaoID() || _resX != widthPX || _resY != heightPX)
    {
        // texture width and height not yet valid on first call
        _resX = widthPX;
        _resY = heightPX;
        _vao.clearAttribs();

        SLfloat left = 0, right = (float)_resX, bottom = 0, top = (float)_resY;

        // the background is centered and stretched to the screen boarders while keeping the textures aspect ratio
        if (_texture && _fixAspectRatio)
        {
            SLfloat backgroundW, backgroundH;
            if ((SLfloat)_resX / (SLfloat)_resY > (SLfloat)_texture->width() / (SLfloat)_texture->height())
            {
                // screen is wider than texture -> adjust background width
                backgroundH = (float)_resY;
                backgroundW = (float)_resY / (SLfloat)_texture->height() * (SLfloat)_texture->width();
            }
            else
            {
                // screen is more narrow than texture -> adjust background height
                backgroundW = (float)_resX;
                backgroundH = (float)_resX / (SLfloat)_texture->width() * (SLfloat)_texture->height();
            }

            left   = (_resX - backgroundW) * 0.5f;
            right  = backgroundW + left;
            bottom = (_resY - backgroundH) * 0.5f;
            top    = backgroundH + bottom;

            _rect.set(left, bottom, backgroundW, backgroundH);
            SL_LOG("SLBackground: width:%f height:%f left:%f bottom:%f", rect().width, rect().height, rect().x, rect().x);
        }
        else
            _rect.set(0, 0, (float)widthPX, (float)heightPX);

        // Float array with vertex X & Y of corners
        SLVVec2f P = {{left, top},
                      {left, bottom},
                      {right, top},
                      {right, bottom}};

        _vao.setAttrib(AT_position, AT_position, &P);

        // Indexes for a triangle strip
        SLVushort I = {0, 1, 2, 3};
        _vao.setIndices(&I, nullptr);

        if (_texture)
        { // Float array of texture coordinates
            SLVVec2f T = {{0.0f, 1.0f}, {0.0f, 0.0f}, {1.0f, 1.0f}, {1.0f, 0.0f}};
            _vao.setAttrib(AT_uv1, AT_uv1, &T);
            _vao.generate(4);
        }
        else
        { // Float array of colors of corners
            SLVVec3f C = {{_colors[0].r, _colors[0].g, _colors[0].b},
                          {_colors[1].r, _colors[1].g, _colors[1].b},
                          {_colors[2].r, _colors[2].g, _colors[2].b},
                          {_colors[3].r, _colors[3].g, _colors[3].b}};
            _vao.setAttrib(AT_color, AT_color, &C);
            _vao.generate(4);
        }
    }

    // draw a textured or colored quad
    if (_texture)
    {
        _texture->bindActive(0);
        sp->uniform1i("u_matTexture0", 0);
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
void SLBackground::renderInScene(const SLMat4f& wm,
                                 const SLVec3f& LT,
                                 const SLVec3f& LB,
                                 const SLVec3f& RT,
                                 const SLVec3f& RB)
{
    SLGLState* stateGL = SLGLState::instance();

    // Get shader program
    SLGLProgram* sp = _texture
                        ? _textureOnlyProgram
                        : _colorAttributeProgram;
    sp->useProgram();
    sp->uniformMatrix4fv("u_mMatrix", 1, (const SLfloat*)&wm);
    sp->uniformMatrix4fv("u_vMatrix", 1, (const SLfloat*)&stateGL->viewMatrix);
    sp->uniformMatrix4fv("u_pMatrix", 1, (const SLfloat*)&stateGL->projectionMatrix);

    // Create or update buffer for vertex position and indices
    _vao.clearAttribs();

    // Float array with vertices
    SLVVec3f P = {LT, LB, RT, RB};
    _vao.setAttrib(AT_position, AT_position, &P);

    // Indexes for a triangle strip
    SLVushort I = {0, 1, 2, 3};
    _vao.setIndices(&I);

    if (_texture)
    { // Float array of texture coordinates
        SLVVec2f T = {{0.0f, 1.0f}, {0.0f, 0.0f}, {1.0f, 1.0f}, {1.0f, 0.0f}};
        _vao.setAttrib(AT_uv1, AT_uv1, &T);
        _vao.generate(4);
    }
    else
    { // Float array of colors of corners
        SLVVec3f C = {{_colors[0].r, _colors[0].g, _colors[0].b},
                      {_colors[1].r, _colors[1].g, _colors[1].b},
                      {_colors[2].r, _colors[2].g, _colors[2].b},
                      {_colors[3].r, _colors[3].g, _colors[3].b}};
        _vao.setAttrib(AT_color, AT_color, &C);
        _vao.generate(4);
    }

    // draw a textured or colored quad
    if (_texture)
    {
        _texture->bindActive(0);
        sp->uniform1i("u_matTexture0", 0);
    }

    ///////////////////////////////////////
    _vao.drawElementsAs(PT_triangleStrip);
    ///////////////////////////////////////
}
//-----------------------------------------------------------------------------
//! Returns the interpolated color at the pixel position p[x,y] used in raytracing.
/*! Returns the interpolated color at the pixel position p[x,y] for ray tracing.
    x is expected to be between 0 and width of the RT-frame.
    y is expected to be between 0 and height of the RT-frame.
    width is the width of the RT-frame
    height is the height of the RT-frame

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
SLCol4f SLBackground::colorAtPos(SLfloat x,
                                 SLfloat y,
                                 SLfloat width,
                                 SLfloat height)
{
    if (_isUniform)
        return _colors[0];

    if (_texture)
        return _texture->getTexelf(x / width, y / height);

    // top-down gradient
    if (_colors[0] == _colors[2] && _colors[1] == _colors[3])
    {
        SLfloat f = y / height;
        return f * _colors[0] + (1 - f) * _colors[1];
    }
    // left-right gradient
    if (_colors[0] == _colors[1] && _colors[2] == _colors[3])
    {
        SLfloat f = x / width;
        return f * _colors[0] + (1 - f) * _colors[2];
    }

    // Quadrilateral interpolation
    // First check with barycentric coords if p is in the upper left triangle
    SLVec2f p(x, y);
    SLVec3f bc(p.barycentricCoords(SLVec2f(0, 0),
                                   SLVec2f(width, height),
                                   SLVec2f(0, height)));
    SLfloat u = bc.x;
    SLfloat v = bc.y;
    SLfloat w = 1 - bc.x - bc.y;

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
