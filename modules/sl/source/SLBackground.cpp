//#############################################################################
//  File:      SLBackground.cpp
//  Author:    Marcus Hudritsch
//  Date:      August 2015
//  Codestyle: https://github.com/cpvrlab/SLProject/wiki/SLProject-Coding-Style
//  Copyright: Marcus Hudritsch
//             This software is provide under the GNU General Public License
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

    _textureOnlyProgram    = new SLGLProgramGeneric(nullptr,
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
void SLBackground::texture(SLGLTexture* backgroundTexture, bool repeatBlurred)
{
    _texture       = backgroundTexture;
    _repeatBlurred = repeatBlurred;
    _isUniform     = false;
    _avgColor      = SLCol4f::BLACK;

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
    stateGL->modelViewMatrix.identity();

    // Combine modelview-projection matrix
    SLMat4f mvp(stateGL->projectionMatrix * stateGL->modelViewMatrix);

    stateGL->depthTest(false);
    stateGL->multiSample(false);

    // Get shader program
    SLGLProgram* sp = _texture ? _textureOnlyProgram : _colorAttributeProgram;
    sp->useProgram();
    sp->uniformMatrix4fv("u_mvpMatrix", 1, (SLfloat*)&mvp);
    sp->uniform1f("u_oneOverGamma", SLLight::oneOverGamma());

    // Create or update buffer for vertex position and indices
    if (!_vao.vaoID() || _resX != widthPX || _resY != heightPX)
    {
        _resX = widthPX;
        _resY = heightPX;
        _vao.clearAttribs();

        //if repeatBlurred is active and texture aspect ratio does not fit to sceneview aspect ratio
        if (_repeatBlurred && _texture &&
            std::abs((float)_resX / (float)_resY - (float)_texture->width() / (float)_texture->height()) > 0.001f)
        {
            //in this case we add 2 additional triangles left and right for bars containing a small subregion
            defineWithBars();
        }
        else
        {
            // Float array with vertex X & Y of corners
            SLVVec2f P = {{0.0f, (SLfloat)_resY},
                          {0.0f, 0.0f},
                          {(SLfloat)_resX, (SLfloat)_resY},
                          {(SLfloat)_resX, 0.0f}};

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
void SLBackground::renderInScene(const SLVec3f& LT,
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
    sp->uniformMatrix4fv("u_mvpMatrix", 1, (const SLfloat*)stateGL->mvpMatrix());

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
/*! Draws the background as a flat 2D rectangle with a height and a width on two
triangles with zero in the bottom left corner: <br>
           w
      +----+----+----+
       |    /|    /|     /|
       |   / |   / |    / |
     h  |  /  |  /  |   /  |
       | /   | /   |  /   |
       |/    |/    |/     |
     0 +----+----+----+
       0

We render the quad as a triangle strip: <br>
        0     2     4     6
        +----+----+----+
         |    /|    /|     /|
         |   / |   / |    / |
         |  /  |  /  |   /  |
         | /   | /   |  /   |
         |/    |/    |/     |
        +----+----+----+
        1     3    5      7
 
 ( drawings are for svWdivH > texWdivH case )
 */

/*! Draws the background as a flat 2D rectangle with a height and a width on two
triangles with zero in the bottom left corner: <br>
           w
      +----++----++----+
       |    /|  |    /| |     /|
       |   / |  |   / | |    / |
     h  |  /  |  |  /  | |   /  |
       | /   |  | /   | |  /   |
       |/    |  |/    || /     |
     0 +----++----++----+
       0

We render the quad as a triangle strip: <br>
      0     2 4    6 8    10
      +----++----++----+
       |    /|  |    /| |     /|
       |   / |  |   / | |    / |
       |  /  |  |  /  | |   /  |
       | /   |  | /   | |  /   |
       |/    |  |/    || /     |
      +----+ +----++----+
      1     3 5    7 9     11
 
 ( drawings are for svWdivH > texWdivH case
  For the other case triangles and indices are mirrored at the x-y quadrant diagonal )
 */
void SLBackground::defineWithBars()
{
    SLfloat svWdivH  = (SLfloat)_resX / (SLfloat)_resY;
    SLfloat texWdivH = (SLfloat)_texture->width() / (SLfloat)_texture->height();

    //screen width and height
    SLfloat sW = (SLfloat)_resX;
    SLfloat sH = (SLfloat)_resY;

    //scale factor for bar dimensions for texture coordinate definition
    SLfloat barS = 0.1f;

    if (svWdivH > texWdivH)
    {
        //texture width (relative to resolution defined by resX/resY)
        SLfloat tW = texWdivH * sH;
        //bar width
        SLfloat bW = 0.5f * (sW - tW);

        // Float array with vertex X & Y of corners
        SLVVec2f P = {
          {0.0f, sH},      //0
          {0.0f, 0.0f},    //1
          {bW, sH},        //2
          {bW, 0.0f},      //3
          {bW, sH},        //4
          {bW, 0.0f},      //5
          {bW + tW, sH},   //6
          {bW + tW, 0.0f}, //7
          {bW + tW, sH},   //8
          {bW + tW, 0.0f}, //9
          {sW, sH},        //10
          {sW, 0.0f}       //11
        };

        _vao.setAttrib(AT_position, AT_position, &P);

        // Indexes for a triangle strip
        SLVushort I = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11};
        _vao.setIndices(&I);

        SLfloat bWS = bW * barS / sW;
        SLfloat bHS = barS;
        //offset from texture boarder to scaled bar in texture
        SLfloat bHO = 0.5f * (1.0f - bHS);

        SLVVec2f T = {
          {0.0f, bHO + bHS}, //0
          {0.0f, bHO},       //1
          {bWS, bHO + bHS},  //2
          {bWS, bHO},        //3

          {0.0f, 1.0f}, //4
          {0.0f, 0.0f}, //5
          {1.0f, 1.0f}, //6
          {1.0f, 0.0f}, //7

          {1.0f - bWS, bHO + bHS}, //8
          {1.0f - bWS, bHO},       //9
          {1.0f, bHO + bHS},       //10
          {1.0f, bHO}};            //11

        _vao.setAttrib(AT_uv1, AT_uv1, &T);
        _vao.generate(12);
    }
    else
    {
        //texture height (relative to resolution defined by resX/resY)
        SLfloat tH = sW / texWdivH;
        //bar height
        SLfloat bH = 0.5f * (sH - tH);

        // Float array with vertex X & Y of corners
        SLVVec2f P = {
          {sW, 0.0f},      //0
          {0.0f, 0.0f},    //1
          {sW, bH},        //2
          {0.0f, bH},      //3
          {sW, bH},        //4
          {0.0f, bH},      //5
          {sW, bH + tH},   //6
          {0.0f, bH + tH}, //7
          {sW, bH + tH},   //8
          {0.0f, bH + tH}, //9
          {sW, sH},        //10
          {0, sH}          //11
        };

        _vao.setAttrib(AT_position, AT_position, &P);

        // Indexes for a triangle strip
        SLVushort I = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11};
        _vao.setIndices(&I);

        SLfloat bWS = barS;
        SLfloat bHS = bH * barS / sH;
        //offset from texture boarder to scaled bar in texture
        SLfloat bO = 0.5f * (1.0f - bWS);

        SLVVec2f T = {
          {bO + bWS, 0.0f}, //0
          {bO, 0.0f},       //1
          {bO + bWS, bHS},  //2
          {bO, bHS},        //3

          {1.0f, 0.0f}, //4
          {0.0f, 0.0f}, //5
          {1.0f, 1.0f}, //6
          {0.0f, 1.0f}, //7

          {bO + bWS, 1.0f - bHS}, //8
          {bO, 1.0f - bHS},       //9
          {bO + bWS, 1.0f},       //10
          {bO, 1.0f}};            //11
        _vao.setAttrib(AT_uv1, AT_uv1, &T);
        _vao.generate(12);
    }
}
//-----------------------------------------------------------------------------
