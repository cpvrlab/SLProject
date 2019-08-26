//#############################################################################
//  File:      SLTransferFunction.cpp
//  Purpose:   Implements a transfer function functionality
//  Author:    Marcus Hudritsch
//  Date:      July 2017
//  Copyright: Marcus Hudritsch
//             This software is provide under the GNU General Public License
//             Please visit: http://opensource.org/licenses/GPL-3.0
//#############################################################################

#include <stdafx.h> // Must be the 1st include followed by  an empty line

#include <SLApplication.h>
#include <SLScene.h>
#include <SLTransferFunction.h>

//-----------------------------------------------------------------------------
//! ctor with vector of alpha values and a predefined color LUT scheme
SLTransferFunction::SLTransferFunction(SLVTransferAlpha alphaValues,
                                       SLColorLUT       lut,
                                       SLuint           length)
{
    _min_filter = GL_LINEAR;
    _mag_filter = GL_LINEAR;
    _wrap_s     = GL_CLAMP_TO_EDGE;
    _wrap_t     = GL_CLAMP_TO_EDGE;
    _texType    = TT_color;
    _length     = length;
    _target     = GL_TEXTURE_2D; // OpenGL ES doesn't define 1D textures. We just make a 1 pixel high 2D texture

    colors(lut);

    for (auto alpha : alphaValues)
        _alphas.push_back(alpha);

    generateTexture();

    // Add pointer to the global resource vectors for deallocation
    SLApplication::scene->textures().push_back(this);
} //-----------------------------------------------------------------------------
//! ctor with vector of alpha and color values
SLTransferFunction::SLTransferFunction(SLVTransferAlpha alphaValues,
                                       SLVTransferColor colorValues,
                                       SLuint           length)
{
    _min_filter = GL_LINEAR;
    _mag_filter = GL_LINEAR;
    _wrap_s     = GL_CLAMP_TO_EDGE;
    _wrap_t     = GL_CLAMP_TO_EDGE;
    _texType    = TT_color;
    _length     = length;
    _target     = GL_TEXTURE_2D; // OpenGL ES doesn't define 1D textures. We just make a 1 pixel high 2D texture

    for (auto color : colorValues)
        _colors.push_back(color);

    for (auto alpha : alphaValues)
        _alphas.push_back(alpha);

    generateTexture();

    // Add pointer to the global resource vectors for deallocation
    SLApplication::scene->textures().push_back(this);
}
//-----------------------------------------------------------------------------
SLTransferFunction::~SLTransferFunction()
{
    _colors.clear();
    _alphas.clear();
}
//-----------------------------------------------------------------------------
//! Colors setter function by predefined color LUT
void SLTransferFunction::colors(SLColorLUT lut)
{
    assert(lut != CLUT_custom && "SLTransferFunction::colors: Custom LUT now allowed");

    _colors.clear();
    _colorLUT = lut;

    switch (lut)
    {
        case CLUT_BW:
            _colors.push_back(SLTransferColor(SLCol3f::BLACK, 0.0f));
            _colors.push_back(SLTransferColor(SLCol3f::WHITE, 1.0f));
            name("Transfer Function: Color LUT: B-W");
            break;
        case CLUT_WB:
            _colors.push_back(SLTransferColor(SLCol3f::WHITE, 0.0f));
            _colors.push_back(SLTransferColor(SLCol3f::BLACK, 1.0f));
            name("Transfer Function: Color LUT: W-B");
            break;
        case CLUT_RYGCB:
            _colors.push_back(SLTransferColor(SLCol3f::RED, 0.00f));
            _colors.push_back(SLTransferColor(SLCol3f::YELLOW, 0.25f));
            _colors.push_back(SLTransferColor(SLCol3f::GREEN, 0.50f));
            _colors.push_back(SLTransferColor(SLCol3f::CYAN, 0.75f));
            _colors.push_back(SLTransferColor(SLCol3f::BLUE, 1.00f));
            name("Transfer Function: Color LUT: R-Y-G-C-B");
            break;
        case CLUT_BCGYR:
            _colors.push_back(SLTransferColor(SLCol3f::BLUE, 0.00f));
            _colors.push_back(SLTransferColor(SLCol3f::CYAN, 0.25f));
            _colors.push_back(SLTransferColor(SLCol3f::GREEN, 0.50f));
            _colors.push_back(SLTransferColor(SLCol3f::YELLOW, 0.75f));
            _colors.push_back(SLTransferColor(SLCol3f::RED, 1.00f));
            name("Transfer Function: Color LUT: B-C-G-Y-R");
            break;
        case CLUT_RYGCBK:
            _colors.push_back(SLTransferColor(SLCol3f::RED, 0.00f));
            _colors.push_back(SLTransferColor(SLCol3f::YELLOW, 0.20f));
            _colors.push_back(SLTransferColor(SLCol3f::GREEN, 0.40f));
            _colors.push_back(SLTransferColor(SLCol3f::CYAN, 0.60f));
            _colors.push_back(SLTransferColor(SLCol3f::BLUE, 0.80f));
            _colors.push_back(SLTransferColor(SLCol3f::BLACK, 1.00f));
            name("Transfer Function: Color LUT: R-Y-G-C-B-K");
            break;
        case CLUT_KBCGYR:
            _colors.push_back(SLTransferColor(SLCol3f::BLACK, 0.00f));
            _colors.push_back(SLTransferColor(SLCol3f::BLUE, 0.20f));
            _colors.push_back(SLTransferColor(SLCol3f::CYAN, 0.40f));
            _colors.push_back(SLTransferColor(SLCol3f::GREEN, 0.60f));
            _colors.push_back(SLTransferColor(SLCol3f::YELLOW, 0.00f));
            _colors.push_back(SLTransferColor(SLCol3f::RED, 1.00f));
            name("Transfer Function: Color LUT: K-B-C-G-Y-R");
            break;
        case CLUT_RYGCBM:
            _colors.push_back(SLTransferColor(SLCol3f::RED, 0.00f));
            _colors.push_back(SLTransferColor(SLCol3f::YELLOW, 0.20f));
            _colors.push_back(SLTransferColor(SLCol3f::GREEN, 0.40f));
            _colors.push_back(SLTransferColor(SLCol3f::CYAN, 0.60f));
            _colors.push_back(SLTransferColor(SLCol3f::BLUE, 0.80f));
            _colors.push_back(SLTransferColor(SLCol3f::MAGENTA, 1.00f));
            name("Transfer Function: Color LUT: R-Y-G-C-B-M");
            break;
        case CLUT_MBCGYR:
            _colors.push_back(SLTransferColor(SLCol3f::MAGENTA, 0.00f));
            _colors.push_back(SLTransferColor(SLCol3f::BLUE, 0.20f));
            _colors.push_back(SLTransferColor(SLCol3f::CYAN, 0.40f));
            _colors.push_back(SLTransferColor(SLCol3f::GREEN, 0.60f));
            _colors.push_back(SLTransferColor(SLCol3f::YELLOW, 0.00f));
            _colors.push_back(SLTransferColor(SLCol3f::RED, 1.00f));
            name("Transfer Function: Color LUT: M-B-C-G-Y-R");
            break;
        default:
            SL_EXIT_MSG("SLTransferFunction::colors: undefined color LUT.");
            break;
    }
}
//-----------------------------------------------------------------------------
//! Generates the full 256 value LUT as 1x256 RGBA texture
void SLTransferFunction::generateTexture()
{
    assert(_length > 1);
    assert(!_alphas.empty() &&
           !_colors.empty() &&
           "SLTransferFunction::generateTexture: Not enough alpha and/or color values.");

    // Delete old data in case of regeneration
    clearData();

    SLfloat delta = 1.0f / (SLfloat)_length;

    // Sort alphas and colors by position
    sort(_alphas.begin(),
         _alphas.end(),
         [](SLTransferAlpha a, SLTransferAlpha b) { return a.pos < b.pos; });
    sort(_colors.begin(),
         _colors.end(),
         [](SLTransferColor a, SLTransferColor b) { return a.pos < b.pos; });

    // Check out of bounds position (0-1)
    if (_alphas.front().pos < 0.0f)
        SL_EXIT_MSG("SLTransferFunction::generateTexture: Lower alpha pos below 0");
    if (_alphas.back().pos > 1.0f)
        SL_EXIT_MSG("SLTransferFunction::generateTexture: Upper alpha pos above 1");
    if (_colors.front().pos < 0.0f)
        SL_EXIT_MSG("SLTransferFunction::generateTexture: Lower color pos below 0");
    if (_colors.back().pos > 1.0f)
        SL_EXIT_MSG("SLTransferFunction::generateTexture: Upper color pos above 1");

    // Add boundry node if they are not at position 0 or 1
    if (_alphas.front().pos > 0.0f)
        _alphas.insert(_alphas.begin(), SLTransferAlpha(_alphas.front().alpha, 0.0f));
    if (_alphas.back().pos < 1.0f)
        _alphas.push_back(SLTransferAlpha(_alphas.back().alpha, 1.0f));
    if (_colors.front().pos > 0.0f)
        _colors.insert(_colors.begin(), SLTransferColor(_colors.front().color, 0.0f));
    if (_colors.back().pos < 1.0f)
        _colors.push_back(SLTransferColor(_colors.back().color, 1.0f));

    // Check that the delta between positions is larger than delta
    for (SLuint c = 0; c < _colors.size() - 1; ++c)
        if ((_colors[c + 1].pos - _colors[c].pos) < delta)
            SL_EXIT_MSG("SLTransferFunction::generateTexture: Color position deltas to small.");
    for (SLuint a = 0; a < _alphas.size() - 1; ++a)
        if ((_alphas[a + 1].pos - _alphas[a].pos) < delta)
            SL_EXIT_MSG("SLTransferFunction::generateTexture: Alpha position deltas to small.");

    // Clamp all color and alpha values
    for (auto c : _colors) c.color.clampMinMax(0.0f, 1.0f);
    for (auto a : _alphas) a.alpha = SL_clamp(a.alpha, 0.0f, 1.0f);

    // Finally create transfer function vector by lerping color and alpha values
    SLuint  c      = 0;    // current color segment index
    SLfloat pos    = 0.0f; // overall position between 0-1
    SLfloat posC   = 0.0f; // position in color segment
    SLfloat deltaC = 1.0f / ((_colors[c + 1].pos - _colors[c].pos) / delta);

    SLVCol4f tf;
    tf.resize((SLuint)_length);

    // Interpolate color values
    for (SLuint i = 0; i < _length; ++i)
    {
        tf[i].r = SL_lerp(posC, _colors[c].color.r, _colors[c + 1].color.r);
        tf[i].g = SL_lerp(posC, _colors[c].color.g, _colors[c + 1].color.g);
        tf[i].b = SL_lerp(posC, _colors[c].color.b, _colors[c + 1].color.b);

        pos += delta;
        posC += deltaC;

        if (pos > _colors[c + 1].pos && c < _colors.size() - 2)
        {
            c++;
            posC   = 0.0f;
            deltaC = 1.0f / ((_colors[c + 1].pos - _colors[c].pos) / delta);
        }
    }

    // Interpolate alpha value
    SLuint  a      = 0;    // current alpha segment index
    SLfloat posA   = 0.0f; // position in alpha segment
    SLfloat deltaA = 1.0f / ((_alphas[a + 1].pos - _alphas[a].pos) / delta);
    pos            = 0.0f;
    for (SLuint i = 0; i < _length; ++i)
    {
        tf[i].a = SL_lerp(posA, _alphas[a].alpha, _alphas[a + 1].alpha);

        //_allAlphas[i] = tf[i].a;

        pos += delta;
        posA += deltaA;

        if (pos > _alphas[a + 1].pos && a < _alphas.size() - 2)
        {
            a++;
            posA   = 0.0f;
            deltaA = 1.0f / ((_alphas[a + 1].pos - _alphas[a].pos) / delta);
        }
    }

    // Create 1 x lenght sized image from SLCol4f values
    load(tf);
}
//-----------------------------------------------------------------------------
//! Returns all alpha values of the transfer function as a float vector
SLVfloat SLTransferFunction::allAlphas()
{
    SLVfloat allA;
    allA.resize(_length);

    for (SLuint i = 0; i < _length; ++i)
    {
        CVVec4f c4f = _images[0]->getPixeli((SLint)i, 0);
        allA[i] = c4f[3];
    }

    return allA;
}
//-----------------------------------------------------------------------------
