//#############################################################################
//  File:      SLTransferFunction.cpp
//  Purpose:   Implements a transfer function functionality
//  Author:    Marcus Hudritsch
//  Date:      July 2017
//  Copyright: Marcus Hudritsch
//             This software is provide under the GNU General Public License
//             Please visit: http://opensource.org/licenses/GPL-3.0
//#############################################################################

#include <stdafx.h>
#include <SLTransferFunction.h>

//-----------------------------------------------------------------------------
//! ctor with vector of alpha values and a predefined color LUT scheme
SLTransferFunction::SLTransferFunction(SLVTransferAlpha alphaValues,
                                       SLColorLUT lut,
                                       SLint length)
{
    _length = length;

    colors(lut);

    for (auto alpha : alphaValues)
        _alphas.push_back(alpha);

    generateTexture();
}//-----------------------------------------------------------------------------
//! ctor with vector of alpha and color values
SLTransferFunction::SLTransferFunction(SLVTransferAlpha alphaValues,
                                       SLVTransferColor colorValues,
                                       SLint length)
{
    _length = length;

    for (auto color : colorValues)
        _colors.push_back(color);

    for (auto alpha : alphaValues)
        _alphas.push_back(alpha);

    generateTexture();
}
//-----------------------------------------------------------------------------
//! Colors setter function by predefined color LUT
void SLTransferFunction::colors(SLColorLUT lut)
{
    assert(lut!=CLUT_custom && "SLTransferFunction::colors: Custom LUT now allowed");

    _colors.clear();
    _colorLUT = lut;

    switch (lut)
    {   case CLUT_BW:
            _colors.push_back(SLTransferColor(SLCol3f::BLACK,   0.0f));
            _colors.push_back(SLTransferColor(SLCol3f::WHITE,   1.0f));
            break;
        case CLUT_WB:
            _colors.push_back(SLTransferColor(SLCol3f::WHITE,   0.0f));
            _colors.push_back(SLTransferColor(SLCol3f::BLACK,   1.0f));
            break;
        case CLUT_RYGCB:
            _colors.push_back(SLTransferColor(SLCol3f::RED,     0.00f));
            _colors.push_back(SLTransferColor(SLCol3f::YELLOW,  0.25f));
            _colors.push_back(SLTransferColor(SLCol3f::GREEN,   0.50f));
            _colors.push_back(SLTransferColor(SLCol3f::CYAN,    0.75f));
            _colors.push_back(SLTransferColor(SLCol3f::BLUE,    1.00f));
            break;
        case CLUT_BCGYR:
            _colors.push_back(SLTransferColor(SLCol3f::BLUE,    0.00f));
            _colors.push_back(SLTransferColor(SLCol3f::CYAN,    0.25f));
            _colors.push_back(SLTransferColor(SLCol3f::GREEN,   0.50f));
            _colors.push_back(SLTransferColor(SLCol3f::YELLOW,  0.75f));
            _colors.push_back(SLTransferColor(SLCol3f::RED,     1.00f));
            break;
        case CLUT_RYGCBK:
            _colors.push_back(SLTransferColor(SLCol3f::RED,     0.00f));
            _colors.push_back(SLTransferColor(SLCol3f::YELLOW,  0.20f));
            _colors.push_back(SLTransferColor(SLCol3f::GREEN,   0.40f));
            _colors.push_back(SLTransferColor(SLCol3f::CYAN,    0.60f));
            _colors.push_back(SLTransferColor(SLCol3f::BLUE,    0.80f));
            _colors.push_back(SLTransferColor(SLCol3f::BLACK,   1.00f));
            break;
        case CLUT_KBCGYR:
            _colors.push_back(SLTransferColor(SLCol3f::BLACK,   0.00f));
            _colors.push_back(SLTransferColor(SLCol3f::BLUE,    0.20f));
            _colors.push_back(SLTransferColor(SLCol3f::CYAN,    0.40f));
            _colors.push_back(SLTransferColor(SLCol3f::GREEN,   0.60f));
            _colors.push_back(SLTransferColor(SLCol3f::YELLOW,  0.00f));
            _colors.push_back(SLTransferColor(SLCol3f::RED,     1.00f));
            break;
        case CLUT_RYGCBM:
            _colors.push_back(SLTransferColor(SLCol3f::RED,     0.00f));
            _colors.push_back(SLTransferColor(SLCol3f::YELLOW,  0.20f));
            _colors.push_back(SLTransferColor(SLCol3f::GREEN,   0.40f));
            _colors.push_back(SLTransferColor(SLCol3f::CYAN,    0.60f));
            _colors.push_back(SLTransferColor(SLCol3f::BLUE,    0.80f));
            _colors.push_back(SLTransferColor(SLCol3f::MAGENTA, 1.00f));
            break;
        case CLUT_MBCGYR:
            _colors.push_back(SLTransferColor(SLCol3f::MAGENTA, 0.00f));
            _colors.push_back(SLTransferColor(SLCol3f::BLUE,    0.20f));
            _colors.push_back(SLTransferColor(SLCol3f::CYAN,    0.40f));
            _colors.push_back(SLTransferColor(SLCol3f::GREEN,   0.60f));
            _colors.push_back(SLTransferColor(SLCol3f::YELLOW,  0.00f));
            _colors.push_back(SLTransferColor(SLCol3f::RED,     1.00f));
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
    assert(_alphas.size() > 0 &&
           _colors.size() > 0 &&
           "SLTransferFunction::generateTexture: Not enough alpha and/or color values.");

    // Sort alphas and colors by position
    sort(_alphas.begin(), _alphas.end(),
         [](SLTransferAlpha a, SLTransferAlpha b) {return a.pos < b.pos;});
    sort(_colors.begin(), _colors.end(),
         [](SLTransferColor a, SLTransferColor b) {return a.pos < b.pos;});

    // Check out of bounds position (0-1)
    if (_alphas[0].pos < 0.0f)
        SL_EXIT_MSG("SLTransferFunction::generateTexture: Lower alpha pos below 0");
    if (_alphas[_alphas.size()-1].pos > 1.0f)
        SL_EXIT_MSG("SLTransferFunction::generateTexture: Upper alpha pos above 1");
    if (_colors[0].pos < 0.0f)
        SL_EXIT_MSG("SLTransferFunction::generateTexture: Lower color pos below 0");
    if (_colors[_colors.size()-1].pos > 1.0f)
        SL_EXIT_MSG("SLTransferFunction::generateTexture: Upper color pos above 1");

    // Correct boundries
    if (_alphas[0].pos > 0.0f) _alphas[0].pos = 0.0;
    if (_alphas[_alphas.size()-1].pos < 1.0f) _alphas[_alphas.size()-1].pos = 1.0f;
    if (_colors[0].pos > 0.0f) _colors[0].pos = 0.0;
    if (_colors[_colors.size()-1].pos < 1.0f) _colors[_alphas.size()-1].pos = 1.0f;

    // Clamp all color and alpha values
    for (auto c : _colors) c.color.clampMinMax(0.0f, 1.0f);
    for (auto a : _alphas) a.alpha = SL_clamp(a.alpha, 0.0f, 1.0f);

    // Finally linear interpolate color and alpha values
    SLint c = 0, a = 0;
    SLfloat delta = 1.0f / (SLfloat)_length;
    SLfloat pos = 0.0f;

    for (SLint i=0; i < _length; ++i)
    {
//        _colors[c].color.lerp()

//        pos += delta;
    }
}
//-----------------------------------------------------------------------------
