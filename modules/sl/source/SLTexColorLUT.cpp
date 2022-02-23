//#############################################################################
//  File:      SLTexColorLUT.cpp
//  Purpose:   Implements a transfer function functionality
//  Date:      July 2017
//  Authors:   Marcus Hudritsch
//  License:   This software is provided under the GNU General Public License
//             Please visit: http://opensource.org/licenses/GPL-3.0
//#############################################################################

#include <SLTexColorLUT.h>
#include <SLAssetManager.h>

//-----------------------------------------------------------------------------
//! Default ctor color LUT of a specific SLColorLUTType
SLTexColorLUT::SLTexColorLUT(SLAssetManager* assetMgr,
                             SLColorLUTType  lutType,
                             SLuint          length)
{
    _min_filter = GL_LINEAR;
    _mag_filter = GL_LINEAR;
    _wrap_s     = GL_CLAMP_TO_EDGE;
    _wrap_t     = GL_CLAMP_TO_EDGE;
    _texType    = TT_diffuse;
    _length     = length;
    _target     = GL_TEXTURE_2D; // OpenGL ES doesn't define 1D textures. We just make a 1 pixel high 2D texture

    colors(lutType);

    generateTexture();

    // Add pointer to the global resource vectors for deallocation
    if (assetMgr)
        assetMgr->textures().push_back(this);
}
//-----------------------------------------------------------------------------
//! ctor with vector of alpha values and a predefined color LUT scheme
SLTexColorLUT::SLTexColorLUT(SLAssetManager*  assetMgr,
                             SLVAlphaLUTPoint alphaValues,
                             SLColorLUTType   lutType,
                             SLuint           length)
{
    _min_filter = GL_LINEAR;
    _mag_filter = GL_LINEAR;
    _wrap_s     = GL_CLAMP_TO_EDGE;
    _wrap_t     = GL_CLAMP_TO_EDGE;
    _texType    = TT_diffuse;
    _length     = length;
    _target     = GL_TEXTURE_2D; // OpenGL ES doesn't define 1D textures. We just make a 1 pixel high 2D texture

    colors(lutType);

    for (auto alpha : alphaValues)
        _alphas.push_back(alpha);

    generateTexture();

    // Add pointer to the global resource vectors for deallocation
    if (assetMgr)
        assetMgr->textures().push_back(this);
}
//-----------------------------------------------------------------------------
//! ctor with vector of alpha and color values
SLTexColorLUT::SLTexColorLUT(SLAssetManager*  assetMgr,
                             SLVAlphaLUTPoint alphaValues,
                             SLVColorLUTPoint colorValues,
                             SLuint           length)
{
    _min_filter = GL_LINEAR;
    _mag_filter = GL_LINEAR;
    _wrap_s     = GL_CLAMP_TO_EDGE;
    _wrap_t     = GL_CLAMP_TO_EDGE;
    _texType    = TT_diffuse;
    _length     = length;
    _target     = GL_TEXTURE_2D; // OpenGL ES doesn't define 1D textures. We just make a 1 pixel high 2D texture

    for (auto color : colorValues)
        _colors.push_back(color);

    for (auto alpha : alphaValues)
        _alphas.push_back(alpha);

    generateTexture();

    // Add pointer to the global resource vectors for deallocation
    if (assetMgr)
        assetMgr->textures().push_back(this);
}
//-----------------------------------------------------------------------------
SLTexColorLUT::~SLTexColorLUT()
{
    _colors.clear();
    _alphas.clear();
}
//-----------------------------------------------------------------------------
//! Colors setter function by predefined color LUT
void SLTexColorLUT::colors(SLColorLUTType lutType)
{
    assert(lutType != CLUT_custom && "SLTexColorLUT::colors: Custom LUT now allowed");

    _colors.clear();
    _colorLUT = lutType;

    switch (lutType)
    {
        case CLUT_BW:
            _colors.push_back(SLColorLUTPoint(SLCol3f::BLACK, 0.0f));
            _colors.push_back(SLColorLUTPoint(SLCol3f::WHITE, 1.0f));
            name("Transfer Function: Color LUT: B-W");
            break;
        case CLUT_WB:
            _colors.push_back(SLColorLUTPoint(SLCol3f::WHITE, 0.0f));
            _colors.push_back(SLColorLUTPoint(SLCol3f::BLACK, 1.0f));
            name("Transfer Function: Color LUT: W-B");
            break;
        case CLUT_RYGCB:
            _colors.push_back(SLColorLUTPoint(SLCol3f::RED, 0.00f));
            _colors.push_back(SLColorLUTPoint(SLCol3f::YELLOW, 0.25f));
            _colors.push_back(SLColorLUTPoint(SLCol3f::GREEN, 0.50f));
            _colors.push_back(SLColorLUTPoint(SLCol3f::CYAN, 0.75f));
            _colors.push_back(SLColorLUTPoint(SLCol3f::BLUE, 1.00f));
            name("Transfer Function: Color LUT: R-Y-G-C-B");
            break;
        case CLUT_BCGYR:
            _colors.push_back(SLColorLUTPoint(SLCol3f::BLUE, 0.00f));
            _colors.push_back(SLColorLUTPoint(SLCol3f::CYAN, 0.25f));
            _colors.push_back(SLColorLUTPoint(SLCol3f::GREEN, 0.50f));
            _colors.push_back(SLColorLUTPoint(SLCol3f::YELLOW, 0.75f));
            _colors.push_back(SLColorLUTPoint(SLCol3f::RED, 1.00f));
            name("Transfer Function: Color LUT: B-C-G-Y-R");
            break;
        case CLUT_RYGCBK:
            _colors.push_back(SLColorLUTPoint(SLCol3f::RED, 0.00f));
            _colors.push_back(SLColorLUTPoint(SLCol3f::YELLOW, 0.20f));
            _colors.push_back(SLColorLUTPoint(SLCol3f::GREEN, 0.40f));
            _colors.push_back(SLColorLUTPoint(SLCol3f::CYAN, 0.60f));
            _colors.push_back(SLColorLUTPoint(SLCol3f::BLUE, 0.80f));
            _colors.push_back(SLColorLUTPoint(SLCol3f::BLACK, 1.00f));
            name("Transfer Function: Color LUT: R-Y-G-C-B-K");
            break;
        case CLUT_KBCGYR:
            _colors.push_back(SLColorLUTPoint(SLCol3f::BLACK, 0.00f));
            _colors.push_back(SLColorLUTPoint(SLCol3f::BLUE, 0.20f));
            _colors.push_back(SLColorLUTPoint(SLCol3f::CYAN, 0.40f));
            _colors.push_back(SLColorLUTPoint(SLCol3f::GREEN, 0.60f));
            _colors.push_back(SLColorLUTPoint(SLCol3f::YELLOW, 0.00f));
            _colors.push_back(SLColorLUTPoint(SLCol3f::RED, 1.00f));
            name("Transfer Function: Color LUT: K-B-C-G-Y-R");
            break;
        case CLUT_RYGCBM:
            _colors.push_back(SLColorLUTPoint(SLCol3f::RED, 0.00f));
            _colors.push_back(SLColorLUTPoint(SLCol3f::YELLOW, 0.20f));
            _colors.push_back(SLColorLUTPoint(SLCol3f::GREEN, 0.40f));
            _colors.push_back(SLColorLUTPoint(SLCol3f::CYAN, 0.60f));
            _colors.push_back(SLColorLUTPoint(SLCol3f::BLUE, 0.80f));
            _colors.push_back(SLColorLUTPoint(SLCol3f::MAGENTA, 1.00f));
            name("Transfer Function: Color LUT: R-Y-G-C-B-M");
            break;
        case CLUT_MBCGYR:
            _colors.push_back(SLColorLUTPoint(SLCol3f::MAGENTA, 0.00f));
            _colors.push_back(SLColorLUTPoint(SLCol3f::BLUE, 0.20f));
            _colors.push_back(SLColorLUTPoint(SLCol3f::CYAN, 0.40f));
            _colors.push_back(SLColorLUTPoint(SLCol3f::GREEN, 0.60f));
            _colors.push_back(SLColorLUTPoint(SLCol3f::YELLOW, 0.00f));
            _colors.push_back(SLColorLUTPoint(SLCol3f::RED, 1.00f));
            name("Transfer Function: Color LUT: M-B-C-G-Y-R");
            break;
        case CLUT_DAYLIGHT:
            // Daylight color ramp with white at noon in the middle
            _colors.push_back(SLColorLUTPoint(SLCol3f(1, 36.0f / 255.0f, 36.0f / 255.0f), 0.00f));
            _colors.push_back(SLColorLUTPoint(SLCol3f(1, 113.0f / 255.0f, 6.0f / 255.0f), 0.05f));
            _colors.push_back(SLColorLUTPoint(SLCol3f(1, 243.0f / 255.0f, 6.0f / 255.0f), 0.10f));
            _colors.push_back(SLColorLUTPoint(SLCol3f(1, 1, 245.0f / 255.0f), 0.20f));
            _colors.push_back(SLColorLUTPoint(SLCol3f(1, 1, 1), 1.00f));
            name("Color LUT: For daylight");
            break;
        default:
            SL_EXIT_MSG("SLTexColorLUT::colors: undefined color LUT.");
            break;
    }
}
//-----------------------------------------------------------------------------
//! Generates the full 256 value LUT as 1x256 RGBA texture
void SLTexColorLUT::generateTexture()
{
    assert(_length > 1);
    assert(!_colors.empty() &&
           "SLTexColorLUT::generateTexture: Not enough color values.");

    // Delete old data in case of regeneration
    deleteData();

    SLfloat delta = 1.0f / (SLfloat)_length;

    // Check and sort alpha values
    if (!_alphas.empty())
    {
        sort(_alphas.begin(),
             _alphas.end(),
             [](SLAlphaLUTPoint a, SLAlphaLUTPoint b)
             { return a.pos < b.pos; });

        // Check out of bounds position (0-1)
        if (_alphas.front().pos < 0.0f)
            SL_EXIT_MSG("SLTexColorLUT::generateTexture: Lower alpha pos below 0");
        if (_alphas.back().pos > 1.0f)
            SL_EXIT_MSG("SLTexColorLUT::generateTexture: Upper alpha pos above 1");
        if (_colors.front().pos < 0.0f)
            SL_EXIT_MSG("SLTexColorLUT::generateTexture: Lower color pos below 0");
        if (_colors.back().pos > 1.0f)
            SL_EXIT_MSG("SLTexColorLUT::generateTexture: Upper color pos above 1");

        // Add boundary node if they are not at position 0 or 1
        if (_alphas.front().pos > 0.0f)
            _alphas.insert(_alphas.begin(), SLAlphaLUTPoint(_alphas.front().alpha, 0.0f));
        if (_alphas.back().pos < 1.0f)
            _alphas.push_back(SLAlphaLUTPoint(_alphas.back().alpha, 1.0f));
        if (_colors.front().pos > 0.0f)
            _colors.insert(_colors.begin(), SLColorLUTPoint(_colors.front().color, 0.0f));
        if (_colors.back().pos < 1.0f)
            _colors.push_back(SLColorLUTPoint(_colors.back().color, 1.0f));

        // Check that the delta between positions is larger than delta
        for (SLuint a = 0; a < _alphas.size() - 1; ++a)
            if ((_alphas[a + 1].pos - _alphas[a].pos) < delta)
                SL_EXIT_MSG("SLTexColorLUT::generateTexture: Alpha position deltas to small.");

        // Clamp alpha values
        for (auto a : _alphas) a.alpha = Utils::clamp(a.alpha, 0.0f, 1.0f);
    }

    // Check and sort color values
    sort(_colors.begin(),
         _colors.end(),
         [](SLColorLUTPoint a, SLColorLUTPoint b)
         { return a.pos < b.pos; });

    // Check that the delta between positions is larger than delta
    for (SLuint c = 0; c < _colors.size() - 1; ++c)
        if ((_colors[c + 1].pos - _colors[c].pos) < delta)
            SL_EXIT_MSG("SLTexColorLUT::generateTexture: Color position deltas to small.");

    // Clamp all colors
    for (auto c : _colors) c.color.clampMinMax(0.0f, 1.0f);

    // Finally create color LUT vector by lerping color and alpha values
    SLuint  c      = 0;    // current color segment index
    SLfloat pos    = 0.0f; // overall position between 0-1
    SLfloat posC   = 0.0f; // position in color segment
    SLfloat deltaC = 1.0f / ((_colors[c + 1].pos - _colors[c].pos) / delta);

    SLVCol4f lut;
    lut.resize((SLuint)_length);

    // Interpolate color values
    for (SLuint i = 0; i < _length; ++i)
    {
        lut[i].r = Utils::lerp(posC, _colors[c].color.r, _colors[c + 1].color.r);
        lut[i].g = Utils::lerp(posC, _colors[c].color.g, _colors[c + 1].color.g);
        lut[i].b = Utils::lerp(posC, _colors[c].color.b, _colors[c + 1].color.b);

        pos += delta;
        posC += deltaC;

        if (pos > _colors[c + 1].pos && c < _colors.size() - 2)
        {
            c++;
            posC   = 0.0f;
            deltaC = 1.0f / ((_colors[c + 1].pos - _colors[c].pos) / delta);
        }
    }

    if (!_alphas.empty())
    {
        // Interpolate alpha value
        SLuint  a      = 0;    // current alpha segment index
        SLfloat posA   = 0.0f; // position in alpha segment
        SLfloat deltaA = 1.0f / ((_alphas[a + 1].pos - _alphas[a].pos) / delta);
        pos            = 0.0f;
        for (SLuint i = 0; i < _length; ++i)
        {
            lut[i].a = Utils::lerp(posA, _alphas[a].alpha, _alphas[a + 1].alpha);
            pos += delta;
            posA += deltaA;

            if (pos > _alphas[a + 1].pos && a < _alphas.size() - 2)
            {
                a++;
                posA   = 0.0f;
                deltaA = 1.0f / ((_alphas[a + 1].pos - _alphas[a].pos) / delta);
            }
        }
    }
    else
    {
        for (SLuint i = 0; i < _length; ++i)
            lut[i].a = 1.0f;
    }

    // Create 1 x length sized image from SLCol4f values
    load(lut);
    _width  = _images[0]->width();
    _height = _images[0]->height();
    _depth  = (SLint)_images.size();
}
//-----------------------------------------------------------------------------
//! Returns all alpha values of the transfer function as a float vector
SLVfloat SLTexColorLUT::allAlphas()
{
    SLVfloat allA;
    allA.resize(_length);

    for (SLuint i = 0; i < _length; ++i)
    {
        CVVec4f c4f = _images[0]->getPixeli((SLint)i, 0);
        allA[i]     = c4f[3];
    }

    return allA;
}
//-----------------------------------------------------------------------------
