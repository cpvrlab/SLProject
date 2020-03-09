//#############################################################################
//  File:      SLText.cpp
//  Author:    Marcus Hudritsch
//  Date:      July 2014
//  Codestyle: https://github.com/cpvrlab/SLProject/wiki/SLProject-Coding-Style
//  Copyright: Marcus Hudritsch
//             This software is provide under the GNU General Public License
//             Please visit: http://opensource.org/licenses/GPL-3.0
//#############################################################################

#include <stdafx.h> // Must be the 1st include followed by  an empty line

#ifdef SL_MEMLEAKDETECT    // set in SL.h for debug config only
#    include <debug_new.h> // memory leak detector
#endif

#include <SLApplication.h>
#include <SLScene.h>
#include <SLText.h>

//-----------------------------------------------------------------------------
/*! 
The ctor sets all members and translates to the min. position.
*/
SLText::SLText(SLstring   text,
               SLTexFont* font,
               SLCol4f    color,
               SLfloat    maxWidth,
               SLfloat    lineHeightFactor)
  : SLNode("Text_" + text.substr(0, 10))
{
    assert(font);
    _font  = font;
    _text  = text;
    _color = color;
    _maxW  = maxWidth;
    _lineH = lineHeightFactor;

    _aabb.hasAlpha(true);
}
//-----------------------------------------------------------------------------
/*! 
SLText::shapeDraw draws the text buffer objects
*/
void SLText::drawRec(SLSceneView* sv)
{
    if (_drawBits.get(SL_DB_HIDDEN) || !SLGLState::instance()->blend()) return;

    // create buffer object for text once
    if (!_vao.vaoID())
    {
        _font->buildTextBuffers(_vao, _text, _maxW, _lineH);
        _font->minFiler(SL_ANISOTROPY_MAX);
        _font->magFiler(GL_LINEAR);
    }
    // Enable & build font texture with active OpenGL context
    _font->bindActive();

    // Setup shader
    SLGLProgram* sp    = SLApplication::scene->programs(SP_fontTex);
    SLGLState*   state = SLGLState::instance();
    sp->useProgram();
    sp->uniformMatrix4fv("u_mvpMatrix", 1, (const SLfloat*)state->mvpMatrix());
    sp->uniform4fv("u_textColor", 1, (float*)&_color);
    sp->uniform1i("u_texture0", 0);

    _vao.drawElementsAs(PT_triangles, (SLuint)_text.length() * 2 * 3);
}
void SLText::drawMeshes(SLSceneView* sv)
{
    drawRec(sv);
}
//-----------------------------------------------------------------------------
/*! 
SLText::statsRec updates the statistics.
*/
void SLText::statsRec(SLNodeStats& stats)
{
    stats.numBytes += (SLuint)sizeof(SLText);
    stats.numBytes += (SLuint)_text.length();
    stats.numNodes++;
    stats.numTriangles += (SLuint)_text.length() * 2 + 2;
}
//-----------------------------------------------------------------------------
/*! 
SLText::buildAABB builds and returns the axis-aligned bounding box.
*/
SLAABBox& SLText::updateAABBRec()
{
    SLVec2f size = _font->calcTextSize(_text);

    // calculate min & max in object space
    SLVec3f minOS(0, 0, -0.01f);
    SLVec3f maxOS(size.x, size.y, 0.01f);

    // apply world matrix: this overwrites the AABB of the group
    _aabb.fromOStoWS(minOS, maxOS, updateAndGetWM());

    return _aabb;
}
//-----------------------------------------------------------------------------
