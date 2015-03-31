//#############################################################################
//  File:      SLText.cpp
//  Author:    Marcus Hudritsch
//  Date:      July 2014
//  Codestyle: https://github.com/cpvrlab/SLProject/wiki/Coding-Style-Guidelines
//  Copyright: Marcus Hudritsch
//             This software is provide under the GNU General Public License
//             Please visit: http://opensource.org/licenses/GPL-3.0
//#############################################################################

#include <stdafx.h>           // precompiled headers
#ifdef SL_MEMLEAKDETECT       // set in SL.h for debug config only
#include <debug_new.h>        // memory leak detector
#endif

#include <SLText.h>
#include <SLScene.h>
#include <SLGLProgram.h>
#include <SLGLState.h>

//-----------------------------------------------------------------------------
/*! 
The ctor sets all members and translates to the min. position.
*/
SLText::SLText(SLstring text, SLTexFont* font, SLCol4f color, 
               SLfloat maxWidth, SLfloat lineHeightFactor) 
               : SLNode("Text_"+text.substr(0,10))
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
    if (_drawBits.get(SL_DB_HIDDEN) || !_stateGL->blend()) return;
   
    // create buffer object for text once
    if (!_bufP.id() && !_bufT.id() && !_bufI.id())
    {   _font->buildTextBuffers(&_bufP, &_bufT, &_bufI, _text, 
                                _maxW, _lineH);
    }
   
    // Enable & build font texture with active OpenGL context
    _font->bindActive();

    // Setup shader
    SLGLProgram* sp = SLScene::current->programs(FontTex);
    SLGLState* state = SLGLState::getInstance();
    sp->useProgram();
    sp->uniformMatrix4fv("u_mvpMatrix", 1,
                        (SLfloat*)state->mvpMatrix());
    sp->uniform4fv("u_textColor", 1, (float*)&_color);
    sp->uniform1i("u_texture0", 0);
   
    // bind buffers and draw 
    _bufP.bindAndEnableAttrib(sp->getAttribLocation("a_position"));
    _bufT.bindAndEnableAttrib(sp->getAttribLocation("a_texCoord"));
   
    _bufI.bindAndDrawElementsAs(SL_TRIANGLES, (SLint)_text.length()*2*3);
   
    _bufP.disableAttribArray();
    _bufT.disableAttribArray();

    // For debug purpose
    //sp = SLScene::current->programs(ColorUniform);
    //sp->useProgram();
    //sp->uniformMatrix4fv("u_mvpMatrix",1,(SLfloat*)state->mvpMatrix());
    //sp->uniform4fv("u_color", 1, (float*)&SLCol4f::GREEN);
    //_bufP.bindAndEnableAttrib(sp->getAttribLocation("a_position"));
    //_bufT.bindAndEnableAttrib(sp->getAttribLocation("a_texCoord"));
    //_bufI.bindAndDrawElementsAs(SL_LINES, _text.length()*2*3);
    //_bufP.disableAttribArray();
    //_bufT.disableAttribArray();
}
void SLText::drawMeshes(SLSceneView* sv)
{
    drawRec(sv);
}
//-----------------------------------------------------------------------------
/*! 
SLText::statsRec updates the statistics.
*/
void SLText::statsRec(SLNodeStats &stats)
{  
    stats.numBytes += (SLuint)sizeof(SLText); 
    stats.numBytes += (SLuint)_text.length();  
    stats.numNodes++;
    stats.numTriangles += (SLuint)_text.length()*2 + 2;
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
