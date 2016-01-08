//#############################################################################
//  File:      SLGLVertexArrayExt.cpp
//  Purpose:   Extension class with functions for quick line & point drawing
//  Author:    Marcus Hudritsch
//  Date:      January 2016
//  Codestyle: https://github.com/cpvrlab/SLProject/wiki/Coding-Style-Guidelines
//  Copyright: Marcus Hudritsch
//             This software is provide under the GNU General Public License
//             Please visit: http://opensource.org/licenses/GPL-3.0
//#############################################################################

#include <stdafx.h>           // precompiled headers
#ifdef SL_MEMLEAKDETECT       // set in SL.h for debug config only
#include <debug_new.h>        // memory leak detector
#endif

#include <SLGLVertexArrayExt.h>
#include <SLGLProgram.h>
#include <SLScene.h>

//-----------------------------------------------------------------------------
/*! Helper function that sets the vertex position attribute and generates or 
updates the vertex buffer from it. It is used together with the 
drawArrayAsColored function.
*/
void SLGLVertexArrayExt::generateVertexPos(SLuint numVertices,
                                           SLint elementSize,
                                           void* dataPointer)
{
    assert(dataPointer);
    assert(elementSize);
    assert(numVertices);
    
    SLGLProgram* sp = SLScene::current->programs(ColorUniform);
    sp->useProgram();
    SLint location = sp->getAttribLocation("a_position");
    
    if (location == -1)
        SL_EXIT_MSG("The position attribute has no variable location.");
    
    // Add attribute if it doesn't exist
    if (attribIndex(SL_POSITION) == -1)
    {   setAttrib(SL_POSITION, elementSize, location, dataPointer);
        generate(numVertices, SL_STATIC_DRAW, false);
    } else
        updateAttrib(SL_POSITION, elementSize, dataPointer);
}
//-----------------------------------------------------------------------------
/*! Draws the vertex positions as array with a specified primitive & color
*/
void SLGLVertexArrayExt::drawArrayAsColored(SLPrimitive primitiveType,
                                            SLCol4f color,
                                            SLfloat pointSize,
                                            SLuint  indexFirstVertex,
                                            SLuint  countVertices)
{   assert(_idVBOAttribs);
    assert(countVertices <= _numVertices);
   
    // Prepare shader
    SLMaterial::current = 0;
    SLGLProgram* sp = SLScene::current->programs(ColorUniform);
    SLGLState* state = SLGLState::getInstance();
    sp->useProgram();
    sp->uniformMatrix4fv("u_mvpMatrix", 1, (SLfloat*)state->mvpMatrix());
   
    // Set uniform color
    glUniform4fv(sp->getUniformLocation("u_color"), 1, (SLfloat*)&color);
   
    #ifndef SL_GLES2
    if (pointSize!=1.0f)
        if (primitiveType == SL_POINTS)
            glPointSize(pointSize);
    #endif
                
    ///////////////////////////////////////////////////////////
    drawArrayAs(primitiveType, indexFirstVertex, countVertices);
    ///////////////////////////////////////////////////////////
   
    #ifndef SL_GLES2
    if (pointSize!=1.0f)
        if (primitiveType == SL_POINTS)
            glPointSize(1.0f);
    #endif
    
    #ifdef _GLDEBUG
    GET_GL_ERROR;
    #endif
}
//-----------------------------------------------------------------------------
