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

#include <stdafx.h> // Must be the 1st include followed by  an empty line

#ifdef SL_MEMLEAKDETECT    // set in SL.h for debug config only
#    include <debug_new.h> // memory leak detector
#endif

#include <SLApplication.h>
#include <SLGLProgram.h>
#include <SLGLVertexArrayExt.h>
#include <SLScene.h>

//-----------------------------------------------------------------------------
/*! Helper function that sets the vertex position attribute and generates or 
updates the vertex buffer from it. It is used together with the 
drawArrayAsColored function.
*/
void SLGLVertexArrayExt::generateVertexPos(SLuint numVertices,
                                           SLint  elementSize,
                                           void*  dataPointer)
{
    assert(dataPointer);
    assert(elementSize);
    assert(numVertices);

    SLGLProgram* sp = SLApplication::scene->programs(SP_colorUniform);
    sp->useProgram();
    SLint location = sp->getAttribLocation("a_position");

    if (location == -1)
        SL_EXIT_MSG("The position attribute has no variable location.");

    // Add attribute if it doesn't exist
    if (_VBOf.attribIndex(AT_position) == -1)
    {
        setAttrib(AT_position, elementSize, location, dataPointer);
        generate(numVertices, BU_static, false);
    }
    else
        updateAttrib(AT_position, elementSize, dataPointer);
}
//-----------------------------------------------------------------------------
/*! Draws the vertex positions as array with a specified primitive & color
*/
void SLGLVertexArrayExt::drawArrayAsColored(SLGLPrimitiveType primitiveType,
                                            SLCol4f           color,
                                            SLfloat           pointSize,
                                            SLuint            indexFirstVertex,
                                            SLuint            countVertices)
{
    assert(countVertices <= _numVertices);

    if (!_VBOf.id())
        SL_EXIT_MSG("No VBO generated for VAO in drawArrayAsColored.");

    // Prepare shader
    SLMaterial::current = nullptr;
    SLGLProgram* sp     = SLApplication::scene->programs(SP_colorUniform);
    SLGLState*   state  = SLGLState::getInstance();
    sp->useProgram();
    sp->uniformMatrix4fv("u_mvpMatrix", 1, (const SLfloat*)state->mvpMatrix());
    sp->uniform1f("u_oneOverGamma", 1.0f);

    // Set uniform color
    glUniform4fv(sp->getUniformLocation("u_color"), 1, (SLfloat*)&color);

#ifndef SL_GLES
    if (pointSize != 1.0f)
        if (primitiveType == PT_points)
            glPointSize(pointSize);
#endif

    //////////////////////////////////////
    drawArrayAs(primitiveType,
                (SLsizei)indexFirstVertex,
                (SLsizei)countVertices);
    //////////////////////////////////////

#ifndef SL_GLES
    if (pointSize != 1.0f)
        if (primitiveType == PT_points)
            glPointSize(1.0f);
#endif

#ifdef _GLDEBUG
    GET_GL_ERROR;
#endif
}
//-----------------------------------------------------------------------------
/*! Draws the vertex positions as array with a specified primitive & color
*/
void SLGLVertexArrayExt::drawElementAsColored(SLGLPrimitiveType primitiveType,
                                              SLCol4f           color,
                                              SLfloat           pointSize,
                                              SLuint            indexFirstVertex,
                                              SLuint            countVertices)
{
    assert(countVertices <= _numVertices);

    if (!_VBOf.id())
        SL_EXIT_MSG("No VBO generated for VAO in drawArrayAsColored.");

    // Prepare shader
    SLMaterial::current = nullptr;
    SLGLProgram* sp     = SLApplication::scene->programs(SP_colorUniform);
    SLGLState*   state  = SLGLState::getInstance();
    sp->useProgram();
    sp->uniformMatrix4fv("u_mvpMatrix", 1, (const SLfloat*)state->mvpMatrix());
    sp->uniform1f("u_oneOverGamma", 1.0f);

    // Set uniform color
    glUniform4fv(sp->getUniformLocation("u_color"), 1, (SLfloat*)&color);

#ifndef SL_GLES
    if (pointSize != 1.0f)
        if (primitiveType == PT_points)
            glPointSize(pointSize);
#endif

    ////////////////////////////////
    drawElementsAs(primitiveType,
                   indexFirstVertex,
                   countVertices);
    ////////////////////////////////

#ifndef SL_GLES
    if (pointSize != 1.0f)
        if (primitiveType == PT_points)
            glPointSize(1.0f);
#endif

#ifdef _GLDEBUG
    GET_GL_ERROR;
#endif
}
//-----------------------------------------------------------------------------
