//#############################################################################
//  File:      SLGLVertexArrayExt.cpp
//  Purpose:   Extension class with functions for quick line & point drawing
//  Date:      January 2016
//  Codestyle: https://github.com/cpvrlab/SLProject/wiki/SLProject-Coding-Style
//  Authors:   Marcus Hudritsch
//  License:   This software is provided under the GNU General Public License
//             Please visit: http://opensource.org/licenses/GPL-3.0
//#############################################################################

#include <SLGLState.h>
#include <SLGLProgram.h>
#include <SLGLVertexArrayExt.h>
#include <SLGLProgramManager.h>
#include <SLMaterial.h>

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

    SLGLProgram* sp = SLGLProgramManager::get(SP_colorUniform);
    sp->useProgram();
    SLint location = AT_position;

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
    SLGLProgram* sp      = SLGLProgramManager::get(SP_colorUniform);
    SLGLState*   stateGL = SLGLState::instance();
    sp->useProgram();
    sp->uniformMatrix4fv("u_mMatrix", 1, (SLfloat*)&stateGL->modelMatrix);
    sp->uniformMatrix4fv("u_vMatrix", 1, (SLfloat*)&stateGL->viewMatrix);
    sp->uniformMatrix4fv("u_pMatrix", 1, (SLfloat*)&stateGL->projectionMatrix);
    sp->uniform1f("u_oneOverGamma", 1.0f);
    stateGL->currentMaterial(nullptr);

    // Set uniform color
    glUniform4fv(sp->getUniformLocation("u_matDiff"), 1, (SLfloat*)&color);

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

    GET_GL_ERROR;
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
    SLGLProgram* sp      = SLGLProgramManager::get(SP_colorUniform);
    SLGLState*   stateGL = SLGLState::instance();
    sp->useProgram();
    sp->uniformMatrix4fv("u_mMatrix", 1, (SLfloat*)&stateGL->modelMatrix);
    sp->uniformMatrix4fv("u_vMatrix", 1, (SLfloat*)&stateGL->viewMatrix);
    sp->uniformMatrix4fv("u_pMatrix", 1, (SLfloat*)&stateGL->projectionMatrix);
    sp->uniform1f("u_oneOverGamma", 1.0f);
    stateGL->currentMaterial(nullptr);

    // Set uniform color
    glUniform4fv(sp->getUniformLocation("u_matDiff"), 1, (SLfloat*)&color);

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

    GET_GL_ERROR;
}
//-----------------------------------------------------------------------------
