//#############################################################################
//  File:      SLGLVertexArray.cpp
//  Purpose:   Wrapper around an OpenGL Vertex Array Objects
//  Date:      January 2016
//  Codestyle: https://github.com/cpvrlab/SLProject/wiki/SLProject-Coding-Style
//  Authors:   Marcus Hudritsch
//  License:   This software is provided under the GNU General Public License
//             Please visit: http://opensource.org/licenses/GPL-3.0
//#############################################################################

#include <SLGLState.h>
#include <SLGLProgram.h>
#include <SLGLVertexArray.h>
#include <SLScene.h>

//-----------------------------------------------------------------------------
SLuint SLGLVertexArray::totalDrawCalls          = 0;
SLuint SLGLVertexArray::totalPrimitivesRendered = 0;
//-----------------------------------------------------------------------------
/*! Constructor initializing with default values
 */
SLGLVertexArray::SLGLVertexArray()
{
    _vaoID = 0;
    _VBOf.clear();
    _idVBOIndices       = 0;
    _numIndicesElements = 0;
    _numIndicesEdges    = 0;
    _numVertices        = 0;
    _indexDataElements  = nullptr;
    _indexDataEdges     = nullptr;
}
//-----------------------------------------------------------------------------
/*! Deletes the OpenGL objects for the vertex array and the vertex buffer.
The vector _attribs with the attribute information is not cleared.
*/
void SLGLVertexArray::deleteGL()
{
    if (_vaoID)
        glDeleteVertexArrays(1, &_vaoID);
    _vaoID = 0;

    if (_VBOf.id())
        _VBOf.clear();

    if (_idVBOIndices)
    {
        glDeleteBuffers(1, &_idVBOIndices);
        _idVBOIndices = 0;
        SLGLVertexBuffer::totalBufferCount--;
        SLGLVertexBuffer::totalBufferSize -= _numIndicesElements * (SLuint)SLGLVertexBuffer::sizeOfType(_indexDataType);
    }
}
//-----------------------------------------------------------------------------
/*! Defines a vertex attribute for the later generation.
It must be of a specific SLVertexAttribType. Each attribute can appear only
once in an vertex array.
If all attributes of a vertex array have the same data pointer the data input
will be interpreted as an interleaved array. See example in SLGLOculus::init.
Be aware that the VBO for the attribute will not be generated until generate
is called. The data pointer must still be valid when SLGLVertexArray::generate
is called.
*/
void SLGLVertexArray::setAttrib(SLGLAttributeType type,
                                SLint             elementSize,
                                SLint             location,
                                void*             dataPointer,
                                SLGLBufferType    dataType)
{
    assert(dataPointer);
    assert(elementSize);

    if (type == AT_position && location == -1)
        SL_EXIT_MSG("The position attribute has no variable location.");

    if (_VBOf.attribIndex(type) >= 0)
        SL_EXIT_MSG("Attribute type already exists.");

    SLGLAttribute va;
    va.type            = type;
    va.elementSize     = elementSize;
    va.dataType        = dataType;
    va.dataPointer     = dataPointer;
    va.location        = location;
    va.bufferSizeBytes = 0;

    _VBOf.attribs().push_back(va);
}
//-----------------------------------------------------------------------------
/*! Defines the vertex indices for the element drawing. Without indices vertex
array can only be drawn with SLGLVertexArray::drawArrayAs.
Be aware that the VBO for the indices will not be generated until generate
is called. The data pointer must still be valid when generate is called.
*/
void SLGLVertexArray::setIndices(SLuint         numIndicesElements,
                                 SLGLBufferType indexDataType,
                                 void*          indexDataElements,
                                 SLuint         numIndicesEdges,
                                 void*          indexDataEdges)
{
    assert(numIndicesElements);
    assert(indexDataElements);

    if (indexDataType == BT_ushort && _numVertices > 65535)
        SL_EXIT_MSG("Index data type not sufficient.");
    if (indexDataType == BT_ubyte && _numVertices > 255)
        SL_EXIT_MSG("Index data type not sufficient.");

    _numIndicesElements = numIndicesElements;
    _indexDataElements  = indexDataElements;
    _indexDataType      = indexDataType;
    _numIndicesEdges    = numIndicesEdges;
    _indexDataEdges     = indexDataEdges;
}
//-----------------------------------------------------------------------------
/*! Updates the specified vertex attribute. This works only for sequential
attributes and not for interleaved attributes. This is used e.g. for meshes
with vertex skinning. See SLMesh::draw where we have joint attributes.
*/
void SLGLVertexArray::updateAttrib(SLGLAttributeType type,
                                   SLint             elementSize,
                                   void*             dataPointer)
{
    assert(dataPointer && "No data pointer passed");
    assert(elementSize > 0 && elementSize < 5 && "Element size invalid");

    // Get attribute index and check element size
    SLint indexf = _VBOf.attribIndex(type);
    if (indexf == -1)
        SL_EXIT_MSG("Attribute type does not exist in VAO.");

    if (!_vaoID)
        glGenVertexArrays(1, &_vaoID);
    glBindVertexArray(_vaoID);

    // update the appropriate VBO
    if (indexf > -1)
        _VBOf.updateAttrib(type, elementSize, dataPointer);

    glBindVertexArray(0);
    GET_GL_ERROR;
}
//-----------------------------------------------------------------------------
/*! Generates the OpenGL objects for the vertex array and the vertex buffer
 object. If the input data is an interleaved array (all attribute data pointer
 where identical) also the output buffer will be generated as an interleaved
 array. Vertex arrays with attributes that are updated can not be interleaved.
 Vertex attributes with separate arrays can generate an interleaved or a
 sequential vertex buffer.\n\n
 <PRE>
 \n Sequential attribute layout:
 \n           |          Positions          |           Normals           |     TexCoords     |
 \n Attribs:  |   Position0  |   Position1  |    Normal0   |    Normal1   |  UV1_0  |  UV1_1  |
 \n Elements: | PX | PY | PZ | PX | PY | PZ | NX | NY | NZ | NX | NY | NZ | TX | TY | TX | TY |
 \n Bytes:    |#### #### ####|#### #### ####|#### #### ####|#### #### ####|#### ####|#### ####|
 \n           |                             |                             |
 \n           |<------ offset Normals ----->|                             |
 \n           |<----------------------- offset UVs ---------------------->|
 \n
 \n Interleaved attribute layout:
 \n           |               Vertex 0                |               Vertex 1                |
 \n Attribs:  |   Position0  |    Normal0   |  UV1_0  |   Position1  |    Normal1   |  UV1_1  |
 \n Elements: | PX | PY | PZ | NX | NY | NZ | TX | TY | PX | PY | PZ | NX | NY | NZ | TX | TY |
 \n Bytes:    |#### #### ####|#### #### ####|#### ####|#### #### ####|#### #### ####|#### ####|
 \n           |              |              |         |
 \n           |<-offsetN=32->|              |         |
 \n           |<------- offsetUV=32 ------->|         |
 \n           |                                       |
 \n           |<---------- strideBytes=32 ----------->|
 </PRE>
 The VAO has no or one active index buffer. For drawArrayAs no indices are needed.
 For drawElementsAs the index buffer is used. For triangle meshes also hard edges
 are generated. Their indices are stored behind the indices of the triangles.
*/
void SLGLVertexArray::generate(SLuint          numVertices,
                               SLGLBufferUsage usage,
                               SLbool          outputInterleaved)
{
    assert(numVertices);

    // if buffers exist delete them first
    deleteGL();

    _numVertices = numVertices;

    // Generate and bind VAO
    glGenVertexArrays(1, &_vaoID);
    glBindVertexArray(_vaoID);

    ///////////////////////////////
    // Create Vertex Buffer Objects
    ///////////////////////////////

    // Generate the vertex buffer object for float attributes
    if (_VBOf.attribs().size())
        _VBOf.generate(numVertices, usage, outputInterleaved);

    /////////////////////////////////////////////////////////////////
    // Create Element Array Buffer for Indices for elements and edges
    /////////////////////////////////////////////////////////////////

    if (_numIndicesElements && _indexDataElements &&
        _numIndicesEdges && _indexDataEdges)
    {
        // create temp. buffer with both index arrays
        SLuint   typeSize  = SLGLVertexBuffer::sizeOfType(_indexDataType);
        SLuint   tmBufSize = (_numIndicesElements + _numIndicesEdges) * (SLuint)typeSize;
        SLubyte* tmpBuf    = new SLubyte[tmBufSize];
        memcpy(tmpBuf,
               _indexDataElements,
               _numIndicesElements * (SLuint)typeSize);
        memcpy(tmpBuf + _numIndicesElements * (SLuint)typeSize,
               _indexDataEdges,
               _numIndicesEdges * (SLuint)typeSize);

        glGenBuffers(1, &_idVBOIndices);
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, _idVBOIndices);
        glBufferData(GL_ELEMENT_ARRAY_BUFFER, tmBufSize, tmpBuf, GL_STATIC_DRAW);
        SLGLVertexBuffer::totalBufferCount++;
        SLGLVertexBuffer::totalBufferSize += tmBufSize;
        delete[] tmpBuf;
    }
    else if (_numIndicesElements && _indexDataElements) // for elements only
    {
        SLuint typeSize = SLGLVertexBuffer::sizeOfType(_indexDataType);
        glGenBuffers(1, &_idVBOIndices);
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, _idVBOIndices);
        glBufferData(GL_ELEMENT_ARRAY_BUFFER,
                     _numIndicesElements * (SLuint)typeSize,
                     _indexDataElements,
                     GL_STATIC_DRAW);
        SLGLVertexBuffer::totalBufferCount++;
        SLGLVertexBuffer::totalBufferSize += _numIndicesElements * (SLuint)typeSize;
    }

    glBindVertexArray(0);
    GET_GL_ERROR;
}
//-----------------------------------------------------------------------------
/*! Same as generate but with transform feedback */
void SLGLVertexArray::generateTF(SLuint          numVertices,
                                 SLGLBufferUsage usage,
                                 SLbool          outputInterleaved)
{
    assert(numVertices);

    // if buffers exist delete them first
    deleteGL();

    _numVertices = numVertices;

    // Generate TFO
    glGenTransformFeedbacks(1, &_tfoID);

    // Generate and bind VAO
    glGenVertexArrays(1, &_vaoID);
    glBindVertexArray(_vaoID);

    ///////////////////////////////
    // Create Vertex Buffer Objects
    ///////////////////////////////

    // Generate the vertex buffer object for float attributes
    if (_VBOf.attribs().size())
        _VBOf.generate(numVertices, usage, outputInterleaved);

    ///////////////////////////////
    // Bind transform feedback
    ///////////////////////////////

    glBindTransformFeedback(GL_TRANSFORM_FEEDBACK, _tfoID);
    glBindBufferBase(GL_TRANSFORM_FEEDBACK_BUFFER, 0, _VBOf.id());

    /////////////////////////////////////////////////////////////////
    // Create Element Array Buffer for Indices for elements and edges
    /////////////////////////////////////////////////////////////////

    if (_numIndicesElements && _indexDataElements &&
        _numIndicesEdges && _indexDataEdges)
    {
        // create temp. buffer with both index arrays
        SLuint   typeSize  = SLGLVertexBuffer::sizeOfType(_indexDataType);
        SLuint   tmBufSize = (_numIndicesElements + _numIndicesEdges) * (SLuint)typeSize;
        SLubyte* tmpBuf    = new SLubyte[tmBufSize];
        memcpy(tmpBuf,
               _indexDataElements,
               _numIndicesElements * (SLuint)typeSize);
        memcpy(tmpBuf + _numIndicesElements * (SLuint)typeSize,
               _indexDataEdges,
               _numIndicesEdges * (SLuint)typeSize);

        glGenBuffers(1, &_idVBOIndices);
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, _idVBOIndices);
        glBufferData(GL_ELEMENT_ARRAY_BUFFER, tmBufSize, tmpBuf, GL_STATIC_DRAW);
        SLGLVertexBuffer::totalBufferCount++;
        SLGLVertexBuffer::totalBufferSize += tmBufSize;
        delete[] tmpBuf;
    }
    else if (_numIndicesElements && _indexDataElements) // for elements only
    {
        SLuint typeSize = SLGLVertexBuffer::sizeOfType(_indexDataType);
        glGenBuffers(1, &_idVBOIndices);
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, _idVBOIndices);
        glBufferData(GL_ELEMENT_ARRAY_BUFFER,
                     _numIndicesElements * (SLuint)typeSize,
                     _indexDataElements,
                     GL_STATIC_DRAW);
        SLGLVertexBuffer::totalBufferCount++;
        SLGLVertexBuffer::totalBufferSize += _numIndicesElements * (SLuint)typeSize;
    }

    glBindVertexArray(0);
    GET_GL_ERROR;
}
//-----------------------------------------------------------------------------
/*! Discard the rendering because we just compute next position with the
 * transform feedback. We need to bind a transform feedback object but not the
 * same from this vao, because we want to read from one vao and write on another.
 */
void SLGLVertexArray::beginTF(SLuint tfoID)
{
    // Disable rendering
    glEnable(GL_RASTERIZER_DISCARD);

    // Bind the feedback object for the buffers to be drawn next
    glBindTransformFeedback(GL_TRANSFORM_FEEDBACK, tfoID);

    // Draw points from input buffer with transform feedback
    glBeginTransformFeedback(GL_POINTS);
}

//-----------------------------------------------------------------------------
/*! We activate back the rendering and stop the transform feedback.
 */
void SLGLVertexArray::endTF()
{
    // End transform feedback
    glEndTransformFeedback();

    // Enable rendering
    glDisable(GL_RASTERIZER_DISCARD);

    // Un-bind the feedback object.
    glBindTransformFeedback(GL_TRANSFORM_FEEDBACK, 0);
}
//-----------------------------------------------------------------------------
/*! Draws the vertex attributes as a specified primitive type by elements with
the indices from the index buffer defined in setIndices.
*/
void SLGLVertexArray::drawElementsAs(SLGLPrimitiveType primitiveType,
                                     SLuint            numIndexes,
                                     SLuint            indexOffset)
{
    assert(_numIndicesElements && _idVBOIndices && "No index VBO generated for VAO");

    // From OpenGL 3.0 on we have the OpenGL Vertex Arrays
    // Binding the VAO saves all the commands after the else (per draw call!)
    glBindVertexArray(_vaoID);
    GET_GL_ERROR;

    // Do the draw call with indices
    if (numIndexes == 0)
        numIndexes = _numIndicesElements;

    SLuint indexTypeSize = SLGLVertexBuffer::sizeOfType(_indexDataType);

    /////////////////////////////////////////////////////////////////////
    glDrawElements(primitiveType,
                   (SLsizei)numIndexes,
                   _indexDataType,
                   (void*)(size_t)(indexOffset * (SLuint)indexTypeSize));
    /////////////////////////////////////////////////////////////////////

    GET_GL_ERROR;
    totalDrawCalls++;
    switch (primitiveType)
    {
        case PT_triangles:
            totalPrimitivesRendered += (numIndexes / 3);
            break;
        case PT_lines:
            totalPrimitivesRendered += (numIndexes / 2);
            break;
        case PT_points:
            totalPrimitivesRendered += numIndexes;
            break;
        default: break;
    }

    glBindVertexArray(0);
    GET_GL_ERROR;
}
//-----------------------------------------------------------------------------
/*! Draws the vertex attributes as a specified primitive type as the vertices
are defined in the attribute arrays.
*/
void SLGLVertexArray::drawArrayAs(SLGLPrimitiveType primitiveType,
                                  SLint             firstVertex,
                                  SLsizei           countVertices)
{
    assert((_VBOf.id()) && "No VBO generated for VAO.");

    glBindVertexArray(_vaoID);

    if (countVertices == 0)
        countVertices = (SLsizei)_numVertices;

    ////////////////////////////////////////////////////////
    glDrawArrays(primitiveType, firstVertex, countVertices);
    ////////////////////////////////////////////////////////

    // Update statistics
    totalDrawCalls++;
    SLint numVertices = countVertices - firstVertex;
    switch (primitiveType)
    {
        case PT_triangles:
            totalPrimitivesRendered += (numVertices / 3);
            break;
        case PT_lines:
            totalPrimitivesRendered += (numVertices / 2);
            break;
        case PT_points:
            totalPrimitivesRendered += numVertices;
            break;
        default: break;
    }

    glBindVertexArray(0);

    GET_GL_ERROR;
}
//-----------------------------------------------------------------------------
/*! Draws the hard edges with the specified color.
 The VAO has no or one active index buffer. For drawArrayAs no indices are needed.
 For drawElementsAs the index buffer is used. For triangle meshes also hard edges
 are generated. Their indices are stored behind the indices of the triangles.
*/
void SLGLVertexArray::drawEdges(SLCol4f color,
                                SLfloat lineWidth)
{
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
    if (lineWidth != 1.0f)
        glLineWidth(lineWidth);
#endif

    //////////////////////////////////////////////////////////////////
    drawElementsAs(PT_lines, _numIndicesEdges, _numIndicesElements);
    //////////////////////////////////////////////////////////////////

#ifndef SL_GLES
    if (lineWidth != 1.0f)
        glPointSize(1.0f);
#endif

    GET_GL_ERROR;
}
//-----------------------------------------------------------------------------
