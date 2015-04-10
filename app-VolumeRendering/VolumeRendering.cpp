//#############################################################################
//  File:      VolumeRendering.cpp
//  Purpose:   Standalone volume rendering test application.
//  Date:      February 2014
//  Author:    Manuel Frischknecht
//  Codestyle: https://github.com/cpvrlab/SLProject/wiki/Coding-Style-Guidelines
//  Copyright: Marcus Hudritsch, Manuel Frischknecht
//             This software is provide under the GNU General Public License
//             Please visit: http://opensource.org/licenses/GPL-3.0
//#############################################################################

#include "stdafx.h"
#ifdef SL_MEMLEAKDETECT       // set in SL.h for debug config only
#include <debug_new.h>        // memory leak detector
#endif

#include "SL.h"        // Basic SL type definitions
#include "glUtils.h"   // Basics for OpenGL shaders, buffers & textures
#include "SLImage.h"   // Image class for image loading
#include "SLVec3.h"    // 3D vector class
#include "SLMat4.h"    // 4x4 matrix class

#include "../lib-SLExternal/glew/include/GL/glew.h"     // OpenGL headers
#include "../lib-SLExternal/glfw3/include/GLFW/glfw3.h" // GLFW GUI library

#include <iomanip>
#include <sstream>
#include <iostream>

//-----------------------------------------------------------------------------
GLFWwindow* window;             //!< The global glfw window handle

// GLobal application variables
SLVec3f _voxelScaling = { 1.0f, 1.0f, 1.0f };
SLMat4f _volumeRotationMatrix;  //!< 4x4 volume rotation matrix
SLMat4f _modelViewMatrix;       //!< 4x4 modelview matrix
SLMat4f _projectionMatrix;      //!< 4x4 projection matrix

SLint   _scrWidth;              //!< Window width at start up
SLint   _scrHeight;             //!< Window height at start up
SLfloat _scr2fbX;               //!< Factor from screen to framebuffer coords
SLfloat _scr2fbY;               //!< Factor from screen to framebuffer coords

//Cube geometry
GLuint  _cubeNumI = 0;          //!< Number of vertex indices
GLuint  _cubeVboV = 0;          //!< Handle for the vertex VBO on the GPU
GLuint  _cubeVboI = 0;          //!< Handle for the vertex index VBO on the GPU

//Slices geometry
SLint   _numQuads = 350;        //!< Number of quads (slices) used
GLuint  _quadNumI = 0;          //!< Number of vertex indices
GLuint  _quadVboV = 0;          //!< Handle for the vertex VBO on the GPU
GLuint  _quadVboI = 0;          //!< Handle for the vertex index VBO on the GPU

enum RenderMethod
{
	SAMPLING = 1 << 0,
	SIDDON   = 1 << 1,
	SLICING  = 1 << 2
};

enum DisplayMethod
{
	MAXIMUM_INTENSITY_PROJECTION = 1 << 3,
	ALPHA_BLENDING_TF_LUT        = 1 << 4,
	ALPHA_BLENDING_CUSTOM_TF_LUT = 1 << 5
};

DisplayMethod _displayMethod = MAXIMUM_INTENSITY_PROJECTION; //!< The display method in use
RenderMethod  _renderMethod = SAMPLING;                      //!< The render method in use
SLstring      _renderMethodDescription = "";                 //!< A description of the current render and display methods


float    _camZ;                     //!< z-distance of camera
float    _rotX, _rotY;              //!< rotation angles around x & y axis
int      _deltaX, _deltaY;          //!< delta mouse motion
int      _startX, _startY;          //!< x,y mouse start positions
int      _mouseX, _mouseY;          //!< current mouse position
bool     _mouseLeftDown;            //!< Flag if mouse is down
GLuint   _modifiers = 0;            //!< modifier bit flags
const GLuint NONE  = 0;             //!< constant for no modifier
const GLuint SHIFT = 0x00200000;    //!< constant for shift key modifier
const GLuint CTRL  = 0x00400000;    //!< constant for control key modifier
const GLuint ALT   = 0x00800000;    //!< constant for alt key modifier

// Sampling shaders and attributes/uniforms:

//Sampling w/ maximum intensity projection
GLuint   _mipSamplingVertShader = 0;
GLuint   _mipSamplingFragShader = 0;
GLuint   _mipSamplingProgram = 0;

GLint    _mipSamplingPos = 0;
GLint    _mipSamplingMVP = 0;
GLint    _mipSamplingEyePos  = 0;
GLint    _mipSamplingVolume = 0;
GLint    _mipSamplingVoxelScale  = 0;
GLint    _mipSamplingTextureSize = 0;

//Sampling w/ transfer function
GLuint   _tfSamplingVertShader = 0;
GLuint   _tfSamplingFragShader = 0;
GLuint   _tfSamplingProgram = 0;

GLint    _tfSamplingPos = 0;
GLint    _tfSamplingMVP = 0;
GLint    _tfSamplingEyePos = 0;
GLint    _tfSamplingVolume = 0;
GLint    _tfSamplingTfLut = 0;
GLint    _tfSamplingVoxelScale = 0;
GLint    _tfSamplingTextureSize = 0;


// Voxel walking shaders and attributes/uniforms:

//Voxel walking with maximum intensity projection
GLuint   _mipSiddonVertShader = 0;
GLuint   _mipSiddonFragShader  = 0;
GLuint   _mipSiddonProgram = 0;

GLint    _mipSiddonPos = 0;
GLint    _mipSiddonMVP = 0;
GLint    _mipSiddonEyePos = 0;
GLint    _mipSiddonVolume = 0;
GLint    _mipSiddonVoxelScale = 0;
GLint    _mipSiddonTextureSize = 0;

//Voxel walking with transfer function
GLuint   _tfSiddonVertShader = 0;
GLuint   _tfSiddonFragShader  = 0;
GLuint   _tfSiddonProgram = 0;

GLint    _tfSiddonPos = 0;
GLint    _tfSiddonMVP = 0;
GLint    _tfSiddonEyePos = 0;
GLint    _tfSiddonVolume = 0;
GLint    _tfSiddonTfLut = 0;
GLint    _tfSiddonVoxelScale = 0;
GLint    _tfSiddonTextureSize = 0;

//Slice shader and attributes/uniforms
GLuint   _sliceVertShader = 0;
GLuint   _sliceFragShader  = 0;
GLuint   _sliceProgram = 0;

GLint    _slicePos = 0;
GLint    _sliceMVP = 0;
GLint    _sliceVolumeRot = 0;
GLint    _sliceVolume = 0;
GLint    _sliceTfLut = 0;
GLint    _sliceVoxelScale = 0;
GLint    _sliceTextureSize = 0;

//Texture handles
GLuint    _volumeTexture = 0;           //!< OpenGL handle of the 3d volume texture
GLuint    _tfLutTexture = 0;            //!< OpenGL handle of the transform function LUT texture
std::array<SLCol4f, 256> _tfLutBuffer;  //!< The buffer used to generate the LUT
GLfloat _intensityFocus = 0.8f;         //!< The currently focused intensity (for the custom LUT)

//The size of the volume texture in use
int _volumeWidth = 0;
int _volumeHeight = 0;
int _volumeDepth = 0;

// Triangle with 3 vertex indices
struct Triangle
{   Triangle(GLuint i1=0, GLuint i2=0, GLuint i3=0)
    {   indices[0] = i1;
        indices[1] = i2;
        indices[2] = i3;
    }
    std::array<GLuint, 3> indices;
};

void compilePrograms()
{
    SLstring glslDir = "../app-VolumeRendering/";
    _mipSamplingVertShader = glUtils::buildShader(glslDir + "VolumeRenderingRayCast.vert", GL_VERTEX_SHADER);
    _mipSamplingFragShader = glUtils::buildShader(glslDir + "VolumeRenderingSampling_MIP.frag", GL_FRAGMENT_SHADER);
    _mipSamplingProgram    = glUtils::buildProgram(_mipSamplingVertShader,_mipSamplingFragShader);

    _tfSamplingVertShader  = glUtils::buildShader(glslDir + "VolumeRenderingRayCast.vert", GL_VERTEX_SHADER);
    _tfSamplingFragShader  = glUtils::buildShader(glslDir + "VolumeRenderingSampling_TF.frag", GL_FRAGMENT_SHADER);
    _tfSamplingProgram     = glUtils::buildProgram(_tfSamplingVertShader,_tfSamplingFragShader);

    _mipSiddonVertShader   = glUtils::buildShader(glslDir + "VolumeRenderingRayCast.vert", GL_VERTEX_SHADER);
    _mipSiddonFragShader   = glUtils::buildShader(glslDir + "VolumeRenderingSiddon_MIP.frag", GL_FRAGMENT_SHADER);
    _mipSiddonProgram      = glUtils::buildProgram(_mipSiddonVertShader,_mipSiddonFragShader);

    _tfSiddonVertShader    = glUtils::buildShader(glslDir + "VolumeRenderingRayCast.vert", GL_VERTEX_SHADER);
    _tfSiddonFragShader    = glUtils::buildShader(glslDir + "VolumeRenderingSiddon_TF.frag", GL_FRAGMENT_SHADER);
    _tfSiddonProgram       = glUtils::buildProgram(_tfSiddonVertShader,_tfSiddonFragShader);

    _sliceVertShader       = glUtils::buildShader(glslDir + "VolumeRenderingSlicing.vert", GL_VERTEX_SHADER);
    _sliceFragShader       = glUtils::buildShader(glslDir + "VolumeRenderingSlicing.frag", GL_FRAGMENT_SHADER);
    _sliceProgram          = glUtils::buildProgram(_sliceVertShader,_sliceFragShader);

    _mipSamplingPos        = glGetAttribLocation (_mipSamplingProgram, "a_position");
    _mipSamplingMVP        = glGetUniformLocation(_mipSamplingProgram, "u_mvpMatrix");
    _mipSamplingEyePos     = glGetUniformLocation(_mipSamplingProgram, "u_eyePosition");
    _mipSamplingVolume     = glGetUniformLocation(_mipSamplingProgram, "u_volume");
    _mipSamplingVoxelScale = glGetUniformLocation(_mipSamplingProgram, "u_voxelScale");
    _mipSamplingTextureSize= glGetUniformLocation(_mipSamplingProgram, "u_textureSize");

    _tfSamplingPos         = glGetAttribLocation (_tfSamplingProgram, "a_position");
    _tfSamplingMVP         = glGetUniformLocation(_tfSamplingProgram, "u_mvpMatrix");
    _tfSamplingEyePos      = glGetUniformLocation(_tfSamplingProgram, "u_eyePosition");
    _tfSamplingVolume      = glGetUniformLocation(_tfSamplingProgram, "u_volume");
    _tfSamplingTfLut       = glGetUniformLocation(_tfSamplingProgram, "u_TfLut");
    _tfSamplingVoxelScale  = glGetUniformLocation(_tfSamplingProgram, "u_voxelScale");
    _tfSamplingTextureSize = glGetUniformLocation(_tfSamplingProgram, "u_textureSize");

    _mipSiddonPos          = glGetAttribLocation (_mipSiddonProgram, "a_position");
    _mipSiddonMVP          = glGetUniformLocation(_mipSiddonProgram, "u_mvpMatrix");
    _mipSiddonEyePos       = glGetUniformLocation(_mipSiddonProgram, "u_eyePosition");
    _mipSiddonVolume       = glGetUniformLocation(_mipSiddonProgram, "u_volume");
    _mipSiddonVoxelScale   = glGetUniformLocation(_mipSiddonProgram, "u_voxelScale");
    _mipSiddonTextureSize  = glGetUniformLocation(_mipSiddonProgram, "u_textureSize");

    _tfSiddonPos           = glGetAttribLocation (_tfSiddonProgram, "a_position");
    _tfSiddonMVP           = glGetUniformLocation(_tfSiddonProgram, "u_mvpMatrix");
    _tfSiddonEyePos        = glGetUniformLocation(_tfSiddonProgram, "u_eyePosition");
    _tfSiddonVolume        = glGetUniformLocation(_tfSiddonProgram, "u_volume");
    _tfSiddonTfLut         = glGetUniformLocation(_tfSiddonProgram, "u_TfLut");
    _tfSiddonVoxelScale    = glGetUniformLocation(_tfSiddonProgram, "u_voxelScale");
    _tfSiddonTextureSize   = glGetUniformLocation(_tfSiddonProgram, "u_textureSize");

    _slicePos              = glGetAttribLocation (_sliceProgram, "a_position");
    _sliceMVP              = glGetUniformLocation(_sliceProgram, "u_mvpMatrix");
    _sliceVolumeRot        = glGetUniformLocation(_sliceProgram, "u_volumeRotationMatrix");
    _sliceVolume           = glGetUniformLocation(_sliceProgram, "u_volume");
    _sliceTfLut            = glGetUniformLocation(_sliceProgram, "u_TfLut");
    _sliceVoxelScale       = glGetUniformLocation(_sliceProgram, "u_voxelScale");
    _sliceTextureSize      = glGetUniformLocation(_sliceProgram, "u_textureSize");
}

void deletePrograms()
{
    glDeleteShader(_mipSamplingVertShader);
    glDeleteShader(_mipSamplingFragShader );
    glDeleteProgram(_mipSamplingProgram);

    glDeleteShader(_tfSamplingVertShader);
    glDeleteShader(_tfSamplingFragShader );
    glDeleteProgram(_tfSamplingProgram);

    glDeleteShader(_mipSiddonVertShader);
    glDeleteShader(_mipSiddonFragShader);
    glDeleteProgram(_mipSiddonProgram);

    glDeleteShader(_tfSiddonVertShader);
    glDeleteShader(_tfSiddonFragShader );
    glDeleteProgram(_tfSiddonProgram);
}

void buildSliceQuads()
{
    std::vector<SLVec3f> vertices;
    std::vector<Triangle> triangles;

    // The maximal length in the cube in any dimension is
    // reached when the cube is seen at a 45° angle.
    // Thus, the length of the enclosing bounding cube is
    // sqrt(1^2 + 1^2 + 1^2) = sqrt(3) in any direction.
    const float sqrt3 = sqrt(3.0f);

    for (int i = 0; i < _numQuads; ++i)
    {
        // add 4 verices of a quad
        SLfloat sliceZ = ((2.0f*i) / _numQuads - 1.0f) * sqrt3; // [-1 .. 1] * sqrt(3)
        vertices.push_back(SLVec3f(-sqrt3, -sqrt3, sliceZ));
        vertices.push_back(SLVec3f( sqrt3, -sqrt3, sliceZ));
        vertices.push_back(SLVec3f(-sqrt3,  sqrt3, sliceZ));
        vertices.push_back(SLVec3f( sqrt3,  sqrt3, sliceZ));

        // add 2 triangle indexes
        triangles.push_back(Triangle(i*4+0, i*4+1, i*4+2));
        triangles.push_back(Triangle(i*4+1, i*4+2, i*4+3));
    }

	_quadVboV = glUtils::buildVBO(vertices.data(),
                                  vertices.size(),
                                  1,
                                  sizeof(SLVec3f),
                                  GL_ARRAY_BUFFER,
                                  GL_STATIC_DRAW
                                 );
	_quadNumI = triangles.size() * 3;
    _quadVboI = glUtils::buildVBO(triangles.data(),
                                  triangles.size(),
                                  1,
                                  sizeof(Triangle),
                                  GL_ELEMENT_ARRAY_BUFFER,
                                  GL_STATIC_DRAW
                                 );
}

void destroyQuads()
{
	glDeleteBuffers(1, &_quadVboV);
	glDeleteBuffers(1, &_quadVboI);
}

void buildCube()
{
    std::array<SLVec3f, 8> vertices = {SLVec3f(-1,-1,-1),
                                       SLVec3f( 1,-1,-1),
                                       SLVec3f(-1, 1,-1),
                                       SLVec3f( 1, 1,-1),
                                       SLVec3f(-1,-1, 1),
                                       SLVec3f( 1,-1, 1),
                                       SLVec3f(-1, 1, 1),
                                       SLVec3f( 1, 1, 1)};

    _cubeVboV = glUtils::buildVBO(vertices.data(),
                                  vertices.size(),
                                  1,
                                  sizeof(SLVec3f),
                                  GL_ARRAY_BUFFER,
                                  GL_STATIC_DRAW
                                 );

    std::array<Triangle, 12> triangles =
    {   Triangle(1,0,2), Triangle(1,2,3),   //Back face
        Triangle(4,5,6), Triangle(5,7,6),   //Front face
        Triangle(0,4,2), Triangle(4,6,2),   //Left face
        Triangle(1,3,5), Triangle(5,3,7),   //Right face
        Triangle(0,1,5), Triangle(0,5,4),   //Bottom face
        Triangle(3,2,7), Triangle(2,6,7)    //Top face
    };


    _cubeNumI = triangles.size()*3;
    _cubeVboI = glUtils::buildVBO(triangles.data(),
                              triangles.size(),
                              1,
                              sizeof(Triangle),
                              GL_ELEMENT_ARRAY_BUFFER,
                              GL_STATIC_DRAW
                              );
    GET_GL_ERROR;
}

void drawSamplingMIP()
{
	_modelViewMatrix.multiply(_volumeRotationMatrix);

	// Build the combined modelview-projection matrix
	SLMat4f mvp(_projectionMatrix);
	mvp.multiply(_modelViewMatrix);

	SLVec4f eye = _modelViewMatrix.inverse()*SLVec4f(0.0f, 0.0f, 0.0f, 1.0f);

	glEnable(GL_CULL_FACE);
	glFrontFace(GL_CCW);
	glCullFace(GL_BACK);
	glEnable(GL_DEPTH_TEST);
	glClearDepth(1.0);
	glDepthFunc(GL_LESS);
    glUseProgram(_mipSamplingProgram);

	glBindFramebuffer(GL_FRAMEBUFFER, 0);
	glClearColor(0.0, 0.0, 0.0, 0.0);
	glClear(GL_COLOR_BUFFER_BIT);

	glEnable(GL_TEXTURE_3D);
	glActiveTexture(GL_TEXTURE0);
	glBindTexture(GL_TEXTURE_3D, _volumeTexture);
    glUniform1i(_mipSamplingVolume, 0);
	glDisable(GL_TEXTURE_3D);
		
    glUniformMatrix4fv(_mipSamplingMVP, 1, 0, (float*)&mvp);
    glUniform3fv(_mipSamplingEyePos , 1, (float*)&eye);
    glUniform3fv(_mipSamplingVoxelScale , 1, (float*)&_voxelScaling);

    SLVec3f size((float)_volumeWidth, (float)_volumeHeight, (float)_volumeDepth);
    glUniform3fv(_mipSamplingTextureSize, 1, (float*)&size);

	glBindBuffer(GL_ARRAY_BUFFER, _cubeVboV);
	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, _cubeVboI);

    glEnableVertexAttribArray(_mipSamplingPos);

    glVertexAttribPointer(_mipSamplingPos,
                          3,
                          GL_FLOAT,
                          GL_FALSE,
                          sizeof(SLVec3f),
                          0);

	glDrawElements(GL_TRIANGLES, _cubeNumI, GL_UNSIGNED_INT, 0);

    glDisableVertexAttribArray(_mipSamplingPos);

	glBindBuffer(GL_ARRAY_BUFFER, 0);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);
}

void drawSamplingTF()
{
	_modelViewMatrix.multiply(_volumeRotationMatrix);

	SLMat4f mvp(_projectionMatrix);
	mvp.multiply(_modelViewMatrix);

	SLVec4f eye = _modelViewMatrix.inverse()*SLVec4f(0.0f, 0.0f, 0.0f, 1.0f);

	glEnable(GL_CULL_FACE);
	glFrontFace(GL_CCW);
	glCullFace(GL_BACK);
	glEnable(GL_DEPTH_TEST);
	glClearDepth(1.0);
	glDepthFunc(GL_LESS);
	glUseProgram(_tfSamplingProgram);

	glBindFramebuffer(GL_FRAMEBUFFER, 0);
	glClearColor(0.0, 0.0, 0.0, 0.0);
	glClear(GL_COLOR_BUFFER_BIT);

	glEnable(GL_TEXTURE_3D);
	glActiveTexture(GL_TEXTURE0);
	glBindTexture(GL_TEXTURE_3D, _volumeTexture);
	glUniform1i(_tfSamplingVolume, 0);
	glDisable(GL_TEXTURE_3D);

	glEnable(GL_TEXTURE_1D);
	glActiveTexture(GL_TEXTURE0 + 1);
	glBindTexture(GL_TEXTURE_1D, _tfLutTexture);
	glUniform1i(_tfSamplingTfLut, 1);
	glDisable(GL_TEXTURE_1D);

	glUniformMatrix4fv(_tfSamplingMVP, 1, 0, (float*)&mvp);
    glUniform3fv(_tfSamplingEyePos, 1, (float*)&eye);
    glUniform3fv(_tfSamplingVoxelScale, 1, (float*)&_voxelScaling);

    SLVec3f size((float)_volumeWidth, (float)_volumeHeight, (float)_volumeDepth);
    glUniform3fv(_tfSamplingTextureSize, 1, (float*)&size);

	glBindBuffer(GL_ARRAY_BUFFER, _cubeVboV);
	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, _cubeVboI);

    glEnableVertexAttribArray(_tfSamplingPos);

    glVertexAttribPointer(_tfSamplingPos,
                          3,
                          GL_FLOAT,
                          GL_FALSE,
                          sizeof(SLVec3f),
                          0);

	glDrawElements(GL_TRIANGLES, _cubeNumI, GL_UNSIGNED_INT, 0);

    glDisableVertexAttribArray(_tfSamplingPos);

	glBindBuffer(GL_ARRAY_BUFFER, 0);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);
}

void drawSiddonMIP()
{
	_modelViewMatrix.multiply(_volumeRotationMatrix);

	SLMat4f mvp(_projectionMatrix);
	mvp.multiply(_modelViewMatrix);

	SLVec4f eye = _modelViewMatrix.inverse()*SLVec4f(0.0f, 0.0f, 0.0f, 1.0f);

	glEnable(GL_CULL_FACE);
	glFrontFace(GL_CCW);
	glCullFace(GL_BACK);

	glEnable(GL_DEPTH_TEST);
	glClearDepth(1.0);
	glDepthFunc(GL_LESS);

	glDisable(GL_BLEND);

	glUseProgram(_mipSiddonProgram);

	glClearColor(0.0, 0.0, 0.0, 0.0);
	glClear(GL_COLOR_BUFFER_BIT);

	glEnable(GL_TEXTURE_3D);
	glActiveTexture(GL_TEXTURE0);
	glBindTexture(GL_TEXTURE_3D, _volumeTexture);
	glUniform1i(_mipSiddonVolume, 0);
	glDisable(GL_TEXTURE_3D);

	glUniformMatrix4fv(_mipSiddonMVP, 1, 0, (float*)&mvp);
    glUniform3fv(_mipSiddonEyePos, 1, (float*)&eye);
    glUniform3fv(_mipSiddonVoxelScale, 1, (float*)&_voxelScaling);

    SLVec3f size((float)_volumeWidth, (float)_volumeHeight, (float)_volumeDepth);
    glUniform3fv(_mipSiddonTextureSize, 1, (float*)&size);

	glBindBuffer(GL_ARRAY_BUFFER, _cubeVboV);
	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, _cubeVboI);

    glEnableVertexAttribArray(_mipSiddonPos);

    glVertexAttribPointer(_mipSiddonPos,
                          3,
                          GL_FLOAT,
                          GL_FALSE,
                          sizeof(SLVec3f),
                          0);

	glDrawElements(GL_TRIANGLES, _cubeNumI, GL_UNSIGNED_INT, 0);

    glDisableVertexAttribArray(_mipSiddonPos);

	glBindBuffer(GL_ARRAY_BUFFER, 0);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);
}

void drawSiddonTF()
{
	_modelViewMatrix.multiply(_volumeRotationMatrix);

	SLMat4f mvp(_projectionMatrix);
	mvp.multiply(_modelViewMatrix);

	SLVec4f eye = _modelViewMatrix.inverse()*SLVec4f(0.0f, 0.0f, 0.0f, 1.0f);

	glEnable(GL_CULL_FACE);
	glFrontFace(GL_CCW);
	glCullFace(GL_BACK);
	glEnable(GL_DEPTH_TEST);
	glClearDepth(1.0);
	glDepthFunc(GL_LESS);
	glUseProgram(_tfSiddonProgram);

	glBindFramebuffer(GL_FRAMEBUFFER, 0);
	glClearColor(0.0, 0.0, 0.0, 0.0);
	glClear(GL_COLOR_BUFFER_BIT);

	glEnable(GL_TEXTURE_3D);
	glActiveTexture(GL_TEXTURE0);
	glBindTexture(GL_TEXTURE_3D, _volumeTexture);
	glUniform1i(_tfSiddonVolume, 0);
	glDisable(GL_TEXTURE_3D);

	glEnable(GL_TEXTURE_1D);
	glActiveTexture(GL_TEXTURE0 + 1);
	glBindTexture(GL_TEXTURE_1D, _tfLutTexture);
	glUniform1i(_tfSiddonTfLut, 1);
	glDisable(GL_TEXTURE_1D);

	glUniformMatrix4fv(_tfSiddonMVP, 1, 0, (float*)&mvp);
    glUniform3fv(_tfSiddonEyePos, 1, (float*)&eye);
    glUniform3fv(_tfSiddonVoxelScale, 1, (float*)&_voxelScaling);

    SLVec3f size((float)_volumeWidth, (float)_volumeHeight, (float)_volumeDepth);
    glUniform3fv(_tfSiddonTextureSize, 1, (float*)&size);

	glBindBuffer(GL_ARRAY_BUFFER, _cubeVboV);
	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, _cubeVboI);

    glEnableVertexAttribArray(_tfSiddonPos);

    glVertexAttribPointer(_tfSiddonPos,
                          3, GL_FLOAT, GL_FALSE,
                          sizeof(SLVec3f),
                          0);

	glDrawElements(GL_TRIANGLES, _cubeNumI, GL_UNSIGNED_INT, 0);

    glDisableVertexAttribArray(_tfSiddonPos);

	glBindBuffer(GL_ARRAY_BUFFER, 0);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);
}

void drawSlicesMIP()
{
	_volumeRotationMatrix.invert();
	SLMat4f mvp(_projectionMatrix);
	mvp.multiply(_modelViewMatrix);

	glClearColor(0.0f,0.0f,0.0f,0.0f);
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
	glEnable(GL_BLEND);

	glBlendEquation(GL_MAX); // MIP, Maximum Intensity Projection
	glBlendFunc(GL_ONE, GL_ONE);

	glDisable(GL_CULL_FACE);
	glDisable(GL_DEPTH_TEST);
	glUseProgram(_sliceProgram);

	glUniformMatrix4fv(_sliceMVP, 1, 0, (float*)&mvp);
    glUniformMatrix4fv(_sliceVolumeRot, 1, 0, (float*)&_volumeRotationMatrix);
    glUniform3fv(_sliceVoxelScale, 1, (float*)&_voxelScaling);

    SLVec3f size((float)_volumeWidth, (float)_volumeHeight, (float)_volumeDepth);
    glUniform3fv(_sliceTextureSize, 1, (float*)&size);

	glEnable(GL_TEXTURE_3D);
	glActiveTexture(GL_TEXTURE0);
	glBindTexture(GL_TEXTURE_3D, _volumeTexture);
	glUniform1i(_sliceVolume, 0);
	glDisable(GL_TEXTURE_3D);

	glEnable(GL_TEXTURE_1D);
	glActiveTexture(GL_TEXTURE0 + 1);
	glBindTexture(GL_TEXTURE_1D, _tfLutTexture);
	glUniform1i(_sliceTfLut, 1);
	glDisable(GL_TEXTURE_1D);

	glBindBuffer(GL_ARRAY_BUFFER, _quadVboV);
	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, _quadVboI);

    glEnableVertexAttribArray(_slicePos);
    glVertexAttribPointer(_slicePos,
		                  3, GL_FLOAT, GL_FALSE,
                          sizeof(SLVec3f),
                          0);

	glDrawElements(GL_TRIANGLES, _quadNumI, GL_UNSIGNED_INT, 0);
    glDisableVertexAttribArray(_slicePos);

	glBindBuffer(GL_ARRAY_BUFFER, 0);
	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);
	glEnable(GL_CULL_FACE);
}

void drawSlicesTF()
{
	_volumeRotationMatrix.invert();
	SLMat4f mvp(_projectionMatrix);
	mvp.multiply(_modelViewMatrix);
	
	glClearColor(0.0f,0.0f,0.0f,0.0f);
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
	glEnable(GL_BLEND);

	glBlendEquation(GL_FUNC_ADD);
	glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

	glDisable(GL_CULL_FACE);
	glDisable(GL_DEPTH_TEST);
	glUseProgram(_sliceProgram);

	glUniformMatrix4fv(_sliceMVP, 1, 0, (float*)&mvp);
    glUniformMatrix4fv(_sliceVolumeRot, 1, 0, (float*)&_volumeRotationMatrix);

    SLVec3f size((float)_volumeWidth, (float)_volumeHeight, (float)_volumeDepth);
    glUniform3fv(_sliceTextureSize, 1, (float*)&size);

	glEnable(GL_TEXTURE_3D);
	glActiveTexture(GL_TEXTURE0);
	glBindTexture(GL_TEXTURE_3D, _volumeTexture);
	glUniform1i(_sliceVolume, 0);
	glDisable(GL_TEXTURE_3D);

	glEnable(GL_TEXTURE_1D);
	glActiveTexture(GL_TEXTURE0 + 1);
	glBindTexture(GL_TEXTURE_1D, _tfLutTexture);
	glUniform1i(_sliceTfLut, 1);
	glDisable(GL_TEXTURE_1D);

	glBindBuffer(GL_ARRAY_BUFFER, _quadVboV);
	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, _quadVboI);

    glEnableVertexAttribArray(_slicePos);
    glVertexAttribPointer(_slicePos,
                          3, GL_FLOAT, GL_FALSE,
                          sizeof(SLVec3f),
                          0);

	glDrawElements(GL_TRIANGLES, _quadNumI, GL_UNSIGNED_INT, 0);
    glDisableVertexAttribArray(_slicePos);

	glBindBuffer(GL_ARRAY_BUFFER, 0);
	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);
	glEnable(GL_CULL_FACE);
}

void updateRenderMethodDescription()
{
	std::stringstream ss;

	switch (_renderMethod)
	{
        default:
            ss << "Sampling";
            break;
        case SIDDON:
            ss << "Voxel Walking";
            break;
        case SLICING:
            ss << "Slicing (" << _numQuads << " Slices)";
            break;
	}

	switch (_displayMethod)
	{
        default:
            ss << " (Maximum Intensity Projection)";
            break;
        case ALPHA_BLENDING_TF_LUT:
            ss << " (Alpha Blending w/ Default Transfer Function)";
            break;
        case ALPHA_BLENDING_CUSTOM_TF_LUT:
            ss << " (Alpha Blending w/ Custom Transfer Function)";
            break;
	}

	_renderMethodDescription = ss.str();
}

void applyLut()
{
	// bind the texture as the active one
	glBindTexture(GL_TEXTURE_1D, _tfLutTexture);

	// apply minification & magnification filter
	glTexParameteri(GL_TEXTURE_1D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
	glTexParameteri(GL_TEXTURE_1D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);

	// apply texture wrapping modes
	glTexParameteri(GL_TEXTURE_1D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
	GET_GL_ERROR;

	glTexImage1D(GL_TEXTURE_1D, //Copy the new buffer to the GPU
                 0, //Mipmap level,
                 GL_RGBA,
                 _tfLutBuffer.size(),
                 0, //Border
                 GL_RGBA, //Format
                 GL_FLOAT, //Data type
                 &_tfLutBuffer[0]
                );
	GET_GL_ERROR;

	glBindTexture(GL_TEXTURE_1D, 0);
	GET_GL_ERROR;
}

void buildMaxIntensityLut()
{ 
	// Generate a maximum intensity projection LUT that will result in white voxels that
	// are dimmed by their alpha values. This LUT is only used in the slicing rendering method,
	// because MIP for the other methods implies some major changes to the shaders (and thus the
	// LUT-part is stripped out for performance).
	int i = 0;
    for (SLCol4f &color : _tfLutBuffer)
	{
		float f = float(i++) / _tfLutBuffer.size();
        color.set(f, f, f, f);
	}
	applyLut();
}

void generateHeatmapLut()
{
	int i = 0;
	for (auto &color : _tfLutBuffer)
	{
		//Gradually move from blue to red (heatmap-like)
		float hue = fmod((4.0f*M_PI / 3.0f) //240°
                    * (1.0f - float(i) / _tfLutBuffer.size())
                    + 2.0f*M_PI, // + 360°
                    2.0f*M_PI //mod 360°
                    );
        color.hsva2rgba({hue, 0.5f, 1.0f, float(i) / _tfLutBuffer.size()});
		++i;
	}
}

void buildDefaultLut()
{
    generateHeatmapLut();

	//Generate new alpha values from a simple exponential equation that focuses on the
	//highest intensity value while keeping some basic transparency.
	int i = 0;
	for (auto &color : _tfLutBuffer)
	{
		float t = float(i++) / _tfLutBuffer.size();
        color.a = 0.6f*pow(2.5f, 10.0f*t - 10.0f);
	}

	applyLut();
}

//Gaussian PDF. See: http://stackoverflow.com/a/10848293
template <typename T>
inline T normal_pdf(T x, T m, T s)
{
	static const T inv_sqrt_2pi = (T)0.3989422804014327;
	T a = (x - m) / s;

	return inv_sqrt_2pi / s * (T)std::exp(-T(0.5) * a * a);
}

void updateFocusLut()
{
	//Generate new alpha values from a simple flipped parabola equation that focuses on
	//a specific intensity value while keeping some basic transparency. 
	int i = 0;

	static const float sigma = .05f;
	static const float base_alpha = 0.002f;

	float max = normal_pdf(_intensityFocus, _intensityFocus, sigma);
	for (auto &color : _tfLutBuffer)
	{
		float x = float(i++) / _tfLutBuffer.size();
		double gaussian = normal_pdf<float>(x, _intensityFocus, sigma);
        color.a = base_alpha + (1.0f - base_alpha)*(gaussian/max);
	}

	applyLut();
}

void buildFocusLut()
{
	generateHeatmapLut(); 
	updateFocusLut();
}

//-----------------------------------------------------------------------------
/*!
calcFPS determines the frame per second measurement by averaging 60 frames.
*/
float calcFPS(float deltaTime)
{
    const  SLint   FILTERSIZE = 60;
    static SLfloat frameTimes[FILTERSIZE];
    static SLuint  frameNo = 0;

    frameTimes[frameNo % FILTERSIZE] = deltaTime;
    float sumTime = 0.0f;
    for (SLuint i=0; i<FILTERSIZE; ++i) sumTime += frameTimes[i];
    frameNo++;
    float frameTimeSec = sumTime / (SLfloat)FILTERSIZE;
    float fps = 1 / frameTimeSec;

    return fps;
}
//-----------------------------------------------------------------------------
/*!
onInit initializes the global variables and builds the shader program. It
should be called after a window with a valid OpenGL context is present.
*/
void onInit()
{
    updateRenderMethodDescription();

    buildSliceQuads();
    GET_GL_ERROR;

    buildCube();
    GET_GL_ERROR;

    // backwards movement of the camera
    _camZ = -3.0f;

    // Mouse rotation paramters
    _rotX = 0;
    _rotY = 0;
    _deltaX = 0;
    _deltaY = 0;
    _mouseLeftDown = false;

    // Load, compile & link shaders
    compilePrograms();
    GET_GL_ERROR;

    // Set some OpenGL states
    glClearColor(0.0f, 0.0f, 0.0f, 1.0f);  // Set the background color
    glEnable(GL_DEPTH_TEST);            // Enables depth test
    glEnable(GL_CULL_FACE);             // Enables the culling of back faces
    GET_GL_ERROR;                       // Check for OpenGL errors
}
//-----------------------------------------------------------------------------
/*!
onClose is called when the user closes the window and can be used for proper
deallocation of resources.
*/
void onClose(GLFWwindow* window)
{
    // Delete shaders & programs on GPU
    deletePrograms();

    // Delete arrays & buffers on GPU
    glDeleteBuffers(1, &_cubeVboV);
    glDeleteBuffers(1, &_cubeVboI);

    glDeleteBuffers(1, &_quadVboV);
    glDeleteBuffers(1, &_quadVboI);
}
//-----------------------------------------------------------------------------
/*!
onPaint does all the rendering for one frame from scratch with OpenGL (in core
profile).
*/
bool onPaint()
{
    // Clear the color & depth buffer
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    // Start with identity every frame
    _modelViewMatrix.identity();

    // View transform: move the coordinate system away from the camera
    _modelViewMatrix.translate(0, 0, _camZ);

    // View transform: rotate the coordinate system increasingly by the mouse
    _volumeRotationMatrix.identity();
    _volumeRotationMatrix.rotate(_rotX + _deltaX, 1, 0, 0);
    _volumeRotationMatrix.rotate(_rotY + _deltaY, 0, 1, 0);

    switch (_renderMethod + _displayMethod)
    {   case SAMPLING + MAXIMUM_INTENSITY_PROJECTION:    drawSamplingMIP();	break;
        case SAMPLING + ALPHA_BLENDING_TF_LUT:
        case SAMPLING + ALPHA_BLENDING_CUSTOM_TF_LUT:    drawSamplingTF();	break;
        case SIDDON   + MAXIMUM_INTENSITY_PROJECTION:    drawSiddonMIP();	break;
        case SIDDON   + ALPHA_BLENDING_TF_LUT:
        case SIDDON   + ALPHA_BLENDING_CUSTOM_TF_LUT:    drawSamplingTF();	break;
        case SLICING  + MAXIMUM_INTENSITY_PROJECTION:    drawSlicesMIP();	break;
        case SLICING  + ALPHA_BLENDING_TF_LUT:
        case SLICING  + ALPHA_BLENDING_CUSTOM_TF_LUT:    drawSlicesTF();    break;
    }

    // Check for errors from time to time
    GET_GL_ERROR;

    // Fast copy the back buffer to the front buffer. This is OS dependent.
    glfwSwapBuffers(window);

    // Calculate frames per second
    char title[255];
    static float lastTimeSec = 0;
    float timeNowSec = (float)glfwGetTime();
    float fps = calcFPS(timeNowSec-lastTimeSec);

    sprintf(title, "VolumeRendering. Method: %s - FPS: %4.0f", _renderMethodDescription.c_str(), fps);
    glfwSetWindowTitle(window, title);
    lastTimeSec = timeNowSec;

    // Return true to get an immediate refresh
    return true;
}
//-----------------------------------------------------------------------------
/*!
onResize: Event handler called on the resize event of the window. This event
should called once before the onPaint event. Do everything that is dependent on
the size and ratio of the window.
*/
void onResize(GLFWwindow* window, int width, int height)
{
    double w = (double)width;
    double h = (double)height;

    // define the projection matrix
    _projectionMatrix.perspective(45, w/h, 0.01f, 10.0f);

    // define the viewport
    glViewport(0, 0, width, height);

    onPaint();
}
//-----------------------------------------------------------------------------
/*!
Mouse button down & release eventhandler starts and end mouse rotation
*/
void onMouseButton(GLFWwindow* window, int button, int action, int mods)
{
    SLint x = _mouseX;
    SLint y = _mouseY;

    _mouseLeftDown = (action==GLFW_PRESS);
    if (_mouseLeftDown)
    {   _startX = x;
        _startY = y;

        // Renders only the lines of a polygon during mouse moves
        if (button==GLFW_MOUSE_BUTTON_RIGHT)
            glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
    } else
    {   _rotX += _deltaX;
        _rotY += _deltaY;
        _deltaX = 0;
        _deltaY = 0;

        // Renders filled polygons
        if (button==GLFW_MOUSE_BUTTON_RIGHT)
            glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
    }
}
//-----------------------------------------------------------------------------
/*!
Mouse move eventhandler tracks the mouse delta since touch down (_deltaX/_deltaY)
*/
void onMouseMove(GLFWwindow* window, double x, double y)
{
    _mouseX  = (int)x;
    _mouseY  = (int)y;

    if (_mouseLeftDown)
    {   _deltaY = (int)x - _startX;
        _deltaX = (int)y - _startY;
        onPaint();
    }
}
//-----------------------------------------------------------------------------
/*!
Mouse wheel eventhandler that moves the camera foreward or backwards
*/
void onMouseWheel(GLFWwindow* window, double xscroll, double yscroll)
{
    if (_modifiers == NONE)
    {   _camZ += (SLfloat)SL_sign(yscroll)*0.1f;
        onPaint();
    }
    else if (_modifiers == SHIFT)
    {   if (_displayMethod == ALPHA_BLENDING_CUSTOM_TF_LUT)
        {
           _intensityFocus = min(max(_intensityFocus + (SLfloat)SL_sign(yscroll)*0.01f, 0.0f), 1.0f);
           updateFocusLut();
        }
    }
    else if (_modifiers == CTRL)
    {
        int minNumSlices = 10;
        int maxNumSlices = (int)ceil(sqrt(3)*max(max(_volumeWidth, _volumeHeight), _volumeDepth));
        _numQuads += int(SL_sign(yscroll)) * 10;
        _numQuads = max(min(_numQuads, maxNumSlices), minNumSlices);
        destroyQuads();
        buildSliceQuads();
        updateRenderMethodDescription();
    }
}
//-----------------------------------------------------------------------------
/*!
Key action eventhandler handles key down & release events
*/
void onKey(GLFWwindow* window, int GLFWKey, int scancode, int action, int mods)
{
    if (action==GLFW_PRESS)
    {
        switch (GLFWKey)
        {
            case GLFW_KEY_ESCAPE:
            onClose(window);
            glfwSetWindowShouldClose(window, GL_TRUE);
            break;
            case GLFW_KEY_LEFT_SHIFT:     _modifiers = _modifiers|SHIFT; break;
            case GLFW_KEY_RIGHT_SHIFT:    _modifiers = _modifiers|SHIFT; break;
            case GLFW_KEY_LEFT_CONTROL:   _modifiers = _modifiers|CTRL; break;
            case GLFW_KEY_RIGHT_CONTROL:  _modifiers = _modifiers|CTRL; break;
            case GLFW_KEY_LEFT_ALT:       _modifiers = _modifiers|ALT; break;
            case GLFW_KEY_RIGHT_ALT:      _modifiers = _modifiers|ALT; break;
        }
    } else
    if (action == GLFW_RELEASE)
    {
        switch (GLFWKey)
        {   case GLFW_KEY_LEFT_SHIFT:     _modifiers = _modifiers^SHIFT; break;
            case GLFW_KEY_RIGHT_SHIFT:    _modifiers = _modifiers^SHIFT; break;
            case GLFW_KEY_LEFT_CONTROL:   _modifiers = _modifiers^CTRL; break;
            case GLFW_KEY_RIGHT_CONTROL:  _modifiers = _modifiers^CTRL; break;
            case GLFW_KEY_LEFT_ALT:       _modifiers = _modifiers^ALT; break;
            case GLFW_KEY_RIGHT_ALT:      _modifiers = _modifiers^ALT; break;

            case GLFW_KEY_1:
                 _renderMethod = SAMPLING;
                 updateRenderMethodDescription();
                 break;
            case GLFW_KEY_2:
                 _renderMethod = SIDDON;
                 updateRenderMethodDescription();
                 break;
            case GLFW_KEY_3:
                 _renderMethod = SLICING;
                 updateRenderMethodDescription();
                 break;
            case GLFW_KEY_Q:
                 _displayMethod = MAXIMUM_INTENSITY_PROJECTION;
                 updateRenderMethodDescription();
                 buildMaxIntensityLut();
                 break;
            case GLFW_KEY_W:
                 _displayMethod = ALPHA_BLENDING_TF_LUT;
                 updateRenderMethodDescription();
                 buildDefaultLut();
                 break;
            case GLFW_KEY_E:
                 _displayMethod = ALPHA_BLENDING_CUSTOM_TF_LUT;
                 updateRenderMethodDescription();
                 buildFocusLut();
                 break;
        }
    }
}
//-----------------------------------------------------------------------------
/*!
Error callback handler for GLFW.
*/
void onGLFWError(int error, const char* description)
{
    fputs(description, stderr);
}
//-----------------------------------------------------------------------------
/*!
The C main procedure running the GLFW GUI application.
*/
int main()
{
    if (!glfwInit())
    {    fprintf(stderr, "Failed to initialize GLFW\n");
        exit(EXIT_FAILURE);
    }

    glfwSetErrorCallback(onGLFWError);

    // Enable fullscreen anti aliasing with 4 samples
    glfwWindowHint(GLFW_SAMPLES, 4);
    //glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    //glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 2);
    //glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);
    //glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

    _scrWidth = 640;
    _scrHeight = 480;

    window = glfwCreateWindow(_scrWidth, _scrHeight, "My Title", NULL, NULL);
    if (!window)
    {   glfwTerminate();
        exit(EXIT_FAILURE);
    }

    // Get the currenct GL context. After this you can call GL
    glfwMakeContextCurrent(window);

    // On some systems screen & framebuffer size are different
    // All commands in GLFW are in screen coords but rendering in GL is
    // in framebuffer coords
    SLint fbWidth, fbHeight;
    glfwGetFramebufferSize(window, &fbWidth, &fbHeight);
    _scr2fbX = (float)fbWidth  / (float)_scrWidth;
    _scr2fbY = (float)fbHeight / (float)_scrHeight;

    // Include OpenGL via GLEW
    GLenum err = glewInit();
    if (GLEW_OK != err)
    {  fprintf(stderr, "Error: %s\n", glewGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    glfwSetWindowTitle(window, "Volume Rendering Test Application");

    // Set number of monitor refreshes between 2 buffer swaps
    glfwSwapInterval(1);

    // Load and build 3D texture from multiple images of the same size
    _voxelScaling = { 1.0f, 1.0f, 1.0f };
    SLstring  path = "../_data/images/textures/3d/volumes/mri_head_front_to_back/";
    std::vector<std::string> files = glUtils::getFileNamesInDir(path);
    _volumeTexture = glUtils::build3DTexture(files,
                                             _volumeWidth,
                                             _volumeHeight,
                                             _volumeDepth,
                                             GL_LINEAR,
                                             GL_LINEAR,
                                             GL_CLAMP_TO_BORDER,
                                             GL_CLAMP_TO_BORDER,
                                             GL_CLAMP_TO_BORDER);
    glGenTextures(1, &_tfLutTexture);
    glBindTexture(GL_TEXTURE_1D, _tfLutTexture);
    glTexParameteri(GL_TEXTURE_1D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_1D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_1D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glBindTexture(GL_TEXTURE_1D, 0);

    buildMaxIntensityLut();

    GET_GL_ERROR;

    onInit();
    onResize(window, (SLint)(_scrWidth  * _scr2fbX),
                     (SLint)(_scrHeight * _scr2fbY));

    // Set GLFW callback functions
    glfwSetKeyCallback(window, onKey);
    glfwSetFramebufferSizeCallback(window, onResize);
    glfwSetMouseButtonCallback(window, onMouseButton);
    glfwSetCursorPosCallback(window, onMouseMove);
    glfwSetScrollCallback(window, onMouseWheel);
    glfwSetWindowCloseCallback(window, onClose);

    // Event loop
    while (!glfwWindowShouldClose(window))
    {
        // if no updated occured wait for the next event (power saving)
        if (!onPaint())
           glfwWaitEvents();
        else glfwPollEvents();
    }

    glfwDestroyWindow(window);
    glfwTerminate();
    exit(0);
}
//-----------------------------------------------------------------------------
