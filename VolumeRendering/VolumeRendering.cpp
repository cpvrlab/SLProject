//#############################################################################
//  File:      TextureMapping.cpp
//  Purpose:   Minimal core profile OpenGL application for ambient-diffuse-
//             specular lighting shaders with Textures.
//  Date:      February 2014
//  Copyright: 2002-2014 Marcus Hudritsch
//             This software is provide under the GNU General Public License
//             Please visit: http://opensource.org/licenses/GPL-3.0
//#############################################################################

#include "stdafx.h"

#include "SL.h"        // Basic SL type definitions
#include "SLMath.h"
#include "glUtils.h"   // Basics for OpenGL shaders, buffers & textures
#include "SLImage.h"   // Image class for image loading
#include "SLVec3.h"    // 3D vector class
#include "SLMat4.h"    // 4x4 matrix class

#include "../lib-SLExternal/glew/include/GL/glew.h"     // OpenGL headers
#include "../lib-SLExternal/glfw3/include/GLFW/glfw3.h" // GLFW GUI library

#include <iomanip>
#include <sstream>
#include <iostream>
#include <random>
#include <regex>

//-----------------------------------------------------------------------------
//! Struct defintion for vertex attributes
struct VertexPNT
{
   SLVec3f p;  // vertex position [x,y,z]
   SLVec3f n;  // vertex normal [x,y,z]
   SLVec2f t;  // vertex texture coord. [s,t]
};
//-----------------------------------------------------------------------------
GLFWwindow* window;                 //!< The global glfw window handle

// GLobal application variables
SLVec3f  _voxelScaling = { 1.0f, 1.0f, 1.0f };
SLMat4f  _volumeRotationMatrix;     //!< 4x4 volume rotation matrix
SLMat4f  _modelViewMatrix;          //!< 4x4 modelview matrix
SLMat4f  _projectionMatrix;         //!< 4x4 projection matrix

SLint     _scrWidth;                //!< Window width at start up
SLint     _scrHeight;               //!< Window height at start up
SLfloat   _scr2fbX;                 //!< Factor from screen to framebuffer coords
SLfloat   _scr2fbY;                 //!< Factor from screen to framebuffer coords

//Cube geometry
GLuint    _cubeNumI = 0; //!< Number of vertex indices
GLuint    _cubeVboV = 0; //!< Handle for the vertex VBO on the GPU
GLuint    _cubeVboI = 0; //!< Handle for the vertex index VBO on the GPU

//Slices geometry
int _numQuads = 350;   //!< Number of quads (slices) used
GLuint _quadNumI = 0;  //!< Number of vertex indices
GLuint _quadVboV = 0;  //!< Handle for the vertex VBO on the GPU
GLuint _quadVboI = 0;  //!< Handle for the vertex index VBO on the GPU

enum RenderMethod
{
	SAMPLING = 1 << 0,
	SIDDON = 1 << 1,
	SLICING = 1 << 2
};

enum DisplayMethod
{
	MAXIMUM_INTENSITY_PROJECTION = 1 << 3,
	ALPHA_BLENDING_TF_LUT = 1 << 4,
	ALPHA_BLENDING_CUSTOM_TF_LUT = 1 << 5
};

DisplayMethod _displayMethod = MAXIMUM_INTENSITY_PROJECTION; //!< The display method in use
RenderMethod  _renderMethod = SAMPLING;                      //!< The render method in use
std::string   _renderMethodDescription = "";                 //!< A description of the current render and display methods


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
GLuint   _mipSamplingVertexShader = 0;
GLuint   _mipSamplingFragmentShader = 0;
GLuint   _mipSamplingProgram = 0;
GLint    _mipSamplingPosition = 0;
GLint    _mipSamplingMVP = 0;
GLint    _mipSamplingEyePosition = 0;
GLint    _mipSamplingVolume = 0;
GLint    _mipSamplingVoxelScaling = 0;
GLint    _mipSamplingTextureSize = 0;

//Sampling w/ transfer function
GLuint   _tfSamplingVertexShader = 0;
GLuint   _tfSamplingFragmentShader = 0;
GLuint   _tfSamplingProgram = 0;
GLint    _tfSamplingPosition = 0;
GLint    _tfSamplingMVP = 0;
GLint    _tfSamplingEyePosition = 0;
GLint    _tfSamplingVolume = 0;
GLint    _tfSamplingTfLut = 0;
GLint    _tfSamplingVoxelScaling = 0;
GLint    _tfSamplingTextureSize = 0;


// Voxel walking shaders and attributes/uniforms:

//Voxel walking w/ maximum intensity projection
GLuint   _mipSiddonVertexShader = 0;
GLuint   _mipSiddonFragmentShader = 0;
GLuint   _mipSiddonProgram = 0;
GLint    _mipSiddonPosition = 0;
GLint    _mipSiddonMVP = 0;
GLint    _mipSiddonEyePosition = 0;
GLint    _mipSiddonVolume = 0;
GLint    _mipSiddonVoxelScaling = 0;
GLint    _mipSiddonTextureSize = 0;

//Voxel walking w/ transfer function
GLuint   _tfSiddonVertexShader = 0;
GLuint   _tfSiddonFragmentShader = 0;
GLuint   _tfSiddonProgram = 0;
GLint    _tfSiddonPosition = 0;
GLint    _tfSiddonMVP = 0;
GLint    _tfSiddonEyePosition = 0;
GLint    _tfSiddonVolume = 0;
GLint    _tfSiddonTfLut = 0;
GLint    _tfSiddonVoxelScaling = 0;
GLint    _tfSiddonTextureSize = 0;

//Slice shader and attributes/uniforms
GLuint   _sliceVertexShader = 0;
GLuint   _sliceFragmentShader = 0;
GLuint   _sliceProgram = 0;

GLint    _slicePosition = 0;
GLint    _sliceMVP = 0;
GLint    _sliceVolumeRotation = 0;
GLint    _sliceVolume = 0;
GLint    _sliceTfLut = 0;
GLint    _sliceVoxelScaling = 0;
GLint    _sliceTextureSize = 0;

//Texture handles
GLuint    _volumeTexture = 0;  //!< OpenGL handle of the 3d volume texture
GLuint    _tfLutTexture = 0;   //!< OpenGL handle of the transform function LUT texture
std::array<std::array<GLfloat, 4>, 256> _tfLutBuffer; //!< The buffer used to generate the LUT
GLfloat _intensityFocus = 0.8f;                       //!< The currently focused intensity (for the custom LUT)

//The size of the volume texture in use
int _volumeWidth = 0;
int _volumeHeight = 0;
int _volumeDepth = 0;

struct Triangle
{
    std::array<GLuint, 3> indices;
};

struct Vertex
{
    std::array<GLfloat,3> position;
};

enum Dimensions
{
    DIM_X = 0,
    DIM_Y = 1,
    DIM_Z = 2,

    RIGHT = (1 << DIM_X),
    TOP   = (1 << DIM_Y),
    FRONT = (1 << DIM_Z),

    LEFT   = !RIGHT,
    BOTTOM = !TOP,
    BACK   = !FRONT
};

void compileProgram(GLuint &vertexShader, std::string vfile,
                    GLuint &fragmentShader, std::string ffile,
                    GLuint &program)
{
    vertexShader = glUtils::buildShader(vfile, GL_VERTEX_SHADER);
    fragmentShader = glUtils::buildShader(ffile, GL_FRAGMENT_SHADER);
    program = glUtils::buildProgram(vertexShader,fragmentShader);
}

void getVariables(GLint program,
                  std::initializer_list<std::pair<GLint&,std::string>> attributes,
                  std::initializer_list<std::pair<GLint&,std::string>> uniforms)
{
    for (auto &entry: attributes)
        entry.first = glGetAttribLocation(program,entry.second.c_str());

    for (auto &entry: uniforms)
        entry.first = glGetUniformLocation(program,entry.second.c_str());
}

void compilePrograms()
{
    compileProgram(_mipSamplingVertexShader, "../lib-SLProject/source/oglsl/RayCastVolumeRendering.vert",
                   _mipSamplingFragmentShader, "../lib-SLProject/source/oglsl/SamplingVolumeRendering_MIP.frag",
                   _mipSamplingProgram);

    compileProgram(_tfSamplingVertexShader, "../lib-SLProject/source/oglsl/RayCastVolumeRendering.vert",
                   _tfSamplingFragmentShader, "../lib-SLProject/source/oglsl/SamplingVolumeRendering_TF.frag",
                   _tfSamplingProgram);

    compileProgram(_mipSiddonVertexShader, "../lib-SLProject/source/oglsl/RayCastVolumeRendering.vert",
                   _mipSiddonFragmentShader, "../lib-SLProject/source/oglsl/SiddonVolumeRendering_MIP.frag",
                   _mipSiddonProgram);

    compileProgram(_tfSiddonVertexShader, "../lib-SLProject/source/oglsl/RayCastVolumeRendering.vert",
                   _tfSiddonFragmentShader, "../lib-SLProject/source/oglsl/SiddonVolumeRendering_TF.frag",
                   _tfSiddonProgram);

	compileProgram(_sliceVertexShader,   "../lib-SLProject/source/oglsl/SliceVolumeRendering.vert",
                   _sliceFragmentShader, "../lib-SLProject/source/oglsl/SliceVolumeRendering.frag",
				   _sliceProgram);

    getVariables(_mipSamplingProgram, {
                        { _mipSamplingPosition, "a_position" },
                    }, {
                        { _mipSamplingMVP, "u_mvpMatrix" },
                        { _mipSamplingEyePosition, "u_eyePosition" },
						{ _mipSamplingVolume, "u_volume" },
                        { _mipSamplingVoxelScaling, "u_voxelScale" },
                        { _mipSamplingTextureSize, "u_textureSize" }
                    });

    getVariables(_tfSamplingProgram, {
                        { _tfSamplingPosition, "a_position" },
                    }, {
                        { _tfSamplingMVP, "u_mvpMatrix" },
                        { _tfSamplingEyePosition, "u_eyePosition" },
                        { _tfSamplingVolume, "u_volume" },
						{ _tfSamplingTfLut, "u_TfLut" },
                        { _tfSamplingVoxelScaling, "u_voxelScale" },
                        { _tfSamplingTextureSize, "u_textureSize" }
                    });

    getVariables(_mipSiddonProgram, {
                        { _mipSiddonPosition, "a_position" },
                    }, {
                        { _mipSiddonMVP, "u_mvpMatrix" },
                        { _mipSiddonEyePosition, "u_eyePosition" },
						{ _mipSiddonVolume, "u_volume" },
                        { _mipSiddonVoxelScaling, "u_voxelScale" },
                        { _mipSiddonTextureSize, "u_textureSize" }
                    });

    getVariables(_tfSiddonProgram, {
                        { _tfSiddonPosition, "a_position" },
                    }, {
                        { _tfSiddonMVP, "u_mvpMatrix" },
                        { _tfSiddonEyePosition, "u_eyePosition" },
                        { _tfSiddonVolume, "u_volume" },
						{ _tfSiddonTfLut, "u_TfLut" },
                        { _tfSiddonVoxelScaling, "u_voxelScale" },
                        { _tfSiddonTextureSize, "u_textureSize" }
					});

	getVariables(_sliceProgram, {
						{ _slicePosition, "a_position" },
					}, {
						{ _sliceMVP, "u_mvpMatrix" },
						{ _sliceVolumeRotation, "u_volumeRotationMatrix" },
						{ _sliceVolume, "u_volume" },
						{ _sliceTfLut, "u_TfLut" },
                        { _sliceVoxelScaling, "u_voxelScale" },
                        { _sliceTextureSize, "u_textureSize" }
					});
}

void deletePrograms()
{
    glDeleteShader(_mipSamplingVertexShader);
    glDeleteShader(_mipSamplingFragmentShader);
    glDeleteProgram(_mipSamplingProgram);

    glDeleteShader(_tfSamplingVertexShader);
    glDeleteShader(_tfSamplingFragmentShader);
    glDeleteProgram(_tfSamplingProgram);

    glDeleteShader(_mipSiddonVertexShader);
    glDeleteShader(_mipSiddonFragmentShader);
    glDeleteProgram(_mipSiddonProgram);

    glDeleteShader(_tfSiddonVertexShader);
    glDeleteShader(_tfSiddonFragmentShader);
    glDeleteProgram(_tfSiddonProgram);
}


void buildQuads()
{
	struct Quad
	{
		Quad(int slice, int slices)
		{
			//The maximal length in the cube in any dimension is reached when
			//the cube is seen at a 45° angle.
			//Thus, the length of the enclosing bounding cube is sqrt(1^2 + 1^2 + 1^2) = sqrt(3)
			//in any direction.
			const float length_factor = sqrt(3.0f);

			for (int i = 0; i < 4; ++i)
			{
				vertices[i] = Vertex{ 
					(i & RIGHT ? 1.0f : -1.0f) * length_factor,
					(i & TOP ? 1.0f : -1.0f) * length_factor,
					((2.0f*slice) / slices - 1.0f) * length_factor
				};
			}
			triangles[0] = { 0, 1, 2 };
			triangles[1] = { 1, 2, 3 };

			for (auto &t : triangles)
				for (auto &i : t.indices)
					i += 4 * slice; //Adjust to the "global" vertex number
		}

		std::array<Vertex, 4> vertices;
		std::array<Triangle, 2> triangles;
	};

	std::vector<Vertex> vertices;
	std::vector<Triangle> triangles;

	vertices.reserve(4 * _numQuads);
	triangles.reserve(2 * _numQuads);

	for (int i = 0; i < _numQuads; ++i)
	{
		Quad quad(i, _numQuads);
		for (auto &v : quad.vertices) vertices.push_back(v);
		for (auto &t : quad.triangles) triangles.push_back(t);
	}

	_quadVboV = glUtils::buildVBO(vertices.data(),
		vertices.size(),
		1,
		sizeof(Vertex),
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
    auto createVertex = [](int i) {
        return Vertex {
            { //Position
                i & RIGHT ? 1.0f : -1.0f,
                i & TOP   ? 1.0f : -1.0f,
                i & FRONT ? 1.0f : -1.0f,
            }
        };
    };

    std::array<Vertex, 8> vertices = {
        createVertex(0),
        createVertex(1),
        createVertex(2),
        createVertex(3),
        createVertex(4),
        createVertex(5),
        createVertex(6),
        createVertex(7)
    };

    _cubeVboV = glUtils::buildVBO(vertices.data(),
                              vertices.size(),
                              1,
                              sizeof(Vertex),
                              GL_ARRAY_BUFFER,
                              GL_STATIC_DRAW
                              );

    std::array<Triangle, 12> triangles = {
        //Back face
		Triangle{ RIGHT + BOTTOM + BACK, LEFT + BOTTOM + BACK, LEFT  + TOP + BACK },
		Triangle{ RIGHT + BOTTOM + BACK, LEFT + TOP    + BACK, RIGHT + TOP + BACK },

        //Front face
        Triangle{ LEFT + BOTTOM + FRONT,  RIGHT + BOTTOM + FRONT, LEFT + TOP + FRONT },
        Triangle{ RIGHT + BOTTOM + FRONT, RIGHT + TOP + FRONT,    LEFT + TOP + FRONT },

        //Left face
		Triangle{ LEFT + BOTTOM + BACK,  LEFT + BOTTOM + FRONT, LEFT + TOP + BACK },
		Triangle{ LEFT + BOTTOM + FRONT, LEFT + TOP    + FRONT, LEFT + TOP + BACK },

        //Right face
        Triangle{ RIGHT + BOTTOM + BACK,  RIGHT + TOP + BACK, RIGHT + BOTTOM + FRONT },
        Triangle{ RIGHT + BOTTOM + FRONT, RIGHT + TOP + BACK, RIGHT + TOP + FRONT },

        //Bottom face
        Triangle{ LEFT + BOTTOM + BACK, RIGHT + BOTTOM + BACK,  RIGHT + BOTTOM + FRONT },
		Triangle{ LEFT + BOTTOM + BACK, RIGHT + BOTTOM + FRONT, LEFT  + BOTTOM + FRONT, },

        //Top face
		Triangle{ RIGHT + TOP + BACK, LEFT + TOP + BACK,  RIGHT + TOP + FRONT },
        Triangle{ LEFT  + TOP + BACK, LEFT + TOP + FRONT, RIGHT + TOP + FRONT }
    };

    _cubeNumI = triangles.size()*3;
    _cubeVboI = glUtils::buildVBO(triangles.data(),
                              triangles.size(),
                              1,
                              sizeof(Triangle),
                              GL_ELEMENT_ARRAY_BUFFER,
                              GL_STATIC_DRAW
                              );
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
	glUniform3fv(_mipSamplingEyePosition, 1, (float*)&eye);
	glUniform3fv(_mipSamplingVoxelScaling, 1, (float*)&_voxelScaling);

    SLVec3f size(_volumeWidth, _volumeHeight, _volumeDepth);
    glUniform3fv(_mipSamplingTextureSize, 1, (float*)&size);

	glBindBuffer(GL_ARRAY_BUFFER, _cubeVboV);
	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, _cubeVboI);

    glEnableVertexAttribArray(_mipSamplingPosition);

    glVertexAttribPointer(_mipSamplingPosition,
		3, GL_FLOAT, GL_FALSE,
		sizeof(Vertex),
		(void*)offsetof(Vertex, position));

	glDrawElements(GL_TRIANGLES, _cubeNumI, GL_UNSIGNED_INT, 0);

    glDisableVertexAttribArray(_mipSamplingPosition);

	glBindBuffer(GL_ARRAY_BUFFER, 0);
	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);

	GET_GL_ERROR;
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
	glUniform3fv(_tfSamplingEyePosition, 1, (float*)&eye);
	glUniform3fv(_tfSamplingVoxelScaling, 1, (float*)&_voxelScaling);

    SLVec3f size(_volumeWidth, _volumeHeight, _volumeDepth);
    glUniform3fv(_tfSamplingTextureSize, 1, (float*)&size);

	glBindBuffer(GL_ARRAY_BUFFER, _cubeVboV);
	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, _cubeVboI);

	glEnableVertexAttribArray(_tfSamplingPosition);

	glVertexAttribPointer(_tfSamplingPosition,
		3, GL_FLOAT, GL_FALSE,
		sizeof(Vertex),
		(void*)offsetof(Vertex, position));

	glDrawElements(GL_TRIANGLES, _cubeNumI, GL_UNSIGNED_INT, 0);

	glDisableVertexAttribArray(_tfSamplingPosition);

	glBindBuffer(GL_ARRAY_BUFFER, 0);
	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);

	GET_GL_ERROR;
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
	glUniform3fv(_mipSiddonEyePosition, 1, (float*)&eye);
	glUniform3fv(_mipSiddonVoxelScaling, 1, (float*)&_voxelScaling);

    SLVec3f size(_volumeWidth, _volumeHeight, _volumeDepth);
    glUniform3fv(_mipSiddonTextureSize, 1, (float*)&size);

	glBindBuffer(GL_ARRAY_BUFFER, _cubeVboV);
	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, _cubeVboI);

	glEnableVertexAttribArray(_mipSiddonPosition);

	{
		glVertexAttribPointer(_mipSiddonPosition,
			3, GL_FLOAT, GL_FALSE,
			sizeof(Vertex),
			(void*)offsetof(Vertex, position));
	}

	glDrawElements(GL_TRIANGLES, _cubeNumI, GL_UNSIGNED_INT, 0);

	glDisableVertexAttribArray(_mipSiddonPosition);

	glBindBuffer(GL_ARRAY_BUFFER, 0);
	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);

	GET_GL_ERROR;
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
	glUniform3fv(_tfSiddonEyePosition, 1, (float*)&eye);
	glUniform3fv(_tfSiddonVoxelScaling, 1, (float*)&_voxelScaling);

    SLVec3f size(_volumeWidth, _volumeHeight, _volumeDepth);
    glUniform3fv(_tfSiddonTextureSize, 1, (float*)&size);

	glBindBuffer(GL_ARRAY_BUFFER, _cubeVboV);
	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, _cubeVboI);

	glEnableVertexAttribArray(_tfSiddonPosition);

	glVertexAttribPointer(_tfSiddonPosition,
		3, GL_FLOAT, GL_FALSE,
		sizeof(Vertex),
		(void*)offsetof(Vertex, position));

	glDrawElements(GL_TRIANGLES, _cubeNumI, GL_UNSIGNED_INT, 0);

	glDisableVertexAttribArray(_tfSiddonPosition);

	glBindBuffer(GL_ARRAY_BUFFER, 0);
	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);

	GET_GL_ERROR;
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
	glUniformMatrix4fv(_sliceVolumeRotation, 1, 0, (float*)&_volumeRotationMatrix);
	glUniform3fv(_sliceVoxelScaling, 1, (float*)&_voxelScaling);

    SLVec3f size(_volumeWidth, _volumeHeight, _volumeDepth);
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

	glEnableVertexAttribArray(_slicePosition);
	glVertexAttribPointer(_slicePosition,
		                  3, GL_FLOAT, GL_FALSE,
		                  sizeof(Vertex),
		                  (void*)offsetof(Vertex, position));

	glDrawElements(GL_TRIANGLES, _quadNumI, GL_UNSIGNED_INT, 0);
	glDisableVertexAttribArray(_slicePosition);

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
	glUniformMatrix4fv(_sliceVolumeRotation, 1, 0, (float*)&_volumeRotationMatrix);

    SLVec3f size(_volumeWidth, _volumeHeight, _volumeDepth);
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

	glEnableVertexAttribArray(_slicePosition);
	glVertexAttribPointer(_slicePosition,
		3, GL_FLOAT, GL_FALSE,
		sizeof(Vertex),
		(void*)offsetof(Vertex, position));

	glDrawElements(GL_TRIANGLES, _quadNumI, GL_UNSIGNED_INT, 0);
	glDisableVertexAttribArray(_slicePosition);

	glBindBuffer(GL_ARRAY_BUFFER, 0);
	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);
	glEnable(GL_CULL_FACE);
}

void updateRenderMethodDescription()
{
	std::stringstream ss;

	switch (_renderMethod)
	{
	default: //SAMPLING
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
	default: //MAXIMUM_INTENSITY_PROJECTION
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
template <class T>
T clip(const T& n, const T& lower, const T& upper) {
	return std::max(lower, std::min(n, upper));
}

typedef std::array<GLfloat, 4> Color;

//Conversion according to http://www.rapidtables.com/convert/color/hsv-to-rgb.htm
Color hsva2rgba(const Color &hsva)
{
	GLfloat h = fmod(fmod(hsva[0], 2 * SL_PI) + 2 * SL_PI, 2 * SL_PI); // 0° <= H <= 360°
	GLfloat s = clip(hsva[1], 0.0f, 1.0f);
	GLfloat v = clip(hsva[2], 0.0f, 1.0f);
	GLfloat a = clip(hsva[3], 0.0f, 1.0f);

	float c = v * s;
	float x = c * (1.0f - fabs(fmod(h*3.0f / M_PI, 2.0f) - 1.0f));
	float m = v - c;

	switch (int(floor(h*3.0f / SL_PI)))
	{
	case 0: //[0°,60°)
		return{ m + c, m + x, m, a };
	case 1: //[60°,120°)
		return{ m + x, m + c, m, a };
	case 2: //[120°,180°)
		return{ m, m + c, m + x, a };
	case 3: //[180°,240°)
		return{ m, m + x, m + c, a };
	case 4: //[240°,300°)
		return{ m + x, m, m + c, a };
	case 5: //[300°,360°)
		return{ m + c, m, m + x, a };
	}
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
	for (auto &color : _tfLutBuffer)
	{
		float f = float(i++) / _tfLutBuffer.size();
		for (auto &channel : color) channel = f;
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
		color = hsva2rgba({ hue, 0.5f, 1.0f, float(i) / _tfLutBuffer.size() });
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
		color[3] = 0.6f*pow(2.5f, 10.0f*t - 10.0f);
	}

	applyLut();
}

//Gaussian PDF. See: http://stackoverflow.com/a/10848293
template <typename T>
inline T normal_pdf(T x, T m, T s)
{
	static const T inv_sqrt_2pi = 0.3989422804014327;
	T a = (x - m) / s;

	return inv_sqrt_2pi / s * std::exp(-T(0.5) * a * a);
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
		color[3] = base_alpha + (1.0f - base_alpha)*(gaussian/max);
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
{  const  SLint   FILTERSIZE = 60;
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

   buildQuads();
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
   //_modelViewMatrix.rotate(_rotX + _deltaX, 1,0,0);
   //_modelViewMatrix.rotate(_rotY + _deltaY, 0,1,0);

   _volumeRotationMatrix.identity();
   _volumeRotationMatrix.rotate(_rotX + _deltaX, 1, 0, 0);
   _volumeRotationMatrix.rotate(_rotY + _deltaY, 0, 1, 0);

   switch (_renderMethod + _displayMethod)
   {
   case SAMPLING + MAXIMUM_INTENSITY_PROJECTION:
	   drawSamplingMIP();
	   break;
   case SAMPLING + ALPHA_BLENDING_TF_LUT:
   case SAMPLING + ALPHA_BLENDING_CUSTOM_TF_LUT:
	   drawSamplingTF();
	   break;
   case SIDDON + MAXIMUM_INTENSITY_PROJECTION:
	   drawSiddonMIP();
	   break;
   case SIDDON + ALPHA_BLENDING_TF_LUT:
   case SIDDON + ALPHA_BLENDING_CUSTOM_TF_LUT:
	   drawSamplingTF();
	   break;
   case SLICING + MAXIMUM_INTENSITY_PROJECTION:
	   drawSlicesMIP();
	   break;
   case SLICING + ALPHA_BLENDING_TF_LUT:
   case SLICING + ALPHA_BLENDING_CUSTOM_TF_LUT:
	   drawSlicesTF();
	   break;
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
   {  _startX = x;
      _startY = y;

      // Renders only the lines of a polygon during mouse moves
      if (button==GLFW_MOUSE_BUTTON_RIGHT)
         glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
   } else
   {  _rotX += _deltaX;
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
   {  _deltaY = (int)x - _startX;
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
   {
      _camZ += (SLfloat)SL_sign(yscroll)*0.1f;
      onPaint();
   }
   else if (_modifiers == SHIFT)
   {
	   if (_displayMethod == ALPHA_BLENDING_CUSTOM_TF_LUT)
	   {
		   _intensityFocus = std::min(std::max(_intensityFocus + (SLfloat)SL_sign(yscroll)*0.01f, 0.0f), 1.0f);
		   updateFocusLut();
	   }
   }
   else if (_modifiers == CTRL)
   {
	   int minNumSlices = 10;
	   int maxNumSlices = ceil(sqrt(3)*std::max(std::max(_volumeWidth, _volumeHeight), _volumeDepth));
	   _numQuads += int(SL_sign(yscroll)) * 10;
	   _numQuads = std::max(std::min(_numQuads, maxNumSlices), minNumSlices);
	   destroyQuads();
	   buildQuads();
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
   {  switch (GLFWKey)
      {  case GLFW_KEY_LEFT_SHIFT:     _modifiers = _modifiers^SHIFT; break;
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
   {  fprintf(stderr, "Failed to initialize GLFW\n");
      exit(EXIT_FAILURE);
   }

   glfwSetErrorCallback(onGLFWError);

   // Enable fullscreen anti aliasing with 4 samples
//   glfwWindowHint(GLFW_SAMPLES, 4);
//   glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
//   glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 2);
//   glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);
//   glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

   _scrWidth = 640;
   _scrHeight = 480;

   window = glfwCreateWindow(_scrWidth, _scrHeight, "My Title", NULL, NULL);
   if (!window)
   {  glfwTerminate();
      exit(EXIT_FAILURE);
   }

   // Get the currenct GL context. After this you can call GL
   glfwMakeContextCurrent(window);

   // On some systems screen & framebuffer size are different
   // All commands in GLFW are in screen coords but rendering in GL is
   // in framebuffer coords
   SLint fbWidth, fbHeight;
   glfwGetFramebufferSize(window, &fbWidth, &fbHeight);
   _scr2fbX = (float)fbWidth / (float)_scrWidth;
   _scr2fbY = (float)fbHeight / (float)_scrHeight;

   // Include OpenGL via GLEW
   GLenum err = glewInit();
   if (GLEW_OK != err)
   {  fprintf(stderr, "Error: %s\n", glewGetErrorString(err));
      exit(EXIT_FAILURE);
   }

   glfwSetWindowTitle(window, "SLProject Test Application");

   // Set number of monitor refreshes between 2 buffer swaps
   glfwSwapInterval(1);

   {
	   _voxelScaling = { 1.0f, 1.0f, 1.0f };
       const std::string path = "../_data/images/textures/3d/volumes/mri_head_front_to_back/";
       const int numFiles = 207;
       std::vector<std::string> files;
       files.reserve(numFiles);
       for (int i = 0; i < numFiles; ++i)
       {
            std::stringstream ss;
            ss << path
               << "i" << std::setw(4) << std::setfill('0') << i
               << "_0000b.png";

            files.emplace_back(ss.str());
       }

	   _volumeTexture = glUtils::build3DTexture(files,
		   _volumeWidth,
		   _volumeHeight,
		   _volumeDepth,
		   GL_LINEAR,
		   GL_LINEAR,
		   GL_CLAMP_TO_BORDER,
		   GL_CLAMP_TO_BORDER,
		   GL_CLAMP_TO_BORDER
		   );

	   glGenTextures(1, &_tfLutTexture);
	   glBindTexture(GL_TEXTURE_1D, _tfLutTexture);
	   glTexParameteri(GL_TEXTURE_1D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
	   glTexParameteri(GL_TEXTURE_1D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
	   glTexParameteri(GL_TEXTURE_1D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
	   glBindTexture(GL_TEXTURE_1D, 0);

	   buildMaxIntensityLut();
   }
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
