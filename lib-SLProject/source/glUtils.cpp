//#############################################################################
//  File:      glUtils.cpp
//  Purpose:   General OpenGL utility functions for simple OpenGL demo apps
//  Author:    Marcus Hudritsch
//  Date:      July 2014
//  Copyright: Marcus Hudritsch
//             This software is provide under the GNU General Public License
//             Please visit: http://opensource.org/licenses/GPL-3.0
//#############################################################################

#include <stdafx.h>           // precompiled headers

#include "glUtils.h"
#include <SLImage.h>          // for image loading

#include <algorithm>
#include <numeric>
#include <dirent.h>           // opendir

std::vector<string> errors;   // global vector for errors used in getGLError    

//-----------------------------------------------------------------------------
void glUtils::printGLInfo()
{
	std::cout << "OpenGL Version  : " << glGetString(GL_VERSION) << std::endl;
	std::cout << "GLSL Version    : " << glGetString(GL_SHADING_LANGUAGE_VERSION) << std::endl;
	std::cout << "OpenGL Renderer : " << glGetString(GL_RENDERER) << std::endl;
	std::cout << "OpenGL Vendor   : " << glGetString(GL_VENDOR) << std::endl;
    /*
	std::cout << "OpenGL Extensions:" << std::endl;
	std::stringstream ss;
	int n;
	glGetIntegerv(GL_NUM_EXTENSIONS, &n);
	for (int i = 0; i < n; ++i)
	{
		ss << glGetStringi(GL_EXTENSIONS, i) << " ";
	}
	std::cout << ss.str() << std::endl;
    */
}

//-----------------------------------------------------------------------------
/*! 
loadShader loads the ASCII content of a shader file and returns it as a string.
If the file can not be opened an error message is sent to stdout before the app
exits with code 1.
*/
string glUtils::loadShader(string filename)
{  
    // Loader file and return it as a string
    fstream shaderFile(filename.c_str(), ios::in);
    if (shaderFile.is_open())
    {   std::stringstream buffer;
        buffer << shaderFile.rdbuf();
        return buffer.str();
    }  
    cout << "File open failed: " << filename.c_str() << endl;
    exit(1);
    return "";
} 
//-----------------------------------------------------------------------------
/*! 
buildShader load the shader file, creates an OpenGL shader object, compiles the 
source code and returns the handle to the internal shader object. If the 
compilation fails the compiler log is sent to the stdout before the app exits 
with code 1.
All shaders are written with the initial GLSL version 110 without version number
in the code and are therefore backwards compatible with the compatibility
profile from OpenGL 2.1 and OpenGL ES 2 that runs on most mobile devices.
To be upwards compatible some modification have to be done.
*/
GLuint glUtils::buildShader(string shaderFile, 
                            GLenum shaderType)
{  
    // Load shader file, create shader and compile it
    string source = loadShader(shaderFile);

    string verGLSL = glUtils::glSLVersionNO();
    string srcVersion = "#version " + verGLSL + "\n";

    // Replace "attribute" and "varying" that came in GLSL 310
    if (verGLSL > "120")
    {   if (shaderType == GL_VERTEX_SHADER)
        {   SLUtils::replaceString(source, "attribute", "in       ");
            SLUtils::replaceString(source, "varying", "out    ");
        }
        if (shaderType == GL_FRAGMENT_SHADER)
        {   SLUtils::replaceString(source, "varying", "in     ");
        }
    }

    // Replace "gl_FragColor" that was deprecated in GLSL 140 (OpenGL 3.1) by a custom out variable
    if (verGLSL > "130")
    {   if (shaderType == GL_FRAGMENT_SHADER)
        {   SLUtils::replaceString(source, "gl_FragColor", "fragColor");
            SLUtils::replaceString(source, "void main", "out vec4 fragColor; \n\nvoid main");
        }
    }

    // Replace deprecated texture functions
    if (verGLSL > "140")
    {   if (shaderType == GL_FRAGMENT_SHADER)
        {   SLUtils::replaceString(source, "texture1D", "texture");
            SLUtils::replaceString(source, "texture2D", "texture");
            SLUtils::replaceString(source, "texture3D", "texture");
            SLUtils::replaceString(source, "textureCube", "texture");
        }
    }

    // Prepend the GLSL version as the first statement in the shader code
    SLstring scrComplete = srcVersion + source;


    // Compile Shader code
    GLuint shaderHandle = glCreateShader(shaderType);
    const char* src = scrComplete.c_str();
    glShaderSource(shaderHandle, 1, &src, 0);
    glCompileShader(shaderHandle);
   
    // Check compile success
    GLint compileSuccess;
    glGetShaderiv(shaderHandle, GL_COMPILE_STATUS, &compileSuccess);
    if (compileSuccess == GL_FALSE) 
    {   GLchar log[1024];
        glGetShaderInfoLog(shaderHandle, sizeof(log), 0, &log[0]);
        cout << "**** Compile Error ****" << endl;
        cout << "In File: " << shaderFile.c_str() << endl;
        cout << log;
        exit(1);
    }

    GETGLERROR;

    return shaderHandle;
}
//-----------------------------------------------------------------------------
/*! 
buildProgram creates a program object, attaches the shaders, links them and 
returns the OpenGL handle of the program. If the linking fails the linker log 
is sent to the stdout before the app exits with code 1.
*/
GLuint glUtils::buildProgram(GLuint vertShaderID, 
                             GLuint fragShaderID)
{  
    // Create program, attach shaders and link them 
    GLuint programHandle = glCreateProgram();
    glAttachShader(programHandle, vertShaderID);
    glAttachShader(programHandle, fragShaderID);
    glLinkProgram(programHandle);
   
    // Check linker success
    GLint linkSuccess;
    glGetProgramiv(programHandle, GL_LINK_STATUS, &linkSuccess);
    if (linkSuccess == GL_FALSE) 
    {   GLchar log[256];
        glGetProgramInfoLog(programHandle, sizeof(log), 0, &log[0]);
        cout << "**** Link Error ****" << endl;
        cout << log;
        exit(1);
    }
    return programHandle;
}
//-----------------------------------------------------------------------------
/*! Generates a Vertex Buffer Object (VBO) and copies the data into the
buffer on the GPU. The targetTypeGL distincts between GL_ARRAY_BUFFER for 
attribute data and GL_ELEMENT_ARRAY_BUFFER for index data. The usageTypeGL 
distincts between GL_STREAM_DRAW, GL_STATIC_DRAW and GL_DYNAMIC_DRAW.
*/
void glUtils::buildVBO(GLuint& vboID,
                       void*   dataPointer, 
                       GLint   numElements, 
                       GLint   elementSizeBytes, 
                       GLuint  targetTypeGL,
                       GLuint  usageTypeGL)
{  
    // Delete, generates and activates the VBO
    if (vboID) glDeleteBuffers(1, &vboID);
    glGenBuffers(1, &vboID);
    glBindBuffer(targetTypeGL, vboID);
   
    // determine the buffer size in bytes
    int bufSize = numElements * elementSizeBytes;
   
    // Copy data to the VBO on the GPU. The data could be delete afterwards.
    glBufferData(targetTypeGL, bufSize, dataPointer, usageTypeGL);
}
//-----------------------------------------------------------------------------
/* Builds the OpenGL Vertex Array Object (VAO) with it associated vertex buffer 
objects. VAOs where introduces OpenGL 3.0 and reduce the overhead per draw call. 
All vertex attributes (e.g. position, normals, texture coords, etc.) are float
and are stored in one big VBO. We expect the data in interleaved order: First
3 floats for position, then optionally 3 floats for the normal and last and
optional 2 floats for a texture coordinate.
*/
void glUtils::buildVAO(GLuint& vaoID,
                       GLuint& vboIDVertices, 
                       GLuint& vboIDIndices,
                       void*   dataPointerVertices, 
                       GLint   numVertices,
                       GLint   sizeVertexBytes,
                       void*   dataPointerIndices,
                       GLint   numIndices,
                       GLuint  sizeIndexBytes,
                       GLint   shaderProgramID,
                       GLint   attributePositionLoc, 
                       GLint   attributeNormalLoc,
                       GLint   attributeTexCoordLoc)
{
    assert(shaderProgramID);
    assert(dataPointerVertices && numVertices);
    assert(dataPointerIndices && numIndices);
    assert(attributePositionLoc > -1);

    //1) Generate and bind OpenGL vertex array object
    if (vaoID) glDeleteVertexArrays(1, &vaoID);
    glGenVertexArrays(1, &vaoID);
    glBindVertexArray(vaoID);

    //2) Generate array buffer vbo for float vertices
    buildVBO(vboIDVertices, 
             dataPointerVertices, 
             numVertices,
             sizeVertexBytes, 
             GL_ARRAY_BUFFER, GL_STATIC_DRAW);

    //3) Generate element array buffer for indices
    buildVBO(vboIDIndices, 
             dataPointerIndices, 
             numIndices,
             sizeIndexBytes, 
             GL_ELEMENT_ARRAY_BUFFER, GL_STATIC_DRAW);

    // Tell OpenGL how to interpret the vertex buffer                                                                                             
    // We use an interleaved attribute layout:
    // With vertex position, normals and texture coordinates it would look like this:                                           
    //           |               Vertex 0                |               Vertex 1                |   
    // Attribs:  |   Position0  |    Normal0   |TexCoord0|   Position1  |    Normal1   |TexCoord1|   
    // Elements: | PX | PY | PZ | NX | NY | NZ | TX | TY | PX | PY | PZ | NX | NY | NZ | TX | TY |   
    // Bytes:    |#### #### ####|#### #### ####|#### ####|#### #### ####|#### #### ####|#### ####|    
    //           |                                       |                                            
    //           |<--------- sizeVertexBytes = 32 ------>|
    //           |<-------- offsetT = 24 ----->|
    //           |<offsetN = 12>|

    //4) Activate GLSL shader program
    glUseProgram(shaderProgramID);

    //5a) We always must have a position attribute
    glVertexAttribPointer(attributePositionLoc, 3, GL_FLOAT, GL_FALSE, sizeVertexBytes, 0);
    glEnableVertexAttribArray(attributePositionLoc);

    //5b) If we have normals they are the second attribute with 12 bytes offset
    if (attributeNormalLoc > -1)
    {   glVertexAttribPointer(attributeNormalLoc, 3, GL_FLOAT, GL_FALSE, sizeVertexBytes, (void*)12);
        glEnableVertexAttribArray(attributeNormalLoc);
    }

    //5c) If we have texture coords they are the third attribute with 24 bytes offset
    if (attributeTexCoordLoc > -1)
    {   glVertexAttribPointer(attributeTexCoordLoc, 2, GL_FLOAT, GL_FALSE, sizeVertexBytes, (void*)24);
        glEnableVertexAttribArray(attributeTexCoordLoc);
    }
}
//-----------------------------------------------------------------------------
/*!
buildTexture loads and build the OpenGL texture on the GPU. The loaded image
data in the client memory is deleted again. The parameters min_filter and
mag_filter set the minification and magnification. The wrapS and wrapT parameters
set the texture wrapping mode. See the GL spec.
*/
GLuint glUtils::buildTexture(string textureFile,
                             GLint min_filter,
                             GLint mag_filter,
                             GLint wrapS, 
                             GLint wrapT)
{  
    // load texture image
    SLImage img(textureFile);

    // check max. size
    GLint maxSize = 0;
    glGetIntegerv(GL_MAX_TEXTURE_SIZE, &maxSize);
    if (img.width()  > (GLuint)maxSize || img.height() > (GLuint)maxSize) 
    {   cout << "SLGLTexture::build: Texture height is too big." << endl;
        exit(0);
    }

    // generate texture name (= internal texture object)
    GLuint textureHandle;
    glGenTextures(1, &textureHandle);

    // bind the texture as the active one
    glBindTexture(GL_TEXTURE_2D, textureHandle);

    // apply minification & magnification filter
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, min_filter);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, mag_filter);
      
    // apply texture wrapping modes
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, wrapS);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, wrapT);

    // Copy image data to the GPU. The image can be delete afterwards
    glTexImage2D(GL_TEXTURE_2D,     // target texture type 1D, 2D or 3D
                 0,                 // Base level for mipmapped textures
                 img.format(),      // internal format: e.g. GL_RGBA, see spec.
                 img.width(),       // image width
                 img.height(),      // image height
                 0,                 // border pixels: must be 0
                 img.format(),      // data format: e.g. GL_RGBA, see spec. 
                 GL_UNSIGNED_BYTE,  // data type
                 (GLvoid*)img.data()); // image data pointer
   
   // generate the mipmap levels 
   if (min_filter>=GL_NEAREST_MIPMAP_NEAREST)
	{  glGenerateMipmap(GL_TEXTURE_2D);
   }
   
   return textureHandle;
}

//-----------------------------------------------------------------------------
GLuint glUtils::build3DTexture(const std::vector<std::string> &files,
                               int &x_extend,
                               int &y_extend,
                               int &z_extend,
							   GLint min_filter,
							   GLint mag_filter,
							   GLint wrapR,
							   GLint wrapS,
							   GLint wrapT,
							   const std::array<GLfloat, 4> &borderColor
							  )
{
	// check max. size
	GLint maxSize = 0;
	glGetIntegerv(GL_MAX_3D_TEXTURE_SIZE, &maxSize);

	//The checks takes up valuable runtime; only do it in debug builds
    assert(files.size() > 0);

    SLImage first(files.front());
    if (min(min((SLuint)files.size(), first.height()), first.width()) > maxSize)
    {
        std::cout << "glUtils: Texture is too big in at least one dimension."<< std::endl;
        exit(0);
    }

    int imageSize = first.width()*first.height()*first.bytesPerPixel();
    std::vector<unsigned char> buffer(imageSize*files.size());
    unsigned char *imageData = &buffer[0]; //Concatenate the image data in a new buffer
    for (auto &file : files)
    {
        SLImage image(file);
        assert(image.height() == first.height());
        assert(image.width()  == first.width());
        assert(image.format() == first.format());

        memcpy(imageData, image.data(), imageSize);
        imageData += imageSize;
    }

	// generate texture name (= internal texture object)
	GLuint textureHandle;
	glGenTextures(1, &textureHandle);

	// bind the texture as the active one
    glBindTexture(GL_TEXTURE_3D, textureHandle);

	// apply minification & magnification filter
	glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MAG_FILTER, mag_filter);
	glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MIN_FILTER, min_filter);

	// apply texture wrapping modes
	glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_S, wrapS);
	glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_T, wrapT);
	glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_R, wrapR);
	glTexParameterfv(GL_TEXTURE_3D, GL_TEXTURE_BORDER_COLOR, borderColor.data());

    x_extend = first.width();
    y_extend = first.height();
	z_extend = files.size();

	buffer.emplace_back(0);

    glTexImage3D(GL_TEXTURE_3D, //Copy the new buffer to the GPU
                 0, //Mipmap level,
                 first.format(), //Internal format
                 x_extend,
                 y_extend,
                 z_extend,
                 0, //Border
                 first.format(), //Format
                 GL_UNSIGNED_BYTE, //Data type
                 &buffer[0]
                );

    glBindTexture(GL_TEXTURE_3D, 0);
    GETGLERROR;

	return textureHandle;
}

//-----------------------------------------------------------------------------
void glUtils::getGLError(char* file, 
                         int line, 
                         bool quit)
{  
    #if defined(DEBUG) || defined(_DEBUG)
    GLenum err;
    if ((err = glGetError()) != GL_NO_ERROR) 
    {   string errStr;
        switch(err)
        {   case GL_INVALID_ENUM: 
                errStr = "GL_INVALID_ENUM"; break;
            case GL_INVALID_VALUE: 
                errStr = "GL_INVALID_VALUE"; break;
            case GL_INVALID_OPERATION: 
                errStr = "GL_INVALID_OPERATION"; break;
            case GL_INVALID_FRAMEBUFFER_OPERATION: 
                errStr = "GL_INVALID_FRAMEBUFFER_OPERATION"; break;
            case GL_OUT_OF_MEMORY: 
                errStr = "GL_OUT_OF_MEMORY"; break;
            default: 
                errStr = "Unknown error";
        }

        // Build error string as a concatenation of file, line & error
        char sLine[32];
        sprintf(sLine, "%d", line);

        string newErr(file);
        newErr += ": line:";
        newErr += sLine;
        newErr += ": ";
        newErr += errStr;

        // Check if error exists already
        bool errExists = std::find(errors.begin(), errors.end(), newErr)!=errors.end();
      
        // Only print
        if (!errExists)
        {   errors.push_back(newErr);
            fprintf(stderr, 
                    "OpenGL Error in %s, line %d: %s\n", 
                    file, line, errStr.c_str());
        }
      
        if (quit) 
        {  
            #ifdef SL_MEMLEAKDETECT       // set in SL.h for debug config only
            // turn off leak checks on forced exit
            //new_autocheck_flag = false;
            #endif
            exit(1);
        }
    }
    #endif
}
//-----------------------------------------------------------------------------
//! Returns a vector of sorted file names with path within a directory
SLVstring glUtils::getFileNamesInDir(SLstring dirName)
{
    SLVstring fileNames;
    DIR* dir;
    struct dirent *dirContent;
    int i=0;
    dir = opendir(dirName.c_str());

    if (dir)
    {   while ((dirContent = readdir(dir)) != NULL)
        {   i++;
            //cout << sizeof(dirent) << endl;
            //printf("%s",dirContent->d_name);
            SLstring name(dirContent->d_name);
            if(name != "." && name != "..")
                fileNames.push_back(dirName+"/"+name);
        }
        closedir(dir);
    }
    return fileNames;
}
//-----------------------------------------------------------------------------
//! Returns the OpenGL Shading Language version number as a string.
/*! The string returned by glGetString can contain additional vendor
information such as the build number and the brand name.
For the shading language string "Nvidia GLSL 4.5" the function returns "450"
*/
string glUtils::glSLVersionNO()
{
    string versionStr = SLstring((char*)glGetString(GL_SHADING_LANGUAGE_VERSION));
    size_t dotPos = versionStr.find(".");
    char NO[4];
    NO[0] = versionStr[dotPos - 1];
    NO[1] = versionStr[dotPos + 1];
    NO[2] = '0';
    NO[3] = 0;
    return SLstring(NO);
}
//-----------------------------------------------------------------------------
