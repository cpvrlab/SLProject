//#############################################################################
//  File:      SLGLState.cpp
//  Purpose:   Singleton class implementation for global OpenGL replacement
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

#include <SLGLEnums.h>
#include <SLGLState.h>

//-----------------------------------------------------------------------------
SLGLState* SLGLState::instance = nullptr;
//-----------------------------------------------------------------------------
std::vector<string> errors;   // global vector for errors used in getGLError  
//-----------------------------------------------------------------------------
/*! Public static creator and getter function. Guarantees the the static 
instance is created only once. The constructor is therefore private.
*/
SLGLState* SLGLState::getInstance()     
{
    if(!instance)
    {   instance = new SLGLState();
        return instance;
    } else return instance;
}
//-----------------------------------------------------------------------------
/*! Public static destruction.
*/
void SLGLState::deleteInstance()     
{
    delete instance;
    instance = 0;
}
//-----------------------------------------------------------------------------
/*! Private constructor should be called only once for a singleton class.
*/
SLGLState::SLGLState()
{ 
    initAll();
}
//-----------------------------------------------------------------------------
/*! Initializes all states.
*/
void SLGLState::initAll()
{  
    viewMatrix.identity();
    modelViewMatrix.identity();
    projectionMatrix.identity();
    textureMatrix.identity();
   
    numLightsUsed = 0;
   
    for (SLint i=0; i<SL_MAX_LIGHTS; ++i)
    {   lightIsOn[i] = 0;
        lightPosWS[i] = SLVec4f(0,0,1,1);
        lightPosVS[i] = SLVec4f(0,0,1,1);
        lightAmbient[i] = SLCol4f::BLACK;
        lightDiffuse[i] = SLCol4f::BLACK;
        lightSpecular[i] = SLCol4f::BLACK;
        lightSpotDirWS[i] = SLVec3f(0,0,-1);
        lightSpotDirVS[i] = SLVec3f(0,0,-1);
        lightSpotCutoff[i] = 180.0f;
        lightSpotCosCut[i] = cos(SL_DEG2RAD*lightSpotCutoff[i]);
        lightSpotExp[i] = 1.0f;
        lightAtt[i].set(1.0f, 0.0f, 0.0f);
        lightDoAtt[i] = 0;
    }
   
    matAmbient     = SLCol4f::WHITE;
    matDiffuse     = SLCol4f::WHITE;      
    matSpecular    = SLCol4f::WHITE;     
    matEmissive    = SLCol4f::BLACK;     
    matShininess   = 100;
   
    fogIsOn = false;                 
    fogMode = GL_LINEAR;
    fogDensity = 0.2f;
    fogDistStart = 1.0f;
    fogDistEnd = 6.0f;
    fogColor = SLCol4f::BLACK;
   
    globalAmbientLight.set(0.2f,0.2f,0.2f,0.0f);
   
    _glVersion      = SLstring((char*)glGetString(GL_VERSION));
    _glVersionNO    = getGLVersionNO();
    _glVersionNOf   = (SLfloat)atof(_glVersionNO.c_str());
    _glVendor       = SLstring((char*)glGetString(GL_VENDOR));
    _glRenderer     = SLstring((char*)glGetString(GL_RENDERER));
    _glSLVersion    = SLstring((char*)glGetString(GL_SHADING_LANGUAGE_VERSION));
    _glSLVersionNO  = getSLVersionNO();
    _glIsES2        = (_glVersion.find("OpenGL ES 2")!=string::npos);
    _glIsES3        = (_glVersion.find("OpenGL ES 3")!=string::npos);

    // Get extensions
    #ifndef SL_GLES2
    if (_glVersionNOf > 3.0f)
    {   GLint n;
        glGetIntegerv(GL_NUM_EXTENSIONS, &n);
        for (int i = 0; i < n; i++)
            _glExtensions += SLstring((char*)glGetStringi(GL_EXTENSIONS, i)) + ", ";
    } else
    #endif
    {   const GLubyte* ext = glGetString(GL_EXTENSIONS);
        if (ext) _glExtensions = SLstring((char*)ext);
    }
   
    //initialize states a unset
    _blend = false;
    _cullFace = false;
    _depthTest = false;
    _depthMask = false;
    _multisample = false;
    _polygonLine = false;
    _polygonOffsetEnabled = false;
    _polygonOffsetFactor = -1.0f;
    _polygonOffsetUnits = -1.0f;
    _viewport.set(-1,-1,-1,-1);
    _clearColor.set(-1,-1,-1,-1);

    // Reset all cached states to an invalid state
    _programID = 0;
    _textureUnit = -1;
    _textureTarget = -1;
    _textureID = 0;
    _colorMaskR = -1;
    _colorMaskG = -1;
    _colorMaskB = -1;
    _colorMaskA = -1;

    _isInitialized = true;   
    
    glGetIntegerv(GL_SAMPLES, &_multiSampleSamples);

    /* After over 10 years of OpenGL experience I used once a texture that is
    not divisible by 4 and this caused distorted texture displays. By default
    OpenGL has a pixel alignment of 4 which means that all images must be
    divisible by 4! If you want to use textures of any size you have to set
    a pixel alignment of 1:*/
    glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
    glPixelStorei(GL_PACK_ALIGNMENT, 1);

    #ifndef SL_GLES2
    glEnable(GL_PROGRAM_POINT_SIZE);
    #endif

    #ifdef _GLDEBUG
    GET_GL_ERROR;
    #endif
}
//-----------------------------------------------------------------------------
/*! The destructor only empties the stacks
*/
SLGLState::~SLGLState()
{  
    _modelViewMatrixStack.clear();
}
//-----------------------------------------------------------------------------
/*! One time initialization
*/
void SLGLState::onInitialize(SLCol4f clearColor)
{  
    // Reset all internal states
    if (!_isInitialized) initAll();

    // enable depth_test
    glDepthFunc(GL_LESS);
    glEnable(GL_DEPTH_TEST);

    // set blend function for classic transparency
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA); 
     
    // set background color
    glClearColor(clearColor.r,
                 clearColor.g,
                 clearColor.b,
                 clearColor.a);
    
    #ifdef _GLDEBUG
    GET_GL_ERROR;
    #endif
}
//-----------------------------------------------------------------------------
/*! Builds the 4x4 inverse matrix from the modelview matrix.
*/
void SLGLState::buildInverseMatrix()
{  
    _invModelViewMatrix.setMatrix(modelViewMatrix);
    _invModelViewMatrix.invert();
}
//-----------------------------------------------------------------------------
/*! Builds the normal matrix by the inverse transposed modelview matrix. Only
the linear 3x3 submatrix of the modelview matrix with the rotation is inversed.
The inverse transposed could be ignored as long as we would only have rotation
and uniform scaling in the 3x3 submatrix.
*/
void SLGLState::buildNormalMatrix()
{
    _normalMatrix.setMatrix(modelViewMatrix.mat3());
    _normalMatrix.invert();
    _normalMatrix.transpose();
}
//-----------------------------------------------------------------------------
/*! Builds the 4x4 inverse matrix and the 3x3 normal matrix from the modelview 
matrix. If only the normal matrix is needed use the method buildNormalMatrix
because inverses only the 3x3 submatrix of the modelview matrix.
*/
void SLGLState::buildInverseAndNormalMatrix()
{  
    _invModelViewMatrix.setMatrix(modelViewMatrix);
    _invModelViewMatrix.invert();
    _normalMatrix.setMatrix(_invModelViewMatrix.mat3());
    _normalMatrix.transpose();
}
//-----------------------------------------------------------------------------
/*! Returns the combined modelview projection matrix
*/
const SLMat4f* SLGLState::mvpMatrix()
{
    _mvpMatrix.setMatrix(projectionMatrix);
    _mvpMatrix.multiply(modelViewMatrix);
    return &_mvpMatrix;
}
//-----------------------------------------------------------------------------
/*! Transforms the light position into the view space
*/
void SLGLState::calcLightPosVS(SLint nLights)
{
    assert(nLights>=0 && nLights<=SL_MAX_LIGHTS);
    for (SLint i=0; i<nLights; ++i)
        lightPosVS[i].set(viewMatrix * lightPosWS[i]);
}
//-----------------------------------------------------------------------------
/*! Transforms the lights spot direction into the view space
*/
void SLGLState::calcLightDirVS(SLint nLights)
{
    assert(nLights>=0 && nLights<=SL_MAX_LIGHTS);
    SLMat4f vRot(viewMatrix);
    vRot.translation(0,0,0); // delete translation part, only rotation needed
   
    for (SLint i=0; i<nLights; ++i)
        lightSpotDirVS[i].set(vRot.multVec(lightSpotDirWS[i]));
}
//-----------------------------------------------------------------------------
/*! Returns the global ambient color as the component wise product of the global
ambient light intensity and the materials ambient reflection. This is used to
give the scene a minimal ambient lighting.
*/
const SLCol4f* SLGLState::globalAmbient()
{
    _globalAmbient.set(globalAmbientLight & matAmbient);
    return &_globalAmbient;
}
//-----------------------------------------------------------------------------
void SLGLState::clearColor(SLCol4f newColor)
{
    if (_clearColor != newColor)
    {   glClearColor(newColor.r, newColor.g, newColor.b, newColor.a);
        _clearColor = newColor;
    
        #ifdef _GLDEBUG
        GET_GL_ERROR;
        #endif
    }
}
//-----------------------------------------------------------------------------
/*! SLGLState::depthTest enables or disables depth testing but only if the 
state really changes. The depth test decides for each pixel in the depth buffer 
which polygon is the closest to the eye.
*/
void SLGLState::depthTest(SLbool stateNew)
{
    if (_depthTest != stateNew)
    {   if (stateNew) glEnable(GL_DEPTH_TEST);
        else glDisable(GL_DEPTH_TEST);
        _depthTest = stateNew;
    
        #ifdef _GLDEBUG
        GET_GL_ERROR;
        #endif
    }
}
//-----------------------------------------------------------------------------
/*! SLGLState::depthTest enables or disables depth buffer writing but only if 
the state really changes. Turning on depth masking freezes the depth test but 
keeps all values in the depth buffer.
*/
void SLGLState::depthMask(SLbool stateNew)
{
    if (_depthMask != stateNew)
    {   glDepthMask(stateNew ? GL_TRUE : GL_FALSE);
        _depthMask = stateNew;
    
        #ifdef _GLDEBUG
        GET_GL_ERROR;
        #endif
    }
}
//-----------------------------------------------------------------------------
/*! SLGLState::cullFace sets the GL_CULL_FACE state but only if the state 
really changes. If face culling is turned on no back faces are processed. 
*/
void SLGLState::cullFace(SLbool stateNew)
{
    if (_cullFace != stateNew)
    {   if (stateNew) glEnable(GL_CULL_FACE);
        else glDisable(GL_CULL_FACE);
        _cullFace = stateNew;
    
        #ifdef _GLDEBUG
        GET_GL_ERROR;
        #endif
    }
}
//-----------------------------------------------------------------------------
/*! SLGLState::blend enables or disables alpha blending but only if the state 
really changes.
*/
void SLGLState::blend(SLbool stateNew)
{
    if (_blend != stateNew)
    {   if (stateNew) glEnable(GL_BLEND);
        else glDisable(GL_BLEND);
        _blend = stateNew;
    
        #ifdef _GLDEBUG
        GET_GL_ERROR;
        #endif
    }
}
//-----------------------------------------------------------------------------
/*! SLGLState::multiSample enables or disables multisampling but only if the 
state really changes. Multisampling turns on fullscreen anti aliasing on the GPU
witch produces smooth polygon edges, lines and points.
*/
void SLGLState::multiSample(SLbool stateNew)
{  
    #ifndef SL_GLES2
    if (_multisample != stateNew)
    {   
        if (_multiSampleSamples > 0)
        {
            #ifndef SL_GLES3
            if (stateNew) glEnable(GL_MULTISAMPLE);
            else glDisable(GL_MULTISAMPLE);
            _multisample = stateNew;
            #endif
        }
    
        #ifdef _GLDEBUG
        GET_GL_ERROR;
        #endif
    }
    #endif
}
//-----------------------------------------------------------------------------
/*! SLGLState::polygonMode sets the polygonMode to GL_LINE but only if the 
state really changes. OpenGL ES doesn't support glPolygonMode. It has to be 
mimicked with GL_LINE_LOOP drawing.
*/
void SLGLState::polygonLine(SLbool stateNew)
{  
    #ifndef SL_GLES2
    if (_polygonLine != stateNew)
    {
        #ifndef SL_GLES3
        if (stateNew)
             glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
        else glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
        _polygonLine = stateNew;
        #endif
    
        #ifdef _GLDEBUG
        GET_GL_ERROR;
        #endif
    }
    #endif
}
//-----------------------------------------------------------------------------
/*! SLGLState::polygonOffset turns on/off polygon offset (for filled polygons)
and sets the factor and unit for glPolygonOffset but only if the state really 
changes. Polygon offset is used to reduce z-fighting due to parallel planes or
lines.
*/
void SLGLState::polygonOffset(SLbool stateNew, SLfloat factor, SLfloat units)
{   
    if (_polygonOffsetEnabled != stateNew)
    {
        if (stateNew) 
        {   glEnable(GL_POLYGON_OFFSET_FILL);   
            if (_polygonOffsetFactor != factor || _polygonOffsetUnits != units)
            {   glPolygonOffset(factor, units);
                _polygonOffsetFactor = factor;
                _polygonOffsetUnits = units;
            }
        }
        else glDisable(GL_POLYGON_OFFSET_FILL);
        _polygonOffsetEnabled = stateNew;
    
        #ifdef _GLDEBUG
        GET_GL_ERROR;
        #endif
    }
}
//-----------------------------------------------------------------------------
/*! SLGLState::viewport sets the OpenGL viewport position and size 
*/
void SLGLState::viewport(SLint x, SLint y, SLsizei width, SLsizei height)
{

    if (_viewport.x!=x || _viewport.y!=y || _viewport.z!=width || _viewport.w!=height)
    {   glViewport(x, y, width, height);
        _viewport.set(x, y, width, height);
    
        #ifdef _GLDEBUG
        GET_GL_ERROR;
        #endif
    }
}
//-----------------------------------------------------------------------------
/*! SLGLState::colorMask sets the OpenGL colorMask for framebuffer masking 
*/
void SLGLState::colorMask(SLint r, SLint g, SLint b, SLint a)
{
    if (r != _colorMaskR || g != _colorMaskG || b != _colorMaskB || a != _colorMaskA)
    {   glColorMask(r, g, b, a);
        _colorMaskR = r;
        _colorMaskG = g;
        _colorMaskB = b;
        _colorMaskA = a;
    
        #ifdef _GLDEBUG
        GET_GL_ERROR;
        #endif
    }
}
//-----------------------------------------------------------------------------
/*! SLGLState::useProgram sets the _rent active shader program
*/
void SLGLState::useProgram(SLuint progID)
{
    if (_programID != progID)
    {   glUseProgram(progID);
        _programID = progID;
    
        #ifdef _GLDEBUG
        GET_GL_ERROR;
        #endif
    }
}
//-----------------------------------------------------------------------------
/*! SLGLState::bindTexture sets the current active texture.
*/
void SLGLState::bindTexture(GLenum target, SLuint textureID)
{
    if (target != _textureTarget || textureID != _textureID)
    {
        glBindTexture(target, textureID);

        _textureTarget = target;
        _textureID = textureID;
               
        #ifdef _GLDEBUG
        GET_GL_ERROR;
        #endif
    }
}
//-----------------------------------------------------------------------------
/*! SLGLState::activeTexture sets the current active texture unit
*/
void SLGLState::activeTexture(SLenum textureUnit)
{
    if (textureUnit != _textureUnit) 
    {   glActiveTexture(textureUnit);
        _textureUnit = textureUnit;
    
        #ifdef _GLDEBUG
        GET_GL_ERROR;
        #endif
    }
}
//-----------------------------------------------------------------------------
/*! SLGLState::unbindAnythingAndFlush unbinds all shaderprograms and buffers in 
use and calls glFinish. This should be the last call to GL before buffer
swapping.
*/
void SLGLState::unbindAnythingAndFlush()
{
    useProgram(0);
    
    // reset the bound texture unit
    // This is needed since leaving one texture unit bound over multiple windows
    // sometimes (most of the time) causes bugs
    //glBindTexture(GL_TEXTURE_2D, 0);
    //glBindVertexArray(0);
    //glBindBuffer(GL_ARRAY_BUFFER, 0);
    //glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);
    
    // The iOS OpenGL ES Analyzer suggests not to use flush or finish
    //glFlush();
    //glFinish();
    
    #ifdef _GLDEBUG
    GET_GL_ERROR;
    #endif
}
//-----------------------------------------------------------------------------
void SLGLState::getGLError(char* file, 
                           int line, 
                           bool quit)
{  
    #if defined(DEBUG) || defined(_DEBUG)
    GLenum err;
    if ((err = glGetError()) != GL_NO_ERROR) 
    {   
        string errStr;
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

        //fprintf(stderr, "OpenGL Error in %s, line %d: %s\n", file, line, errStr.c_str());

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
        {
            errors.push_back(newErr);
            #ifdef SL_OS_ANDROID
            __android_log_print(ANDROID_LOG_INFO, "SLProject",
                                "OpenGL Error in %s, line %d: %s\n",
                                file, line, errStr.c_str());
            #else
            fprintf(stderr,
                    "OpenGL Error in %s, line %d: %s\n",
                    file, line, errStr.c_str());
            #endif
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
/// Returns the OpenGL version number as a string
/*! The string returned by glGetString can contain additional vendor 
information such as the build number and the brand name. 
For the OpenGL version string "4.5.0 NVIDIA 347.68" the function returns "4.5" 
*/
SLstring SLGLState::getGLVersionNO()
{
    SLstring versionStr = SLstring((char*)glGetString(GL_VERSION));
    size_t dotPos = versionStr.find(".");
    SLchar NO[4];
    NO[0] = versionStr[dotPos - 1];
    NO[1] = '.';
    NO[2] = versionStr[dotPos + 1];
    NO[3] = 0;
    
    if (versionStr.find("OpenGL ES") > -1)
    {   SLstring strNO = "ES";
        return strNO + NO;    
    } else return SLstring(NO);
}
//-----------------------------------------------------------------------------
//! Returns the OpenGL Shading Language version number as a string.
/*! The string returned by glGetString can contain additional vendor 
information such as the build number and the brand name. 
For the shading language string "Nvidia GLSL 4.5" the function returns "450" 
*/
SLstring SLGLState::getSLVersionNO()
{
    SLstring versionStr = SLstring((char*)glGetString(GL_SHADING_LANGUAGE_VERSION));
    size_t dotPos = versionStr.find(".");
    SLchar NO[4];
    NO[0] = versionStr[dotPos - 1];
    NO[1] = versionStr[dotPos + 1];
    NO[2] = '0';
    NO[3] = 0;
    return SLstring(NO);
}
//-----------------------------------------------------------------------------
//! Returns true if the according GL pixelformat is valid in the GL context
SLbool SLGLState::pixelFormatIsSupported(SLint pixelFormat)
{   /*
    #define SL_ALPHA 0x1906             // ES2 ES3 GL2
    #define SL_LUMINANCE 0x1909         // ES2 ES3 GL2
    #define SL_LUMINANCE_ALPHA 0x190A   // ES2 ES3 GL2
    #define SL_INTENSITY 0x8049         //         GL2
    #define SL_GREEN 0x1904             //         GL2
    #define SL_BLUE 0x1905              //         GL2

    #define SL_RED  0x1903              //     ES3 GL2 GL3 GL4
    #define SL_RG   0x8227              //     ES3     GL3 GL4
    #define SL_RGB  0x1907              // ES2 ES3 GL2 GL3 GL4
    #define SL_RGBA 0x1908              // ES2 ES3 GL2 GL3 GL4
    #define SL_BGR  0x80E0              //         GL2 GL3 GL4
    #define SL_BGRA 0x80E1              //         GL2 GL3 GL4 (iOS defines it but it doesn't work)

    #define SL_RG_INTEGER 0x8228        //     ES3         GL4
    #define SL_RED_INTEGER 0x8D94       //     ES3         GL4
    #define SL_RGB_INTEGER 0x8D98       //     ES3         GL4
    #define SL_RGBA_INTEGER 0x8D99      //     ES3         GL4
    #define SL_BGR_INTEGER 0x8D9A       //                 GL4
    #define SL_BGRA_INTEGER 0x8D9B      //                 GL4
    */
    switch(pixelFormat)
    {
        case PF_rgb:
        case PF_rgba: return true;
        case PF_red: return (!_glIsES2);
        case PF_bgr:
        case PF_bgra: return (!_glIsES2 && !_glIsES3);
        case PF_luminance:
        case PF_luminance_alpha:
        case PF_alpha: return (_glIsES2 || _glIsES3 || (((SLint)_glVersionNOf) == 2));
        case PF_intensity:
        case PF_green:
        case PF_blue: return (!_glIsES2 && !_glIsES3 && (((SLint)_glVersionNOf) == 2));
        case PF_rg: return (!_glIsES2 && _glVersionNOf >= 3);
        case PF_rg_integer:
        case PF_red_integer:
        case PF_rgb_integer:
        case PF_rgba_integer: return (!_glIsES2 && _glVersionNOf >= 4);
        case PF_bgr_integer:
        case PF_bgra_integer: return (_glVersionNOf >= 4);
        default: return false;
    }
}
//-----------------------------------------------------------------------------
