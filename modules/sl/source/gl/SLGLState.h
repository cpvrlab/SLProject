//#############################################################################
//  File:      SLGLState.h
//  Purpose:   Singleton class for global render state
//  Date:      July 2014
//  Codestyle: https://github.com/cpvrlab/SLProject/wiki/SLProject-Coding-Style
//  Authors:   Marcus Hudritsch
//  License:   This software is provided under the GNU General Public License
//             Please visit: http://opensource.org/licenses/GPL-3.0
//#############################################################################

#ifndef SLGLSTATE_H
#define SLGLSTATE_H

#include <SL.h>

#if defined(SL_OS_MACIOS)
#    include <OpenGLES/ES3/gl.h>
#    include <OpenGLES/ES3/glext.h>
#elif defined(SL_OS_MACOS)
#    include <GL/gl3w.h>
#elif defined(SL_OS_ANDROID)
// https://stackoverflow.com/questions/31003863/gles-3-0-including-gl2ext-h
#    include <GLES3/gl3.h>
#    include <GLES2/gl2ext.h>
#    ifndef GL_CLAMP_TO_BORDER // see #define GL_CLAMP_TO_BORDER_OES 0x812D in gl2ext.h
#        define GL_CLAMP_TO_BORDER GL_CLAMP_TO_BORDER_OES
#    endif
//#    include <GLES3/gl31.h>
//#    include <GLES3/gl3ext.h>
#elif defined(SL_OS_WINDOWS)
#    include <GL/gl3w.h>
#elif defined(SL_OS_LINUX)
#    include <GL/gl3w.h>
#elif defined(SL_EMSCRIPTEN)
#    include <GLES3/gl3.h>
#else
#    error "SL has not been ported to this OS"
#endif

#include <SLVec3.h>
#include <SLVec4.h>
#include <SLMat4.h>

class SLDrawBits;
class SLGLDepthBuffer;
class SLMaterial;

//-----------------------------------------------------------------------------
static const SLint SL_MAX_LIGHTS = 8; //!< max. number of used lights
//-----------------------------------------------------------------------------

// glGetError turns WebGL rendering into a slideshow, so we disable it when compiling with Emscripten
#if (defined(DEBUG) || defined(_DEBUG)) && !defined(SL_EMSCRIPTEN)
#    define GET_GL_ERROR SLGLState::getGLError((const char*)__FILE__, __LINE__, false)
#else
#    define GET_GL_ERROR
#endif
//-----------------------------------------------------------------------------
//! Singleton class holding all OpenGL states
/*!
 The main purpose of the SLGLState class is to replace all the OpenGL states and
 functionality that has been removed from the core profile of OpenGL. The core
 profile started from OpenGL version 3.0 has e.g. no more internal matrices,
 lighting or material states. It also has no more fixed function pipeline on the
 GPU witch means, that core profile OpenGL only works with custom shader
 programs written in OpenGL Shading Language (GLSL).
 The second purpose is to concentrate OpenGL functionality and to reduce
 redundant state changes.
 */
class SLGLState
{
public:
    //! Public static instance getter for singleton pattern
    static SLGLState* instance()
    {
        if (!_instance)
        {
            _instance = new SLGLState();
            return _instance;
        }
        else
            return _instance;
    }
    static void deleteInstance();                        //!< global destruction
    void        onInitialize(const SLCol4f& clearColor); //!< On init GL
    void        initAll();                               //! Init all states

    // matrices
    SLMat4f modelMatrix;      //!< matrix for model to world transform
    SLMat4f projectionMatrix; //!< matrix for projection transform
    SLMat4f viewMatrix;       //!< matrix for the active cameras view transform
    SLMat4f textureMatrix;    //!< matrix for the texture transform

    // getters
    inline bool hasMultiSampling() const { return _multiSampleSamples > 0; }

    // misc.
    void   unbindAnythingAndFlush(); //!< finishes all GL commands
    SLbool pixelFormatIsSupported(SLint pixelFormat);
    void   readPixels(void* buffer);

    // state setters
    void depthTest(SLbool state);
    void depthMask(SLbool state);
    void depthFunc(SLenum func);
    void cullFace(SLbool state);
    void blend(SLbool state);
    void blendFunc(SLenum blendFuncSFactor, SLenum blendFuncDFactor);
    void multiSample(SLbool state);
    void polygonLine(SLbool state);
    void polygonOffsetPoint(SLbool enabled, SLfloat factor = -1.0f, SLfloat units = -1.0f);
    void polygonOffsetLine(SLbool enabled, SLfloat factor = -1.0f, SLfloat units = -1.0f);
    void polygonOffsetFill(SLbool enabled, SLfloat factor = -1.0f, SLfloat units = -1.0f);
    void viewport(SLint x, SLint y, SLsizei width, SLsizei height);
    void colorMask(GLboolean r, GLboolean g, GLboolean b, GLboolean a);
    void useProgram(SLuint progID);
    void bindTexture(SLenum target, SLuint textureID);
    void activeTexture(SLenum textureUnit);
    void clearColor(const SLCol4f& c);
    void currentMaterial(SLMaterial* mat) { _currentMaterial = mat; }
    void clearColorBuffer() { glClear(GL_COLOR_BUFFER_BIT); }
    void clearDepthBuffer() { glClear(GL_DEPTH_BUFFER_BIT); }
    void clearColorDepthBuffer() { glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT); }

    // state getters
    SLbool   blend() const { return _blend; }
    SLstring glVersion() { return _glVersion; }
    SLstring glVersionNO() { return _glVersionNO; }
    SLfloat  glVersionNOf() const { return _glVersionNOf; }
    SLstring glVendor() { return _glVendor; }
    SLstring glRenderer() { return _glRenderer; }
    SLstring glSLVersion() { return _glSLVersion; }
    SLstring glSLVersionNO() { return _glSLVersionNO; }
    SLbool   glIsES() const { return _glIsES2 || _glIsES3; }
    SLbool   glIsES2() const { return _glIsES2; }
    SLbool   glIsES3() const { return _glIsES3; }
    SLint    glMaxTexUnits() const { return _glMaxTexUnits; }
    SLint    glMaxTexSize() const { return _glMaxTexSize; }
    SLbool   glHasGeometryShaders() const { return (_glIsES3 && _glVersionNOf > 3.1f) || (!glIsES() && _glVersionNOf >= 4.1f); }
    SLbool   hasExtension(const SLstring& e) { return _glExtensions.find(e) != string::npos; }
    SLVec4i  viewport() { return _viewport; }
    SLMat4f  viewportMatrix()
    {
        SLMat4f vpm;
        vpm.viewport((SLfloat)_viewport.x,
                     (SLfloat)_viewport.y,
                     (SLfloat)_viewport.z,
                     (SLfloat)_viewport.w);
        return vpm;
    }
    SLMaterial* currentMaterial() { return _currentMaterial; }

    //! Checks if an OpenGL error occurred
    static void getGLError(const char* file, int line, bool quit);

    SLstring getGLVersionNO();
    SLstring getSLVersionNO();

private:
    SLGLState();                 //!< private onetime constructor
    ~SLGLState();                //!< destruction in ~SLScene

    static SLGLState* _instance; //!< global singleton object

    SLbool _isInitialized;       //!< flag for first init

    SLstring _glVersion;         //!< OpenGL Version string
    SLstring _glVersionNO;       //!< OpenGL Version number string
    SLfloat  _glVersionNOf;      //!< OpenGL Version number as float
    SLstring _glVendor;          //!< OpenGL Vendor string
    SLstring _glRenderer;        //!< OpenGL Renderer string
    SLstring _glSLVersion;       //!< GLSL Version string
    SLstring _glSLVersionNO;     //!< GLSL Version number string
    SLstring _glExtensions;      //!< OpenGL extensions string
    SLbool   _glIsES2;           //!< Flag if OpenGL ES2
    SLbool   _glIsES3;           //!< Flag if OpenGL ES3
    SLint    _glMaxTexUnits;     //!< glGetIntegerv(GL_MAX_COMBINED_TEXTURE_IMAGE_UNITS, &_glMaxTexUnits);
    SLint    _glMaxTexSize;      //!< glGetIntegerv(GL_MAX_TEXTURE_SIZE, &_glMaxTexSize);

    // read/write states
    SLbool  _blend;                     //!< blending default false;
    SLenum  _blendFuncSfactor;          //!< blend function source factor enum
    SLenum  _blendFuncDfactor;          //!< blend function destination factor enum
    SLbool  _depthTest;                 //!< GL_DEPTH_TEST state
    SLbool  _depthMask;                 //!< glDepthMask state
    SLenum  _depthFunc;                 //!< depth buffer comparison function
    SLbool  _cullFace;                  //!< Face culling state
    SLbool  _multisample;               //!< Multisampling state
    SLint   _multiSampleSamples;        //!< NO. of multisampling samples
    SLbool  _polygonLine;               //!< Line polygon state
    SLbool  _polygonOffsetPointEnabled; //!< GL_POLYGON_OFFSET_POINT state enabled
    SLbool  _polygonOffsetLineEnabled;  //!< GL_POLYGON_OFFSET_LINE state enabled
    SLbool  _polygonOffsetFillEnabled;  //!< GL_POLYGON_OFFSET_FILL state enabled
    SLVec4i _viewport;                  //!< viewport size (x,y,w,h) of the framebuffer
    SLCol4f _clearColor;                //!< clear color

    // states
    SLuint    _programID;     //!< current shader program id
    SLenum    _textureUnit;   //!< current texture unit
    SLenum    _textureTarget; //!< current texture target
    SLuint    _textureID;     //!< current texture id
    GLboolean _colorMaskR;    //!< current color mask for R
    GLboolean _colorMaskG;    //!< current color mask for G
    GLboolean _colorMaskB;    //!< current color mask for B
    GLboolean _colorMaskA;    //!< current color mask for A

    SLVstring errors;         //!< vector for errors collected in getGLError

    SLMaterial* _currentMaterial;
};
//-----------------------------------------------------------------------------
#endif
