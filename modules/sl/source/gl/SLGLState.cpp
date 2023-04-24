//#############################################################################
//   File:      SLGLState.cpp
//   Purpose:   Singleton class implementation for global OpenGL replacement
//   Date:      July 2014
//   Codestyle: https://github.com/cpvrlab/SLProject/wiki/SLProject-Coding-Style
//   Authors:   Marcus Hudritsch
//   License:   This software is provided under the GNU General Public License
//              Please visit: http://opensource.org/licenses/GPL-3.0
//#############################################################################

#include <SLGLState.h>
#include <SLMaterial.h>
#include <CVImage.h>
#ifdef SL_OS_ANDROID
#    include <android/log.h>
#endif
//-----------------------------------------------------------------------------
SLGLState* SLGLState::_instance = nullptr;
//-----------------------------------------------------------------------------
/*! Public static destruction.
 */
void SLGLState::deleteInstance()
{
    delete _instance;
    _instance = nullptr;
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
    modelMatrix.identity();
    projectionMatrix.identity();
    textureMatrix.identity();

    _glVersion     = SLstring((const char*)glGetString(GL_VERSION));
    _glVersionNO   = getGLVersionNO();
    _glVersionNOf  = (SLfloat)atof(_glVersionNO.c_str());
    _glVendor      = SLstring((const char*)glGetString(GL_VENDOR));
    _glRenderer    = SLstring((const char*)glGetString(GL_RENDERER));
    _glSLVersion   = SLstring((const char*)glGetString(GL_SHADING_LANGUAGE_VERSION));
    _glSLVersionNO = getSLVersionNO();
    _glIsES2       = (_glVersion.find("OpenGL ES 2") != string::npos);
    _glIsES3       = (_glVersion.find("OpenGL ES 3") != string::npos);
    glGetIntegerv(GL_MAX_COMBINED_TEXTURE_IMAGE_UNITS, &_glMaxTexUnits);
    glGetIntegerv(GL_MAX_TEXTURE_SIZE, &_glMaxTexSize);

// Get extensions
#ifndef APP_USES_GLES
    if (_glVersionNOf > 3.0f)
    {
        GLint n;
        glGetIntegerv(GL_NUM_EXTENSIONS, &n);
        for (SLuint i = 0; i < (SLuint)n; i++)
            _glExtensions += SLstring((const char*)glGetStringi(GL_EXTENSIONS, i)) + ", ";
    }
    else
#endif
    {
        const GLubyte* ext = glGetString(GL_EXTENSIONS);
        if (ext) _glExtensions = SLstring((const char*)ext);
    }

    // initialize states a unset
    _blend                     = false;
    _blendFuncSfactor          = GL_SRC_ALPHA;
    _blendFuncDfactor          = GL_ONE_MINUS_SRC_ALPHA;
    _cullFace                  = false;
    _depthTest                 = false;
    _depthMask                 = false;
    _depthFunc                 = GL_LESS;
    _multisample               = false;
    _polygonLine               = false;
    _polygonOffsetPointEnabled = false;
    _polygonOffsetLineEnabled  = false;
    _polygonOffsetFillEnabled  = false;
    _viewport.set(-1, -1, -1, -1);
    _clearColor.set(-1, -1, -1, -1);

    // Reset all cached states to an invalid state
    _programID  = 0;
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

#ifndef SL_GLES
    glEnable(GL_PROGRAM_POINT_SIZE);
#endif

    GET_GL_ERROR;

    _currentMaterial = nullptr;
}
//-----------------------------------------------------------------------------
/*! The destructor only empties the stacks
 */
SLGLState::~SLGLState()
{
}
//-----------------------------------------------------------------------------
/*! One time initialization
 */
void SLGLState::onInitialize(const SLCol4f& clearColor)
{
    // Reset all internal states
    if (!_isInitialized) initAll();

    // enable depth_test
    glDepthFunc(GL_LESS);
    glEnable(GL_DEPTH_TEST);

    // set blend function for classic transparency
    glBlendFunc(_blendFuncSfactor, _blendFuncDfactor);

    // set background color
    glClearColor(clearColor.r,
                 clearColor.g,
                 clearColor.b,
                 clearColor.a);
    GET_GL_ERROR;
}
//-----------------------------------------------------------------------------
void SLGLState::clearColor(const SLCol4f& newColor)
{
    if (_clearColor != newColor)
    {
        glClearColor(newColor.r, newColor.g, newColor.b, newColor.a);
        _clearColor = newColor;

        GET_GL_ERROR;
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
    {
        if (stateNew)
            glEnable(GL_DEPTH_TEST);
        else
            glDisable(GL_DEPTH_TEST);
        _depthTest = stateNew;

        GET_GL_ERROR;
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
    {
        glDepthMask(stateNew ? GL_TRUE : GL_FALSE);
        _depthMask = stateNew;

        GET_GL_ERROR;
    }
}
//-----------------------------------------------------------------------------
/*! SLGLState::depthFunc specifies the depth comparison function.
Symbolic constants GL_NEVER, GL_LESS, GL_EQUAL, GL_LEQUAL, GL_GREATER,
GL_NOTEQUAL, GL_GEQUAL, and GL_ALWAYS are accepted. The initial value is GL_LESS.
*/
void SLGLState::depthFunc(SLenum func)
{
    if (_depthFunc != func)
    {
        glDepthFunc(func);
        _depthFunc = func;

        GET_GL_ERROR;
    }
}
//-----------------------------------------------------------------------------
/*! SLGLState::cullFace sets the GL_CULL_FACE state but only if the state
 really changes. If face culling is turned on no back faces are processed.
 */
void SLGLState::cullFace(SLbool stateNew)
{
    if (_cullFace != stateNew)
    {
        if (stateNew)
            glEnable(GL_CULL_FACE);
        else
            glDisable(GL_CULL_FACE);
        _cullFace = stateNew;

        GET_GL_ERROR;
    }
}
//-----------------------------------------------------------------------------
/*! SLGLState::blend enables or disables alpha blending but only if the state
 really changes.
 */
void SLGLState::blend(SLbool stateNew)
{
    if (_blend != stateNew)
    {
        if (stateNew)
            glEnable(GL_BLEND);
        else
            glDisable(GL_BLEND);
        _blend = stateNew;
        GET_GL_ERROR;
    }
}
//-----------------------------------------------------------------------------
//! Sets new blend function source and destination factors
void SLGLState::blendFunc(SLenum newBlendFuncSFactor,
                          SLenum newBlendFuncDFactor)
{
    if (_blendFuncSfactor != newBlendFuncSFactor ||
        _blendFuncDfactor != newBlendFuncDFactor)
    {
        glBlendFunc(newBlendFuncSFactor, newBlendFuncDFactor);
        _blendFuncSfactor = newBlendFuncSFactor;
        _blendFuncDfactor = newBlendFuncDFactor;
        GET_GL_ERROR;
    }
}
//-----------------------------------------------------------------------------
/*! SLGLState::multiSample enables or disables multisampling but only if the
 state really changes. Multisampling turns on fullscreen anti aliasing on the GPU
 witch produces smooth polygon edges, lines and points.
 */
void SLGLState::multiSample(SLbool stateNew)
{
#ifndef SL_GLES3
    if (_multisample != stateNew)
    {
        if (_multiSampleSamples > 0)
        {
            if (stateNew)
                glEnable(GL_MULTISAMPLE);
            else
                glDisable(GL_MULTISAMPLE);
            _multisample = stateNew;
        }

        GET_GL_ERROR;
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
#ifndef SL_GLES3
    if (_polygonLine != stateNew)
    {
        if (stateNew)
            glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
        else
            glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
        _polygonLine = stateNew;

        GET_GL_ERROR;
    }
#endif
}
//-----------------------------------------------------------------------------
/*! SLGLState::polygonOffsetPoint turns on/off polygon offset for points
 and sets the factor and unit for glPolygonOffset but only if the state really
 changes. Polygon offset is used to reduce z-fighting due to parallel planes or
 lines. See: http://www.zeuscmd.com/tutorials/opengl/15-PolygonOffset.php
 */
void SLGLState::polygonOffsetPoint(SLbool enabled, SLfloat factor, SLfloat units)
{
#ifndef SL_GLES3
    if (_polygonOffsetPointEnabled != enabled)
    {
        if (enabled)
        {
            glEnable(GL_POLYGON_OFFSET_POINT);
            glPolygonOffset(factor, units);
        }
        else
            glDisable(GL_POLYGON_OFFSET_POINT);
        _polygonOffsetPointEnabled = enabled;

        GET_GL_ERROR;
    }
#endif
}
//-----------------------------------------------------------------------------
/*! SLGLState::polygonOffsetLine turns on/off polygon offset for lines
 and sets the factor and unit for glPolygonOffset but only if the state really
 changes. Polygon offset is used to reduce z-fighting due to parallel planes or
 lines. See: http://www.zeuscmd.com/tutorials/opengl/15-PolygonOffset.php
 */
void SLGLState::polygonOffsetLine(SLbool enabled, SLfloat factor, SLfloat units)
{
#ifndef SL_GLES3
    if (_polygonOffsetLineEnabled != enabled)
    {
        if (enabled)
        {
            glEnable(GL_POLYGON_OFFSET_LINE);
            glPolygonOffset(factor, units);
        }
        else
            glDisable(GL_POLYGON_OFFSET_LINE);
        _polygonOffsetLineEnabled = enabled;

        GET_GL_ERROR;
    }
#endif
}
//-----------------------------------------------------------------------------
/*! SLGLState::polygonOffsetFill turns on/off polygon offset for filled polygons
 and sets the factor and unit for glPolygonOffset but only if the state really
 changes. Polygon offset is used to reduce z-fighting due to parallel planes or
 lines. See: http://www.zeuscmd.com/tutorials/opengl/15-PolygonOffset.php
 */
void SLGLState::polygonOffsetFill(SLbool enabled, SLfloat factor, SLfloat units)
{
    if (_polygonOffsetFillEnabled != enabled)
    {
        if (enabled)
        {
            glEnable(GL_POLYGON_OFFSET_FILL);
            glPolygonOffset(factor, units);
        }
        else
            glDisable(GL_POLYGON_OFFSET_FILL);
        _polygonOffsetFillEnabled = enabled;

        GET_GL_ERROR;
    }
}
//-----------------------------------------------------------------------------
/*! SLGLState::viewport sets the OpenGL viewport position and size
 */
void SLGLState::viewport(SLint x, SLint y, SLsizei width, SLsizei height)
{
    if (_viewport.x != x ||
        _viewport.y != y ||
        _viewport.z != width ||
        _viewport.w != height)
    {
        glViewport(x, y, width, height);
        _viewport.set(x, y, width, height);

        GET_GL_ERROR;
    }
}
//-----------------------------------------------------------------------------
/*! SLGLState::colorMask sets the OpenGL colorMask for framebuffer masking
 */
void SLGLState::colorMask(GLboolean r, GLboolean g, GLboolean b, GLboolean a)
{
    if (r != _colorMaskR || g != _colorMaskG || b != _colorMaskB || a != _colorMaskA)
    {
        glColorMask(r, g, b, a);
        _colorMaskR = r;
        _colorMaskG = g;
        _colorMaskB = b;
        _colorMaskA = a;

        GET_GL_ERROR;
    }
}
//-----------------------------------------------------------------------------
/*! SLGLState::useProgram sets the _rent active shader program
 */
void SLGLState::useProgram(SLuint progID)
{
    if (_programID != progID)
    {
        glUseProgram(progID);
        _programID = progID;

        GET_GL_ERROR;
    }
}
//-----------------------------------------------------------------------------
/*! SLGLState::bindTexture sets the current active texture.
 */
void SLGLState::bindTexture(SLenum target, SLuint textureID)
{
    // (luc) If there we call glActiveTexture and glBindTexture from outside,
    // This will lead to problems as the global state in SLGLState will not be
    // equivalent to the OpenGL state.
    // We should solve this by querying opengl for the last binded texture.
    // glGetIntegeriv(GL_ACTIVE_TEXTURE, active_texture)
    // glGetIntegeriv(GL_TEXTURE_BINDING_2D, textureID)

    // if (target != _textureTarget || textureID != _textureID)
    {
        glBindTexture(target, textureID);

        _textureTarget = target;
        _textureID     = textureID;

        GET_GL_ERROR;
    }
}
//-----------------------------------------------------------------------------
/*! SLGLState::activeTexture sets the current active texture unit
 */
void SLGLState::activeTexture(SLenum textureUnit)
{
    if ((textureUnit - GL_TEXTURE0) > _glMaxTexUnits)
        SL_LOG("******* To many texture units: %i used of %i",
               (SLint)textureUnit - GL_TEXTURE0,
               _glMaxTexUnits);

    assert((textureUnit - GL_TEXTURE0) <= _glMaxTexUnits &&
           "To many texture units!");

    glActiveTexture(textureUnit);
    _textureUnit = textureUnit;

    GET_GL_ERROR;
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
    // glBindTexture(GL_TEXTURE_2D, 0);
    // glBindVertexArray(0);
    // glBindBuffer(GL_ARRAY_BUFFER, 0);
    // glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);

    // The iOS OpenGL ES Analyzer suggests not to use flush or finish
    // glFlush();
    // glFinish();

    GET_GL_ERROR;
}
//-----------------------------------------------------------------------------
void SLGLState::getGLError(const char* file,
                           int         line,
                           bool        quit)
{
    GLenum err;
    if ((err = glGetError()) != GL_NO_ERROR)
    {
        string errStr;
        switch (err)
        {
            case GL_INVALID_ENUM:
                errStr = "GL_INVALID_ENUM";
                break;
            case GL_INVALID_VALUE:
                errStr = "GL_INVALID_VALUE";
                break;
            case GL_INVALID_OPERATION:
                errStr = "GL_INVALID_OPERATION";
                break;
            case GL_INVALID_FRAMEBUFFER_OPERATION:
                errStr = "GL_INVALID_FRAMEBUFFER_OPERATION";
                break;
            case GL_OUT_OF_MEMORY:
                errStr = "GL_OUT_OF_MEMORY";
                break;
            default:
                errStr = "Unknown error";
        }

        // Build error string as a concatenation of file, line & error
        char sLine[32];
        snprintf(sLine, sizeof(sLine), "%d", line);

        string newErr(file);
        newErr += ": line:";
        newErr += sLine;
        newErr += ": ";
        newErr += errStr;

        // Check if error exists already
        SLGLState* state     = SLGLState::instance();
        bool       errExists = std::find(state->errors.begin(),
                                   state->errors.end(),
                                   newErr) != state->errors.end();
        // Only print
        if (!errExists)
        {
            state->errors.push_back(newErr);
            SL_LOG("OpenGL Error in %s, line %d: %s\n", file, line, errStr.c_str());
        }

        if (quit)
            exit(1);
    }
}
//-----------------------------------------------------------------------------
/// Returns the OpenGL version number as a string
/*! The string returned by glGetString can contain additional vendor
 information such as the build number and the brand name.
 For the OpenGL version string "4.5.0 NVIDIA 347.68" the function returns "4.5"
 */
SLstring SLGLState::getGLVersionNO()
{
    SLstring versionStr = SLstring((const char*)glGetString(GL_VERSION));
    size_t   dotPos     = versionStr.find('.');
    SLchar   NO[4];
    NO[0] = versionStr[dotPos - 1];
    NO[1] = '.';
    NO[2] = versionStr[dotPos + 1];
    NO[3] = 0;

    if (versionStr.find("OpenGL ES") != string::npos)
    {
        return SLstring(NO) + "ES";
    }
    else
        return SLstring(NO);
}
//-----------------------------------------------------------------------------
//! Returns the OpenGL Shading Language version number as a string.
/*! The string returned by glGetString can contain additional vendor
 information such as the build number and the brand name.
 For the shading language string "Nvidia GLSL 4.5" the function returns "450"
 */
SLstring SLGLState::getSLVersionNO()
{
    SLstring versionStr = SLstring((const char*)glGetString(GL_SHADING_LANGUAGE_VERSION));
    size_t   dotPos     = versionStr.find('.');
    SLchar   NO[4];
    NO[0] = versionStr[dotPos - 1];
    NO[1] = versionStr[dotPos + 1];
    NO[2] = '0';
    NO[3] = 0;
    return SLstring(NO);
}
//-----------------------------------------------------------------------------
//! Returns true if the according GL pixel format is valid in the GL context
SLbool SLGLState::pixelFormatIsSupported(SLint pixelFormat)
{ /*
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
    switch (pixelFormat)
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
//! Reads the front framebuffer pixels into the passed buffer
/*!
 * @param buffer Pointer to a 4 byte aligned buffer with the correct size.
 */
void SLGLState::readPixels(void* buffer)
{
    glPixelStorei(GL_PACK_ALIGNMENT, 4);

#ifndef SL_EMSCRIPTEN
    glReadBuffer(GL_FRONT);
#endif

    // Get viewport size
    GLint vp[4];
    glGetIntegerv(GL_VIEWPORT, vp);

    glReadPixels(vp[0],
                 vp[1],
                 vp[2],
                 vp[3],
                 SL_READ_PIXELS_GL_FORMAT,
                 GL_UNSIGNED_BYTE,
                 buffer);
}
//-----------------------------------------------------------------------------
