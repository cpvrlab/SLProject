//#############################################################################
//  File:      SLOculus.cpp
//  Purpose:   Wrapper around Oculus Rift
//  Date:      July 2014
//  Codestyle: https://github.com/cpvrlab/SLProject/wiki/SLProject-Coding-Style
//  Authors:   Marc Wacker, Roman Kuehne, Marcus Hudritsch
//  License:   This software is provided under the GNU General Public License
//             Please visit: http://opensource.org/licenses/GPL-3.0
//#############################################################################

#include <SLGLState.h>
#include <SLGLOVRWorkaround.h>
#include <SLGLOculus.h>
#include <SLGLProgram.h>

//-----------------------------------------------------------------------------
/*! Constructor initializing with default values
 */
SLGLOculus::SLGLOculus(SLstring shaderDir)
  : _usingDebugHmd(false),
    _positionTrackingEnabled(false),
    _lowPersistanceEnabled(false),
    _timeWarpEnabled(false),
    _displaySleep(false),
    _isConnected(false),
    _isCameraConnected(false),
    _isPositionTracked(false),
    _shaderFileDir(shaderDir)
{
}
//-----------------------------------------------------------------------------
/*! Destructor calling dispose
 */
SLGLOculus::~SLGLOculus()
{
    dispose();

    if (_stereoOculusDistProgram)
    {
        delete _stereoOculusDistProgram;
        _stereoOculusDistProgram = nullptr;
    }
}
//-----------------------------------------------------------------------------
/*! Deletes the buffer object
 */
void SLGLOculus::dispose()
{
}
//-----------------------------------------------------------------------------
/*! Initialization of the Oculus Rift SDK and the device recognition.
 */
void SLGLOculus::init()
{
    _stereoOculusDistProgram = new SLGLProgramGeneric(nullptr,
                                                      _shaderFileDir + "StereoOculusDistortionMesh.vert",
                                                      _shaderFileDir + "StereoOculusDistortionMesh.frag");
    _resolutionScale         = 1.25f;
    _resolution.set(1920, 1080);
    renderResolution(1920, 1080);

    for (SLint i = 0; i < 2; ++i)
    {
        _position[i].set(0, 0, 0);
        _orientation[i].set(0, 0, 0, 1);
        _viewAdjust[i].set((i * 2 - 1) * 0.03f, 0, 0); //[-0.03, 0.03]m

        // not 100% correct projections but it just has to look somewhat right
        _projection[i].perspective(125.0f, 0.88f, 0.1f, 1000.0f);
        _projection[i].translate(-_viewAdjust[i]);
    }

    createSLDistortionMesh(_stereoOculusDistProgram, ET_left, _distortionMeshVAO[0]);
    createSLDistortionMesh(_stereoOculusDistProgram, ET_right, _distortionMeshVAO[1]);
}
//-----------------------------------------------------------------------------
/*! Renders the distortion mesh with time warp and chromatic abberation
 */
void SLGLOculus::renderDistortion(SLint          width,
                                  SLint          height,
                                  SLuint         tex,
                                  const SLCol4f& background)
{
    assert(_stereoOculusDistProgram && "SLGLOculus::renderDistortion: shader program not set");
    SLGLProgram* sp = _stereoOculusDistProgram;

    glViewport(0, 0, width, height);
    glBindFramebuffer(GL_FRAMEBUFFER, 0);
    glClearColor(0, 0, 0, 1);
    glClear(GL_COLOR_BUFFER_BIT);
    glDisable(GL_DEPTH_TEST);
    glDisable(GL_CULL_FACE);
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, tex);

    sp->beginUse(nullptr, nullptr, nullptr);

    for (auto& eye : _distortionMeshVAO)
    {
        sp->uniform1i("u_texture", 0);
        sp->uniform2f("u_eyeToSourceUVScale", 0.232f, -0.376f);
        sp->uniform2f("u_eyeToSourceUVOffset", 0.246f, 0.5f);
        SLMat4f identity;

        sp->uniformMatrix4fv("u_eyeRotationStart", 1, identity.m());
        sp->uniformMatrix4fv("u_eyeRotationEnd", 1, identity.m());
        eye.drawElementsAs(PT_triangles);
    }

    sp->endUse();

    glEnable(GL_DEPTH_TEST);
    glEnable(GL_CULL_FACE);
}
//-----------------------------------------------------------------------------
/*! Returns the view adjust vector as reported by the HMD for the specified eye
 */
const SLVec3f& SLGLOculus::viewAdjust(SLEyeType eye)
{
    //@todo find a nicer way to store this (SLEye has a -1 for left and +1 for right eye)
    if (eye == ET_left)
        return _viewAdjust[0];
    else
        return _viewAdjust[1];
}
//-----------------------------------------------------------------------------
/*! Returns an perspective projection matrix for the specified eye
 */
const SLMat4f&
SLGLOculus::projection(SLEyeType eye)
{
    if (eye == ET_left)
        return _projection[0];
    else
        return _projection[1];
}
//-----------------------------------------------------------------------------
/*! Returns an orthogonal projection matrix for the specified eye
 */
const SLMat4f&
SLGLOculus::orthoProjection(SLEyeType eye)
{
    if (eye == ET_left)
        return _orthoProjection[0];
    else
        return _orthoProjection[1];
}
//-----------------------------------------------------------------------------
/*! Recalculates values such as projection or render target size
This function gets called whenever some settings changed.
*/
void SLGLOculus::calculateHmdValues()
{
    for (SLint i = 0; i < 2; ++i)
    {
        _position[i].set(0, 0, 0);
        _orientation[i].set(0, 0, 0, 1);
        _viewAdjust[i].set((i * 2 - 1) * 0.03f, 0, 0); //[-0.03, 0.03]m

        ovrFovPort fov;
        fov.DownTan    = 1.329f;
        fov.UpTan      = 1.329f;
        fov.LeftTan    = 1.058f;
        fov.RightTan   = 1.092f;
        _projection[i] = CreateProjection(true, fov, 0.01f, 10000.0f);

        _orthoProjection[i] = ovrMatrix4f_OrthoSubProjection(_projection[i],
                                                             SLVec2f(1.0f / (549.618286f * ((SLfloat)_outputRes.x / _resolution.x)),
                                                                     1.0f / (549.618286f * ((SLfloat)_outputRes.x / _resolution.x))),
                                                             0.8f,
                                                             _viewAdjust[i].x);

        SLMat4f flipY(1.0f, 0.0f, 0.0f, 0.0f, 0.0f, -1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f);
        _orthoProjection[i] = flipY * _orthoProjection[i];
    }

    // done
    _hmdSettingsChanged = false;
}
//-----------------------------------------------------------------------------
/*! Specify the final output resolution for this rift
 */
void SLGLOculus::renderResolution(SLint width, SLint height)
{
    if (width == _outputRes.x && height == _outputRes.y)
        return;

    _outputRes.x = width;
    _outputRes.y = height;

    _hmdSettingsChanged = true;
}
//-----------------------------------------------------------------------------
/*! Updates rift status and collects data for timewarp
 */
void SLGLOculus::beginFrame()
{
    // update changed settings
    if (_hmdSettingsChanged)
        calculateHmdValues();
}
//-----------------------------------------------------------------------------
/*! Returns the Oculus orientation as quaternion. If no Oculus Rift is
recognized it returns a unit quaternion.
*/
const SLQuat4f& SLGLOculus::orientation(SLEyeType eye)
{
    if (eye == ET_left)
        return _orientation[0];
    else
        return _orientation[1];
}
//-----------------------------------------------------------------------------
/*! Returns the Oculus position.
 */
const SLVec3f& SLGLOculus::position(SLEyeType eye)
{
    if (eye == ET_left)
        return _position[0];
    else
        return _position[1];
}
//-----------------------------------------------------------------------------
/*! enable or disable low persistance
 */
void SLGLOculus::lowPersistance(SLbool val)
{
    if (val == _lowPersistanceEnabled)
        return;

    _lowPersistanceEnabled = val;
    _hmdSettingsChanged    = true;
}
//-----------------------------------------------------------------------------
/*! enable or disable timewarp
 */
void SLGLOculus::timeWarp(SLbool val)
{
    if (val == _timeWarpEnabled)
        return;

    _timeWarpEnabled    = val;
    _hmdSettingsChanged = true;
}
//-----------------------------------------------------------------------------
/*! enable or disable position tracking
 */
void SLGLOculus::positionTracking(SLbool val)
{
    if (val == _positionTrackingEnabled)
        return;

    _positionTrackingEnabled = val;
    _hmdSettingsChanged      = true;
}
//-----------------------------------------------------------------------------
/*! enable or disable position tracking
 */
void SLGLOculus::displaySleep(SLbool val)
{
    if (val == _displaySleep)
        return;

    _displaySleep       = val;
    _hmdSettingsChanged = true;
}
//-----------------------------------------------------------------------------
