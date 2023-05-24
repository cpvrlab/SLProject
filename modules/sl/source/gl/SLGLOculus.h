//#############################################################################
//  File:      SLOculus.h
//  Purpose:   Wrapper around Oculus Rift
//  Date:      July 2014
//  Codestyle: https://github.com/cpvrlab/SLProject/wiki/SLProject-Coding-Style
//  Authors:   Marc Wacker, Roman Kuehne, Marcus Hudritsch
//  License:   This software is provided under the GNU General Public License
//             Please visit: http://opensource.org/licenses/GPL-3.0
//#############################################################################

#ifndef SLOCULUS_H
#define SLOCULUS_H

#include <SLCamera.h>

//-----------------------------------------------------------------------------
//! Distorted vertex used to draw in the Oculus frame buffer.
struct SLVertexOculus
{
    SLVec2f screenPosNDC;
    float   timeWarpFactor;
    float   vignetteFactor;
    SLVec2f tanEyeAnglesR;
    SLVec2f tanEyeAnglesG;
    SLVec2f tanEyeAnglesB;
};

typedef vector<SLVertexOculus> SLVVertexOculus;

//-----------------------------------------------------------------------------
//! Wrapper around Oculus Rift Devkit 2.
/*! This class is depricated since the lib_ovr from Oculus completely changed
The lib_ovr that connects the Oculus hardware was removed since it only worked
for devkit2 under windows.
*/
class SLGLOculus
{
public:
    SLGLOculus(SLstring shaderDir);
    ~SLGLOculus();

    void init();

    const SLQuat4f& orientation(SLEyeType eye);
    const SLVec3f&  position(SLEyeType eye);

    const SLVec3f& viewAdjust(SLEyeType eye);
    const SLMat4f& projection(SLEyeType eye);
    const SLMat4f& orthoProjection(SLEyeType eye);

    SLfloat resolutionScale() { return _resolutionScale; }
    void    renderResolution(SLint width, SLint height);
    void    beginFrame();
    void    renderDistortion(SLint          width,
                             SLint          height,
                             SLuint         tex,
                             const SLCol4f& background);
    // Setters
    void lowPersistance(SLbool val);
    void timeWarp(SLbool val);
    void positionTracking(SLbool val);
    void displaySleep(SLbool val);

    // Getters
    SLbool isConnected() { return _isConnected; }
    SLbool isCameraConnected() { return _isCameraConnected; }
    SLbool isPositionTracked() { return _isPositionTracked; }

    SLbool isPositionTrackingEnabled() { return _positionTrackingEnabled; }
    SLbool isLowPersistanceEnabled() { return _lowPersistanceEnabled; }
    SLbool isTimeWarpEnabled() { return _timeWarpEnabled; }

private:
    void dispose();
    void calculateHmdValues(); //!< recalculate HMD settings changed

    // SL variables that can be accessed via getters
    SLVec2i _outputRes;                    //!< output resolution used for ortho projection

    SLQuat4f        _orientation[2];       //!< eye orientation
    SLVec3f         _position[2];          //!< eye position
    SLMat4f         _projection[2];        //!< projection matrices for left and right eye
    SLMat4f         _orthoProjection[2];   //!< projection for 2d elements
    SLVec3f         _viewAdjust[2];        //!< view adjust vector
    SLGLVertexArray _distortionMeshVAO[2]; //!< distortion meshes for left and right eye

    SLfloat _resolutionScale;              //!< required resolution scale for a 1.0 min pixel density

    // distortion
    SLbool _usingDebugHmd;           //!< we're using a debug HMD
    SLbool _positionTrackingEnabled; //!< is position tracking enabled
    SLbool _lowPersistanceEnabled;   //!< low persistence rendering enabled
    SLbool _timeWarpEnabled;         //!< time warp correction enabled
    SLbool _displaySleep;            //!< is the display of the rift currently off

    SLbool _isConnected;             //!< is HMD connected
    SLbool _isCameraConnected;       //!< is position tracker camera connected
    SLbool _isPositionTracked;       //!< is the position tracked (false if out of range)

    SLVec2i _resolution;             //!< Resolution of the HMD
    SLVec2i _rtSize;                 //!< Required resolution for the render target

    SLbool _hmdSettingsChanged;      //!< settings need to be updated flag

    SLGLProgram* _stereoOculusDistProgram = nullptr;

    SLstring _shaderFileDir;
};
//-----------------------------------------------------------------------------

#endif
