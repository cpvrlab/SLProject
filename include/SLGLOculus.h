//#############################################################################
//  File:      SLOculus.h
//  Purpose:   Wrapper around Oculus Rift
//  Author:    Marc Wacker, Roman Kühne, Marcus Hudritsch
//  Date:      July 2014
//  Codestyle: https://github.com/cpvrlab/SLProject/wiki/Coding-Style-Guidelines
//  Copyright: Marcus Hudritsch
//             This software is provide under the GNU General Public License
//             Please visit: http://opensource.org/licenses/GPL-3.0
//#############################################################################

#ifndef SLOCULUS_H
#define SLOCULUS_H

#include <stdafx.h>
#include <SLCamera.h>

#ifdef SL_OVR
#include <OVR.h>
#endif

//-----------------------------------------------------------------------------
//! Distorted vertex used to draw in the Occulus frame buffer.
struct SLGLOcculusDistortionVertex
{
    SLVec2f screenPosNDC;
    float   timeWarpFactor;
    float   vignetteFactor;
    SLVec2f tanEyeAnglesR;
    SLVec2f tanEyeAnglesG;
    SLVec2f tanEyeAnglesB;
};

//-----------------------------------------------------------------------------
//! Wrapper around Oculus Rift
class SLGLOculus
{
    public:
                        SLGLOculus          ();
                       ~SLGLOculus          ();
      
    const   SLQuat4f&   orientation         (SLEye eye);
    const   SLVec3f&    position            (SLEye eye);

    const   SLVec3f&    viewAdjust          (SLEye eye);
    const   SLMat4f&    projection          (SLEye eye);
    const   SLMat4f&    orthoProjection     (SLEye eye);
            
            SLfloat     resolutionScale     () { return _resolutionScale; }

            void        renderResolution    (SLint width, SLint height);

            void        beginFrame          ();
            void        endFrame            (SLint width, SLint height, SLuint tex);


            void        renderDistortion    (SLint width, SLint height, SLuint tex);
            

            // Setters
            void        lowPersistance      (SLbool val);
            void        timeWarp            (SLbool val);
            void        positionTracking    (SLbool val);
            void        displaySleep        (SLbool val);

            // Getters
            SLbool      isConnected         () { return _isConnected; }
            SLbool      isCameraConnected   () { return _isCameraConnected; }
            SLbool      isPositionTracked   () { return _isPositionTracked; }
            
            SLbool      isPositionTrackingEnabled() { return _positionTrackingEnabled; }
            SLbool      isLowPersistanceEnabled() { return _lowPersistanceEnabled; }
            SLbool      isTimeWarpEnabled   () { return _timeWarpEnabled; } 


    private:
            void        init                ();
            void        dispose             ();
            
#ifdef SL_OVR
            // ovr variables
            ovrHmd              _hmd;
            ovrFrameTiming      _frameTiming;
            SLuint              _startTrackingCaps;     //!< the current ovr tracking configuration
            
            ovrEyeRenderDesc    _eyeRenderDesc[2];      //!< ovr eye render description
            ovrPosef            _eyeRenderPose[2];      //!< ovr individual eye render pose

            ovrRecti            _viewports[2];          //!< viewport size and offset for both eyes
            ovrVector2f         _uvScaleOffset[2][2];   //!< uv scale and offset for each eye
#endif


            // SL variables that can be accessed via getters
            SLVec2i             _outputRes;                 //!< output resolution used for ortho projection
            SLQuat4f            _orientation[2];            //!< eye orientation
            SLVec3f             _position[2];               //!< eye position
                        
            SLMat4f             _projection[2];             //!< projection matrices for left and right eye
            SLMat4f             _orthoProjection[2];        //!< projection for 2d elements
            SLVec3f             _viewAdjust[2];             //!< view ajust vector
            
            SLGLBuffer          _distortionMeshVB[2];       //!< distortion meshes for left and right eye 
            SLGLBuffer          _distortionMeshIB[2];
            
            SLfloat             _resolutionScale;           //!< required resolution scale for a 1.0 min pixel density
            
            // distortion                
            SLbool              _usingDebugHmd;             //!< we're using a debug hmd
            SLbool              _positionTrackingEnabled;   //!< is position tracking enabled
            SLbool              _lowPersistanceEnabled;     //!< low persistance rendering enabled
            SLbool              _timeWarpEnabled;           //!< time warp correction enabled
            SLbool              _displaySleep;              //!< is the display of the rift currently off

            SLbool              _isConnected;               //!< is hmd connected
            SLbool              _isCameraConnected;         //!< is position tracker camera connected
            SLbool              _isPositionTracked;         //!< is the position tracked (false if out of range)

            SLVec2i             _resolution;                //!< Resolution of the hmd
            SLVec2i             _rtSize;                    //!< Required resolution for the render target
            
            SLbool              _hmdSettingsChanged;        //!< settings need to be updated flag
            void                calculateHmdValues();       //!< recalculate hmd settings changed


};
//-----------------------------------------------------------------------------

#endif
