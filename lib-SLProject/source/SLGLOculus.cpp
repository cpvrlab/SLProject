//#############################################################################
//  File:      SLOculus.cpp
//  Purpose:   Wrapper around Oculus Rift
//  Author:    Marc Wacker, Roman Kühne, Marcus Hudritsch
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

#include <SLGLOculus.h>
#include <SLGLProgram.h>
#include <SLScene.h>

#ifndef SL_OVR
#include <SLGLOVRWorkaround.h>
#endif


//-----------------------------------------------------------------------------
/*! Constructor initializing with default values
*/
SLGLOculus::SLGLOculus() : _usingDebugHmd(false),
                           _positionTrackingEnabled(false),
                           _lowPersistanceEnabled(false),
                           _timeWarpEnabled(false),
                           _displaySleep(false),
                           _isConnected(false),
                           _isCameraConnected(false),
                           _isPositionTracked(false)
{
    #ifdef SL_OVR
    _hmd = nullptr;
    #endif
    init();
}
//-----------------------------------------------------------------------------
/*! Destructor calling dispose
*/
SLGLOculus::~SLGLOculus() 
{  
    dispose();
}
//-----------------------------------------------------------------------------
/*! Deletes the buffer object
*/
void SLGLOculus::dispose()
{  
    #ifdef SL_OVR
    // dispose Oculus
    if (_hmd)
        ovrHmd_Destroy(_hmd);

    ovr_Shutdown();

    #endif
}

//-----------------------------------------------------------------------------
/*! Initialization of the Oculus Rift SDK and the device recognition.
*/
void SLGLOculus::init()
{
#ifdef SL_OVR
    ovr_Initialize();
    
    // for now we just support one device
    _hmd = ovrHmd_Create(0);
    //SL_LOG("%s", ovrHmd_GetLastError(_hmd));
    

    if (!_hmd)
    {
        // create a debug device if we didn't find a physical one
        _hmd = ovrHmd_CreateDebug(ovrHmd_DK2);
        _usingDebugHmd = true;
        assert(_hmd);
    }
    
    // get hmd resolution
    ovrSizei resolution = _hmd->Resolution;
    _resolution.set(resolution.w, resolution.h);

    // set output resolution to the above set hmd resolution for now
    // this can be changed later however if we want to output to a smaller screen
    renderResolution(_resolution.x, _resolution.y);

    // are we running in extended desktop mode or are we using the oculus driver
    SLbool useAppWindowFrame = (_hmd->HmdCaps & ovrHmdCap_ExtendDesktop) ? false : true;
    // TODO: we need to call ovrHmd_AttachToWindow with a windw handle

    _positionTrackingEnabled = (_hmd->TrackingCaps & ovrTrackingCap_Position) ? true : false;
    _lowPersistanceEnabled = (_hmd->HmdCaps & ovrHmdCap_LowPersistence) ? true : false;
    
    calculateHmdValues();

    _viewports[0].Pos = OVR::Vector2i(0, 0);
    _viewports[0].Size = OVR::Sizei(_rtSize.x / 2, _rtSize.y);
    _viewports[1].Pos = OVR::Vector2i((_rtSize.x + 1) / 2, 0);
    _viewports[1].Size = _viewports[0].Size;


    //Generate distortion mesh for each eye
    SLuint distortionCaps = ovrDistortionCap_Chromatic | ovrDistortionCap_TimeWarp 
        | ovrDistortionCap_Vignette | ovrDistortionCap_FlipInput;

    // TODO: careful here, the actual size of the framebuffer might differ due to hardware limitations
    ovrFovPort viewports[2] = { _hmd->DefaultEyeFov[0], _hmd->DefaultEyeFov[1] };
    OVR::Sizei rtSize(_rtSize.x, _rtSize.y);



    for ( int eyeNum = 0; eyeNum < 2; eyeNum++ )
    {
        // Allocate & generate distortion mesh vertices.
        ovrDistortionMesh meshData;
        ovrHmd_CreateDistortionMesh(_hmd, 
                                    (ovrEyeType)eyeNum, 
                                    _eyeRenderDesc[eyeNum].Fov,
                                    distortionCaps, 
                                    &meshData);

        ovrHmd_GetRenderScaleAndOffset(_eyeRenderDesc[eyeNum].Fov,
                                       rtSize, 
                                       _viewports[eyeNum], 
                                       _uvScaleOffset[eyeNum]);

        // Now parse the vertex data and create a render ready vertex buffer from it
        SLGLOcculusDistortionVertex* pVBVerts = new SLGLOcculusDistortionVertex[meshData.VertexCount];

        vector<SLuint> tempIndex;

        SLGLOcculusDistortionVertex* v = pVBVerts;
        ovrDistortionVertex * ov = meshData.pVertexData;
        for ( unsigned vertNum = 0; vertNum < meshData.VertexCount; vertNum++ )
        {
            v->screenPosNDC.x = ov->ScreenPosNDC.x;
            v->screenPosNDC.y = ov->ScreenPosNDC.y;

            v->timeWarpFactor = ov->TimeWarpFactor;
            v->vignetteFactor = ov->VignetteFactor;
            
            v->tanEyeAnglesR.x = ov->TanEyeAnglesR.x;
            v->tanEyeAnglesR.y = ov->TanEyeAnglesR.y;

            v->tanEyeAnglesG.x = ov->TanEyeAnglesG.x;
            v->tanEyeAnglesG.y = ov->TanEyeAnglesG.y;

            v->tanEyeAnglesB.x = ov->TanEyeAnglesB.x;
            v->tanEyeAnglesB.y = ov->TanEyeAnglesB.y;
            
            v++; ov++;
        }

        for (unsigned i = 0; i < meshData.IndexCount; i++)
            tempIndex.push_back(meshData.pIndexData[i]);

        //@todo the SLGLBuffer isn't made for this kind of interleaved usage
        //       rework it so it is easier to use and more dynamic.
        _distortionMeshVB[eyeNum].generate(pVBVerts, meshData.VertexCount, 10,
                                         SL_FLOAT, SL_ARRAY_BUFFER, SL_STATIC_DRAW);
        // somehow passing in meshData.pIndexData doesn't work...
        _distortionMeshIB[eyeNum].generate(&tempIndex[0], meshData.IndexCount, 1,
                                           SL_UNSIGNED_INT, SL_ELEMENT_ARRAY_BUFFER, SL_STATIC_DRAW);
        delete[] pVBVerts;
        ovrHmd_DestroyDistortionMesh( &meshData );  
    }
#else

	_resolutionScale = 1.25f;
    _resolution.set(1920, 1080);
    renderResolution(1920, 1080);


    for (SLint i = 0; i < 2; ++i)
    {
        _position[i].set(0, 0, 0);
        _orientation[i].set(0, 0, 0, 1);
        _viewAdjust[i].set((i*2-1)*0.03f, 0, 0); //[-0.03, 0.03]m
    
        // not 100% correct projctions but it just has to look somewhat right
        _projection[i].perspective(125.0f, 0.88f, 0.1f, 1000.0f);
        _projection[i].translate(-_viewAdjust[i]);
    }
    
    createSLDistortionMesh(leftEye, _distortionMeshVB[0], _distortionMeshIB[0]);
    createSLDistortionMesh(rightEye, _distortionMeshVB[1], _distortionMeshIB[1]);

#endif

}

//-----------------------------------------------------------------------------
/*! Renders the distortion mesh with timewarp and chromatic abberation
*/
void SLGLOculus::renderDistortion(SLint width, SLint height, SLuint tex)
{
    SLGLProgram* sp = SLScene::current->programs(StereoOculusDistortionMesh);

    glViewport(0, 0, width, height);
    glBindFramebuffer(GL_FRAMEBUFFER, 0);
    glClearColor(0, 0, 0, 1);
    glClear(GL_COLOR_BUFFER_BIT);
    glDisable(GL_DEPTH_TEST);
    glDisable(GL_CULL_FACE);
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, tex);


    sp->beginUse();
    
    for (int eye = 0; eye < 2; eye++) {
        
        sp->uniform1i("u_texture", 0);

#ifdef SL_OVR
        sp->uniform2f("u_eyeToSourceUVScale",  _uvScaleOffset[eye][0].x, -_uvScaleOffset[eye][0].y);
        sp->uniform2f("u_eyeToSourceUVOffset", _uvScaleOffset[eye][1].x,  _uvScaleOffset[eye][1].y);
    
		ovrPosef eyeRenderPose = ovrHmd_GetHmdPosePerEye(_hmd, (ovrEyeType)0);
        ovrMatrix4f timeWarpMatrices[2];
        ovrHmd_GetEyeTimewarpMatrices(_hmd, (ovrEyeType)0, eyeRenderPose, timeWarpMatrices);
    
        sp->uniformMatrix4fv("u_eyeRotationStart", 1, (SLfloat*)&timeWarpMatrices[0]);
        sp->uniformMatrix4fv("u_eyeRotationEnd",   1, (SLfloat*)&timeWarpMatrices[1]);
#else
        sp->uniform2f("u_eyeToSourceUVScale",  0.232f, -0.376f);
        sp->uniform2f("u_eyeToSourceUVOffset", 0.246f, 0.5f);

        SLMat4f identity;

        sp->uniformMatrix4fv("u_eyeRotationStart", 1, identity.m());
        sp->uniformMatrix4fv("u_eyeRotationEnd", 1, identity.m());
#endif

         // manually bind the array buffer since SLGLBuffer doesn't support interleaved
        glBindBuffer(GL_ARRAY_BUFFER, _distortionMeshVB[eye].id());     

        SLint attrPos       = sp->getAttribLocation("a_position");
        SLint attrTimeWarp  = sp->getAttribLocation("a_timeWarpFactor");
        SLint attrVignette  = sp->getAttribLocation("a_vignetteFactor");
        SLint attrTexCoordR = sp->getAttribLocation("a_texCoordR");
        SLint attrTexCoordG = sp->getAttribLocation("a_texCoordG");
        SLint attrTexCoordB = sp->getAttribLocation("a_texCoordB");

        // enable the vertex attribute data array by index
        glEnableVertexAttribArray(attrPos);
        glEnableVertexAttribArray(attrTimeWarp);
        glEnableVertexAttribArray(attrVignette);
        glEnableVertexAttribArray(attrTexCoordR);
        glEnableVertexAttribArray(attrTexCoordG);
        glEnableVertexAttribArray(attrTexCoordB);
      
        // defines the vertex attribute data array by index
        glVertexAttribPointer(attrPos,       2, GL_FLOAT, GL_FALSE, sizeof(GLfloat)*10, 0);
        glVertexAttribPointer(attrTimeWarp,  1, GL_FLOAT, GL_FALSE, sizeof(GLfloat)*10, (GLvoid*)(sizeof(GLfloat)*2));
        glVertexAttribPointer(attrVignette,  1, GL_FLOAT, GL_FALSE, sizeof(GLfloat)*10, (GLvoid*)(sizeof(GLfloat)*3));
        glVertexAttribPointer(attrTexCoordR, 2, GL_FLOAT, GL_FALSE, sizeof(GLfloat)*10, (GLvoid*)(sizeof(GLfloat)*4));
        glVertexAttribPointer(attrTexCoordG, 2, GL_FLOAT, GL_FALSE, sizeof(GLfloat)*10, (GLvoid*)(sizeof(GLfloat)*6));
        glVertexAttribPointer(attrTexCoordB, 2, GL_FLOAT, GL_FALSE, sizeof(GLfloat)*10, (GLvoid*)(sizeof(GLfloat)*8));
    
        //glPolygonMode(GL_FRONT_AND_BACK, GL_LINES);
        //glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, _distortionMeshIB[eye].id());
        //glDrawElements(GL_TRIANGLES, _distortionMeshIB[eye].numElements(), GL_UNSIGNED_INT, 0);

        _distortionMeshIB[eye].bindAndDrawElementsAs(SL_TRIANGLES);

        glDisableVertexAttribArray(attrPos);
        glDisableVertexAttribArray(attrTimeWarp);
        glDisableVertexAttribArray(attrVignette);
        glDisableVertexAttribArray(attrTexCoordR);
        glDisableVertexAttribArray(attrTexCoordG);
        glDisableVertexAttribArray(attrTexCoordB);
    }

    sp->endUse();

    glEnable(GL_DEPTH_TEST);
    glEnable(GL_CULL_FACE);
}



//-----------------------------------------------------------------------------
/*! Returns the view adjust vector as reported by the hmd for the specified eye
*/
const SLVec3f& SLGLOculus::viewAdjust(SLEye eye)
{
    //@todo find a nicer way to store this (SLEye has a -1 for left and +1 for right eye)
    if (eye == leftEye)
        return _viewAdjust[0];
    else
        return _viewAdjust[1];
}
//-----------------------------------------------------------------------------
/*! Returns an perspective projection matrix for the specified eye
*/
const SLMat4f& SLGLOculus::projection(SLEye eye)
{
    if (eye == leftEye)
        return _projection[0];
    else
        return _projection[1];
}
//-----------------------------------------------------------------------------
/*! Returns an orthogonal projection matrix for the specified eye
*/
const SLMat4f& SLGLOculus::orthoProjection(SLEye eye)
{
    if (eye == leftEye)
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
#ifdef SL_OVR
    ovrFovPort eyeFov[2];
    eyeFov[0] = _hmd->DefaultEyeFov[0];
    eyeFov[1] = _hmd->DefaultEyeFov[1];

    _eyeRenderDesc[0] = ovrHmd_GetRenderDesc(_hmd, ovrEye_Left, eyeFov[0]);
    _eyeRenderDesc[1] = ovrHmd_GetRenderDesc(_hmd, ovrEye_Right, eyeFov[1]);

    float desiredPixelDensity = 1.0f;

    
    OVR::Sizei recommendTex0Size = ovrHmd_GetFovTextureSize(_hmd, ovrEye_Left, eyeFov[0], desiredPixelDensity);
    OVR::Sizei recommendTex1Size = ovrHmd_GetFovTextureSize(_hmd, ovrEye_Right, eyeFov[1], desiredPixelDensity);

    //@todo the calculated size below might not be achievable due to hw limits
    //       make sure to query for actual max dimensions
    _rtSize.set(recommendTex0Size.w + recommendTex1Size.w,
                max(recommendTex0Size.h, recommendTex1Size.h));

    _resolutionScale =  (SLfloat)_rtSize.x / _resolution.x;

    
    // hmd caps
    SLuint hmdCaps = 0;
    if (_lowPersistanceEnabled)
        hmdCaps |= ovrHmdCap_LowPersistence;
    if (_displaySleep)
        hmdCaps |= ovrHmdCap_DisplayOff;

    ovrHmd_SetEnabledCaps(_hmd, hmdCaps);

    SLuint sensorCaps = ovrTrackingCap_Orientation|ovrTrackingCap_MagYawCorrection;
    if (_positionTrackingEnabled)
        sensorCaps |= ovrTrackingCap_Position;

    // update current tracking config if it changed
    if (_startTrackingCaps != sensorCaps)
    {
        ovrHmd_ConfigureTracking(_hmd, sensorCaps, 0);
        _startTrackingCaps = sensorCaps;
    }

    // calculate projections
    ovrMatrix4f projLeft  = ovrMatrix4f_Projection(_eyeRenderDesc[0].Fov, 0.01f, 1000.0f, true);
    ovrMatrix4f projRight = ovrMatrix4f_Projection(_eyeRenderDesc[1].Fov, 0.01f, 1000.0f, true);
    
    float orthoDistance = 0.8f; // 2D is 0.8 meter from camera     
    OVR::Vector2f orthoScale0 = OVR::Vector2f(1.0f) / OVR::Vector2f(_eyeRenderDesc[0].PixelsPerTanAngleAtCenter);
    OVR::Vector2f orthoScale1 = OVR::Vector2f(1.0f) / OVR::Vector2f(_eyeRenderDesc[0].PixelsPerTanAngleAtCenter);
    orthoScale0.x /= (SLfloat)_outputRes.x /_resolution.x;
    orthoScale0.y /= (SLfloat)_outputRes.y /_resolution.y;
    orthoScale1.x /= (SLfloat)_outputRes.x /_resolution.x;
    orthoScale1.y /= (SLfloat)_outputRes.y /_resolution.y;

    ovrMatrix4f orthoProjLeft  = ovrMatrix4f_OrthoSubProjection(projLeft, orthoScale0, orthoDistance,
                                                                _eyeRenderDesc[0].HmdToEyeViewOffset.x);
    ovrMatrix4f orthoProjRight = ovrMatrix4f_OrthoSubProjection(projRight, orthoScale1, orthoDistance,
																_eyeRenderDesc[1].HmdToEyeViewOffset.x);

    
    memcpy(&_projection[0], &projLeft, sizeof(ovrMatrix4f));
    memcpy(&_projection[1], &projRight, sizeof(ovrMatrix4f));
    _projection[0].transpose();
    _projection[1].transpose();
    
    memcpy(&_orthoProjection[0], &orthoProjLeft, sizeof(ovrMatrix4f));
    memcpy(&_orthoProjection[1], &orthoProjRight, sizeof(ovrMatrix4f));
    _orthoProjection[0].transpose();
    _orthoProjection[1].transpose();

    SLMat4f flipY(1.0f, 0.0f, 0.0f, 0.0f,
                  0.0f,-1.0f, 0.0f, 0.0f,
                  0.0f, 0.0f, 1.0f, 0.0f,
                  0.0f, 0.0f, 0.0f, 1.0f);
    _orthoProjection[0] = flipY * _orthoProjection[0];
    _orthoProjection[1] = flipY * _orthoProjection[1];
    
    memcpy(&_viewAdjust[0], &_eyeRenderDesc[0].HmdToEyeViewOffset, sizeof(SLVec3f));
    memcpy(&_viewAdjust[1], &_eyeRenderDesc[1].HmdToEyeViewOffset, sizeof(SLVec3f));
    
#else
    for (SLint i = 0; i < 2; ++i)
    {
        _position[i].set(0, 0, 0);
        _orientation[i].set(0, 0, 0, 1);
        _viewAdjust[i].set((i*2-1)*0.03f, 0, 0); //[-0.03, 0.03]m
    
        ovrFovPort fov;
        fov.DownTan = 1.329f;
        fov.UpTan = 1.329f;
        fov.LeftTan = 1.058f;
        fov.RightTan = 1.092f;
        _projection[i] =  CreateProjection( true, fov,0.01f, 10000.0f );
    
        _orthoProjection[i] = ovrMatrix4f_OrthoSubProjection(_projection[i], SLVec2f(1.0f/(549.618286 * ((SLfloat)_outputRes.x /_resolution.x)), 1.0f/(549.618286 * ((SLfloat)_outputRes.x /_resolution.x))), 0.8f, _viewAdjust[i].x);

        SLMat4f flipY(1.0f, 0.0f, 0.0f, 0.0f,
                      0.0f,-1.0f, 0.0f, 0.0f,
                      0.0f, 0.0f, 1.0f, 0.0f,
                      0.0f, 0.0f, 0.0f, 1.0f);
        _orthoProjection[i] = flipY * _orthoProjection[i];
    }
#endif


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

#ifdef SL_OVR
    _frameTiming = ovrHmd_BeginFrameTiming(_hmd, 0);
    
    // update sensor status
    ovrTrackingState trackState = ovrHmd_GetTrackingState(_hmd, _frameTiming.ScanoutMidpointSeconds);
    
    _isConnected = (trackState.StatusFlags & ovrStatus_HmdConnected) ? true : false;
    _isCameraConnected = (trackState.StatusFlags & ovrStatus_PositionConnected) ? true : false;
    _isPositionTracked = (trackState.StatusFlags & ovrStatus_PositionTracked) ? true : false;

    // code for binding frame buffer here 
    for (int eyeIndex = 0; eyeIndex < 2; ++eyeIndex)
    {
        ovrEyeType eye = (ovrEyeType) eyeIndex; // _hmd->EyeRenderOrder[eyeIndex]; <-- would be the better way
		_eyeRenderPose[eye] = ovrHmd_GetHmdPosePerEye(_hmd, eye);

        _orientation[eyeIndex].set(_eyeRenderPose[eye].Orientation.x, 
                                   _eyeRenderPose[eye].Orientation.y, 
                                   _eyeRenderPose[eye].Orientation.z, 
                                   _eyeRenderPose[eye].Orientation.w);
        if (!_positionTrackingEnabled)
        {   _position[eyeIndex].set(0,0,0);
        }
        else
        {   _position[eyeIndex].set(_eyeRenderPose[eye].Position.x, 
                                    _eyeRenderPose[eye].Position.y, 
                                    _eyeRenderPose[eye].Position.z);
        }
    }
#endif
}

//-----------------------------------------------------------------------------
/*! endFrame handles correct frame timing
*/
void SLGLOculus::endFrame(SLint width, SLint height, SLuint tex)
{
#ifdef SL_OVR
    // wait till timewarp point to reduce latency
    ovr_WaitTillTime(_frameTiming.TimewarpPointSeconds);
    
    renderDistortion(width, height, tex);

    if (_hmd->HmdCaps & ovrHmdCap_ExtendDesktop)
        glFlush();

    ovrHmd_EndFrameTiming(_hmd);
#else
	renderDistortion(width, height, tex);
#endif
}
//-----------------------------------------------------------------------------
/*! Returns the Oculus orientation as quaternion. If no Oculus Rift is 
recognized it returns a unit quaternion.
*/
const SLQuat4f& SLGLOculus::orientation(SLEye eye)
{
    if (eye == leftEye) return _orientation[0];
    else                return _orientation[1];
}
//-----------------------------------------------------------------------------

/*! Returns the Oculus position.
*/
const SLVec3f& SLGLOculus::position(SLEye eye)
{
    if (eye == leftEye) return _position[0];
    else                return _position[1];
}
//-----------------------------------------------------------------------------

/*! enable or disable low persistance
*/
void SLGLOculus::lowPersistance(SLbool val)
{
    if (val == _lowPersistanceEnabled)
        return;

    _lowPersistanceEnabled = val;
    _hmdSettingsChanged = true;
}
//-----------------------------------------------------------------------------

/*! enable or disable timewarp
*/
void SLGLOculus::timeWarp(SLbool val)
{
    if (val == _timeWarpEnabled)
        return;

    _timeWarpEnabled = val;
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
    _hmdSettingsChanged = true;
}
//-----------------------------------------------------------------------------

/*! enable or disable position tracking
*/
void SLGLOculus::displaySleep(SLbool val)
{
    if (val == _displaySleep)
        return;

    _displaySleep = val;
    _hmdSettingsChanged = true;
}
//-----------------------------------------------------------------------------

