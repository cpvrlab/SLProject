//#############################################################################
//  File:      SL/SLEnums.h
//  Author:    Marcus Hudritsch
//  Date:      July 2014
//  Codestyle: https://github.com/cpvrlab/SLProject/wiki/Coding-Style-Guidelines
//  Copyright: Marcus Hudritsch
//             This software is provide under the GNU General Public License
//             Please visit: http://opensource.org/licenses/GPL-3.0
//#############################################################################

#ifndef SLENUMS_H
#define SLENUMS_H

//#include <stdafx.h> // gets included before SL.h

//-----------------------------------------------------------------------------
//! Keyboard key codes enumeration
enum SLKey
{   K_none=0, 
    K_space=32,
    K_tab=256, K_enter, K_esc, K_backspace, K_delete,
    K_up, K_down, K_right, K_left, 
    K_home, K_end, K_insert, K_pageUp, K_pageDown,
    K_NP0, K_NP1, K_NP2, K_NP3, K_NP4, K_NP5, K_NP6, K_NP7, K_NP8, K_NP9,
    K_NPDivide, K_NPMultiply, K_NPAdd, K_NPSubtract, K_NPEnter, K_NPDecimal,
    K_F1, K_F2, K_F3, K_F4, K_F5, K_F6, K_F7, K_F8, K_F9, K_F10, K_F11, K_F12, 
    K_super=0x00100000, K_shift=0x00200000, K_ctrl=0x00400000, K_alt=0x00800000
};
//-----------------------------------------------------------------------------
//! Rendering type enumeration
enum SLRenderType
{   RT_gl=0,    //!< OpenGL
    RT_rt=1,    //!< Ray Tracing
    RT_pt=2     //!< Path Tracing
};
//-----------------------------------------------------------------------------
//! Coordinate axis enumeration
enum SLAxis
{   A_x=0, 
    A_Y=1,
    A_z=2
};
//-----------------------------------------------------------------------------
//! Pixel format according to OpenGL pixel format defines
enum SLPixelFormat
{
    PF_unknown = 0,
    PF_yuv_420_888 = 1,         // YUV format from Android not supported in GL

    PF_alpha = 0x1906,          // ES2 ES3 GL2
    PF_luminance = 0x1909,      // ES2 ES3 GL2
    PF_luminance_alpha = 0x190A,// ES2 ES3 GL2
    PF_intensity = 0x8049,      //         GL2
    PF_green = 0x1904,          //         GL2
    PF_blue = 0x1905,           //         GL2
    PF_depth_component = 0x1902,//     ES3 GL2     GL4

    PF_red  = 0x1903,           //     ES3 GL2 GL3 GL4
    PF_rg   = 0x8227,           //     ES3     GL3 GL4
    PF_rgb  = 0x1907,           // ES2 ES3 GL2 GL3 GL4
    PF_rgba = 0x1908,           // ES2 ES3 GL2 GL3 GL4
    PF_bgr  = 0x80E0,           //         GL2 GL3 GL4
    PF_bgra = 0x80E1,           //         GL2 GL3 GL4

    PF_rg_integer = 0x8228,     //     ES3         GL4
    PF_red_integer = 0x8D94,    //     ES3         GL4
    PF_rgb_integer = 0x8D98,    //     ES3         GL4
    PF_rgba_integer = 0x8D99,   //     ES3         GL4
    PF_bgr_integer = 0x8D9A,    //                 GL4
    PF_bgra_integer = 0x8D9B,   //                 GL4

};
//-----------------------------------------------------------------------------
//! SLCommand enumerates all possible menu and keyboard commands
enum SLCommand
{   
    C_sceneFromFile = -2,   // Custom assted loaded over menu
    C_sceneEmpty = -1,      // No data in scene
    C_sceneAll = 0,         // Loads all scenes one after the other
    C_sceneMinimal,
    C_sceneFigure,
    C_sceneMeshLoad,
    C_sceneVRSizeTest,
    C_sceneLargeModel,
    C_sceneChristoffel,
    C_sceneRevolver,
    C_sceneTextureFilter,
    C_sceneTextureBlend,
    C_sceneTextureVideo,
    C_sceneFrustumCull,
    C_sceneMassiveData,

    C_scenePerVertexBlinn,
    C_scenePerPixelBlinn,
    C_scenePerVertexWave,
    C_sceneWater,
    C_sceneBumpNormal,
    C_sceneBumpParallax,
    C_sceneEarth,
    C_sceneMassAnimation,
    C_sceneTerrain,

    C_sceneSkeletalAnimation,
    C_sceneNodeAnimation,
    C_sceneAstroboyArmy,

    C_sceneTrackChessboard,
    C_sceneTrackAruco,
    C_sceneTrackFeatures2D,

    C_sceneRTMuttenzerBox,
    C_sceneRTSpheres,
    C_sceneRTSoftShadows,
    C_sceneRTDoF,
    C_sceneRTLens,
    C_sceneRTTest,
    
    C_menu,
    C_aboutToggle,
    C_helpToggle,
    C_creditsToggle,
    C_noCalibToggle,
    C_sceneInfoToggle,
    C_quit,
    C_clearCalibration,

    C_multiSampleToggle,// Toggles multisampling
    C_depthTestToggle,  // Toggles the depth test flag
    C_frustCullToggle,  // Toggles frustum culling
    C_waitEventsToggle, // Toggles the wait event flag

    C_skeletonToggle,   // Toggles skeleton drawing bit
    C_bBoxToggle,       // Toggles bounding box drawing bit
    C_axisToggle,       // Toggles axis drawing bit
    C_faceCullToggle,   // Toggles face culling
    C_wireMeshToggle,   // Toggles wireframe drawing bit
    C_normalsToggle,    // Toggles normal drawing bit
    C_animationToggle,  // Animation bit toggle
    C_textureToggle,    // Texture drawing bit toggle
    C_voxelsToggle,     // Voxel drawing bit toggle
   
    C_projPersp,        // Perspective projection
    C_projOrtho,        // Orthographic projection
    C_projSideBySide,   // side-by-side
    C_projSideBySideP,  // side-by-side proportional
    C_projSideBySideD,  // Oculus Rift stereo mode
    C_projLineByLine,   // line-by-line
    C_projColumnByColumn,// column-by-column
    C_projPixelByPixel, // checkerboard pattern (DLP3D)
    C_projColorRC,      // color masking for red-cyan anaglyphs
    C_projColorRG,      // color masking for red-green anaglyphs
    C_projColorRB,      // color masking for red-blue anaglyphs
    C_projColorYB,      // color masking for yellow-blue anaglyphs (ColorCode 3D)
   
    C_camReset,         // Resets to the initial camera view
    C_useSceneViewCamera,  // make the editor camera active
    C_camDeviceRotOn,   // Use devices rotation (mobile or Oculus Rift) for camera view
    C_camDeviceRotOff,  // Don't use devices rotation (mobile or Oculus Rift) for camera view
    C_camDeviceRotToggle, // Toggle devices rotation (mobile or Oculus Rift) for camera view
    C_camEyeSepInc,     // Cameras eye separation distance increase
    C_camEyeSepDec,     // Cameras eye separation distance decrease
    C_camFocalDistInc,  // Cameras focal distance increase
    C_camFocalDistDec,  // Cameras focal distance decrease
    C_camFOVInc,        // Cameras field of view increase
    C_camFOVDec,        // Cameras field of view decrease
    C_camAnimTurnYUp,   // Sets turntable camera animation w. Y axis up
    C_camAnimTurnZUp,   // Sets turntable camera animation w. Z axis up
    C_camAnimWalkYUp,   // Sets 1st person walking camera animation w. Y axis up
    C_camAnimWalkZUp,   // Sets 1st person walking camera animation w. Z axis up
    C_camAnimFly1stP,   // Sets 1st person flying camera animation
    C_camSpeedLimitInc, // Increments the speed limit by 10%
    C_camSpeedLimitDec, // Decrements the speed limit by 10%

    C_statsTimingToggle,
    C_statsRendererToggle,
    C_statsMemoryToggle,
    C_statsVideoToggle,
    C_statsCameraToggle,

    C_dpiInc,           // Increase DPI 10%
    C_dpiDec,           // Decrease DPI 10%

    C_renderOpenGL,     // Render with GL
    C_rtContinuously,   // Do ray tracing continuously
    C_rtDistributed,    // Do ray tracing distributed
    C_rtStop,           // Stop ray tracing
    C_rt1,              //1: Do ray tracing with max. depth 1
    C_rt2,              //2: Do ray tracing with max. depth 2
    C_rt3,              //3: Do ray tracing with max. depth 3
    C_rt4,              //4: Do ray tracing with max. depth 4
    C_rt5,              //5: Do ray tracing with max. depth 5
    C_rt6,              //6: Do ray tracing with max. depth 6
    C_rt7,              //7: Do ray tracing with max. depth 7
    C_rt8,              //8: Do ray tracing with max. depth 8
    C_rt9,              //9: Do ray tracing with max. depth 9
    C_rt0,              //0: Do ray tracing with max. depth
    C_rtSaveImage,      // Save the ray tracing image
    C_pt1,              // Do pathtracing 1 Rays
    C_pt10,             // Do pathtracing 10 Rays
    C_pt50,             // Do pathtracing 50 Rays
    C_pt100,            // Do pathtracing 100 Rays
    C_pt500,            // Do pathtracing 500 Rays
    C_pt1000,           // Do pathtracing 1000 Rays
    C_pt5000,           // Do pathtracing 5000 Rays
    C_pt10000,          // Do pathtracing 10000 Rays
    C_ptSaveImage       // Save the ray tracing image
};
//-----------------------------------------------------------------------------
//! Mouse button codes
enum SLMouseButton
{   MB_none,
    MB_left,
    MB_middle,
    MB_right
};
//-----------------------------------------------------------------------------
//! Enumeration for text alignment in a box
enum SLTextAlign
{   TA_topLeft, TA_topCenter, TA_topRight,
    TA_centerLeft, TA_centerCenter, TA_centerRight,
    TA_bottomLeft, TA_bottomCenter, TA_bottomRight
};
//-----------------------------------------------------------------------------
//! Enumeration for possible camera animation types
enum SLCamAnim
{   CA_turntableYUp,
    CA_turntableZUp,
    CA_walkingYUp,
    CA_walkingZUp
};
//-----------------------------------------------------------------------------
//! Enumeration for different camera projections
enum SLProjection
{   P_monoPerspective,     //!< standard mono pinhole perspective projection
    P_monoOrthographic,    //!< standard mono orthographic projection
    P_stereoSideBySide,    //!< side-by-side
    P_stereoSideBySideP,   //!< side-by-side proportional for mirror stereoscopes
    P_stereoSideBySideD,   //!< side-by-side distorted for Oculus Rift like glasses
    P_stereoLineByLine,    //!< line-by-line
    P_stereoColumnByColumn,//!< column-by-column
    P_stereoPixelByPixel,  //!< checkerboard pattern (DLP3D)
    P_stereoColorRC,       //!< color masking for red-cyan anaglyphs
    P_stereoColorRG,       //!< color masking for red-green anaglyphs
    P_stereoColorRB,       //!< color masking for red-blue anaglyphs
    P_stereoColorYB        //!< color masking for yellow-blue anaglyphs (ColorCode 3D)
};
//-----------------------------------------------------------------------------
//! Enumeration for stereo eye type used for camera projection
enum SLEyeType
{   ET_left   =-1,
    ET_center = 0,
    ET_right  = 1
};
//-----------------------------------------------------------------------------
//! Enumeration for animation modes
enum SLAnimInterpolation
{   AI_linear,
    AI_bezier
};
//-----------------------------------------------------------------------------
//! Enumeration for animation modes
enum SLAnimLooping
{   AL_once = 0,          //!< play once
    AL_loop = 1,          //!< loop
    AL_pingPong = 2,      //!< play once in two directions
    AL_pingPongLoop = 3   //!< loop forward and backwards
};
//-----------------------------------------------------------------------------
//! Enumeration for animation easing curves
/*! 
Enumerations copied from Qt class QEasingCurve. 
See http://qt-project.org/doc/qt-4.8/qeasingcurve.html#Type-enum
*/
enum SLEasingCurve
{   EC_linear = 0,      //!< linear easing with constant velocity
    EC_inQuad = 1,      //!< quadratic easing in, acceleration from zero velocity
    EC_outQuad = 2,     //!< quadratic easing out, decelerating to zero velocity
    EC_inOutQuad = 3,   //!< quadratic easing in and then out  
    EC_outInQuad = 4,   //!< quadratic easing out and then in
    EC_inCubic = 5,     //!< cubic in easing in, acceleration from zero velocity
    EC_outCubic = 6,    //!< cubic easing out, decelerating to zero velocity
    EC_inOutCubic = 7,  //!< cubic easing in and then out 
    EC_outInCubic = 8,  //!< cubic easing out and then in
    EC_inQuart = 9,     //!< quartic easing in, acceleration from zero velocity
    EC_outQuart = 10,   //!< quartic easing out, decelerating to zero velocity
    EC_inOutQuart = 11, //!< quartic easing in and then out 
    EC_outInQuart = 12, //!< quartic easing out and then in
    EC_inQuint = 13,    //!< quintic easing in, acceleration from zero velocity
    EC_outQuint = 14,   //!< quintic easing out, decelerating to zero velocity
    EC_inOutQuint = 15, //!< quintic easing in and then out 
    EC_outInQuint = 16, //!< quintic easing out and then in
    EC_inSine = 17,     //!< sine easing in, acceleration from zero velocity
    EC_outSine = 18,    //!< sine easing out, decelerating to zero velocity
    EC_inOutSine = 19,  //!< sine easing in and then out  
    EC_outInSine = 20,  //!< sine easing out and then in
};
//-----------------------------------------------------------------------------
//! Describes the relative space a transformation is applied in.
enum SLTransformSpace
{   TS_world, 
    TS_parent,  
    TS_object,
};
//-----------------------------------------------------------------------------
//! Skinning methods
enum SLSkinMethod
{   SM_hardware, //!< Do vertex skinning on the GPU
    SM_software  //!< Do vertex skinning on the CPU
};
//-----------------------------------------------------------------------------
//! Shader type enumeration for vertex or fragment (pixel) shader
enum SLShaderType
{   ST_none,
    ST_vertex,
    ST_fragment,
    ST_geometry,
    ST_tesselation
};
//-----------------------------------------------------------------------------
//! Enumeration for standard preloaded shader programs in SLScene::_shaderProgs
enum SLShaderProg
{   SP_colorAttribute,
    SP_colorUniform,
    SP_perVrtBlinn,
    SP_perVrtBlinnColorAttrib,
    SP_perVrtBlinnTex,
    SP_TextureOnly,
    SP_perPixBlinn,
    SP_perPixBlinnTex,
    SP_bumpNormal,
    SP_bumpNormalParallax,
    SP_fontTex,
    SP_stereoOculus,
    SP_stereoOculusDistortion
};
//-----------------------------------------------------------------------------
//! Type definition for GLSL uniform1f variables that change per frame.
enum SLUniformType
{   UT_const,   //!< constant value
    UT_incDec,  //!< never ending loop from min to max and max to min
    UT_incInc,  //!< never ending loop from min to max
    UT_inc,     //!< never ending increment
    UT_random,  //!< random values between min and max
    UT_seconds  //!< seconds since the process has started
};
//-----------------------------------------------------------------------------
// @todo build a dedicated log class that defines this verbosity levels
enum SLLogVerbosity
{   LV_quiet = 0,
    LV_minimal = 1,
    LV_normal = 2,
    LV_detailed = 3,
    LV_diagnostic = 4
};
//-----------------------------------------------------------------------------
//! OpenCV Calibration state
enum SLCVCalibState 
{   CS_uncalibrated,    //!< The camera is not calibrated (no calibration found)
    CS_calibrateStream, //!< The calibration is running with live video stream
    CS_calibrateGrab,   //!< The calibration is running and an image should be grabbed
    CS_startCalculating,//!< The calibration starts during the next frame
    CS_calibrated,      //!< The camera is calibrated 
    CS_approximated     //!< The camera intrinsics where approximated
};
//-----------------------------------------------------------------------------
//! OpenCV feature type
enum SLCVFeatureType
{   FT_SIFT,    //!<
    FT_SURF,    //!<
    FT_ORB      //!< 
};
//-----------------------------------------------------------------------------
//! Video type if multiple exist on mobile devices
enum SLVideoType
{   VT_NONE =  0,  //!< No camera needed
    VT_MAIN =  1,  //!< Back facing camera on mobile devices
    VT_SCND =  2,  //!< Front facing camera on mobile devices
};
//-----------------------------------------------------------------------------
#endif
