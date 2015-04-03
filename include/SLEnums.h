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

#include <stdafx.h>

//-----------------------------------------------------------------------------
//! Keyboard key codes enumeration
enum SLKey
{
    KeyNone=0, 
    KeySpace=32,
    KeyTab=256,KeyEnter,KeyEsc,KeyBackspace,KeyDelete,
    KeyUp,KeyDown,KeyRight,KeyLeft, 
    KeyHome, KeyEnd, KeyInsert, KeyPageUp, KeyPageDown,
    KeyNP0,KeyNP1,KeyNP2,KeyNP3,KeyNP4,KeyNP5,KeyNP6,KeyNP7,KeyNP8,KeyNP9,
    KeyNPDivide,KeyNPMultiply,KeyNPAdd,KeyNPSubtract,KeyNPEnter,KeyNPDecimal,
    KeyF1,KeyF2,KeyF3,KeyF4,KeyF5,KeyF6,KeyF7,KeyF8,KeyF9,KeyF10,KeyF11,KeyF12, 
    KeySuper=0x00100000,KeyShift=0x00200000,KeyCtrl=0x00400000,KeyAlt=0x00800000
};

//-----------------------------------------------------------------------------
//! Rendering type enumeration
enum SLRenderer
{
    renderGL=0,
    renderRT=1,
    renderPT=2
};

//-----------------------------------------------------------------------------
//! Coordinate axis enumeration
enum SLAxis
{
    XAxis=0, 
    YAxis=1,
    ZAxis=2
};

//-----------------------------------------------------------------------------
//! SLCmd enumerates all possible menu and keyboard commands
enum SLCmd
{
    cmdMenu,
    cmdAboutToggle,
    cmdHelpToggle,
    cmdCreditsToggle,
    cmdSceneInfoToggle,
    cmdQuit,
   
    cmdSceneSmallTest,   // Loads the different scenes
    cmdSceneFigure,   
    cmdSceneMeshLoad,
    cmdSceneVRSizeTest,
    cmdSceneLargeModel,
    cmdSceneRevolver,
    cmdSceneTextureFilter,
    cmdSceneTextureBlend,
    cmdSceneFrustumCull1,
    cmdSceneFrustumCull2,

    cmdScenePerVertexBlinn,
    cmdScenePerPixelBlinn,
    cmdScenePerVertexWave,
    cmdSceneWater,
    cmdSceneBumpNormal,
    cmdSceneBumpParallax,
    cmdSceneEarth,
    cmdSceneMassAnimation,
    cmdSceneTerrain,

    cmdSceneSkeletalAnimation,
    cmdSceneNodeAnimation,
    cmdSceneAstroboyArmyGPU,
    cmdSceneAstroboyArmyCPU,

    cmdSceneRTMuttenzerBox,
    cmdSceneRTSpheres,
    cmdSceneRTSoftShadows,
    cmdSceneRTDoF,
    cmdSceneRTLens,

    cmdMultiSampleToggle,// Toggles multisampling
    cmdDepthTestToggle,  // Toggles the depth test flag
    cmdFrustCullToggle,  // Toggles frustum culling
    cmdWaitEventsToggle, // Toggles the wait event flag

    cmdBBoxToggle,       // Toggles bounding box drawing bit
    cmdAxisToggle,       // Toggles bounding box drawing bit
    cmdFaceCullToggle,   // Toggles face culling
    cmdWireMeshToggle,   // Toggles wireframe drawing bit
    cmdNormalsToggle,    // Toggles normale drawing bit
    cmdAnimationToggle,  // Animation bit toggle
    cmdTextureToggle,    // Texture drawing bit toggle
    cmdVoxelsToggle,     // Voxel drawing bit toggle
   
    cmdProjPersp,        // Perspective projection
    cmdProjOrtho,        // Orthographic projection
    cmdProjSideBySide,   // side-by-side
    cmdProjSideBySideP,  // side-by-side proportional
    cmdProjSideBySideD,  // Oculus Rift stereo mode
    cmdProjLineByLine,   // line-by-line
    cmdProjColumnByColumn,// column-by-column
    cmdProjPixelByPixel, // checkerboard pattern (DLP3D)
    cmdProjColorRC,      // color masking for red-cyan anaglyphs
    cmdProjColorRG,      // color masking for red-green anaglyphs
    cmdProjColorRB,      // color masking for red-blue anaglyphs
    cmdProjColorYB,      // color masking for yellow-blue anaglyphs (ColorCode 3D)
   
    cmdCamReset,         // Resets to the initial camera view
    cmdUseSceneViewCamera,  // make the editor camera active
    cmdCamDeviceRotOn,   // Use devices rotation (mobile or Oculus Rift) for camera view
    cmdCamDeviceRotOff,  // Don't use devices rotation (mobile or Oculus Rift) for camera view
    cmdCamDeviceRotToggle, // Toggle devices rotation (mobile or Oculus Rift) for camera view
    cmdCamEyeSepInc,     // Cameras eye separation distance increase
    cmdCamEyeSepDec,     // Cameras eye separation distance decrease
    cmdCamFocalDistInc,  // Cameras focal distance increase
    cmdCamFocalDistDec,  // Cameras focal distance decrease
    cmdCamFOVInc,        // Cameras field of view increase
    cmdCamFOVDec,        // Cameras field of view decrease
    cmdCamAnimTurnYUp,   // Sets turntable camera animation w. Y axis up
    cmdCamAnimTurnZUp,   // Sets turntable camera animation w. Z axis up
    cmdCamAnimWalkYUp,   // Sets 1st person walking camera animation w. Y axis up
    cmdCamAnimWalkZUp,   // Sets 1st person walking camera animation w. Z axis up
    cmdCamAnimFly1stP,   // Sets 1st person flying camera animation
    cmdCamSpeedLimitInc, // Increments the speed limit by 10%
    cmdCamSpeedLimitDec, // Decrements the speed limit by 10%


    cmdStatsToggle,      // Toggles statistics on/off

    cmdRenderOpenGL,     // Render with GL
    cmdRTContinuously,   // Do ray tracing continuously
    cmdRTDistributed,    // Do ray tracing distributed
    cmdRTStop,           // Stop ray tracing
    cmdRT1,              //1: Do ray tracing with max. depth 1
    cmdRT2,              //2: Do ray tracing with max. depth 2
    cmdRT3,              //3: Do ray tracing with max. depth 3
    cmdRT4,              //4: Do ray tracing with max. depth 4
    cmdRT5,              //5: Do ray tracing with max. depth 5
    cmdRT6,              //6: Do ray tracing with max. depth 6
    cmdRT7,              //7: Do ray tracing with max. depth 7
    cmdRT8,              //8: Do ray tracing with max. depth 8
    cmdRT9,              //9: Do ray tracing with max. depth 9
    cmdRT0,              //0: Do ray tracing with max. depth
    cmdRTSaveImage,      // Save the ray tracing image
    cmdPT1,              // Do pathtracing 1 Rays
    cmdPT10,             // Do pathtracing 10 Rays
    cmdPT50,             // Do pathtracing 50 Rays
    cmdPT100,            // Do pathtracing 100 Rays
    cmdPT500,            // Do pathtracing 500 Rays
    cmdPT1000,           // Do pathtracing 1000 Rays
    cmdPT5000,           // Do pathtracing 5000 Rays
    cmdPT10000,          // Do pathtracing 10000 Rays
    cmdPTSaveImage       // Save the ray tracing image
};

//-----------------------------------------------------------------------------
//! Mouse button codes
enum SLMouseButton
{
    ButtonNone=0,
    ButtonLeft,
    ButtonMiddle,
    ButtonRight
};

//-----------------------------------------------------------------------------
//! Enumeration for text alignment in a box
enum SLTextAlign
{
    topLeft, topCenter, topRight,
    centerLeft, centerCenter, centerRight,
    bottomLeft, bottomCenter, bottomRight
};

//-----------------------------------------------------------------------------
//! Enumeration for possible camera animation types
enum SLCamAnim
{
    turntableYUp = 0,
    turntableZUp,
    walkingYUp,
    walkingZUp
};

//-----------------------------------------------------------------------------
//! Enumeration for differen camera projections
enum SLProjection
{
    monoPerspective = 0, //! standard mono pinhole perspective projection
    monoOrthographic,    //! standard mono orthgraphic projection
    stereoSideBySide,    //! side-by-side
    stereoSideBySideP,   //! side-by-side proportional for mirror stereoscopes
    stereoSideBySideD,   //! side-by-side distorted for Oculus Rift like glasses
    stereoLineByLine,    //! line-by-line
    stereoColumnByColumn,//! column-by-column
    stereoPixelByPixel,  //! checkerboard pattern (DLP3D)
    stereoColorRC,       //! color masking for red-cyan anaglyphs
    stereoColorRG,       //! color masking for red-green anaglyphs
    stereoColorRB,       //! color masking for red-blue anaglyphs
    stereoColorYB        //! color masking for yellow-blue anaglyphs (ColorCode 3D)
};
//-----------------------------------------------------------------------------
//! Enumeration for stereo eye type used for camera projection
enum SLEye
{
    leftEye   =-1,
    centerEye = 0,
    rightEye  = 1
};
//-----------------------------------------------------------------------------
//! Enumeration for animation modes
enum SLAnimInterpolation
{
    AI_Linear,
    AI_Bezier
};
//-----------------------------------------------------------------------------
//! Enumeration for animation modes
enum SLAnimLooping
{
    AL_once = 0,          //!< play once
    AL_loop = 1,          //!< loop
    AL_pingPong = 2,      //!< play once in two directions
    AL_pingPongLoop = 3   //!< loop forward and backwards
};

//-----------------------------------------------------------------------------
//! Enumeration for animation easing curves
/*! 
Enumatrations copied from Qt class QEasingCurve. 
See http://qt-project.org/doc/qt-4.8/qeasingcurve.html#Type-enum
*/
enum SLEasingCurve
{
    EC_linear = 0,      //!< linear easing with constant velocity
    EC_inQuad = 1,      //!< quadratic easing in, acceleration from zero velocity
    EC_outQuad = 2,     //!< quadratic easing out, decelerating to zero velocity
    EC_inOutQuad = 3,   //!< quadratic easing in and then out  
    EC_outInQuad = 4,   //!< quadratic easing out and then in
    EC_inCubic = 5,     //!< qubic in easing in, acceleration from zero velocity
    EC_outCubic = 6,    //!< qubic easing out, decelerating to zero velocity
    EC_inOutCubic = 7,  //!< qubic easing in and then out 
    EC_outInCubic = 8,  //!< qubic easing out and then in
    EC_inQuart = 9,     //!< quartic easing in, acceleration from zero velocity
    EC_outQuart = 10,   //!< quartic easing out, decelerating to zero velocity
    EC_inOutQuart = 11, //!< quartic easing in and then out 
    EC_outInQuart = 12, //!< quartic easing out and then in
    EC_inQuint = 13,    //!< quintic easing in, acceleration from zero velocity
    EC_outQuint = 14,   //!< quintic easing out, decelerating to zero velocity
    EC_inOutQuint = 15, //!< quintic easing in and then out 
    EC_outInQuint = 16, //!< quintic easing out and then in
    EC_inSine = 17,     //!< sine ieasing in, acceleration from zero velocity
    EC_outSine = 18,    //!< sine easing out, decelerating to zero velocity
    EC_inOutSine = 19,  //!< sine easing in and then out  
    EC_outInSine = 20,  //!< sine easing out and then in
};
//-----------------------------------------------------------------------------
//! Describes the relative space a transformation is applied in.
enum SLTransformSpace
{
    TS_World, 
    TS_Parent,  
    TS_Object,
};
//-----------------------------------------------------------------------------
//! Skinning methods
enum SLSkinMethod
{
    SM_HardwareSkinning, //!< Do vertex skinning on the GPU
    SM_SoftwareSkinning  //!< Do vertex skinning on the CPU
};

//-----------------------------------------------------------------------------
//! Shader type enumeration for vertex or fragment (pixel) shader
enum SLShaderType
{
    NoShader=0,
    VertexShader=1,
    FragmentShader=2
};

//-----------------------------------------------------------------------------
//! Enumeration for standard preloaded shader programs in SLScene::_shaderProgs
enum SLStdShaderProg
{
    ColorAttribute,
    ColorUniform,
    PerVrtBlinn,
    PerVrtBlinnTex,
    TextureOnly,
    PerPixBlinn,
    PerPixBlinnTex,
    BumpNormal,
    BumpNormalParallax,
    FontTex,
    StereoOculus,
    StereoOculusDistortionMesh
};

//-----------------------------------------------------------------------------
//! Type definition for GLSL uniform1f variables that change per frame.
enum SLUF1Type
{
    UF1Const,   //!< constant value
    UF1IncDec,  //!< never ending loop from min to max and max to min
    UF1IncInc,  //!< never ending loop from min to max
    UF1Inc,     //!< never ending increment
    UF1Random,  //!< random values between min and max
    UF1Seconds  //!< seconds since the process has started
};

//-----------------------------------------------------------------------------
#endif
