//#############################################################################
//  File:      sl/SLEnums.h
//  Date:      July 2014
//  Codestyle: https://github.com/cpvrlab/SLProject/wiki/SLProject-Coding-Style
//  Authors:   Marcus Hudritsch
//  License:   This software is provided under the GNU General Public License
//             Please visit: http://opensource.org/licenses/GPL-3.0
//#############################################################################

#ifndef SLENUMSID_H
#define SLENUMSID_H

//-----------------------------------------------------------------------------
//! Keyboard key codes enumeration
enum SLKey
{
    K_none  = 0,
    K_space = 32,
    K_tab   = 256,
    K_enter,
    K_esc,
    K_backspace,
    K_delete,
    K_up,
    K_down,
    K_right,
    K_left,
    K_home,
    K_end,
    K_insert,
    K_pageUp,
    K_pageDown,
    K_NP0,
    K_NP1,
    K_NP2,
    K_NP3,
    K_NP4,
    K_NP5,
    K_NP6,
    K_NP7,
    K_NP8,
    K_NP9,
    K_NPDivide,
    K_NPMultiply,
    K_NPAdd,
    K_NPSubtract,
    K_NPEnter,
    K_NPDecimal,
    K_F1,
    K_F2,
    K_F3,
    K_F4,
    K_F5,
    K_F6,
    K_F7,
    K_F8,
    K_F9,
    K_F10,
    K_F11,
    K_F12,
    K_super = 0x00100000,
    K_shift = 0x00200000,
    K_ctrl  = 0x00400000,
    K_alt   = 0x00800000
};
//-----------------------------------------------------------------------------
//! Rendering type enumeration
enum SLRenderType
{
    RT_gl       = 0, //!< OpenGL
    RT_rt       = 1, //!< Ray Tracing
    RT_pt       = 2, //!< Path Tracing
    RT_ct       = 3, //!< Voxel Cone Tracing
    RT_optix_rt = 4, //!< Ray Tracing with OptiX
    RT_optix_pt = 5  //!< Path Tracing with OptiX
};
//-----------------------------------------------------------------------------
//! Coordinate axis enumeration
enum SLAxis
{
    A_x = 0,
    A_Y = 1,
    A_z = 2
};
//-----------------------------------------------------------------------------
//! SLCommand enumerates all possible menu and keyboard commands
enum SLSceneID
{
    SID_FromFile = -2, // Custom asset loaded over menu
    SID_Empty    = 0,  // No data in scene
    SID_Minimal,
    SID_Figure,
    SID_MeshLoad,
    SID_Revolver,
    SID_TextureFilter,
    SID_TextureBlend,
#ifdef SL_BUILD_WITH_KTX
    SID_TextureCompression,
#endif
    SID_FrustumCull,
    SID_2Dand3DText,
    SID_PointClouds,

    SID_ShaderPerVertexBlinn,
    SID_ShaderPerPixelBlinn,
    SID_ShaderPerVertexWave,
    SID_ShaderCook,
    SID_ShaderIBL,
    SID_ShaderBumpNormal,
    SID_ShaderBumpParallax,
    SID_ShaderSkyBox,
    SID_ShaderEarth,
    SID_ShaderVoxelConeDemo,
    SID_ShadowMappingBasicScene,
    SID_ShadowMappingLightTypes,
    SID_ShadowMappingPointLights,
    SID_ShadowMappingSpotLights,
    SID_ShadowMappingCascaded,

    SID_SuzannePerPixBlinn,
    SID_SuzannePerPixBlinnTm,
    SID_SuzannePerPixBlinnNm,
    SID_SuzannePerPixBlinnAo,
    SID_SuzannePerPixBlinnSm,
    SID_SuzannePerPixBlinnTmNm,
    SID_SuzannePerPixBlinnTmAo,
    SID_SuzannePerPixBlinnNmAo,
    SID_SuzannePerPixBlinnTmSm,
    SID_SuzannePerPixBlinnNmSm,
    SID_SuzannePerPixBlinnAoSm,
    SID_SuzannePerPixBlinnTmNmAo,
    SID_SuzannePerPixBlinnTmNmSm,
    SID_SuzannePerPixBlinnTmNmAoSm,

    SID_VolumeRayCast,
    SID_VolumeRayCastLighted,

    SID_AnimationMass,
    SID_AnimationSkeletal,
    SID_AnimationNode,
    SID_AnimationAstroboyArmy,

    SID_VideoTextureLive,
    SID_VideoTextureFile,
    SID_VideoCalibrateMain,
    SID_VideoCalibrateScnd,
    SID_VideoTrackChessMain,
    SID_VideoTrackChessScnd,
    SID_VideoTrackArucoMain,
    SID_VideoTrackArucoScnd,
    SID_VideoTrackArucoCubeMain,
    SID_VideoTrackFeature2DMain,
    SID_VideoTrackFeature2DScnd,
    SID_VideoTrackFaceMain,
    SID_VideoTrackFaceScnd,
    SID_VideoSensorAR,
#ifdef SL_BUILD_WAI
    SID_VideoTrackWAI,
#endif

    SID_RTMuttenzerBox,
    SID_RTSpheres,
    SID_RTSoftShadows,
    SID_RTDoF,
    SID_RTLens,
    SID_RTTest,

    SID_Benchmark1_LargeModel,
    SID_Benchmark2_MassiveNodes,
    SID_Benchmark3_NodeAnimations,
    SID_Benchmark4_SkinnedAnimations,
    SID_Benchmark5_LevelOfDetail,

    SID_ErlebARBielBFH,
    SID_ErlebARBernChristoffel,
    SID_ErlebARAugustaRauricaTmp,
    SID_ErlebARAugustaRauricaTht,
    SID_ErlebARAugustaRauricaTmpTht,
    SID_ErlebARAventicumAmphiteatre,
    SID_ErlebARAventicumCigognier,
    SID_ErlebARAventicumTheatre,
    SID_ErlebARSutzKirchrain18,

    SID_Maximal
};
//-----------------------------------------------------------------------------
//! Mouse button codes
enum SLMouseButton
{
    MB_none,
    MB_left,
    MB_middle,
    MB_right
};
//-----------------------------------------------------------------------------
//! Enumeration for text alignment in a box
enum SLTextAlign
{
    TA_topLeft,
    TA_topCenter,
    TA_topRight,
    TA_centerLeft,
    TA_centerCenter,
    TA_centerRight,
    TA_bottomLeft,
    TA_bottomCenter,
    TA_bottomRight
};
//-----------------------------------------------------------------------------
//! Enumeration for available camera animation types
enum SLCamAnim
{
    CA_turntableYUp,    //!< Orbiting around central object w. turntable rotation around y & right axis.
    CA_turntableZUp,    //!< Orbiting around central object w. turntable rotation around z & right axis.
    CA_trackball,       //!< Orbiting around central object w. one rotation around one axis
    CA_walkingYUp,      //!< Walk translation with AWSD and look around rotation around y & right axis.
    CA_walkingZUp,      //!< Walk translation with AWSD and look around rotation around z & right axis.
    CA_deviceRotYUp,    //!< The device rotation controls the camera rotation.
    CA_deviceRotLocYUp, //!< The device rotation controls the camera rotation and the GPS controls the Camera Translation
    CA_off              //!< No camera animation
};
//-----------------------------------------------------------------------------
//! Enumeration for different camera projections
enum SLProjection
{
    P_monoPerspective = 0,  //!< standard mono pinhole perspective projection
    P_monoIntrinsic,        //!< standard mono pinhole perspective projection from intrinsic calibration
    P_monoOrthographic,     //!< standard mono orthographic projection
    P_stereoSideBySide,     //!< side-by-side
    P_stereoSideBySideP,    //!< side-by-side proportional for mirror stereoscopes
    P_stereoSideBySideD,    //!< side-by-side distorted for Oculus Rift like glasses
    P_stereoLineByLine,     //!< line-by-line
    P_stereoColumnByColumn, //!< column-by-column
    P_stereoPixelByPixel,   //!< checkerboard pattern (DLP3D)
    P_stereoColorRC,        //!< color masking for red-cyan anaglyphs
    P_stereoColorRG,        //!< color masking for red-green anaglyphs
    P_stereoColorRB,        //!< color masking for red-blue anaglyphs
    P_stereoColorYB         //!< color masking for yellow-blue anaglyphs (ColorCode 3D)
};
//-----------------------------------------------------------------------------
//! Enumeration for stereo eye type used for camera projection
enum SLEyeType
{
    ET_left   = -1,
    ET_center = 0,
    ET_right  = 1
};
//-----------------------------------------------------------------------------
//! Enumeration for animation modes
enum SLAnimInterpolation
{
    AI_linear,
    AI_bezier
};
//-----------------------------------------------------------------------------
//! Enumeration for animation modes
enum SLAnimLooping
{
    AL_once         = 0, //!< play once
    AL_loop         = 1, //!< loop
    AL_pingPong     = 2, //!< play once in two directions
    AL_pingPongLoop = 3  //!< loop forward and backwards
};
//-----------------------------------------------------------------------------
//! Enumeration for animation easing curves
/*!
Enumerations copied from Qt class QEasingCurve.
See http://qt-project.org/doc/qt-4.8/qeasingcurve.html#Type-enum
*/
enum SLEasingCurve
{
    EC_linear     = 0,  //!< linear easing with constant velocity
    EC_inQuad     = 1,  //!< quadratic easing in, acceleration from zero velocity
    EC_outQuad    = 2,  //!< quadratic easing out, decelerating to zero velocity
    EC_inOutQuad  = 3,  //!< quadratic easing in and then out
    EC_outInQuad  = 4,  //!< quadratic easing out and then in
    EC_inCubic    = 5,  //!< cubic in easing in, acceleration from zero velocity
    EC_outCubic   = 6,  //!< cubic easing out, decelerating to zero velocity
    EC_inOutCubic = 7,  //!< cubic easing in and then out
    EC_outInCubic = 8,  //!< cubic easing out and then in
    EC_inQuart    = 9,  //!< quartic easing in, acceleration from zero velocity
    EC_outQuart   = 10, //!< quartic easing out, decelerating to zero velocity
    EC_inOutQuart = 11, //!< quartic easing in and then out
    EC_outInQuart = 12, //!< quartic easing out and then in
    EC_inQuint    = 13, //!< quintic easing in, acceleration from zero velocity
    EC_outQuint   = 14, //!< quintic easing out, decelerating to zero velocity
    EC_inOutQuint = 15, //!< quintic easing in and then out
    EC_outInQuint = 16, //!< quintic easing out and then in
    EC_inSine     = 17, //!< sine easing in, acceleration from zero velocity
    EC_outSine    = 18, //!< sine easing out, decelerating to zero velocity
    EC_inOutSine  = 19, //!< sine easing in and then out
    EC_outInSine  = 20, //!< sine easing out and then in
};
//-----------------------------------------------------------------------------
//! Describes the relative space a transformation is applied in.
enum SLTransformSpace
{
    // Do not change order!
    TS_world = 0,
    TS_parent,
    TS_object
};
//-----------------------------------------------------------------------------
//! Skinning methods
enum SLSkinMethod
{
    SM_hardware, //!< Do vertex skinning on the GPU
    SM_software  //!< Do vertex skinning on the CPU
};
//-----------------------------------------------------------------------------
//! Shader type enumeration for vertex or fragment (pixel) shader
enum SLShaderType
{
    ST_none,
    ST_vertex,
    ST_fragment,
    ST_geometry,
    ST_tesselation
};
//-----------------------------------------------------------------------------
//! Type definition for GLSL uniform1f variables that change per frame.
enum SLUniformType
{
    UT_const,   //!< constant value
    UT_incDec,  //!< never ending loop from min to max and max to min
    UT_incInc,  //!< never ending loop from min to max
    UT_inc,     //!< never ending increment
    UT_random,  //!< random values between min and max
    UT_seconds, //!< seconds since the process has started
    UT_lambda   //!< lambda getter function
};
//-----------------------------------------------------------------------------
// @todo build a dedicated log class that defines this verbosity levels
enum SLLogVerbosity
{
    LV_quiet      = 0,
    LV_minimal    = 1,
    LV_normal     = 2,
    LV_detailed   = 3,
    LV_diagnostic = 4
};
//-----------------------------------------------------------------------------
//! Mouse button codes
enum SLViewportAlign
{
    VA_center = 0,
    VA_leftOrTop,
    VA_rightOrBottom
};
//-----------------------------------------------------------------------------
//! Corresponds to the old fog modes in OpenGL 2.1
/*! See also: https://www.khronos.org/registry/OpenGL-Refpages/gl2.1/xhtml/glFog.xml
*/
enum SLFogMode
{
    FM_linear = 0,
    FM_exp,
    FM_exp2
};
//-----------------------------------------------------------------------------
//! Light Models for shader generation
enum SLLightModel
{
    LM_BlinnPhong = 0,
    LM_CookTorrance,
    LM_Custom
};
//-----------------------------------------------------------------------------
#endif
