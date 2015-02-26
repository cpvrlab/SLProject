##############################################################################
#  File:      oculus.pro
#  Purpose:   QMake project definition file for the asset import library
#  Author:    Marcus Hudritsch
#  Date:      September 2012 (HS12)
#  Copyright: Marcus Hudritsch, Switzerland
#             THIS SOFTWARE IS PROVIDED FOR EDUCATIONAL PURPOSE ONLY AND
#             WITHOUT ANY WARRANTIES WHETHER EXPRESSED OR IMPLIED.
##############################################################################

TEMPLATE = lib
TARGET = lib-ovr

CONFIG += staticlib
CONFIG -= qt
CONFIG += warn_off

CONFIG += oculus
linux:CONFIG -= oculus

REQUIRES += oculus

DEFINES += _UNICODE


#define platform variable for folder name
win32 {contains(QMAKE_TARGET.arch, x86_64) {PLATFORM = x64} else {PLATFORM = Win32}}
macx {PLATFORM = macx}
unix:!macx:!android {PLATFORM = linux}
android {PLATFORM = android}

#define configuration variable for folder name
CONFIG(debug, debug|release) {CONFIGURATION = Debug} else {CONFIGURATION = Release}

DESTDIR = ../../_lib/$$CONFIGURATION/$$PLATFORM
OBJECTS_DIR = ../../intermediate/$$TARGET/$$CONFIGURATION/$$PLATFORM


INCLUDEPATH += \
    LibOVR/Include \
    LibOVR/Src \
    3rdParty \
    3rdParty/TinyXml \
    3rdParty/glext


HEADERS += \
    LibOVR/Src/OVR_CAPI.h \
    LibOVR/Src/OVR_CAPI_Keys.h \
    LibOVR/Src/Util/Util_SystemInfo.h \
    LibOVR/Src/OVR_CAPI_D3D.h \
    LibOVR/Src/OVR_CAPI_GL.h \
    LibOVR/Src/OVR_JSON.h \
    LibOVR/Src/OVR_Profile.h \
    LibOVR/Src/OVR_SerialFormat.h \
    LibOVR/Src/OVR_Stereo.h \
    LibOVR/Src/CAPI/CAPI_LatencyStatistics.h \
    LibOVR/Src/CAPI/GL/CAPI_GL_DistortionRenderer.h \
    LibOVR/Src/CAPI/GL/CAPI_GL_DistortionShaders.h \
    LibOVR/Src/CAPI/GL/CAPI_GL_HSWDisplay.h \
    LibOVR/Src/CAPI/GL/CAPI_GL_Util.h \
    LibOVR/Src/CAPI/GL/CAPI_GLE.h \
    LibOVR/Src/CAPI/GL/CAPI_GLE_GL.h \
    LibOVR/Src/CAPI/Textures/healthAndSafety.tga.h \
    LibOVR/Src/CAPI/CAPI_DistortionRenderer.h \
    LibOVR/Src/CAPI/CAPI_FrameTimeManager.h \
    LibOVR/Src/CAPI/CAPI_HMDRenderState.h \
    LibOVR/Src/CAPI/CAPI_HMDState.h \
    LibOVR/Src/CAPI/CAPI_HSWDisplay.h \
    LibOVR/Src/Displays/OVR_Display.h \
    LibOVR/Src/Kernel/OVR_Alg.h \
    LibOVR/Src/Kernel/OVR_Allocator.h \
    LibOVR/Src/Kernel/OVR_Array.h \
    LibOVR/Src/Kernel/OVR_Atomic.h \
    LibOVR/Src/Kernel/OVR_Color.h \
    LibOVR/Src/Kernel/OVR_Compiler.h \
    LibOVR/Src/Kernel/OVR_ContainerAllocator.h \
    LibOVR/Src/Kernel/OVR_CRC32.h \
    LibOVR/Src/Kernel/OVR_DebugHelp.h \
    LibOVR/Src/Kernel/OVR_Delegates.h \
    LibOVR/Src/Kernel/OVR_Deque.h \
    LibOVR/Src/Kernel/OVR_File.h \
    LibOVR/Src/Kernel/OVR_Hash.h \
    LibOVR/Src/Kernel/OVR_KeyCodes.h \
    LibOVR/Src/Kernel/OVR_List.h \
    LibOVR/Src/Kernel/OVR_Lockless.h \
    LibOVR/Src/Kernel/OVR_Log.h \
    LibOVR/Src/Kernel/OVR_Math.h \
    LibOVR/Src/Kernel/OVR_Nullptr.h \
    LibOVR/Src/Kernel/OVR_Observer.h \
    LibOVR/Src/Kernel/OVR_RefCount.h \
    LibOVR/Src/Kernel/OVR_SharedMemory.h \
    LibOVR/Src/Kernel/OVR_Std.h \
    LibOVR/Src/Kernel/OVR_String.h \
    LibOVR/Src/Kernel/OVR_StringHash.h \
    LibOVR/Src/Kernel/OVR_SysFile.h \
    LibOVR/Src/Kernel/OVR_System.h \
    LibOVR/Src/Kernel/OVR_ThreadCommandQueue.h \
    LibOVR/Src/Kernel/OVR_Threads.h \
    LibOVR/Src/Kernel/OVR_Timer.h \
    LibOVR/Src/Kernel/OVR_Types.h \
    LibOVR/Src/Kernel/OVR_UTF8Util.h \
    LibOVR/Src/Net/OVR_BitStream.h \
    LibOVR/Src/Net/OVR_NetworkPlugin.h \
    LibOVR/Src/Net/OVR_NetworkTypes.h \
    LibOVR/Src/Net/OVR_PacketizedTCPSocket.h \
    LibOVR/Src/Net/OVR_RPC1.h \
    LibOVR/Src/Net/OVR_Session.h \
    LibOVR/Src/Net/OVR_Socket.h \
    LibOVR/Src/Sensors/OVR_DeviceConstants.h \
    LibOVR/Src/Service/Service_NetClient.h \
    LibOVR/Src/Service/Service_NetSessionCommon.h \
    LibOVR/Src/Tracking/Tracking_PoseState.h \
    LibOVR/Src/Tracking/Tracking_SensorState.h \
    LibOVR/Src/Tracking/Tracking_SensorStateReader.h \
    LibOVR/Src/Util/Util_ImageWindow.h \
    LibOVR/Src/Util/Util_Interface.h \
    LibOVR/Src/Util/Util_LatencyTest2Reader.h \
    LibOVR/Src/Util/Util_LatencyTest2State.h \
    LibOVR/Src/Util/Util_Render_Stereo.h \
    LibOVR/Src/Util/Util_SystemGUI.h

SOURCES += \
    LibOVR/Src/OVR_CAPI.cpp \
    LibOVR/Src/OVR_JSON.cpp \
    LibOVR/Src/OVR_Profile.cpp \
    LibOVR/Src/OVR_SerialFormat.cpp \
    LibOVR/Src/OVR_Stereo.cpp \
    LibOVR/Src/CAPI/GL/CAPI_GLE.cpp \
    LibOVR/Src/CAPI/GL/CAPI_GL_DistortionRenderer.cpp \
    LibOVR/Src/CAPI/GL/CAPI_GL_HSWDisplay.cpp \
    LibOVR/Src/CAPI/GL/CAPI_GL_Util.cpp \
    LibOVR/Src/CAPI/GL/CAPI_GLE.cpp \
    LibOVR/Src/CAPI/CAPI_DistortionRenderer.cpp \
    LibOVR/Src/CAPI/CAPI_FrameTimeManager.cpp \
    LibOVR/Src/CAPI/CAPI_HMDRenderState.cpp \
    LibOVR/Src/CAPI/CAPI_HMDState.cpp \
    LibOVR/Src/CAPI/CAPI_HSWDisplay.cpp \
    LibOVR/Src/CAPI/CAPI_LatencyStatistics.cpp \
    LibOVR/Src/Kernel/OVR_Alg.cpp \
    LibOVR/Src/Kernel/OVR_Allocator.cpp \
    LibOVR/Src/Kernel/OVR_Atomic.cpp \
    LibOVR/Src/Kernel/OVR_CRC32.cpp \
    LibOVR/Src/Kernel/OVR_File.cpp \
    LibOVR/Src/Kernel/OVR_FileFILE.cpp \
    LibOVR/Src/Kernel/OVR_Log.cpp \
    LibOVR/Src/Kernel/OVR_Math.cpp \
    LibOVR/Src/Kernel/OVR_RefCount.cpp \
    LibOVR/Src/Kernel/OVR_SharedMemory.cpp \
    LibOVR/Src/Kernel/OVR_Std.cpp \
    LibOVR/Src/Kernel/OVR_String.cpp \
    LibOVR/Src/Kernel/OVR_String_FormatUtil.cpp \
    LibOVR/Src/Kernel/OVR_String_PathUtil.cpp \
    LibOVR/Src/Kernel/OVR_SysFile.cpp \
    LibOVR/Src/Kernel/OVR_System.cpp \
    LibOVR/Src/Kernel/OVR_ThreadCommandQueue.cpp \
    LibOVR/Src/Kernel/OVR_Timer.cpp \
    LibOVR/Src/Kernel/OVR_UTF8Util.cpp \
    LibOVR/Src/Kernel/OVR_DebugHelp.cpp \
    LibOVR/Src/Net/OVR_BitStream.cpp \
    LibOVR/Src/Net/OVR_NetworkPlugin.cpp \
    LibOVR/Src/Net/OVR_PacketizedTCPSocket.cpp \
    LibOVR/Src/Net/OVR_RPC1.cpp \
    LibOVR/Src/Net/OVR_Session.cpp \
    LibOVR/Src/Net/OVR_Socket.cpp \
    LibOVR/Src/Service/Service_NetClient.cpp \
    LibOVR/Src/Service/Service_NetSessionCommon.cpp \
    LibOVR/Src/Tracking/Tracking_SensorStateReader.cpp \
    LibOVR/Src/Util/Util_ImageWindow.cpp \
    LibOVR/Src/Util/Util_LatencyTest2Reader.cpp \
    LibOVR/Src/Util/Util_Render_Stereo.cpp \
    LibOVR/Src/Util/Util_SystemInfo.cpp \
    LibOVR/Src/Util/Util_SystemGUI.cpp

macx {
HEADERS += \
    LibOVR/Src/Displays/OVR_OSX_Display.h \
    LibOVR/Src/Net/OVR_Unix_Socket.h \
    LibOVR/Src/Kernel/OVR_mach_exc_OSX.h \

SOURCES += \
    LibOVR/Src/Displays/OVR_OSX_Display.cpp \
    LibOVR/Src/Net/OVR_Unix_Socket.cpp \
    LibOVR/Src/Kernel/OVR_ThreadsPthread.cpp \
    LibOVR/Src/Kernel/OVR_mach_exc_OSX.c \

OBJECTIVE_SOURCES += \
    LibOVR/Src/Util/Util_SystemInfo_OSX.mm \
    LibOVR/Src/Util/Util_SystemGUI_OSX.mm \
}

win32 {
HEADERS += \
    LibOVR/Src/CAPI/D3D1X/CAPI_D3D10_DistortionRenderer.h \
    LibOVR/Src/CAPI/D3D1X/CAPI_D3D10_HSWDisplay.h \
    LibOVR/Src/CAPI/D3D1X/CAPI_D3D11_DistortionRenderer.h \
    LibOVR/Src/CAPI/D3D1X/CAPI_D3D11_HSWDisplay.h \
    LibOVR/Src/CAPI/D3D1X/CAPI_D3D1X_DistortionRenderer.h \
    LibOVR/Src/CAPI/D3D1X/CAPI_D3D1X_HSWDisplay.h \
    LibOVR/Src/CAPI/D3D1X/CAPI_D3D1X_Util.h \
    LibOVR/Src/CAPI/D3D9/CAPI_D3D9_DistortionRenderer.h \
    LibOVR/Src/CAPI/D3D9/CAPI_D3D9_HSWDisplay.h \
    LibOVR/Src/CAPI/Shaders/Distortion_ps.h \
    LibOVR/Src/CAPI/Shaders/Distortion_ps_refl.h \
    LibOVR/Src/CAPI/Shaders/Distortion_vs.h \
    LibOVR/Src/CAPI/Shaders/Distortion_vs_refl.h \
    LibOVR/Src/CAPI/Shaders/DistortionChroma_ps.h \
    LibOVR/Src/CAPI/Shaders/DistortionChroma_ps_refl.h \
    LibOVR/Src/CAPI/Shaders/DistortionChroma_vs.h \
    LibOVR/Src/CAPI/Shaders/DistortionChroma_vs_refl.h \
    LibOVR/Src/CAPI/Shaders/DistortionTimewarp_vs.h \
    LibOVR/Src/CAPI/Shaders/DistortionTimewarp_vs_refl.h \
    LibOVR/Src/CAPI/Shaders/DistortionTimewarpChroma_vs.h \
    LibOVR/Src/CAPI/Shaders/DistortionTimewarpChroma_vs_refl.h \
    LibOVR/Src/CAPI/Shaders/SimpleQuad_ps.h \
    LibOVR/Src/CAPI/Shaders/SimpleQuad_ps_refl.h \
    LibOVR/Src/CAPI/Shaders/SimpleQuad_vs.h \
    LibOVR/Src/CAPI/Shaders/SimpleQuad_vs_refl.h \
    LibOVR/Src/CAPI/Shaders/SimpleTexturedQuad_ps.h \
    LibOVR/Src/CAPI/Shaders/SimpleTexturedQuad_ps_refl.h \
    LibOVR/Src/CAPI/Shaders/SimpleTexturedQuad_vs.h \
    LibOVR/Src/CAPI/Shaders/SimpleTexturedQuad_vs_refl.h \
    LibOVR/Src/Displays/OVR_Win32_Display.h \
    LibOVR/Src/Displays/OVR_Win32_Dxgi_Display.h \
    LibOVR/Src/Displays/OVR_Win32_FocusReader.h \
    LibOVR/Src/Displays/OVR_Win32_ShimFunctions.h \
    LibOVR/Src/Net/OVR_Win32_Socket.h

SOURCES += \
    LibOVR/Src/CAPI/D3D1X/CAPI_D3D10_DistortionRenderer.cpp \
    LibOVR/Src/CAPI/D3D1X/CAPI_D3D10_HSWDisplay.cpp \
    LibOVR/Src/CAPI/D3D1X/CAPI_D3D11_DistortionRenderer.cpp \
    LibOVR/Src/CAPI/D3D1X/CAPI_D3D11_HSWDisplay.cpp \
    LibOVR/Src/CAPI/D3D1X/CAPI_D3D1X_DistortionRenderer.cpp \
    LibOVR/Src/CAPI/D3D1X/CAPI_D3D1X_HSWDisplay.cpp \
    LibOVR/Src/CAPI/D3D1X/CAPI_D3D1X_Util.cpp \
    LibOVR/Src/CAPI/D3D9/CAPI_D3D9_DistortionRenderer.cpp \
    LibOVR/Src/CAPI/D3D9/CAPI_D3D9_HSWDisplay.cpp \
    LibOVR/Src/CAPI/D3D9/CAPI_D3D9_Util.cpp \
    LibOVR/Src/Displays/OVR_Win32_Display.cpp \
    LibOVR/Src/Displays/OVR_Win32_FocusReader.cpp \
    LibOVR/Src/Displays/OVR_Win32_RenderShim.cpp \
    LibOVR/Src/Displays/OVR_Win32_ShimFunctions.cpp \
    LibOVR/Src/Net/OVR_Win32_Socket.cpp \
    LibOVR/Src/Kernel/OVR_ThreadsWinAPI.cpp \
    LibOVR/Src/Displays/OVR_Display.cpp \
    LibOVR/Src/Kernel/OVR_Lockless.cpp \
    LibOVR/Src/Util/Util_Interface.cpp


    #the last 3 source files are needed in both mac and windows
    #but the mac version fails if the source files dont define any symbols themselfes
    #files: (ovr_display.cpp, ovr_lockless.cpp, util_interface.cpp)
}
