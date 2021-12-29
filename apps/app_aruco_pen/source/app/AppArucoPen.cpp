//#############################################################################
//  File:      AppArucoPen.cpp
//  Date:      October 2021
//  Codestyle: https://github.com/cpvrlab/SLProject/wiki/SLProject-Coding-Style
//  Authors:   Marino von Wattenwyl
//  License:   This software is provided under the GNU General Public License
//             Please visit: http://opensource.org/licenses/GPL-3.0
//#############################################################################

#include "AppArucoPen.h"

#include <SL.h>
#include <GlobalTimer.h>
#include <CVCapture.h>
#include <Utils.h>
#include <Instrumentor.h>
#include <AppDemo.h>
#include <app/AppArucoPenROSNode.h>
#include <IDSPeakInterface.h>
#include <CVCaptureProviderSpryTrack.h>

extern void trackVideo(CVCaptureProvider* provider);

//-----------------------------------------------------------------------------
void AppArucoPen::openCaptureProviders()
{
    //    SL_LOG("Loading capture providers...");
    //
    //    IMFAttributes* attributes = nullptr;
    //    HRESULT        hr         = MFCreateAttributes(&attributes, 1);
    //    if (FAILED(hr)) SL_LOG("Failed to create MF attributes");
    //
    //    hr = attributes->SetGUID(MF_DEVSOURCE_ATTRIBUTE_SOURCE_TYPE,
    //                             MF_DEVSOURCE_ATTRIBUTE_SOURCE_TYPE_VIDCAP_GUID);
    //    if (FAILED(hr)) SL_LOG("Failed to set source type");
    //
    //    IMFActivate** devices = nullptr;
    //    UINT32        count;
    //    hr = MFEnumDeviceSources(attributes, &devices, &count);
    //    if (FAILED(hr)) SL_LOG("Failed to enumerate devices");
    //
    //    SL_LOG("Device count: %d", count);
    //    for (UINT32 i = 0; i < count; i++) {
    //        IMFActivate* device = devices[i];
    //
    //        WCHAR name[128];
    //        UINT32 len;
    //        device->GetString(MF_DEVSOURCE_ATTRIBUTE_FRIENDLY_NAME,
    //                          name,
    //                          128,
    //                          &len);
    //
    //        std::string cppString = CW2A(name);
    //        SL_LOG("%s", cppString.c_str());
    //    }

    // Logitech + IDS Peak
    // openCaptureProvider(new CVCaptureProviderStandard(0, CVSize(640, 480)));
    // openCaptureProvider(new CVCaptureProviderIDSPeak(0, CVSize(1920, 1280)));

    // Logitech
    // 1280x960 actually provides better results than 1920x1080
    // Are you kidding
    // openCaptureProvider(new CVCaptureProviderStandard(0, CVSize(1280, 960)));

    // Logitech + Intel
    //    openCaptureProvider(new CVCaptureProviderStandard(2, CVSize(1280, 960)));
    //    openCaptureProvider(new CVCaptureProviderStandard(1, CVSize(1280, 960)));

    // Intel + IDS Peak
    //    openCaptureProvider(new CVCaptureProviderStandard(1, CVSize(1280, 960)));

    // Every single IDS camera that is connected
    // for (int i = 0; i < IDSPeakInterface::instance().numAvailableDevices(); i++)
    // {
    //     openCaptureProvider(new CVCaptureProviderIDSPeak(i, CVSize(2768, 1840)));
    // }

    openCaptureProvider(new CVCaptureProviderSpryTrack(CVSize(1280, 960)));

    _currentCaptureProvider = _captureProviders.empty() ? nullptr : _captureProviders[0];

    SL_LOG("Opening done");
}
//-----------------------------------------------------------------------------
void AppArucoPen::openCaptureProvider(CVCaptureProvider* captureProvider)
{
    float before = GlobalTimer::timeS();
    captureProvider->open();

    if (!captureProvider->isOpened())
    {
        SL_LOG("Failed to open capture provider \"%s\"", captureProvider->name().c_str());
        return;
    }

    _captureProviders.push_back(captureProvider);
    float delta = GlobalTimer::timeS() - before;

    SL_LOG("%s capture provider opened in %f s",
           captureProvider->name().c_str(),
           delta);

    SLstring configPath    = AppDemo::calibFilePath;
    SLstring calibFilename = "camCalib_" + captureProvider->uid() + ".xml";
    SLstring calibPath     = Utils::unifySlashes(configPath) + calibFilename;

    if (Utils::fileExists(calibPath))
    {
        SL_LOG("Found calibration for capture at %s", calibPath.c_str());
        if (captureProvider->camera().calibration.load(configPath,
                                                       calibFilename,
                                                       true))
        {
            SL_LOG("Calibration successfully loaded");
        }
        else
        {
            SL_LOG("Failed to load calibration");
        }
    }
}
//-----------------------------------------------------------------------------
void AppArucoPen::closeCaptureProviders()
{
    SL_LOG("Closing capture providers...");

    for (CVCaptureProvider* captureProvider : _captureProviders)
    {
        captureProvider->close();
        delete captureProvider;
    }

    SL_LOG("All capture providers closed");
}
//-----------------------------------------------------------------------------
void AppArucoPen::grabFrameImagesAndTrack(SLSceneView* sv)
{
    CVCapture::instance()->camSizes.clear();

    // Grab and track all non-displayed capture providers if we are in the ArUco cube scene
    if (AppArucoPen::doMultiTracking() && (AppDemo::sceneID == SID_VideoTrackArucoCubeMain || AppDemo::sceneID == SID_VirtualArucoPen))
    {
        for (CVCaptureProvider* provider : _captureProviders)
        {
            if (provider != _currentCaptureProvider)
            {
                grabFrameImageAndTrack(provider, sv);
            }
        }
    }

    // Grab and track the currently displayed capture provider
    grabFrameImageAndTrack(_currentCaptureProvider, sv);

    if (AppDemo::sceneID == SID_VideoTrackArucoCubeMain || AppDemo::sceneID == SID_VirtualArucoPen)
    {
        AppArucoPen::instance().arucoPen().multiTracker().combine();
        if (AppArucoPen::instance().arucoPen().state() == SLArucoPen::Tracing)
        {
            AppArucoPen::instance().publishTipPose();
        }
    }
}
//-----------------------------------------------------------------------------
void AppArucoPen::grabFrameImageAndTrack(CVCaptureProvider* provider, SLSceneView* sv)
{
    PROFILE_FUNCTION();

    grabFrameImage(provider, sv);
    trackVideo(provider);
}
//-----------------------------------------------------------------------------
void AppArucoPen::grabFrameImage(CVCaptureProvider* provider, SLSceneView* sv)
{
    PROFILE_FUNCTION();

    CVCapture::instance()->startCaptureTimeMS = GlobalTimer::timeMS();

    provider->grab();
    provider->cropToAspectRatio(sv->viewportWdivH());

    CVCapture::instance()->camSizes.push_back(provider->captureSize());

    CVCapture::instance()->lastFrame     = provider->lastFrameBGR();
    CVCapture::instance()->lastFrameGray = provider->lastFrameGray();
    CVCapture::instance()->captureSize   = provider->captureSize();
    CVCapture::instance()->format        = PF_bgr;
}
//-----------------------------------------------------------------------------
CVTracked* AppArucoPen::currentTracker()
{
    if (_trackers.empty())
    {
        return nullptr;
    }

    CVTracked* tracked = _trackers.at(_currentCaptureProvider);
    return tracked;
}
//-----------------------------------------------------------------------------
void AppArucoPen::publishTipPose()
{
    SLArucoPen pen = AppArucoPen::instance().arucoPen();
    AppArucoPenROSNode::instance().publishPose(pen.rosPosition(), pen.rosOrientation());
}
//-----------------------------------------------------------------------------