#include <AppArucoPen.h>
#include <SL.h>
#include <GlobalTimer.h>

void AppArucoPen::openCaptureProviders()
{
    float before;
    float delta;

    SL_LOG("Loading capture providers...");

    before                 = GlobalTimer::timeS();
    auto* standardProvider = new CVStandardCaptureProvider();
    standardProvider->open();
    captureProviders.push_back(standardProvider);
    delta = GlobalTimer::timeS() - before;

    SL_LOG("CV capture provider opened in %f s", delta);

    before                = GlobalTimer::timeS();
    auto* idsPeakProvider = new IDSPeakCaptureProvider();
    idsPeakProvider->open();
    captureProviders.push_back(idsPeakProvider);
    delta = GlobalTimer::timeS() - before;

    SL_LOG("IDS Peak capture provider opened in %f s", delta);

    currentCaptureProvider = standardProvider;
}