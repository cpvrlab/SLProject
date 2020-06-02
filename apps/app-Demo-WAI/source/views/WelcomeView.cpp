#include <views/WelcomeView.h>

WelcomeView::WelcomeView(SLInputManager&     inputManager,
                         ErlebAR::Resources& resources,
                         const ImGuiEngine&  imGuiEngine,
                         const DeviceData&   deviceData,
                         std::string         version)
  : SLSceneView(nullptr, deviceData.dpi(), inputManager),
    _gui(imGuiEngine,
         resources,
         deviceData.dpi(),
         deviceData.scrWidth(),
         deviceData.scrHeight(),
         deviceData.fontDir(),
         deviceData.textureDir(),
         version)
{
    init("WelcomeView",
         deviceData.scrWidth(),
         deviceData.scrHeight(),
         nullptr,
         nullptr,
         &_gui,
         deviceData.writableDir());
    onInitialize();
}

bool WelcomeView::update()
{
    return onPaint();
}
