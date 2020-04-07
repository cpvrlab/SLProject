#include <views/WelcomeView.h>
#include <SLGLTexture.h>
#include <SLGLProgram.h>
#include <SLLightSpot.h>
#include <SL/SLTexFont.h>
#include <SLSphere.h>
#include <SLText.h>
#include <SelectionGui.h>
#include <SLGLProgramManager.h>

WelcomeView::WelcomeView(SLInputManager& inputManager,
                         int             screenWidth,
                         int             screenHeight,
                         int             dotsPerInch,
                         std::string     fontPath,
                         std::string     texturePath,
                         std::string     imguiIniPath,
                         std::string     version)
  : SLSceneView(nullptr, dotsPerInch, inputManager),
    _gui(dotsPerInch, screenWidth, screenHeight, fontPath, texturePath, version),
    _pixPerMM((float)dotsPerInch / 25.4f)
{
    init("WelcomeView", screenWidth, screenHeight, nullptr, nullptr, &_gui, imguiIniPath);
    onInitialize();
}

WelcomeView::~WelcomeView()
{
    if (_textFont)
    {
        delete _textFont;
        _textFont = nullptr;
    }
}

bool WelcomeView::update()
{
    return onPaint();
}
