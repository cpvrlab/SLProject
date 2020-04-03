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
                         std::string     imguiIniPath,
                         std::string     version)
  : _gui(dotsPerInch, screenWidth, screenHeight, fontPath, version),
    _sv(nullptr, dotsPerInch, inputManager),
    _pixPerMM((float)dotsPerInch / 25.4f)
{
    _sv.init("WelcomeView", screenWidth, screenHeight, nullptr, nullptr, &_gui, imguiIniPath);

    _sv.doWaitOnIdle(false);
    _sv.onInitialize();
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
    return _sv.onPaint();
}
