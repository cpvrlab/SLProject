#include <states/StartUpState.h>
#include <SLGLTexture.h>
#include <SLGLProgram.h>
#include <SLLightSpot.h>
#include <SL/SLTexFont.h>
#include <SLSphere.h>
#include <SLText.h>
#include <SelectionGui.h>

StartUpState::StartUpState(SLInputManager& inputManager,
                           int             screenWidth,
                           int             screenHeight,
                           int             dotsPerInch,
                           std::string     imguiIniPath)
  : _s("StartUpScene", nullptr, inputManager),
    _sv(&_s, dotsPerInch),
    _pixPerMM((float)dotsPerInch / 25.4f)
{
    _sv.init("StartUpSceneView", screenWidth, screenHeight, nullptr, nullptr, nullptr, imguiIniPath);
    _s.init();

    SLMaterial* m1 = new SLMaterial(&_assets, "m1", SLCol4f::BLUE);

    SLCamera* sceneCamera = new SLCamera("Camera 1");
    sceneCamera->clipNear(0.1f);
    sceneCamera->clipFar(100);
    sceneCamera->translation(0, 0, 5);
    sceneCamera->lookAt(0, 0, 0);
    sceneCamera->focalDist(5);
    sceneCamera->background().colors(BFHColors::OrangePrimary,
                                     BFHColors::OrangePrimary);
    sceneCamera->setInitialState();

    // Because all text objects get their sizes in pixels we have to scale them down
    //SLfloat  scale = 0.01f;
    //SLstring txt   = "";
    //SLVec2f  size  = SLTexFont::font22->calcTextSize(txt);
    //SLNode*  t22   = new SLText(txt, SLTexFont::font22);
    //t22->translate(-size.x * 0.5f * scale, -1.2f, 0);
    //t22->scale(scale);

    // Now create 2D text but don't scale it (all sizes in pixels)
    std::string txt       = "ErlebAR";
    SLTexFont*  _textFont = new SLTexFont("Font24.png", SLGLProgramManager::get(SP_fontTex));
    SLVec2f     size      = _textFont->calcTextSize(txt);
    //calc scale for target height mm
    float   targetHeightMM  = 10.f;
    float   targetHeightPix = targetHeightMM * _pixPerMM;
    float   scale           = targetHeightPix / size.y;
    SLNode* text2D          = new SLText(txt, _textFont);
    text2D->translate(-size.x * 0.5f * scale, -size.y * 0.5f * scale, 0);
    text2D->scale(scale);
    // Assemble 3D scene as usual with camera and light
    SLNode* scene3D = new SLNode("root3D");
    scene3D->addChild(sceneCamera);
    //scene3D->addChild(t22);
    _s.root3D(scene3D);

    // Assemble 2D scene
    SLNode* scene2D = new SLNode("root2D");
    scene2D->addChild(text2D);
    _s.root2D(scene2D);

    _sv.camera(sceneCamera);
    _sv.doWaitOnIdle(false);

    _sv.onInitialize();
}

StartUpState::~StartUpState()
{
    if (_textFont)
    {
        delete _textFont;
        _textFont = nullptr;
    }
}

bool StartUpState::update()
{
    if (_firstUpdate)
    {
        _firstUpdate = false;
        _timer.start();
    }

    if (_timer.elapsedTimeInMilliSec() > 5000.f)
        setStateReady();

    _s.onUpdate();
    return _sv.onPaint();
}

void StartUpState::doStart()
{
    _started = true;
}
