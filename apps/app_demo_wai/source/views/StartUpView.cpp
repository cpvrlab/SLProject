#include <views/StartUpView.h>
#include <SLGLTexture.h>
#include <SLGLProgram.h>
#include <SLLightSpot.h>
#include <SLTexFont.h>
#include <SLSphere.h>
#include <SLText.h>
#include <SelectionGui.h>
#include <SLGLProgramManager.h>

StartUpView::StartUpView(SLInputManager&   inputManager,
                         const DeviceData& deviceData)
  : _s("StartUpScene", nullptr),
    _sv(&_s, deviceData.dpi(), inputManager),
    _pixPerMM((float)deviceData.dpi() / 25.4f)
{
    _sv.init("StartUpSceneView", deviceData.scrWidth(), deviceData.scrHeight(), nullptr, nullptr, nullptr, deviceData.writableDir());
    _s.init();

    SLMaterial* m1 = new SLMaterial(&_assets, "m1", SLCol4f::BLUE);

    SLCamera* sceneCamera = new SLCamera("Camera 1");
    sceneCamera->clipNear(0.1f);
    sceneCamera->clipFar(100);
    sceneCamera->translation(0, 0, 5);
    sceneCamera->lookAt(0, 0, 0);
    sceneCamera->focalDist(5);
    sceneCamera->background().colors(BFHColors::OrangeLogo,
                                     BFHColors::OrangeLogo);
    sceneCamera->setInitialState();

    // Now create 2D text but don't scale it (all sizes in pixels)
    //std::string txt       = "ErlebAR";
    //SLTexFont*  _textFont = new SLTexFont("Font24.png", SLGLProgramManager::get(SP_fontTex));
    //SLVec2f     size      = _textFont->calcTextSize(txt);
    //calc scale for target height mm
    //float   targetHeightMM  = 10.f;
    //float   targetHeightPix = targetHeightMM * _pixPerMM;
    //float   scale           = targetHeightPix / size.y;
    //SLNode* text2D          = new SLText(txt, _textFont);
    //text2D->translate(-size.x * 0.5f * scale, -size.y * 0.5f * scale, 0);
    //text2D->scale(scale);
    // Assemble 3D scene as usual with camera and light
    SLNode* scene3D = new SLNode("root3D");
    scene3D->addChild(sceneCamera);
    //scene3D->addChild(t22);
    _s.root3D(scene3D);

    // Assemble 2D scene
    SLNode* scene2D = new SLNode("root2D");
    //scene2D->addChild(text2D);
    _s.root2D(scene2D);

    _sv.camera(sceneCamera);
    _sv.doWaitOnIdle(false);

    _sv.onInitialize();
}

StartUpView::~StartUpView()
{
    if (_textFont)
    {
        delete _textFont;
        _textFont = nullptr;
    }
}

bool StartUpView::update()
{
    if (_firstUpdate)
    {
        _firstUpdate = false;
        _timer.start();
    }

    //if (_timer.elapsedTimeInMilliSec() > 1000.f)
    //    setStateReady();

    //_s.onUpdate();
    return _sv.onPaint();
}
