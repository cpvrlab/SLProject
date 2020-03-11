#include <states/SelectionState.h>
#include <SLSceneView.h>
#include <SLInputManager.h>

#include <SLGLTexture.h>
#include <SLGLProgram.h>
#include <SLLightSpot.h>
#include <SL/SLTexFont.h>
#include <SLSphere.h>
#include <SLText.h>

SelectionState::SelectionState(SLInputManager& inputManager,
                               int             screenWidth,
                               int             screenHeight,
                               int             dotsPerInch,
                               std::string     fontPath,
                               std::string     imguiIniPath)
  : _gui(dotsPerInch, fontPath),
    _s("SelectionScene", nullptr, inputManager),
    _sv(&_s, dotsPerInch)
{
    _sv.init("SelectionSceneView", screenWidth, screenHeight, nullptr, nullptr, &_gui, imguiIniPath);
    _s.init();

    SLMaterial* m1 = new SLMaterial(&_assets, "m1", SLCol4f::RED);

    SLCamera* sceneCamera = new SLCamera("Camera 1");
    sceneCamera->clipNear(0.1f);
    sceneCamera->clipFar(100);
    sceneCamera->translation(0, 0, 5);
    sceneCamera->lookAt(0, 0, 0);
    sceneCamera->focalDist(5);
    sceneCamera->background().colors(SLCol4f(0.7f, 0.7f, 0.7f),
                                     SLCol4f(0.2f, 0.2f, 0.2f));
    sceneCamera->setInitialState();

    SLLightSpot* light1 = new SLLightSpot(&_assets, &_s, 10, 10, 10, 0.3f);
    light1->ambient(SLCol4f(0.2f, 0.2f, 0.2f));
    light1->diffuse(SLCol4f(0.8f, 0.8f, 0.8f));
    light1->specular(SLCol4f(1, 1, 1));
    light1->attenuation(1, 0, 0);

    // Because all text objects get their sizes in pixels we have to scale them down
    SLfloat  scale = 0.01f;
    SLstring txt   = "This is text in 3D with font07";
    SLVec2f  size  = SLTexFont::font07->calcTextSize(txt);
    SLNode*  t07   = new SLText(txt, SLTexFont::font07);
    t07->translate(-size.x * 0.5f * scale, 1.0f, 0);
    t07->scale(scale);

    txt         = "This is text in 3D with font22";
    size        = SLTexFont::font22->calcTextSize(txt);
    SLNode* t22 = new SLText(txt, SLTexFont::font22);
    t22->translate(-size.x * 0.5f * scale, -1.2f, 0);
    t22->scale(scale);

    // Assemble 3D scene as usual with camera and light
    SLNode* scene3D = new SLNode("root3D");
    scene3D->addChild(sceneCamera);
    scene3D->addChild(light1);
    scene3D->addChild(new SLNode(new SLSphere(&_assets, 0.5f, 32, 32, "Sphere", m1)));
    scene3D->addChild(t07);
    scene3D->addChild(t22);

    _sv.camera(sceneCamera);
    _sv.doWaitOnIdle(false);

    _s.root3D(scene3D);

    _sv.onInitialize();
}

bool SelectionState::update()
{
    _s.onUpdate();
    return _sv.onPaint();
}

void SelectionState::doStart()
{
    _started = true;
}
