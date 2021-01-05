#include "AppWAIScene.h"
#include <SLBox.h>
#include <SLLightDirect.h>
#include <SLLightSpot.h>
#include <SLCoordAxis.h>
#include <SLPoints.h>
#include <SLAssimpImporter.h>
#include <SLVec4.h>
#include <SLKeyframeCamera.h>
#include <SLGLProgramManager.h>
#include <SLHorizonNode.h>
#include <HttpUtils.h>
#include <ZipUtils.h>
#define PASSWORD "http_password"

AppWAIScene::AppWAIScene(SLstring name, std::string dataDir, std::string erlebARDir)
  : SLScene(name, nullptr),
    _dataDir(Utils::unifySlashes(dataDir)),
    _erlebARDir(Utils::unifySlashes(erlebARDir))
{
}

AppWAIScene::~AppWAIScene()
{
    if (_font16)
        delete _font16;
}

void AppWAIScene::unInit()
{
    SLScene::unInit();
    assets.clear();

    mapNode  = nullptr;
    camera   = nullptr;
    sunLight = nullptr;

    mapPC             = nullptr;
    mapMatchedPC      = nullptr;
    mapLocalPC        = nullptr;
    mapMarkerCornerPC = nullptr;
    keyFrameNode      = nullptr;
    covisibilityGraph = nullptr;
    spanningTree      = nullptr;
    loopEdges         = nullptr;

    redMat               = nullptr;
    greenMat             = nullptr;
    blueMat              = nullptr;
    covisibilityGraphMat = nullptr;
    spanningTreeMat      = nullptr;
    loopEdgesMat         = nullptr;

    mappointsMesh             = nullptr;
    mappointsMatchedMesh      = nullptr;
    mappointsLocalMesh        = nullptr;
    mappointsMarkerCornerMesh = nullptr;
    covisibilityGraphMesh     = nullptr;
    spanningTreeMesh          = nullptr;
    loopEdgesMesh             = nullptr;
}

void AppWAIScene::initScene(ErlebAR::LocationId locationId, ErlebAR::AreaId areaId, SLDeviceRotation* devRot, SLDeviceLocation* devLoc, int svW, int svH)
{
    unInit();

    _root3D = new SLNode("scene");

    //init map visualizaton (common to all areas)
    initMapVisualization();
    //init area dependent visualization
    initAreaVisualization(locationId, areaId, devRot, devLoc, svW, svH);
}

void AppWAIScene::initMapVisualization()
{
    mapNode           = new SLNode("map");
    mapPC             = new SLNode("MapPC");
    mapMatchedPC      = new SLNode("MapMatchedPC");
    mapLocalPC        = new SLNode("MapLocalPC");
    mapMarkerCornerPC = new SLNode("MapMarkerCornerPC");
    keyFrameNode      = new SLNode("KeyFrames");
    covisibilityGraph = new SLNode("CovisibilityGraph");
    spanningTree      = new SLNode("SpanningTree");
    loopEdges         = new SLNode("LoopEdges");

    redMat = new SLMaterial(&assets, SLGLProgramManager::get(SP_colorUniform), SLCol4f::RED, "Red");
    //todo ghm1: the shader program was assigned already!!??
    redMat->program(new SLGLProgramGeneric(&assets, _dataDir + "shaders/ColorUniformPoint.vert", _dataDir + "shaders/Color.frag"));
    redMat->program()->addUniform1f(new SLGLUniform1f(UT_const, "u_pointSize", 3.0f));
    greenMat = new SLMaterial(&assets, SLGLProgramManager::get(SP_colorUniform), BFHColors::GreenLight, "Green");
    greenMat->program(new SLGLProgramGeneric(&assets, _dataDir + "shaders/ColorUniformPoint.vert", _dataDir + "shaders/Color.frag"));
    greenMat->program()->addUniform1f(new SLGLUniform1f(UT_const, "u_pointSize", 5.0f));
    blueMat = new SLMaterial(&assets, SLGLProgramManager::get(SP_colorUniform), BFHColors::BlueImgui1, "Blue");
    blueMat->program(new SLGLProgramGeneric(&assets, _dataDir + "shaders/ColorUniformPoint.vert", _dataDir + "shaders/Color.frag"));
    blueMat->program()->addUniform1f(new SLGLUniform1f(UT_const, "u_pointSize", 4.0f));

    covisibilityGraphMat = new SLMaterial(&assets, "covisibilityGraphMat", SLCol4f::YELLOW);
    covisibilityGraphMat->program(new SLGLProgramGeneric(&assets, _dataDir + "shaders/ColorUniform.vert", _dataDir + "shaders/Color.frag"));
    spanningTreeMat = new SLMaterial(&assets, "spanningTreeMat", SLCol4f::GREEN);
    spanningTreeMat->program(new SLGLProgramGeneric(&assets, _dataDir + "shaders/ColorUniform.vert", _dataDir + "shaders/Color.frag"));
    loopEdgesMat = new SLMaterial(&assets, "loopEdgesMat", SLCol4f::RED);
    loopEdgesMat->program(new SLGLProgramGeneric(&assets, _dataDir + "shaders/ColorUniform.vert", _dataDir + "shaders/Color.frag"));

    mapNode->addChild(mapPC);
    mapNode->addChild(mapMatchedPC);
    mapNode->addChild(mapLocalPC);
    mapNode->addChild(mapMarkerCornerPC);
    mapNode->addChild(keyFrameNode);
    mapNode->addChild(covisibilityGraph);
    mapNode->addChild(spanningTree);
    mapNode->addChild(loopEdges);

    //todo: remove?
    mapNode->rotate(180, 1, 0, 0);

    _root3D->addChild(mapNode);
}

void AppWAIScene::initAreaVisualization(ErlebAR::LocationId locationId, ErlebAR::AreaId areaId, SLDeviceRotation* devRot, SLDeviceLocation* devLoc, int svW, int svH)
{
    //search and delete old node
    if (!_root3D)
        return;

    if (locationId == ErlebAR::LocationId::AUGST)
        initLocationAugst();
    else if (locationId == ErlebAR::LocationId::AVENCHES)
        initLocationAvenches(areaId);
    else if (locationId == ErlebAR::LocationId::BERN)
        initLocationBern();
    else if (locationId == ErlebAR::LocationId::BIEL)
        initLocationBiel(devRot, devLoc);
    else if (locationId == ErlebAR::LocationId::EVILARD)
    {
        if (areaId == ErlebAR::AreaId::EVILARD_OFFICE)
            initAreaEvilardOffice(devRot, devLoc, svW, svH);
        else
            initLocationDefault();
    }
    else
        initLocationDefault();
}

void AppWAIScene::initLocationAugst()
{
    if (!Utils::dirExists(_dataDir + "erleb-AR/models/augst/"))
    {
        HttpUtils::download("https://pallas.ti.bfh.ch/erlebar/models/augst.zip", _dataDir + "erleb-AR/models/", "erlebar", PASSWORD);
        ZipUtils::unzip(_dataDir + "erleb-AR/models/augst.zip", _dataDir + "erleb-AR/models/");
    }
    // Create directional light for the sun light
    sunLight = new SLLightDirect(&assets, this, 5.0f);
    sunLight->powers(1.0f, 1.0f, 1.0f);
    sunLight->attenuation(1, 0, 0);
    sunLight->translation(0, 10, 0);
    sunLight->lookAt(10, 0, 10);
    sunLight->doSunPowerAdaptation(true);
    sunLight->createsShadows(true);
    sunLight->createShadowMap(-100, 250, SLVec2f(250, 150), SLVec2i(2048, 2048));
    sunLight->doSmoothShadows(true);
    sunLight->castsShadows(false);
    _root3D->addChild(sunLight);

    //init camera
    camera = new VideoBackgroundCamera("AppWAIScene Camera", _dataDir + "images/textures/LiveVideoError.png", _dataDir + "shaders/");
    camera->translation(0, 50, -150);
    camera->lookAt(0, 0, 0);
    camera->clipNear(1);
    camera->clipFar(400);
    camera->focalDist(150);
    camera->camAnim(SLCamAnim::CA_off);
    camera->setInitialState();
    _root3D->addChild(camera);

    //load area model
    loadAugstTempelTheater();

    // Add axis object a world origin
    SLNode* axis = new SLNode(new SLCoordAxis(&assets), "Axis Node");
    axis->setDrawBitsRec(SL_DB_MESHWIRED, false);
    axis->scale(10);
    axis->rotate(-90, 1, 0, 0);
    _root3D->addChild(axis);
}

void AppWAIScene::initLocationAvenches(ErlebAR::AreaId areaId)
{
    if (!Utils::dirExists(_dataDir + "erleb-AR/models/avenches/"))
    {
        HttpUtils::download("https://pallas.ti.bfh.ch/erlebar/models/avenches.zip", _dataDir + "erleb-AR/models/", "erlebar", PASSWORD);
        ZipUtils::unzip(_dataDir + "erleb-AR/models/avenches.zip", _dataDir + "erleb-AR/models/");
    }

    // Create directional light for the sun light
    sunLight = new SLLightDirect(&assets, this, 5.0f);
    sunLight->powers(1.0f, 1.5f, 1.0f);
    sunLight->attenuation(1, 0, 0);
    sunLight->translation(0, 10, 0);
    sunLight->lookAt(10, 0, 10);
    sunLight->doSunPowerAdaptation(true);
    sunLight->createsShadows(true);
    sunLight->createShadowMap(-100, 150, SLVec2f(150, 150), SLVec2i(2048, 2048));
    sunLight->doSmoothShadows(true);
    sunLight->shadowMaxBias(0.02f);
    sunLight->castsShadows(false);
    _root3D->addChild(sunLight);

    //init camera
    camera = new VideoBackgroundCamera("AppWAIScene Camera", _dataDir + "images/textures/LiveVideoError.png", _dataDir + "shaders/");
    camera->translation(0, 50, -150);
    camera->lookAt(0, 0, 0);
    camera->clipNear(1);
    camera->clipFar(400);
    camera->focalDist(150);
    camera->camAnim(SLCamAnim::CA_off);
    camera->setInitialState();
    _root3D->addChild(camera);

    if (areaId == ErlebAR::AreaId::AVENCHES_AMPHITHEATER ||
        areaId == ErlebAR::AreaId::AVENCHES_AMPHITHEATER_ENTRANCE)
        initAreaAvenchesAmphitheater();
    else if (areaId == ErlebAR::AreaId::AVENCHES_CIGOGNIER)
        initAreaAvenchesCigognier();
    else if (areaId == ErlebAR::AreaId::AVENCHES_THEATER)
        initAreaAvenchesTheatre();

    // Add axis object a world origin
    //SLNode* axis = new SLNode(new SLCoordAxis(&assets), "Axis Node");
    //axis->setDrawBitsRec(SL_DB_MESHWIRED, false);
    //axis->scale(10);
    //axis->rotate(-90, 1, 0, 0);
    //_root3D->addChild(axis);
}

void AppWAIScene::initAreaAvenchesAmphitheater()
{
    loadAvenchesAmphitheater();
}

void AppWAIScene::initAreaAvenchesCigognier()
{
    loadAvenchesCigognier();
}

void AppWAIScene::initAreaAvenchesTheatre()
{
    loadAvenchesTheatre();
}

void AppWAIScene::initLocationBern()
{
    if (!Utils::dirExists(_dataDir + "erleb-AR/models/bern/"))
    {
        HttpUtils::download("https://pallas.ti.bfh.ch/erlebar/models/bern.zip", _dataDir + "erleb-AR/models/", "erlebar", PASSWORD);
        ZipUtils::unzip(_dataDir + "erleb-AR/models/bern.zip", _dataDir + "erleb-AR/models/");
    }
    // Create directional light for the sun light
    sunLight = new SLLightDirect(&assets, this, 5.0f);
    sunLight->powers(1.0f, 1.5f, 1.0f);
    sunLight->attenuation(1, 0, 0);
    sunLight->doSunPowerAdaptation(true);
    sunLight->createsShadows(true);
    sunLight->createShadowMap(-100, 150, SLVec2f(150, 150), SLVec2i(2048, 2048));
    sunLight->doSmoothShadows(true);
    sunLight->castsShadows(false);
    _root3D->addChild(sunLight);

    //init camera
    camera = new VideoBackgroundCamera("AppWAIScene Camera", _dataDir + "images/textures/LiveVideoError.png", _dataDir + "shaders/");
    camera->translation(0, 2, 0);
    camera->lookAt(-10, 2, 0);
    camera->clipNear(0.1);
    camera->clipFar(500);
    camera->camAnim(SLCamAnim::CA_off);
    camera->setInitialState();
    _root3D->addChild(camera);

    //load area model
    loadChristoffelBernBahnhofsplatz();

    // Add axis object a world origin (Loeb Ecke)
    /*
    SLNode* axis = new SLNode(new SLCoordAxis(&assets), "Axis Node");
    axis->setDrawBitsRec(SL_DB_MESHWIRED, false);
    axis->scale(10);
    axis->rotate(-90, 1, 0, 0);
    _root3D->addChild(axis);
    */
}

void AppWAIScene::initLocationBiel(SLDeviceRotation* devRot, SLDeviceLocation* devLoc)
{
    if (!Utils::dirExists(_dataDir + "erleb-AR/models/biel/"))
    {
        HttpUtils::download("https://pallas.ti.bfh.ch/erlebar/models/biel.zip", _dataDir + "erleb-AR/models/", "erlebar", PASSWORD);
        ZipUtils::unzip(_dataDir + "erleb-AR/models/biel.zip", _dataDir + "erleb-AR/models/");
    }
    // Create directional light for the sun light
    sunLight = new SLLightDirect(&assets, this, 5.0f);
    sunLight->powers(1.0f, 1.0f, 1.0f);
    sunLight->attenuation(1, 0, 0);
    sunLight->doSunPowerAdaptation(true);
    sunLight->createsShadows(true);
    sunLight->createShadowMap(-100, 150, SLVec2f(150, 150), SLVec2i(2048, 2048));
    sunLight->doSmoothShadows(true);
    sunLight->castsShadows(false);
    _root3D->addChild(sunLight);

    //init camera
    camera = new VideoBackgroundCamera("AppWAIScene Camera", _dataDir + "images/textures/LiveVideoError.png", _dataDir + "shaders/");
    camera->translation(0, 2, 0);
    camera->lookAt(-10, 2, 0);
    camera->clipNear(1);
    camera->clipFar(1000);

    //camera->camAnim(SLCamAnim::CA_off);
    camera->camAnim(SLCamAnim::CA_deviceRotLocYUp);
    camera->devRotLoc(devRot, devLoc);
    if (devRot)
    {
        devRot->offsetMode(SLOffsetMode::OM_fingerX);
        devRot->isUsed(true);
    }
    if (devLoc)
        devLoc->isUsed(true);

    camera->setInitialState();
    _root3D->addChild(camera);

    //load area model
    loadBielBFHRolex();

    // Add axis object at world origin
    SLNode* axis = new SLNode(new SLCoordAxis(&assets), "Axis Node");
    axis->setDrawBitsRec(SL_DB_MESHWIRED, false);
    axis->scale(2);
    axis->rotate(-90, 1, 0, 0);
    _root3D->addChild(axis);
}

void AppWAIScene::initAreaEvilardOffice(SLDeviceRotation* devRot, SLDeviceLocation* devLoc, int svW, int svH)
{
    SLNode* world = new SLNode("World");
    _root3D->addChild(world);

    SLNode* worldAxisNode = new SLNode(new SLCoordAxis(&assets), "World Axis Node");
    worldAxisNode->setDrawBitsRec(SL_DB_MESHWIRED, false);
    world->addChild(worldAxisNode);

    // Create directional light for the sun light
    sunLight = new SLLightDirect(&assets, this, 5.0f);
    sunLight->powers(1.0f, 1.0f, 1.0f);
    sunLight->attenuation(1, 0, 0);
    sunLight->doSunPowerAdaptation(true);
    //sunLight->createsShadows(true);
    //sunLight->createShadowMap(-100, 150, SLVec2f(150, 150), SLVec2i(2048, 2048));
    //sunLight->doSmoothShadows(true);
    //sunLight->castsShadows(false);
    sunLight->drawBits()->set(SL_DB_HIDDEN, true);
    world->addChild(sunLight);

    //init camera
    camera = new VideoBackgroundCamera("AppWAIScene Camera", _dataDir + "images/textures/LiveVideoError.png", _dataDir + "shaders/");
    camera->translation(1.29f, 1.57f, 3.85f);
    camera->lookAt(1.29f, 1.57f, 0.f);
    camera->clipNear(0.5f);
    camera->clipFar(10.f);
    //camera->camAnim(SLCamAnim::CA_deviceRotYUp);
    camera->camAnim(SLCamAnim::CA_off);
    camera->devRotLoc(devRot, nullptr);
    camera->setInitialState();
    world->addChild(camera);

    SLMaterial* yellow    = new SLMaterial(&assets, "mY", SLCol4f(1, 1, 0, 0.5f));
    SLBox*      piano     = new SLBox(&assets, 0.0f, 0.0f, 0.0f, 1.467f, 0.908f, 0.515f, "Box", yellow);
    SLNode*     pianoNode = new SLNode(piano, "Piano Node");
    pianoNode->setDrawBitsRec(SL_DB_CULLOFF, true);
    pianoNode->translation(0.523f, 0.f, 0.f);

    SLNode* axisNode = new SLNode(new SLCoordAxis(&assets), "Axis Node");
    axisNode->setDrawBitsRec(SL_DB_MESHWIRED, false);
    pianoNode->addChild(axisNode);

    SLNode* room = new SLNode("Room");
    room->addChild(pianoNode);
    world->addChild(room);

    //correct room orientation to world (rotation around camera)
    SLVec3f center = {camera->translationOS().x, 0, camera->translationOS().z};
    SLMat4f rot;
    rot.translate(center);
    rot.rotate(60, SLVec3f(0, 1, 0));
    rot.translate(-center); //this translation comes first because of left multiplication
    room->om(rot * room->om());

    if (devRot)
    {
        devRot->offsetMode(SLOffsetMode::OM_fingerX);
        //add horizon visualization
        if (!_root2D)
            _root2D = new SLNode("root2D");

        if (!_font16)
            _font16 = new SLTexFont(_dataDir + "images/fonts/Font16.png", SLGLProgramManager::get(SP_fontTex));
        SLHorizonNode* horizonNode = new SLHorizonNode("Horizon", devRot, _font16, _dataDir + "shaders/", svW, svH);
        _root2D->addChild(horizonNode);
    }
}

void AppWAIScene::initLocationDefault()
{
    // Create directional light for the sun light
    sunLight = new SLLightDirect(&assets, this, 5.0f);
    sunLight->powers(1.0f, 1.0f, 1.0f);
    sunLight->attenuation(1, 0, 0);
    sunLight->doSunPowerAdaptation(true);
    //sunLight->createsShadows(true);
    //sunLight->createShadowMap(-100, 150, SLVec2f(150, 150), SLVec2i(2048, 2048));
    //sunLight->doSmoothShadows(true);
    //sunLight->castsShadows(false);
    _root3D->addChild(sunLight);

    //init camera
    camera = new VideoBackgroundCamera("AppWAIScene Camera", _dataDir + "images/textures/LiveVideoError.png", _dataDir + "shaders/");
    camera->translation(0, 2, 0);
    camera->lookAt(-10, 2, 0);
    camera->clipNear(1);
    camera->clipFar(300);
    camera->camAnim(SLCamAnim::CA_turntableYUp);
    camera->setInitialState();
    _root3D->addChild(camera);

    SLMaterial* yellow  = new SLMaterial(&assets, "mY", SLCol4f(1, 1, 0, 0.5f));
    float       e       = 10.f; //edge length
    SLBox*      box     = new SLBox(&assets, 0.0f, 0.0f, 0.0f, e, e, e, "Box", yellow);
    SLNode*     boxNode = new SLNode(box, "Box Node");
    boxNode->setDrawBitsRec(SL_DB_CULLOFF, true);
    SLNode* axisNode = new SLNode(new SLCoordAxis(&assets), "Axis Node");
    axisNode->setDrawBitsRec(SL_DB_MESHWIRED, false);
    axisNode->scale(e);
    boxNode->addChild(axisNode);
    _root3D->addChild(boxNode);
}

void AppWAIScene::loadChristoffelBernBahnhofsplatz()
{
    SLAssimpImporter importer;
    SLNode*          bern = importer.load(_animManager,
                                 &assets,
                                 _dataDir + "erleb-AR/models/bern/Bern-Bahnhofsplatz2.gltf",
                                 _dataDir + "images/textures/");

    // Make city transparent
    SLNode* UmgD = bern->findChild<SLNode>("Umgebung-Daecher");
    if (!UmgD)
        throw std::runtime_error("Node: Umgebung-Daecher not found!");

    auto updateKtAmbiFnc = [](SLMaterial* m) {
        m->kt(0.5f);
        m->ambient(SLCol4f(.3f, .3f, .3f));
    };

    UmgD->updateMeshMat(updateKtAmbiFnc, true);
    SLNode* UmgF = bern->findChild<SLNode>("Umgebung-Fassaden");
    if (!UmgF)
        throw std::runtime_error("Node: Umgebung-Fassaden not found!");
    UmgF->updateMeshMat(updateKtAmbiFnc, true);

    /*
    bern->findChild<SLNode>("Mauer-Wand")->drawBits()->set(SL_DB_HIDDEN, true);
    bern->findChild<SLNode>("Mauer-Turm")->drawBits()->set(SL_DB_HIDDEN, true);
    bern->findChild<SLNode>("Mauer-Dach")->drawBits()->set(SL_DB_HIDDEN, true);
    bern->findChild<SLNode>("Mauer-Weg")->drawBits()->set(SL_DB_HIDDEN, true);
    bern->findChild<SLNode>("Boden")->drawBits()->set(SL_DB_HIDDEN, true);
    bern->findChild<SLNode>("Graben-Mauern")->drawBits()->set(SL_DB_HIDDEN, true);
    bern->findChild<SLNode>("Graben-Bruecken")->drawBits()->set(SL_DB_HIDDEN, true);
    bern->findChild<SLNode>("Graben-Grass")->drawBits()->set(SL_DB_HIDDEN, true);
    bern->findChild<SLNode>("Graben-Turm-Dach")->drawBits()->set(SL_DB_HIDDEN, true);
    bern->findChild<SLNode>("Graben-Turm-Fahne")->drawBits()->set(SL_DB_HIDDEN, true);
    bern->findChild<SLNode>("Graben-Turm-Stein")->drawBits()->set(SL_DB_HIDDEN, true);
    */

    // Hide some objects
    //bern->findChild<SLNode>("Umgebung-Daecher")->drawBits()->set(SL_DB_HIDDEN, true);
    //bern->findChild<SLNode>("Umgebung-Fassaden")->drawBits()->set(SL_DB_HIDDEN, true);
    bern->findChild<SLNode>("Baldachin-Glas")->drawBits()->set(SL_DB_HIDDEN, true);
    bern->findChild<SLNode>("Baldachin-Stahl")->drawBits()->set(SL_DB_HIDDEN, true);

    // Set the video background shader on the baldachin and the ground
    bern->findChild<SLNode>("Boden")->setMeshMat(camera->matVideoBackground(), true);
    //bern->findChild<SLNode>("Umgebung-Daecher")->setMeshMat(camera->matVideoBackground(), true);
    //bern->findChild<SLNode>("Umgebung-Fassaden")->setMeshMat(camera->matVideoBackground(), true);
    //bern->findChild<SLNode>("Baldachin-Stahl")->setMeshMat(camera->matVideoBackground(), true);
    //bern->findChild<SLNode>("Baldachin-Glas")->setMeshMat(camera->matVideoBackground(), true);

    // Set ambient on all child nodes
    bern->updateMeshMat([](SLMaterial* m) {
        if (m->name() != "Kupfer-dunkel") m->ambient(SLCol4f(.3f, .3f, .3f)); }, true);

    _root3D->addChild(bern);
}

void AppWAIScene::loadBielBFHRolex()
{
    SLAssimpImporter importer;
    SLNode*          bfh = importer.load(_animManager,
                                &assets,
                                _dataDir + "erleb-AR/models/biel/Biel-BFH-Rolex.gltf",
                                _dataDir + "images/textures/");

    // Setup shadow mapping material and replace shader from loader
    SLGLProgram* progPerPixNrmSM = new SLGLProgramGeneric(&assets,
                                                          _dataDir + "shaders/PerPixBlinnSm.vert",
                                                          _dataDir + "shaders/PerPixBlinnSm.frag");
    auto         updateMat       = [=](SLMaterial* mat) { mat->program(progPerPixNrmSM); };
    bfh->updateMeshMat(updateMat, true);

    // Make terrain a video shine trough
    //bfh->findChild<SLNode>("Terrain")->setMeshMat(camera->matVideoBackground(), true);

    // Make buildings transparent
    SLNode* buildings       = bfh->findChild<SLNode>("Buildings");
    SLNode* roofs           = bfh->findChild<SLNode>("Roofs");
    auto    updateTranspFnc = [](SLMaterial* m) { m->kt(0.5f); };
    buildings->updateMeshMat(updateTranspFnc, true);
    roofs->updateMeshMat(updateTranspFnc, true);

    // Set ambient on all child nodes
    bfh->updateMeshMat([](SLMaterial* m) { m->ambient(SLCol4f(.2f, .2f, .2f)); }, true);

    _root3D->addChild(bfh);
}

void AppWAIScene::loadAugstTempelTheater()
{
    SLAssimpImporter importer;
    SLNode*          theaterAndTempel = importer.load(_animManager,
                                             &assets,
                                             _dataDir + "erleb-AR/models/augst/Tempel-Theater-02.gltf",
                                             _dataDir + "images/textures/",
                                             true,    // only meshes
                                             nullptr, // no replacement material
                                             0.4f);   // 40% ambient reflection

    // Rotate to the true geographic rotation
    theaterAndTempel->rotate(16.7f, 0, 1, 0, TS_parent);

    // Setup shadow mapping material and replace shader from loader
    SLGLProgram* progPerPixNrmSM = new SLGLProgramGeneric(&assets,
                                                          _dataDir + "shaders/PerPixBlinnTmNmSm.vert",
                                                          _dataDir + "shaders/PerPixBlinnTmNmSm.frag");
    auto         updateMat       = [=](SLMaterial* mat) { mat->program(progPerPixNrmSM); };
    theaterAndTempel->updateMeshMat(updateMat, true);

    // Let the video shine through on some objects
    theaterAndTempel->findChild<SLNode>("Tmp-Boden")->setMeshMat(camera->matVideoBackground(), true);
    theaterAndTempel->findChild<SLNode>("Tht-Boden")->setMeshMat(camera->matVideoBackground(), true);
    theaterAndTempel->updateMeshMat([](SLMaterial* m) { m->ambient(SLCol4f(.25f, .23f, .15f)); }, true);
    _root3D->addChild(theaterAndTempel);
}

void AppWAIScene::loadAvenchesAmphitheater()
{
    SLAssimpImporter importer;
    SLNode*          amphiTheatre = importer.load(_animManager,
                                         &assets,
                                         _dataDir + "erleb-AR/models/avenches/Aventicum-Amphitheater-AO.gltf",
                                         _dataDir + "images/textures/",
                                         true,    // only meshes
                                         nullptr, // no replacement material
                                         0.4f);   // 40% ambient reflection

    // Rotate to the true geographic rotation
    amphiTheatre->rotate(13.7f, 0, 1, 0, TS_parent);

    // Let the video shine through some objects
    amphiTheatre->findChild<SLNode>("Tht-Aussen-Untergrund")->setMeshMat(camera->matVideoBackground(), true);
    amphiTheatre->findChild<SLNode>("Tht-Eingang-Ost-Boden")->setMeshMat(camera->matVideoBackground(), true);
    amphiTheatre->findChild<SLNode>("Tht-Arenaboden")->setMeshMat(camera->matVideoBackground(), true);
    amphiTheatre->findChild<SLNode>("Tht-Aussen-Terrain")->setMeshMat(camera->matVideoBackground(), true);
    _root3D->addChild(amphiTheatre);
}

void AppWAIScene::loadAvenchesCigognier()
{
    SLAssimpImporter importer;
    SLNode*          cigognier = importer.load(_animManager,
                                      &assets,
                                      _dataDir + "erleb-AR/models/avenches/Aventicum-Cigognier-AO.gltf",
                                      _dataDir + "images/textures/",
                                      true,    // only meshes
                                      nullptr, // no replacement material
                                      0.4f);   // 40% ambient reflection

    cigognier->findChild<SLNode>("Tmp-Parois-Sud")->drawBits()->set(SL_DB_HIDDEN, true);

    // Rotate to the true geographic rotation
    cigognier->rotate(-37.0f, 0, 1, 0, TS_parent);

    _root3D->addChild(cigognier);
}

void AppWAIScene::loadAvenchesTheatre()
{
    SLAssimpImporter importer;
    SLNode*          theatre = importer.load(_animManager,
                                    &assets,
                                    _dataDir + "erleb-AR/models/avenches/Aventicum-Theater1.gltf",
                                    _dataDir + "images/textures/",
                                    true,    // only meshes
                                    nullptr, // no replacement material
                                    0.4f);   // 40% ambient reflection

    // Setup shadow mapping material and replace shader from loader
    SLGLProgram* progPerPixNrmSM = new SLGLProgramGeneric(&assets,
                                                          _dataDir + "shaders/PerPixBlinnTmNmSm.vert",
                                                          _dataDir + "shaders/PerPixBlinnTmNmSm.frag");
    auto         updateMat       = [=](SLMaterial* mat) { mat->program(progPerPixNrmSM); };
    theatre->updateMeshMat(updateMat, true);

    // Rotate to the true geographic rotation
    theatre->rotate(-36.7f, 0, 1, 0, TS_parent);

    theatre->findChild<SLNode>("Tht-Buehnenhaus")->drawBits()->set(SL_DB_HIDDEN, true);

    // Let the video shine through some objects
    theatre->findChild<SLNode>("Tht-Rasen")->setMeshMat(camera->matVideoBackground(), true);
    theatre->findChild<SLNode>("Tht-Boden")->setMeshMat(camera->matVideoBackground(), true);

    _root3D->addChild(theatre);
}

void AppWAIScene::adjustAugmentationTransparency(float kt)
{
    /*
    if (augmentationRoot)
    {
        for (SLNode* child : augmentationRoot->children())
        {
            child->mesh()->mat()->kt(kt);
            child->mesh()->mat()->ambient(SLCol4f(0.3f, 0.3f, 0.3f));
            child->mesh()->init(child);
        }
    }
     */
}

void AppWAIScene::resetMapNode()
{
    mapNode->translation(0, 0, 0);
    mapNode->lookAt(0, 0, -1);
}

/*
 cTw is the camera extrinsic: the world w.r.t. the camera coordinate system.
 We invert to get the camera pose (camera w.r.t the world coordinate system).
 Additionally we change the direction of the axes y and z:
 Wai coordinate axes are like in an opencv image (x right, y down, z into plane), in SLProject
 they are aligned like in opengl (x right, y up, z back)
 */
void AppWAIScene::updateCameraPose(const cv::Mat& cTw)
{
    // update camera node position
    cv::Mat wRc(3, 3, CV_32F);
    cv::Mat wtc(3, 1, CV_32F);

    //inversion of orthogonal rotation matrix
    wRc = (cTw.rowRange(0, 3).colRange(0, 3)).t();
    //inversion of vector
    wtc = -wRc * cTw.rowRange(0, 3).col(3);

    cv::Mat wTc = cv::Mat::eye(4, 4, CV_32F);
    wRc.copyTo(wTc.colRange(0, 3).rowRange(0, 3));
    wtc.copyTo(wTc.rowRange(0, 3).col(3));

    SLMat4f om;
    // clang-format off
    //set and invert y and z axes
    om.setMatrix(wTc.at<float>(0, 0), -wTc.at<float>(0, 1), -wTc.at<float>(0, 2), wTc.at<float>(0, 3),
                 wTc.at<float>(1, 0), -wTc.at<float>(1, 1), -wTc.at<float>(1, 2), wTc.at<float>(1, 3),
                 wTc.at<float>(2, 0), -wTc.at<float>(2, 1), -wTc.at<float>(2, 2), wTc.at<float>(2, 3),
                 wTc.at<float>(3, 0), -wTc.at<float>(3, 1), -wTc.at<float>(3, 2), wTc.at<float>(3, 3));
    // clang-format on
    camera->om(om);
}

void AppWAIScene::renderMapPoints(const std::vector<WAIMapPoint*>& pts)
{
    renderMapPoints("MapPoints", pts, mapPC, mappointsMesh, redMat);
}

void AppWAIScene::renderMarkerCornerMapPoints(const std::vector<WAIMapPoint*>& pts)
{
    renderMapPoints("MarkerCornerMapPoints", pts, mapMarkerCornerPC, mappointsMarkerCornerMesh, blueMat);
}

void AppWAIScene::renderLocalMapPoints(const std::vector<WAIMapPoint*>& pts)
{
    renderMapPoints("LocalMapPoints", pts, mapLocalPC, mappointsLocalMesh, greenMat);
}

void AppWAIScene::renderMatchedMapPoints(const std::vector<WAIMapPoint*>& pts, float opacity)
{
    renderMapPoints("MatchedMapPoints",
                    pts,
                    mapMatchedPC,
                    mappointsMatchedMesh,
                    blueMat,
                    opacity);
}

void AppWAIScene::removeMapPoints()
{
    removeMesh(mapPC, mappointsMesh);
}

void AppWAIScene::removeMarkerCornerMapPoints()
{
    removeMesh(mapMarkerCornerPC, mappointsMarkerCornerMesh);
}

void AppWAIScene::removeLocalMapPoints()
{
    removeMesh(mapLocalPC, mappointsLocalMesh);
}

void AppWAIScene::removeMatchedMapPoints()
{
    removeMesh(mapMatchedPC, mappointsMatchedMesh);
}

void AppWAIScene::renderKeyframes(const std::vector<WAIKeyFrame*>& keyframes, const std::vector<WAIKeyFrame*>& candidates)
{
    keyFrameNode->deleteChildren();
    // TODO(jan): delete keyframe textures

    for (WAIKeyFrame* kf : keyframes)
    {
        if (kf->isBad())
            continue;

        SLKeyframeCamera* cam = new SLKeyframeCamera("KeyFrame " + std::to_string(kf->mnId));
        //set background
        if (kf->getTexturePath().size())
        {
            // TODO(jan): textures are saved in a global textures vector (scene->textures)
            // and should be deleted from there. Otherwise we have a yuuuuge memory leak.
#if 0
        SLGLTexture* texture = new SLGLTexture(kf->getTexturePath());
        _kfTextures.push_back(texture);
        cam->background().texture(texture);
#endif
        }

        cam->setDrawColor();
        for (WAIKeyFrame* ckf : candidates)
        {
            if (kf->mnId == ckf->mnId)
                cam->setDrawColor(SLCol4f::BLUE);
        }

        cv::Mat Twc = kf->getObjectMatrix();

        SLMat4f om;
        om.setMatrix(Twc.at<float>(0, 0),
                     -Twc.at<float>(0, 1),
                     -Twc.at<float>(0, 2),
                     Twc.at<float>(0, 3),
                     Twc.at<float>(1, 0),
                     -Twc.at<float>(1, 1),
                     -Twc.at<float>(1, 2),
                     Twc.at<float>(1, 3),
                     Twc.at<float>(2, 0),
                     -Twc.at<float>(2, 1),
                     -Twc.at<float>(2, 2),
                     Twc.at<float>(2, 3),
                     Twc.at<float>(3, 0),
                     -Twc.at<float>(3, 1),
                     -Twc.at<float>(3, 2),
                     Twc.at<float>(3, 3));
        //om.rotate(180, 1, 0, 0);

        cam->om(om);

        //calculate vertical field of view
        SLfloat fy     = (SLfloat)kf->fy;
        SLfloat cy     = (SLfloat)kf->cy;
        SLfloat fovDeg = 2 * (SLfloat)atan2(cy, fy) * Utils::RAD2DEG;
        cam->fov(fovDeg);
        cam->focalDist(0.11f);
        cam->clipNear(0.1f);
        cam->clipFar(1.0f);

        keyFrameNode->addChild(cam);
    }
}

void AppWAIScene::removeKeyframes()
{
    keyFrameNode->deleteChildren();
}

void AppWAIScene::renderGraphs(const std::vector<WAIKeyFrame*>& kfs,
                               const int&                       minNumOfCovisibles,
                               const bool                       showCovisibilityGraph,
                               const bool                       showSpanningTree,
                               const bool                       showLoopEdges)
{
    SLVVec3f covisGraphPts;
    SLVVec3f spanningTreePts;
    SLVVec3f loopEdgesPts;

    for (auto* kf : kfs)
    {
        cv::Mat Ow = kf->GetCameraCenter();

        //covisibility graph
        const std::vector<WAIKeyFrame*> vCovKFs = kf->GetBestCovisibilityKeyFrames(minNumOfCovisibles);

        if (!vCovKFs.empty())
        {
            for (vector<WAIKeyFrame*>::const_iterator vit = vCovKFs.begin(), vend = vCovKFs.end(); vit != vend; vit++)
            {
                if ((*vit)->mnId < kf->mnId)
                    continue;
                cv::Mat Ow2 = (*vit)->GetCameraCenter();

                covisGraphPts.push_back(SLVec3f(Ow.at<float>(0), Ow.at<float>(1), Ow.at<float>(2)));
                covisGraphPts.push_back(SLVec3f(Ow2.at<float>(0), Ow2.at<float>(1), Ow2.at<float>(2)));
            }
        }

        //spanning tree
        WAIKeyFrame* parent = kf->GetParent();
        if (parent)
        {
            cv::Mat Owp = parent->GetCameraCenter();
            spanningTreePts.push_back(SLVec3f(Ow.at<float>(0), Ow.at<float>(1), Ow.at<float>(2)));
            spanningTreePts.push_back(SLVec3f(Owp.at<float>(0), Owp.at<float>(1), Owp.at<float>(2)));
        }

        //loop edges
        std::set<WAIKeyFrame*> loopKFs = kf->GetLoopEdges();
        for (set<WAIKeyFrame*>::iterator sit = loopKFs.begin(), send = loopKFs.end(); sit != send; sit++)
        {
            if ((*sit)->mnId < kf->mnId)
                continue;
            cv::Mat Owl = (*sit)->GetCameraCenter();
            loopEdgesPts.push_back(SLVec3f(Ow.at<float>(0), Ow.at<float>(1), Ow.at<float>(2)));
            loopEdgesPts.push_back(SLVec3f(Owl.at<float>(0), Owl.at<float>(1), Owl.at<float>(2)));
        }
    }

    removeGraphs();

    if (covisGraphPts.size() && showCovisibilityGraph)
    {
        covisibilityGraphMesh = new SLPolyline(&assets, covisGraphPts, false, "CovisibilityGraph", covisibilityGraphMat);
        covisibilityGraph->addMesh(covisibilityGraphMesh);
        covisibilityGraph->updateAABBRec();
    }

    if (spanningTreePts.size() && showSpanningTree)
    {
        spanningTreeMesh = new SLPolyline(&assets, spanningTreePts, false, "SpanningTree", spanningTreeMat);
        spanningTree->addMesh(spanningTreeMesh);
        spanningTree->updateAABBRec();
    }

    if (loopEdgesPts.size() && showLoopEdges)
    {
        loopEdgesMesh = new SLPolyline(&assets, loopEdgesPts, false, "LoopEdges", loopEdgesMat);
        loopEdges->addMesh(loopEdgesMesh);
        loopEdges->updateAABBRec();
    }
}

void AppWAIScene::removeGraphs()
{
    if (covisibilityGraphMesh)
    {
        if (covisibilityGraph->removeMesh(covisibilityGraphMesh))
        {
            assets.removeMesh(covisibilityGraphMesh);
            delete covisibilityGraphMesh;
            covisibilityGraphMesh = nullptr;
        }
    }
    if (spanningTreeMesh)
    {
        if (spanningTree->removeMesh(spanningTreeMesh))
        {
            assets.removeMesh(spanningTreeMesh);
            delete spanningTreeMesh;
            spanningTreeMesh = nullptr;
        }
    }
    if (loopEdgesMesh)
    {
        if (loopEdges->removeMesh(loopEdgesMesh))
        {
            assets.removeMesh(loopEdgesMesh);
            delete loopEdgesMesh;
            loopEdgesMesh = nullptr;
        }
    }
}

void AppWAIScene::renderMapPoints(std::string                      name,
                                  const std::vector<WAIMapPoint*>& pts,
                                  SLNode*&                         node,
                                  SLPoints*&                       mesh,
                                  SLMaterial*&                     material,
                                  float                            opacity)
{
    //remove old mesh, if it exists
    if (mesh)
    {
        if (node->removeMesh(mesh))
        {
            assets.removeMesh(mesh);
            delete mesh;
            mesh = nullptr;
        }
    }

    //instantiate and add new mesh
    if (pts.size())
    {
        //get points as Vec3f
        std::vector<SLVec3f> points, normals;
        for (auto mapPt : pts)
        {
            if (mapPt->isBad())
                continue;
            WAI::V3 wP = mapPt->worldPosVec();
            WAI::V3 wN = mapPt->normalVec();
            points.push_back(SLVec3f(wP.x, wP.y, wP.z));
            normals.push_back(SLVec3f(wN.x, wN.y, wN.z));
        }

        material->kt(1.f - opacity);

        mesh = new SLPoints(&assets, points, normals, name, material);
        node->addMesh(mesh);
        node->updateAABBRec();
    }
}

void AppWAIScene::removeMesh(SLNode* node, SLMesh* mesh)
{
    if (mesh)
    {
        if (node->removeMesh(mesh))
        {
            assets.removeMesh(mesh);
            delete mesh;
            mesh = nullptr;
        }
    }
}
