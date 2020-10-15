#include "UserGuidanceScene.h"
#include <SLArrow.h>
#include <SLLightSpot.h>

DirectionArrow::DirectionArrow(SLAssetManager& assets, std::string name, SLCamera* cameraNode)
  : SLNode(name)
{
    SLMaterial* blueMat = new SLMaterial(&assets, "m2", SLCol4f::BLUE * 0.3f, SLCol4f::BLUE, 128, 0.5f, 0.0f, 1.0f);

    //define arrow size in the middle of camera frustum
    float   distance            = (cameraNode->clipFar() - cameraNode->clipNear()) * 0.5f;
    SLVec2i frustumSize         = cameraNode->frustumSizeAtDistance(distance);
    SLfloat length              = (float)frustumSize.x / 10.f;
    SLfloat arrowCylinderRadius = length * 1.5f / 5.f;
    SLfloat headLength          = length * 2.f / 5.f;
    SLfloat headWidth           = length * 3.f / 5.f;
    SLuint  slices              = 20;
    SLNode* arrowNode           = new SLNode(new SLArrow(&assets, arrowCylinderRadius, length, headLength, headWidth, slices, "ArrowMesh", blueMat), "ArrowNode");
    arrowNode->rotate(-90, {0, 1, 0});
    arrowNode->translate(0, 0, -length * 0.5f);

    //coordinate axis
    //SLNode* axisNode = new SLNode(new SLCoordAxis(s), "AxisNode");
    //setup final direction arrow
    //directionArrow->addChild(axisNode);
    addChild(arrowNode);
    translate(0, 0, -distance);
}

UserGuidanceScene::UserGuidanceScene(std::string dataDir)
    : SLScene("TestScene", nullptr),
      _dataDir(dataDir)
{
    // Create textures and materials
    //SLGLTexture* texC = new SLGLTexture(s, SLApplication::texturePath + "earth1024_C.jpg");
    //SLMaterial*  m1   = new SLMaterial(&_assets, "m1", texC);
    //SLMaterial* blueMat = new SLMaterial(&_assets, "m2", SLCol4f::BLUE * 0.3f, SLCol4f::BLUE, 128, 0.5f, 0.0f, 1.0f);
    
    // Create a scene group node
    SLNode* scene = new SLNode("scene node");

    // Create a light source node
    SLLightSpot* light1 = new SLLightSpot(&_assets, this, 0.3f);
    light1->translation(0, 0, 5);
    light1->lookAt(0, 0, 0);
    light1->name("light node");
    scene->addChild(light1);

    // Create meshes and nodes
    //SLMesh* rectMesh = new SLRectangle(&_assets, SLVec2f(-5, -5), SLVec2f(5, 5), 25, 25, "rectangle mesh", blueMat);
    //SLNode* rectNode = new SLNode(rectMesh, "rectangle node");
    //rectNode->translation(0, 0, -10);
    //scene->addChild(rectNode);
           
    // Set background color and the root scene node
    //sv->sceneViewCamera()->background().colors(SLCol4f(0.7f, 0.7f, 0.7f), SLCol4f(0.2f, 0.2f, 0.2f));
    camera = new VideoBackgroundCamera("UserGuidance Camera", _dataDir + "images/textures/LiveVideoError.png");
    camera->translation(0, 0, 0.f);
    camera->lookAt(0, 0, -1);
    //for tracking we have to use the field of view from calibration
    camera->clipNear(0.1f);
    camera->clipFar(1000.0f); // Increase to infinity?
    camera->focalDist(0);
    camera->setInitialState();
    //camera->background().colors(SLCol4f(0.7f, 0.7f, 0.7f), SLCol4f(0.2f, 0.2f, 0.2f));
    
    scene->addChild(camera);
    
    _dirArrow = new DirectionArrow(_assets, "DirArrow", camera);
    camera->addChild(_dirArrow);
    _dirArrow->drawBits()->set(SL_DB_HIDDEN, true);
    
    // pass the scene group as root node
    root3D(scene);
}

void UserGuidanceScene::updateArrowRot(SLMat3f camRarrow)
{
    SLMat4f cTa;
    cTa.setTranslation(_dirArrow->om().translation());
    cTa.setRotation(camRarrow);
    _dirArrow->om(cTa);
}

void UserGuidanceScene::hideDirArrow()
{
    _dirArrow->drawBits()->set(SL_DB_HIDDEN, true);
}

void UserGuidanceScene::showDirArrow()
{
    _dirArrow->drawBits()->set(SL_DB_HIDDEN, false);
}
