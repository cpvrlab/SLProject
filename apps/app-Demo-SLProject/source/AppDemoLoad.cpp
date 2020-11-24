//#############################################################################
//  File:      AppDemoSceneLoad.cpp
//  Author:    Marcus Hudritsch
//  Date:      Februar 2018
//  Codestyle: https://github.com/cpvrlab/SLProject/wiki/SLProject-Coding-Style
//  Copyright: Marcus Hudritsch
//             This software is provide under the GNU General Public License
//             Please visit: http://opensource.org/licenses/GPL-3.0
//#############################################################################

#include <GlobalTimer.h>

#include <CVCapture.h>
#include <CVTrackedAruco.h>
#include <CVTrackedChessboard.h>
#include <CVTrackedFaces.h>
#include <CVTrackedFeatures.h>
#include <CVCalibrationEstimator.h>

#include <SLApplication.h>
#include <SLAssimpImporter.h>
#include <SLScene.h>
#include <SLSceneView.h>
#include <SLBox.h>
#include <SLCone.h>
#include <SLCoordAxis.h>
#include <SLCylinder.h>
#include <SLDisk.h>
#include <SLGrid.h>
#include <SLLens.h>
#include <SLLightDirect.h>
#include <SLLightRect.h>
#include <SLLightSpot.h>
#include <SLPoints.h>
#include <SLPolygon.h>
#include <SLRectangle.h>
#include <SLSkybox.h>
#include <SLSphere.h>
#include <SLText.h>
#include <SLColorLUT.h>
#include <SLProjectScene.h>
#include <SLGLProgramManager.h>
#include <Instrumentor.h>
#include <SLArrow.h>

#include <AppDemoGui.h>

#ifdef SL_BUILD_WAI
#    include <CVTrackedWAI.h>
#endif
//-----------------------------------------------------------------------------
// Global pointers declared in AppDemoVideo
extern SLGLTexture* videoTexture;
extern CVTracked*   tracker;
extern SLNode*      trackedNode;
//-----------------------------------------------------------------------------
//! Global pointer to 3D MRI texture for volume rendering for threaded loading
SLGLTexture* gTexMRI3D = nullptr;
//! Creates a recursive sphere group used for the ray tracing scenes
SLNode* SphereGroup(SLProjectScene* s,
                    SLint           depth, // depth of recursion
                    SLfloat         x,
                    SLfloat         y,
                    SLfloat         z,          // position of group
                    SLfloat         scale,      // scale factor
                    SLuint          resolution, // resolution of spheres
                    SLMaterial*     matGlass,   // material for center sphere
                    SLMaterial*     matRed)         // material for orbiting spheres
{
    PROFILE_FUNCTION();

    SLstring name = matGlass->kt() > 0 ? "GlassSphere" : "RedSphere";
    if (depth == 0)
    {
        SLSphere* sphere  = new SLSphere(s, 0.5f * scale, resolution, resolution, name, matRed);
        SLNode*   sphNode = new SLNode(sphere, "Sphere");
        sphNode->translate(x, y, z, TS_object);
        return sphNode;
    }
    else
    {
        depth--;
        SLNode* sGroup = new SLNode("SphereGroup");
        sGroup->translate(x, y, z, TS_object);
        SLuint newRes = (SLuint)std::max((SLint)resolution - 4, 8);
        sGroup->addChild(new SLNode(new SLSphere(s, 0.5f * scale, resolution, resolution, name, matGlass)));
        sGroup->addChild(SphereGroup(s, depth, 0.643951f * scale, 0, 0.172546f * scale, scale / 3, newRes, matRed, matRed));
        sGroup->addChild(SphereGroup(s, depth, 0.172546f * scale, 0, 0.643951f * scale, scale / 3, newRes, matRed, matRed));
        sGroup->addChild(SphereGroup(s, depth, -0.471405f * scale, 0, 0.471405f * scale, scale / 3, newRes, matRed, matRed));
        sGroup->addChild(SphereGroup(s, depth, -0.643951f * scale, 0, -0.172546f * scale, scale / 3, newRes, matRed, matRed));
        sGroup->addChild(SphereGroup(s, depth, -0.172546f * scale, 0, -0.643951f * scale, scale / 3, newRes, matRed, matRed));
        sGroup->addChild(SphereGroup(s, depth, 0.471405f * scale, 0, -0.471405f * scale, scale / 3, newRes, matRed, matRed));
        sGroup->addChild(SphereGroup(s, depth, 0.272166f * scale, 0.544331f * scale, 0.272166f * scale, scale / 3, newRes, matRed, matRed));
        sGroup->addChild(SphereGroup(s, depth, -0.371785f * scale, 0.544331f * scale, 0.099619f * scale, scale / 3, newRes, matRed, matRed));
        sGroup->addChild(SphereGroup(s, depth, 0.099619f * scale, 0.544331f * scale, -0.371785f * scale, scale / 3, newRes, matRed, matRed));
        return sGroup;
    }
}
//-----------------------------------------------------------------------------
//! Build a hierarchical figurine with arms and legs
SLNode* BuildFigureGroup(SLProjectScene* s, SLMaterial* mat, SLbool withAnimation)
{
    SLNode* cyl;
    SLuint  res = 16;

    // Feet
    SLNode* feet = new SLNode("feet group (T13,R6)");
    feet->addMesh(new SLSphere(s, 0.2f, 16, 16, "ankle", mat));
    SLNode* feetbox = new SLNode(new SLBox(s, -0.2f, -0.1f, 0.0f, 0.2f, 0.1f, 0.8f, "foot", mat), "feet (T14)");
    feetbox->translate(0.0f, -0.25f, -0.15f, TS_object);
    feet->addChild(feetbox);
    feet->translate(0.0f, 0.0f, 1.6f, TS_object);
    feet->rotate(-90.0f, 1.0f, 0.0f, 0.0f);

    // Assemble low leg
    SLNode* leglow = new SLNode("low leg group (T11, R5)");
    leglow->addMesh(new SLSphere(s, 0.3f, res, res, "knee", mat));
    cyl = new SLNode(new SLCylinder(s, 0.2f, 1.4f, 1, res, false, false, "shin", mat), "shin (T12)");
    cyl->translate(0.0f, 0.0f, 0.2f, TS_object);
    leglow->addChild(cyl);
    leglow->addChild(feet);
    leglow->translate(0.0f, 0.0f, 1.27f, TS_object);
    leglow->rotate(0, 1.0f, 0.0f, 0.0f);

    // Assemble leg
    SLNode* leg = new SLNode("leg group ()");
    leg->addMesh(new SLSphere(s, 0.4f, res, res, "hip joint", mat));
    cyl = new SLNode(new SLCylinder(s, 0.3f, 1.0f, 1, res, false, false, "thigh", mat), "thigh (T10)");
    cyl->translate(0.0f, 0.0f, 0.27f, TS_object);
    leg->addChild(cyl);
    leg->addChild(leglow);

    // Assemble left & right leg
    SLNode* legLeft = new SLNode("left leg group (T8)");
    legLeft->translate(-0.4f, 0.0f, 2.2f, TS_object);
    legLeft->addChild(leg);
    SLNode* legRight = new SLNode("right leg group (T9)");
    legRight->translate(0.4f, 0.0f, 2.2f, TS_object);
    legRight->addChild(leg->copyRec());

    // Assemble low arm
    SLNode* armlow = new SLNode("low arm group (T6,R4)");
    armlow->addMesh(new SLSphere(s, 0.2f, 16, 16, "elbow", mat));
    cyl = new SLNode(new SLCylinder(s, 0.15f, 1.0f, 1, res, true, false, "low arm", mat), "T7");
    cyl->translate(0.0f, 0.0f, 0.14f, TS_object);
    armlow->addChild(cyl);
    armlow->translate(0.0f, 0.0f, 1.2f, TS_object);
    armlow->rotate(45, -1.0f, 0.0f, 0.0f);

    // Assemble arm
    SLNode* arm = new SLNode("arm group ()");
    arm->addMesh(new SLSphere(s, 0.3f, 16, 16, "shoulder", mat));
    cyl = new SLNode(new SLCylinder(s, 0.2f, 1.0f, 1, res, false, false, "upper arm", mat), "upper arm (T5)");
    cyl->translate(0.0f, 0.0f, 0.2f, TS_object);
    arm->addChild(cyl);
    arm->addChild(armlow);

    // Assemble left & right arm
    SLNode* armLeft = new SLNode("left arm group (T3,R2)");
    armLeft->translate(-1.1f, 0.0f, 0.3f, TS_object);
    armLeft->rotate(10, -1, 0, 0);
    armLeft->addChild(arm);
    SLNode* armRight = new SLNode("right arm group (T4,R3)");
    armRight->translate(1.1f, 0.0f, 0.3f, TS_object);
    armRight->rotate(-60, -1, 0, 0);
    armRight->addChild(arm->copyRec());

    // Assemble head & neck
    SLNode* head = new SLNode(new SLSphere(s, 0.5f, res, res, "head", mat), "head (T1)");
    head->translate(0.0f, 0.0f, -0.7f, TS_object);
    SLNode* neck = new SLNode(new SLCylinder(s, 0.25f, 0.3f, 1, res, false, false, "neck", mat), "neck (T2)");
    neck->translate(0.0f, 0.0f, -0.3f, TS_object);

    // Assemble figure Left
    SLNode* figure = new SLNode("figure group (R1)");
    figure->addChild(new SLNode(new SLBox(s, -0.8f, -0.4f, 0.0f, 0.8f, 0.4f, 2.0f, "chest", mat), "chest"));
    figure->addChild(head);
    figure->addChild(neck);
    figure->addChild(armLeft);
    figure->addChild(armRight);
    figure->addChild(legLeft);
    figure->addChild(legRight);
    figure->rotate(90, 1, 0, 0);

    // Add animations for left leg
    if (withAnimation)
    {
        legLeft = figure->findChild<SLNode>("left leg group (T8)");
        legLeft->rotate(30, -1, 0, 0);
        SLAnimation* anim = s->animManager().createNodeAnimation("figure animation", 2.0f, true, EC_inOutQuint, AL_pingPongLoop);
        anim->createSimpleRotationNodeTrack(legLeft, 60, SLVec3f(1, 0, 0));

        SLNode* legLowLeft = legLeft->findChild<SLNode>("low leg group (T11, R5)");
        anim->createSimpleRotationNodeTrack(legLowLeft, 40, SLVec3f(1, 0, 0));

        SLNode* feetLeft = legLeft->findChild<SLNode>("feet group (T13,R6)");
        anim->createSimpleRotationNodeTrack(feetLeft, 40, SLVec3f(1, 0, 0));
    }

    return figure;
}
//-----------------------------------------------------------------------------
//! appDemoLoadScene builds a scene from source code.
/*! appDemoLoadScene builds a scene from source code. Such a function must be
 passed as a void*-pointer to slCreateScene. It will be called from within
 slCreateSceneView as soon as the view is initialized. You could separate
 different scene by a different sceneID.<br>
 The purpose is to assemble a scene by creating scenegraph objects with nodes
 (SLNode) and meshes (SLMesh). See the scene with SID_Minimal for a minimal
 example of the different steps.
*/
void appDemoLoadScene(SLProjectScene* s, SLSceneView* sv, SLSceneID sceneID)
{
    PROFILE_FUNCTION();

    SLfloat startLoadMS = GlobalTimer::timeMS();

    // Reset non CVTracked and CVCapture infos
    CVTracked::resetTimes();                   // delete all tracker times
    CVCapture::instance()->videoType(VT_NONE); // turn off any video

    // Reset asset pointer from previous scenes
    delete tracker;
    tracker      = nullptr;
    videoTexture = nullptr; // The video texture will be deleted by scene uninit
    trackedNode  = nullptr; // The tracked node will be deleted by scene uninit
    if (sceneID != SID_VolumeRayCastLighted)
        gTexMRI3D = nullptr; // The 3D MRI texture will be deleted by scene uninit

    SLApplication::sceneID = sceneID;

    // reset existing sceneviews
    for (auto* sceneview : SLApplication::sceneViews)
        sceneview->unInit();

    // Initialize all preloaded stuff from SLScene
    s->init();
    // clear gui stuff that depends on scene and sceneview
    AppDemoGui::clear();

    // Deactivate in general the device sensors
    SLApplication::devRot.isUsed(false);
    SLApplication::devLoc.isUsed(false);

    if (SLApplication::sceneID == SID_Empty) //..........................................................
    {
        s->name("No Scene loaded.");
        s->info("No Scene loaded.");
        s->root3D(nullptr);
        sv->sceneViewCamera()->background().colors(SLCol4f(0.7f, 0.7f, 0.7f),
                                                   SLCol4f(0.2f, 0.2f, 0.2f));
        sv->camera(nullptr);
        sv->doWaitOnIdle(true);
    }
    else if (SLApplication::sceneID == SID_Minimal) //...................................................
    {
        // Set scene name and info string
        s->name("Minimal Scene Test");
        s->info("Minimal texture mapping example with one light source.");

        // Create a scene group node
        SLNode* scene = new SLNode("scene node");

        // Create textures and materials
        SLGLTexture* texC = new SLGLTexture(s, SLApplication::texturePath + "earth1024_C.jpg");
        SLMaterial*  m1   = new SLMaterial(s, "m1", texC);

        // Create a light source node
        SLLightSpot* light1 = new SLLightSpot(s, s, 0.3f);
        light1->translation(0, 0, 5);
        light1->lookAt(0, 0, 0);
        light1->name("light node");
        scene->addChild(light1);

        // Create meshes and nodes
        SLMesh* rectMesh = new SLRectangle(s, SLVec2f(-5, -5), SLVec2f(5, 5), 25, 25, "rectangle mesh", m1);
        SLNode* rectNode = new SLNode(rectMesh, "rectangle node");
        scene->addChild(rectNode);

        // Set background color and the root scene node
        sv->sceneViewCamera()->background().colors(SLCol4f(0.7f, 0.7f, 0.7f), SLCol4f(0.2f, 0.2f, 0.2f));

        // pass the scene group as root node
        s->root3D(scene);

        // Save energy
        sv->doWaitOnIdle(true);
    }
    else if (SLApplication::sceneID == SID_Figure) //....................................................
    {
        s->name("Hierarchical Figure Test");
        s->info("Hierarchical scenegraph with multiple subgroups in the figure. The goal is to visualize how you can design a figure with hierarchical transforms containing only rotations and translations. View the bounding boxes with GL > Bounding Boxes. Try also ray tracing.");

        // Create textures and materials
        SLMaterial* m1 = new SLMaterial(s, "m1", SLCol4f::BLACK, SLCol4f::WHITE, 128, 0.2f, 0.8f, 1.5f);
        SLMaterial* m2 = new SLMaterial(s, "m2", SLCol4f::WHITE * 0.3f, SLCol4f::WHITE, 128, 0.5f, 0.0f, 1.0f);

        SLuint  res         = 20;
        SLMesh* rectangle   = new SLRectangle(s, SLVec2f(-5, -5), SLVec2f(5, 5), res, res, "rectangle", m2);
        SLNode* floorRect   = new SLNode(rectangle);
        SLNode* ceilingRect = new SLNode(rectangle);
        floorRect->rotate(90, -1, 0, 0);
        floorRect->translate(0, 0, -5.5f);
        ceilingRect->rotate(90, 1, 0, 0);
        ceilingRect->translate(0, 0, -5.5f);
        ceilingRect->drawBits()->on(SL_DB_NOTSELECTABLE);

        SLCamera* cam1 = new SLCamera("Camera 1");
        cam1->translation(0, 0, 22);
        cam1->lookAt(0, 0, 0);
        cam1->focalDist(22);
        cam1->background().colors(SLCol4f(0.1f, 0.4f, 0.8f));
        cam1->setInitialState();
        cam1->devRotLoc(&SLApplication::devRot, &SLApplication::devLoc);

        SLLightSpot* light1 = new SLLightSpot(s, s, 5, 0, 5, 0.5f);
        light1->powers(0.2f, 0.9f, 0.9f);
        light1->attenuation(1, 0, 0);

        SLNode* figure = BuildFigureGroup(s, m1, false);

        SLNode* scene = new SLNode("scene node");
        scene->addChild(light1);
        scene->addChild(cam1);
        scene->addChild(floorRect);
        scene->addChild(ceilingRect);
        scene->addChild(figure);

        // Set background color, active camera & the root pointer
        sv->camera(cam1);
        s->root3D(scene);
    }
    else if (SLApplication::sceneID == SID_MeshLoad) //..................................................
    {
        s->name("Mesh 3D Loader Test");
        s->info("3D file import test. We use the assimp library to load all 3D file formats including materials, skeletons and animations. ");

        SLMaterial* matBlu = new SLMaterial(s, "Blue", SLCol4f(0, 0, 0.2f), SLCol4f(1, 1, 1), 100, 0.8f, 0);
        SLMaterial* matRed = new SLMaterial(s, "Red", SLCol4f(0.2f, 0, 0), SLCol4f(1, 1, 1), 100, 0.8f, 0);
        SLMaterial* matGre = new SLMaterial(s, "Green", SLCol4f(0, 0.2f, 0), SLCol4f(1, 1, 1), 100, 0.8f, 0);
        SLMaterial* matGra = new SLMaterial(s, "Gray", SLCol4f(0.3f, 0.3f, 0.3f), SLCol4f(1, 1, 1), 100, 0, 0);

        SLCamera* cam1 = new SLCamera("Camera 1");
        cam1->clipNear(.1f);
        cam1->clipFar(30);
        cam1->translation(0, 0, 12);
        cam1->lookAt(0, 0, 0);
        cam1->maxSpeed(20);
        cam1->moveAccel(160);
        cam1->brakeAccel(160);
        cam1->focalDist(12);
        cam1->stereoEyeSeparation(cam1->focalDist() / 30.0f);
        cam1->background().colors(SLCol4f(0.6f, 0.6f, 0.6f), SLCol4f(0.3f, 0.3f, 0.3f));
        cam1->setInitialState();
        cam1->devRotLoc(&SLApplication::devRot, &SLApplication::devLoc);

        SLLightSpot* light1 = new SLLightSpot(s, s, 2.5f, 2.5f, 2.5f, 0.2f);
        light1->powers(0.1f, 1.0f, 1.0f);
        light1->attenuation(1, 0, 0);
        SLAnimation* anim = s->animManager().createNodeAnimation("anim_light1_backforth", 2.0f, true, EC_inOutQuad, AL_pingPongLoop);
        anim->createSimpleTranslationNodeTrack(light1, SLVec3f(0.0f, 0.0f, -5.0f));

        SLLightSpot* light2 = new SLLightSpot(s, s, -2.5f, -2.5f, 2.5f, 0.2f);
        light2->powers(0.1f, 1.0f, 1.0f);
        light2->attenuation(1, 0, 0);
        anim = s->animManager().createNodeAnimation("anim_light2_updown", 2.0f, true, EC_inOutQuint, AL_pingPongLoop);
        anim->createSimpleTranslationNodeTrack(light2, SLVec3f(0.0f, 5.0f, 0.0f));

        SLAssimpImporter importer;

        SLNode* mesh3DS = importer.load(s->animManager(), s, SLApplication::modelPath + "3DS/Halloween/jackolan.3ds", SLApplication::texturePath);
        SLNode* meshFBX = importer.load(s->animManager(), s, SLApplication::modelPath + "FBX/Duck/duck.fbx", SLApplication::texturePath);
        SLNode* meshDAE = importer.load(s->animManager(), s, SLApplication::modelPath + "DAE/AstroBoy/AstroBoy.dae", SLApplication::texturePath);

        // Start animation
        SLAnimPlayback* charAnim = s->animManager().lastAnimPlayback();
        charAnim->playForward();
        charAnim->playbackRate(0.8f);

        // Scale to so that the AstroBoy is about 2 (meters) high.
        if (mesh3DS)
        {
            mesh3DS->scale(0.1f);
            mesh3DS->translate(-22.0f, 1.9f, 3.5f, TS_object);
        }
        if (meshDAE)
        {
            meshDAE->translate(0, -3, 0, TS_object);
            meshDAE->scale(2.7f);
        }
        if (meshFBX)
        {
            meshFBX->scale(0.1f);
            meshFBX->scale(0.1f);
            meshFBX->translate(200, 30, -30, TS_object);
            meshFBX->rotate(-90, 0, 1, 0);
        }

        // define rectangles for the surrounding box
        SLfloat b = 3; // edge size of rectangles
        SLNode *rb, *rl, *rr, *rf, *rt;
        SLuint  res = 20;
        rb          = new SLNode(new SLRectangle(s, SLVec2f(-b, -b), SLVec2f(b, b), res, res, "rectB", matBlu), "rectBNode");
        rb->translate(0, 0, -b, TS_object);
        rl = new SLNode(new SLRectangle(s, SLVec2f(-b, -b), SLVec2f(b, b), res, res, "rectL", matRed), "rectLNode");
        rl->rotate(90, 0, 1, 0);
        rl->translate(0, 0, -b, TS_object);
        rr = new SLNode(new SLRectangle(s, SLVec2f(-b, -b), SLVec2f(b, b), res, res, "rectR", matGre), "rectRNode");
        rr->rotate(-90, 0, 1, 0);
        rr->translate(0, 0, -b, TS_object);
        rf = new SLNode(new SLRectangle(s, SLVec2f(-b, -b), SLVec2f(b, b), res, res, "rectF", matGra), "rectFNode");
        rf->rotate(-90, 1, 0, 0);
        rf->translate(0, 0, -b, TS_object);
        rt = new SLNode(new SLRectangle(s, SLVec2f(-b, -b), SLVec2f(b, b), res, res, "rectT", matGra), "rectTNode");
        rt->rotate(90, 1, 0, 0);
        rt->translate(0, 0, -b, TS_object);

        SLNode* scene = new SLNode("Scene");
        scene->addChild(light1);
        scene->addChild(light2);
        scene->addChild(rb);
        scene->addChild(rl);
        scene->addChild(rr);
        scene->addChild(rf);
        scene->addChild(rt);
        if (mesh3DS) scene->addChild(mesh3DS);
        if (meshFBX) scene->addChild(meshFBX);
        if (meshDAE) scene->addChild(meshDAE);
        scene->addChild(cam1);

        sv->camera(cam1);
        s->root3D(scene);
    }
    else if (SLApplication::sceneID == SID_Revolver) //..................................................
    {
        s->name("Revolving Mesh Test");
        s->info("Examples of revolving mesh objects constructed by rotating a 2D curve. The glass shader reflects and refracts the environment map. Try ray tracing.");

        // Test map material
        SLGLTexture* tex1 = new SLGLTexture(s, SLApplication::texturePath + "Testmap_0512_C.png");
        SLMaterial*  mat1 = new SLMaterial(s, "mat1", tex1);

        // floor material
        SLGLTexture* tex2 = new SLGLTexture(s, SLApplication::texturePath + "wood0_0512_C.jpg");
        SLMaterial*  mat2 = new SLMaterial(s, "mat2", tex2);
        mat2->specular(SLCol4f::BLACK);

        // Back wall material
        SLGLTexture* tex3 = new SLGLTexture(s, SLApplication::texturePath + "bricks1_0256_C.jpg");
        SLMaterial*  mat3 = new SLMaterial(s, "mat3", tex3);
        mat3->specular(SLCol4f::BLACK);

        // Left wall material
        SLGLTexture* tex4 = new SLGLTexture(s, SLApplication::texturePath + "wood2_0512_C.jpg");
        SLMaterial*  mat4 = new SLMaterial(s, "mat4", tex4);
        mat4->specular(SLCol4f::BLACK);

        // Glass material
        SLGLTexture* tex5 = new SLGLTexture(s,
                                            SLApplication::texturePath + "wood2_0256_C.jpg",
                                            SLApplication::texturePath + "wood2_0256_C.jpg",
                                            SLApplication::texturePath + "gray_0256_C.jpg",
                                            SLApplication::texturePath + "wood0_0256_C.jpg",
                                            SLApplication::texturePath + "gray_0256_C.jpg",
                                            SLApplication::texturePath + "bricks1_0256_C.jpg");
        SLMaterial*  mat5 = new SLMaterial(s, "glass", SLCol4f::BLACK, SLCol4f::WHITE, 255, 0.1f, 0.9f, 1.5f);
        mat5->textures().push_back(tex5);
        SLGLProgram* sp1 = new SLGLGenericProgram(s, SLApplication::shaderPath + "RefractReflect.vert", SLApplication::shaderPath + "RefractReflect.frag");
        mat5->program(sp1);

        // Wine material
        SLMaterial* mat6 = new SLMaterial(s, "wine", SLCol4f(0.4f, 0.0f, 0.2f), SLCol4f::BLACK, 255, 0.2f, 0.7f, 1.3f);
        mat6->textures().push_back(tex5);
        mat6->program(sp1);

        // camera
        SLCamera* cam1 = new SLCamera();
        cam1->name("cam1");
        cam1->translation(0, 1, 17);
        cam1->lookAt(0, 1, 0);
        cam1->focalDist(17);
        cam1->background().colors(SLCol4f(0.7f, 0.7f, 0.7f), SLCol4f(0.2f, 0.2f, 0.2f));
        cam1->setInitialState();
        cam1->devRotLoc(&SLApplication::devRot, &SLApplication::devLoc);

        // light
        SLLightSpot* light1 = new SLLightSpot(s, s, 0, 4, 0, 0.3f);
        light1->powers(0.2f, 1.0f, 1.0f);
        light1->attenuation(1, 0, 0);
        SLAnimation* anim = s->animManager().createNodeAnimation("light1_anim", 4.0f);
        anim->createEllipticNodeTrack(light1, 6.0f, A_z, 6.0f, A_x);

        // glass 2D polygon definition for revolution
        SLVVec3f revG;
        revG.push_back(SLVec3f(0.00f, 0.00f)); // foot
        revG.push_back(SLVec3f(2.00f, 0.00f));
        revG.push_back(SLVec3f(2.00f, 0.00f));
        revG.push_back(SLVec3f(2.00f, 0.10f));
        revG.push_back(SLVec3f(1.95f, 0.15f));
        revG.push_back(SLVec3f(0.40f, 0.50f)); // stand
        revG.push_back(SLVec3f(0.25f, 0.60f));
        revG.push_back(SLVec3f(0.20f, 0.70f));
        revG.push_back(SLVec3f(0.30f, 3.00f));
        revG.push_back(SLVec3f(0.30f, 3.00f)); // crack
        revG.push_back(SLVec3f(0.20f, 3.10f));
        revG.push_back(SLVec3f(0.20f, 3.10f));
        revG.push_back(SLVec3f(1.20f, 3.90f)); // outer cup
        revG.push_back(SLVec3f(1.60f, 4.30f));
        revG.push_back(SLVec3f(1.95f, 4.80f));
        revG.push_back(SLVec3f(2.15f, 5.40f));
        revG.push_back(SLVec3f(2.20f, 6.20f));
        revG.push_back(SLVec3f(2.10f, 7.10f));
        revG.push_back(SLVec3f(2.05f, 7.15f));
        revG.push_back(SLVec3f(2.00f, 7.10f)); // inner cup
        revG.push_back(SLVec3f(2.05f, 6.00f));
        SLuint  res   = 30;
        SLNode* glass = new SLNode(new SLRevolver(s, revG, SLVec3f(0, 1, 0), res, true, false, "GlassRev", mat5));
        glass->translate(0.0f, -3.5f, 0.0f, TS_object);

        // wine 2D polyline definition for revolution with two sided material
        SLVVec3f revW;
        revW.push_back(SLVec3f(0.00f, 3.82f));
        revW.push_back(SLVec3f(0.20f, 3.80f));
        revW.push_back(SLVec3f(0.80f, 4.00f));
        revW.push_back(SLVec3f(1.30f, 4.30f));
        revW.push_back(SLVec3f(1.70f, 4.80f));
        revW.push_back(SLVec3f(1.95f, 5.40f));
        revW.push_back(SLVec3f(2.05f, 6.00f));
        SLMesh* wineMesh = new SLRevolver(s, revW, SLVec3f(0, 1, 0), res, true, false, "WineRev", mat6);
        wineMesh->matOut(mat5);
        SLNode* wine = new SLNode(wineMesh);
        wine->translate(0.0f, -3.5f, 0.0f, TS_object);

        // wine fluid top
        SLNode* wineTop = new SLNode(new SLDisk(s, 2.05f, -SLVec3f::AXISY, res, false, "WineRevTop", mat6));
        wineTop->translate(0.0f, 2.5f, 0.0f, TS_object);

        // Other revolver objects
        SLNode* sphere = new SLNode(new SLSphere(s, 1, 16, 16, "sphere", mat1));
        sphere->translate(3, 0, 0, TS_object);
        SLNode* cylinder = new SLNode(new SLCylinder(s, 0.1f, 7, 3, 16, true, true, "cylinder", mat1));
        cylinder->translate(0, 0.5f, 0);
        cylinder->rotate(90, -1, 0, 0);
        cylinder->rotate(30, 0, 1, 0);
        SLNode* cone = new SLNode(new SLCone(s, 1, 3, 3, 16, true, "cone", mat1));
        cone->translate(-3, -1, 0, TS_object);
        cone->rotate(90, -1, 0, 0);

        // Cube dimensions
        SLfloat pL = -9.0f, pR = 9.0f;  // left/right
        SLfloat pB = -3.5f, pT = 14.5f; // bottom/top
        SLfloat pN = 9.0f, pF = -9.0f;  // near/far

        //// bottom rectangle
        SLNode* b = new SLNode(new SLRectangle(s, SLVec2f(pL, -pN), SLVec2f(pR, -pF), 10, 10, "PolygonFloor", mat2));
        b->rotate(90, -1, 0, 0);
        b->translate(0, 0, pB, TS_object);

        // top rectangle
        SLNode* t = new SLNode(new SLRectangle(s, SLVec2f(pL, pF), SLVec2f(pR, pN), 10, 10, "top", mat2));
        t->rotate(90, 1, 0, 0);
        t->translate(0, 0, -pT, TS_object);

        // far rectangle
        SLNode* f = new SLNode(new SLRectangle(s, SLVec2f(pL, pB), SLVec2f(pR, pT), 10, 10, "far", mat3));
        f->translate(0, 0, pF, TS_object);

        // left rectangle
        SLNode* l = new SLNode(new SLRectangle(s, SLVec2f(-pN, pB), SLVec2f(-pF, pT), 10, 10, "left", mat4));
        l->rotate(90, 0, 1, 0);
        l->translate(0, 0, pL, TS_object);

        // right rectangle
        SLNode* r = new SLNode(new SLRectangle(s, SLVec2f(pF, pB), SLVec2f(pN, pT), 10, 10, "right", mat4));
        r->rotate(90, 0, -1, 0);
        r->translate(0, 0, -pR, TS_object);

        SLNode* scene = new SLNode;
        scene->addChild(light1);
        scene->addChild(glass);
        scene->addChild(wine);
        scene->addChild(wineTop);
        scene->addChild(sphere);
        scene->addChild(cylinder);
        scene->addChild(cone);
        scene->addChild(b);
        scene->addChild(f);
        scene->addChild(t);
        scene->addChild(l);
        scene->addChild(r);
        scene->addChild(cam1);

        sv->camera(cam1);
        s->root3D(scene);
    }
    else if (SLApplication::sceneID == SID_LargeModel) //................................................
    {
        SLstring largeFile = SLApplication::modelPath + "PLY/xyzrgb_dragon.ply";

        if (Utils::fileExists(largeFile))
        {
            s->name("Large Model Test");
            s->info("Large Model with 7.2 mio. triangles.");

            // Material for glass
            SLMaterial* diffuseMat = new SLMaterial(s, "diffuseMat", SLCol4f::WHITE, SLCol4f::WHITE);

            SLCamera* cam1 = new SLCamera("Camera 1");
            cam1->translation(0, 0, 220);
            cam1->lookAt(0, 0, 0);
            cam1->clipNear(1);
            cam1->clipFar(10000);
            cam1->focalDist(220);
            cam1->background().colors(SLCol4f(0.7f, 0.7f, 0.7f), SLCol4f(0.2f, 0.2f, 0.2f));
            cam1->setInitialState();
            cam1->devRotLoc(&SLApplication::devRot, &SLApplication::devLoc);

            SLLightSpot* light1 = new SLLightSpot(s, s, 200, 200, 200, 1);
            light1->powers(0.1f, 1.0f, 1.0f);
            light1->attenuation(1, 0, 0);

            SLAssimpImporter importer;
            SLNode*          dragonModel = importer.load(s->animManager(),
                                                s,
                                                largeFile,
                                                SLApplication::texturePath,
                                                true,
                                                diffuseMat,
                                                0.2f,
                                                SLProcess_Triangulate |
                                                  SLProcess_JoinIdenticalVertices);

            SLNode* scene = new SLNode("Scene");
            scene->addChild(light1);
            scene->addChild(dragonModel);
            scene->addChild(cam1);

            sv->camera(cam1);
            s->root3D(scene);
        }
    }
    else if (SLApplication::sceneID == SID_TextureBlend) //..............................................
    {
        s->name("Texture Blending Test");
        s->info("Texture map blending with depth sorting. Trees in view frustum are rendered back to front.");

        SLGLTexture* t1 = new SLGLTexture(s,
                                          SLApplication::texturePath + "tree1_1024_C.png",
                                          GL_LINEAR_MIPMAP_LINEAR,
                                          GL_LINEAR,
                                          TT_diffuse,
                                          GL_CLAMP_TO_EDGE,
                                          GL_CLAMP_TO_EDGE);
        SLGLTexture* t2 = new SLGLTexture(s,
                                          SLApplication::texturePath + "grass0512_C.jpg",
                                          GL_LINEAR_MIPMAP_LINEAR,
                                          GL_LINEAR);

        SLMaterial* m1 = new SLMaterial(s, "m1", SLCol4f(1, 1, 1), SLCol4f(0, 0, 0), 100);
        SLMaterial* m2 = new SLMaterial(s, "m2", SLCol4f(1, 1, 1), SLCol4f(0, 0, 0), 100);

        SLGLProgram* sp = new SLGLGenericProgram(s,
                                                 SLApplication::shaderPath + "PerVrtTextureOnly.vert",
                                                 SLApplication::shaderPath + "PerVrtTextureOnly.frag");
        m1->program(sp);
        m1->textures().push_back(t1);
        m2->textures().push_back(t2);

        SLCamera* cam1 = new SLCamera("Camera 1");
        cam1->translation(0, 3, 25);
        cam1->lookAt(0, 0, 10);
        cam1->focalDist(25);
        cam1->background().colors(SLCol4f(0.6f, 0.6f, 1));
        cam1->setInitialState();
        cam1->devRotLoc(&SLApplication::devRot, &SLApplication::devLoc);

        SLLightSpot* light = new SLLightSpot(s, s, 0.1f);
        light->translation(5, 5, 5);
        light->lookAt(0, 0, 0);
        light->attenuation(1, 0, 0);

        // Build arrays for polygon vertices and texcoords for tree
        SLVVec3f pNW, pSE;
        SLVVec2f tNW, tSE;
        pNW.push_back(SLVec3f(0, 0, 0));
        tNW.push_back(SLVec2f(0.5f, 0.0f));
        pNW.push_back(SLVec3f(1, 0, 0));
        tNW.push_back(SLVec2f(1.0f, 0.0f));
        pNW.push_back(SLVec3f(1, 2, 0));
        tNW.push_back(SLVec2f(1.0f, 1.0f));
        pNW.push_back(SLVec3f(0, 2, 0));
        tNW.push_back(SLVec2f(0.5f, 1.0f));
        pSE.push_back(SLVec3f(-1, 0, 0));
        tSE.push_back(SLVec2f(0.0f, 0.0f));
        pSE.push_back(SLVec3f(0, 0, 0));
        tSE.push_back(SLVec2f(0.5f, 0.0f));
        pSE.push_back(SLVec3f(0, 2, 0));
        tSE.push_back(SLVec2f(0.5f, 1.0f));
        pSE.push_back(SLVec3f(-1, 2, 0));
        tSE.push_back(SLVec2f(0.0f, 1.0f));

        // Build tree out of 4 polygons
        SLNode* p1 = new SLNode(new SLPolygon(s, pNW, tNW, "Tree+X", m1));
        SLNode* p2 = new SLNode(new SLPolygon(s, pNW, tNW, "Tree-Z", m1));
        p2->rotate(90, 0, 1, 0);
        SLNode* p3 = new SLNode(new SLPolygon(s, pSE, tSE, "Tree-X", m1));
        SLNode* p4 = new SLNode(new SLPolygon(s, pSE, tSE, "Tree+Z", m1));
        p4->rotate(90, 0, 1, 0);

        // Turn face culling off so that we see both sides
        p1->drawBits()->on(SL_DB_CULLOFF);
        p2->drawBits()->on(SL_DB_CULLOFF);
        p3->drawBits()->on(SL_DB_CULLOFF);
        p4->drawBits()->on(SL_DB_CULLOFF);

        // Build tree group
        SLNode* tree = new SLNode("grTree");
        tree->addChild(p1);
        tree->addChild(p2);
        tree->addChild(p3);
        tree->addChild(p4);

        // Build arrays for polygon vertices and texcoords for ground
        SLVVec3f pG;
        SLVVec2f tG;
        SLfloat  size = 22.0f;
        pG.push_back(SLVec3f(-size, 0, size));
        tG.push_back(SLVec2f(0, 0));
        pG.push_back(SLVec3f(size, 0, size));
        tG.push_back(SLVec2f(30, 0));
        pG.push_back(SLVec3f(size, 0, -size));
        tG.push_back(SLVec2f(30, 30));
        pG.push_back(SLVec3f(-size, 0, -size));
        tG.push_back(SLVec2f(0, 30));

        SLNode* scene = new SLNode("grScene");
        scene->addChild(light);
        scene->addChild(tree);
        scene->addChild(new SLNode(new SLPolygon(s, pG, tG, "Ground", m2)));

        //create 21*21*21-1 references around the center tree
        SLint res = 10;
        for (SLint iZ = -res; iZ <= res; ++iZ)
        {
            for (SLint iX = -res; iX <= res; ++iX)
            {
                if (iX != 0 || iZ != 0)
                {
                    SLNode* t = tree->copyRec();
                    t->translate(float(iX) * 2 + Utils::random(0.7f, 1.4f),
                                 0,
                                 float(iZ) * 2 + Utils::random(0.7f, 1.4f),
                                 TS_object);
                    t->rotate(Utils::random(0, 90), 0, 1, 0);
                    t->scale(Utils::random(0.5f, 1.0f));
                    scene->addChild(t);
                }
            }
        }

        scene->addChild(cam1);

        sv->camera(cam1);
        s->root3D(scene);
    }
    else if (SLApplication::sceneID == SID_TextureFilter) //.............................................
    {
        s->name("Texture Filer Test");
        s->info("Texture filters: Bottom: nearest, left: linear, top: linear mipmap, right: anisotropic");

        // Create 4 textures with different filter modes
        SLGLTexture* texB = new SLGLTexture(s, SLApplication::texturePath + "brick0512_C.png", GL_NEAREST, GL_NEAREST);
        SLGLTexture* texL = new SLGLTexture(s, SLApplication::texturePath + "brick0512_C.png", GL_LINEAR, GL_LINEAR);
        SLGLTexture* texT = new SLGLTexture(s, SLApplication::texturePath + "brick0512_C.png", GL_LINEAR_MIPMAP_LINEAR, GL_LINEAR);
        SLGLTexture* texR = new SLGLTexture(s, SLApplication::texturePath + "brick0512_C.png", SL_ANISOTROPY_MAX, GL_LINEAR);

        // define materials with textureOnly shader, no light needed
        SLMaterial* matB = new SLMaterial(s, "matB", texB, nullptr, nullptr, nullptr, SLGLProgramManager::get(SP_TextureOnly));
        SLMaterial* matL = new SLMaterial(s, "matL", texL, nullptr, nullptr, nullptr, SLGLProgramManager::get(SP_TextureOnly));
        SLMaterial* matT = new SLMaterial(s, "matT", texT, nullptr, nullptr, nullptr, SLGLProgramManager::get(SP_TextureOnly));
        SLMaterial* matR = new SLMaterial(s, "matR", texR, nullptr, nullptr, nullptr, SLGLProgramManager::get(SP_TextureOnly));

        // build polygons for bottom, left, top & right side
        SLVVec3f VB;
        VB.push_back(SLVec3f(-0.5f, -0.5f, 1.0f));
        VB.push_back(SLVec3f(0.5f, -0.5f, 1.0f));
        VB.push_back(SLVec3f(0.5f, -0.5f, -2.0f));
        VB.push_back(SLVec3f(-0.5f, -0.5f, -2.0f));
        SLVVec2f T;
        T.push_back(SLVec2f(0.0f, 2.0f));
        T.push_back(SLVec2f(0.0f, 0.0f));
        T.push_back(SLVec2f(6.0f, 0.0f));
        T.push_back(SLVec2f(6.0f, 2.0f));
        SLNode* polyB = new SLNode(new SLPolygon(s, VB, T, "PolygonB", matB));

        SLVVec3f VL;
        VL.push_back(SLVec3f(-0.5f, 0.5f, 1.0f));
        VL.push_back(SLVec3f(-0.5f, -0.5f, 1.0f));
        VL.push_back(SLVec3f(-0.5f, -0.5f, -2.0f));
        VL.push_back(SLVec3f(-0.5f, 0.5f, -2.0f));
        SLNode* polyL = new SLNode(new SLPolygon(s, VL, T, "PolygonL", matL));

        SLVVec3f VT;
        VT.push_back(SLVec3f(0.5f, 0.5f, 1.0f));
        VT.push_back(SLVec3f(-0.5f, 0.5f, 1.0f));
        VT.push_back(SLVec3f(-0.5f, 0.5f, -2.0f));
        VT.push_back(SLVec3f(0.5f, 0.5f, -2.0f));
        SLNode* polyT = new SLNode(new SLPolygon(s, VT, T, "PolygonT", matT));

        SLVVec3f VR;
        VR.push_back(SLVec3f(0.5f, -0.5f, 1.0f));
        VR.push_back(SLVec3f(0.5f, 0.5f, 1.0f));
        VR.push_back(SLVec3f(0.5f, 0.5f, -2.0f));
        VR.push_back(SLVec3f(0.5f, -0.5f, -2.0f));
        SLNode* polyR = new SLNode(new SLPolygon(s, VR, T, "PolygonR", matR));

        // 3D Texture Mapping on a pyramid
        SLVstring tex3DFiles;
        for (SLint i = 0; i < 256; ++i) tex3DFiles.push_back(SLApplication::texturePath + "Wave_radial10_256C.jpg");
        SLGLTexture* tex3D = new SLGLTexture(s, tex3DFiles);
        SLGLProgram* spr3D = new SLGLGenericProgram(s, SLApplication::shaderPath + "TextureOnly3D.vert", SLApplication::shaderPath + "TextureOnly3D.frag");
        SLMaterial*  mat3D = new SLMaterial(s, "mat3D", tex3D, nullptr, nullptr, nullptr, spr3D);

        // Create 3D textured pyramid mesh and node
        SLMesh* pyramid = new SLMesh(s, "Pyramid");
        pyramid->mat(mat3D);
        pyramid->P          = {{-1, -1, 1}, {1, -1, 1}, {1, -1, -1}, {-1, -1, -1}, {0, 2, 0}};
        pyramid->I16        = {0, 3, 1, 1, 3, 2, 4, 0, 1, 4, 1, 2, 4, 2, 3, 4, 3, 0};
        SLNode* pyramidNode = new SLNode(pyramid, "Pyramid");
        pyramidNode->scale(0.2f);
        pyramidNode->translate(0, 0, -3);

        // Create 3D textured sphere mesh and node
        SLNode*   sphere = new SLNode(new SLSphere(s, 0.2f, 16, 16, "Sphere", mat3D));
        SLCamera* cam1   = new SLCamera("Camera 1");
        cam1->translation(0, 0, 2.2f);
        cam1->lookAt(0, 0, 0);
        cam1->focalDist(2.2f);
        cam1->background().colors(SLCol4f(0.2f, 0.2f, 0.2f));
        cam1->setInitialState();
        cam1->devRotLoc(&SLApplication::devRot, &SLApplication::devLoc);

        SLNode* scene = new SLNode();
        scene->addChild(polyB);
        scene->addChild(polyL);
        scene->addChild(polyT);
        scene->addChild(polyR);
        scene->addChild(sphere);
        scene->addChild(cam1);
        sv->camera(cam1);
        s->root3D(scene);
    }
    else if (SLApplication::sceneID == SID_FrustumCull) //...............................................
    {
        s->name("Frustum Culling Test");
        s->info("View frustum culling: Only objects in view frustum are rendered. You can turn view culling off in the render flags.");

        // create texture
        SLGLTexture* tex  = new SLGLTexture(s, SLApplication::texturePath + "earth1024_C.jpg");
        SLMaterial*  mat1 = new SLMaterial(s, "mat1", tex);

        SLCamera* cam1 = new SLCamera("Camera 1");
        cam1->clipNear(0.1f);
        cam1->clipFar(100);
        cam1->translation(0, 0, 1);
        cam1->lookAt(0, 0, 0);
        cam1->focalDist(5);
        cam1->background().colors(SLCol4f(0.1f, 0.1f, 0.1f));
        cam1->setInitialState();
        cam1->devRotLoc(&SLApplication::devRot, &SLApplication::devLoc);

        SLLightSpot* light1 = new SLLightSpot(s, s, 10, 10, 10, 0.3f);
        light1->powers(0.2f, 0.8f, 1.0f);
        light1->attenuation(1, 0, 0);

        SLNode* scene = new SLNode;
        scene->addChild(cam1);
        scene->addChild(light1);

        // add one single sphere in the center
        SLuint  res    = 16;
        SLNode* sphere = new SLNode(new SLSphere(s, 0.15f, res, res, "mySphere", mat1));
        scene->addChild(sphere);

        // create spheres around the center sphere
        SLint size = 10;
        for (SLint iZ = -size; iZ <= size; ++iZ)
        {
            for (SLint iY = -size; iY <= size; ++iY)
            {
                for (SLint iX = -size; iX <= size; ++iX)
                {
                    if (iX != 0 || iY != 0 || iZ != 0)
                    {
                        SLNode* sph = sphere->copyRec();
                        sph->translate(float(iX), float(iY), float(iZ), TS_object);
                        scene->addChild(sph);
                    }
                }
            }
        }

        SLuint num = (SLuint)(size + size + 1);
        SL_LOG("Triangles on GPU: %u", res * res * 2 * num * num * num);

        sv->camera(cam1);
        sv->doWaitOnIdle(false);
        s->root3D(scene);
    }
    else if (SLApplication::sceneID == SID_MassiveScene) //..............................................
    {
        s->name("Massive Data Test");
        s->info("No data is shared on the GPU. Check Memory consumption.");

        SLCamera* cam1 = new SLCamera("Camera 1");
        cam1->clipNear(0.1f);
        cam1->clipFar(100);
        cam1->translation(0, 0, 50);
        cam1->lookAt(0, 0, 0);
        cam1->focalDist(50);
        cam1->background().colors(SLCol4f(0.1f, 0.1f, 0.1f));
        cam1->setInitialState();

        SLLightSpot* light1 = new SLLightSpot(s, s, 15, 15, 15, 0.3f);
        light1->powers(0.2f, 0.8f, 1.0f);
        light1->attenuation(1, 0, 0);

        SLNode* scene = new SLNode;
        scene->addChild(cam1);
        scene->addChild(light1);

        // Generate NUM_MAT materials
        const int   NUM_MAT = 20;
        SLMaterial* mat[NUM_MAT];
        for (int i = 0; i < NUM_MAT; ++i)
        {
            SLGLProgram* sp      = new SLGLGenericProgram(s,
                                                     SLApplication::shaderPath + "PerPixBlinnTex.vert",
                                                     SLApplication::shaderPath + "PerPixBlinnTex.frag");
            SLGLTexture* texC    = new SLGLTexture(s, SLApplication::texturePath + "earth2048_C.jpg"); // color map
            SLstring     matName = "mat-" + std::to_string(i);
            mat[i]               = new SLMaterial(s, matName.c_str(), texC, nullptr, nullptr, nullptr, sp);
            SLCol4f color;
            color.hsva2rgba(SLVec3f(Utils::TWOPI * (float)i / (float)NUM_MAT, 1.0f, 1.0f));
            mat[i]->diffuse(color);
        }

        // create a 3D array of spheres
        SLint  halfSize = 10;
        SLuint n        = 0;
        for (SLint iZ = -halfSize; iZ <= halfSize; ++iZ)
        {
            for (SLint iY = -halfSize; iY <= halfSize; ++iY)
            {
                for (SLint iX = -halfSize; iX <= halfSize; ++iX)
                {
                    // Choose a random material index
                    SLuint   res      = 36;
                    SLint    iMat     = (SLint)Utils::random(0, NUM_MAT - 1);
                    SLstring nodeName = "earth-" + std::to_string(n);

                    // Create a new sphere and node and translate it
                    SLSphere* earth  = new SLSphere(s, 0.3f, res, res, nodeName, mat[iMat]);
                    SLNode*   sphere = new SLNode(earth);
                    sphere->translate(float(iX), float(iY), float(iZ), TS_object);
                    scene->addChild(sphere);
                    //SL_LOG("Earth: %000d (Mat: %00d)", n, iMat);
                    n++;
                }
            }
        }

        sv->camera(cam1);
        sv->doWaitOnIdle(false);
        s->root3D(scene);
    }
    else if (SLApplication::sceneID == SID_2Dand3DText) //...............................................
    {
        s->name("2D & 3D Text Test");
        s->info("All 3D objects are in the _root3D scene and the center text is in the _root2D scene and rendered in orthographic projection in screen space.");

        SLMaterial* m1 = new SLMaterial(s, "m1", SLCol4f::RED);

        SLCamera* cam1 = new SLCamera("Camera 1");
        cam1->clipNear(0.1f);
        cam1->clipFar(100);
        cam1->translation(0, 0, 5);
        cam1->lookAt(0, 0, 0);
        cam1->focalDist(5);
        cam1->background().colors(SLCol4f(0.1f, 0.1f, 0.1f));
        cam1->setInitialState();
        cam1->devRotLoc(&SLApplication::devRot, &SLApplication::devLoc);

        SLLightSpot* light1 = new SLLightSpot(s, s, 10, 10, 10, 0.3f);
        light1->powers(0.2f, 0.8f, 1.0f);
        light1->attenuation(1, 0, 0);

        // Because all text objects get their sizes in pixels we have to scale them down
        SLfloat  scale = 0.01f;
        SLstring txt   = "This is text in 3D with font07";
        SLVec2f  size  = SLProjectScene::font07->calcTextSize(txt);
        SLNode*  t07   = new SLText(txt, SLProjectScene::font07);
        t07->translate(-size.x * 0.5f * scale, 1.0f, 0);
        t07->scale(scale);

        txt         = "This is text in 3D with font09";
        size        = SLProjectScene::font09->calcTextSize(txt);
        SLNode* t09 = new SLText(txt, SLProjectScene::font09);
        t09->translate(-size.x * 0.5f * scale, 0.8f, 0);
        t09->scale(scale);

        txt         = "This is text in 3D with font12";
        size        = SLProjectScene::font12->calcTextSize(txt);
        SLNode* t12 = new SLText(txt, SLProjectScene::font12);
        t12->translate(-size.x * 0.5f * scale, 0.6f, 0);
        t12->scale(scale);

        txt         = "This is text in 3D with font20";
        size        = SLProjectScene::font20->calcTextSize(txt);
        SLNode* t20 = new SLText(txt, SLProjectScene::font20);
        t20->translate(-size.x * 0.5f * scale, -0.8f, 0);
        t20->scale(scale);

        txt         = "This is text in 3D with font22";
        size        = SLProjectScene::font22->calcTextSize(txt);
        SLNode* t22 = new SLText(txt, SLProjectScene::font22);
        t22->translate(-size.x * 0.5f * scale, -1.2f, 0);
        t22->scale(scale);

        // Now create 2D text but don't scale it (all sizes in pixels)
        txt           = "This is text in 2D with font16";
        size          = SLProjectScene::font16->calcTextSize(txt);
        SLNode* t2D16 = new SLText(txt, SLProjectScene::font16);
        t2D16->translate(-size.x * 0.5f, 0, 0);

        // Assemble 3D scene as usual with camera and light
        SLNode* scene3D = new SLNode("root3D");
        scene3D->addChild(cam1);
        scene3D->addChild(light1);
        scene3D->addChild(new SLNode(new SLSphere(s, 0.5f, 32, 32, "Sphere", m1)));
        scene3D->addChild(t07);
        scene3D->addChild(t09);
        scene3D->addChild(t12);
        scene3D->addChild(t20);
        scene3D->addChild(t22);

        // Assemble 2D scene
        SLNode* scene2D = new SLNode("root2D");
        scene2D->addChild(t2D16);

        sv->camera(cam1);
        sv->doWaitOnIdle(true);

        s->root3D(scene3D);
        s->root2D(scene2D);
    }
    else if (SLApplication::sceneID == SID_PointClouds) //...............................................
    {
        s->name("Point Clouds Test");
        s->info("Point Clouds with normal and uniform distribution");

        SLCamera* cam1 = new SLCamera("Camera 1");
        cam1->clipNear(0.1f);
        cam1->clipFar(100);
        cam1->translation(0, 0, 15);
        cam1->lookAt(0, 0, 0);
        cam1->focalDist(15);
        cam1->background().colors(SLCol4f(0.1f, 0.1f, 0.1f));
        cam1->setInitialState();
        cam1->devRotLoc(&SLApplication::devRot, &SLApplication::devLoc);

        SLLightSpot* light1 = new SLLightSpot(s, s, 10, 10, 10, 0.3f);
        light1->powers(0.2f, 0.8f, 1.0f);
        light1->attenuation(1, 0, 0);

        SLMaterial* pcMat1 = new SLMaterial(s, "Red", SLCol4f::RED);
        pcMat1->program(new SLGLGenericProgram(s, SLApplication::shaderPath + "ColorUniformPoint.vert", SLApplication::shaderPath + "Color.frag"));
        pcMat1->program()->addUniform1f(new SLGLUniform1f(UT_const, "u_pointSize", 3.0f));
        SLRnd3fNormal rndN(SLVec3f(0, 0, 0), SLVec3f(5, 2, 1));
        SLNode*       pc1 = new SLNode(new SLPoints(s, 1000, rndN, "PC1", pcMat1));
        pc1->translate(-5, 0, 0);

        SLMaterial* pcMat2 = new SLMaterial(s, "Green", SLCol4f::GREEN);
        pcMat2->program(new SLGLGenericProgram(s, SLApplication::shaderPath + "ColorUniform.vert", SLApplication::shaderPath + "Color.frag"));
        SLRnd3fUniform rndU(SLVec3f(0, 0, 0), SLVec3f(2, 3, 5));
        SLNode*        pc2 = new SLNode(new SLPoints(s, 1000, rndU, "PC2", pcMat2));
        pc2->translate(5, 0, 0);

        SLNode* scene = new SLNode("scene");
        scene->addChild(cam1);
        scene->addChild(light1);
        scene->addChild(pc1);
        scene->addChild(pc2);

        sv->camera(cam1);
        sv->doWaitOnIdle(false);
        s->root3D(scene);
    }
    else if (SLApplication::sceneID == SID_ShaderPerPixelBlinn ||
             SLApplication::sceneID == SID_ShaderPerVertexBlinn) //......................................
    {
        SLMaterial*  mL = nullptr;
        SLMaterial*  mM = nullptr;
        SLMaterial*  mR = nullptr;
        SLGLProgram* pL = new SLGLGenericProgram(s,
                                                 SLApplication::shaderPath + "PerPixBlinnTex.vert",
                                                 SLApplication::shaderPath + "PerPixBlinnTex.frag");
        SLGLProgram* pM = new SLGLGenericProgram(s,
                                                 SLApplication::shaderPath + "PerPixBlinn.vert",
                                                 SLApplication::shaderPath + "PerPixBlinn.frag");

        SLGLTexture* texC = new SLGLTexture(s, SLApplication::texturePath + "earth2048_C.jpg"); // color map

        if (SLApplication::sceneID == SID_ShaderPerPixelBlinn)
        {
            s->name("Blinn-Phong per pixel lighting");
            s->info("Per-pixel lighting with Blinn-Phong light model. "
                    "The reflection of 5 light sources is calculated per pixel. "
                    "Some of the lights are attached to the camera, some are in the scene.");
            SLGLTexture*   texN   = new SLGLTexture(s, SLApplication::texturePath + "earth2048_N.jpg"); // normal map
            SLGLTexture*   texH   = new SLGLTexture(s, SLApplication::texturePath + "earth2048_H.jpg"); // height map
            SLGLProgram*   pR     = new SLGLGenericProgram(s,
                                                     SLApplication::shaderPath + "PerPixBlinnTexNrm.vert",
                                                     SLApplication::shaderPath + "PerPixBlinnTexNrmParallax.frag");
            SLGLUniform1f* scale  = new SLGLUniform1f(UT_const, "u_scale", 0.02f, 0.002f, 0, 1);
            SLGLUniform1f* offset = new SLGLUniform1f(UT_const, "u_offset", -0.02f, 0.002f, -1, 1);
            pR->addUniform1f(scale);
            pR->addUniform1f(offset);
            mL = new SLMaterial(s, "mL", texC, nullptr, nullptr, nullptr, pL);
            mM = new SLMaterial(s, "mM", nullptr, nullptr, nullptr, nullptr, pM);
            mR = new SLMaterial(s, "mR", texC, texN, texH, nullptr, pR);
        }
        else
        {
            s->name("Blinn-Phong per vertex lighting");
            s->info("Per-vertex lighting with Blinn-Phong light model. "
                    "The reflection of 5 light sources is calculated per vertex. "
                    "Some of the lights are attached to the camera, some are in the scene.");
            mL = new SLMaterial(s, "mL", texC);
            mM = new SLMaterial(s, "mM");
            mR = new SLMaterial(s, "mR", texC);
        }

        mM->shininess(500);

        // Base root group node for the scene
        SLNode* scene = new SLNode;

        SLCamera* cam1 = new SLCamera("Camera 1");
        cam1->translation(0, 0, 7);
        cam1->lookAt(0, 0, 0);
        cam1->focalDist(7);
        cam1->background().colors(SLCol4f(0.1f, 0.1f, 0.1f));
        cam1->setInitialState();
        scene->addChild(cam1);

        // Define 5 light sources

        // A rectangular white light attached to the camera
        SLLightRect* lightW = new SLLightRect(s, s, 2.0f, 1.0f);
        lightW->ambiDiffPowers(0, 5);
        lightW->translation(0, 2.5f, 0);
        lightW->translation(0, 2.5f, -7);
        lightW->rotate(-90, 1, 0, 0);
        lightW->attenuation(0, 0, 1);
        cam1->addChild(lightW);

        // A red point light from the front attached in the scene
        SLLightSpot* lightR = new SLLightSpot(s, s, 0.1f);
        lightR->ambientColor(SLCol4f(0, 0, 0));
        lightR->diffuseColor(SLCol4f(1, 0, 0));
        lightR->specularColor(SLCol4f(1, 0, 0));
        lightR->translation(0, 0, 2);
        lightR->lookAt(0, 0, 0);
        lightR->attenuation(0, 0, 1);
        scene->addChild(lightR);

        // A green spot head light with 40 deg. spot angle from front right
        SLLightSpot* lightG = new SLLightSpot(s, s, 0.1f, 20, true);
        lightG->ambientColor(SLCol4f(0, 0, 0));
        lightG->diffuseColor(SLCol4f(0, 1, 0));
        lightG->specularColor(SLCol4f(0, 1, 0));
        lightG->translation(1.5f, 1, -5.5f);
        lightG->lookAt(0, 0, -7);
        lightG->attenuation(1, 0, 0);
        cam1->addChild(lightG);

        // A blue spot light with 40 deg. spot angle from front left
        SLLightSpot* lightB = new SLLightSpot(s, s, 0.1f, 20.0f, true);
        lightB->ambientColor(SLCol4f(0, 0, 0));
        lightB->diffuseColor(SLCol4f(0, 0, 1));
        lightB->specularColor(SLCol4f(0, 0, 1));
        lightB->translation(-1.5f, 1.5f, 1.5f);
        lightB->lookAt(0, 0, 0);
        lightB->attenuation(1, 0, 0);
        SLAnimation* light3Anim = s->animManager().createNodeAnimation("Ball3_anim",
                                                                       1.0f,
                                                                       true,
                                                                       EC_outQuad,
                                                                       AL_pingPongLoop);
        light3Anim->createSimpleTranslationNodeTrack(lightB, SLVec3f(0, -2, 0));
        scene->addChild(lightB);

        // A yellow directional light from the back-bottom
        // Do constant attenuation for directional lights since it is infinitely far away
        SLLightDirect* lightY = new SLLightDirect(s, s);
        lightY->ambientColor(SLCol4f(0, 0, 0));
        lightY->diffuseColor(SLCol4f(1, 1, 0));
        lightY->specularColor(SLCol4f(1, 1, 0));
        lightY->translation(-1.5f, -1.5f, 1.5f);
        lightY->lookAt(0, 0, 0);
        lightY->attenuation(1, 0, 0);
        scene->addChild(lightY);

        // Add some meshes to be lighted
        SLNode* sphereL = new SLNode(new SLSpheric(s, 1.0f, 0.0f, 180.0f, 36, 36, "Sphere", mL));
        sphereL->translate(-2, 0, 0);
        sphereL->rotate(90, -1, 0, 0);
        SLNode* sphereM = new SLNode(new SLSpheric(s, 1.0f, 0.0f, 180.0f, 36, 36, "Sphere", mM));
        SLNode* sphereR = new SLNode(new SLSpheric(s, 1.0f, 0.0f, 180.0f, 36, 36, "Sphere", mR));
        sphereR->translate(2, 0, 0);
        sphereR->rotate(90, -1, 0, 0);

        scene->addChild(sphereL);
        scene->addChild(sphereM);
        scene->addChild(sphereR);
        sv->camera(cam1);
        s->root3D(scene);
    }
    else if (SLApplication::sceneID == SID_ShaderCook) //................................................
    {
        s->name("Cook-Torrance Test");
        s->info("Cook-Torrance light model. Left-Right: roughness 0.05-1, Top-Down: metallic: 1-0. The center sphere has roughness and metallic encoded in textures.");

        // Base root group node for the scene
        SLNode* scene = new SLNode;

        SLCamera* cam1 = new SLCamera("Camera 1");
        cam1->translation(0, 0, 28);
        cam1->lookAt(0, 0, 0);
        cam1->background().colors(SLCol4f(0.2f, 0.2f, 0.2f));
        cam1->focalDist(28);
        cam1->setInitialState();
        cam1->devRotLoc(&SLApplication::devRot, &SLApplication::devLoc);
        scene->addChild(cam1);

        // Create spheres and materials with roughness & metallic values between 0 and 1
        const SLint nrRows  = 7;
        const SLint nrCols  = 7;
        SLfloat     spacing = 2.5f;
        SLfloat     maxX    = (float)(nrCols - 1) * spacing * 0.5f;
        SLfloat     maxY    = (float)(nrRows - 1) * spacing * 0.5f;
        SLfloat     deltaR  = 1.0f / (float)(nrRows - 1);
        SLfloat     deltaM  = 1.0f / (float)(nrCols - 1);

        SLGLProgram* sp = new SLGLGenericProgram(s,
                                                 SLApplication::shaderPath + "PerPixCook.vert",
                                                 SLApplication::shaderPath + "PerPixCook.frag");

        SLGLProgram* spTex = new SLGLGenericProgram(s,
                                                    SLApplication::shaderPath + "PerPixCookTex.vert",
                                                    SLApplication::shaderPath + "PerPixCookTex.frag");

        SLMaterial* mat[nrRows * nrCols];
        SLint       i = 0;
        SLfloat     y = -maxY;
        for (SLint m = 0; m < nrRows; ++m)
        {
            SLfloat x = -maxX;
            for (SLint r = 0; r < nrCols; ++r)
            {
                if (m == nrRows / 2 && r == nrCols / 2)
                {
                    // The center sphere has roughness and metallic encoded in textures
                    mat[i] = new SLMaterial(s,
                                            "CookTorranceMatTex",
                                            new SLGLTexture(s, SLApplication::texturePath + "rusty-metal_2048C.jpg"),
                                            new SLGLTexture(s, SLApplication::texturePath + "rusty-metal_2048N.jpg"),
                                            new SLGLTexture(s, SLApplication::texturePath + "rusty-metal_2048M.jpg"),
                                            new SLGLTexture(s, SLApplication::texturePath + "rusty-metal_2048R.jpg"),
                                            spTex);
                }
                else
                {
                    // Cook-Torrance material without textures
                    mat[i] = new SLMaterial(s,
                                            sp,
                                            "CookTorranceMat",
                                            SLCol4f::RED * 0.5f,
                                            Utils::clamp((float)r * deltaR, 0.05f, 1.0f),
                                            (float)m * deltaM);
                }

                SLNode* node = new SLNode(new SLSpheric(s, 1.0f, 0.0f, 180.0f, 32, 32, "Sphere", mat[i]));
                node->translate(x, y, 0);
                scene->addChild(node);
                x += spacing;
                i++;
            }
            y += spacing;
        }

        // Add 5 Lights: 2 point lights, 2 directional lights and 1 spot light in the center.
        SLLightSpot* light1 = new SLLightSpot(s, s, -maxX, maxY, maxY, 0.2f, 180, 0, 1000, 1000);
        light1->attenuation(0, 0, 1);
        SLLightDirect* light2 = new SLLightDirect(s, s, maxX, maxY, maxY, 0.5f, 0, 10, 10);
        light2->lookAt(0, 0, 0);
        light2->attenuation(0, 0, 1);
        SLLightSpot* light3 = new SLLightSpot(s, s, 0, 0, maxY, 0.2f, 36, 0, 1000, 1000);
        light3->attenuation(0, 0, 1);
        SLLightDirect* light4 = new SLLightDirect(s, s, -maxX, -maxY, maxY, 0.5f, 0, 10, 10);
        light4->lookAt(0, 0, 0);
        light4->attenuation(0, 0, 1);
        SLLightSpot* light5 = new SLLightSpot(s, s, maxX, -maxY, maxY, 0.2f, 180, 0, 1000, 1000);
        light5->attenuation(0, 0, 1);
        scene->addChild(light1);
        scene->addChild(light2);
        scene->addChild(light3);
        scene->addChild(light4);
        scene->addChild(light5);
        sv->camera(cam1);
        s->root3D(scene);
    }
    else if (SLApplication::sceneID == SID_ShaderPerVertexWave) //.......................................
    {
        s->name("Wave Shader Test");
        s->info("Vertex Shader with wave displacment.");
        SL_LOG("Use H-Key to increment (decrement w. shift) the wave height.\n");

        SLCamera* cam1 = new SLCamera("Camera 1");
        cam1->translation(0, 3, 8);
        cam1->lookAt(0, 0, 0);
        cam1->focalDist(cam1->translationOS().length());
        cam1->background().colors(SLCol4f(0.1f, 0.4f, 0.8f));
        cam1->setInitialState();
        cam1->devRotLoc(&SLApplication::devRot, &SLApplication::devLoc);

        // Create generic shader program with 4 custom uniforms
        SLGLProgram*   sp  = new SLGLGenericProgram(s, SLApplication::shaderPath + "Wave.vert", SLApplication::shaderPath + "Wave.frag");
        SLGLUniform1f* u_h = new SLGLUniform1f(UT_const, "u_h", 0.1f, 0.05f, 0.0f, 0.5f, (SLKey)'H');
        s->eventHandlers().push_back(u_h);
        sp->addUniform1f(u_h);
        sp->addUniform1f(new SLGLUniform1f(UT_inc, "u_t", 0.0f, 0.06f));
        sp->addUniform1f(new SLGLUniform1f(UT_const, "u_a", 2.5f));
        sp->addUniform1f(new SLGLUniform1f(UT_incDec, "u_b", 2.2f, 0.01f, 2.0f, 2.5f));

        // Create materials
        SLMaterial* matWater = new SLMaterial(s, "matWater", SLCol4f(0.45f, 0.65f, 0.70f), SLCol4f::WHITE, 300);
        matWater->program(sp);
        SLMaterial* matRed = new SLMaterial(s, "matRed", SLCol4f(1.00f, 0.00f, 0.00f));

        // water rectangle in the y=0 plane
        SLNode* wave = new SLNode(new SLRectangle(s, SLVec2f(-Utils::PI, -Utils::PI), SLVec2f(Utils::PI, Utils::PI), 40, 40, "WaterRect", matWater));
        wave->rotate(90, -1, 0, 0);

        SLLightSpot* light0 = new SLLightSpot(s, s);
        light0->ambiDiffPowers(0, 1);
        light0->translate(0, 4, -4, TS_object);
        light0->attenuation(1, 0, 0);

        SLNode* scene = new SLNode;
        scene->addChild(light0);
        scene->addChild(wave);
        scene->addChild(new SLNode(new SLSphere(s, 1, 32, 32, "Red Sphere", matRed)));
        scene->addChild(cam1);

        sv->camera(cam1);
        s->root3D(scene);
        sv->doWaitOnIdle(false);
    }
    else if (SLApplication::sceneID == SID_ShaderWater) //...............................................
    {
        s->name("Water Shader Test");
        s->info("Water Shader with reflection & refraction mapping.");
        SL_LOG("Use H-Key to increment (decrement w. shift) the wave height.\n");

        SLCamera* cam1 = new SLCamera("Camera 1");
        cam1->translation(0, 3, 8);
        cam1->lookAt(0, 0, 0);
        cam1->focalDist(cam1->translationOS().length());
        cam1->background().colors(SLCol4f(0.1f, 0.4f, 0.8f));
        cam1->setInitialState();
        cam1->devRotLoc(&SLApplication::devRot, &SLApplication::devLoc);

        // create texture
        SLGLTexture* tex1 = new SLGLTexture(s,
                                            SLApplication::texturePath + "Pool+X0512_C.png",
                                            SLApplication::texturePath + "Pool-X0512_C.png",
                                            SLApplication::texturePath + "Pool+Y0512_C.png",
                                            SLApplication::texturePath + "Pool-Y0512_C.png",
                                            SLApplication::texturePath + "Pool+Z0512_C.png",
                                            SLApplication::texturePath + "Pool-Z0512_C.png");
        SLGLTexture* tex2 = new SLGLTexture(s, SLApplication::texturePath + "tile1_0256_C.jpg");

        // Create generic shader program with 4 custom uniforms
        SLGLProgram*   sp  = new SLGLGenericProgram(s, SLApplication::shaderPath + "WaveRefractReflect.vert", SLApplication::shaderPath + "RefractReflect.frag");
        SLGLUniform1f* u_h = new SLGLUniform1f(UT_const, "u_h", 0.1f, 0.05f, 0.0f, 0.5f, (SLKey)'H');
        s->eventHandlers().push_back(u_h);
        sp->addUniform1f(u_h);
        sp->addUniform1f(new SLGLUniform1f(UT_inc, "u_t", 0.0f, 0.06f));
        sp->addUniform1f(new SLGLUniform1f(UT_const, "u_a", 2.5f));
        sp->addUniform1f(new SLGLUniform1f(UT_incDec, "u_b", 2.2f, 0.01f, 2.0f, 2.5f));

        // Create materials
        SLMaterial* matWater = new SLMaterial(s, "matWater", SLCol4f(0.45f, 0.65f, 0.70f), SLCol4f::WHITE, 100, 0.1f, 0.9f, 1.5f);
        matWater->program(sp);
        matWater->textures().push_back(tex1);
        SLMaterial* matRed  = new SLMaterial(s, "matRed", SLCol4f(1.00f, 0.00f, 0.00f));
        SLMaterial* matTile = new SLMaterial(s, "matTile");
        matTile->textures().push_back(tex2);

        // water rectangle in the y=0 plane
        SLNode* rect = new SLNode(new SLRectangle(s,
                                                  SLVec2f(-Utils::PI, -Utils::PI),
                                                  SLVec2f(Utils::PI, Utils::PI),
                                                  40,
                                                  40,
                                                  "WaterRect",
                                                  matWater));
        rect->rotate(90, -1, 0, 0);

        // Pool rectangles
        SLuint  res   = 10;
        SLNode* rectF = new SLNode(new SLRectangle(s, SLVec2f(-Utils::PI, -Utils::PI / 6), SLVec2f(Utils::PI, Utils::PI / 6), SLVec2f(0, 0), SLVec2f(10, 2.5f), res, res, "rectF", matTile));
        SLNode* rectN = new SLNode(new SLRectangle(s, SLVec2f(-Utils::PI, -Utils::PI / 6), SLVec2f(Utils::PI, Utils::PI / 6), SLVec2f(0, 0), SLVec2f(10, 2.5f), res, res, "rectN", matTile));
        SLNode* rectL = new SLNode(new SLRectangle(s, SLVec2f(-Utils::PI, -Utils::PI / 6), SLVec2f(Utils::PI, Utils::PI / 6), SLVec2f(0, 0), SLVec2f(10, 2.5f), res, res, "rectL", matTile));
        SLNode* rectR = new SLNode(new SLRectangle(s, SLVec2f(-Utils::PI, -Utils::PI / 6), SLVec2f(Utils::PI, Utils::PI / 6), SLVec2f(0, 0), SLVec2f(10, 2.5f), res, res, "rectR", matTile));
        SLNode* rectB = new SLNode(new SLRectangle(s, SLVec2f(-Utils::PI, -Utils::PI), SLVec2f(Utils::PI, Utils::PI), SLVec2f(0, 0), SLVec2f(10, 10), res, res, "rectB", matTile));
        rectF->translate(0, 0, -Utils::PI, TS_object);
        rectL->rotate(90, 0, 1, 0);
        rectL->translate(0, 0, -Utils::PI, TS_object);
        rectN->rotate(180, 0, 1, 0);
        rectN->translate(0, 0, -Utils::PI, TS_object);
        rectR->rotate(270, 0, 1, 0);
        rectR->translate(0, 0, -Utils::PI, TS_object);
        rectB->rotate(90, -1, 0, 0);
        rectB->translate(0, 0, -Utils::PI / 6, TS_object);

        SLLightSpot* light0 = new SLLightSpot(s, s);
        light0->ambiDiffPowers(0, 1);
        light0->translate(0, 4, -4, TS_object);
        light0->attenuation(1, 0, 0);

        SLNode* scene = new SLNode;
        scene->addChild(light0);
        scene->addChild(rectF);
        scene->addChild(rectL);
        scene->addChild(rectN);
        scene->addChild(rectR);
        scene->addChild(rectB);
        scene->addChild(rect);
        scene->addChild(new SLNode(new SLSphere(s,
                                                1,
                                                32,
                                                32,
                                                "Red Sphere",
                                                matRed)));
        scene->addChild(cam1);

        sv->camera(cam1);
        s->root3D(scene);
        sv->doWaitOnIdle(false);
    }
    else if (SLApplication::sceneID == SID_ShaderBumpNormal) //..........................................
    {
        s->name("Normal Map Test");
        s->info("Normal map bump mapping combined with a spot and a directional lighting.");

        // Create textures
        SLGLTexture* texC = new SLGLTexture(s, SLApplication::texturePath + "brickwall0512_C.jpg");
        SLGLTexture* texN = new SLGLTexture(s, SLApplication::texturePath + "brickwall0512_N.jpg");

        SLGLProgram* sp = new SLGLGenericProgram(s,
                                                 SLApplication::shaderPath + "PerPixBlinnTexNrm.vert",
                                                 SLApplication::shaderPath + "PerPixBlinnTexNrm.frag");

        // Create materials
        SLMaterial* m1 = new SLMaterial(s,
                                        "m1",
                                        texC,
                                        texN,
                                        nullptr,
                                        nullptr,
                                        sp);

        SLCamera* cam1 = new SLCamera("Camera 1");
        cam1->translation(-10, 10, 10);
        cam1->lookAt(0, 0, 0);
        cam1->focalDist(20);
        cam1->background().colors(SLCol4f(0.5f, 0.5f, 0.5f));
        cam1->setInitialState();

        SLLightSpot* light1 = new SLLightSpot(s, s, 0.3f, 40, true);
        light1->powers(0.1f, 1.0f, 1.0f);
        light1->attenuation(1, 0, 0);
        light1->translation(0, 0, 5);
        light1->lookAt(0, 0, 0);

        SLLightDirect* light2 = new SLLightDirect(s, s);
        light2->ambientColor(SLCol4f(0, 0, 0));
        light2->diffuseColor(SLCol4f(1, 1, 0));
        light2->specularColor(SLCol4f(1, 1, 0));
        light2->translation(-5, -5, 5);
        light2->lookAt(0, 0, 0);
        light2->attenuation(1, 0, 0);

        SLAnimation* anim = s->animManager().createNodeAnimation("light1_anim", 2.0f);
        anim->createEllipticNodeTrack(light1, 2.0f, A_x, 2.0f, A_Y);

        SLNode* scene = new SLNode;
        scene->addChild(light1);
        scene->addChild(light2);
        scene->addChild(new SLNode(new SLRectangle(s, SLVec2f(-5, -5), SLVec2f(5, 5), 1, 1, "Rect", m1)));
        scene->addChild(cam1);

        sv->camera(cam1);
        s->root3D(scene);
    }
    else if (SLApplication::sceneID == SID_ShaderBumpParallax) //........................................
    {
        s->name("Parallax Map Test");
        s->info("Normal map parallax mapping with a spot and a directional light");
        SL_LOG("Demo application for parallax bump mapping.");
        SL_LOG("Use X-Key to increment (decrement w. shift) parallax scale.");
        SL_LOG("Use O-Key to increment (decrement w. shift) parallax offset.\n");

        // Create shader program with 4 uniforms
        SLGLProgram*   sp     = new SLGLGenericProgram(s, SLApplication::shaderPath + "PerPixBlinnTexNrm.vert", SLApplication::shaderPath + "PerPixBlinnTexNrmParallax.frag");
        SLGLUniform1f* scale  = new SLGLUniform1f(UT_const, "u_scale", 0.04f, 0.002f, 0, 1, (SLKey)'X');
        SLGLUniform1f* offset = new SLGLUniform1f(UT_const, "u_offset", -0.03f, 0.002f, -1, 1, (SLKey)'O');
        s->eventHandlers().push_back(scale);
        s->eventHandlers().push_back(offset);
        sp->addUniform1f(scale);
        sp->addUniform1f(offset);

        // Create textures
        SLGLTexture* texC = new SLGLTexture(s, SLApplication::texturePath + "brickwall0512_C.jpg");
        SLGLTexture* texN = new SLGLTexture(s, SLApplication::texturePath + "brickwall0512_N.jpg");
        SLGLTexture* texH = new SLGLTexture(s, SLApplication::texturePath + "brickwall0512_H.jpg");

        // Create materials
        SLMaterial* m1 = new SLMaterial(s, "mat1", texC, texN, texH, nullptr, sp);

        SLCamera* cam1 = new SLCamera("Camera 1");
        cam1->translation(-10, 10, 10);
        cam1->lookAt(0, 0, 0);
        cam1->focalDist(20);
        cam1->background().colors(SLCol4f(0.5f, 0.5f, 0.5f));
        cam1->setInitialState();

        SLLightSpot* light1 = new SLLightSpot(s, s, 0.3f, 40, true);
        light1->powers(0.1f, 1.0f, 1.0f);
        light1->attenuation(1, 0, 0);
        light1->translation(0, 0, 5);
        light1->lookAt(0, 0, 0);

        SLLightDirect* light2 = new SLLightDirect(s, s);
        light2->ambientColor(SLCol4f(0, 0, 0));
        light2->diffuseColor(SLCol4f(1, 1, 0));
        light2->specularColor(SLCol4f(1, 1, 0));
        light2->translation(-5, -5, 5);
        light2->lookAt(0, 0, 0);
        light2->attenuation(1, 0, 0);

        SLAnimation* anim = s->animManager().createNodeAnimation("light1_anim", 2.0f);
        anim->createEllipticNodeTrack(light1, 2.0f, A_x, 2.0f, A_Y);

        SLNode* scene = new SLNode;
        scene->addChild(light1);
        scene->addChild(light2);
        scene->addChild(new SLNode(new SLRectangle(s, SLVec2f(-5, -5), SLVec2f(5, 5), 1, 1, "Rect", m1)));
        scene->addChild(cam1);

        sv->camera(cam1);
        s->root3D(scene);
    }
    else if (SLApplication::sceneID == SID_ShaderSkyBox) //..............................................
    {
        // Set scene name and info string
        s->name("Sky Box Test");
        s->info("Sky box cube with cubemap skybox shader");

        // Create textures and materials
        SLSkybox*    skybox    = new SLSkybox(s,
                                        SLApplication::shaderPath,
                                        SLApplication::texturePath + "Desert+X1024_C.jpg",
                                        SLApplication::texturePath + "Desert-X1024_C.jpg",
                                        SLApplication::texturePath + "Desert+Y1024_C.jpg",
                                        SLApplication::texturePath + "Desert-Y1024_C.jpg",
                                        SLApplication::texturePath + "Desert+Z1024_C.jpg",
                                        SLApplication::texturePath + "Desert-Z1024_C.jpg");
        SLGLTexture* skyboxTex = skybox->mesh()->mat()->textures()[0];

        // Material for mirror
        SLMaterial* refl = new SLMaterial(s, "refl", SLCol4f::BLACK, SLCol4f::WHITE, 1000, 1.0f);
        refl->textures().push_back(skyboxTex);
        refl->program(new SLGLGenericProgram(s, SLApplication::shaderPath + "Reflect.vert", SLApplication::shaderPath + "Reflect.frag"));

        // Material for glass
        SLMaterial* refr = new SLMaterial(s, "refr", SLCol4f::BLACK, SLCol4f::BLACK, 100, 0.1f, 0.9f, 1.5f);
        refr->translucency(1000);
        refr->transmissive(SLCol4f::WHITE);
        refr->textures().push_back(skyboxTex);
        refr->program(new SLGLGenericProgram(s, SLApplication::shaderPath + "RefractReflect.vert", SLApplication::shaderPath + "RefractReflect.frag"));

        // Create a scene group node
        SLNode* scene = new SLNode("scene node");

        // Create camera in the center
        SLCamera* cam1 = new SLCamera("Camera 1");
        cam1->translation(0, 0, 5);
        cam1->setInitialState();
        scene->addChild(cam1);

        // There is no light needed in this scene. All reflections come from cube maps
        // But ray tracing needs light sources
        // Create directional light for the sun light
        SLLightDirect* light = new SLLightDirect(s, s, 0.5f);
        light->ambientColor(SLCol4f(0.3f, 0.3f, 0.3f));
        light->attenuation(1, 0, 0);
        light->translate(1, 1, -1);
        light->lookAt(-1, -1, 1);
        scene->addChild(light);

        // Center sphere
        SLNode* sphere = new SLNode(new SLSphere(s, 0.5f, 32, 32, "Sphere", refr));
        scene->addChild(sphere);

        // load teapot
        SLAssimpImporter importer;
        SLNode*          teapot = importer.load(s->animManager(), s, SLApplication::modelPath + "FBX/Teapot/Teapot.fbx", SLApplication::texturePath, true, refl);
        teapot->translate(-1.5f, -0.5f, 0);
        scene->addChild(teapot);

        // load Suzanne
        SLNode* suzanne = importer.load(s->animManager(), s, SLApplication::modelPath + "FBX/Suzanne/Suzanne.fbx", SLApplication::texturePath, true, refr);
        suzanne->translate(1.5f, -0.5f, 0);
        scene->addChild(suzanne);

        sv->camera(cam1);
        sv->skybox(skybox);

        // pass the scene group as root node
        s->root3D(scene);

        // Save energy
        sv->doWaitOnIdle(true);
    }
    else if (SLApplication::sceneID == SID_ShaderEarth) //...............................................
    {
        s->name("Earth Shader Test");
        s->info("Complex earth shader with 7 textures: day color, night color, normal, height & gloss map of earth, color & alphamap of clouds");
        SL_LOG("Earth Shader from Markus Knecht");
        SL_LOG("Use (SHIFT) & key X to change scale of the parallax mapping");
        SL_LOG("Use (SHIFT) & key O to change offset of the parallax mapping");

        // Create shader program with 4 uniforms
        SLGLProgram*   sp     = new SLGLGenericProgram(s, SLApplication::shaderPath + "PerPixBlinnTexNrm.vert", SLApplication::shaderPath + "PerPixBlinnTexNrmEarth.frag");
        SLGLUniform1f* scale  = new SLGLUniform1f(UT_const, "u_scale", 0.02f, 0.002f, 0, 1, (SLKey)'X');
        SLGLUniform1f* offset = new SLGLUniform1f(UT_const, "u_offset", -0.02f, 0.002f, -1, 1, (SLKey)'O');
        s->eventHandlers().push_back(scale);
        s->eventHandlers().push_back(offset);
        sp->addUniform1f(scale);
        sp->addUniform1f(offset);

        // Create textures
        SLGLTexture* texC   = new SLGLTexture(s, SLApplication::texturePath + "earth2048_C.jpg");      // color map
        SLGLTexture* texN   = new SLGLTexture(s, SLApplication::texturePath + "earth2048_N.jpg");      // normal map
        SLGLTexture* texH   = new SLGLTexture(s, SLApplication::texturePath + "earth2048_H.jpg");      // height map
        SLGLTexture* texG   = new SLGLTexture(s, SLApplication::texturePath + "earth2048_G.jpg");      // gloss map
        SLGLTexture* texNC  = new SLGLTexture(s, SLApplication::texturePath + "earthNight2048_C.jpg"); // night color  map
        SLGLTexture* texClC = new SLGLTexture(s, SLApplication::texturePath + "earthCloud1024_C.jpg"); // cloud color map
        SLGLTexture* texClA = new SLGLTexture(s, SLApplication::texturePath + "earthCloud1024_A.jpg"); // cloud alpha map

        // Create materials
        SLMaterial* matEarth = new SLMaterial(s, "matEarth", texC, texN, texH, texG, sp);
        matEarth->textures().push_back(texClC);
        matEarth->textures().push_back(texClA);
        matEarth->textures().push_back(texNC);
        matEarth->shininess(4000);
        matEarth->program(sp);

        SLCamera* cam1 = new SLCamera("Camera 1");
        cam1->translation(0, 0, 4);
        cam1->lookAt(0, 0, 0);
        cam1->focalDist(4);
        cam1->background().colors(SLCol4f(0, 0, 0));
        cam1->setInitialState();
        cam1->devRotLoc(&SLApplication::devRot, &SLApplication::devLoc);

        SLLightSpot* sun = new SLLightSpot(s, s);
        sun->powers(0.0f, 1.0f, 0.2f);
        sun->attenuation(1, 0, 0);

        SLAnimation* anim = s->animManager().createNodeAnimation("light1_anim", 24.0f);
        anim->createEllipticNodeTrack(sun, 50.0f, A_x, 50.0f, A_z);

        SLuint  res   = 30;
        SLNode* earth = new SLNode(new SLSphere(s, 1, res, res, "Earth", matEarth));
        earth->rotate(90, -1, 0, 0);

        SLNode* scene = new SLNode;
        scene->addChild(sun);
        scene->addChild(earth);
        scene->addChild(cam1);

        sv->camera(cam1);
        s->root3D(scene);
    }
    else if (SLApplication::sceneID == SID_ShaderVoxelConeDemo) //.......................................
    {
        s->name("Voxelization Test");
        s->info("Voxelizing a Scnene and Display result");

        // Base root group node for the scene
        SLNode* scene = new SLNode;

        SLCamera* cam1 = new SLCamera("Camera 1");
        cam1->translation(0, 0, 1.8f);
        cam1->lookAt(0, 0, 0);
        cam1->background().colors(SLCol4f(0.2f, 0.2f, 0.2f));
        cam1->fov(75.0f);
        cam1->focalDist(1.8f);
        cam1->devRotLoc(&SLApplication::devRot, &SLApplication::devLoc);

        scene->addChild(cam1);

        SLCol4f grayRGB(0.75f, 0.75f, 0.75f);
        SLCol4f redRGB(0.75f, 0.25f, 0.25f);
        SLCol4f yellowRGB(1.0f, 1.0f, 0.0);
        SLCol4f blueRGB(0.25f, 0.25f, 0.75f);
        SLCol4f blackRGB(0.00f, 0.00f, 0.00f);

        SLGLProgram* sp = new SLGLGenericProgram(s,
                                                 SLApplication::shaderPath + "PerPixBlinn.vert",
                                                 SLApplication::shaderPath + "PerPixBlinn.frag");

        SLMaterial* cream     = new SLMaterial(s, "cream", grayRGB, SLCol4f::BLACK, 100.f, 0.f, 0.f, 1.f, sp);
        SLMaterial* teapotMat = new SLMaterial(s, "teapot", grayRGB, SLCol4f::WHITE, 100.f, 0.f, 0.f, 1.f, sp);

        SLAssimpImporter importer;
        SLNode*          teapot = importer.load(s->animManager(),
                                       s,
                                       SLApplication::modelPath + "FBX/Teapot/Teapot.fbx",
                                       SLApplication::texturePath,
                                       true,
                                       teapotMat);

        teapot->scale(0.5);
        teapot->translate(-0.6f, -0.2f, -0.4f, TS_world);
        scene->addChild(teapot);

        SLMaterial* red    = new SLMaterial(s, "red", redRGB, SLCol4f::BLACK, 100.f, 0.f, 0.f, 1.f, sp);
        SLMaterial* yellow = new SLMaterial(s, "yellow", yellowRGB, SLCol4f::BLACK, 100.f, 0.f, 0.f, 1.f, sp);
        SLMaterial* refl   = new SLMaterial(s, "refl", SLCol4f::BLACK, SLCol4f::WHITE, 1000, 1.0f);

        SLNode* sphere = new SLNode(new SLSphere(s, 0.3f, 32, 32, "Sphere1", refl));
        scene->addChild(sphere);

        SLNode* box = new SLNode(new SLBox(s, 0, 0, 0, 0.6f, 0.8f, 0.8f, "Box", yellow));
        box->translation(SLVec3f(-0.9f, -1, -0.7f));
        scene->addChild(box);

        // animate teapot
        SLAnimation*     light2Anim = s->animManager().createNodeAnimation("sphere_anim",
                                                                       5.0f,
                                                                       true,
                                                                       EC_linear,
                                                                       AL_loop);
        SLNodeAnimTrack* track      = light2Anim->createNodeAnimationTrack();
        track->animatedNode(sphere);
        SLTransformKeyframe* k1 = track->createNodeKeyframe(0.0f);
        k1->translation(SLVec3f(0.3f, 0.2f, -0.3f));
        SLTransformKeyframe* k2 = track->createNodeKeyframe(2.5f);
        k2->translation(SLVec3f(0.3f, -0.65f, -0.3f));
        SLTransformKeyframe* k3 = track->createNodeKeyframe(5.0f);
        k3->translation(SLVec3f(0.3f, 0.2f, -0.3f));

        SLMaterial* pink = new SLMaterial(s, "cream", SLCol4f(1, 0.35f, 0.65f), SLCol4f::BLACK, 100.f, 0.f, 0.f, 1.f, sp);

        // create wall polygons
        SLfloat pL = -0.99f, pR = 0.99f; // left/right
        SLfloat pB = -0.99f, pT = 0.99f; // bottom/top
        SLfloat pN = 0.99f, pF = -0.99f; // near/far

        SLMaterial* blue = new SLMaterial(s, "blue", blueRGB, SLCol4f::BLACK, 100.f, 0.f, 0.f, 1.f, sp);

        // bottom plane
        SLNode* b = new SLNode(new SLRectangle(s, SLVec2f(pL, -pN), SLVec2f(pR, -pF), 6, 6, "bottom", cream));
        b->rotate(90, -1, 0, 0);
        b->translate(0, 0, pB, TS_object);
        scene->addChild(b);

        // top plane
        SLNode* t = new SLNode(new SLRectangle(s, SLVec2f(pL, pF), SLVec2f(pR, pN), 6, 6, "top", cream));
        t->rotate(90, 1, 0, 0);
        t->translate(0, 0, -pT, TS_object);
        scene->addChild(t);

        // far plane
        SLNode* f = new SLNode(new SLRectangle(s, SLVec2f(pL, pB), SLVec2f(pR, pT), 6, 6, "far", cream));
        f->translate(0, 0, pF, TS_object);
        scene->addChild(f);

        // left plane
        SLNode* l = new SLNode(new SLRectangle(s, SLVec2f(-pN, pB), SLVec2f(-pF, pT), 6, 6, "left", red));
        l->rotate(90, 0, 1, 0);
        l->translate(0, 0, pL, TS_object);
        scene->addChild(l);

        // right plane
        SLNode* r = new SLNode(new SLRectangle(s, SLVec2f(pF, pB), SLVec2f(pN, pT), 6, 6, "right", blue));
        r->rotate(90, 0, -1, 0);
        r->translate(0, 0, -pR, TS_object);
        scene->addChild(r);

        // Rectangular light
        SLLightRect* light0 = new SLLightRect(s, s, 0.9f, 0.6f, true);
        //SLLightRect *light0 = new SLLightRect(0.9, 0.6f, true);
        light0->rotate(90, -1.0f, 0.0f, 0.0f);
        light0->translate(0.0f, 0.f, 0.95f, TS_object);
        //light0->init();
        light0->spotCutOffDEG(170);
        light0->spotExponent(1.0);
        light0->powers(0.3f, 2.0f, 1.0f);
        light0->attenuation(0, 0, 1);
        scene->addChild(light0);

        sv->camera(cam1);
        s->root3D(scene);
    }
    else if (SLApplication::sceneID == SID_ShadowMappingBasicScene) //...................................
    {
        s->name("Shadow Mapping Basic Scene");
        s->info("Shadow Mapping is a technique to render shadows.");

        // Setup shadow mapping material
        SLGLProgram* progPerPixSM = new SLGLGenericProgram(s,
                                                           SLApplication::shaderPath + "PerPixBlinnSM.vert",
                                                           SLApplication::shaderPath + "PerPixBlinnSM.frag");
        SLMaterial*  matPerPixSM  = new SLMaterial(s, "m1", SLCol4f::WHITE, SLCol4f::WHITE, 500, 0, 0, 1, progPerPixSM);

        // Base root group node for the scene
        SLNode* scene = new SLNode;

        SLCamera* cam1 = new SLCamera("Camera 1");
        cam1->translation(0, 7, 12);
        cam1->lookAt(0, 1, 0);
        cam1->focalDist(8);
        cam1->background().colors(SLCol4f(0.1f, 0.1f, 0.1f));
        cam1->setInitialState();
        scene->addChild(cam1);

        // Create light source
        // Do constant attenuation for directional lights since it is infinitely far away
        SLLightDirect* light = new SLLightDirect(s, s);
        light->powers(0.0f, 1.0f, 1.0f);
        light->translation(0, 5, 0);
        light->lookAt(0, 0, 0);
        light->attenuation(1, 0, 0);
        light->createsShadows(true);
        light->createShadowMap();
        light->shadowMap()->rayCount(SLVec2i(16, 16));
        light->castsShadows(false);
        scene->addChild(light);

        // Add a sphere which casts shadows
        SLNode* sphereNode = new SLNode(new SLSpheric(s, 1, 0, 180, 20, 20, "Sphere", matPerPixSM));
        sphereNode->translate(0, 2.0, 0);
        sphereNode->castsShadows(true);
        scene->addChild(sphereNode);

        SLAnimation* anim = s->animManager().createNodeAnimation("sphere_anim", 2.0f);
        anim->createEllipticNodeTrack(sphereNode, 0.5f, A_x, 0.5f, A_z);

        // Add a box which receives shadows
        SLNode* boxNode = new SLNode(new SLBox(s, -5, -1, -5, 5, 0, 5, "Box", matPerPixSM));
        boxNode->castsShadows(false);
        scene->addChild(boxNode);

        sv->camera(cam1);
        s->root3D(scene);
    }
    else if (SLApplication::sceneID == SID_ShadowMappingLightTypes) //...................................
    {
        s->name("Shadow Mapping light types");
        s->info("Shadow Mapping is implemented for these light types.");

        // Setup shadow mapping material
        SLGLProgram* progPerPixSM = new SLGLGenericProgram(s,
                                                           SLApplication::shaderPath + "PerPixBlinnSM.vert",
                                                           SLApplication::shaderPath + "PerPixBlinnSM8CM.frag");
        SLMaterial*  matPerPixSM  = new SLMaterial(s, "m1", SLCol4f::WHITE, SLCol4f::WHITE, 500, 0, 0, 1, progPerPixSM);

        // Base root group node for the scene
        SLNode* scene = new SLNode;

        SLCamera* cam1 = new SLCamera("Camera 1");
        cam1->translation(0, 2, 15);
        cam1->lookAt(0, 2, 0);
        cam1->focalDist(8);
        cam1->background().colors(SLCol4f(0.1f, 0.1f, 0.1f));
        cam1->setInitialState();
        scene->addChild(cam1);

        // Create light sources
        std::vector<SLLight*> lights = {
          new SLLightDirect(s, s),
          new SLLightRect(s, s),
          new SLLightSpot(s, s, 0.3f, 25.0f),
          new SLLightSpot(s, s, 0.1f, 180.0f)};

        for (SLint i = 0; i < lights.size(); ++i)
        {
            SLLight* light = lights[i];
            SLNode*  node  = dynamic_cast<SLNode*>(light);
            SLfloat  x     = (i - (lights.size() - 1.0f) / 2.0f) * 5;

            if (i == 0) // Make direct light less bright
            {
                light->powers(0.0f, 0.4f, 0.4f);
                light->attenuation(1, 0, 0);
            }
            else
            {
                light->powers(0.0f, 2.0f, 2.0f);
                light->attenuation(0, 0, 1);
            }

            node->translation(x, 5, 0);
            node->lookAt(x, 0, 0);
            light->createsShadows(true);
            light->createShadowMap();
            light->shadowMap()->rayCount(SLVec2i(16, 16));
            scene->addChild(node);
        }

        // Add teapots which cast shadows
        SLAssimpImporter importer;
        SLAnimation*     teapotAnim  = s->animManager().createNodeAnimation("teapot_anim", 8.0f, true, EC_linear, AL_loop);
        SLNode*          teapotModel = importer.load(s->animManager(), s, SLApplication::modelPath + "FBX/Teapot/Teapot.fbx", SLApplication::texturePath, true, matPerPixSM);

        for (SLLight* light : lights)
        {
            SLNode* teapot = teapotModel->copyRec();

            teapot->translate(light->positionWS().x, 2, 0);
            teapot->children()[0]->castsShadows(true);
            scene->addChild(teapot);

            // Create animation
            SLNodeAnimTrack* track = teapotAnim->createNodeAnimationTrack();
            track->animatedNode(teapot);

            SLTransformKeyframe* frame0 = track->createNodeKeyframe(0.0f);
            frame0->translation(teapot->translationWS());
            frame0->rotation(SLQuat4f(0, 0, 0));

            SLTransformKeyframe* frame1 = track->createNodeKeyframe(4.0f);
            frame1->translation(teapot->translationWS());
            frame1->rotation(SLQuat4f(0, 1 * PI, 0));

            SLTransformKeyframe* frame2 = track->createNodeKeyframe(8.0f);
            frame2->translation(teapot->translationWS());
            frame2->rotation(SLQuat4f(0, 2 * PI, 0));
        }

        delete teapotModel;

        // Add a box which receives shadows
        SLfloat minx    = lights.front()->positionWS().x - 3;
        SLfloat maxx    = lights.back()->positionWS().x + 3;
        SLNode* boxNode = new SLNode(new SLBox(s, minx, -1, -5, maxx, 0, 5, "Box", matPerPixSM));
        boxNode->castsShadows(false);
        scene->addChild(boxNode);

        sv->camera(cam1);
        s->root3D(scene);
    }
    else if (SLApplication::sceneID == SID_ShadowMappingSpotLights) //...................................
    {
        s->name("Shadow Mapping for Spot lights");
        s->info("8 Spot lights use a perspective projection for their light space.");

        // Setup shadow mapping material
        SLGLProgram* progPerPixSM = new SLGLGenericProgram(s,
                                                           SLApplication::shaderPath + "PerPixBlinnSM.vert",
                                                           SLApplication::shaderPath + "PerPixBlinnSM8CM.frag");
        SLMaterial*  matPerPixSM  = new SLMaterial(s, "m1", SLCol4f::WHITE, SLCol4f::WHITE, 500, 0, 0, 1, progPerPixSM);

        // Base root group node for the scene
        SLNode* scene = new SLNode;

        SLCamera* cam1 = new SLCamera("Camera 1");
        cam1->translation(0, 5, 13);
        cam1->lookAt(0, 1, 0);
        cam1->focalDist(8);
        cam1->background().colors(SLCol4f(0.1f, 0.1f, 0.1f));
        cam1->setInitialState();
        scene->addChild(cam1);

        // Create light sources
        for (int i = 0; i < SL_MAX_LIGHTS; ++i)
        {
            SLLightSpot* light = new SLLightSpot(s, s, 0.3f, 45.0f);
            SLCol4f      color;
            color.hsva2rgba(SLVec3f(Utils::TWOPI * (float)i / (float)SL_MAX_LIGHTS, 1.0f, 1.0f));
            light->powers(0.0f, 5.0f, 5.0f, color);
            light->translation(2 * sin((Utils::TWOPI / (float)SL_MAX_LIGHTS) * (float)i),
                               5,
                               2 * cos((Utils::TWOPI / (float)SL_MAX_LIGHTS) * (float)i));
            light->lookAt(0, 0, 0);
            light->attenuation(0, 0, 1);
            light->createsShadows(true);
            light->createShadowMap();
            light->shadowMap()->rayCount(SLVec2i(16, 16));
            scene->addChild(light);
        }

        // Add a sphere which casts shadows
        SLNode* sphereNode = new SLNode(new SLSpheric(s, 1, 0, 180, 20, 20, "Sphere", matPerPixSM));
        sphereNode->translate(0, 2.0, 0);
        sphereNode->castsShadows(true);
        scene->addChild(sphereNode);

        SLAnimation* anim = s->animManager().createNodeAnimation("sphere_anim", 2.0f);
        anim->createEllipticNodeTrack(sphereNode, 1.0f, A_x, 1.0f, A_z);

        // Add a box which receives shadows
        SLNode* boxNode = new SLNode(new SLBox(s, -5, -1, -5, 5, 0, 5, "Box", matPerPixSM));
        boxNode->castsShadows(false);
        scene->addChild(boxNode);

        sv->camera(cam1);
        s->root3D(scene);
    }
    else if (SLApplication::sceneID == SID_ShadowMappingPointLights) //..................................
    {
        s->name("Shadow Mapping for point lights");
        s->info("Point lights use cubemaps to store shadow maps.");

        // Setup shadow mapping material
        SLGLProgram* progPerPixSM = new SLGLGenericProgram(s,
                                                           SLApplication::shaderPath + "PerPixBlinnSM.vert",
                                                           SLApplication::shaderPath + "PerPixBlinnSM8CM.frag");
        SLMaterial*  matPerPixSM  = new SLMaterial(s, "m1", SLCol4f::WHITE, SLCol4f::WHITE, 500, 0, 0, 1, progPerPixSM);

        // Base root group node for the scene
        SLNode* scene = new SLNode;

        // Create camera
        SLCamera* cam1 = new SLCamera;
        cam1->translation(0, 0, 8);
        cam1->lookAt(0, 0, 0);
        cam1->fov(27);
        cam1->focalDist(cam1->translationOS().length());
        cam1->background().colors(SLCol4f(0.1f, 0.1f, 0.1f));
        cam1->setInitialState();
        cam1->devRotLoc(&SLApplication::devRot, &SLApplication::devLoc);

        // Create lights
        SLAnimation* anim = s->animManager().createNodeAnimation("light_anim", 4.0f);

        for (SLint i = 0; i < 3; ++i)
        {
            SLLightSpot* light = new SLLightSpot(s, s, 0.1f);
            light->powers(0.2f, 1.5f, 1.0f, SLCol4f(i == 0, i == 1, i == 2));
            light->attenuation(0, 0, 1);
            light->translate(i - 1.0f, i - 1.0f, i - 1.0f);
            light->createsShadows(true);
            light->createShadowMap();
            light->shadowMap()->rayCount(SLVec2i(16, 16));
            scene->addChild(light);
            anim->createEllipticNodeTrack(light, 0.2f, A_x, 0.2f, A_z);
        }

        // Create wall polygons
        SLfloat pL = -1.48f, pR = 1.48f; // left/right
        SLfloat pB = -1.25f, pT = 1.19f; // bottom/top
        SLfloat pN = 1.79f, pF = -1.55f; // near/far

        // Bottom plane
        SLNode* b = new SLNode(new SLRectangle(s, SLVec2f(pL, -pN), SLVec2f(pR, -pF), 6, 6, "bottom", matPerPixSM));
        b->rotate(90, -1, 0, 0);
        b->translate(0, 0, pB, TS_object);
        scene->addChild(b);

        // Top plane
        SLNode* t = new SLNode(new SLRectangle(s, SLVec2f(pL, pF), SLVec2f(pR, pN), 6, 6, "top", matPerPixSM));
        t->rotate(90, 1, 0, 0);
        t->translate(0, 0, -pT, TS_object);
        scene->addChild(t);

        // Far plane
        SLNode* f = new SLNode(new SLRectangle(s, SLVec2f(pL, pB), SLVec2f(pR, pT), 6, 6, "far", matPerPixSM));
        f->translate(0, 0, pF, TS_object);
        scene->addChild(f);

        // near plane
        SLNode* n = new SLNode(new SLRectangle(s, SLVec2f(pL, pT), SLVec2f(pR, pB), 6, 6, "near", matPerPixSM));
        n->translate(0, 0, pN, TS_object);
        scene->addChild(n);

        // left plane
        SLNode* l = new SLNode(new SLRectangle(s, SLVec2f(-pN, pB), SLVec2f(-pF, pT), 6, 6, "left", matPerPixSM));
        l->rotate(90, 0, 1, 0);
        l->translate(0, 0, pL, TS_object);
        scene->addChild(l);

        // Right plane
        SLNode* r = new SLNode(new SLRectangle(s, SLVec2f(pF, pB), SLVec2f(pN, pT), 6, 6, "right", matPerPixSM));
        r->rotate(90, 0, -1, 0);
        r->translate(0, 0, -pR, TS_object);
        scene->addChild(r);

        // Create cubes which cast shadows
        for (SLint i = 0; i < 64; ++i)
        {
            SLBox* box = new SLBox(s);
            box->mat(matPerPixSM);
            SLNode* boxNode = new SLNode(box);

            boxNode->scale(Utils::random(0.01f, 0.1f));
            boxNode->translate(Utils::random(pL + 0.3f, pR - 0.3f),
                               Utils::random(pB + 0.3f, pT - 0.3f),
                               Utils::random(pF + 0.3f, pN - 0.3f),
                               TS_world);
            boxNode->castsShadows(true);

            scene->addChild(boxNode);
        }

        sv->camera(cam1);
        s->root3D(scene);
    }
    else if (SLApplication::sceneID == SID_SuzannePerPixBlinn) //........................................
    {
        // Set scene name and info string
        s->name("Suzanne with per pixel lighting");
        s->info(s->name());

        // Create a scene group node
        SLNode* scene = new SLNode("scene node");

        // Create camera in the center
        SLCamera* cam1 = new SLCamera("Camera 1");
        cam1->translation(0, 0.5f, 2);
        cam1->lookAt(0, 0.5f, 0);
        cam1->setInitialState();
        cam1->focalDist(2);
        scene->addChild(cam1);

        // Create directional light for the sun light
        SLLightDirect* light = new SLLightDirect(s, s, 0.1f);
        light->ambientPower(0.5f);
        light->diffusePower(0.5f);
        light->attenuation(1, 0, 0);
        light->translate(0, 0, 0.5);
        light->lookAt(1, -1, 0.5);
        SLAnimation* lightAnim = s->animManager().createNodeAnimation("LightAnim", 4.0f, true, EC_inOutSine, AL_pingPongLoop);
        lightAnim->createSimpleRotationNodeTrack(light, -180, SLVec3f(0, 1, 0));
        scene->addChild(light);

        // load teapot
        SLAssimpImporter importer;
        SLNode*          suzanneInCube = importer.load(s->animManager(),
                                              s,
                                              SLApplication::modelPath + "GLTF/AO-Baked-Test/AO-Baked-Test.gltf",
                                              SLApplication::texturePath,
                                              true,    // load meshes only
                                              nullptr, // override material
                                              0.5f);   // ambient factor

        // Setup shadow mapping material and replace shader from loader
        SLGLProgram* progPerPixNrm = new SLGLGenericProgram(s,
                                                            SLApplication::shaderPath + "PerPixBlinn.vert",
                                                            SLApplication::shaderPath + "PerPixBlinn.frag");
        auto         updateMat     = [=](SLMaterial* mat) { mat->program(progPerPixNrm); };
        suzanneInCube->updateMeshMat(updateMat, true);

        scene->addChild(suzanneInCube);

        sv->camera(cam1);

        // pass the scene group as root node
        s->root3D(scene);

        // Save energy
        sv->doWaitOnIdle(true);
    }
    else if (SLApplication::sceneID == SID_SuzannePerPixBlinnSM) //......................................
    {
        // Set scene name and info string
        s->name("Suzanne with per pixel lighting and shadow mapping");
        s->info(s->name());

        // Create a scene group node
        SLNode* scene = new SLNode("scene node");

        // Create camera in the center
        SLCamera* cam1 = new SLCamera("Camera 1");
        cam1->translation(0, 0.5f, 2);
        cam1->lookAt(0, 0.5f, 0);
        cam1->setInitialState();
        cam1->focalDist(2);
        scene->addChild(cam1);

        // Create directional light for the sun light
        SLLightDirect* light = new SLLightDirect(s, s, 0.1f);
        light->ambientPower(0.5f);
        light->diffusePower(0.5f);
        light->attenuation(1, 0, 0);
        light->translate(0, 0, 0.5);
        light->lookAt(1, -1, 0.5);
        light->createsShadows(true);
        light->createShadowMap(-3, 3, SLVec2f(5, 5), SLVec2i(2048, 2048));
        light->doSmoothShadows(true);
        SLAnimation* lightAnim = s->animManager().createNodeAnimation("LightAnim", 4.0f, true, EC_inOutSine, AL_pingPongLoop);
        lightAnim->createSimpleRotationNodeTrack(light, -180, SLVec3f(0, 1, 0));
        scene->addChild(light);

        // load teapot
        SLAssimpImporter importer;
        SLNode*          suzanneInCube = importer.load(s->animManager(),
                                              s,
                                              SLApplication::modelPath + "GLTF/AO-Baked-Test/AO-Baked-Test.gltf",
                                              SLApplication::texturePath,
                                              true,    // load meshes only
                                              nullptr, // override material
                                              0.5f);   // ambient factor

        // Setup shadow mapping material and replace shader from loader
        SLGLProgram* progPerPixNrm = new SLGLGenericProgram(s,
                                                            SLApplication::shaderPath + "PerPixBlinnSM.vert",
                                                            SLApplication::shaderPath + "PerPixBlinnSM.frag");
        auto         updateMat     = [=](SLMaterial* mat) { mat->program(progPerPixNrm); };
        suzanneInCube->updateMeshMat(updateMat, true);

        scene->addChild(suzanneInCube);

        sv->camera(cam1);

        // pass the scene group as root node
        s->root3D(scene);

        // Save energy
        sv->doWaitOnIdle(true);
    }
    else if (SLApplication::sceneID == SID_SuzannePerPixBlinnSMAO) //....................................
    {
        // Set scene name and info string
        s->name("Suzanne with per pixel lighting, shadow mapping and ambient occlusion");
        s->info(s->name());

        // Create a scene group node
        SLNode* scene = new SLNode("scene node");

        // Create camera in the center
        SLCamera* cam1 = new SLCamera("Camera 1");
        cam1->translation(0, 0.5f, 2);
        cam1->lookAt(0, 0.5f, 0);
        cam1->setInitialState();
        cam1->focalDist(2);
        scene->addChild(cam1);

        // Create directional light for the sun light
        SLLightDirect* light = new SLLightDirect(s, s, 0.1f);
        light->ambientPower(0.5f);
        light->diffusePower(0.5f);
        light->attenuation(1, 0, 0);
        light->translate(0, 0, 0.5);
        light->lookAt(1, -1, 0.5);
        light->createsShadows(true);
        light->createShadowMap(-3, 3, SLVec2f(5, 5), SLVec2i(2048, 2048));
        light->doSmoothShadows(true);
        SLAnimation* lightAnim = s->animManager().createNodeAnimation("LightAnim", 4.0f, true, EC_inOutSine, AL_pingPongLoop);
        lightAnim->createSimpleRotationNodeTrack(light, -180, SLVec3f(0, 1, 0));
        scene->addChild(light);

        // load teapot
        SLAssimpImporter importer;
        SLNode*          suzanneInCube = importer.load(s->animManager(),
                                              s,
                                              SLApplication::modelPath + "GLTF/AO-Baked-Test/AO-Baked-Test.gltf",
                                              SLApplication::texturePath,
                                              true,    // load meshes only
                                              nullptr, // override material
                                              0.5f);   // ambient factor

        // Setup shadow mapping material and replace shader from loader
        SLGLProgram* progPerPixNrm = new SLGLGenericProgram(s,
                                                            SLApplication::shaderPath + "PerPixBlinnSMAO.vert",
                                                            SLApplication::shaderPath + "PerPixBlinnSMAO.frag");
        auto         updateMat     = [=](SLMaterial* mat) { mat->program(progPerPixNrm); };
        suzanneInCube->updateMeshMat(updateMat, true);

        scene->addChild(suzanneInCube);

        sv->camera(cam1);

        // pass the scene group as root node
        s->root3D(scene);

        // Save energy
        sv->doWaitOnIdle(true);
    }
    else if (SLApplication::sceneID == SID_SuzannePerPixBlinnTex) //.....................................
    {
        // Set scene name and info string
        s->name("Suzanne with per pixel lighting and texture mapping");
        s->info(s->name());

        // Create a scene group node
        SLNode* scene = new SLNode("scene node");

        // Create camera in the center
        SLCamera* cam1 = new SLCamera("Camera 1");
        cam1->translation(0, 0.5f, 2);
        cam1->lookAt(0, 0.5f, 0);
        cam1->setInitialState();
        cam1->focalDist(2);
        scene->addChild(cam1);

        // Create directional light for the sun light
        SLLightDirect* light = new SLLightDirect(s, s, 0.1f);
        light->ambientPower(0.8f);
        light->diffusePower(0.8f);
        light->attenuation(1, 0, 0);
        light->translate(0, 0, 0.5);
        light->lookAt(1, -1, 0.5);
        SLAnimation* lightAnim = s->animManager().createNodeAnimation("LightAnim", 4.0f, true, EC_inOutSine, AL_pingPongLoop);
        lightAnim->createSimpleRotationNodeTrack(light, -180, SLVec3f(0, 1, 0));
        scene->addChild(light);

        // load teapot
        SLAssimpImporter importer;
        SLNode*          suzanneInCube = importer.load(s->animManager(),
                                              s,
                                              SLApplication::modelPath + "GLTF/AO-Baked-Test/AO-Baked-Test.gltf",
                                              SLApplication::texturePath,
                                              true,    // load meshes only
                                              nullptr, // override material
                                              0.5f);   // ambient factor

        // Setup shadow mapping material and replace shader from loader
        SLGLProgram* progPerPixNrm = new SLGLGenericProgram(s,
                                                            SLApplication::shaderPath + "PerPixBlinnTex.vert",
                                                            SLApplication::shaderPath + "PerPixBlinnTex.frag");
        auto         updateMat     = [=](SLMaterial* mat) { mat->program(progPerPixNrm); };
        suzanneInCube->updateMeshMat(updateMat, true);

        scene->addChild(suzanneInCube);

        sv->camera(cam1);

        // pass the scene group as root node
        s->root3D(scene);

        // Save energy
        sv->doWaitOnIdle(true);
    }
    else if (SLApplication::sceneID == SID_SuzannePerPixBlinnTexNrm) //..................................
    {
        // Set scene name and info string
        s->name("Suzanne with per pixel lighting and texture and normal mapping");
        s->info(s->name());

        // Create a scene group node
        SLNode* scene = new SLNode("scene node");

        // Create camera in the center
        SLCamera* cam1 = new SLCamera("Camera 1");
        cam1->translation(0, 0.5f, 2);
        cam1->lookAt(0, 0.5f, 0);
        cam1->setInitialState();
        cam1->focalDist(2);
        scene->addChild(cam1);

        // Create directional light for the sun light
        SLLightDirect* light = new SLLightDirect(s, s, 0.1f);
        light->ambientPower(0.8f);
        light->diffusePower(0.8f);
        light->attenuation(1, 0, 0);
        light->translate(0, 0, 0.5);
        light->lookAt(1, -1, 0.5);
        SLAnimation* lightAnim = s->animManager().createNodeAnimation("LightAnim", 4.0f, true, EC_inOutSine, AL_pingPongLoop);
        lightAnim->createSimpleRotationNodeTrack(light, -180, SLVec3f(0, 1, 0));
        scene->addChild(light);

        // load teapot
        SLAssimpImporter importer;
        SLNode*          suzanneInCube = importer.load(s->animManager(),
                                              s,
                                              SLApplication::modelPath + "GLTF/AO-Baked-Test/AO-Baked-Test.gltf",
                                              SLApplication::texturePath,
                                              true,    // load meshes only
                                              nullptr, // override material
                                              0.5f);   // ambient factor

        // Setup shadow mapping material and replace shader from loader
        SLGLProgram* progPerPixNrm = new SLGLGenericProgram(s,
                                                            SLApplication::shaderPath + "PerPixBlinnTexNrm.vert",
                                                            SLApplication::shaderPath + "PerPixBlinnTexNrm.frag");
        auto         updateMat     = [=](SLMaterial* mat) { mat->program(progPerPixNrm); };
        suzanneInCube->updateMeshMat(updateMat, true);

        scene->addChild(suzanneInCube);

        sv->camera(cam1);

        // pass the scene group as root node
        s->root3D(scene);

        // Save energy
        sv->doWaitOnIdle(true);
    }
    else if (SLApplication::sceneID == SID_SuzannePerPixBlinnTexNrmAO) //................................
    {
        // Set scene name and info string
        s->name("Suzanne with per pixel lighting and diffuse, normal and ambient occlusion mapping");
        s->info(s->name());

        // Create a scene group node
        SLNode* scene = new SLNode("scene node");

        // Create camera in the center
        SLCamera* cam1 = new SLCamera("Camera 1");
        cam1->translation(0, 0.5f, 2);
        cam1->lookAt(0, 0.5f, 0);
        cam1->setInitialState();
        cam1->focalDist(2);
        scene->addChild(cam1);

        // Create directional light for the sun light
        SLLightDirect* light = new SLLightDirect(s, s, 0.1f);
        light->ambientPower(0.8f);
        light->diffusePower(0.8f);
        light->attenuation(1, 0, 0);
        light->translate(0, 0, 0.5);
        light->lookAt(1, -1, 0.5);
        SLAnimation* lightAnim = s->animManager().createNodeAnimation("LightAnim", 4.0f, true, EC_inOutSine, AL_pingPongLoop);
        lightAnim->createSimpleRotationNodeTrack(light, -180, SLVec3f(0, 1, 0));
        scene->addChild(light);

        // load teapot
        SLAssimpImporter importer;
        SLNode*          suzanneInCube = importer.load(s->animManager(),
                                              s,
                                              SLApplication::modelPath + "GLTF/AO-Baked-Test/AO-Baked-Test.gltf",
                                              SLApplication::texturePath,
                                              true,    // load meshes only
                                              nullptr, // override material
                                              0.5f);   // ambient factor

        // Setup shadow mapping material and replace shader from loader
        SLGLProgram* progPerPixNrm = new SLGLGenericProgram(s,
                                                            SLApplication::shaderPath + "PerPixBlinnTexNrmAO.vert",
                                                            SLApplication::shaderPath + "PerPixBlinnTexNrmAO.frag");
        auto         updateMat     = [=](SLMaterial* mat) { mat->program(progPerPixNrm); };
        suzanneInCube->updateMeshMat(updateMat, true);

        scene->addChild(suzanneInCube);

        sv->camera(cam1);

        // pass the scene group as root node
        s->root3D(scene);

        // Save energy
        sv->doWaitOnIdle(true);
    }
    else if (SLApplication::sceneID == SID_SuzannePerPixBlinnTexNrmSM) //................................
    {
        // Set scene name and info string
        s->name("Suzanne with per pixel lighting and diffuse, normal and shadow mapping but without ambient occlusion");
        s->info(s->name());

        // Create a scene group node
        SLNode* scene = new SLNode("scene node");

        // Create camera in the center
        SLCamera* cam1 = new SLCamera("Camera 1");
        cam1->translation(0, 0.5f, 2);
        cam1->lookAt(0, 0.5f, 0);
        cam1->setInitialState();
        cam1->focalDist(2);
        scene->addChild(cam1);

        // Create directional light for the sun light
        SLLightDirect* light = new SLLightDirect(s, s, 0.1f);
        light->ambientPower(0.8f);
        light->diffusePower(0.8f);
        light->attenuation(1, 0, 0);
        light->translate(0, 0, 0.5);
        light->lookAt(1, -1, 0.5);
        light->createsShadows(true);
        light->createShadowMap(-3, 3, SLVec2f(5, 5), SLVec2i(2048, 2048));
        light->doSmoothShadows(true);
        SLAnimation* lightAnim = s->animManager().createNodeAnimation("LightAnim", 4.0f, true, EC_inOutSine, AL_pingPongLoop);
        lightAnim->createSimpleRotationNodeTrack(light, -180, SLVec3f(0, 1, 0));
        scene->addChild(light);

        // load teapot
        SLAssimpImporter importer;
        SLNode*          suzanneInCube = importer.load(s->animManager(),
                                              s,
                                              SLApplication::modelPath + "GLTF/AO-Baked-Test/AO-Baked-Test.gltf",
                                              SLApplication::texturePath,
                                              true,    // load meshes only
                                              nullptr, // override material
                                              0.5f);   // ambient factor

        // Setup shadow mapping material and replace shader from loader
        SLGLProgram* progPerPixNrm = new SLGLGenericProgram(s,
                                                            SLApplication::shaderPath + "PerPixBlinnTexNrmSM.vert",
                                                            SLApplication::shaderPath + "PerPixBlinnTexNrmSM.frag");
        auto         updateMat     = [=](SLMaterial* mat) { mat->program(progPerPixNrm); };
        suzanneInCube->updateMeshMat(updateMat, true);

        scene->addChild(suzanneInCube);

        sv->camera(cam1);

        // pass the scene group as root node
        s->root3D(scene);

        // Save energy
        sv->doWaitOnIdle(true);
    }
    else if (SLApplication::sceneID == SID_SuzannePerPixBlinnTexNrmAOSM) //..............................
    {
        // Set scene name and info string
        s->name("Suzanne with per pixel lighting and diffuse, normal, shadow and ambient occlusion mapping");
        s->info(s->name());

        // Create a scene group node
        SLNode* scene = new SLNode("scene node");

        // Create camera in the center
        SLCamera* cam1 = new SLCamera("Camera 1");
        cam1->translation(0, 0.5f, 2);
        cam1->lookAt(0, 0.5f, 0);
        cam1->setInitialState();
        cam1->focalDist(2);
        scene->addChild(cam1);

        // Create directional light for the sun light
        SLLightDirect* light = new SLLightDirect(s, s, 0.1f);
        light->ambientPower(1.0f);
        light->diffusePower(0.8f);
        light->attenuation(1, 0, 0);
        light->translate(0, 0, 0.5);
        light->lookAt(1, -1, 0.5);
        light->createsShadows(true);
        light->createShadowMap(-3, 3, SLVec2f(5, 5), SLVec2i(2048, 2048));
        light->doSmoothShadows(true);
        SLAnimation* lightAnim = s->animManager().createNodeAnimation("LightAnim", 4.0f, true, EC_inOutSine, AL_pingPongLoop);
        lightAnim->createSimpleRotationNodeTrack(light, -180, SLVec3f(0, 1, 0));
        scene->addChild(light);

        // load teapot
        SLAssimpImporter importer;
        SLNode*          suzanneInCube = importer.load(s->animManager(),
                                              s,
                                              SLApplication::modelPath + "GLTF/AO-Baked-Test/AO-Baked-Test.gltf",
                                              SLApplication::texturePath,
                                              true,    // load meshes only
                                              nullptr, // override material
                                              0.5f);   // ambient factor

        scene->addChild(suzanneInCube);

        sv->camera(cam1);

        // pass the scene group as root node
        s->root3D(scene);

        // Save energy
        sv->doWaitOnIdle(true);
    }
    else if (SLApplication::sceneID == SID_VolumeRayCast) //.............................................
    {
        s->name("Volume Ray Cast Test");
        s->info("Volume Rendering of an angiographic MRI scan");

        // Load volume data into 3D texture
        SLVstring mriImages;
        for (SLint i = 0; i < 207; ++i)
            mriImages.push_back(Utils::formatString(SLApplication::texturePath + "i%04u_0000b.png", i));

        SLint clamping3D = GL_CLAMP_TO_EDGE;
        if (SLGLState::instance()->getSLVersionNO() > "320")
            clamping3D = 0x812D; // GL_CLAMP_TO_BORDER

        SLGLTexture* texMRI = new SLGLTexture(s,
                                              mriImages,
                                              GL_LINEAR,
                                              GL_LINEAR,
                                              clamping3D,
                                              clamping3D,
                                              "mri_head_front_to_back");

        // Create transfer LUT 1D texture
        SLVAlphaLUTPoint tfAlphas = {SLAlphaLUTPoint(0.00f, 0.00f),
                                     SLAlphaLUTPoint(0.01f, 0.75f),
                                     SLAlphaLUTPoint(1.00f, 1.00f)};
        SLColorLUT*      tf       = new SLColorLUT(s, tfAlphas, CLUT_BCGYR);

        // Load shader and uniforms for volume size
        SLGLProgram*   sp   = new SLGLGenericProgram(s,
                                                 SLApplication::shaderPath + "VolumeRenderingRayCast.vert",
                                                 SLApplication::shaderPath + "VolumeRenderingRayCast.frag");
        SLGLUniform1f* volX = new SLGLUniform1f(UT_const, "u_volumeX", (SLfloat)texMRI->images()[0]->width());
        SLGLUniform1f* volY = new SLGLUniform1f(UT_const, "u_volumeY", (SLfloat)texMRI->images()[0]->height());
        SLGLUniform1f* volZ = new SLGLUniform1f(UT_const, "u_volumeZ", (SLfloat)mriImages.size());
        sp->addUniform1f(volX);
        sp->addUniform1f(volY);
        sp->addUniform1f(volZ);

        // Create volume rendering material
        SLMaterial* matVR = new SLMaterial(s, "matVR", texMRI, tf, nullptr, nullptr, sp);

        // Create camera
        SLCamera* cam1 = new SLCamera("Camera 1");
        cam1->translation(0, 0, 3);
        cam1->lookAt(0, 0, 0);
        cam1->focalDist(3);
        cam1->background().colors(SLCol4f(0, 0, 0));
        cam1->setInitialState();
        cam1->devRotLoc(&SLApplication::devRot, &SLApplication::devLoc);

        // Set light
        SLLightSpot* light1 = new SLLightSpot(s, s, 0.3f);
        light1->powers(0.1f, 1.0f, 1.0f);
        light1->attenuation(1, 0, 0);
        light1->translation(5, 5, 5);

        // Assemble scene with box node
        SLNode* scene = new SLNode("Scene");
        scene->addChild(light1);
        scene->addChild(new SLNode(new SLBox(s, -1, -1, -1, 1, 1, 1, "Box", matVR)));
        scene->addChild(cam1);

        sv->camera(cam1);
        s->root3D(scene);
    }
    else if (SLApplication::sceneID == SID_VolumeRayCastLighted) //......................................
    {
        s->name("Volume Ray Cast Lighted Test");
        s->info("Volume Rendering of an angiographic MRI scan with lighting");

        // The MRI Images got loaded in advance
        if (gTexMRI3D && !gTexMRI3D->images().empty())
        {
            // Add pointer to the global resource vectors for deallocation
            if (s)
                s->textures().push_back(gTexMRI3D);
        }
        else
        {
            // Load volume data into 3D texture
            SLVstring mriImages;
            for (SLint i = 0; i < 207; ++i)
                mriImages.push_back(Utils::formatString(SLApplication::texturePath + "i%04u_0000b.png", i));

            gTexMRI3D = new SLGLTexture(s,
                                        mriImages,
                                        GL_LINEAR,
                                        GL_LINEAR,
                                        0x812D, // GL_CLAMP_TO_BORDER (GLSL 320)
                                        0x812D, // GL_CLAMP_TO_BORDER (GLSL 320)
                                        "mri_head_front_to_back",
                                        true);

            gTexMRI3D->calc3DGradients(1, [](int progress) { SLApplication::jobProgressNum(progress); });
            //gTexMRI3D->smooth3DGradients(1, [](int progress) {SLApplication::jobProgressNum(progress);});
        }

        // Create transfer LUT 1D texture
        SLVAlphaLUTPoint tfAlphas = {SLAlphaLUTPoint(0.00f, 0.00f),
                                     SLAlphaLUTPoint(0.01f, 0.75f),
                                     SLAlphaLUTPoint(1.00f, 1.00f)};
        SLColorLUT*      tf       = new SLColorLUT(s, tfAlphas, CLUT_BCGYR);

        // Load shader and uniforms for volume size
        SLGLProgram*   sp   = new SLGLGenericProgram(s, SLApplication::shaderPath + "VolumeRenderingRayCast.vert", SLApplication::shaderPath + "VolumeRenderingRayCastLighted.frag");
        SLGLUniform1f* volX = new SLGLUniform1f(UT_const, "u_volumeX", (SLfloat)gTexMRI3D->images()[0]->width());
        SLGLUniform1f* volY = new SLGLUniform1f(UT_const, "u_volumeY", (SLfloat)gTexMRI3D->images()[0]->height());
        SLGLUniform1f* volZ = new SLGLUniform1f(UT_const, "u_volumeZ", (SLfloat)gTexMRI3D->images().size());
        sp->addUniform1f(volX);
        sp->addUniform1f(volY);
        sp->addUniform1f(volZ);

        // Create volume rendering material
        SLMaterial* matVR = new SLMaterial(s, "matVR", gTexMRI3D, tf, nullptr, nullptr, sp);

        // Create camera
        SLCamera* cam1 = new SLCamera("Camera 1");
        cam1->translation(0, 0, 3);
        cam1->lookAt(0, 0, 0);
        cam1->focalDist(3);
        cam1->background().colors(SLCol4f(0, 0, 0));
        cam1->setInitialState();
        cam1->devRotLoc(&SLApplication::devRot, &SLApplication::devLoc);

        // Set light
        SLLightSpot* light1 = new SLLightSpot(s, s, 0.3f);
        light1->powers(0.1f, 1.0f, 1.0f);
        light1->attenuation(1, 0, 0);
        light1->translation(5, 5, 5);

        // Assemble scene with box node
        SLNode* scene = new SLNode("Scene");
        scene->addChild(light1);
        scene->addChild(new SLNode(new SLBox(s, -1, -1, -1, 1, 1, 1, "Box", matVR)));
        scene->addChild(cam1);

        sv->camera(cam1);
        s->root3D(scene);
    }
    else if (SLApplication::sceneID == SID_AnimationSkeletal) //.........................................
    {
        s->name("Skeletal Animation Test");
        s->info("Skeletal Animation Test Scene");

        SLAssimpImporter importer;

        // Root scene node
        SLNode* scene = new SLNode("scene group");

        // camera
        SLCamera* cam1 = new SLCamera("Camera 1");
        cam1->translation(0, 2, 10);
        cam1->lookAt(0, 2, 0);
        cam1->focalDist(10);
        cam1->setInitialState();
        cam1->background().colors(SLCol4f(0.1f, 0.4f, 0.8f));
        cam1->setInitialState();
        cam1->devRotLoc(&SLApplication::devRot, &SLApplication::devLoc);
        scene->addChild(cam1);

        // light
        SLLightSpot* light1 = new SLLightSpot(s, s, 10, 10, 5, 0.5f);
        light1->powers(0.2f, 1.0f, 1.0f);
        light1->attenuation(1, 0, 0);
        scene->addChild(light1);

        // Floor grid
        SLMaterial* m2   = new SLMaterial(s, "m2", SLCol4f::WHITE);
        SLGrid*     grid = new SLGrid(s, SLVec3f(-5, 0, -5), SLVec3f(5, 0, 5), 20, 20, "Grid", m2);
        scene->addChild(new SLNode(grid, "grid"));

        // Astro boy character
        SLNode* char1 = importer.load(s->animManager(), s, SLApplication::modelPath + "DAE/AstroBoy/AstroBoy.dae", SLApplication::texturePath);
        char1->translate(-1, 0, 0);
        SLAnimPlayback* char1Anim = s->animManager().lastAnimPlayback();
        char1Anim->playForward();
        scene->addChild(char1);

        // Sintel character
        SLNode* char2 = importer.load(s->animManager(), s, SLApplication::modelPath + "DAE/Sintel/SintelLowResOwnRig.dae", SLApplication::texturePath
                                      //,true
                                      //,SLProcess_JoinIdenticalVertices
                                      //|SLProcess_RemoveRedundantMaterials
                                      //|SLProcess_SortByPType
                                      //|SLProcess_FindDegenerates
                                      //|SLProcess_FindInvalidData
                                      //|SLProcess_SplitLargeMeshes
        );
        char2->translate(1, 0, 0);
        SLAnimPlayback* char2Anim = s->animManager().lastAnimPlayback();
        char2Anim->playForward();
        scene->addChild(char2);

        // Skinned cube 1
        SLNode* cube1 = importer.load(s->animManager(), s, SLApplication::modelPath + "DAE/SkinnedCube/skinnedcube2.dae", SLApplication::texturePath);
        cube1->translate(3, 0, 0);
        SLAnimPlayback* cube1Anim = s->animManager().lastAnimPlayback();
        cube1Anim->easing(EC_inOutSine);
        cube1Anim->playForward();
        scene->addChild(cube1);

        // Skinned cube 2
        SLNode* cube2 = importer.load(s->animManager(), s, SLApplication::modelPath + "DAE/SkinnedCube/skinnedcube4.dae", SLApplication::texturePath);
        cube2->translate(-3, 0, 0);
        SLAnimPlayback* cube2Anim = s->animManager().lastAnimPlayback();
        cube2Anim->easing(EC_inOutSine);
        cube2Anim->playForward();
        scene->addChild(cube2);

        // Skinned cube 3
        SLNode* cube3 = importer.load(s->animManager(), s, SLApplication::modelPath + "DAE/SkinnedCube/skinnedcube5.dae", SLApplication::texturePath);
        cube3->translate(0, 3, 0);
        SLAnimPlayback* cube3Anim = s->animManager().lastAnimPlayback();
        cube3Anim->loop(AL_pingPongLoop);
        cube3Anim->easing(EC_inOutCubic);
        cube3Anim->playForward();
        scene->addChild(cube3);

        // Set active camera & the root pointer
        sv->camera(cam1);
        s->root3D(scene);
    }
    else if (SLApplication::sceneID == SID_AnimationNode) //.............................................
    {
        s->name("Node Animations Test");
        s->info("Node animations with different easing curves.");

        // Create textures and materials
        SLGLTexture* tex1 = new SLGLTexture(s, SLApplication::texturePath + "Checkerboard0512_C.png");
        SLMaterial*  m1   = new SLMaterial(s, "m1", tex1);
        m1->kr(0.5f);
        SLMaterial* m2 = new SLMaterial(s, "m2", SLCol4f::WHITE * 0.5, SLCol4f::WHITE, 128, 0.5f, 0.0f, 1.0f);

        SLMesh* floorMesh = new SLRectangle(s, SLVec2f(-5, -5), SLVec2f(5, 5), 20, 20, "FloorMesh", m1);
        SLNode* floorRect = new SLNode(floorMesh);
        floorRect->rotate(90, -1, 0, 0);
        floorRect->translate(0, 0, -5.5f);

        // Bouncing balls
        SLNode* ball1 = new SLNode(new SLSphere(s, 0.3f, 16, 16, "Ball1", m2));
        ball1->translate(0, 0, 4, TS_object);
        SLAnimation* ball1Anim = s->animManager().createNodeAnimation("Ball1_anim", 1.0f, true, EC_linear, AL_pingPongLoop);
        ball1Anim->createSimpleTranslationNodeTrack(ball1, SLVec3f(0.0f, -5.2f, 0.0f));

        SLNode* ball2 = new SLNode(new SLSphere(s, 0.3f, 16, 16, "Ball2", m2));
        ball2->translate(-1.5f, 0, 4, TS_object);
        SLAnimation* ball2Anim = s->animManager().createNodeAnimation("Ball2_anim", 1.0f, true, EC_inQuad, AL_pingPongLoop);
        ball2Anim->createSimpleTranslationNodeTrack(ball2, SLVec3f(0.0f, -5.2f, 0.0f));

        SLNode* ball3 = new SLNode(new SLSphere(s, 0.3f, 16, 16, "Ball3", m2));
        ball3->translate(-2.5f, 0, 4, TS_object);
        SLAnimation* ball3Anim = s->animManager().createNodeAnimation("Ball3_anim", 1.0f, true, EC_outQuad, AL_pingPongLoop);
        ball3Anim->createSimpleTranslationNodeTrack(ball3, SLVec3f(0.0f, -5.2f, 0.0f));

        SLNode* ball4 = new SLNode(new SLSphere(s, 0.3f, 16, 16, "Ball4", m2));
        ball4->translate(1.5f, 0, 4, TS_object);
        SLAnimation* ball4Anim = s->animManager().createNodeAnimation("Ball4_anim", 1.0f, true, EC_inOutQuad, AL_pingPongLoop);
        ball4Anim->createSimpleTranslationNodeTrack(ball4, SLVec3f(0.0f, -5.2f, 0.0f));

        SLNode* ball5 = new SLNode(new SLSphere(s, 0.3f, 16, 16, "Ball5", m2));
        ball5->translate(2.5f, 0, 4, TS_object);
        SLAnimation* ball5Anim = s->animManager().createNodeAnimation("Ball5_anim", 1.0f, true, EC_outInQuad, AL_pingPongLoop);
        ball5Anim->createSimpleTranslationNodeTrack(ball5, SLVec3f(0.0f, -5.2f, 0.0f));

        SLCamera* cam1 = new SLCamera("Camera 1");
        cam1->translation(0, 0, 22);
        cam1->lookAt(0, 0, 0);
        cam1->focalDist(22);
        cam1->setInitialState();
        cam1->devRotLoc(&SLApplication::devRot, &SLApplication::devLoc);

        SLCamera* cam2 = new SLCamera("Camera 2");
        cam2->translation(5, 0, 0);
        cam2->lookAt(0, 0, 0);
        cam2->focalDist(5);
        cam2->clipFar(10);
        cam2->background().colors(SLCol4f(0, 0, 0.6f), SLCol4f(0, 0, 0.3f));
        cam2->setInitialState();
        cam2->devRotLoc(&SLApplication::devRot, &SLApplication::devLoc);

        SLCamera* cam3 = new SLCamera("Camera 3");
        cam3->translation(-5, -2, 0);
        cam3->lookAt(0, 0, 0);
        cam3->focalDist(5);
        cam3->clipFar(10);
        cam3->background().colors(SLCol4f(0.6f, 0, 0), SLCol4f(0.3f, 0, 0));
        cam3->setInitialState();
        cam3->devRotLoc(&SLApplication::devRot, &SLApplication::devLoc);

        SLLightSpot* light1 = new SLLightSpot(s, s, 0, 2, 0, 0.5f);
        light1->powers(0.2f, 1.0f, 1.0f);
        light1->attenuation(1, 0, 0);
        SLAnimation* light1Anim = s->animManager().createNodeAnimation("Light1_anim", 4.0f);
        light1Anim->createEllipticNodeTrack(light1, 6, A_z, 6, A_x);

        SLLightSpot* light2 = new SLLightSpot(s, s, 0, 0, 0, 0.2f);
        light2->powers(0.1f, 1.0f, 1.0f);
        light2->attenuation(1, 0, 0);
        light2->translate(-8, -4, 0, TS_world);
        light2->setInitialState();
        SLAnimation*     light2Anim = s->animManager().createNodeAnimation("light2_anim", 2.0f, true, EC_linear, AL_pingPongLoop);
        SLNodeAnimTrack* track      = light2Anim->createNodeAnimationTrack();
        track->animatedNode(light2);
        track->createNodeKeyframe(0.0f);
        track->createNodeKeyframe(1.0f)->translation(SLVec3f(8, 8, 0));
        track->createNodeKeyframe(2.0f)->translation(SLVec3f(16, 0, 0));
        track->translationInterpolation(AI_bezier);

        SLNode* figure = BuildFigureGroup(s, m2, true);

        SLNode* scene = new SLNode("Scene");
        scene->addChild(light1);
        scene->addChild(light2);
        scene->addChild(cam1);
        scene->addChild(cam2);
        scene->addChild(cam3);
        scene->addChild(floorRect);
        scene->addChild(ball1);
        scene->addChild(ball2);
        scene->addChild(ball3);
        scene->addChild(ball4);
        scene->addChild(ball5);
        scene->addChild(figure);

        // Set active camera & the root pointer
        sv->camera(cam1);
        s->root3D(scene);
    }
    else if (SLApplication::sceneID == SID_AnimationMass) //.............................................
    {
        s->name("Mass Animation Test");
        s->info("Performance test for transform updates from many animations.");

        SLLightSpot* light1 = new SLLightSpot(s, s, 0.1f);
        light1->translate(0, 10, 0);

        // build a basic scene to have a reference for the occuring rotations
        SLMaterial* genericMat = new SLMaterial(s, "some material");

        // we use the same mesh to viasualize all the nodes
        SLBox* box = new SLBox(s, -0.5f, -0.5f, -0.5f, 0.5f, 0.5f, 0.5f, "box", genericMat);

        s->root3D(new SLNode);
        s->root3D()->addChild(light1);

        // we build a stack of levels, each level has a grid of boxes on it
        // each box on this grid has an other grid above it with child nodes
        // best results are achieved if gridSize is an uneven number.
        // (gridSize^2)^levels = num nodes. handle with care.
        const SLint levels      = 3;
        const SLint gridSize    = 3;
        const SLint gridHalf    = gridSize / 2;
        const SLint nodesPerLvl = gridSize * gridSize;

        // node spacing per level
        // nodes are 1^3 in size, we want to space the levels so that the densest levels meet
        // (so exactly 1 unit spacing between blocks)
        SLfloat nodeSpacing[levels];
        for (SLint i = 0; i < levels; ++i)
            nodeSpacing[(levels - 1) - i] = (SLfloat)pow((SLfloat)gridSize, (SLfloat)i);

        // lists to keep track of previous grid level to set parents correctly
        vector<SLNode*> parents;
        vector<SLNode*> curParentsVector;

        // first parent is the scene root
        parents.push_back(s->root3D());

        SLint nodeIndex = 0;
        for (float lvl : nodeSpacing)
        {
            curParentsVector = parents;
            parents.clear();

            // for each parent in the previous level, add a completely new grid
            for (auto parent : curParentsVector)
            {
                for (SLint i = 0; i < nodesPerLvl; ++i)
                {
                    SLNode* node = new SLNode("MassAnimNode");
                    node->addMesh(box);
                    parent->addChild(node);
                    parents.push_back(node);

                    // position
                    SLfloat x = (SLfloat)(i % gridSize - gridHalf);
                    SLfloat z = (SLfloat)((i > 0) ? i / gridSize - gridHalf : -gridHalf);
                    SLVec3f pos(x * lvl * 1.1f, 1.5f, z * lvl * 1.1f);

                    node->translate(pos, TS_object);
                    //node->scale(1.1f);

                    SLfloat       duration = 1.0f + 5.0f * ((SLfloat)i / (SLfloat)nodesPerLvl);
                    ostringstream oss;

                    oss << "random anim " << nodeIndex++;
                    SLAnimation* anim = s->animManager().createNodeAnimation(oss.str(), duration, true, EC_inOutSine, AL_pingPongLoop);
                    anim->createSimpleTranslationNodeTrack(node, SLVec3f(0.0f, 1.0f, 0.0f));
                }
            }
        }
    }
    else if (SLApplication::sceneID == SID_AnimationArmy) //.............................................
    {
        s->name("Astroboy Army Test");
        s->info("Mass animation scene of identical Astroboy models");

        // Create materials
        SLMaterial* m1 = new SLMaterial(s, "m1", SLCol4f::GRAY);
        m1->specular(SLCol4f::BLACK);

        // Define a light
        SLLightSpot* light1 = new SLLightSpot(s, s, 100, 40, 100, 1);
        light1->powers(0.1f, 1.0f, 1.0f);
        light1->attenuation(1, 0, 0);

        // Define camera
        SLCamera* cam1 = new SLCamera;
        cam1->translation(0, 10, 10);
        cam1->lookAt(0, 0, 0);
        cam1->focalDist(cam1->translationOS().length());
        cam1->background().colors(SLCol4f(0.1f, 0.4f, 0.8f));
        cam1->setInitialState();
        cam1->devRotLoc(&SLApplication::devRot, &SLApplication::devLoc);

        // Floor rectangle
        SLNode* rect = new SLNode(new SLRectangle(s,
                                                  SLVec2f(-20, -20),
                                                  SLVec2f(20, 20),
                                                  SLVec2f(0, 0),
                                                  SLVec2f(50, 50),
                                                  50,
                                                  50,
                                                  "Floor",
                                                  m1));
        rect->rotate(90, -1, 0, 0);

        SLAssimpImporter importer;
        SLNode*          center = importer.load(s->animManager(), s, SLApplication::modelPath + "DAE/AstroBoy/AstroBoy.dae", SLApplication::texturePath);
        s->animManager().lastAnimPlayback()->playForward();

        // Assemble scene
        SLNode* scene = new SLNode("scene group");
        scene->addChild(light1);
        scene->addChild(rect);
        scene->addChild(center);
        scene->addChild(cam1);

        // create astroboys around the center astroboy
        SLint size = 4;
        for (SLint iZ = -size; iZ <= size; ++iZ)
        {
            for (SLint iX = -size; iX <= size; ++iX)
            {
                SLbool shift = iX % 2 != 0;
                if (iX != 0 || iZ != 0)
                {
                    float   xt = float(iX) * 1.0f;
                    float   zt = float(iZ) * 1.0f + ((shift) ? 0.5f : 0.0f);
                    SLNode* n  = center->copyRec();
                    n->translate(xt, 0, zt, TS_object);
                    scene->addChild(n);
                }
            }
        }

        // Set active camera & the root pointer
        sv->camera(cam1);
        s->root3D(scene);
    }
    else if (SLApplication::sceneID == SID_VideoTextureLive ||
             SLApplication::sceneID == SID_VideoTextureFile) //..........................................
    {
        // Set scene name and info string
        if (SLApplication::sceneID == SID_VideoTextureLive)
        {
            s->name("Live Video Texture");
            s->info("Minimal texture mapping example with live video source.");
            CVCapture::instance()->videoType(VT_MAIN); // on desktop it will be the main camera
        }
        else
        {
            s->name("File Video Texture");
            s->info("Minimal texture mapping example with video file source.");
            CVCapture::instance()->videoType(VT_FILE);
            CVCapture::instance()->videoFilename = SLApplication::videoPath + "street3.mp4";
            CVCapture::instance()->videoLoops    = true;
        }
        sv->viewportSameAsVideo(true);

        // Create video texture on global pointer updated in AppDemoVideo
        videoTexture   = new SLGLTexture(s, SLApplication::texturePath + "LiveVideoError.png", GL_LINEAR, GL_LINEAR);
        SLMaterial* m1 = new SLMaterial(s, "VideoMat", videoTexture);

        // Create a root scene group for all nodes
        SLNode* scene = new SLNode("scene node");

        // Create a camera node
        SLCamera* cam1 = new SLCamera("Camera 1");
        cam1->translation(0, 0, 20);
        cam1->focalDist(20);
        cam1->lookAt(0, 0, 0);
        cam1->background().texture(videoTexture);
        cam1->setInitialState();
        cam1->devRotLoc(&SLApplication::devRot, &SLApplication::devLoc);
        scene->addChild(cam1);

        // Create rectangle meshe and nodes
        SLfloat h        = 5.0f;
        SLfloat w        = h * sv->viewportWdivH();
        SLMesh* rectMesh = new SLRectangle(s, SLVec2f(-w, -h), SLVec2f(w, h), 1, 1, "rect mesh", m1);
        SLNode* rectNode = new SLNode(rectMesh, "rect node");
        rectNode->translation(0, 0, -5);
        scene->addChild(rectNode);

        // Center sphere
        SLNode* sphere = new SLNode(new SLSphere(s, 2, 32, 32, "Sphere", m1));
        sphere->rotate(-90, 1, 0, 0);
        scene->addChild(sphere);

        // Create a light source node
        SLLightSpot* light1 = new SLLightSpot(s, s, 0.3f);
        light1->translation(0, 0, 5);
        light1->lookAt(0, 0, 0);
        light1->name("light node");
        scene->addChild(light1);

        s->root3D(scene);

        // Set active camera
        sv->camera(cam1);
        sv->doWaitOnIdle(false);
    }
    else if (SLApplication::sceneID == SID_VideoTrackChessMain ||
             SLApplication::sceneID == SID_VideoTrackChessScnd ||
             SLApplication::sceneID == SID_VideoCalibrateMain ||
             SLApplication::sceneID == SID_VideoCalibrateScnd) //........................................
    {
        /*
        The tracking of markers is done in AppDemoVideo::onUpdateTracking by calling the specific
        CVTracked::track method. If a marker was found it overwrites the linked nodes
        object matrix (SLNode::_om). If the linked node is the active camera the found
        transform is additionally inversed. This would be the standard augmented realtiy
        use case.
        The chessboard marker used in these scenes is also used for the camera
        calibration. The different calibration state changes are also handled in
        AppDemoVideo::onUpdateVideo.
        */

        // Setup here only the requested scene.
        if (SLApplication::sceneID == SID_VideoTrackChessMain ||
            SLApplication::sceneID == SID_VideoTrackChessScnd)
        {
            if (SLApplication::sceneID == SID_VideoTrackChessMain)
            {
                CVCapture::instance()->videoType(VT_MAIN);
                s->name("Track Chessboard (main cam.)");
            }
            else
            {
                CVCapture::instance()->videoType(VT_SCND);
                s->name("Track Chessboard (scnd cam.");
            }
        }
        else if (SLApplication::sceneID == SID_VideoCalibrateMain)
        {
            if (SLApplication::calibrationEstimator)
            {
                delete SLApplication::calibrationEstimator;
                SLApplication::calibrationEstimator = nullptr;
            }
            CVCapture::instance()->videoType(VT_MAIN);
            s->name("Calibrate Main Cam.");
        }
        else if (SLApplication::sceneID == SID_VideoCalibrateScnd)
        {
            if (SLApplication::calibrationEstimator)
            {
                delete SLApplication::calibrationEstimator;
                SLApplication::calibrationEstimator = nullptr;
            }
            CVCapture::instance()->videoType(VT_SCND);
            s->name("Calibrate Scnd Cam.");
        }

        // Create video texture on global pointer updated in AppDemoVideo
        videoTexture = new SLGLTexture(s, SLApplication::texturePath + "LiveVideoError.png", GL_LINEAR, GL_LINEAR);

        // Material
        SLMaterial* yellow = new SLMaterial(s, "mY", SLCol4f(1, 1, 0, 0.5f));

        // set the edge length of a chessboard square
        SLfloat e1 = 0.028f;
        SLfloat e3 = e1 * 3.0f;
        SLfloat e9 = e3 * 3.0f;

        // Create a scene group node
        SLNode* scene = new SLNode("scene node");

        // Create a camera node
        SLCamera* cam1 = new SLCamera();
        cam1->name("camera node");
        cam1->translation(0, 0, 5);
        cam1->lookAt(0, 0, 0);
        cam1->focalDist(5);
        cam1->clipFar(10);
        cam1->fov(CVCapture::instance()->activeCamera->calibration.cameraFovVDeg());
        cam1->background().texture(videoTexture);
        cam1->setInitialState();
        cam1->devRotLoc(&SLApplication::devRot, &SLApplication::devLoc);
        scene->addChild(cam1);

        // Create a light source node
        SLLightSpot* light1 = new SLLightSpot(s, s, e1 * 0.5f);
        light1->translate(e9, e9, e9);
        light1->name("light node");
        scene->addChild(light1);

        // Build mesh & node
        if (SLApplication::sceneID == SID_VideoTrackChessMain ||
            SLApplication::sceneID == SID_VideoTrackChessScnd)
        {
            SLBox*  box     = new SLBox(s, 0.0f, 0.0f, 0.0f, e3, e3, e3, "Box", yellow);
            SLNode* boxNode = new SLNode(box, "Box Node");
            boxNode->setDrawBitsRec(SL_DB_CULLOFF, true);
            SLNode* axisNode = new SLNode(new SLCoordAxis(s), "Axis Node");
            axisNode->setDrawBitsRec(SL_DB_MESHWIRED, false);
            axisNode->scale(e3);
            boxNode->addChild(axisNode);
            scene->addChild(boxNode);
        }

        // Create OpenCV Tracker for the camera node for AR camera.
        tracker = new CVTrackedChessboard(SLApplication::calibIniPath);
        tracker->drawDetection(true);
        trackedNode = cam1;

        // pass the scene group as root node
        s->root3D(scene);

        // Set active camera
        sv->camera(cam1);
        sv->doWaitOnIdle(false);
    }
    else if (SLApplication::sceneID == SID_VideoTrackArucoMain ||
             SLApplication::sceneID == SID_VideoTrackArucoScnd) //.......................................
    {
        /*
        The tracking of markers is done in AppDemoVideo::onUpdateVideo by calling the specific
        CVTracked::track method. If a marker was found it overwrites the linked nodes
        object matrix (SLNode::_om). If the linked node is the active camera the found
        transform is additionally inversed. This would be the standard augmented realtiy
        use case.
        */

        if (SLApplication::sceneID == SID_VideoTrackArucoMain)
        {
            CVCapture::instance()->videoType(VT_MAIN);
            s->name("Track Aruco (main cam.)");
            s->info("Hold the Aruco board dictionary 0 into the field of view of the main camera. You can find the Aruco markers in the file data/Calibrations. If not all markers are tracked you may have the mirror the video horizontally.");
        }
        else
        {
            CVCapture::instance()->videoType(VT_SCND);
            s->name("Track Aruco (scnd. cam.)");
            s->info("Hold the Aruco board dictionary 0 into the field of view of the secondary camera. You can find the Aruco markers in the file data/Calibrations. If not all markers are tracked you may have the mirror the video horizontally.");
        }

        // Create video texture on global pointer updated in AppDemoVideo
        videoTexture = new SLGLTexture(s, SLApplication::texturePath + "LiveVideoError.png", GL_LINEAR, GL_LINEAR);

        // Material
        SLMaterial* yellow = new SLMaterial(s, "mY", SLCol4f(1, 1, 0, 0.5f));
        SLMaterial* cyan   = new SLMaterial(s, "mY", SLCol4f(0, 1, 1, 0.5f));

        // Create a scene group node
        SLNode* scene = new SLNode("scene node");

        // Create a camera node 1
        SLCamera* cam1 = new SLCamera("Camera 1");
        cam1->translation(0, 0, 5);
        cam1->lookAt(0, 0, 0);
        cam1->fov(CVCapture::instance()->activeCamera->calibration.cameraFovVDeg());
        cam1->background().texture(videoTexture);
        cam1->setInitialState();
        cam1->devRotLoc(&SLApplication::devRot, &SLApplication::devLoc);
        scene->addChild(cam1);

        // Create a light source node
        SLLightSpot* light1 = new SLLightSpot(s, s, 0.02f);
        light1->translation(0.12f, 0.12f, 0.12f);
        light1->name("light node");
        scene->addChild(light1);

        // Get the half edge length of the aruco marker
        SLfloat edgeLen = CVTrackedAruco::params.edgeLength;
        SLfloat he      = edgeLen * 0.5f;

        // Build mesh & node that will be tracked by the 1st marker (camera)
        SLBox*  box1      = new SLBox(s, -he, -he, 0.0f, he, he, 2 * he, "Box 1", yellow);
        SLNode* boxNode1  = new SLNode(box1, "Box Node 1");
        SLNode* axisNode1 = new SLNode(new SLCoordAxis(s), "Axis Node 1");
        axisNode1->setDrawBitsRec(SL_DB_MESHWIRED, false);
        axisNode1->scale(edgeLen);
        boxNode1->addChild(axisNode1);
        boxNode1->setDrawBitsRec(SL_DB_CULLOFF, true);
        scene->addChild(boxNode1);

        // Create OpenCV Tracker for the box node
        tracker = new CVTrackedAruco(9, SLApplication::calibIniPath);
        tracker->drawDetection(true);
        trackedNode = boxNode1;

        // pass the scene group as root node
        s->root3D(scene);

        // Set active camera
        sv->camera(cam1);

        // Turn on constant redraw
        sv->doWaitOnIdle(false);
    }
    else if (SLApplication::sceneID == SID_VideoTrackFeature2DMain) //...................................
    {
        /*
        The tracking of markers is done in AppDemoVideo::onUpdateVideo by calling the specific
        CVTracked::track method. If a marker was found it overwrites the linked nodes
        object matrix (SLNode::_om). If the linked node is the active camera the found
        transform is additionally inversed. This would be the standard augmented realtiy
        use case.
        */

        s->name("Track 2D Features");
        s->info("Augmented Reality 2D Feature Tracking: You need to print out the stones image target from the file data/calibrations/vuforia_markers.pdf");

        // Create video texture on global pointer updated in AppDemoVideo
        videoTexture = new SLGLTexture(s, SLApplication::texturePath + "LiveVideoError.png", GL_LINEAR, GL_LINEAR);

        SLCamera* cam1 = new SLCamera("Camera 1");
        cam1->translation(0, 2, 60);
        cam1->lookAt(15, 15, 0);
        cam1->clipNear(0.1f);
        cam1->clipFar(1000.0f); // Increase to infinity?
        cam1->setInitialState();
        cam1->devRotLoc(&SLApplication::devRot, &SLApplication::devLoc);
        cam1->background().texture(videoTexture);
        CVCapture::instance()->videoType(VT_MAIN);

        SLLightSpot* light1 = new SLLightSpot(s, s, 420, 420, 420, 1);
        light1->powers(1.0f, 1.0f, 1.0f);

        SLLightSpot* light2 = new SLLightSpot(s, s, -450, -340, 420, 1);
        light2->powers(1.0f, 1.0f, 1.0f);
        light2->attenuation(1, 0, 0);

        SLLightSpot* light3 = new SLLightSpot(s, s, 450, -370, 0, 1);
        light3->powers(1.0f, 1.0f, 1.0f);
        light3->attenuation(1, 0, 0);

        // Coordinate axis node
        SLNode* axis = new SLNode(new SLCoordAxis(s), "Axis Node");
        axis->setDrawBitsRec(SL_DB_MESHWIRED, false);
        axis->scale(100);
        axis->rotate(-90, 1, 0, 0);

        // Yellow center box
        SLMaterial* yellow = new SLMaterial(s, "mY", SLCol4f(1, 1, 0, 0.5f));
        SLNode*     box    = new SLNode(new SLBox(s, 0, 0, 0, 100, 100, 100, "Box", yellow), "Box Node");
        box->rotate(-90, 1, 0, 0);

        // Scene structure
        SLNode* scene = new SLNode("Scene");
        scene->addChild(light1);
        scene->addChild(light2);
        scene->addChild(light3);
        scene->addChild(axis);
        scene->addChild(box);
        scene->addChild(cam1);

        // Create feature tracker and let it pose the camera for AR posing
        tracker = new CVTrackedFeatures(SLApplication::texturePath + "features_stones.jpg");
        //tracker = new CVTrackedFeatures("features_abstract.jpg");
        tracker->drawDetection(true);
        trackedNode = cam1;

        sv->doWaitOnIdle(false); // for constant video feed
        sv->camera(cam1);

        s->root3D(scene);
        SLApplication::devRot.isUsed(true);
    }
    else if (SLApplication::sceneID == SID_VideoTrackFaceMain ||
             SLApplication::sceneID == SID_VideoTrackFaceScnd) //........................................
    {
        /*
        The tracking of markers is done in AppDemoVideo::onUpdateVideo by calling the specific
        CVTracked::track method. If a marker was found it overwrites the linked nodes
        object matrix (SLNode::_om). If the linked node is the active camera the found
        transform is additionally inversed. This would be the standard augmented realtiy
        use case.
        */

        if (SLApplication::sceneID == SID_VideoTrackFaceMain)
        {
            CVCapture::instance()->videoType(VT_MAIN);
            s->name("Track Face (main cam.)");
        }
        else
        {
            CVCapture::instance()->videoType(VT_SCND);
            s->name("Track Face (scnd. cam.)");
        }
        s->info("Face and facial landmark detection.");

        // Create video texture on global pointer updated in AppDemoVideo
        videoTexture = new SLGLTexture(s, SLApplication::texturePath + "LiveVideoError.png", GL_LINEAR, GL_LINEAR);

        SLCamera* cam1 = new SLCamera("Camera 1");
        cam1->translation(0, 0, 0.5f);
        cam1->clipNear(0.1f);
        cam1->clipFar(1000.0f); // Increase to infinity?
        cam1->setInitialState();
        cam1->devRotLoc(&SLApplication::devRot, &SLApplication::devLoc);
        cam1->background().texture(videoTexture);

        SLLightSpot* light1 = new SLLightSpot(s, s, 10, 10, 10, 1);
        light1->powers(1.0f, 1.0f, 1.0f);
        light1->attenuation(1, 0, 0);

        // Load sunglasses
        SLAssimpImporter importer;
        SLNode*          glasses = importer.load(s->animManager(), s, SLApplication::modelPath + "FBX/Sunglasses.fbx", SLApplication::texturePath);
        glasses->scale(0.01f);

        // Add axis arrows at world center
        SLNode* axis = new SLNode(new SLCoordAxis(s), "Axis Node");
        axis->setDrawBitsRec(SL_DB_MESHWIRED, false);
        axis->scale(0.03f);

        // Scene structure
        SLNode* scene = new SLNode("Scene");
        scene->addChild(light1);
        scene->addChild(cam1);
        scene->addChild(glasses);
        scene->addChild(axis);

        // Add a face tracker that moves the camera node
        tracker     = new CVTrackedFaces(Utils::findFile("haarcascade_frontalface_alt2.xml", {SLApplication::calibIniPath, SLApplication::exePath}),
                                     Utils::findFile("lbfmodel.yaml", {SLApplication::calibIniPath, SLApplication::exePath}),
                                     3);
        trackedNode = cam1;
        tracker->drawDetection(true);

        sv->doWaitOnIdle(false); // for constant video feed
        sv->camera(cam1);

        s->root3D(scene);
    }
#ifdef SL_BUILD_WAI
    else if (SLApplication::sceneID == SID_VideoTrackWAI) //.............................................
    {
        CVCapture::instance()->videoType(VT_MAIN);
        s->name("Track WAI (main cam.)");
        s->info("Track the scene with a point cloud built with the WAI (Where Am I) library.");

        // Create video texture on global pointer updated in AppDemoVideo
        videoTexture = new SLGLTexture(s, SLApplication::texturePath + "LiveVideoError.png", GL_LINEAR, GL_LINEAR);

        // Material
        SLMaterial* yellow = new SLMaterial(s, "mY", SLCol4f(1, 1, 0, 0.5f));
        SLMaterial* cyan   = new SLMaterial(s, "mY", SLCol4f(0, 1, 1, 0.5f));

        // Create a scene group node
        SLNode* scene = new SLNode("scene node");

        // Create a camera node 1
        SLCamera* cam1 = new SLCamera("Camera 1");
        cam1->translation(0, 0, 5);
        cam1->lookAt(0, 0, 0);
        cam1->fov(CVCapture::instance()->activeCamera->calibration.cameraFovVDeg());
        cam1->background().texture(videoTexture);
        cam1->setInitialState();
        scene->addChild(cam1);

        // Create a light source node
        SLLightSpot* light1 = new SLLightSpot(s, s, 0.02f);
        light1->translation(0.12f, 0.12f, 0.12f);
        light1->name("light node");
        scene->addChild(light1);

        // Get the half edge length of the aruco marker
        SLfloat edgeLen = 0.1f;
        SLfloat he      = edgeLen * 0.5f;

        // Build mesh & node that will be tracked by the 1st marker (camera)
        SLBox*  box1      = new SLBox(s, -he, -he, -he, he, he, he, "Box 1", yellow);
        SLNode* boxNode1  = new SLNode(box1, "Box Node 1");
        SLNode* axisNode1 = new SLNode(new SLCoordAxis(s), "Axis Node 1");
        axisNode1->setDrawBitsRec(SL_DB_MESHWIRED, false);
        axisNode1->scale(edgeLen);
        axisNode1->translate(-he, -he, -he, TS_parent);
        boxNode1->addChild(axisNode1);
        boxNode1->setDrawBitsRec(SL_DB_CULLOFF, true);
        boxNode1->translate(0.0f, 0.0f, 1.0f, TS_world);
        scene->addChild(boxNode1);

        // Create OpenCV Tracker for the box node
        std::string vocFileName;
#    if USE_FBOW
        vocFileName = "voc_fbow.bin";
#    else
        vocFileName = "ORBvoc.bin";
#    endif
        tracker = new CVTrackedWAI(Utils::findFile(vocFileName, {SLApplication::calibIniPath, SLApplication::exePath}));
        tracker->drawDetection(true);
        trackedNode = cam1;

        // pass the scene group as root node
        s->root3D(scene);

        // Set active camera
        sv->camera(cam1);

        // Turn on constant redraw
        sv->doWaitOnIdle(false);
    }
#endif
    else if (SLApplication::sceneID == SID_VideoSensorAR) //.............................................
    {
        // Set scene name and info string
        s->name("Video Sensor AR");
        s->info("Minimal scene to test the devices IMU and GPS Sensors. See the sensor information. GPS needs a few sec. to improve the accuracy.");

        // Create video texture on global pointer updated in AppDemoVideo
        videoTexture = new SLGLTexture(s, SLApplication::texturePath + "LiveVideoError.png", GL_LINEAR, GL_LINEAR);

        SLCamera* cam1 = new SLCamera("Camera 1");
        cam1->translation(0, 0, 60);
        cam1->lookAt(0, 0, 0);
        cam1->fov(CVCapture::instance()->activeCamera->calibration.cameraFovVDeg());
        cam1->clipNear(0.1f);
        cam1->clipFar(10000.0f);
        cam1->setInitialState();
        cam1->devRotLoc(&SLApplication::devRot, &SLApplication::devLoc);
        cam1->background().texture(videoTexture);

        // Turn on main video
        CVCapture::instance()->videoType(VT_MAIN);

        // Create directional light for the sun light
        SLLightDirect* light = new SLLightDirect(s, s, 1.0f);
        light->powers(1.0f, 1.0f, 1.0f);
        light->attenuation(1, 0, 0);

        // Let the sun be rotated by time and location
        SLApplication::devLoc.sunLightNode(light);

        SLNode* axis = new SLNode(new SLCoordAxis(s), "Axis Node");
        axis->setDrawBitsRec(SL_DB_MESHWIRED, false);
        axis->scale(2);
        axis->rotate(-90, 1, 0, 0);

        // Yellow center box
        SLMaterial* yellow = new SLMaterial(s, "mY", SLCol4f(1, 1, 0, 0.5f));
        SLNode*     box    = new SLNode(new SLBox(s, -.5f, -.5f, -.5f, .5f, .5f, .5f, "Box", yellow), "Box Node");

        // Scene structure
        SLNode* scene = new SLNode("Scene");
        scene->addChild(light);
        scene->addChild(cam1);
        scene->addChild(box);
        scene->addChild(axis);

        sv->camera(cam1);

        s->root3D(scene);

#if defined(SL_OS_MACIOS) || defined(SL_OS_ANDROID)
        //activate rotation and gps sensor
        SLApplication::devRot.isUsed(true);
        SLApplication::devRot.zeroYawAtStart(false);
        SLApplication::devLoc.isUsed(true);
        SLApplication::devLoc.useOriginAltitude(true);
        SLApplication::devLoc.hasOrigin(false);
        cam1->camAnim(SLCamAnim::CA_deviceRotLocYUp);
#else
        cam1->camAnim(SLCamAnim::CA_turntableYUp);
        SLApplication::devRot.zeroYawAtStart(true);
#endif

        sv->doWaitOnIdle(false); // for constant video feed
    }
    else if (SLApplication::sceneID == SID_ErlebARBielBFH) //............................................
    {
        s->name("Biel-BFH AR");
        s->info("Augmented Reality at Biel-BFH");

        // Create video texture on global pointer updated in AppDemoVideo
        videoTexture = new SLGLTexture(s, SLApplication::texturePath + "LiveVideoError.png", GL_LINEAR, GL_LINEAR);

        // Define shader that shows on all pixels the video background
        SLGLProgram* spVideoBackground  = new SLGLGenericProgram(s,
                                                                SLApplication::shaderPath + "PerVrtTextureBackground.vert",
                                                                SLApplication::shaderPath + "PerVrtTextureBackground.frag");
        SLMaterial*  matVideoBackground = new SLMaterial(s,
                                                        "matVideoBackground",
                                                        videoTexture,
                                                        nullptr,
                                                        nullptr,
                                                        nullptr,
                                                        spVideoBackground);

        SLCamera* cam1 = new SLCamera("Camera 1");
        cam1->translation(0, 2, 0);
        cam1->lookAt(-10, 2, 0);
        cam1->clipNear(1);
        cam1->clipFar(1000);
        cam1->setInitialState();
        cam1->devRotLoc(&SLApplication::devRot, &SLApplication::devLoc);
        cam1->background().texture(videoTexture);

        // Turn on main video
        CVCapture::instance()->videoType(VT_MAIN);

        // Create directional light for the sun light
        SLLightDirect* sunLight = new SLLightDirect(s, s, 5.0f);
        sunLight->powers(1.0f, 1.0f, 1.0f);
        sunLight->attenuation(1, 0, 0);
        sunLight->doSunPowerAdaptation(true);
        sunLight->createsShadows(true);
        sunLight->createShadowMap(-100, 150, SLVec2f(150, 150), SLVec2i(2048, 2048));
        sunLight->doSmoothShadows(true);
        sunLight->castsShadows(false);

        // Let the sun be rotated by time and location
        SLApplication::devLoc.sunLightNode(sunLight);

        SLAssimpImporter importer;
        SLNode*          bfh = importer.load(s->animManager(),
                                    s,
                                    SLApplication::dataPath + "erleb-AR/models/biel/Biel-BFH-Rolex.gltf",
                                    SLApplication::texturePath);

        /* Setup shadow mapping material and replace shader from loader
        SLGLProgram* progPerPixSM = new SLGLGenericProgram(s,
                                                              SLApplication::shaderPath + "PerPixBlinnSM.vert",
                                                              SLApplication::shaderPath + "PerPixBlinnSM.frag");
        auto         updateMat       = [=](SLMaterial* mat) { mat->program(progPerPixSM); };
        bfh->updateMeshMat(updateMat, true);
        */
        bfh->setMeshMat(matVideoBackground, true);

        // Make terrain a video shine trough
        //bfh->findChild<SLNode>("Terrain")->setMeshMat(matVideoBackground, true);

        /* Make buildings transparent
        SLNode* buildings = bfh->findChild<SLNode>("Buildings");
        SLNode* roofs = bfh->findChild<SLNode>("Roofs");
        auto updateTranspFnc = [](SLMaterial* m) {m->kt(0.5f);};
        buildings->updateMeshMat(updateTranspFnc, true);
        roofs->updateMeshMat(updateTranspFnc, true);

        // Set ambient on all child nodes
        bfh->updateMeshMat([](SLMaterial* m) { m->ambient(SLCol4f(.2f, .2f, .2f)); }, true);
        */

        // Add axis object a world origin
        SLNode* axis = new SLNode(new SLCoordAxis(s), "Axis Node");
        axis->scale(2);
        axis->rotate(-90, 1, 0, 0);

        SLNode* scene = new SLNode("Scene");
        scene->addChild(sunLight);
        scene->addChild(axis);
        scene->addChild(bfh);
        scene->addChild(cam1);

        //initialize sensor stuff
        SLApplication::devLoc.originLatLonAlt(47.14271, 7.24337, 488.2);        // Ecke Giosa
        SLApplication::devLoc.defaultLatLonAlt(47.14260, 7.24310, 488.7 + 1.7); // auf Parkplatz
        SLApplication::devLoc.locMaxDistanceM(1000.0f);
        SLApplication::devLoc.improveOrigin(false);
        SLApplication::devLoc.useOriginAltitude(true);
        SLApplication::devLoc.hasOrigin(true);
        SLApplication::devRot.zeroYawAtStart(false);

        // This loads the DEM file and overwrites the altitude of originLatLonAlt and defaultLatLonAlt
        SLstring tif = SLApplication::dataPath + "erleb-AR/models/biel/DEM_Biel-BFH_WGS84.tif";
        SLApplication::devLoc.loadGeoTiff(tif);

#if defined(SL_OS_MACIOS) || defined(SL_OS_ANDROID)
        SLApplication::devLoc.isUsed(true);
        SLApplication::devRot.isUsed(true);
        cam1->camAnim(SLCamAnim::CA_deviceRotLocYUp);
#else
        SLApplication::devLoc.isUsed(false);
        SLApplication::devRot.isUsed(false);
        SLVec3d pos_d = SLApplication::devLoc.defaultENU() - SLApplication::devLoc.originENU();
        SLVec3f pos_f((SLfloat)pos_d.x, (SLfloat)pos_d.y, (SLfloat)pos_d.z);
        cam1->translation(pos_f);
        cam1->focalDist(pos_f.length());
        cam1->lookAt(SLVec3f::ZERO);
        cam1->camAnim(SLCamAnim::CA_turntableYUp);
#endif

        sv->doWaitOnIdle(false); // for constant video feed
        sv->camera(cam1);
        sv->drawBits()->on(SL_DB_ONLYEDGES);
        s->root3D(scene);
    }
    else if (SLApplication::sceneID == SID_ErlebARChristoffel) //........................................
    {
        s->name("Christoffel Tower AR");
        s->info("Augmented Reality Christoffel Tower");

        // Create video texture on global pointer updated in AppDemoVideo
        videoTexture = new SLGLTexture(s, SLApplication::texturePath + "LiveVideoError.png", GL_LINEAR, GL_LINEAR);

        // Define shader that shows on all pixels the video background
        SLGLProgram* spVideoBackground  = new SLGLGenericProgram(s,
                                                                SLApplication::shaderPath + "PerVrtTextureBackground.vert",
                                                                SLApplication::shaderPath + "PerVrtTextureBackground.frag");
        SLMaterial*  matVideoBackground = new SLMaterial(s,
                                                        "matVideoBackground",
                                                        videoTexture,
                                                        nullptr,
                                                        nullptr,
                                                        nullptr,
                                                        spVideoBackground);

        SLCamera* cam1 = new SLCamera("Camera 1");
        cam1->translation(0, 2, 0);
        cam1->lookAt(-10, 2, 0);
        cam1->clipNear(1);
        cam1->clipFar(500);
        cam1->setInitialState();
        cam1->devRotLoc(&SLApplication::devRot, &SLApplication::devLoc);
        cam1->background().texture(videoTexture);

        // Turn on main video
        CVCapture::instance()->videoType(VT_MAIN);

        // Create directional light for the sun light
        SLLightDirect* sunLight = new SLLightDirect(s, s, 5.0f);
        sunLight->powers(1.0f, 1.5f, 1.0f);
        sunLight->attenuation(1, 0, 0);
        sunLight->doSunPowerAdaptation(true);
        sunLight->createsShadows(true);
        sunLight->createShadowMap(-100, 150, SLVec2f(150, 150), SLVec2i(2048, 2048));
        sunLight->doSmoothShadows(true);
        sunLight->castsShadows(false);

        // Let the sun be rotated by time and location
        SLApplication::devLoc.sunLightNode(sunLight);

        SLAssimpImporter importer;
        SLNode*          bern = importer.load(s->animManager(),
                                     s,
                                     SLApplication::dataPath + "erleb-AR/models/bern/Bern-Bahnhofsplatz2.gltf",
                                     SLApplication::texturePath);

        // Make city with hard edges
        SLNode* UmgD = bern->findChild<SLNode>("Umgebung-Daecher");
        if (!UmgD) SL_EXIT_MSG("Node: Umgebung-Daecher not found!");
        SLNode* UmgF = bern->findChild<SLNode>("Umgebung-Fassaden");
        if (!UmgF) SL_EXIT_MSG("Node: Umgebung-Fassaden not found!");
        UmgD->setMeshMat(matVideoBackground, true);
        UmgF->setMeshMat(matVideoBackground, true);
        UmgD->setDrawBitsRec(SL_DB_WITHEDGES, true);
        UmgF->setDrawBitsRec(SL_DB_WITHEDGES, true);

        // Hide some objects
        //bern->findChild<SLNode>("Umgebung-Daecher")->drawBits()->set(SL_DB_HIDDEN, true);
        //bern->findChild<SLNode>("Umgebung-Fassaden")->drawBits()->set(SL_DB_HIDDEN, true);
        bern->findChild<SLNode>("Baldachin-Glas")->drawBits()->set(SL_DB_HIDDEN, true);
        bern->findChild<SLNode>("Baldachin-Stahl")->drawBits()->set(SL_DB_HIDDEN, true);

        // Set the video background shader on the baldachin and the ground
        bern->findChild<SLNode>("Baldachin-Stahl")->setMeshMat(matVideoBackground, true);
        bern->findChild<SLNode>("Baldachin-Glas")->setMeshMat(matVideoBackground, true);
        bern->findChild<SLNode>("Boden")->setMeshMat(matVideoBackground, true);

        // Set ambient on all child nodes
        bern->updateMeshMat([](SLMaterial* m)
                            {   if (m->name() != "Kupfer-dunkel")
                                    m->ambient(SLCol4f(.3f, .3f, .3f));
                            }, true);

        // Add axis object a world origin (Loeb Ecke)
        SLNode* axis = new SLNode(new SLCoordAxis(s), "Axis Node");
        axis->setDrawBitsRec(SL_DB_MESHWIRED, false);
        axis->scale(10);
        axis->rotate(-90, 1, 0, 0);

        SLMaterial* yellow = new SLMaterial(s, "mY", SLCol4f(1, 1, 0, 0.5f));
        SLNode*     box2m  = new SLNode(new SLBox(s, 0, 0, 0, 2, 2, 2, "Box2m", yellow));

        SLNode* scene = new SLNode("Scene");
        scene->addChild(sunLight);
        scene->addChild(axis);
        scene->addChild(box2m);
        scene->addChild(bern);
        scene->addChild(cam1);

        //initialize sensor stuff
        SLApplication::devLoc.originLatLonAlt(46.94763, 7.44074, 542.2);        // Loeb Ecken
        SLApplication::devLoc.defaultLatLonAlt(46.94841, 7.43970, 541.1 + 1.7); // Bahnhof Ausgang in Augenhöhe
        SLApplication::devLoc.locMaxDistanceM(1000.0f);                         // Max. Distanz. zum Loeb Ecken
        SLApplication::devLoc.improveOrigin(false);                             // Keine autom. Verbesserung vom Origin
        SLApplication::devLoc.useOriginAltitude(true);
        SLApplication::devLoc.hasOrigin(true);
        SLApplication::devRot.zeroYawAtStart(false);

        // This loads the DEM file and overwrites the altitude of originLatLonAlt and defaultLatLonAlt
        SLstring tif = SLApplication::dataPath + "erleb-AR/models/bern/DEM-Bern-2600_1199-WGS84.tif";
        SLApplication::devLoc.loadGeoTiff(tif);

#if defined(SL_OS_MACIOS) || defined(SL_OS_ANDROID)
        SLApplication::devLoc.isUsed(true);
        SLApplication::devRot.isUsed(true);
        cam1->camAnim(SLCamAnim::CA_deviceRotLocYUp);
#else
        SLApplication::devLoc.isUsed(false);
        SLApplication::devRot.isUsed(false);
        SLVec3d pos_d = SLApplication::devLoc.defaultENU() - SLApplication::devLoc.originENU();
        SLVec3f pos_f((SLfloat)pos_d.x, (SLfloat)pos_d.y, (SLfloat)pos_d.z);
        cam1->translation(pos_f);
        cam1->focalDist(pos_f.length());
        cam1->lookAt(SLVec3f::ZERO);
        cam1->camAnim(SLCamAnim::CA_turntableYUp);
#endif

        sv->doWaitOnIdle(false); // for constant video feed
        sv->camera(cam1);
        s->root3D(scene);
    }
    else if (SLApplication::sceneID == SID_ErlebARAugustaRaurica) //.....................................
    {
        s->name("Augusta Raurica AR");
        s->info("Augmented Reality for Augusta Raurica");

        SLCamera* cam1 = new SLCamera("Camera 1");
        cam1->translation(0, 50, -150);
        cam1->lookAt(0, 0, 0);
        cam1->clipNear(1);
        cam1->clipFar(400);
        cam1->focalDist(150);
        cam1->devRotLoc(&SLApplication::devRot, &SLApplication::devLoc);

        // Create video texture and turn on live video
        videoTexture = new SLGLTexture(s, SLApplication::texturePath + "LiveVideoError.png", GL_LINEAR, GL_LINEAR);
        cam1->background().texture(videoTexture);
        CVCapture::instance()->videoType(VT_MAIN);

        // Define shader that shows on all pixels the video background
        SLGLProgram* spVideoBackground  = new SLGLGenericProgram(s,
                                                                SLApplication::shaderPath + "PerVrtTextureBackground.vert",
                                                                SLApplication::shaderPath + "PerVrtTextureBackground.frag");
        SLMaterial*  matVideoBackground = new SLMaterial(s,
                                                        "matVideoBackground",
                                                        videoTexture,
                                                        nullptr,
                                                        nullptr,
                                                        nullptr,
                                                        spVideoBackground);
        // Create directional light for the sun light
        SLLightDirect* sunLight = new SLLightDirect(s, s, 5.0f);
        sunLight->powers(1.0f, 1.5f, 1.0f);
        sunLight->attenuation(1, 0, 0);
        sunLight->translation(0, 10, 0);
        sunLight->lookAt(10, 0, 10);
        sunLight->doSunPowerAdaptation(true);
        sunLight->createsShadows(true);
        sunLight->createShadowMap(-100, 250, SLVec2f(250, 150), SLVec2i(2048, 2048));
        sunLight->doSmoothShadows(true);
        sunLight->castsShadows(false);

        // Let the sun be rotated by time and location
        SLApplication::devLoc.sunLightNode(sunLight);

        SLAssimpImporter importer;
        SLNode*          TheaterAndTempel = importer.load(s->animManager(),
                                                 s,
                                                 SLApplication::dataPath + "erleb-AR/models/augst/Tempel-Theater-02.gltf",
                                                 SLApplication::texturePath,
                                                 true,    // only meshes
                                                 nullptr, // no replacement material
                                                 0.4f);   // 40% ambient reflection

        // Rotate to the true geographic rotation
        TheaterAndTempel->rotate(16.7f, 0, 1, 0, TS_parent);

        // Setup shadow mapping material and replace shader from loader
        SLGLProgram* progPerPixNrmSM = new SLGLGenericProgram(s,
                                                              SLApplication::shaderPath + "PerPixBlinnTexNrmSM.vert",
                                                              SLApplication::shaderPath + "PerPixBlinnTexNrmSM.frag");
        auto         updateMat       = [=](SLMaterial* mat) { mat->program(progPerPixNrmSM); };
        TheaterAndTempel->updateMeshMat(updateMat, true);

        // Let the video shine through on some objects
        TheaterAndTempel->findChild<SLNode>("Tmp-Boden")->setMeshMat(matVideoBackground, true);
        TheaterAndTempel->findChild<SLNode>("Tht-Boden")->setMeshMat(matVideoBackground, true);

        // Add axis object a world origin
        SLNode* axis = new SLNode(new SLCoordAxis(s), "Axis Node");
        axis->setDrawBitsRec(SL_DB_MESHWIRED, false);
        axis->scale(10);
        axis->rotate(-90, 1, 0, 0);

        // Set some ambient light
        TheaterAndTempel->updateMeshMat([](SLMaterial* m) { m->ambient(SLCol4f(.25f, .23f, .15f)); }, true);
        SLNode* scene = new SLNode("Scene");
        scene->addChild(sunLight);
        scene->addChild(axis);
        scene->addChild(TheaterAndTempel);
        scene->addChild(cam1);

        //initialize sensor stuff
        SLApplication::devLoc.useOriginAltitude(false);                   // Use
        SLApplication::devLoc.originLatLonAlt(47.53319, 7.72207, 0);      // At the center of the theater
        SLApplication::devLoc.defaultLatLonAlt(47.5329758, 7.7210428, 0); // At the entrance of the tempel
        SLApplication::devLoc.locMaxDistanceM(1000.0f);                   // Max. allowed distance to origin
        SLApplication::devLoc.improveOrigin(false);                       // No autom. origin improvement
        SLApplication::devLoc.hasOrigin(true);
        SLApplication::devRot.zeroYawAtStart(false); // Use the real yaw from the IMU

        // This loads the DEM file and overwrites the altitude of originLatLonAlt and defaultLatLonAlt
        SLstring tif = SLApplication::dataPath + "erleb-AR/models/augst/DTM-Theater-Tempel-WGS84.tif";
        SLApplication::devLoc.loadGeoTiff(tif);

#if defined(SL_OS_MACIOS) || defined(SL_OS_ANDROID)
        SLApplication::devLoc.isUsed(true);
        SLApplication::devRot.isUsed(true);
        cam1->camAnim(SLCamAnim::CA_deviceRotLocYUp);
#else
        SLApplication::devLoc.isUsed(false);
        SLApplication::devRot.isUsed(false);
        SLVec3d pos_d = SLApplication::devLoc.defaultENU() - SLApplication::devLoc.originENU();
        SLVec3f pos_f((SLfloat)pos_d.x, (SLfloat)pos_d.y, (SLfloat)pos_d.z);
        cam1->translation(pos_f);
        cam1->focalDist(pos_f.length());
        cam1->lookAt(SLVec3f::ZERO);
        cam1->camAnim(SLCamAnim::CA_turntableYUp);
#endif

        sv->doWaitOnIdle(false); // for constant video feed
        sv->camera(cam1);
        s->root3D(scene);
    }
    else if (SLApplication::sceneID == SID_ErlebARAventicumAmphiAO) //...................................
    {
        s->name("Aventicum Amphitheatre AR (AO)");
        s->info("Augmented Reality for Aventicum Amphitheatre (AO)");

        SLCamera* cam1 = new SLCamera("Camera 1");
        cam1->translation(0, 50, -150);
        cam1->lookAt(0, 0, 0);
        cam1->clipNear(1);
        cam1->clipFar(300);
        cam1->focalDist(150);
        cam1->setInitialState();
        cam1->devRotLoc(&SLApplication::devRot, &SLApplication::devLoc);

        // Create video texture and turn on live video
        videoTexture = new SLGLTexture(s, SLApplication::texturePath + "LiveVideoError.png", GL_LINEAR, GL_LINEAR);
        cam1->background().texture(videoTexture);
        CVCapture::instance()->videoType(VT_MAIN);

        // Define shader that shows on all pixels the video background
        SLGLProgram* spVideoBackground  = new SLGLGenericProgram(s,
                                                                SLApplication::shaderPath + "PerVrtTextureBackground.vert",
                                                                SLApplication::shaderPath + "PerVrtTextureBackground.frag");
        SLMaterial*  matVideoBackground = new SLMaterial(s,
                                                        "matVideoBackground",
                                                        videoTexture,
                                                        nullptr,
                                                        nullptr,
                                                        nullptr,
                                                        spVideoBackground);

        // Create directional light for the sun light
        SLLightDirect* sunLight = new SLLightDirect(s, s, 5.0f);
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

        // Let the sun be rotated by time and location
        SLApplication::devLoc.sunLightNode(sunLight);

        SLAssimpImporter importer;
        SLNode*          amphiTheatre = importer.load(s->animManager(),
                                             s,
                                             SLApplication::dataPath + "erleb-AR/models/avenches/Aventicum-Amphitheater-AO.gltf",
                                             SLApplication::texturePath,
                                             true,    // only meshes
                                             nullptr, // no replacement material
                                             0.4f);   // 40% ambient reflection

        // Rotate to the true geographic rotation
        amphiTheatre->rotate(13.7f, 0, 1, 0, TS_parent);

        // Let the video shine through some objects
        amphiTheatre->findChild<SLNode>("Tht-Aussen-Untergrund")->setMeshMat(matVideoBackground, true);
        amphiTheatre->findChild<SLNode>("Tht-Eingang-Ost-Boden")->setMeshMat(matVideoBackground, true);
        amphiTheatre->findChild<SLNode>("Tht-Arenaboden")->setMeshMat(matVideoBackground, true);
        amphiTheatre->findChild<SLNode>("Tht-Aussen-Terrain")->setMeshMat(matVideoBackground, true);

        // Add axis object a world origin
        SLNode* axis = new SLNode(new SLCoordAxis(s), "Axis Node");
        axis->setDrawBitsRec(SL_DB_MESHWIRED, false);
        axis->scale(10);
        axis->rotate(-90, 1, 0, 0);

        SLNode* scene = new SLNode("Scene");
        scene->addChild(sunLight);
        scene->addChild(axis);
        scene->addChild(amphiTheatre);
        scene->addChild(cam1);

        //initialize sensor stuff
        SLApplication::devLoc.useOriginAltitude(false);
        SLApplication::devLoc.originLatLonAlt(46.881013677, 7.042621953, 442.0);        // Zentrum Amphitheater
        SLApplication::devLoc.defaultLatLonAlt(46.881210148, 7.043767122, 442.0 + 1.7); // Ecke Vorplatz Ost
        SLApplication::devLoc.locMaxDistanceM(1000.0f);                                 // Max. Distanz. zum Nullpunkt
        SLApplication::devLoc.improveOrigin(false);                                     // Keine autom. Verbesserung vom Origin
        SLApplication::devLoc.hasOrigin(true);
        SLApplication::devRot.zeroYawAtStart(false);

        // This loads the DEM file and overwrites the altitude of originLatLonAlt and defaultLatLonAlt
        SLstring tif = SLApplication::dataPath + "erleb-AR/models/avenches/DTM-Aventicum-WGS84.tif";
        SLApplication::devLoc.loadGeoTiff(tif);

#if defined(SL_OS_MACIOS) || defined(SL_OS_ANDROID)
        SLApplication::devLoc.isUsed(true);
        SLApplication::devRot.isUsed(true);
        cam1->camAnim(SLCamAnim::CA_deviceRotLocYUp);
#else
        SLApplication::devLoc.isUsed(false);
        SLApplication::devRot.isUsed(false);
        SLVec3d pos_d = SLApplication::devLoc.defaultENU() - SLApplication::devLoc.originENU();
        SLVec3f pos_f((SLfloat)pos_d.x, (SLfloat)pos_d.y, (SLfloat)pos_d.z);
        cam1->translation(pos_f);
        cam1->focalDist(pos_f.length());
        cam1->lookAt(SLVec3f::ZERO);
        cam1->camAnim(SLCamAnim::CA_turntableYUp);
#endif

        sv->doWaitOnIdle(false); // for constant video feed
        sv->camera(cam1);
        s->root3D(scene);
    }
    else if (SLApplication::sceneID == SID_ErlebARAventicumCigognier) //.................................
    {
        s->name("Aventicum Cigognier AR");
        s->info("Augmented Reality for Aventicum Cigognier Temple");

        SLCamera* cam1 = new SLCamera("Camera 1");
        cam1->translation(0, 50, -150);
        cam1->lookAt(0, 0, 0);
        cam1->clipNear(1);
        cam1->clipFar(400);
        cam1->focalDist(150);
        cam1->setInitialState();
        cam1->devRotLoc(&SLApplication::devRot, &SLApplication::devLoc);

        // Create video texture and turn on live video
        videoTexture = new SLGLTexture(s, SLApplication::texturePath + "LiveVideoError.png", GL_LINEAR, GL_LINEAR);
        cam1->background().texture(videoTexture);

        CVCapture::instance()->videoType(VT_MAIN);

        // Create directional light for the sun light
        SLLightDirect* sunLight = new SLLightDirect(s, s, 5.0f);
        sunLight->powers(1.0f, 1.5f, 1.0f);
        sunLight->attenuation(1, 0, 0);
        sunLight->translation(0, 10, 0);
        sunLight->lookAt(10, 0, 10);
        sunLight->doSunPowerAdaptation(true);
        sunLight->createsShadows(true);
        sunLight->createShadowMap(-100, 150, SLVec2f(150, 150), SLVec2i(2048, 2048));
        sunLight->doSmoothShadows(true);
        sunLight->castsShadows(false);

        // Let the sun be rotated by time and location
        SLApplication::devLoc.sunLightNode(sunLight);

        SLAssimpImporter importer;
        SLNode*          cigognier = importer.load(s->animManager(),
                                          s,
                                          SLApplication::dataPath + "erleb-AR/models/avenches/Aventicum-Cigognier2.gltf",
                                          SLApplication::texturePath,
                                          true,    // only meshes
                                          nullptr, // no replacement material
                                          0.4f);   // 40% ambient reflection

        cigognier->findChild<SLNode>("Tmp-Parois-Sud")->drawBits()->set(SL_DB_HIDDEN, true);

        // Rotate to the true geographic rotation
        cigognier->rotate(-37.0f, 0, 1, 0, TS_parent);

        // Setup shadow mapping material and replace shader from loader
        SLGLProgram* progPerPixNrmSM = new SLGLGenericProgram(s,
                                                              SLApplication::shaderPath + "PerPixBlinnTexNrmSM.vert",
                                                              SLApplication::shaderPath + "PerPixBlinnTexNrmSM.frag");
        auto         updateMat       = [=](SLMaterial* mat) { mat->program(progPerPixNrmSM); };
        cigognier->updateMeshMat(updateMat, true);

        // Add axis object a world origin
        SLNode* axis = new SLNode(new SLCoordAxis(s), "Axis Node");
        axis->setDrawBitsRec(SL_DB_MESHWIRED, false);
        axis->rotate(-90, 1, 0, 0);

        SLNode* scene = new SLNode("Scene");
        scene->addChild(sunLight);
        scene->addChild(axis);
        scene->addChild(cigognier);
        scene->addChild(cam1);

        //initialize sensor stuff
        SLApplication::devLoc.useOriginAltitude(false);
        //https://map.geo.admin.ch/?lang=de&topic=ech&bgLayer=ch.swisstopo.swissimage&layers=ch.swisstopo.zeitreihen,ch.bfs.gebaeude_wohnungs_register,ch.bav.haltestellen-oev,ch.swisstopo.swisstlm3d-wanderwege&layers_opacity=1,1,1,0.8&layers_visibility=false,false,false,false&layers_timestamp=18641231,,,&E=2570106&N=1192334&zoom=13&crosshair=marker
        SLApplication::devLoc.originLatLonAlt(46.88145, 7.04645, 450.9); // In the center of the place before the Cigognier temple
        //https://map.geo.admin.ch/?lang=de&topic=ech&bgLayer=ch.swisstopo.swissimage&layers=ch.swisstopo.zeitreihen,ch.bfs.gebaeude_wohnungs_register,ch.bav.haltestellen-oev,ch.swisstopo.swisstlm3d-wanderwege&layers_opacity=1,1,1,0.8&layers_visibility=false,false,false,false&layers_timestamp=18641231,,,&E=2570135&N=1192315&zoom=13&crosshair=marker
        SLApplication::devLoc.defaultLatLonAlt(46.88124, 7.04686, 451.5 + 1.7); // Before the entry if the Cigognier sanctuary
        SLApplication::devLoc.locMaxDistanceM(1000.0f);                         // Max. allowed distance from origin
        SLApplication::devLoc.improveOrigin(false);                             // No auto improvement from
        SLApplication::devLoc.hasOrigin(true);
        SLApplication::devRot.zeroYawAtStart(false);

        // This loads the DEM file and overwrites the altitude of originLatLonAlt and defaultLatLonAlt
        SLstring tif = SLApplication::dataPath + "erleb-AR/models/avenches/DTM-Aventicum-WGS84.tif";
        SLApplication::devLoc.loadGeoTiff(tif);

#if defined(SL_OS_MACIOS) || defined(SL_OS_ANDROID)
        SLApplication::devLoc.isUsed(true);
        SLApplication::devRot.isUsed(true);
        cam1->camAnim(SLCamAnim::CA_deviceRotLocYUp);
#else
        SLApplication::devLoc.isUsed(false);
        SLApplication::devRot.isUsed(false);
        SLVec3d pos_d = SLApplication::devLoc.defaultENU() - SLApplication::devLoc.originENU();
        SLVec3f pos_f((SLfloat)pos_d.x, (SLfloat)pos_d.y, (SLfloat)pos_d.z);
        cam1->translation(pos_f);
        cam1->focalDist(pos_f.length());
        cam1->lookAt(SLVec3f::ZERO);
        cam1->camAnim(SLCamAnim::CA_turntableYUp);
#endif

        sv->doWaitOnIdle(false); // for constant video feed
        sv->camera(cam1);
        s->root3D(scene);
    }
    else if (SLApplication::sceneID == SID_ErlebARAventicumCigognierAO) //...............................
    {
        s->name("Aventicum Cigognier AR (AO)");
        s->info("Augmented Reality for Aventicum Cigognier Temple");

        SLCamera* cam1 = new SLCamera("Camera 1");
        cam1->translation(0, 50, -150);
        cam1->lookAt(0, 0, 0);
        cam1->clipNear(1);
        cam1->clipFar(400);
        cam1->focalDist(150);
        cam1->setInitialState();
        cam1->devRotLoc(&SLApplication::devRot, &SLApplication::devLoc);

        // Create video texture and turn on live video
        videoTexture = new SLGLTexture(s, SLApplication::texturePath + "LiveVideoError.png", GL_LINEAR, GL_LINEAR);
        cam1->background().texture(videoTexture);

        CVCapture::instance()->videoType(VT_MAIN);

        // Create directional light for the sun light
        SLLightDirect* sunLight = new SLLightDirect(s, s, 5.0f);
        sunLight->powers(1.0f, 1.5f, 1.0f);
        sunLight->attenuation(1, 0, 0);
        sunLight->translation(0, 10, 0);
        sunLight->lookAt(10, 0, 10);
        sunLight->doSunPowerAdaptation(true);
        sunLight->createsShadows(true);
        sunLight->createShadowMap(-100, 150, SLVec2f(150, 150), SLVec2i(2048, 2048));
        sunLight->doSmoothShadows(true);
        sunLight->castsShadows(false);

        // Let the sun be rotated by time and location
        SLApplication::devLoc.sunLightNode(sunLight);

        SLAssimpImporter importer;
        SLNode*          cigognier = importer.load(s->animManager(),
                                          s,
                                          SLApplication::dataPath + "erleb-AR/models/avenches/Aventicum-Cigognier-AO.gltf",
                                          SLApplication::texturePath,
                                          true,    // only meshes
                                          nullptr, // no replacement material
                                          0.4f);   // 40% ambient reflection

        cigognier->findChild<SLNode>("Tmp-Parois-Sud")->drawBits()->set(SL_DB_HIDDEN, true);

        // Rotate to the true geographic rotation
        cigognier->rotate(-37.0f, 0, 1, 0, TS_parent);

        // Add axis object a world origin
        SLNode* axis = new SLNode(new SLCoordAxis(s), "Axis Node");
        axis->setDrawBitsRec(SL_DB_MESHWIRED, false);
        axis->rotate(-90, 1, 0, 0);

        SLNode* scene = new SLNode("Scene");
        scene->addChild(sunLight);
        scene->addChild(axis);
        scene->addChild(cigognier);
        scene->addChild(cam1);

        //initialize sensor stuff
        SLApplication::devLoc.useOriginAltitude(false);
        //https://map.geo.admin.ch/?lang=de&topic=ech&bgLayer=ch.swisstopo.swissimage&layers=ch.swisstopo.zeitreihen,ch.bfs.gebaeude_wohnungs_register,ch.bav.haltestellen-oev,ch.swisstopo.swisstlm3d-wanderwege&layers_opacity=1,1,1,0.8&layers_visibility=false,false,false,false&layers_timestamp=18641231,,,&E=2570106&N=1192334&zoom=13&crosshair=marker
        SLApplication::devLoc.originLatLonAlt(46.88145, 7.04645, 450.9); // In the center of the place before the Cigognier temple
        //https://map.geo.admin.ch/?lang=de&topic=ech&bgLayer=ch.swisstopo.swissimage&layers=ch.swisstopo.zeitreihen,ch.bfs.gebaeude_wohnungs_register,ch.bav.haltestellen-oev,ch.swisstopo.swisstlm3d-wanderwege&layers_opacity=1,1,1,0.8&layers_visibility=false,false,false,false&layers_timestamp=18641231,,,&E=2570135&N=1192315&zoom=13&crosshair=marker
        SLApplication::devLoc.defaultLatLonAlt(46.88124, 7.04686, 451.5 + 1.7); // Before the entry if the Cigognier sanctuary
        SLApplication::devLoc.locMaxDistanceM(1000.0f);                         // Max. allowed distance from origin
        SLApplication::devLoc.improveOrigin(false);                             // No auto improvement from
        SLApplication::devLoc.hasOrigin(true);
        SLApplication::devRot.zeroYawAtStart(false);

        // This loads the DEM file and overwrites the altitude of originLatLonAlt and defaultLatLonAlt
        SLstring tif = SLApplication::dataPath + "erleb-AR/models/avenches/DTM-Aventicum-WGS84.tif";
        SLApplication::devLoc.loadGeoTiff(tif);

#if defined(SL_OS_MACIOS) || defined(SL_OS_ANDROID)
        SLApplication::devLoc.isUsed(true);
        SLApplication::devRot.isUsed(true);
        cam1->camAnim(SLCamAnim::CA_deviceRotLocYUp);
#else
        SLApplication::devLoc.isUsed(false);
        SLApplication::devRot.isUsed(false);
        SLVec3d pos_d = SLApplication::devLoc.defaultENU() - SLApplication::devLoc.originENU();
        SLVec3f pos_f((SLfloat)pos_d.x, (SLfloat)pos_d.y, (SLfloat)pos_d.z);
        cam1->translation(pos_f);
        cam1->focalDist(pos_f.length());
        cam1->lookAt(SLVec3f::ZERO);
        cam1->camAnim(SLCamAnim::CA_turntableYUp);
#endif

        sv->doWaitOnIdle(false); // for constant video feed
        sv->camera(cam1);
        s->root3D(scene);
    }
    else if (SLApplication::sceneID == SID_ErlebARAventicumTheatre) //...................................
    {
        s->name("Aventicum Theatre AR");
        s->info("Augmented Reality for Aventicum Theatre");

        SLCamera* cam1 = new SLCamera("Camera 1");
        cam1->translation(0, 50, -150);
        cam1->lookAt(0, 0, 0);
        cam1->clipNear(1);
        cam1->clipFar(300);
        cam1->focalDist(150);
        cam1->setInitialState();
        cam1->devRotLoc(&SLApplication::devRot, &SLApplication::devLoc);

        // Create video texture and turn on live video
        videoTexture = new SLGLTexture(s, SLApplication::texturePath + "LiveVideoError.png", GL_LINEAR, GL_LINEAR);
        cam1->background().texture(videoTexture);
        CVCapture::instance()->videoType(VT_MAIN);

        // Define shader that shows on all pixels the video background
        SLGLProgram* spVideoBackground  = new SLGLGenericProgram(s,
                                                                SLApplication::shaderPath + "PerVrtTextureBackground.vert",
                                                                SLApplication::shaderPath + "PerVrtTextureBackground.frag");
        SLMaterial*  matVideoBackground = new SLMaterial(s,
                                                        "matVideoBackground",
                                                        videoTexture,
                                                        nullptr,
                                                        nullptr,
                                                        nullptr,
                                                        spVideoBackground);

        // Create directional light for the sun light
        SLLightDirect* sunLight = new SLLightDirect(s, s, 5.0f);
        sunLight->powers(1.0f, 1.0f, 1.0f);
        sunLight->attenuation(1, 0, 0);
        sunLight->translation(0, 10, 0);
        sunLight->lookAt(10, 0, 10);
        sunLight->doSunPowerAdaptation(true);
        sunLight->createsShadows(true);
        sunLight->createShadowMap(-100, 150, SLVec2f(150, 150), SLVec2i(2048, 2048));
        sunLight->doSmoothShadows(true);
        sunLight->castsShadows(false);

        // Let the sun be rotated by time and location
        SLApplication::devLoc.sunLightNode(sunLight);

        SLAssimpImporter importer;
        SLNode*          theatre = importer.load(s->animManager(),
                                        s,
                                        SLApplication::dataPath + "erleb-AR/models/avenches/Aventicum-Theater1.gltf",
                                        SLApplication::texturePath,
                                        true,    // only meshes
                                        nullptr, // no replacement material
                                        0.4f);   // 40% ambient reflection

        // Setup shadow mapping material and replace shader from loader
        SLGLProgram* progPerPixNrmSM = new SLGLGenericProgram(s,
                                                              SLApplication::shaderPath + "PerPixBlinnTexNrmSM.vert",
                                                              SLApplication::shaderPath + "PerPixBlinnTexNrmSM.frag");
        auto         updateMat       = [=](SLMaterial* mat) { mat->program(progPerPixNrmSM); };
        theatre->updateMeshMat(updateMat, true);

        // Rotate to the true geographic rotation
        theatre->rotate(-36.7f, 0, 1, 0, TS_parent);

        theatre->findChild<SLNode>("Tht-Buehnenhaus")->drawBits()->set(SL_DB_HIDDEN, true);

        // Let the video shine through some objects
        theatre->findChild<SLNode>("Tht-Rasen")->setMeshMat(matVideoBackground, true);
        theatre->findChild<SLNode>("Tht-Boden")->setMeshMat(matVideoBackground, true);

        // Add axis object a world origin
        SLNode* axis = new SLNode(new SLCoordAxis(s), "Axis Node");
        axis->setDrawBitsRec(SL_DB_MESHWIRED, false);
        axis->scale(10);
        axis->rotate(-90, 1, 0, 0);

        SLNode* scene = new SLNode("Scene");
        scene->addChild(sunLight);
        scene->addChild(axis);
        scene->addChild(theatre);
        scene->addChild(cam1);

        //initialize sensor stuff
        //https://map.geo.admin.ch/?lang=de&topic=ech&bgLayer=ch.swisstopo.swissimage&layers=ch.swisstopo.zeitreihen,ch.bfs.gebaeude_wohnungs_register,ch.bav.haltestellen-oev,ch.swisstopo.swisstlm3d-wanderwege&layers_opacity=1,1,1,0.8&layers_visibility=false,false,false,false&layers_timestamp=18641231,,,&E=2570281&N=1192204&zoom=13&crosshair=marker
        SLApplication::devLoc.useOriginAltitude(false);
        SLApplication::devLoc.originLatLonAlt(46.88029, 7.04876, 454.9f);        // Zentrum Orchestra
        SLApplication::devLoc.defaultLatLonAlt(46.88044, 7.04846, 455.3f + 1.7); // Vor dem Bühnenhaus
        SLApplication::devLoc.locMaxDistanceM(1000.0f);                          // Max. Distanz. zum Nullpunkt
        SLApplication::devLoc.improveOrigin(false);                              // Keine autom. Verbesserung vom Origin
        SLApplication::devLoc.hasOrigin(true);
        SLApplication::devRot.zeroYawAtStart(false);

        // This loads the DEM file and overwrites the altitude of originLatLonAlt and defaultLatLonAlt
        SLstring tif = SLApplication::dataPath + "erleb-AR/models/avenches/DTM-Aventicum-WGS84.tif";
        SLApplication::devLoc.loadGeoTiff(tif);

#if defined(SL_OS_MACIOS) || defined(SL_OS_ANDROID)
        SLApplication::devLoc.isUsed(true);
        SLApplication::devRot.isUsed(true);
        cam1->camAnim(SLCamAnim::CA_deviceRotLocYUp);
#else
        SLApplication::devLoc.isUsed(false);
        SLApplication::devRot.isUsed(false);
        SLVec3d pos_d = SLApplication::devLoc.defaultENU() - SLApplication::devLoc.originENU();
        SLVec3f pos_f((SLfloat)pos_d.x, (SLfloat)pos_d.y, (SLfloat)pos_d.z);
        cam1->translation(pos_f);
        cam1->focalDist(pos_f.length());
        cam1->lookAt(SLVec3f::ZERO);
        cam1->camAnim(SLCamAnim::CA_turntableYUp);
#endif

        sv->doWaitOnIdle(false); // for constant video feed
        sv->camera(cam1);
        s->root3D(scene);
    }
    else if (SLApplication::sceneID == SID_RTMuttenzerBox) //............................................
    {
        s->name("Muttenzer Box");
        s->info("Muttenzer Box with environment mapped reflective sphere and transparenz refractive glass sphere. Try ray tracing for real reflections and soft shadows.");

        // Create reflection & glass shaders
        SLGLProgram* sp1 = new SLGLGenericProgram(s, SLApplication::shaderPath + "Reflect.vert", SLApplication::shaderPath + "Reflect.frag");
        SLGLProgram* sp2 = new SLGLGenericProgram(s, SLApplication::shaderPath + "RefractReflect.vert", SLApplication::shaderPath + "RefractReflect.frag");

        // Create cube mapping texture
        SLGLTexture* tex1 = new SLGLTexture(s,
                                            SLApplication::texturePath + "MuttenzerBox+X0512_C.png",
                                            SLApplication::texturePath + "MuttenzerBox-X0512_C.png",
                                            SLApplication::texturePath + "MuttenzerBox+Y0512_C.png",
                                            SLApplication::texturePath + "MuttenzerBox-Y0512_C.png",
                                            SLApplication::texturePath + "MuttenzerBox+Z0512_C.png",
                                            SLApplication::texturePath + "MuttenzerBox-Z0512_C.png");

        SLCol4f lightEmisRGB(7.0f, 7.0f, 7.0f);
        SLCol4f grayRGB(0.75f, 0.75f, 0.75f);
        SLCol4f redRGB(0.75f, 0.25f, 0.25f);
        SLCol4f blueRGB(0.25f, 0.25f, 0.75f);
        SLCol4f blackRGB(0.00f, 0.00f, 0.00f);

        // create materials
        SLMaterial* cream = new SLMaterial(s, "cream", grayRGB, SLCol4f::BLACK, 0);
        SLMaterial* red   = new SLMaterial(s, "red", redRGB, SLCol4f::BLACK, 0);
        SLMaterial* blue  = new SLMaterial(s, "blue", blueRGB, SLCol4f::BLACK, 0);

        // Material for mirror sphere
        SLMaterial* refl = new SLMaterial(s, "refl", blackRGB, SLCol4f::WHITE, 1000, 1.0f);
        refl->textures().push_back(tex1);
        refl->program(sp1);

        // Material for glass sphere
        SLMaterial* refr = new SLMaterial(s, "refr", blackRGB, blackRGB, 100, 0.05f, 0.95f, 1.5f);
        refr->translucency(1000);
        refr->transmissive(SLCol4f::WHITE);
        refr->textures().push_back(tex1);
        refr->program(sp2);

        SLNode* sphere1 = new SLNode(new SLSphere(s, 0.5f, 32, 32, "Sphere1", refl));
        sphere1->translate(-0.65f, -0.75f, -0.55f, TS_object);

        SLNode* sphere2 = new SLNode(new SLSphere(s, 0.45f, 32, 32, "Sphere2", refr));
        sphere2->translate(0.73f, -0.8f, 0.10f, TS_object);

        SLNode* balls = new SLNode;
        balls->addChild(sphere1);
        balls->addChild(sphere2);

        // Rectangular light
        SLLightRect* lightRect = new SLLightRect(s, s, 1, 0.65f);
        lightRect->rotate(90, -1.0f, 0.0f, 0.0f);
        lightRect->translate(0.0f, -0.25f, 1.18f, TS_object);
        lightRect->spotCutOffDEG(90);
        lightRect->spotExponent(1.0);
        lightRect->ambientColor(SLCol4f::BLACK);
        lightRect->diffuseColor(lightEmisRGB);
        lightRect->attenuation(0, 0, 1);
        lightRect->samplesXY(11, 7);

        SLLight::globalAmbient.set(lightEmisRGB * 0.01f);

        // create camera
        SLCamera* cam1 = new SLCamera();
        cam1->translation(0.0f, 0.40f, 6.35f);
        cam1->lookAt(0.0f, -0.05f, 0.0f);
        cam1->fov(27);
        cam1->focalDist(cam1->translationOS().length());
        cam1->background().colors(SLCol4f(0.0f, 0.0f, 0.0f));
        cam1->setInitialState();
        cam1->devRotLoc(&SLApplication::devRot, &SLApplication::devLoc);

        // assemble scene
        SLNode* scene = new SLNode;
        scene->addChild(cam1);
        scene->addChild(lightRect);

        // create wall polygons
        SLfloat pL = -1.48f, pR = 1.48f; // left/right
        SLfloat pB = -1.25f, pT = 1.19f; // bottom/top
        SLfloat pN = 1.79f, pF = -1.55f; // near/far

        // bottom plane
        SLNode* b = new SLNode(new SLRectangle(s, SLVec2f(pL, -pN), SLVec2f(pR, -pF), 6, 6, "bottom", cream));
        b->rotate(90, -1, 0, 0);
        b->translate(0, 0, pB, TS_object);
        scene->addChild(b);

        // top plane
        SLNode* t = new SLNode(new SLRectangle(s, SLVec2f(pL, pF), SLVec2f(pR, pN), 6, 6, "top", cream));
        t->rotate(90, 1, 0, 0);
        t->translate(0, 0, -pT, TS_object);
        scene->addChild(t);

        // far plane
        SLNode* f = new SLNode(new SLRectangle(s, SLVec2f(pL, pB), SLVec2f(pR, pT), 6, 6, "far", cream));
        f->translate(0, 0, pF, TS_object);
        scene->addChild(f);

        // // near plane
        // SLNode* n = new SLNode(new SLRectangle(SLVec2f(pL, pT), SLVec2f(pR, pB), 6, 6, "near", cream));
        // n->translate(0, 0, pN, TS_object);
        // scene->addChild(n);

        // left plane
        SLNode* l = new SLNode(new SLRectangle(s, SLVec2f(-pN, pB), SLVec2f(-pF, pT), 6, 6, "left", red));
        l->rotate(90, 0, 1, 0);
        l->translate(0, 0, pL, TS_object);
        scene->addChild(l);

        // right plane
        SLNode* r = new SLNode(new SLRectangle(s, SLVec2f(pF, pB), SLVec2f(pN, pT), 6, 6, "right", blue));
        r->rotate(90, 0, -1, 0);
        r->translate(0, 0, -pR, TS_object);
        scene->addChild(r);

        scene->addChild(balls);

        sv->camera(cam1);
        s->root3D(scene);
    }
    else if (SLApplication::sceneID == SID_RTSpheres) //.................................................
    {
        s->name("Ray tracing Spheres");
        s->info("Classic ray tracing scene with transparent and reflective spheres. Be patient on mobile devices.");

        // define materials
        SLMaterial* matGla = new SLMaterial(s, "Glass", SLCol4f(0.0f, 0.0f, 0.0f), SLCol4f(0.5f, 0.5f, 0.5f), 100, 0.4f, 0.6f, 1.5f);
        SLMaterial* matRed = new SLMaterial(s, "Red", SLCol4f(0.5f, 0.0f, 0.0f), SLCol4f(0.5f, 0.5f, 0.5f), 100, 0.5f, 0.0f, 1.0f);
        SLMaterial* matYel = new SLMaterial(s, "Floor", SLCol4f(0.8f, 0.6f, 0.2f), SLCol4f(0.8f, 0.8f, 0.8f), 100, 0.5f, 0.0f, 1.0f);

        SLCamera* cam1 = new SLCamera();
        cam1->translation(0, 0.1f, 2.5f);
        cam1->lookAt(0, 0, 0);
        cam1->focalDist(cam1->translationOS().length());
        cam1->background().colors(SLCol4f(0.1f, 0.4f, 0.8f));
        cam1->setInitialState();
        cam1->devRotLoc(&SLApplication::devRot, &SLApplication::devLoc);

        SLNode* rect = new SLNode(new SLRectangle(s, SLVec2f(-3, -3), SLVec2f(5, 4), 20, 20, "Floor", matYel));
        rect->rotate(90, -1, 0, 0);
        rect->translate(0, -1, -0.5f, TS_object);

        SLLightSpot* light1 = new SLLightSpot(s, s, 2, 2, 2, 0.1f);
        light1->powers(1, 7, 7);
        light1->attenuation(0, 0, 1);

        SLLightSpot* light2 = new SLLightSpot(s, s, 2, 2, -2, 0.1f);
        light2->powers(1, 7, 7);
        light2->attenuation(0, 0, 1);

        SLNode* scene = new SLNode;
        scene->addChild(light1);
        scene->addChild(light2);
        scene->addChild(SphereGroup(s, 3, 0, 0, 0, 1, 30, matGla, matRed));
        scene->addChild(rect);
        scene->addChild(cam1);

        s->root3D(scene);
        sv->camera(cam1);
    }
    else if (SLApplication::sceneID == SID_RTSoftShadows) //.............................................
    {
        s->name("Ray tracing soft shadows");
        s->info("Ray tracing with soft shadow light sampling. Each light source is sampled 64x per pixel. Be patient on mobile devices.");

        // define materials
        SLCol4f      spec(0.8f, 0.8f, 0.8f);
        SLGLProgram* shadowPrg = new SLGLGenericProgram(s,
                                                        SLApplication::shaderPath + "PerPixBlinnSM.vert",
                                                        SLApplication::shaderPath + "PerPixBlinnSM.frag");
        SLMaterial*  matBlk    = new SLMaterial(s, "Glass", SLCol4f(0.0f, 0.0f, 0.0f), SLCol4f(0.5f, 0.5f, 0.5f), 100, 0.5f, 0.5f, 1.5f, shadowPrg);
        SLMaterial*  matRed    = new SLMaterial(s, "Red", SLCol4f(0.5f, 0.0f, 0.0f), SLCol4f(0.5f, 0.5f, 0.5f), 100, 0.5f, 0.0f, 1.0f, shadowPrg);
        SLMaterial*  matYel    = new SLMaterial(s, "Floor", SLCol4f(0.8f, 0.6f, 0.2f), SLCol4f(0.8f, 0.8f, 0.8f), 100, 0.0f, 0.0f, 1.0f, shadowPrg);

        SLCamera* cam1 = new SLCamera;
        cam1->translation(0, 0.1f, 4);
        cam1->lookAt(0, 0, 0);
        cam1->focalDist(cam1->translationOS().length());
        cam1->background().colors(SLCol4f(0.1f, 0.4f, 0.8f));
        cam1->setInitialState();
        cam1->devRotLoc(&SLApplication::devRot, &SLApplication::devLoc);

        SLNode* rect = new SLNode(new SLRectangle(s, SLVec2f(-5, -5), SLVec2f(5, 5), 1, 1, "Rect", matYel));
        rect->rotate(90, -1, 0, 0);
        rect->translate(0, 0, -0.5f);
        rect->castsShadows(false);

        SLLightSpot* light1 = new SLLightSpot(s, s, 2, 2, 2, 0.3f);
        light1->samples(8, 8);
        light1->attenuation(0, 0, 1);
        light1->createsShadows(true);
        light1->createShadowMap();

        SLLightSpot* light2 = new SLLightSpot(s, s, 2, 2, -2, 0.3f);
        light2->samples(8, 8);
        light2->attenuation(0, 0, 1);
        light2->createsShadows(true);
        light2->createShadowMap();

        SLNode* scene = new SLNode;
        scene->addChild(light1);
        scene->addChild(light2);
        scene->addChild(SphereGroup(s, 1, 0, 0, 0, 1, 32, matBlk, matRed));
        scene->addChild(rect);
        scene->addChild(cam1);

        sv->camera(cam1);
        s->root3D(scene);
    }
    else if (SLApplication::sceneID == SID_RTDoF) //.....................................................
    {
        s->name("Ray tracing depth of field");

        SLGLProgram* p1 = new SLGLGenericProgram(s,
                                                 SLApplication::shaderPath + "PerPixBlinnTex.vert",
                                                 SLApplication::shaderPath + "PerPixBlinnTex.frag");

        // Create textures and materials
        SLGLTexture* texC = new SLGLTexture(s, SLApplication::texturePath + "Checkerboard0512_C.png", SL_ANISOTROPY_MAX, GL_LINEAR);
        SLMaterial*  mT   = new SLMaterial(s, "mT", texC, nullptr, nullptr, nullptr, p1);
        mT->kr(0.5f);
        SLMaterial* mW = new SLMaterial(s, "mW", SLCol4f::WHITE);
        SLMaterial* mB = new SLMaterial(s, "mB", SLCol4f::GRAY);
        SLMaterial* mY = new SLMaterial(s, "mY", SLCol4f::YELLOW);
        SLMaterial* mR = new SLMaterial(s, "mR", SLCol4f::RED);
        SLMaterial* mG = new SLMaterial(s, "mG", SLCol4f::GREEN);
        SLMaterial* mM = new SLMaterial(s, "mM", SLCol4f::MAGENTA);

#ifndef SL_GLES
        SLuint numSamples = 10;
#else
        SLuint numSamples = 4;
#endif

        stringstream ss;
        ss << "Ray tracing with depth of field blur. Each pixel is sampled " << numSamples * numSamples << "x from a lens. Be patient on mobile devices.";
        s->info(ss.str());

        SLCamera* cam1 = new SLCamera("Camera 1");
        cam1->translation(0, 2, 7);
        cam1->lookAt(0, 0, 0);
        cam1->focalDist(cam1->translationOS().length());
        cam1->clipFar(80);
        cam1->lensDiameter(0.4f);
        cam1->lensSamples()->samples(numSamples, numSamples);
        cam1->background().colors(SLCol4f(0.1f, 0.4f, 0.8f));
        cam1->setInitialState();
        cam1->fogIsOn(true);
        cam1->fogMode(FM_exp);
        cam1->fogDensity(0.04f);

        SLuint  res  = 36;
        SLNode* rect = new SLNode(new SLRectangle(s,
                                                  SLVec2f(-40, -10),
                                                  SLVec2f(40, 70),
                                                  SLVec2f(0, 0),
                                                  SLVec2f(4, 4),
                                                  2,
                                                  2,
                                                  "Rect",
                                                  mT));
        rect->rotate(90, -1, 0, 0);
        rect->translate(0, 0, -0.5f, TS_object);

        SLLightSpot* light1 = new SLLightSpot(s, s, 2, 2, 0, 0.1f);
        light1->ambiDiffPowers(0.1f, 1);
        light1->attenuation(1, 0, 0);

        SLNode* balls = new SLNode;
        SLNode* sp;
        sp = new SLNode(new SLSphere(s, 0.5f, res, res, "S1", mW));
        sp->translate(2.0, 0, -4, TS_object);
        balls->addChild(sp);
        sp = new SLNode(new SLSphere(s, 0.5f, res, res, "S2", mB));
        sp->translate(1.5, 0, -3, TS_object);
        balls->addChild(sp);
        sp = new SLNode(new SLSphere(s, 0.5f, res, res, "S3", mY));
        sp->translate(1.0, 0, -2, TS_object);
        balls->addChild(sp);
        sp = new SLNode(new SLSphere(s, 0.5f, res, res, "S4", mR));
        sp->translate(0.5, 0, -1, TS_object);
        balls->addChild(sp);
        sp = new SLNode(new SLSphere(s, 0.5f, res, res, "S5", mG));
        sp->translate(0.0, 0, 0, TS_object);
        balls->addChild(sp);
        sp = new SLNode(new SLSphere(s, 0.5f, res, res, "S6", mM));
        sp->translate(-0.5, 0, 1, TS_object);
        balls->addChild(sp);
        sp = new SLNode(new SLSphere(s, 0.5f, res, res, "S7", mW));
        sp->translate(-1.0, 0, 2, TS_object);
        balls->addChild(sp);

        SLNode* scene = new SLNode;
        scene->addChild(light1);
        scene->addChild(balls);
        scene->addChild(rect);
        scene->addChild(cam1);

        sv->camera(cam1);
        s->root3D(scene);
    }
    else if (SLApplication::sceneID == SID_RTLens) //....................................................
    {
        s->name("Ray tracing lens test");
        s->info("Ray tracing lens test scene.");

        // Create textures and materials
        SLGLTexture* texC = new SLGLTexture(s, SLApplication::texturePath + "VisionExample.jpg");
        //SLGLTexture* texC = new SLGLTexture(s, SLApplication::texturePath + "Checkerboard0512_C.png");

        SLMaterial* mT = new SLMaterial(s, "mT", texC, nullptr, nullptr, nullptr);
        mT->kr(0.5f);

        // Glass material
        // name, ambient, specular,	shininess, kr(reflectivity), kt(transparency), kn(refraction)
        SLMaterial* matLens = new SLMaterial(s, "lens", SLCol4f(0.0f, 0.0f, 0.0f), SLCol4f(0.5f, 0.5f, 0.5f), 100, 0.5f, 0.5f, 1.5f);
        //SLGLShaderProg* sp1 = new SLGLShaderProgGeneric("RefractReflect.vert", "RefractReflect.frag");
        //matLens->shaderProg(sp1);

#ifndef APP_USES_GLES
        SLuint numSamples = 10;
#else
        SLuint numSamples = 6;
#endif

        // Scene
        SLCamera* cam1 = new SLCamera;
        cam1->translation(0, 8, 0);
        cam1->lookAt(0, 0, 0);
        cam1->focalDist(cam1->translationOS().length());
        cam1->lensDiameter(0.4f);
        cam1->lensSamples()->samples(numSamples, numSamples);
        cam1->background().colors(SLCol4f(0.1f, 0.4f, 0.8f));
        cam1->setInitialState();
        cam1->devRotLoc(&SLApplication::devRot, &SLApplication::devLoc);

        // Light
        //SLLightSpot* light1 = new SLLightSpot(s,s,15, 20, 15, 0.1f);
        //light1->attenuation(0, 0, 1);

        // Plane
        //SLNode* rect = new SLNode(new SLRectangle(SLVec2f(-20, -20), SLVec2f(20, 20), 50, 20, "Rect", mT));
        //rect->translate(0, 0, 0, TS_Object);
        //rect->rotate(90, -1, 0, 0);

        SLLightSpot* light1 = new SLLightSpot(s, s, 1, 6, 1, 0.1f);
        light1->attenuation(0, 0, 1);

        SLuint  res  = 20;
        SLNode* rect = new SLNode(new SLRectangle(s, SLVec2f(-5, -5), SLVec2f(5, 5), res, res, "Rect", mT));
        rect->rotate(90, -1, 0, 0);
        rect->translate(0, 0, -0.0f, TS_object);

        // Lens from eye prescription card
        //SLNode* lensA = new SLNode(new SLLens(s, 0.50f, -0.50f, 4.0f, 0.0f, 32, 32, "presbyopic", matLens));   // Weitsichtig
        //lensA->translate(-2, 1, -2);
        //SLNode* lensB = new SLNode(new SLLens(s, -0.65f, -0.10f, 4.0f, 0.0f, 32, 32, "myopic", matLens));      // Kurzsichtig
        //lensB->translate(2, 1, -2);

        // Lens with radius
        //SLNode* lensC = new SLNode(new SLLens(s, 5.0, 4.0, 4.0f, 0.0f, 32, 32, "presbyopic", matLens));        // Weitsichtig
        //lensC->translate(-2, 1, 2);
        SLNode* lensD = new SLNode(new SLLens(s,
                                              -15.0f,
                                              -15.0f,
                                              1.0f,
                                              0.1f,
                                              res,
                                              res,
                                              "myopic",
                                              matLens)); // Kurzsichtig
        lensD->translate(0, 6, 0);

        // Node
        SLNode* scene = new SLNode;
        //scene->addChild(lensA);
        //scene->addChild(lensB);
        //scene->addChild(lensC);
        scene->addChild(lensD);
        scene->addChild(rect);
        scene->addChild(light1);
        scene->addChild(cam1);

        sv->camera(cam1);
        s->root3D(scene);
    }
    else if (SLApplication::sceneID == SID_RTTest) //....................................................
    {
        // Set scene name and info string
        s->name("Ray tracing test");
        s->info("RT Test Scene");

        // Create a camera node
        SLCamera* cam1 = new SLCamera("Camera 1");
        cam1->translation(0, 0, 5);
        cam1->lookAt(0, 0, 0);
        cam1->background().colors(SLCol4f(0.5f, 0.5f, 0.5f));
        cam1->setInitialState();
        cam1->devRotLoc(&SLApplication::devRot, &SLApplication::devLoc);

        // Create a light source node
        SLLightSpot* light1 = new SLLightSpot(s, s, 0.3f);
        light1->translation(5, 5, 5);
        light1->lookAt(0, 0, 0);
        light1->name("light node");

        // Material for glass sphere
        SLMaterial* matBox1  = new SLMaterial(s, "matBox1", SLCol4f(0.0f, 0.0f, 0.0f), SLCol4f(0.5f, 0.5f, 0.5f), 100, 0.0f, 0.9f, 1.5f);
        SLMesh*     boxMesh1 = new SLBox(s, -0.8f, -1, 0.02f, 1.2f, 1, 1, "boxMesh1", matBox1);
        SLNode*     boxNode1 = new SLNode(boxMesh1, "BoxNode1");

        SLMaterial* matBox2  = new SLMaterial(s, "matBox2", SLCol4f(0.0f, 0.0f, 0.0f), SLCol4f(0.5f, 0.5f, 0.5f), 100, 0.0f, 0.9f, 1.3f);
        SLMesh*     boxMesh2 = new SLBox(s, -1.2f, -1, -1, 0.8f, 1, -0.02f, "BoxMesh2", matBox2);
        SLNode*     boxNode2 = new SLNode(boxMesh2, "BoxNode2");

        // Create a scene group and add all nodes
        SLNode* scene = new SLNode("scene node");
        scene->addChild(light1);
        scene->addChild(cam1);
        scene->addChild(boxNode1);
        scene->addChild(boxNode2);

        s->root3D(scene);

        // Set active camera
        sv->camera(cam1);
    }

    ////////////////////////////////////////////////////////////////////////////
    // call onInitialize on all scene views to init the scenegraph and stats
    for (auto* sceneView : SLApplication::sceneViews)
        if (sceneView != nullptr)
            sceneView->onInitialize();

    if (CVCapture::instance()->videoType() != VT_NONE)
    {
        if (sv->viewportSameAsVideo())
        {
            // Pass a negative value to the start function, so that the
            // viewport aspect ratio can be adapted later to the video aspect.
            // This will be know after start.
            CVCapture::instance()->start(-1.0f);
            SLVec2i videoAspect;
            videoAspect.x = CVCapture::instance()->captureSize.width;
            videoAspect.y = CVCapture::instance()->captureSize.height;
            sv->setViewportFromRatio(videoAspect, sv->viewportAlign(), true);
        }
        else
            CVCapture::instance()->start(sv->viewportWdivH());
    }
    s->loadTimeMS(GlobalTimer::timeMS() - startLoadMS);
}
//-----------------------------------------------------------------------------
