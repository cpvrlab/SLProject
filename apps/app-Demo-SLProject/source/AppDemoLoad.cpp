//#############################################################################
//  File:      AppDemoSceneLoad.cpp
//  Author:    Marcus Hudritsch
//  Date:      Februar 2018
//  Codestyle: https://github.com/cpvrlab/SLProject/wiki/Coding-Style-Guidelines
//  Copyright: Marcus Hudritsch
//             This software is provide under the GNU General Public License
//             Please visit: http://opensource.org/licenses/GPL-3.0
//#############################################################################

#include <stdafx.h> // Must be the 1st include followed by  an empty line

#ifdef SL_MEMLEAKDETECT    // set in SL.h for debug config only
#    include <debug_new.h> // memory leak detector
#endif

#include <SLApplication.h>
#include <SLAssimpImporter.h>
#include <SLScene.h>
#include <SLSceneView.h>

#include <SLBox.h>
#include <SLCVCapture.h>
#include <SLCVTrackedAruco.h>
#include <SLCVTrackedChessboard.h>
#include <SLCVTrackedFaces.h>
#include <SLCVTrackedFeatures.h>
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
#include <SLTransferFunction.h>

//-----------------------------------------------------------------------------
// Foreward declarations for helper functions used only in this file
SLNode* SphereGroup(SLint, SLfloat, SLfloat, SLfloat, SLfloat, SLuint, SLMaterial*, SLMaterial*);
SLNode* BuildFigureGroup(SLMaterial* mat, SLbool withAnimation = false);

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
void appDemoLoadScene(SLScene* s, SLSceneView* sv, SLSceneID sceneID)
{
    SLApplication::sceneID = sceneID;

    // Initialize all preloaded stuff from SLScene
    s->init();

    if (SLApplication::sceneID == SID_Empty) //.....................................................
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

        // Create textures and materials
        SLGLTexture* texC = new SLGLTexture("earth1024_C.jpg");
        SLMaterial*  m1   = new SLMaterial("m1", texC);

        // Create a scene group node
        SLNode* scene = new SLNode("scene node");

        // Create a light source node
        SLLightSpot* light1 = new SLLightSpot(0.3f);
        light1->translation(0, 0, 5);
        light1->lookAt(0, 0, 0);
        light1->name("light node");
        scene->addChild(light1);

        // Create meshes and nodes
        SLMesh* rectMesh = new SLRectangle(SLVec2f(-5, -5), SLVec2f(5, 5), 1, 1, "rectangle mesh", m1);
        SLNode* rectNode = new SLNode(rectMesh, "rectangle node");
        scene->addChild(rectNode);

        SLNode* axisNode = new SLNode(new SLCoordAxis(), "axis node");
        scene->addChild(axisNode);

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
        s->info("Hierarchical scenegraph with multiple subgroups.");

        // Create textures and materials
        SLMaterial* m1 = new SLMaterial("m1", SLCol4f::BLACK, SLCol4f::WHITE, 128, 0.2f, 0.8f, 1.5f);
        SLMaterial* m2 = new SLMaterial("m2", SLCol4f::WHITE * 0.3f, SLCol4f::WHITE, 128, 0.5f, 0.0f, 1.0f);

        SLuint  res         = 20;
        SLMesh* rectangle   = new SLRectangle(SLVec2f(-5, -5), SLVec2f(5, 5), res, res, "rectangle", m2);
        SLNode* floorRect   = new SLNode(rectangle);
        SLNode* ceilingRect = new SLNode(rectangle);
        floorRect->rotate(90, -1, 0, 0);
        floorRect->translate(0, 0, -5.5f);
        ceilingRect->rotate(90, 1, 0, 0);
        ceilingRect->translate(0, 0, -5.5f);

        SLCamera* cam1 = new SLCamera("Camera 1");
        cam1->translation(0, 0, 22);
        cam1->lookAt(0, 0, 0);
        cam1->focalDist(22);
        cam1->background().colors(SLCol4f(0.1f, 0.4f, 0.8f));
        cam1->setInitialState();

        SLLightSpot* light1 = new SLLightSpot(5, 0, 5, 0.5f);
        light1->ambient(SLCol4f(0.2f, 0.2f, 0.2f));
        light1->diffuse(SLCol4f(0.9f, 0.9f, 0.9f));
        light1->specular(SLCol4f(0.9f, 0.9f, 0.9f));
        light1->attenuation(1, 0, 0);

        SLNode* figure = BuildFigureGroup(m1);

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
        s->info("3D file import test for: 3DS, DAE & FBX");

        SLMaterial* matBlu = new SLMaterial("Blue", SLCol4f(0, 0, 0.2f), SLCol4f(1, 1, 1), 100, 0.8f, 0);
        SLMaterial* matRed = new SLMaterial("Red", SLCol4f(0.2f, 0, 0), SLCol4f(1, 1, 1), 100, 0.8f, 0);
        SLMaterial* matGre = new SLMaterial("Green", SLCol4f(0, 0.2f, 0), SLCol4f(1, 1, 1), 100, 0.8f, 0);
        SLMaterial* matGra = new SLMaterial("Gray", SLCol4f(0.3f, 0.3f, 0.3f), SLCol4f(1, 1, 1), 100, 0, 0);

        SLCamera* cam1 = new SLCamera("Camera 1");
        cam1->clipNear(.1f);
        cam1->clipFar(30);
        cam1->translation(0, 0, 12);
        cam1->lookAt(0, 0, 0);
        cam1->maxSpeed(20);
        cam1->moveAccel(160);
        cam1->brakeAccel(160);
        cam1->focalDist(12);
        cam1->eyeSeparation(cam1->focalDist() / 30.0f);
        cam1->background().colors(SLCol4f(0.6f, 0.6f, 0.6f), SLCol4f(0.3f, 0.3f, 0.3f));
        cam1->setInitialState();

        SLLightSpot* light1 = new SLLightSpot(2.5f, 2.5f, 2.5f, 0.2f);
        light1->ambient(SLCol4f(0.1f, 0.1f, 0.1f));
        light1->diffuse(SLCol4f(1.0f, 1.0f, 1.0f));
        light1->specular(SLCol4f(1.0f, 1.0f, 1.0f));
        light1->attenuation(1, 0, 0);
        SLAnimation* anim = SLAnimation::create("anim_light1_backforth", 2.0f, true, EC_inOutQuad, AL_pingPongLoop);
        anim->createSimpleTranslationNodeTrack(light1, SLVec3f(0.0f, 0.0f, -5.0f));

        SLLightSpot* light2 = new SLLightSpot(-2.5f, -2.5f, 2.5f, 0.2f);
        light2->ambient(SLCol4f(0.1f, 0.1f, 0.1f));
        light2->diffuse(SLCol4f(1.0f, 1.0f, 1.0f));
        light2->specular(SLCol4f(1.0f, 1.0f, 1.0f));
        light2->attenuation(1, 0, 0);
        anim = SLAnimation::create("anim_light2_updown", 2.0f, true, EC_inOutQuint, AL_pingPongLoop);
        anim->createSimpleTranslationNodeTrack(light2, SLVec3f(0.0f, 5.0f, 0.0f));

        SLAssimpImporter importer;
        SLNode*          mesh3DS = importer.load("3DS/Halloween/jackolan.3ds");
        SLNode*          meshFBX = importer.load("FBX/Duck/duck.fbx");
        SLNode*          meshDAE = importer.load("DAE/AstroBoy/AstroBoy.dae");

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
        rb          = new SLNode(new SLRectangle(SLVec2f(-b, -b), SLVec2f(b, b), res, res, "rectB", matBlu), "rectBNode");
        rb->translate(0, 0, -b, TS_object);
        rl = new SLNode(new SLRectangle(SLVec2f(-b, -b), SLVec2f(b, b), res, res, "rectL", matRed), "rectLNode");
        rl->rotate(90, 0, 1, 0);
        rl->translate(0, 0, -b, TS_object);
        rr = new SLNode(new SLRectangle(SLVec2f(-b, -b), SLVec2f(b, b), res, res, "rectR", matGre), "rectRNode");
        rr->rotate(-90, 0, 1, 0);
        rr->translate(0, 0, -b, TS_object);
        rf = new SLNode(new SLRectangle(SLVec2f(-b, -b), SLVec2f(b, b), res, res, "rectF", matGra), "rectFNode");
        rf->rotate(-90, 1, 0, 0);
        rf->translate(0, 0, -b, TS_object);
        rt = new SLNode(new SLRectangle(SLVec2f(-b, -b), SLVec2f(b, b), res, res, "rectT", matGra), "rectTNode");
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
        SLGLTexture* tex1 = new SLGLTexture("Testmap_0512_C.png");
        SLMaterial*  mat1 = new SLMaterial("mat1", tex1);

        // floor material
        SLGLTexture* tex2 = new SLGLTexture("wood0_0512_C.jpg");
        SLMaterial*  mat2 = new SLMaterial("mat2", tex2);
        mat2->specular(SLCol4f::BLACK);

        // Back wall material
        SLGLTexture* tex3 = new SLGLTexture("bricks1_0256_C.jpg");
        SLMaterial*  mat3 = new SLMaterial("mat3", tex3);
        mat3->specular(SLCol4f::BLACK);

        // Left wall material
        SLGLTexture* tex4 = new SLGLTexture("wood2_0512_C.jpg");
        SLMaterial*  mat4 = new SLMaterial("mat4", tex4);
        mat4->specular(SLCol4f::BLACK);

        // Glass material
        SLGLTexture* tex5 = new SLGLTexture("wood2_0256_C.jpg", "wood2_0256_C.jpg", "gray_0256_C.jpg", "wood0_0256_C.jpg", "gray_0256_C.jpg", "bricks1_0256_C.jpg");
        SLMaterial*  mat5 = new SLMaterial("glass", SLCol4f::BLACK, SLCol4f::WHITE, 255, 0.1f, 0.9f, 1.5f);
        mat5->textures().push_back(tex5);
        SLGLProgram* sp1 = new SLGLGenericProgram("RefractReflect.vert", "RefractReflect.frag");
        mat5->program(sp1);

        // Wine material
        SLMaterial* mat6 = new SLMaterial("wine", SLCol4f(0.4f, 0.0f, 0.2f), SLCol4f::BLACK, 255, 0.2f, 0.7f, 1.3f);
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

        // light
        SLLightSpot* light1 = new SLLightSpot(0, 4, 0, 0.3f);
        light1->diffuse(SLCol4f(1, 1, 1));
        light1->ambient(SLCol4f(0.2f, 0.2f, 0.2f));
        light1->specular(SLCol4f(1, 1, 1));
        light1->attenuation(1, 0, 0);
        SLAnimation* anim = SLAnimation::create("light1_anim", 4.0f);
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
        SLNode* glass = new SLNode(new SLRevolver(revG, SLVec3f(0, 1, 0), res, true, false, "GlassRev", mat5));
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
        SLMesh* wineMesh = new SLRevolver(revW, SLVec3f(0, 1, 0), res, true, false, "WineRev", mat6);
        wineMesh->matOut(mat5);
        SLNode* wine = new SLNode(wineMesh);
        wine->translate(0.0f, -3.5f, 0.0f, TS_object);

        // wine fluid top
        SLNode* wineTop = new SLNode(new SLDisk(2.05f, -SLVec3f::AXISY, res, false, "WineRevTop", mat6));
        wineTop->translate(0.0f, 2.5f, 0.0f, TS_object);

        // Other revolver objects
        SLNode* sphere = new SLNode(new SLSphere(1, 16, 16, "sphere", mat1));
        sphere->translate(3, 0, 0, TS_object);
        SLNode* cylinder = new SLNode(new SLCylinder(0.1f, 7, 3, 16, true, true, "cylinder", mat1));
        cylinder->translate(0, 0.5f, 0);
        cylinder->rotate(90, -1, 0, 0);
        cylinder->rotate(30, 0, 1, 0);
        SLNode* cone = new SLNode(new SLCone(1, 3, 3, 16, true, "cone", mat1));
        cone->translate(-3, -1, 0, TS_object);
        cone->rotate(90, -1, 0, 0);

        // Cube dimensions
        SLfloat pL = -9.0f, pR = 9.0f;  // left/right
        SLfloat pB = -3.5f, pT = 14.5f; // bottom/top
        SLfloat pN = 9.0f, pF = -9.0f;  // near/far

        //// bottom rectangle
        SLNode* b = new SLNode(new SLRectangle(SLVec2f(pL, -pN), SLVec2f(pR, -pF), 10, 10, "PolygonFloor", mat2));
        b->rotate(90, -1, 0, 0);
        b->translate(0, 0, pB, TS_object);

        // top rectangle
        SLNode* t = new SLNode(new SLRectangle(SLVec2f(pL, pF), SLVec2f(pR, pN), 10, 10, "top", mat2));
        t->rotate(90, 1, 0, 0);
        t->translate(0, 0, -pT, TS_object);

        // far rectangle
        SLNode* f = new SLNode(new SLRectangle(SLVec2f(pL, pB), SLVec2f(pR, pT), 10, 10, "far", mat3));
        f->translate(0, 0, pF, TS_object);

        // left rectangle
        SLNode* l = new SLNode(new SLRectangle(SLVec2f(-pN, pB), SLVec2f(-pF, pT), 10, 10, "left", mat4));
        l->rotate(90, 0, 1, 0);
        l->translate(0, 0, pL, TS_object);

        // right rectangle
        SLNode* r = new SLNode(new SLRectangle(SLVec2f(pF, pB), SLVec2f(pN, pT), 10, 10, "right", mat4));
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
        s->name("Large Model Test");
        s->info("Large Model with 7.2 mio. triangles.");

        SLCamera* cam1 = new SLCamera("Camera 1");
        cam1->translation(0, 0, 600000);
        cam1->lookAt(0, 0, 0);
        cam1->clipNear(20);
        cam1->clipFar(1000000);
        cam1->background().colors(SLCol4f(0.5f, 0.5f, 0.5f));
        cam1->setInitialState();

        SLLightSpot* light1 = new SLLightSpot(600000, 600000, 600000, 1);
        light1->ambient(SLCol4f(1, 1, 1));
        light1->diffuse(SLCol4f(1, 1, 1));
        light1->specular(SLCol4f(1, 1, 1));
        light1->attenuation(1, 0, 0);

        SLAssimpImporter importer;
        SLNode*          largeModel = importer.load("PLY/switzerland.ply", true, nullptr
                                           //,SLProcess_JoinIdenticalVertices
                                           //|SLProcess_RemoveRedundantMaterials
                                           //|SLProcess_SortByPType
                                           //|SLProcess_FindDegenerates
                                           //|SLProcess_FindInvalidData
                                           //|SLProcess_SplitLargeMeshes
        );

        SLNode* scene = new SLNode("Scene");
        scene->addChild(light1);
        if (largeModel)
        {
            largeModel->scaleToCenter(100000.0f);
            scene->addChild(largeModel);
        }
        scene->addChild(cam1);

        sv->camera(cam1);
        s->root3D(scene);
    }
    else if (SLApplication::sceneID == SID_TextureBlend) //..............................................
    {
        s->name("Texture Blending Test");
        s->info("Texture map blending with depth sorting. Trees in view frustum are rendered back to front.");

        SLGLTexture* t1 = new SLGLTexture("tree1_1024_C.png",
                                          GL_LINEAR_MIPMAP_LINEAR,
                                          GL_LINEAR,
                                          TT_color,
                                          GL_CLAMP_TO_EDGE,
                                          GL_CLAMP_TO_EDGE);
        SLGLTexture* t2 = new SLGLTexture("grass0512_C.jpg",
                                          GL_LINEAR_MIPMAP_LINEAR,
                                          GL_LINEAR);

        SLMaterial* m1 = new SLMaterial("m1", SLCol4f(1, 1, 1), SLCol4f(0, 0, 0), 100);
        SLMaterial* m2 = new SLMaterial("m2", SLCol4f(1, 1, 1), SLCol4f(0, 0, 0), 100);
        m1->program(s->programs()[SP_TextureOnly]);
        m1->textures().push_back(t1);
        m2->textures().push_back(t2);

        SLCamera* cam1 = new SLCamera("Camera 1");
        cam1->translation(0, 3, 25);
        cam1->lookAt(0, 0, 10);
        cam1->focalDist(25);
        cam1->background().colors(SLCol4f(0.6f, 0.6f, 1));
        cam1->setInitialState();

        SLLightSpot* light = new SLLightSpot(0.1f);
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
        SLNode* p1 = new SLNode(new SLPolygon(pNW, tNW, "Tree+X", m1));
        SLNode* p2 = new SLNode(new SLPolygon(pNW, tNW, "Tree-Z", m1));
        p2->rotate(90, 0, 1, 0);
        SLNode* p3 = new SLNode(new SLPolygon(pSE, tSE, "Tree-X", m1));
        SLNode* p4 = new SLNode(new SLPolygon(pSE, tSE, "Tree+Z", m1));
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
        scene->addChild(new SLNode(new SLPolygon(pG, tG, "Ground", m2)));

        //create 21*21*21-1 references around the center tree
        SLint res = 10;
        for (SLint iZ = -res; iZ <= res; ++iZ)
        {
            for (SLint iX = -res; iX <= res; ++iX)
            {
                if (iX != 0 || iZ != 0)
                {
                    SLNode* t = tree->copyRec();
                    t->translate(float(iX) * 2 + SL_random(0.7f, 1.4f),
                                 0,
                                 float(iZ) * 2 + SL_random(0.7f, 1.4f),
                                 TS_object);
                    t->rotate(SL_random(0, 90), 0, 1, 0);
                    t->scale(SL_random(0.5f, 1.0f));
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
        SLGLTexture* texB = new SLGLTexture("brick0512_C.png", GL_NEAREST, GL_NEAREST);
        SLGLTexture* texL = new SLGLTexture("brick0512_C.png", GL_LINEAR, GL_LINEAR);
        SLGLTexture* texT = new SLGLTexture("brick0512_C.png", GL_LINEAR_MIPMAP_LINEAR, GL_LINEAR);
        SLGLTexture* texR = new SLGLTexture("brick0512_C.png", SL_ANISOTROPY_MAX, GL_LINEAR);

        // define materials with textureOnly shader, no light needed
        SLMaterial* matB = new SLMaterial("matB", texB, nullptr, nullptr, nullptr, s->programs()[SP_TextureOnly]);
        SLMaterial* matL = new SLMaterial("matL", texL, nullptr, nullptr, nullptr, s->programs()[SP_TextureOnly]);
        SLMaterial* matT = new SLMaterial("matT", texT, nullptr, nullptr, nullptr, s->programs()[SP_TextureOnly]);
        SLMaterial* matR = new SLMaterial("matR", texR, nullptr, nullptr, nullptr, s->programs()[SP_TextureOnly]);

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
        SLNode* polyB = new SLNode(new SLPolygon(VB, T, "PolygonB", matB));

        SLVVec3f VL;
        VL.push_back(SLVec3f(-0.5f, 0.5f, 1.0f));
        VL.push_back(SLVec3f(-0.5f, -0.5f, 1.0f));
        VL.push_back(SLVec3f(-0.5f, -0.5f, -2.0f));
        VL.push_back(SLVec3f(-0.5f, 0.5f, -2.0f));
        SLNode* polyL = new SLNode(new SLPolygon(VL, T, "PolygonL", matL));

        SLVVec3f VT;
        VT.push_back(SLVec3f(0.5f, 0.5f, 1.0f));
        VT.push_back(SLVec3f(-0.5f, 0.5f, 1.0f));
        VT.push_back(SLVec3f(-0.5f, 0.5f, -2.0f));
        VT.push_back(SLVec3f(0.5f, 0.5f, -2.0f));
        SLNode* polyT = new SLNode(new SLPolygon(VT, T, "PolygonT", matT));

        SLVVec3f VR;
        VR.push_back(SLVec3f(0.5f, -0.5f, 1.0f));
        VR.push_back(SLVec3f(0.5f, 0.5f, 1.0f));
        VR.push_back(SLVec3f(0.5f, 0.5f, -2.0f));
        VR.push_back(SLVec3f(0.5f, -0.5f, -2.0f));
        SLNode* polyR = new SLNode(new SLPolygon(VR, T, "PolygonR", matR));

#ifdef SL_GLES2
        // Create 3D textured sphere mesh and node
        SLNode* sphere = new SLNode(new SLSphere(0.2f, 16, 16, "Sphere", matL));
#else
        // 3D Texture Mapping on a pyramid
        SLVstring tex3DFiles;
        for (SLint i = 0; i < 256; ++i) tex3DFiles.push_back("Wave_radial10_256C.jpg");
        SLGLTexture* tex3D = new SLGLTexture(tex3DFiles);
        SLGLProgram* spr3D = new SLGLGenericProgram("TextureOnly3D.vert", "TextureOnly3D.frag");
        SLMaterial*  mat3D = new SLMaterial("mat3D", tex3D, nullptr, nullptr, nullptr, spr3D);

        // Create 3D textured pyramid mesh and node
        SLMesh* pyramid = new SLMesh("Pyramid");
        pyramid->mat(mat3D);
        pyramid->P          = {{-1, -1, 1}, {1, -1, 1}, {1, -1, -1}, {-1, -1, -1}, {0, 2, 0}};
        pyramid->I16        = {0, 3, 1, 1, 3, 2, 4, 0, 1, 4, 1, 2, 4, 2, 3, 4, 3, 0};
        SLNode* pyramidNode = new SLNode(pyramid, "Pyramid");
        pyramidNode->scale(0.2f);
        pyramidNode->translate(0, 0, -3);

        // Create 3D textured sphere mesh and node
        SLNode*      sphere = new SLNode(new SLSphere(0.2f, 16, 16, "Sphere", mat3D));
#endif

        SLCamera* cam1 = new SLCamera("Camera 1");
        cam1->translation(0, 0, 2.2f);
        cam1->lookAt(0, 0, 0);
        cam1->focalDist(2.2f);
        cam1->background().colors(SLCol4f(0.2f, 0.2f, 0.2f));
        cam1->setInitialState();

        SLNode* scene = new SLNode();
        scene->addChild(polyB);
        scene->addChild(polyL);
        scene->addChild(polyT);
        scene->addChild(polyR);
        scene->addChild(sphere);
        scene->addChild(cam1);
#ifndef SL_GLES2
        scene->addChild(pyramidNode);
#endif

        sv->camera(cam1);
        s->root3D(scene);
    }
    else if (SLApplication::sceneID == SID_FrustumCull) //...............................................
    {
        s->name("Frustum Culling Test");
        s->info("View frustum culling: Only objects in view frustum are rendered. You can turn view culling off in the render flags.");

        // create texture
        SLGLTexture* tex  = new SLGLTexture("earth1024_C.jpg");
        SLMaterial*  mat1 = new SLMaterial("mat1", tex);

        SLCamera* cam1 = new SLCamera("Camera 1");
        cam1->clipNear(0.1f);
        cam1->clipFar(100);
        cam1->translation(0, 0, 1);
        cam1->lookAt(0, 0, 0);
        cam1->focalDist(5);
        cam1->background().colors(SLCol4f(0.1f, 0.1f, 0.1f));
        cam1->setInitialState();

        SLLightSpot* light1 = new SLLightSpot(10, 10, 10, 0.3f);
        light1->ambient(SLCol4f(0.2f, 0.2f, 0.2f));
        light1->diffuse(SLCol4f(0.8f, 0.8f, 0.8f));
        light1->specular(SLCol4f(1, 1, 1));
        light1->attenuation(1, 0, 0);

        SLNode* scene = new SLNode;
        scene->addChild(cam1);
        scene->addChild(light1);

        // add one single sphere in the center
        SLuint  res    = 16;
        SLNode* sphere = new SLNode(new SLSphere(0.15f, res, res, "mySphere", mat1));
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
                        SLNode* s = sphere->copyRec();
                        s->translate(float(iX), float(iY), float(iZ), TS_object);
                        scene->addChild(s);
                    }
                }
            }
        }

        SLuint num = (SLuint)(size + size + 1);
        SL_LOG("Triangles on GPU: %u\n", res * res * 2 * num * num * num);

        sv->camera(cam1);
        sv->doWaitOnIdle(false);
        s->root3D(scene);
    }
    else if (SLApplication::sceneID == SID_2Dand3DText) //...............................................
    {
        s->name("2D & 3D Text Test");
        s->info("All 3D objects are in the _root3D scene and the center text is in the _root2D scene and rendered in orthographic projection in screen space.");

        SLMaterial* m1 = new SLMaterial("m1", SLCol4f::RED);

        SLCamera* cam1 = new SLCamera("Camera 1");
        cam1->clipNear(0.1f);
        cam1->clipFar(100);
        cam1->translation(0, 0, 5);
        cam1->lookAt(0, 0, 0);
        cam1->focalDist(5);
        cam1->background().colors(SLCol4f(0.1f, 0.1f, 0.1f));
        cam1->setInitialState();

        SLLightSpot* light1 = new SLLightSpot(10, 10, 10, 0.3f);
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

        txt         = "This is text in 3D with font09";
        size        = SLTexFont::font09->calcTextSize(txt);
        SLNode* t09 = new SLText(txt, SLTexFont::font09);
        t09->translate(-size.x * 0.5f * scale, 0.8f, 0);
        t09->scale(scale);

        txt         = "This is text in 3D with font12";
        size        = SLTexFont::font12->calcTextSize(txt);
        SLNode* t12 = new SLText(txt, SLTexFont::font12);
        t12->translate(-size.x * 0.5f * scale, 0.6f, 0);
        t12->scale(scale);

        txt         = "This is text in 3D with font20";
        size        = SLTexFont::font20->calcTextSize(txt);
        SLNode* t20 = new SLText(txt, SLTexFont::font20);
        t20->translate(-size.x * 0.5f * scale, -0.8f, 0);
        t20->scale(scale);

        txt         = "This is text in 3D with font22";
        size        = SLTexFont::font22->calcTextSize(txt);
        SLNode* t22 = new SLText(txt, SLTexFont::font22);
        t22->translate(-size.x * 0.5f * scale, -1.2f, 0);
        t22->scale(scale);

        // Now create 2D text but don't scale it (all sizes in pixels)
        txt           = "This is text in 2D with font16";
        size          = SLTexFont::font16->calcTextSize(txt);
        SLNode* t2D16 = new SLText(txt, SLTexFont::font16);
        t2D16->translate(-size.x * 0.5f, 0, 0);

        // Assemble 3D scene as usual with camera and light
        SLNode* scene3D = new SLNode("root3D");
        scene3D->addChild(cam1);
        scene3D->addChild(light1);
        scene3D->addChild(new SLNode(new SLSphere(0.5f, 32, 32, "Sphere", m1)));
        scene3D->addChild(t07);
        scene3D->addChild(t09);
        scene3D->addChild(t12);
        scene3D->addChild(t20);
        scene3D->addChild(t22);

        // Assemble 2D scene
        SLNode* scene2D = new SLNode("root2D");
        ;
        scene2D->addChild(t2D16);

        sv->camera(cam1);
        sv->doWaitOnIdle(true);

        s->root3D(scene3D);
        s->root2D(scene2D);
    }
    else if (SLApplication::sceneID == SID_MassiveData) //...............................................
    {
        s->name("Massive Data Test");
        s->info("No data is shared on the GPU. Check Memory consumption.");

        SLCamera* cam1 = new SLCamera("Camera 1");
        cam1->clipNear(0.1f);
        cam1->clipFar(100);
        cam1->translation(0, 0, 50);
        cam1->lookAt(0, 0, 0);
        cam1->focalDist(5);
        cam1->background().colors(SLCol4f(0.1f, 0.1f, 0.1f));
        cam1->setInitialState();

        SLLightSpot* light1 = new SLLightSpot(10, 10, 10, 0.3f);
        light1->ambient(SLCol4f(0.2f, 0.2f, 0.2f));
        light1->diffuse(SLCol4f(0.8f, 0.8f, 0.8f));
        light1->specular(SLCol4f(1, 1, 1));
        light1->attenuation(1, 0, 0);

        SLNode* scene = new SLNode;
        scene->addChild(cam1);
        scene->addChild(light1);

        // Create shader program with 4 uniforms
        SLGLProgram*   sp     = new SLGLGenericProgram("BumpNormal.vert", "BumpNormalParallax.frag");
        SLGLUniform1f* scale  = new SLGLUniform1f(UT_const, "u_scale", 0.01f, 0.002f, 0, 1, (SLKey)'X');
        SLGLUniform1f* offset = new SLGLUniform1f(UT_const, "u_offset", 0.01f, 0.002f, -1, 1, (SLKey)'O');
        s->eventHandlers().push_back(scale);
        s->eventHandlers().push_back(offset);
        sp->addUniform1f(scale);
        sp->addUniform1f(offset);

        // create new materials for every sphere
        SLGLTexture* texC = new SLGLTexture("earth2048_C.jpg"); // color map
        SLGLTexture* texN = new SLGLTexture("earth2048_N.jpg"); // normal map
        SLMaterial*  mat  = new SLMaterial("mat1", texC, texN, nullptr, nullptr, sp);

        // create spheres around the center sphere
        SLint size = 8;
        for (SLint iZ = -size; iZ <= size; ++iZ)
        {
            for (SLint iY = -size; iY <= size; ++iY)
            {
                for (SLint iX = -size; iX <= size; ++iX)
                {
                    // add one single sphere in the center
                    SLuint    res    = 30;
                    SLSphere* earth  = new SLSphere(0.3f, res, res, "earth", mat);
                    SLNode*   sphere = new SLNode(earth);
                    sphere->translate(float(iX), float(iY), float(iZ), TS_object);
                    scene->addChild(sphere);
                }
            }
        }

        sv->camera(cam1);
        sv->doWaitOnIdle(false);
        s->root3D(scene);
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

        SLLightSpot* light1 = new SLLightSpot(10, 10, 10, 0.3f);
        light1->ambient(SLCol4f(0.2f, 0.2f, 0.2f));
        light1->diffuse(SLCol4f(0.8f, 0.8f, 0.8f));
        light1->specular(SLCol4f(1, 1, 1));
        light1->attenuation(1, 0, 0);

        SLMaterial* pcMat1 = new SLMaterial("Red", SLCol4f::RED);
        pcMat1->program(new SLGLGenericProgram("ColorUniformPoint.vert", "Color.frag"));
        pcMat1->program()->addUniform1f(new SLGLUniform1f(UT_const, "u_pointSize", 3.0f));
        SLRnd3fNormal rndN(SLVec3f(0, 0, 0), SLVec3f(5, 2, 1));
        SLNode*       pc1 = new SLNode(new SLPoints(1000, rndN, "PC1", pcMat1));
        pc1->translate(-5, 0, 0);

        SLMaterial* pcMat2 = new SLMaterial("Green", SLCol4f::GREEN);
        pcMat2->program(new SLGLGenericProgram("ColorUniform.vert", "Color.frag"));
        SLRnd3fUniform rndU(SLVec3f(0, 0, 0), SLVec3f(2, 3, 5));
        SLNode*        pc2 = new SLNode(new SLPoints(1000, rndU, "PC2", pcMat2));
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
        SLMaterial* m1;

        if (SLApplication::sceneID == SID_ShaderPerPixelBlinn)
        {
            s->name("Blinn-Phong per pixel lighting");
            s->info("Per-pixel lighting with Blinn-Phong lightmodel. The reflection of 5 light sources is calculated per pixel.");
            m1 = new SLMaterial("m1", nullptr, nullptr, nullptr, nullptr, s->programs()[SP_perPixBlinn]);
        }
        else
        {
            s->name("Blinn-Phong per vertex lighting");
            s->info("Per-vertex lighting with Blinn-Phong lightmodel. The reflection of 5 light sources is calculated per vertex.");
            m1 = new SLMaterial("m1", nullptr);
        }

        m1->shininess(500);

        // Base root group node for the scene
        SLNode* scene = new SLNode;

        SLCamera* cam1 = new SLCamera("Camera 1");
        cam1->translation(0, 1, 8);
        cam1->lookAt(0, 1, 0);
        cam1->focalDist(8);
        cam1->background().colors(SLCol4f(0.1f, 0.1f, 0.1f));
        cam1->setInitialState();
        scene->addChild(cam1);

        // Define 5 light sources
        // A rectangluar wight light on top
        SLLightRect* light0 = new SLLightRect(2.0f, 1.0f);
        light0->ambient(SLCol4f(0, 0, 0));
        light0->diffuse(SLCol4f(1, 1, 1));
        light0->translation(0, 3, 0);
        light0->lookAt(0, 0, 0, 0, 0, -1);
        light0->attenuation(0, 0, 1);
        scene->addChild(light0);

        // A red point light from from front left
        SLLightSpot* light1 = new SLLightSpot(0.1f);
        light1->ambient(SLCol4f(0, 0, 0));
        light1->diffuse(SLCol4f(1, 0, 0));
        light1->specular(SLCol4f(1, 0, 0));
        light1->translation(0, 0, 2);
        light1->lookAt(0, 0, 0);
        light1->attenuation(0, 0, 1);
        scene->addChild(light1);

        // A green spot light with 40 deg. spot angle from front right
        //SLLightSpot* light2 = new SLLightSpot(0.1f, 20.0f, true);
        //light2->ambient(SLCol4f(0,0,0));
        //light2->diffuse(SLCol4f(0,1,0));
        //light2->specular(SLCol4f(0,1,0));
        //light2->translation(1.5f, 1.5f, 1.5f);
        //light2->lookAt(0, 0, 0);
        //light2->attenuation(0,0,1);
        //scene->addChild(light2);

        // A green spot head light with 40 deg. spot angle from front right
        SLLightSpot* light2 = new SLLightSpot(0.1f, 20.0f, true);
        light2->ambient(SLCol4f(0, 0, 0));
        light2->diffuse(SLCol4f(0, 1, 0));
        light2->specular(SLCol4f(0, 1, 0));
        light2->translation(1.5f, 0.5f, -6.5f);
        light2->lookAt(0.5f, -0.5f, -7.5f);
        light2->attenuation(0, 0, 1);
        cam1->addChild(light2);

        // A blue spot light with 40 deg. spot angle from front left
        SLLightSpot* light3 = new SLLightSpot(0.1f, 20.0f, true);
        light3->ambient(SLCol4f(0, 0, 0));
        light3->diffuse(SLCol4f(0, 0, 1));
        light3->specular(SLCol4f(0, 0, 1));
        light3->translation(-1.5f, 1.5f, 1.5f);
        light3->lookAt(0, 0, 0);
        light3->attenuation(0, 0, 1);
        scene->addChild(light3);

        // A yellow directional light from the back-bottom
        SLLightDirect* light4 = new SLLightDirect();
        light4->ambient(SLCol4f(0, 0, 0));
        light4->diffuse(SLCol4f(1, 1, 0));
        light4->specular(SLCol4f(1, 1, 0));
        light4->translation(-1.5f, -1.5f, -1.5f);
        light4->lookAt(0, 0, 0);
        scene->addChild(light4);

        // Add some meshes to be lighted
        scene->addChild(new SLNode(new SLSpheric(1.0f, 0.0f, 180.0f, 20, 20, "Sphere", m1)));
        scene->addChild(new SLNode(new SLBox(1, -1, -1, 2, 1, 1, "Box", m1)));

        sv->camera(cam1);
        s->root3D(scene);
    }
    else if (SLApplication::sceneID == SID_ShaderCookTorrance) //........................................
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
        scene->addChild(cam1);

        // Create spheres and materials with roughness & metallic values between 0 and 1
        const SLint nrRows  = 7;
        const SLint nrCols  = 7;
        SLfloat     spacing = 2.5f;
        SLfloat     maxX    = (nrCols / 2) * spacing;
        SLfloat     maxY    = (nrRows / 2) * spacing;
        SLfloat     deltaR  = 1.0f / (float)(nrRows - 1);
        SLfloat     deltaM  = 1.0f / (float)(nrCols - 1);

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
                    mat[i] = new SLMaterial("CookTorranceMatTex",
                                            new SLGLTexture("rusty-metal_2048C.png"),
                                            new SLGLTexture("rusty-metal_2048N.png"),
                                            new SLGLTexture("rusty-metal_2048M.png"),
                                            new SLGLTexture("rusty-metal_2048R.png"),
                                            s->programs()[SP_perPixCookTorranceTex]);
                }
                else
                {
                    // Cook-Torrance material without textures
                    mat[i] = new SLMaterial("CookTorranceMat",
                                            SLCol4f::RED * 0.5f,
                                            SL_clamp((float)r * deltaR, 0.05f, 1.0f),
                                            (float)m * deltaM);
                }

                SLNode* node = new SLNode(new SLSpheric(1.0f, 0.0f, 180.0f, 32, 32, "Sphere", mat[i]));
                node->translate(x, y, 0);
                scene->addChild(node);
                x += spacing;
                i++;
            }
            y += spacing;
        }

        // Add 4 point light
        SLLightSpot* light1 = new SLLightSpot(-maxX, maxY, maxY, 0.1f, 180.0f, 0.0f, 300, 300);
        light1->attenuation(0, 0, 1);
        SLLightSpot* light2 = new SLLightSpot(maxX, maxY, maxY, 0.1f, 180.0f, 0.0f, 300, 300);
        light2->attenuation(0, 0, 1);
        SLLightSpot* light3 = new SLLightSpot(-maxX, -maxY, maxY, 0.1f, 180.0f, 0.0f, 300, 300);
        light3->attenuation(0, 0, 1);
        SLLightSpot* light4 = new SLLightSpot(maxX, -maxY, maxY, 0.1f, 180.0f, 0.0f, 300, 300);
        light4->attenuation(0, 0, 1);
        scene->addChild(light1);
        scene->addChild(light2);
        scene->addChild(light3);
        scene->addChild(light4);

        sv->camera(cam1);
        s->root3D(scene);
    }
    else if (SLApplication::sceneID == SID_ShaderPerVertexWave) //.......................................
    {
        s->name("Wave Shader Test");
        s->info("Vertex Shader with wave displacment.");
        SL_LOG("Use H-Key to increment (decrement w. shift) the wave height.\n\n");

        SLCamera* cam1 = new SLCamera("Camera 1");
        cam1->translation(0, 3, 8);
        cam1->lookAt(0, 0, 0);
        cam1->focalDist(cam1->translationOS().length());
        cam1->background().colors(SLCol4f(0.1f, 0.4f, 0.8f));
        cam1->setInitialState();

        // Create generic shader program with 4 custom uniforms
        SLGLProgram*   sp  = new SLGLGenericProgram("Wave.vert", "Wave.frag");
        SLGLUniform1f* u_h = new SLGLUniform1f(UT_const, "u_h", 0.1f, 0.05f, 0.0f, 0.5f, (SLKey)'H');
        s->eventHandlers().push_back(u_h);
        sp->addUniform1f(u_h);
        sp->addUniform1f(new SLGLUniform1f(UT_inc, "u_t", 0.0f, 0.06f));
        sp->addUniform1f(new SLGLUniform1f(UT_const, "u_a", 2.5f));
        sp->addUniform1f(new SLGLUniform1f(UT_incDec, "u_b", 2.2f, 0.01f, 2.0f, 2.5f));

        // Create materials
        SLMaterial* matWater = new SLMaterial("matWater", SLCol4f(0.45f, 0.65f, 0.70f), SLCol4f::WHITE, 300);
        matWater->program(sp);
        SLMaterial* matRed = new SLMaterial("matRed", SLCol4f(1.00f, 0.00f, 0.00f));

        // water rectangle in the y=0 plane
        SLNode* wave = new SLNode(new SLRectangle(SLVec2f(-SL_PI, -SL_PI), SLVec2f(SL_PI, SL_PI), 40, 40, "WaterRect", matWater));
        wave->rotate(90, -1, 0, 0);

        SLLightSpot* light0 = new SLLightSpot();
        light0->ambient(SLCol4f(0, 0, 0));
        light0->diffuse(SLCol4f(1, 1, 1));
        light0->translate(0, 4, -4, TS_object);
        light0->attenuation(1, 0, 0);

        SLNode* scene = new SLNode;
        scene->addChild(light0);
        scene->addChild(wave);
        scene->addChild(new SLNode(new SLSphere(1, 32, 32, "Red Sphere", matRed)));
        scene->addChild(cam1);

        sv->camera(cam1);
        s->root3D(scene);
        sv->doWaitOnIdle(false);
    }
    else if (SLApplication::sceneID == SID_ShaderWater) //...............................................
    {
        s->name("Water Shader Test");
        s->info("Water Shader with reflection & refraction mapping.");
        SL_LOG("Use H-Key to increment (decrement w. shift) the wave height.\n\n");

        SLCamera* cam1 = new SLCamera("Camera 1");
        cam1->translation(0, 3, 8);
        cam1->lookAt(0, 0, 0);
        cam1->focalDist(cam1->translationOS().length());
        cam1->background().colors(SLCol4f(0.1f, 0.4f, 0.8f));
        cam1->setInitialState();

        // create texture
        SLGLTexture* tex1 = new SLGLTexture("Pool+X0512_C.png", "Pool-X0512_C.png", "Pool+Y0512_C.png", "Pool-Y0512_C.png", "Pool+Z0512_C.png", "Pool-Z0512_C.png");
        SLGLTexture* tex2 = new SLGLTexture("tile1_0256_C.jpg");

        // Create generic shader program with 4 custom uniforms
        SLGLProgram*   sp  = new SLGLGenericProgram("WaveRefractReflect.vert",
                                                 "RefractReflect.frag");
        SLGLUniform1f* u_h = new SLGLUniform1f(UT_const, "u_h", 0.1f, 0.05f, 0.0f, 0.5f, (SLKey)'H');
        s->eventHandlers().push_back(u_h);
        sp->addUniform1f(u_h);
        sp->addUniform1f(new SLGLUniform1f(UT_inc, "u_t", 0.0f, 0.06f));
        sp->addUniform1f(new SLGLUniform1f(UT_const, "u_a", 2.5f));
        sp->addUniform1f(new SLGLUniform1f(UT_incDec, "u_b", 2.2f, 0.01f, 2.0f, 2.5f));

        // Create materials
        SLMaterial* matWater = new SLMaterial("matWater", SLCol4f(0.45f, 0.65f, 0.70f), SLCol4f::WHITE, 100, 0.1f, 0.9f, 1.5f);
        matWater->program(sp);
        matWater->textures().push_back(tex1);
        SLMaterial* matRed  = new SLMaterial("matRed", SLCol4f(1.00f, 0.00f, 0.00f));
        SLMaterial* matTile = new SLMaterial("matTile");
        matTile->textures().push_back(tex2);

        // water rectangle in the y=0 plane
        SLNode* rect = new SLNode(new SLRectangle(SLVec2f(-SL_PI, -SL_PI),
                                                  SLVec2f(SL_PI, SL_PI),
                                                  40,
                                                  40,
                                                  "WaterRect",
                                                  matWater));
        rect->rotate(90, -1, 0, 0);

        // Pool rectangles
        SLuint  res   = 10;
        SLNode* rectF = new SLNode(new SLRectangle(SLVec2f(-SL_PI, -SL_PI / 6), SLVec2f(SL_PI, SL_PI / 6), SLVec2f(0, 0), SLVec2f(10, 2.5f), res, res, "rectF", matTile));
        SLNode* rectN = new SLNode(new SLRectangle(SLVec2f(-SL_PI, -SL_PI / 6), SLVec2f(SL_PI, SL_PI / 6), SLVec2f(0, 0), SLVec2f(10, 2.5f), res, res, "rectN", matTile));
        SLNode* rectL = new SLNode(new SLRectangle(SLVec2f(-SL_PI, -SL_PI / 6), SLVec2f(SL_PI, SL_PI / 6), SLVec2f(0, 0), SLVec2f(10, 2.5f), res, res, "rectL", matTile));
        SLNode* rectR = new SLNode(new SLRectangle(SLVec2f(-SL_PI, -SL_PI / 6), SLVec2f(SL_PI, SL_PI / 6), SLVec2f(0, 0), SLVec2f(10, 2.5f), res, res, "rectR", matTile));
        SLNode* rectB = new SLNode(new SLRectangle(SLVec2f(-SL_PI, -SL_PI), SLVec2f(SL_PI, SL_PI), SLVec2f(0, 0), SLVec2f(10, 10), res, res, "rectB", matTile));
        rectF->translate(0, 0, -SL_PI, TS_object);
        rectL->rotate(90, 0, 1, 0);
        rectL->translate(0, 0, -SL_PI, TS_object);
        rectN->rotate(180, 0, 1, 0);
        rectN->translate(0, 0, -SL_PI, TS_object);
        rectR->rotate(270, 0, 1, 0);
        rectR->translate(0, 0, -SL_PI, TS_object);
        rectB->rotate(90, -1, 0, 0);
        rectB->translate(0, 0, -SL_PI / 6, TS_object);

        SLLightSpot* light0 = new SLLightSpot();
        light0->ambient(SLCol4f(0, 0, 0));
        light0->diffuse(SLCol4f(1, 1, 1));
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
        scene->addChild(new SLNode(new SLSphere(1, 32, 32, "Red Sphere", matRed)));
        scene->addChild(cam1);

        sv->camera(cam1);
        s->root3D(scene);
        sv->doWaitOnIdle(false);
    }
    else if (SLApplication::sceneID == SID_ShaderBumpNormal) //..........................................
    {
        s->name("Normal Map Test");
        s->info("Normal map bump mapping combined with a per pixel spot lighting.");

        // Create textures
        SLGLTexture* texC = new SLGLTexture("brickwall0512_C.jpg");
        SLGLTexture* texN = new SLGLTexture("brickwall0512_N.jpg");

        // Create materials
        SLMaterial* m1 = new SLMaterial("m1", texC, texN, nullptr, nullptr, s->programs()[SP_bumpNormal]);

        SLCamera* cam1 = new SLCamera("Camera 1");
        cam1->translation(0, 0, 20);
        cam1->lookAt(0, 0, 0);
        cam1->focalDist(20);
        cam1->background().colors(SLCol4f(0.5f, 0.5f, 0.5f));
        cam1->setInitialState();

        SLLightSpot* light1 = new SLLightSpot(0.3f);
        light1->ambient(SLCol4f(0.1f, 0.1f, 0.1f));
        light1->diffuse(SLCol4f(1, 1, 1));
        light1->specular(SLCol4f(1, 1, 1));
        light1->attenuation(1, 0, 0);
        light1->translation(0, 0, 5);
        light1->lookAt(0, 0, 0);
        light1->spotCutOffDEG(40);

        SLAnimation* anim = SLAnimation::create("light1_anim", 2.0f);
        anim->createEllipticNodeTrack(light1, 2.0f, A_x, 2.0f, A_Y);

        SLNode* scene = new SLNode;
        scene->addChild(light1);
        scene->addChild(new SLNode(new SLRectangle(SLVec2f(-5, -5), SLVec2f(5, 5), 1, 1, "Rect", m1)));
        scene->addChild(cam1);

        sv->camera(cam1);
        s->root3D(scene);
    }
    else if (SLApplication::sceneID == SID_ShaderBumpParallax) //........................................
    {
        s->name("Parallax Map Test");
        s->info("Normal map parallax mapping.");
        SL_LOG("Demo application for parallax bump mapping.\n");
        SL_LOG("Use S-Key to increment (decrement w. shift) parallax scale.\n");
        SL_LOG("Use O-Key to increment (decrement w. shift) parallax offset.\n\n");

        // Create shader program with 4 uniforms
        SLGLProgram*   sp     = new SLGLGenericProgram("BumpNormal.vert", "BumpNormalParallax.frag");
        SLGLUniform1f* scale  = new SLGLUniform1f(UT_const, "u_scale", 0.04f, 0.002f, 0, 1, (SLKey)'X');
        SLGLUniform1f* offset = new SLGLUniform1f(UT_const, "u_offset", -0.03f, 0.002f, -1, 1, (SLKey)'O');
        s->eventHandlers().push_back(scale);
        s->eventHandlers().push_back(offset);
        sp->addUniform1f(scale);
        sp->addUniform1f(offset);

        // Create textures
        SLGLTexture* texC = new SLGLTexture("brickwall0512_C.jpg");
        SLGLTexture* texN = new SLGLTexture("brickwall0512_N.jpg");
        SLGLTexture* texH = new SLGLTexture("brickwall0512_H.jpg");

        // Create materials
        SLMaterial* m1 = new SLMaterial("mat1", texC, texN, texH, nullptr, sp);

        SLCamera* cam1 = new SLCamera("Camera 1");
        cam1->translation(0, 0, 20);
        cam1->lookAt(0, 0, 0);
        cam1->focalDist(20);
        cam1->background().colors(SLCol4f(0.5f, 0.5f, 0.5f));
        cam1->setInitialState();

        SLLightSpot* light1 = new SLLightSpot(0.3f);
        light1->ambient(SLCol4f(0.1f, 0.1f, 0.1f));
        light1->diffuse(SLCol4f(1, 1, 1));
        light1->specular(SLCol4f(1, 1, 1));
        light1->attenuation(1, 0, 0);
        light1->translation(0, 0, 5);
        light1->lookAt(0, 0, 0);
        light1->spotCutOffDEG(50);

        SLAnimation* anim = SLAnimation::create("light1_anim", 2.0f);
        anim->createEllipticNodeTrack(light1, 2.0f, A_x, 2.0f, A_Y);

        SLNode* scene = new SLNode;
        scene->addChild(light1);
        scene->addChild(new SLNode(new SLRectangle(SLVec2f(-5, -5), SLVec2f(5, 5), 1, 1, "Rect", m1)));
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
        SLSkybox*    skybox    = new SLSkybox("Desert+X1024_C.jpg", "Desert-X1024_C.jpg", "Desert+Y1024_C.jpg", "Desert-Y1024_C.jpg", "Desert+Z1024_C.jpg", "Desert-Z1024_C.jpg");
        SLGLTexture* skyboxTex = skybox->meshes()[0]->mat()->textures()[0];

        // Material for mirror
        SLMaterial* refl = new SLMaterial("refl", SLCol4f::BLACK, SLCol4f::WHITE, 1000, 1.0f);
        refl->textures().push_back(skyboxTex);
        refl->program(new SLGLGenericProgram("Reflect.vert", "Reflect.frag"));

        // Material for glass
        SLMaterial* refr = new SLMaterial("refr", SLCol4f::BLACK, SLCol4f::BLACK, 100, 0.1f, 0.9f, 1.5f);
        refr->translucency(1000);
        refr->transmissiv(SLCol4f::WHITE);
        refr->textures().push_back(skyboxTex);
        refr->program(new SLGLGenericProgram("RefractReflect.vert", "RefractReflect.frag"));

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
        SLLightDirect* light = new SLLightDirect(0.5f);
        light->ambient(SLCol4f(0.3f, 0.3f, 0.3f));
        light->attenuation(1, 0, 0);
        light->translate(1, 1, -1);
        light->lookAt(-1, -1, 1);
        scene->addChild(light);

        // Center sphere
        SLNode* sphere = new SLNode(new SLSphere(0.5f, 32, 32, "Sphere", refr));
        scene->addChild(sphere);

        // load teapot
        SLAssimpImporter importer;
        SLNode*          teapot = importer.load("FBX/Teapot/Teapot.fbx", true, refl);
        teapot->translate(-1.5f, -0.5f, 0);
        scene->addChild(teapot);

        // load Suzanne
        SLNode* suzanne = importer.load("FBX/Suzanne/Suzanne.fbx", true, refr);
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
        s->info("Complex earth shader with 7 textures: daycolor, nightcolor, normal, height & gloss map of earth, color & alphamap of clouds");
        SL_LOG("Earth Shader from Markus Knecht\n");
        SL_LOG("Use (SHIFT) & key Y to change scale of the parallax mapping\n");
        SL_LOG("Use (SHIFT) & key X to change bias of the parallax mapping\n");
        SL_LOG("Use (SHIFT) & key C to change cloud height\n");

        // Create shader program with 4 uniforms
        SLGLProgram*   sp     = new SLGLGenericProgram("BumpNormal.vert", "BumpNormalEarth.frag");
        SLGLUniform1f* scale  = new SLGLUniform1f(UT_const, "u_scale", 0.02f, 0.002f, 0, 1, (SLKey)'X');
        SLGLUniform1f* offset = new SLGLUniform1f(UT_const, "u_offset", -0.02f, 0.002f, -1, 1, (SLKey)'O');
        s->eventHandlers().push_back(scale);
        s->eventHandlers().push_back(offset);
        sp->addUniform1f(scale);
        sp->addUniform1f(offset);

// Create textures
#ifndef SL_GLES
        SLGLTexture* texC  = new SLGLTexture("earth2048_C.jpg");      // color map
        SLGLTexture* texN  = new SLGLTexture("earth2048_N.jpg");      // normal map
        SLGLTexture* texH  = new SLGLTexture("earth2048_H.jpg");      // height map
        SLGLTexture* texG  = new SLGLTexture("earth2048_G.jpg");      // gloss map
        SLGLTexture* texNC = new SLGLTexture("earthNight2048_C.jpg"); // night color  map
#else
        SLGLTexture* texC   = new SLGLTexture("earth1024_C.jpg");      // color map
        SLGLTexture* texN   = new SLGLTexture("earth1024_N.jpg");      // normal map
        SLGLTexture* texH   = new SLGLTexture("earth1024_H.jpg");      // height map
        SLGLTexture* texG   = new SLGLTexture("earth1024_G.jpg");      // gloss map
        SLGLTexture* texNC  = new SLGLTexture("earthNight1024_C.jpg"); // night color  map
#endif
        SLGLTexture* texClC = new SLGLTexture("earthCloud1024_C.jpg"); // cloud color map
        SLGLTexture* texClA = new SLGLTexture("earthCloud1024_A.jpg"); // cloud alpha map

        // Create materials
        SLMaterial* matEarth = new SLMaterial("matEarth", texC, texN, texH, texG, sp);
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

        SLLightSpot* sun = new SLLightSpot();
        sun->ambient(SLCol4f(0, 0, 0));
        sun->diffuse(SLCol4f(1, 1, 1));
        sun->specular(SLCol4f(0.2f, 0.2f, 0.2f));
        sun->attenuation(1, 0, 0);

        SLAnimation* anim = SLAnimation::create("light1_anim", 24.0f);
        anim->createEllipticNodeTrack(sun, 50.0f, A_x, 50.0f, A_z);

        SLuint  res   = 30;
        SLNode* earth = new SLNode(new SLSphere(1, res, res, "Earth", matEarth));
        earth->rotate(90, -1, 0, 0);

        SLNode* scene = new SLNode;
        scene->addChild(sun);
        scene->addChild(earth);
        scene->addChild(cam1);

        sv->camera(cam1);
        s->root3D(scene);
    }
    else if (SLApplication::sceneID == SID_VolumeRayCast) //.............................................
    {
        s->name("Volume Ray Cast Test");
        s->info("Volume Rendering of an angiographic MRI scan");

        // Load volume data into 3D texture
        SLVstring mriImages;
        for (SLint i = 0; i < 207; ++i)
            mriImages.push_back(SLUtils::formatString("i%04u_0000b.png", i));

        SLint clamping3D = GL_CLAMP_TO_EDGE;
        if (SLGLState::getInstance()->getSLVersionNO() > "320")
            clamping3D = 0x812D; // GL_CLAMP_TO_BORDER

        SLGLTexture* texMRI = new SLGLTexture(mriImages,
                                              GL_LINEAR,
                                              GL_LINEAR,
                                              clamping3D,
                                              clamping3D,
                                              "mri_head_front_to_back");

        // Create transfer LUT 1D texture
        SLVTransferAlpha    tfAlphas = {SLTransferAlpha(0.00f, 0.00f),
                                     SLTransferAlpha(0.01f, 0.75f),
                                     SLTransferAlpha(1.00f, 1.00f)};
        SLTransferFunction* tf       = new SLTransferFunction(tfAlphas, CLUT_BCGYR);

        // Load shader and uniforms for volume size
        SLGLProgram*   sp   = new SLGLGenericProgram("VolumeRenderingRayCast.vert",
                                                 "VolumeRenderingRayCast.frag");
        SLGLUniform1f* volX = new SLGLUniform1f(UT_const, "u_volumeX", (SLfloat)texMRI->images()[0]->width());
        SLGLUniform1f* volY = new SLGLUniform1f(UT_const, "u_volumeY", (SLfloat)texMRI->images()[0]->height());
        SLGLUniform1f* volZ = new SLGLUniform1f(UT_const, "u_volumeZ", (SLfloat)mriImages.size());
        sp->addUniform1f(volX);
        sp->addUniform1f(volY);
        sp->addUniform1f(volZ);

        // Create volume rendering material
        SLMaterial* matVR = new SLMaterial("matVR", texMRI, tf, nullptr, nullptr, sp);

        // Create camera
        SLCamera* cam1 = new SLCamera("Camera 1");
        cam1->translation(0, 0, 3);
        cam1->lookAt(0, 0, 0);
        cam1->focalDist(3);
        cam1->background().colors(SLCol4f(0, 0, 0));
        cam1->setInitialState();

        // Set light
        SLLightSpot* light1 = new SLLightSpot(0.3f);
        light1->ambient(SLCol4f(0.1f, 0.1f, 0.1f));
        light1->diffuse(SLCol4f(1, 1, 1));
        light1->specular(SLCol4f(1, 1, 1));
        light1->attenuation(1, 0, 0);
        light1->translation(5, 5, 5);

        // Assemble scene with box node
        SLNode* scene = new SLNode("Scene");
        scene->addChild(light1);
        scene->addChild(new SLNode(new SLBox(-1, -1, -1, 1, 1, 1, "Box", matVR)));
        scene->addChild(cam1);

        sv->camera(cam1);
        s->root3D(scene);
    }
    else if (SLApplication::sceneID == SID_VolumeRayCastLighted) //......................................
    {
        s->name("Volume Ray Cast Lighted Test");
        s->info("Volume Rendering of an angiographic MRI scan with lighting");

        // Load volume data into 3D texture
        SLVstring mriImages;
        for (SLint i = 0; i < 207; ++i)
            mriImages.push_back(SLUtils::formatString("i%04u_0000b.png", i));

        SLint clamping3D = GL_CLAMP_TO_EDGE;
        if (SLGLState::getInstance()->getSLVersionNO() > "320")
            clamping3D = 0x812D; // GL_CLAMP_TO_BORDER

        SLGLTexture* texMRI = new SLGLTexture(mriImages,
                                              GL_LINEAR,
                                              GL_LINEAR,
                                              clamping3D,
                                              clamping3D,
                                              "mri_head_front_to_back",
                                              true);
        texMRI->calc3DGradients(1);

        // Create transfer LUT 1D texture
        SLVTransferAlpha    tfAlphas = {SLTransferAlpha(0.00f, 0.00f),
                                     SLTransferAlpha(0.01f, 0.75f),
                                     SLTransferAlpha(1.00f, 1.00f)};
        SLTransferFunction* tf       = new SLTransferFunction(tfAlphas, CLUT_BCGYR);

        // Load shader and uniforms for volume size
        SLGLProgram*   sp   = new SLGLGenericProgram("VolumeRenderingRayCast.vert",
                                                 "VolumeRenderingRayCastLighted.frag");
        SLGLUniform1f* volX = new SLGLUniform1f(UT_const, "u_volumeX", (SLfloat)texMRI->images()[0]->width());
        SLGLUniform1f* volY = new SLGLUniform1f(UT_const, "u_volumeY", (SLfloat)texMRI->images()[0]->height());
        SLGLUniform1f* volZ = new SLGLUniform1f(UT_const, "u_volumeZ", (SLfloat)mriImages.size());
        sp->addUniform1f(volX);
        sp->addUniform1f(volY);
        sp->addUniform1f(volZ);

        // Create volume rendering material
        SLMaterial* matVR = new SLMaterial("matVR", texMRI, tf, nullptr, nullptr, sp);

        // Create camera
        SLCamera* cam1 = new SLCamera("Camera 1");
        cam1->translation(0, 0, 3);
        cam1->lookAt(0, 0, 0);
        cam1->focalDist(3);
        cam1->background().colors(SLCol4f(0, 0, 0));
        cam1->setInitialState();

        // Set light
        SLLightSpot* light1 = new SLLightSpot(0.3f);
        light1->ambient(SLCol4f(0.1f, 0.1f, 0.1f));
        light1->diffuse(SLCol4f(1, 1, 1));
        light1->specular(SLCol4f(1, 1, 1));
        light1->attenuation(1, 0, 0);
        light1->translation(5, 5, 5);

        // Assemble scene with box node
        SLNode* scene = new SLNode("Scene");
        scene->addChild(light1);
        scene->addChild(new SLNode(new SLBox(-1, -1, -1, 1, 1, 1, "Box", matVR)));
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
        scene->addChild(cam1);

        // light
        SLLightSpot* light1 = new SLLightSpot(10, 10, 5, 0.5f);
        light1->ambient(SLCol4f(0.2f, 0.2f, 0.2f));
        light1->diffuse(SLCol4f(1, 1, 1));
        light1->specular(SLCol4f(1, 1, 1));
        light1->attenuation(1, 0, 0);
        scene->addChild(light1);

        // Floor grid
        SLMaterial* m2   = new SLMaterial(SLCol4f::WHITE);
        SLGrid*     grid = new SLGrid(SLVec3f(-5, 0, -5), SLVec3f(5, 0, 5), 20, 20, "Grid", m2);
        scene->addChild(new SLNode(grid, "grid"));

        // Astro boy character
        SLNode* char1 = importer.load("DAE/AstroBoy/AstroBoy.dae");
        char1->translate(-1, 0, 0);
        SLAnimPlayback* char1Anim = s->animManager().lastAnimPlayback();
        char1Anim->playForward();
        scene->addChild(char1);

        // Sintel character
        SLNode* char2 = importer.load("DAE/Sintel/SintelLowResOwnRig.dae"
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
        SLNode* cube1 = importer.load("DAE/SkinnedCube/skinnedcube2.dae");
        cube1->translate(3, 0, 0);
        SLAnimPlayback* cube1Anim = s->animManager().lastAnimPlayback();
        cube1Anim->easing(EC_inOutSine);
        cube1Anim->playForward();
        scene->addChild(cube1);

        // Skinned cube 2
        SLNode* cube2 = importer.load("DAE/SkinnedCube/skinnedcube4.dae");
        cube2->translate(-3, 0, 0);
        SLAnimPlayback* cube2Anim = s->animManager().lastAnimPlayback();
        cube2Anim->easing(EC_inOutSine);
        cube2Anim->playForward();
        scene->addChild(cube2);

        // Skinned cube 3
        SLNode* cube3 = importer.load("DAE/SkinnedCube/skinnedcube5.dae");
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
        SLGLTexture* tex1 = new SLGLTexture("Checkerboard0512_C.png");
        SLMaterial*  m1   = new SLMaterial("m1", tex1);
        m1->kr(0.5f);
        SLMaterial* m2 = new SLMaterial("m2", SLCol4f::WHITE * 0.5, SLCol4f::WHITE, 128, 0.5f, 0.0f, 1.0f);

        SLMesh* floorMesh = new SLRectangle(SLVec2f(-5, -5), SLVec2f(5, 5), 20, 20, "FloorMesh", m1);
        SLNode* floorRect = new SLNode(floorMesh);
        floorRect->rotate(90, -1, 0, 0);
        floorRect->translate(0, 0, -5.5f);

        // Bouncing balls
        SLNode* ball1 = new SLNode(new SLSphere(0.3f, 16, 16, "Ball1", m2));
        ball1->translate(0, 0, 4, TS_object);
        SLAnimation* ball1Anim = SLAnimation::create("Ball1_anim", 1.0f, true, EC_linear, AL_pingPongLoop);
        ball1Anim->createSimpleTranslationNodeTrack(ball1, SLVec3f(0.0f, -5.2f, 0.0f));

        SLNode* ball2 = new SLNode(new SLSphere(0.3f, 16, 16, "Ball2", m2));
        ball2->translate(-1.5f, 0, 4, TS_object);
        SLAnimation* ball2Anim = SLAnimation::create("Ball2_anim", 1.0f, true, EC_inQuad, AL_pingPongLoop);
        ball2Anim->createSimpleTranslationNodeTrack(ball2, SLVec3f(0.0f, -5.2f, 0.0f));

        SLNode* ball3 = new SLNode(new SLSphere(0.3f, 16, 16, "Ball3", m2));
        ball3->translate(-2.5f, 0, 4, TS_object);
        SLAnimation* ball3Anim = SLAnimation::create("Ball3_anim", 1.0f, true, EC_outQuad, AL_pingPongLoop);
        ball3Anim->createSimpleTranslationNodeTrack(ball3, SLVec3f(0.0f, -5.2f, 0.0f));

        SLNode* ball4 = new SLNode(new SLSphere(0.3f, 16, 16, "Ball4", m2));
        ball4->translate(1.5f, 0, 4, TS_object);
        SLAnimation* ball4Anim = SLAnimation::create("Ball4_anim", 1.0f, true, EC_inOutQuad, AL_pingPongLoop);
        ball4Anim->createSimpleTranslationNodeTrack(ball4, SLVec3f(0.0f, -5.2f, 0.0f));

        SLNode* ball5 = new SLNode(new SLSphere(0.3f, 16, 16, "Ball5", m2));
        ball5->translate(2.5f, 0, 4, TS_object);
        SLAnimation* ball5Anim = SLAnimation::create("Ball5_anim", 1.0f, true, EC_outInQuad, AL_pingPongLoop);
        ball5Anim->createSimpleTranslationNodeTrack(ball5, SLVec3f(0.0f, -5.2f, 0.0f));

        SLCamera* cam1 = new SLCamera("Camera 1");
        cam1->translation(0, 0, 22);
        cam1->lookAt(0, 0, 0);
        cam1->focalDist(22);
        cam1->setInitialState();

        SLCamera* cam2 = new SLCamera("Camera 2");
        cam2->translation(5, 0, 0);
        cam2->lookAt(0, 0, 0);
        cam2->focalDist(5);
        cam2->clipFar(10);
        cam2->background().colors(SLCol4f(0, 0, 0.6f), SLCol4f(0, 0, 0.3f));
        cam2->setInitialState();

        SLCamera* cam3 = new SLCamera("Camera 3");
        cam3->translation(-5, -2, 0);
        cam3->lookAt(0, 0, 0);
        cam3->focalDist(5);
        cam3->clipFar(10);
        cam3->background().colors(SLCol4f(0.6f, 0, 0), SLCol4f(0.3f, 0, 0));
        cam3->setInitialState();

        SLLightSpot* light1 = new SLLightSpot(0, 2, 0, 0.5f);
        light1->ambient(SLCol4f(0.2f, 0.2f, 0.2f));
        light1->diffuse(SLCol4f(0.9f, 0.9f, 0.9f));
        light1->specular(SLCol4f(0.9f, 0.9f, 0.9f));
        light1->attenuation(1, 0, 0);
        SLAnimation* light1Anim = SLAnimation::create("Light1_anim", 4.0f);
        light1Anim->createEllipticNodeTrack(light1, 6, A_z, 6, A_x);

        SLLightSpot* light2 = new SLLightSpot(0, 0, 0, 0.2f);
        light2->ambient(SLCol4f(0.2f, 0.0f, 0.0f));
        light2->diffuse(SLCol4f(0.9f, 0.0f, 0.0f));
        light2->specular(SLCol4f(0.9f, 0.9f, 0.9f));
        light2->attenuation(1, 0, 0);
        light2->translate(-8, -4, 0, TS_world);
        light2->setInitialState();

        SLAnimation*     light2Anim = SLAnimation::create("light2_anim", 2.0f, true, EC_linear, AL_pingPongLoop);
        SLNodeAnimTrack* track      = light2Anim->createNodeAnimationTrack();
        track->animatedNode(light2);
        track->createNodeKeyframe(0.0f);
        track->createNodeKeyframe(1.0f)->translation(SLVec3f(8, 8, 0));
        track->createNodeKeyframe(2.0f)->translation(SLVec3f(16, 0, 0));
        track->translationInterpolation(AI_bezier);

        SLNode* figure = BuildFigureGroup(m2, true);

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

        s->init();

        SLLightSpot* light1 = new SLLightSpot(0.1f);
        light1->translate(0, 10, 0);

        // build a basic scene to have a reference for the occuring rotations
        SLMaterial* genericMat = new SLMaterial("some material");

        // we use the same mesh to viasualize all the nodes
        SLBox* box = new SLBox(-0.5f, -0.5f, -0.5f, 0.5f, 0.5f, 0.5f, "box", genericMat);

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
        for (SLint lvl = 0; lvl < levels; ++lvl)
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
                    SLVec3f pos(x * nodeSpacing[lvl] * 1.1f, 1.5f, z * nodeSpacing[lvl] * 1.1f);

                    node->translate(pos, TS_object);
                    //node->scale(1.1f);

                    SLfloat       duration = 1.0f + 5.0f * ((SLfloat)i / (SLfloat)nodesPerLvl);
                    ostringstream oss;
                    ;
                    oss << "random anim " << nodeIndex++;
                    SLAnimation* anim = SLAnimation::create(oss.str(), duration, true, EC_inOutSine, AL_pingPongLoop);
                    anim->createSimpleTranslationNodeTrack(node, SLVec3f(0.0f, 1.0f, 0.0f));
                }
            }
        }
    }
    else if (SLApplication::sceneID == SID_AnimationArmy) //.............................................
    {
        s->name("Astroboy Army Test");
        s->info("Mass animation scene of identitcal Astroboy models");

        // Create materials
        SLMaterial* m1 = new SLMaterial("m1", SLCol4f::GRAY);
        m1->specular(SLCol4f::BLACK);

        // Define a light
        SLLightSpot* light1 = new SLLightSpot(100, 40, 100, 1);
        light1->ambient(SLCol4f(0.2f, 0.2f, 0.2f));
        light1->diffuse(SLCol4f(0.9f, 0.9f, 0.9f));
        light1->specular(SLCol4f(0.9f, 0.9f, 0.9f));
        light1->attenuation(1, 0, 0);

        // Define camera
        SLCamera* cam1 = new SLCamera;
        cam1->translation(0, 20, 20);
        cam1->lookAt(0, 0, 0);
        cam1->focalDist(cam1->translationOS().length());
        cam1->background().colors(SLCol4f(0.1f, 0.4f, 0.8f));
        cam1->setInitialState();

        // Floor rectangle
        SLNode* rect = new SLNode(new SLRectangle(SLVec2f(-20, -20),
                                                  SLVec2f(20, 20),
                                                  SLVec2f(0, 0),
                                                  SLVec2f(50, 50),
                                                  50,
                                                  50,
                                                  "Floor",
                                                  m1));
        rect->rotate(90, -1, 0, 0);

        SLAssimpImporter importer;
        SLNode*          center = importer.load("DAE/AstroBoy/AstroBoy.dae");
        s->animManager().lastAnimPlayback()->playForward();

        // Assemble scene
        SLNode* scene = new SLNode("scene group");
        scene->addChild(light1);
        scene->addChild(rect);
        scene->addChild(center);
        scene->addChild(cam1);

// create astroboys around the center astroboy
#ifdef SL_GLES2
        SLint size = 4;
#else
        SLint        size   = 8;
#endif
        for (SLint iZ = -size; iZ <= size; ++iZ)
        {
            for (SLint iX = -size; iX <= size; ++iX)
            {
                SLbool shift = iX % 2 != 0;
                if (iX != 0 || iZ != 0)
                {
                    SLNode* n  = new SLNode;
                    float   xt = float(iX) * 1.0f;
                    float   zt = float(iZ) * 1.0f + ((shift) ? 0.5f : 0.0f);
                    n->translate(xt, 0, zt, TS_object);
                    for (auto m : importer.meshes())
                        n->addMesh(m);
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
            s->videoType(VT_SCND); // on desktop it will be the main camera
        }
        else
        {
            s->name("File Video Texture");
            s->info("Minimal texture mapping example with video file source.");
            s->videoType(VT_FILE);
            SLCVCapture::videoFilename = "street3.mp4";
            SLCVCapture::videoLoops    = true;
        }

        // Back wall material with live video texture
        SLMaterial* m1 = new SLMaterial("VideoMat", s->videoTexture());

        // Create a root scene group for all nodes
        SLNode* scene = new SLNode("scene node");

        // Create a camera node
        SLCamera* cam1 = new SLCamera("Camera 1");
        cam1->translation(0, 0, 20);
        cam1->focalDist(20);
        cam1->lookAt(0, 0, 0);
        cam1->background().colors(SLCol4f(0.7f, 0.7f, 0.7f), SLCol4f(0.2f, 0.2f, 0.2f));
        cam1->setInitialState();
        scene->addChild(cam1);

        // Create rectangle meshe and nodes
        SLfloat h        = 5.0f;
        SLfloat w        = h * sv->scrWdivH();
        SLMesh* rectMesh = new SLRectangle(SLVec2f(-w, -h), SLVec2f(w, h), 1, 1, "rect mesh", m1);
        SLNode* rectNode = new SLNode(rectMesh, "rect node");
        rectNode->translation(0, 0, -5);
        scene->addChild(rectNode);

        // Center sphere
        SLNode* sphere = new SLNode(new SLSphere(2, 32, 32, "Sphere", m1));
        sphere->rotate(-90, 1, 0, 0);
        scene->addChild(sphere);

        // Create a light source node
        SLLightSpot* light1 = new SLLightSpot(0.3f);
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
        The tracking of markers is done in SLScene::onUpdate by calling the specific
        SLCVTracked::track method. If a marker was found it overwrites the linked nodes
        object matrix (SLNode::_om). If the linked node is the active camera the found
        transform is additionally inversed. This would be the standard augmented realtiy
        use case.
        The chessboard marker used in these scenes is also used for the camera
        calibration. The different calibration state changes are also handled in
        SLScene::onUpdate.
        */

        // Setup here only the requested scene.
        if (SLApplication::sceneID == SID_VideoTrackChessMain ||
            SLApplication::sceneID == SID_VideoTrackChessScnd)
        {
            if (SLApplication::sceneID == SID_VideoTrackChessMain)
            {
                s->videoType(VT_MAIN);
                s->name("Track Chessboard (main cam.)");
            }
            else
            {
                s->videoType(VT_SCND);
                s->name("Track Chessboard (scnd. cam.");
            }
        }
        else if (SLApplication::sceneID == SID_VideoCalibrateMain)
        {
            s->videoType(VT_MAIN);
            SLApplication::activeCalib->clear();

            s->name("Calibrate Main Cam.");
        }
        else if (SLApplication::sceneID == SID_VideoCalibrateScnd)
        {
            s->videoType(VT_SCND);
            SLApplication::activeCalib->clear();
            s->name("Calibrate Scnd. Cam.");
        }

        // Material
        SLMaterial* yellow = new SLMaterial("mY", SLCol4f(1, 1, 0, 0.5f));

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
        cam1->fov(SLApplication::activeCalib->cameraFovDeg());
        cam1->background().texture(s->videoTexture());
        cam1->setInitialState();
        scene->addChild(cam1);

        // Create a light source node
        SLLightSpot* light1 = new SLLightSpot(e1 * 0.5f);
        light1->translate(e9, e9, e9);
        light1->name("light node");
        scene->addChild(light1);

        // Build mesh & node
        if (SLApplication::sceneID == SID_VideoTrackChessMain ||
            SLApplication::sceneID == SID_VideoTrackChessScnd)
        {
            SLBox*  box     = new SLBox(0.0f, 0.0f, 0.0f, e3, e3, e3, "Box", yellow);
            SLNode* boxNode = new SLNode(box, "Box Node");
            boxNode->setDrawBitsRec(SL_DB_CULLOFF, true);
            SLNode* axisNode = new SLNode(new SLCoordAxis(), "Axis Node");
            axisNode->setDrawBitsRec(SL_DB_WIREMESH, false);
            axisNode->scale(e3);
            boxNode->addChild(axisNode);
            scene->addChild(boxNode);
        }

        // Create OpenCV Tracker for the box node
        s->trackers().push_back(new SLCVTrackedChessboard(cam1));

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
        The tracking of markers is done in SLScene::onUpdate by calling the specific
        SLCVTracked::track method. If a marker was found it overwrites the linked nodes
        object matrix (SLNode::_om). If the linked node is the active camera the found
        transform is additionally inversed. This would be the standard augmented realtiy
        use case.
        */

        if (SLApplication::sceneID == SID_VideoTrackArucoMain)
        {
            s->videoType(VT_MAIN);
            s->name("Track Aruco (main cam.)");
            s->info("Hold Aruco Marker 0 and/or 1 into the field of view of the main camera. You can find the Aruco markers in the file data/Calibrations/ArucoMarkersDict0_Marker0-9.pdf");
        }
        else
        {
            s->videoType(VT_SCND);
            s->name("Track Aruco (scnd. cam.)");
            s->info("Hold Aruco Marker 0 and/or 1 into the field of view of the secondary camera. You can find the Aruco markers in the file data/Calibrations/ArucoMarkersDict0_Marker0-9.pdf");
        }

        // Material
        SLMaterial* yellow = new SLMaterial("mY", SLCol4f(1, 1, 0, 0.5f));
        SLMaterial* cyan   = new SLMaterial("mY", SLCol4f(0, 1, 1, 0.5f));

        // Create a scene group node
        SLNode* scene = new SLNode("scene node");

        // Create a camera node 1
        SLCamera* cam1 = new SLCamera("Camera 1");
        cam1->translation(0, 0, 5);
        cam1->lookAt(0, 0, 0);
        cam1->fov(SLApplication::activeCalib->cameraFovDeg());
        cam1->background().texture(s->videoTexture());
        cam1->setInitialState();
        scene->addChild(cam1);

        // Create a light source node
        SLLightSpot* light1 = new SLLightSpot(0.02f);
        light1->translation(0.12f, 0.12f, 0.12f);
        light1->name("light node");
        scene->addChild(light1);

        // Get the half edge length of the aruco marker
        SLfloat edgeLen = SLCVTrackedAruco::params.edgeLength;
        SLfloat he      = edgeLen * 0.5f;

        // Build mesh & node that will be tracked by the 1st marker (camera)
        SLBox*  box1      = new SLBox(-he, -he, 0.0f, he, he, 2 * he, "Box 1", yellow);
        SLNode* boxNode1  = new SLNode(box1, "Box Node 1");
        SLNode* axisNode1 = new SLNode(new SLCoordAxis(), "Axis Node 1");
        axisNode1->setDrawBitsRec(SL_DB_WIREMESH, false);
        axisNode1->scale(edgeLen);
        boxNode1->addChild(axisNode1);
        boxNode1->setDrawBitsRec(SL_DB_CULLOFF, true);
        scene->addChild(boxNode1);

        // Build mesh & node that will be tracked by the 2nd marker
        SLBox*  box2      = new SLBox(-he, -he, 0.0f, he, he, 2 * he, "Box 2", cyan);
        SLNode* boxNode2  = new SLNode(box2, "Box Node 2");
        SLNode* axisNode2 = new SLNode(new SLCoordAxis(), "Axis Node 2");
        axisNode2->setDrawBitsRec(SL_DB_WIREMESH, false);
        axisNode2->scale(edgeLen);
        boxNode2->addChild(axisNode2);
        boxNode2->setDrawBitsRec(SL_DB_HIDDEN, true);
        boxNode2->setDrawBitsRec(SL_DB_CULLOFF, true);
        scene->addChild(boxNode2);

        // Create OpenCV Tracker for the camera & the 2nd box node
        s->trackers().push_back(new SLCVTrackedAruco(cam1, 0));
        s->trackers().push_back(new SLCVTrackedAruco(boxNode2, 1));

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
        The tracking of markers is done in SLScene::onUpdate by calling the specific
        SLCVTracked::track method. If a marker was found it overwrites the linked nodes
        object matrix (SLNode::_om). If the linked node is the active camera the found
        transform is additionally inversed. This would be the standard augmented realtiy
        use case.
        */

        s->name("Track 2D Features");
        s->info("Augmented Reality 2D Feature Tracking: You need to print out the stones image target from the file data/calibrations/vuforia_markers.pdf");

        SLCamera* cam1 = new SLCamera("Camera 1");
        cam1->translation(0, 2, 60);
        cam1->lookAt(15, 15, 0);
        cam1->clipNear(0.1f);
        cam1->clipFar(1000.0f); // Increase to infinity?
        cam1->setInitialState();
        cam1->background().texture(s->videoTexture());
        s->videoType(VT_MAIN);

        SLLightSpot* light1 = new SLLightSpot(420, 420, 420, 1);
        light1->ambient(SLCol4f(1, 1, 1));
        light1->diffuse(SLCol4f(1, 1, 1));
        light1->specular(SLCol4f(1, 1, 1));
        light1->attenuation(1, 0, 0);

        SLLightSpot* light2 = new SLLightSpot(-450, -340, 420, 1);
        light2->ambient(SLCol4f(1, 1, 1));
        light2->diffuse(SLCol4f(1, 1, 1));
        light2->specular(SLCol4f(1, 1, 1));
        light2->attenuation(1, 0, 0);

        SLLightSpot* light3 = new SLLightSpot(450, -370, 0, 1);
        light3->ambient(SLCol4f(1, 1, 1));
        light3->diffuse(SLCol4f(1, 1, 1));
        light3->specular(SLCol4f(1, 1, 1));
        light3->attenuation(1, 0, 0);

        // Coordinate axis node
        SLNode* axis = new SLNode(new SLCoordAxis(), "Axis Node");
        axis->setDrawBitsRec(SL_DB_WIREMESH, false);
        axis->scale(100);
        axis->rotate(-90, 1, 0, 0);

        // Yellow center box
        SLMaterial* yellow = new SLMaterial("mY", SLCol4f(1, 1, 0, 0.5f));
        SLNode*     box    = new SLNode(new SLBox(0, 0, 0, 100, 100, 100, "Box", yellow), "Box Node");
        box->rotate(-90, 1, 0, 0);

        // Scene structure
        SLNode* scene = new SLNode("Scene");
        scene->addChild(light1);
        scene->addChild(light2);
        scene->addChild(light3);
        scene->addChild(axis);
        scene->addChild(box);
        scene->addChild(cam1);

        s->trackers().push_back(new SLCVTrackedFeatures(cam1, "features_stones.png"));

        sv->doWaitOnIdle(false); // for constant video feed
        sv->camera(cam1);

        s->root3D(scene);
        SLApplication::devRot.isUsed(true);
    }
    else if (SLApplication::sceneID == SID_VideoTrackFaceMain ||
             SLApplication::sceneID == SID_VideoTrackFaceScnd) //........................................
    {
        /*
        The tracking of markers is done in SLScene::onUpdate by calling the specific
        SLCVTracked::track method. If a marker was found it overwrites the linked nodes
        object matrix (SLNode::_om). If the linked node is the active camera the found
        transform is additionally inversed. This would be the standard augmented realtiy
        use case.
        */

        if (SLApplication::sceneID == SID_VideoTrackFaceMain)
        {
            s->videoType(VT_MAIN);
            s->name("Track Face (main cam.)");
        }
        else
        {
            s->videoType(VT_SCND);
            s->name("Track Face (scnd. cam.)");
        }
        s->info("Face and facial landmark detection.");

        SLCamera* cam1 = new SLCamera("Camera 1");
        cam1->translation(0, 0, 0.5f);
        cam1->clipNear(0.1f);
        cam1->clipFar(1000.0f); // Increase to infinity?
        cam1->setInitialState();
        cam1->background().texture(s->videoTexture());

        SLLightSpot* light1 = new SLLightSpot(10, 10, 10, 1);
        light1->ambient(SLCol4f(1, 1, 1));
        light1->diffuse(SLCol4f(1, 1, 1));
        light1->specular(SLCol4f(1, 1, 1));
        light1->attenuation(1, 0, 0);

        // Load sunglasses
        SLAssimpImporter importer;
        SLNode*          glasses = importer.load("FBX/Sunglasses.fbx");
        glasses->scale(0.01f);

        // Add axis arrows at world center
        SLNode* axis = new SLNode(new SLCoordAxis(), "Axis Node");
        axis->setDrawBitsRec(SL_DB_WIREMESH, false);
        axis->scale(0.03f);

        // Scene structure
        SLNode* scene = new SLNode("Scene");
        scene->addChild(light1);
        scene->addChild(cam1);
        scene->addChild(glasses);
        scene->addChild(axis);

        // Add a face tracker that moves the camera node
        s->trackers().push_back(new SLCVTrackedFaces(cam1, 3));

        s->showDetection(true);

        sv->doWaitOnIdle(false); // for constant video feed
        sv->camera(cam1);

        s->root3D(scene);
    }
    else if (SLApplication::sceneID == SID_VideoSensorAR) //.............................................
    {
        // Set scene name and info string
        s->name("Video Sensor AR");
        s->info("Minimal scene to test the devices IMU and GPS Sensors. See the sensor information. GPS needs a few sec. to improve the accuracy.");

        SLCamera* cam1 = new SLCamera("Camera 1");
        cam1->translation(0, 0, 60);
        cam1->lookAt(0, 0, 0);
        cam1->fov(SLApplication::activeCalib->cameraFovDeg());
        cam1->clipNear(0.1f);
        cam1->clipFar(10000.0f);
        cam1->setInitialState();

        // Turn on main video
        cam1->background().texture(s->videoTexture());
        s->videoType(VT_MAIN);

        // Create directional light for the sun light
        SLLightDirect* light = new SLLightDirect(1.0f);
        light->ambient(SLCol4f(1, 1, 1));
        light->diffuse(SLCol4f(1, 1, 1));
        light->specular(SLCol4f(1, 1, 1));
        light->attenuation(1, 0, 0);

        // Let the sun be rotated by time and location
        SLApplication::devLoc.sunLightNode(light);

        SLNode* axis = new SLNode(new SLCoordAxis(), "Axis Node");
        axis->setDrawBitsRec(SL_DB_WIREMESH, false);
        axis->scale(2);
        axis->rotate(-90, 1, 0, 0);

        // Yellow center box
        SLMaterial* yellow = new SLMaterial("mY", SLCol4f(1, 1, 0, 0.5f));
        SLNode*     box    = new SLNode(new SLBox(-.5f, -.5f, -.5f, .5f, .5f, .5f, "Box", yellow), "Box Node");

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
    else if (SLApplication::sceneID == SID_VideoChristoffel) //..........................................
    {
        s->name("Christoffel Tower AR");
        s->info("Augmented Reality Christoffel Tower");

        SLCamera* cam1 = new SLCamera("Camera 1");
        cam1->translation(0, 2, 0);
        cam1->lookAt(-10, 2, 0);
        cam1->clipNear(0.1f);
        cam1->clipFar(500.0f);
        cam1->setInitialState();

        // Turn on main video
        cam1->background().texture(s->videoTexture());
        s->videoType(VT_MAIN);

        // Create directional light for the sun light
        SLLightDirect* light = new SLLightDirect(5.0f);
        light->ambient(SLCol4f(1, 1, 1));
        light->diffuse(SLCol4f(1, 1, 1));
        light->specular(SLCol4f(1, 1, 1));
        light->attenuation(1, 0, 0);

        // Let the sun be rotated by time and location
        SLApplication::devLoc.sunLightNode(light);

        SLAssimpImporter importer;
        SLNode*          bern = importer.load("FBX/Christoffel/Bern-Bahnhofsplatz.fbx");

        // Make city transparent
        SLNode* UmgD = bern->findChild<SLNode>("Umgebung-Daecher");
        if (!UmgD) SL_EXIT_MSG("Node: Umgebung-Daecher not found!");
        for (auto mesh : UmgD->meshes())
        {
            mesh->mat()->kt(0.5f);
            mesh->mat()->ambient(SLCol4f(0.3f, 0.3f, 0.3f));
            mesh->init(UmgD); // reset the correct hasAlpha flag
        }

        SLNode* UmgF = bern->findChild<SLNode>("Umgebung-Fassaden");
        if (!UmgF) SL_EXIT_MSG("Node: Umgebung-Fassaden not found!");
        for (auto mesh : UmgF->meshes())
        {
            mesh->mat()->kt(0.5f);
            mesh->mat()->ambient(SLCol4f(0.3f, 0.3f, 0.3f));
            mesh->init(UmgF); // reset the correct hasAlpha flag
        }

        SLNode* ChrA = bern->findChild<SLNode>("Christoffel-Aussen");
        if (!ChrA) SL_EXIT_MSG("Node: Christoffel-Aussen not found!");
        for (auto mesh : ChrA->meshes())
        {
            mesh->mat()->ambient(SLCol4f(0.3f, 0.3f, 0.3f));
        }

        // Hide some objects
        bern->findChild<SLNode>("Umgebung-Daecher")->drawBits()->set(SL_DB_HIDDEN, true);
        bern->findChild<SLNode>("Umgebung-Fassaden")->drawBits()->set(SL_DB_HIDDEN, true);
        bern->findChild<SLNode>("Baldachin-Glas")->drawBits()->set(SL_DB_HIDDEN, true);
        bern->findChild<SLNode>("Baldachin-Stahl")->drawBits()->set(SL_DB_HIDDEN, true);
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

        // Set ambient on all child nodes
        for (auto node : bern->children())
        {
            for (auto mesh : node->meshes())
            {
                mesh->mat()->ambient(SLCol4f(0.3f, 0.3f, 0.3f));
            }
        }

        // Add axis object a world origin (Loeb Ecke)
        SLNode* axis = new SLNode(new SLCoordAxis(), "Axis Node");
        axis->setDrawBitsRec(SL_DB_WIREMESH, false);
        axis->scale(10);
        axis->rotate(-90, 1, 0, 0);

        SLNode* scene = new SLNode("Scene");
        scene->addChild(light);
        scene->addChild(axis);
        scene->addChild(bern);
        scene->addChild(cam1);

        //initialize sensor stuff
        SLApplication::devLoc.originLLA(46.947629, 7.440754, 442.0);        // Loeb Ecken
        SLApplication::devLoc.defaultLLA(46.948551, 7.440093, 442.0 + 1.7); // Bahnhof Ausgang in Augenhöhe
        SLApplication::devLoc.locMaxDistanceM(1000.0f);                     // Max. Distanz. zum Loeb Ecken
        SLApplication::devLoc.improveOrigin(false);                         // Keine autom. Verbesserung vom Origin
        SLApplication::devLoc.useOriginAltitude(true);
        SLApplication::devLoc.hasOrigin(true);
        SLApplication::devRot.zeroYawAtStart(false);

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
        cam1->lookAt(SLVec3f::ZERO);
        cam1->camAnim(SLCamAnim::CA_turntableYUp);
#endif

        sv->doWaitOnIdle(false); // for constant video feed
        sv->camera(cam1);
        s->root3D(scene);
    }
    else if (SLApplication::sceneID == SID_RTMuttenzerBox) //............................................
    {
        s->name("Muttenzer Box (RT)");
        s->info("Muttenzer Box with environment mapped reflective sphere and transparenz refractive glass sphere. Try ray tracing for real reflections and soft shadows.");

        // Create reflection & glass shaders
        SLGLProgram* sp1 = new SLGLGenericProgram("Reflect.vert", "Reflect.frag");
        SLGLProgram* sp2 = new SLGLGenericProgram("RefractReflect.vert", "RefractReflect.frag");

        // Create cube mapping texture
        SLGLTexture* tex1 = new SLGLTexture("MuttenzerBox+X0512_C.png", "MuttenzerBox-X0512_C.png", "MuttenzerBox+Y0512_C.png", "MuttenzerBox-Y0512_C.png", "MuttenzerBox+Z0512_C.png", "MuttenzerBox-Z0512_C.png");

        SLCol4f lightEmisRGB(7.0f, 7.0f, 7.0f);
        SLCol4f grayRGB(0.75f, 0.75f, 0.75f);
        SLCol4f redRGB(0.75f, 0.25f, 0.25f);
        SLCol4f blueRGB(0.25f, 0.25f, 0.75f);
        SLCol4f blackRGB(0.00f, 0.00f, 0.00f);

        // create materials
        SLMaterial* cream = new SLMaterial("cream", grayRGB, SLCol4f::BLACK, 0);
        SLMaterial* red   = new SLMaterial("red", redRGB, SLCol4f::BLACK, 0);
        SLMaterial* blue  = new SLMaterial("blue", blueRGB, SLCol4f::BLACK, 0);

        // Material for mirror sphere
        SLMaterial* refl = new SLMaterial("refl", blackRGB, SLCol4f::WHITE, 1000, 1.0f);
        refl->textures().push_back(tex1);
        refl->program(sp1);

        // Material for glass sphere
        SLMaterial* refr = new SLMaterial("refr", blackRGB, blackRGB, 100, 0.05f, 0.95f, 1.5f);
        refr->translucency(1000);
        refr->transmissiv(SLCol4f::WHITE);
        refr->textures().push_back(tex1);
        refr->program(sp2);

        SLNode* sphere1 = new SLNode(new SLSphere(0.5f, 32, 32, "Sphere1", refl));
        sphere1->translate(-0.65f, -0.75f, -0.55f, TS_object);

        SLNode* sphere2 = new SLNode(new SLSphere(0.45f, 32, 32, "Sphere2", refr));
        sphere2->translate(0.73f, -0.8f, 0.10f, TS_object);

        SLNode* balls = new SLNode;
        balls->addChild(sphere1);
        balls->addChild(sphere2);

        // Rectangular light
        SLLightRect* lightRect = new SLLightRect(1, 0.65f);
        lightRect->rotate(90, -1.0f, 0.0f, 0.0f);
        lightRect->translate(0.0f, -0.25f, 1.18f, TS_object);
        lightRect->spotCutOffDEG(90);
        lightRect->spotExponent(1.0);
        lightRect->ambient(SLCol4f::BLACK);
        lightRect->diffuse(lightEmisRGB);
        //lightRect->specular(SLCol4f::BLACK);
        lightRect->attenuation(0, 0, 1);
        lightRect->samplesXY(11, 7);

        //_globalAmbiLight.set(SLCol4f::BLACK);
        s->globalAmbiLight().set(lightEmisRGB * 0.05f);

        // create camera
        SLCamera* cam1 = new SLCamera();
        cam1->translation(0.0f, 0.40f, 6.35f);
        cam1->lookAt(0.0f, -0.05f, 0.0f);
        cam1->fov(27);
        cam1->focalDist(cam1->translationOS().length());
        cam1->background().colors(SLCol4f(0.0f, 0.0f, 0.0f));
        cam1->setInitialState();

        // assemble scene
        SLNode* scene = new SLNode;
        scene->addChild(cam1);
        scene->addChild(lightRect);

        // create wall polygons
        SLfloat pL = -1.48f, pR = 1.48f; // left/right
        SLfloat pB = -1.25f, pT = 1.19f; // bottom/top
        SLfloat pN = 1.79f, pF = -1.55f; // near/far

        // bottom plane
        SLNode* b = new SLNode(new SLRectangle(SLVec2f(pL, -pN), SLVec2f(pR, -pF), 6, 6, "bottom", cream));
        b->rotate(90, -1, 0, 0);
        b->translate(0, 0, pB, TS_object);
        scene->addChild(b);

        // top plane
        SLNode* t = new SLNode(new SLRectangle(SLVec2f(pL, pF), SLVec2f(pR, pN), 6, 6, "top", cream));
        t->rotate(90, 1, 0, 0);
        t->translate(0, 0, -pT, TS_object);
        scene->addChild(t);

        // far plane
        SLNode* f = new SLNode(new SLRectangle(SLVec2f(pL, pB), SLVec2f(pR, pT), 6, 6, "far", cream));
        f->translate(0, 0, pF, TS_object);
        scene->addChild(f);

        // left plane
        SLNode* l = new SLNode(new SLRectangle(SLVec2f(-pN, pB), SLVec2f(-pF, pT), 6, 6, "left", red));
        l->rotate(90, 0, 1, 0);
        l->translate(0, 0, pL, TS_object);
        scene->addChild(l);

        // right plane
        SLNode* r = new SLNode(new SLRectangle(SLVec2f(pF, pB), SLVec2f(pN, pT), 6, 6, "right", blue));
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
        SLMaterial* matGla = new SLMaterial("Glass", SLCol4f(0.0f, 0.0f, 0.0f), SLCol4f(0.5f, 0.5f, 0.5f), 100, 0.4f, 0.6f, 1.5f);
        SLMaterial* matRed = new SLMaterial("Red", SLCol4f(0.5f, 0.0f, 0.0f), SLCol4f(0.5f, 0.5f, 0.5f), 100, 0.5f, 0.0f, 1.0f);
        SLMaterial* matYel = new SLMaterial("Floor", SLCol4f(0.8f, 0.6f, 0.2f), SLCol4f(0.8f, 0.8f, 0.8f), 100, 0.5f, 0.0f, 1.0f);

        SLCamera* cam1 = new SLCamera();
        cam1->translation(0, 0.1f, 2.5f);
        cam1->lookAt(0, 0, 0);
        cam1->focalDist(cam1->translationOS().length());
        cam1->background().colors(SLCol4f(0.1f, 0.4f, 0.8f));
        cam1->setInitialState();

        SLNode* rect = new SLNode(new SLRectangle(SLVec2f(-3, -3), SLVec2f(5, 4), 20, 20, "Floor", matYel));
        rect->rotate(90, -1, 0, 0);
        rect->translate(0, -1, -0.5f, TS_object);

        SLLightSpot* light1 = new SLLightSpot(2, 2, 2, 0.1f);
        light1->ambient(SLCol4f(1, 1, 1));
        light1->diffuse(SLCol4f(7, 7, 7));
        light1->specular(SLCol4f(7, 7, 7));
        light1->attenuation(0, 0, 1);

        SLLightSpot* light2 = new SLLightSpot(2, 2, -2, 0.1f);
        light2->ambient(SLCol4f(1, 1, 1));
        light2->diffuse(SLCol4f(7, 7, 7));
        light2->specular(SLCol4f(7, 7, 7));
        light2->attenuation(0, 0, 1);

        SLNode* scene = new SLNode;
        scene->addChild(light1);
        scene->addChild(light2);
        scene->addChild(SphereGroup(1, 0, 0, 0, 1, 30, matGla, matRed));
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
        SLCol4f     spec(0.8f, 0.8f, 0.8f);
        SLMaterial* matBlk = new SLMaterial("Glass", SLCol4f(0.0f, 0.0f, 0.0f), SLCol4f(0.5f, 0.5f, 0.5f), 100, 0.5f, 0.5f, 1.5f);
        SLMaterial* matRed = new SLMaterial("Red", SLCol4f(0.5f, 0.0f, 0.0f), SLCol4f(0.5f, 0.5f, 0.5f), 100, 0.5f, 0.0f, 1.0f);
        SLMaterial* matYel = new SLMaterial("Floor", SLCol4f(0.8f, 0.6f, 0.2f), SLCol4f(0.8f, 0.8f, 0.8f), 100, 0.0f, 0.0f, 1.0f);

        SLCamera* cam1 = new SLCamera;
        cam1->translation(0, 0.1f, 6);
        cam1->lookAt(0, 0, 0);
        cam1->focalDist(cam1->translationOS().length());
        cam1->background().colors(SLCol4f(0.1f, 0.4f, 0.8f));
        cam1->setInitialState();

        SLuint  res  = 30;
        SLNode* rect = new SLNode(new SLRectangle(SLVec2f(-3, -3), SLVec2f(5, 4), res, res, "Rect", matYel));
        rect->rotate(90, -1, 0, 0);
        rect->translate(0, -1, -0.5f, TS_object);

        SLLightSpot* light1 = new SLLightSpot(3, 3, 3, 0.3f);
#ifndef SL_GLES2
        SLuint numSamples = 10;
#else
        SLuint numSamples = 8;
#endif
        light1->samples(numSamples, numSamples);
        light1->attenuation(0, 0, 1);
        //light1->lightAt(2,2,2, 0,0,0);
        //light1->spotCutoff(15);
        light1->translation(2, 2, 2);
        light1->lookAt(0, 0, 0);

        SLLightSpot* light2 = new SLLightSpot(0, 1.5, -1.5, 0.3f);
        light2->samples(8, 8);
        light2->attenuation(0, 0, 1);

        SLNode* scene = new SLNode;
        scene->addChild(light1);
        scene->addChild(light2);
        scene->addChild(SphereGroup(1, 0, 0, 0, 1, res, matBlk, matRed));
        scene->addChild(rect);
        scene->addChild(cam1);

        sv->camera(cam1);
        s->root3D(scene);
    }
    else if (SLApplication::sceneID == SID_RTDoF) //.....................................................
    {
        s->name("Ray tracing depth of field");

        // Create textures and materials
        SLGLTexture* texC = new SLGLTexture("Checkerboard0512_C.png");
        SLMaterial*  mT   = new SLMaterial("mT", texC, nullptr, nullptr, nullptr);
        mT->kr(0.5f);
        SLMaterial* mW = new SLMaterial("mW", SLCol4f::WHITE);
        SLMaterial* mB = new SLMaterial("mB", SLCol4f::GRAY);
        SLMaterial* mY = new SLMaterial("mY", SLCol4f::YELLOW);
        SLMaterial* mR = new SLMaterial("mR", SLCol4f::RED);
        SLMaterial* mG = new SLMaterial("mG", SLCol4f::GREEN);
        SLMaterial* mM = new SLMaterial("mM", SLCol4f::MAGENTA);

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
        cam1->lensDiameter(0.4f);
        cam1->lensSamples()->samples(numSamples, numSamples);
        cam1->background().colors(SLCol4f(0.1f, 0.4f, 0.8f));
        cam1->setInitialState();

        SLuint  res  = 30;
        SLNode* rect = new SLNode(new SLRectangle(SLVec2f(-5, -5), SLVec2f(5, 5), res, res, "Rect", mT));
        rect->rotate(90, -1, 0, 0);
        rect->translate(0, 0, -0.5f, TS_object);

        SLLightSpot* light1 = new SLLightSpot(2, 2, 0, 0.1f);
        light1->attenuation(0, 0, 1);

        SLNode* balls = new SLNode;
        SLNode* sp;
        sp = new SLNode(new SLSphere(0.5f, res, res, "S1", mW));
        sp->translate(2.0, 0, -4, TS_object);
        balls->addChild(sp);
        sp = new SLNode(new SLSphere(0.5f, res, res, "S2", mB));
        sp->translate(1.5, 0, -3, TS_object);
        balls->addChild(sp);
        sp = new SLNode(new SLSphere(0.5f, res, res, "S3", mY));
        sp->translate(1.0, 0, -2, TS_object);
        balls->addChild(sp);
        sp = new SLNode(new SLSphere(0.5f, res, res, "S4", mR));
        sp->translate(0.5, 0, -1, TS_object);
        balls->addChild(sp);
        sp = new SLNode(new SLSphere(0.5f, res, res, "S5", mG));
        sp->translate(0.0, 0, 0, TS_object);
        balls->addChild(sp);
        sp = new SLNode(new SLSphere(0.5f, res, res, "S6", mM));
        sp->translate(-0.5, 0, 1, TS_object);
        balls->addChild(sp);
        sp = new SLNode(new SLSphere(0.5f, res, res, "S7", mW));
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
        SLGLTexture* texC = new SLGLTexture("VisionExample.png");
        //SLGLTexture* texC = new SLGLTexture("Checkerboard0512_C.png");

        SLMaterial* mT = new SLMaterial("mT", texC, nullptr, nullptr, nullptr);
        mT->kr(0.5f);

        // Glass material
        // name, ambient, specular,	shininess, kr(reflectivity), kt(transparency), kn(refraction)
        SLMaterial* matLens = new SLMaterial("lens", SLCol4f(0.0f, 0.0f, 0.0f), SLCol4f(0.5f, 0.5f, 0.5f), 100, 0.5f, 0.5f, 1.5f);
        //SLGLShaderProg* sp1 = new SLGLShaderProgGeneric("RefractReflect.vert", "RefractReflect.frag");
        //matLens->shaderProg(sp1);

#ifndef SL_GLES2
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

        // Light
        //SLLightSpot* light1 = new SLLightSpot(15, 20, 15, 0.1f);
        //light1->attenuation(0, 0, 1);

        // Plane
        //SLNode* rect = new SLNode(new SLRectangle(SLVec2f(-20, -20), SLVec2f(20, 20), 50, 20, "Rect", mT));
        //rect->translate(0, 0, 0, TS_Object);
        //rect->rotate(90, -1, 0, 0);

        SLLightSpot* light1 = new SLLightSpot(1, 6, 1, 0.1f);
        light1->attenuation(0, 0, 1);

        SLuint  res  = 20;
        SLNode* rect = new SLNode(new SLRectangle(SLVec2f(-5, -5), SLVec2f(5, 5), res, res, "Rect", mT));
        rect->rotate(90, -1, 0, 0);
        rect->translate(0, 0, -0.0f, TS_object);

        // Lens from eye prescription card
        //SLNode* lensA = new SLNode(new SLLens(0.50f, -0.50f, 4.0f, 0.0f, 32, 32, "presbyopic", matLens));   // Weitsichtig
        //lensA->translate(-2, 1, -2);
        //SLNode* lensB = new SLNode(new SLLens(-0.65f, -0.10f, 4.0f, 0.0f, 32, 32, "myopic", matLens));      // Kurzsichtig
        //lensB->translate(2, 1, -2);

        // Lens with radius
        //SLNode* lensC = new SLNode(new SLLens(5.0, 4.0, 4.0f, 0.0f, 32, 32, "presbyopic", matLens));        // Weitsichtig
        //lensC->translate(-2, 1, 2);
        SLNode* lensD = new SLNode(new SLLens(-15.0f, -15.0f, 1.0f, 0.1f, res, res, "myopic", matLens)); // Kurzsichtig
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

        // Create a light source node
        SLLightSpot* light1 = new SLLightSpot(0.3f);
        light1->translation(5, 5, 5);
        light1->lookAt(0, 0, 0);
        light1->name("light node");

        // Material for glass sphere
        SLMaterial* matBox1  = new SLMaterial("matBox1", SLCol4f(0.0f, 0.0f, 0.0f), SLCol4f(0.5f, 0.5f, 0.5f), 100, 0.0f, 0.9f, 1.5f);
        SLMesh*     boxMesh1 = new SLBox(-0.8f, -1, 0.02f, 1.2f, 1, 1, "boxMesh1", matBox1);
        SLNode*     boxNode1 = new SLNode(boxMesh1, "BoxNode1");

        SLMaterial* matBox2  = new SLMaterial("matBox2", SLCol4f(0.0f, 0.0f, 0.0f), SLCol4f(0.5f, 0.5f, 0.5f), 100, 0.0f, 0.9f, 1.3f);
        SLMesh*     boxMesh2 = new SLBox(-1.2f, -1, -1, 0.8f, 1, -0.02f, "BoxMesh2", matBox2);
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
    for (auto sv : s->sceneViews())
    {
        if (sv != nullptr)
        {
            sv->onInitialize();
        }
    }

    s->onAfterLoad();
}
//-----------------------------------------------------------------------------
//! Creates a recursive sphere group used for the ray tracing scenes
SLNode* SphereGroup(SLint       depth, // depth of recursion
                    SLfloat     x,
                    SLfloat     y,
                    SLfloat     z,          // position of group
                    SLfloat     scale,      // scale factor
                    SLuint      resolution, // resolution of spheres
                    SLMaterial* matGlass,   // material for center sphere
                    SLMaterial* matRed)     // material for orbiting spheres
{
    SLstring name = matGlass->kt() > 0 ? "GlassSphere" : "RedSphere";
    if (depth == 0)
    {
        SLNode* s = new SLNode(new SLSphere(0.5f * scale, resolution, resolution, name, matRed));
        s->translate(x, y, z, TS_object);
        return s;
    }
    else
    {
        depth--;
        SLNode* sGroup = new SLNode("SphereGroup");
        sGroup->translate(x, y, z, TS_object);
        SLuint newRes = (SLuint)SL_max((SLint)resolution - 8, 8);
        sGroup->addChild(new SLNode(new SLSphere(0.5f * scale, resolution, resolution, name, matGlass)));
        sGroup->addChild(SphereGroup(depth, 0.643951f * scale, 0, 0.172546f * scale, scale / 3, newRes, matRed, matRed));
        sGroup->addChild(SphereGroup(depth, 0.172546f * scale, 0, 0.643951f * scale, scale / 3, newRes, matRed, matRed));
        sGroup->addChild(SphereGroup(depth, -0.471405f * scale, 0, 0.471405f * scale, scale / 3, newRes, matRed, matRed));
        sGroup->addChild(SphereGroup(depth, -0.643951f * scale, 0, -0.172546f * scale, scale / 3, newRes, matRed, matRed));
        sGroup->addChild(SphereGroup(depth, -0.172546f * scale, 0, -0.643951f * scale, scale / 3, newRes, matRed, matRed));
        sGroup->addChild(SphereGroup(depth, 0.471405f * scale, 0, -0.471405f * scale, scale / 3, newRes, matRed, matRed));
        sGroup->addChild(SphereGroup(depth, 0.272166f * scale, 0.544331f * scale, 0.272166f * scale, scale / 3, newRes, matRed, matRed));
        sGroup->addChild(SphereGroup(depth, -0.371785f * scale, 0.544331f * scale, 0.099619f * scale, scale / 3, newRes, matRed, matRed));
        sGroup->addChild(SphereGroup(depth, 0.099619f * scale, 0.544331f * scale, -0.371785f * scale, scale / 3, newRes, matRed, matRed));
        return sGroup;
    }
}
//-----------------------------------------------------------------------------
//! Build a hierarchical figurine with arms and legs
SLNode* BuildFigureGroup(SLMaterial* mat, SLbool withAnimation)
{
    SLNode* cyl;
    SLuint  res = 16;

    // Feet
    SLNode* feet = new SLNode("feet group (T13,R6)");
    feet->addMesh(new SLSphere(0.2f, 16, 16, "ankle", mat));
    SLNode* feetbox = new SLNode(new SLBox(-0.2f, -0.1f, 0.0f, 0.2f, 0.1f, 0.8f, "foot", mat), "feet (T14)");
    feetbox->translate(0.0f, -0.25f, -0.15f, TS_object);
    feet->addChild(feetbox);
    feet->translate(0.0f, 0.0f, 1.6f, TS_object);
    feet->rotate(-90.0f, 1.0f, 0.0f, 0.0f);

    // Assemble low leg
    SLNode* leglow = new SLNode("low leg group (T11, R5)");
    leglow->addMesh(new SLSphere(0.3f, res, res, "knee", mat));
    cyl = new SLNode(new SLCylinder(0.2f, 1.4f, 1, res, false, false, "shin", mat), "shin (T12)");
    cyl->translate(0.0f, 0.0f, 0.2f, TS_object);
    leglow->addChild(cyl);
    leglow->addChild(feet);
    leglow->translate(0.0f, 0.0f, 1.27f, TS_object);
    leglow->rotate(0, 1.0f, 0.0f, 0.0f);

    // Assemble leg
    SLNode* leg = new SLNode("leg group ()");
    leg->addMesh(new SLSphere(0.4f, res, res, "hip joint", mat));
    cyl = new SLNode(new SLCylinder(0.3f, 1.0f, 1, res, false, false, "thigh", mat), "thigh (T10)");
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
    armlow->addMesh(new SLSphere(0.2f, 16, 16, "elbow", mat));
    cyl = new SLNode(new SLCylinder(0.15f, 1.0f, 1, res, true, false, "low arm", mat), "T7");
    cyl->translate(0.0f, 0.0f, 0.14f, TS_object);
    armlow->addChild(cyl);
    armlow->translate(0.0f, 0.0f, 1.2f, TS_object);
    armlow->rotate(45, -1.0f, 0.0f, 0.0f);

    // Assemble arm
    SLNode* arm = new SLNode("arm group ()");
    arm->addMesh(new SLSphere(0.3f, 16, 16, "shoulder", mat));
    cyl = new SLNode(new SLCylinder(0.2f, 1.0f, 1, res, false, false, "upper arm", mat), "upper arm (T5)");
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
    SLNode* head = new SLNode(new SLSphere(0.5f, res, res, "head", mat), "head (T1)");
    head->translate(0.0f, 0.0f, -0.7f, TS_object);
    SLNode* neck = new SLNode(new SLCylinder(0.25f, 0.3f, 1, res, false, false, "neck", mat), "neck (T2)");
    neck->translate(0.0f, 0.0f, -0.3f, TS_object);

    // Assemble figure Left
    SLNode* figure = new SLNode("figure group (R1)");
    figure->addChild(new SLNode(new SLBox(-0.8f, -0.4f, 0.0f, 0.8f, 0.4f, 2.0f, "chest", mat), "chest"));
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
        SLAnimation* anim = SLAnimation::create("figure animation", 2.0f, true, EC_inOutQuint, AL_pingPongLoop);
        anim->createSimpleRotationNodeTrack(legLeft, 60, SLVec3f(1, 0, 0));

        SLNode* legLowLeft = legLeft->findChild<SLNode>("low leg group (T11, R5)");
        anim->createSimpleRotationNodeTrack(legLowLeft, 40, SLVec3f(1, 0, 0));

        SLNode* feetLeft = legLeft->findChild<SLNode>("feet group (T13,R6)");
        anim->createSimpleRotationNodeTrack(feetLeft, 40, SLVec3f(1, 0, 0));
    }

    return figure;
}
//-----------------------------------------------------------------------------
