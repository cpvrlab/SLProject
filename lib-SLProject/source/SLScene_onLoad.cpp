//#############################################################################
//  File:      SLScene_onLoad.cpp
//  Author:    Marcus Hudritsch
//  Date:      July 2014
//  Codestyle: https://github.com/cpvrlab/SLProject/wiki/Coding-Style-Guidelines
//  Copyright: Marcus Hudritsch
//             This software is provide under the GNU General Public License
//             Please visit: http://opensource.org/licenses/GPL-3.0
//#############################################################################

#include <stdafx.h>           // precompiled headers
#ifdef SL_MEMLEAKDETECT       // set in SL.h for debug config only
#include <debug_new.h>        // memory leak detector
#endif

#include <SLScene.h>
#include <SLSceneView.h>
#include <SLKeyframe.h>
#include <SLAnimation.h>
#include <SLAnimManager.h>
#include <SLAssimpImporter.h>

#include <SLCamera.h>
#include <SLLightSphere.h>
#include <SLLightRect.h>
#include <SLMesh.h>
#include <SLPolygon.h>
#include <SLBox.h>
#include <SLCone.h>
#include <SLCylinder.h>
#include <SLDisk.h>
#include <SLSphere.h>
#include <SLRectangle.h>
#include <SLText.h>
#include <SLGrid.h>
#include <SLLens.h>

SLNode* SphereGroup(SLint, SLfloat, SLfloat, SLfloat, SLfloat, SLint, SLMaterial*, SLMaterial*);
//-----------------------------------------------------------------------------
//! Creates a recursive sphere group used for the ray tracing scenes
SLNode* SphereGroup(SLint depth,                    // depth of recursion 
                    SLfloat x, SLfloat y, SLfloat z,// position of group
                    SLfloat scale,                  // scale factor
                    SLint  resolution,              // resolution of spheres
                    SLMaterial* matGlass,           // material for center sphere
                    SLMaterial* matRed)             // material for orbiting spheres
{  
    SLstring name = matGlass->kt() > 0 ? "GlassSphere" : "RedSphere";
    if (depth==0)
    {   SLNode* s = new SLNode(new SLSphere(0.5f*scale,resolution,resolution, name, matRed)); 
        s->translate(x,y,z, TS_Object);
        return s;
    } else
    {   depth--;
        SLNode* sGroup = new SLNode("SphereGroup");
        sGroup->translate(x,y,z, TS_Object);
        SLint newRes = max(resolution-8,8);
        sGroup->addChild(new SLNode(new SLSphere(0.5f*scale,resolution,resolution, name, matGlass)));
        sGroup->addChild(SphereGroup(depth, 0.643951f*scale, 0,               0.172546f*scale, scale/3, newRes, matRed, matRed));
        sGroup->addChild(SphereGroup(depth, 0.172546f*scale, 0,               0.643951f*scale, scale/3, newRes, matRed, matRed));
        sGroup->addChild(SphereGroup(depth,-0.471405f*scale, 0,               0.471405f*scale, scale/3, newRes, matRed, matRed));
        sGroup->addChild(SphereGroup(depth,-0.643951f*scale, 0,              -0.172546f*scale, scale/3, newRes, matRed, matRed));
        sGroup->addChild(SphereGroup(depth,-0.172546f*scale, 0,              -0.643951f*scale, scale/3, newRes, matRed, matRed));
        sGroup->addChild(SphereGroup(depth, 0.471405f*scale, 0,              -0.471405f*scale, scale/3, newRes, matRed, matRed));
        sGroup->addChild(SphereGroup(depth, 0.272166f*scale, 0.544331f*scale, 0.272166f*scale, scale/3, newRes, matRed, matRed));
        sGroup->addChild(SphereGroup(depth,-0.371785f*scale, 0.544331f*scale, 0.099619f*scale, scale/3, newRes, matRed, matRed));
        sGroup->addChild(SphereGroup(depth, 0.099619f*scale, 0.544331f*scale,-0.371785f*scale, scale/3, newRes, matRed, matRed));
        return sGroup;
    }
}
//-----------------------------------------------------------------------------
//! Build a hierarchical figurine with arms and legs
SLNode* BuildFigureGroup(SLMaterial* mat, SLbool withAnimation = false);
SLNode* BuildFigureGroup(SLMaterial* mat, SLbool withAnimation)
{
    SLNode* cyl;
   
    // Feet
    SLNode* feet = new SLNode("feet group (T13,R6)");
    feet->addMesh(new SLSphere(0.2f, 16, 16, "ankle", mat));
    SLNode* feetbox = new SLNode(new SLBox(-0.2f,-0.1f, 0.0f, 0.2f, 0.1f, 0.8f, "foot", mat), "feet (T14)");
    feetbox->translate(0.0f,-0.25f,-0.15f, TS_Object);
    feet->addChild(feetbox);
    feet->translate(0.0f,0.0f,1.6f, TS_Object);
    feet->rotate(-90.0f, 1.0f, 0.0f, 0.0f);
   
    // Assemble low leg
    SLNode* leglow = new SLNode("low leg group (T11, R5)");
    leglow->addMesh(new SLSphere(0.3f, 16, 16, "knee", mat));
    cyl = new SLNode(new SLCylinder(0.2f, 1.4f, 1, 16, false, false, "shin", mat), "shin (T12)");
    cyl->translate(0.0f, 0.0f, 0.2f, TS_Object);            
    leglow->addChild(cyl);
    leglow->addChild(feet);
    leglow->translate(0.0f, 0.0f, 1.27f, TS_Object);
    leglow->rotate(0, 1.0f, 0.0f, 0.0f);
   
    // Assemble leg
    SLNode* leg = new SLNode("leg group ()");
    leg->addMesh(new SLSphere(0.4f, 16, 16, "hip joint", mat));
    cyl = new SLNode(new SLCylinder(0.3f, 1.0f, 1, 16, false, false, "thigh", mat), "thigh (T10)");
    cyl->translate(0.0f, 0.0f, 0.27f, TS_Object);           
    leg->addChild(cyl);
    leg->addChild(leglow);

    // Assemble left & right leg
    SLNode* legLeft = new SLNode("left leg group (T8)");
    legLeft->translate(-0.4f, 0.0f, 2.2f, TS_Object);
    legLeft->addChild(leg);
    SLNode* legRight= new SLNode("right leg group (T9)");
    legRight->translate(0.4f, 0.0f, 2.2f, TS_Object);       
    legRight->addChild(leg->copyRec());

    // Assemble low arm
    SLNode* armlow = new SLNode("low arm group (T6,R4)");
    armlow->addMesh(new SLSphere(0.2f, 16, 16, "ellbow", mat));
    cyl = new SLNode(new SLCylinder(0.15f, 1.0f, 1, 16, true, false, "low arm", mat), "T7");
    cyl->translate(0.0f, 0.0f, 0.14f, TS_Object);           
    armlow->addChild(cyl);
    armlow->translate(0.0f, 0.0f, 1.2f, TS_Object);
    armlow->rotate(45, -1.0f, 0.0f, 0.0f);

    // Assemble arm
    SLNode* arm = new SLNode("arm group ()");
    arm->addMesh(new SLSphere(0.3f, 16, 16, "shoulder", mat));
    cyl = new SLNode(new SLCylinder(0.2f, 1.0f, 1, 16, false, false, "upper arm", mat), "upper arm (T5)");
    cyl->translate(0.0f, 0.0f, 0.2f, TS_Object);            
    arm->addChild(cyl);
    arm->addChild(armlow);

    // Assemble left & right arm
    SLNode* armLeft = new SLNode("left arm group (T3,R2)");
    armLeft->translate(-1.1f, 0.0f, 0.3f, TS_Object);       
    armLeft->rotate(10, -1,0,0);
    armLeft->addChild(arm);
    SLNode* armRight= new SLNode("right arm group (T4,R3)");
    armRight->translate(1.1f, 0.0f, 0.3f, TS_Object);       
    armRight->rotate(-60, -1,0,0);
    armRight->addChild(arm->copyRec());

    // Assemble head & neck
    SLNode* head = new SLNode(new SLSphere(0.5f, 16, 16, "head", mat), "head (T1)");
    head->translate(0.0f, 0.0f,-0.7f, TS_Object);
    SLNode* neck = new SLNode(new SLCylinder(0.25f, 0.3f, 1, 16, false, false, "neck", mat), "neck (T2)");
    neck->translate(0.0f, 0.0f,-0.3f, TS_Object);
      
    // Assemble figure Left
    SLNode* figure = new SLNode("figure group (R1)");
    figure->addChild(new SLNode(new SLBox(-0.8f,-0.4f, 0.0f, 0.8f, 0.4f, 2.0f, "chest", mat), "chest"));
    figure->addChild(head);
    figure->addChild(neck);
    figure->addChild(armLeft);
    figure->addChild(armRight);
    figure->addChild(legLeft);
    figure->addChild(legRight);
    figure->rotate(90, 1,0,0);

    // Add animations for left leg
    if (withAnimation)
    {
        legLeft = figure->findChild<SLNode>("left leg group (T8)");
        legLeft->rotate(30, -1,0,0);
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
//! SLScene::onLoad(int sceneName) builds a scene from source code.
/*! SLScene::onLoad builds a scene from source code.
The parameter sceneName is the scene to choose and corresponds to enumeration 
SLCommand value for the different scenes. The first scene is cmdSceneFigure.
*/
void SLScene::onLoad(SLSceneView* sv, SLCmd sceneName)
{  
    // Initialize all preloaded stuff from SLScene
    cout << "------------------------------------------------------------------" << endl;
    init();

    // Show once the empty loading screen without scene
    // @todo review this, we still pass in the active scene view with sv. Is it necessary?
    for (auto sv : _sceneViews)
        if (sv != nullptr)
            sv->showLoading(true);

    _currentID = sceneName;

    if (sceneName == cmdSceneSmallTest) //......................................
    {
        // Set scene name and info string
        name("Minimal Texture Example");
        info(sv, "Minimal texture mapping example with one light source.");

        // Create textures and materials
        SLGLTexture* texC = new SLGLTexture("earth1024_C.jpg");
        SLMaterial* m1 = new SLMaterial("m1", texC);

        // Create a camera node
        SLCamera* cam1 = new SLCamera();
        cam1->name("camera node");
        cam1->translation(0,0,20);
        cam1->lookAt(0, 0, 0);
        cam1->setInitialState();

        // Create a light source node
        SLLightSphere* light1 = new SLLightSphere(0.3f);
        light1->translation(0,0,5);
        light1->lookAt(0, 0, 0);
        light1->name("light node");

        // Create meshes and nodes
        SLMesh* rectMesh = new SLRectangle(SLVec2f(-5,-5), SLVec2f(5,5), 1,1, "rect mesh", m1);
        SLNode* rectNode = new SLNode(rectMesh, "rect node");

        // Create a scene group and add all nodes
        SLNode* scene = new SLNode("scene node");
        scene->addChild(light1);
        scene->addChild(cam1);
        scene->addChild(rectNode);

        // Set background color and the root scene node
        _background.colors(SLCol4f(0.7f,0.7f,0.7f), SLCol4f(0.2f,0.2f,0.2f));
        _root3D = scene;

        // Set active camera
        sv->camera(cam1);
    }
    else
    if (sceneName == cmdSceneFigure) //.........................................
    {
        name("Hierarchical Figure Scene");
        info(sv, "Hierarchical scenegraph with multiple subgroups.");

        // Create textures and materials
        SLMaterial* m1 = new SLMaterial("m1", SLCol4f::BLACK, SLCol4f::WHITE,128, 0.2f, 0.8f, 1.5f);
        SLMaterial* m2 = new SLMaterial("m2", SLCol4f::WHITE*0.3f, SLCol4f::WHITE,128, 0.5f, 0.0f, 1.0f);

        SLMesh* floorMesh = new SLRectangle(SLVec2f(-5,-5), SLVec2f(5,5), 20, 20, "floor mesh", m2);
        SLNode* floorRect = new SLNode(floorMesh);
        floorRect->rotate(90, -1,0,0);
        floorRect->translate(0,0,-5.5f);

        SLCamera* cam1 = new SLCamera();
        cam1->translation(0, 0, 22);
        cam1->lookAt(0, 0, 0);
        cam1->focalDist(22);
        cam1->setInitialState();

        SLLightSphere* light1 = new SLLightSphere(5, 5, 5, 0.5f);
        light1->ambient (SLCol4f(0.2f,0.2f,0.2f));
        light1->diffuse (SLCol4f(0.9f,0.9f,0.9f));
        light1->specular(SLCol4f(0.9f,0.9f,0.9f));
        light1->attenuation(1,0,0);

        SLNode* figure = BuildFigureGroup(m1);

        SLNode* scene = new SLNode("scene node");
        scene->addChild(light1);
        scene->addChild(cam1);
        scene->addChild(floorRect);
        scene->addChild(figure);
     
        // Set background color, active camera & the root pointer
        _background.colors(SLCol4f(0.1f,0.4f,0.8f));
        sv->camera(cam1);
        _root3D = scene;
    }
    else
    if (sceneName == cmdSceneMeshLoad) //.......................................
    {
        name("Mesh 3D Loader Test");
        info(sv, "3D file import test for: 3DS, DAE & FBX");

        SLMaterial* matBlu = new SLMaterial("Blue",  SLCol4f(0,0,0.2f),       SLCol4f(1,1,1), 100, 0.8f, 0);
        SLMaterial* matRed = new SLMaterial("Red",   SLCol4f(0.2f,0,0),       SLCol4f(1,1,1), 100, 0.8f, 0);
        SLMaterial* matGre = new SLMaterial("Green", SLCol4f(0,0.2f,0),       SLCol4f(1,1,1), 100, 0.8f, 0);
        SLMaterial* matGra = new SLMaterial("Gray",  SLCol4f(0.3f,0.3f,0.3f), SLCol4f(1,1,1), 100, 0,    0);

        SLCamera* cam1 = new SLCamera();
        cam1->name("cam1");
        cam1->clipNear(.1f);
        cam1->clipFar(30);
        cam1->translation(0,0,12);
        cam1->lookAt(0, 0, 0);
        cam1->maxSpeed(20);
        cam1->moveAccel(160);
        cam1->brakeAccel(160);
        cam1->focalDist(12);
        cam1->eyeSeparation(cam1->focalDist()/30.0f);
        cam1->setInitialState();

        SLLightSphere* light1 = new SLLightSphere(2.5f, 2.5f, 2.5f, 0.2f);
        light1->ambient(SLCol4f(0.1f, 0.1f, 0.1f));
        light1->diffuse(SLCol4f(1.0f, 1.0f, 1.0f));
        light1->specular(SLCol4f(1.0f, 1.0f, 1.0f));
        light1->attenuation(1,0,0);
        //light1->samples(8,8); // soft shadows for RT
        SLAnimation* anim = SLAnimation::create("anim_light1_backforth", 2.0f, true, EC_inOutQuad, AL_pingPongLoop);
        anim->createSimpleTranslationNodeTrack(light1, SLVec3f(0.0f, 0.0f, -5.0f));

        SLLightSphere* light2 = new SLLightSphere(-2.5f, -2.5f, 2.5f, 0.2f);
        light2->ambient(SLCol4f(0.1f, 0.1f, 0.1f));
        light2->diffuse(SLCol4f(1.0f, 1.0f, 1.0f));
        light2->specular(SLCol4f(1.0f, 1.0f, 1.0f));
        light2->attenuation(1,0,0);
        //light2->samples(8,8); // soft shadows for RT
        anim = SLAnimation::create("anim_light2_updown", 2.0f, true, EC_inOutQuint, AL_pingPongLoop);
        anim->createSimpleTranslationNodeTrack(light2, SLVec3f(0.0f, 5.0f, 0.0f));

        #if defined(SL_OS_IOS) || defined(SL_OS_ANDROID)
        SLAssimpImporter importer;
        SLNode* mesh3DS = importer.load("jackolan.3ds");
        SLNode* meshFBX = importer.load("duck.fbx");
        SLNode* meshDAE = importer.load("AstroBoy.dae");
      
        #else
        SLAssimpImporter importer;
        SLNode* mesh3DS = importer.load("3DS/Halloween/jackolan.3ds");
        SLNode* meshFBX = importer.load("FBX/Duck/duck.fbx");
        SLNode* meshDAE = importer.load("DAE/AstroBoy/AstroBoy.dae");
        #endif

        // Start animation
        SLAnimPlayback* charAnim = importer.skeleton()->getAnimPlayback("unnamed_anim_0");
        charAnim->playForward();
        charAnim->playbackRate(0.8f);

        // Scale to so that the AstroBoy is about 2 (meters) high.
        if (mesh3DS) {mesh3DS->scale(0.1f);  mesh3DS->translate(-22.0f, 1.9f, 3.5f, TS_Object);}
        if (meshDAE) {meshDAE->translate(0,-3,0, TS_Object); meshDAE->scale(2.7f);}
        if (meshFBX) {meshFBX->scale(0.1f);  meshFBX->scale(0.1f); meshFBX->translate(200, 30, -30, TS_Object); meshFBX->rotate(-90,0,1,0);}
        
        // define rectangles for the surrounding box
        SLfloat b=3; // edge size of rectangles
        SLNode *rb, *rl, *rr, *rf, *rt;
        SLuint res = 20;
        rb = new SLNode(new SLRectangle(SLVec2f(-b,-b), SLVec2f(b,b), res, res, "rectB", matBlu), "rectBNode");                         rb->translate(0,0,-b, TS_Object);
        rl = new SLNode(new SLRectangle(SLVec2f(-b,-b), SLVec2f(b,b), res, res, "rectL", matRed), "rectLNode"); rl->rotate( 90, 0,1,0); rl->translate(0,0,-b, TS_Object);
        rr = new SLNode(new SLRectangle(SLVec2f(-b,-b), SLVec2f(b,b), res, res, "rectR", matGre), "rectRNode"); rr->rotate(-90, 0,1,0); rr->translate(0,0,-b, TS_Object);
        rf = new SLNode(new SLRectangle(SLVec2f(-b,-b), SLVec2f(b,b), res, res, "rectF", matGra), "rectFNode"); rf->rotate(-90, 1,0,0); rf->translate(0,0,-b, TS_Object);
        rt = new SLNode(new SLRectangle(SLVec2f(-b,-b), SLVec2f(b,b), res, res, "rectT", matGra), "rectTNode"); rt->rotate( 90, 1,0,0); rt->translate(0,0,-b, TS_Object);

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

        _background.colors(SLCol4f(0.6f,0.6f,0.6f), SLCol4f(0.3f,0.3f,0.3f));
        sv->camera(cam1);
        _root3D = scene;
    }
    else
    if (sceneName == cmdSceneVRSizeTest) //.....................................
    {
        name("Virtual Reality test scene");
        info(sv, "Test scene for virtual reality size perception.");
        

        SLAssimpImporter importer;
        SLNode* scene = new SLNode;
        scene->scale(1);
        
        // scene floor
        SLMaterial* matFloor = new SLMaterial("floor", new SLGLTexture("tron_floor2.png"
                                            ,SL_ANISOTROPY_MAX
                                            ,GL_LINEAR),
                                            nullptr, nullptr, nullptr,
                                            _programs[TextureOnly]);
        SLNode* floor = new SLNode(
                            new SLRectangle(SLVec2f(-1000, -1000), SLVec2f(1000,1000),
                                            SLVec2f(-1000, -1000), SLVec2f(1000,1000),
                                            1, 1, "rectF", matFloor), "rectFNode"); 
        floor->rotate(-90, 1,0,0);
        scene->addChild(floor);

        // scene sky box
        // TODO...

        // table
        SLNode* table = importer.load("DAE/Table/table.dae");
        table->translate(0, 0, -1);
        scene->addChild(table);

        // create crates of various sizes
        SLNode* crate = importer.load("DAE/Crate/crate.dae");
        SLMesh* crateMesh = importer.meshes()[3];
        
        
        crate->rotate(20, 0, 1, 0);
        crate->translate(2, 0, -1, TS_World);
        scene->addChild(crate);
        
        crate = new SLNode;
        crate->addMesh(crateMesh);
        crate->rotate(20, 0, 1, 0);
        crate->translate(3.1f, 0, -1, TS_World);
        scene->addChild(crate);
        
        crate = new SLNode(crateMesh);
        crate->rotate(-10, 0, 1, 0);
        crate->translate(2.5f, 1, -1, TS_World);
        scene->addChild(crate);

        
        crate = new SLNode(crateMesh);
        crate->rotate(60, 0, 1, 0);
        crate->translate(-4, 0, 1, TS_World);
        crate->scale(2);
        scene->addChild(crate);
        
        crate = new SLNode(crateMesh);
        crate->rotate(30, 0, 1, 0);
        crate->translate(-5, 0, -8, TS_World);
        crate->scale(4);
        scene->addChild(crate);

        SLCamera* cam1 = new SLCamera();
        cam1->translation(0, 1.67f, 0);    // eye height for 180cm high male
        cam1->lookAt(0, 1.67f, -1.0f);
        cam1->focalDist(22);
        cam1->setInitialState();
        cam1->camAnim(walkingYUp);
        scene->addChild(cam1);

        // big astroboy
        // Start animation
        SLNode* astroboyBig = importer.load("DAE/AstroBoy/AstroBoy.dae");
        SLAnimPlayback* charAnim = importer.skeleton()->getAnimPlayback("unnamed_anim_0");
        charAnim->playForward();
        charAnim->playbackRate(0.8f);

        astroboyBig->translate(-1.5f, 0.0f, -1.0f);

        scene->addChild(astroboyBig);

        // small astroboy on table
        SLNode* astroboySmall = importer.load("DAE/AstroBoy/AstroBoy.dae");
        charAnim = importer.skeleton()->getAnimPlayback("unnamed_anim_0");
        charAnim->playForward();
        charAnim->playbackRate(2.0f);
        
        astroboySmall->translate(0.0f, 1.1f, -1.0f);
        astroboySmall->scale(0.1f);
        scene->addChild(astroboySmall);

        sv->camera(cam1);
        
        SLLightSphere* light1 = new SLLightSphere(5, 20, 5, 0.5f, 1.0f, 1.0f, 2.0f);
        light1->ambient(SLCol4f(0.1f, 0.1f, 0.1f));
        light1->diffuse(SLCol4f(1.0f, 0.7f, 0.3f));
        light1->specular(SLCol4f(0.5f, 0.3f, 0.1f));
        light1->attenuation(1,0,0);
                
        SLLightSphere* light2 = new SLLightSphere(-10.0f, -15.0, 10.0f, 0.2f, 1.0f, 1.0f, 0.0f);
        light2->ambient(SLCol4f(0.0f, 0.0f, 0.0f));
        light2->diffuse(SLCol4f(0.0f, 4.0f, 10.0f));
        light2->specular(SLCol4f(0.0f, 0.0f, 0.0f));
        light2->attenuation(1,0.5f,0);
        
        SLLightSphere* light3 = new SLLightSphere(-10.0f, -15.0, -10.0f, 0.2f, 1.0f, 1.0f, 0.0f);
        light3->ambient(SLCol4f(0.0f, 0.0f, 0.0f));
        light3->diffuse(SLCol4f(0.0f, 4.0f, 10.0f));
        light3->specular(SLCol4f(0.0f, 0.0f, 0.0f));
        light3->attenuation(1,0.5f,0);
        
        SLLightSphere* light4 = new SLLightSphere(10.0f, -15.0, -10.0f, 0.2f, 1.0f, 1.0f, 0.0f);
        light4->ambient(SLCol4f(0.0f, 0.0f, 0.0f));
        light4->diffuse(SLCol4f(0.0f, 4.0f, 10.0f));
        light4->specular(SLCol4f(0.0f, 0.0f, 0.0f));
        light4->attenuation(1,0.5f,0);
        
        SLLightSphere* light5 = new SLLightSphere(10.0f, -15.0, 10.0f, 0.2f, 1.0f, 1.0f, 0.0f);
        light5->ambient(SLCol4f(0.0f, 0.0f, 0.0f));
        light5->diffuse(SLCol4f(0.0f, 4.0f, 10.0f));
        light5->specular(SLCol4f(0.0f, 0.0f, 0.0f));
        light5->attenuation(1,0.5f,0);

        SLAnimation* anim = SLAnimation::create("anim_light2_updown", 10.0f, true, EC_inOutSine, AL_pingPongLoop);
        anim->createSimpleTranslationNodeTrack(light2, SLVec3f(0.0f, 1.0f, 0.0f));
        anim->createSimpleTranslationNodeTrack(light3, SLVec3f(0.0f, 2.0f, 0.0f));
        anim->createSimpleTranslationNodeTrack(light4, SLVec3f(0.0f, 1.0f, 0.0f));
        anim->createSimpleTranslationNodeTrack(light5, SLVec3f(0.0f, 2.0f, 0.0f));

        SLMaterial* whiteMat = new SLMaterial("mat", SLCol4f::WHITE, SLCol4f::WHITE, 1.0f, 1.0, 0.0f, 0.0f);
        whiteMat->emission(SLCol4f::WHITE);
        SLRectangle* plane0 = new SLRectangle(SLVec2f(-0.01f, 0.0f), SLVec2f(0.01f, 1.0f), 1, 1, "sizeIndicator0", whiteMat);
        SLRectangle* plane1 = new SLRectangle(SLVec2f(0.005f, 0.0f), SLVec2f(-0.005f, 1.0f), 1, 1, "sizeIndicator1", whiteMat);

        struct indicatorData {
            indicatorData(SLfloat px, SLfloat py, SLfloat pz, SLfloat r, SLfloat s, const SLstring& t)
                : pos(px, py, pz), yRot(r), yScale(s), text(t)
            { }

            SLVec3f pos;
            SLfloat yRot;
            SLfloat yScale;
            SLstring text;
        };
        indicatorData indicators[] = {
            // pos                             y rot    y scale text
            indicatorData(3.0f, 0.0f, -0.2f,    -20.0f,    1.0f,   "1m"),
            indicatorData(0.7f, 0.0f, -0.8f,    0.0f,    1.1f,   "1.10m"),
            indicatorData(0.05f, 1.1f, -1.0f,    0.0f,    0.18f,   "18cm"),
            indicatorData(-1.2f, 0.0f, -1.0f,    0.0f,    1.8f,   "1.80m"),
            indicatorData(-2.8f, 0.0f, 0.2f,    60.0f,    2.0f,   "2m"),
            indicatorData(-2.0f, 0.0f, -7.0f,   20.0f,   4.0f,   "4m")
        };        

        for (SLint i = 0; i < 6; i++)
        {
            SLNode* sizeIndicator = new SLNode;
            sizeIndicator->addMesh(plane0);
            sizeIndicator->addMesh(plane1);
            //sizeIndicator->scale();
            SLVec3f pos = indicators[i].pos;
            sizeIndicator->translate(pos, TS_World);
            sizeIndicator->scale(1, indicators[i].yScale, 1);
            sizeIndicator->rotate(indicators[i].yRot, 0, 1, 0, TS_World);
        
            SLText* sizeText1M = new SLText(indicators[i].text, SLTexFont::font22);
            sizeText1M->translate(pos.x + 0.05f, pos.y + 0.5f * indicators[i].yScale, pos.z);
            sizeText1M->rotate(indicators[i].yRot, 0, 1, 0, TS_World);
            sizeText1M->scale(0.005f);


            scene->addChild(sizeText1M);
            scene->addChild(sizeIndicator);
        }
        
        _background.colors(SLCol4f(0.0f,0.0f,0.0f));
        scene->addChild(light1);
        scene->addChild(light2);
        scene->addChild(light3);
        scene->addChild(light4);
        scene->addChild(light5);

        _root3D = scene;
    }
    else
    if (sceneName == cmdSceneRevolver) //.......................................
    {
        name("Revolving Mesh Test w. glass shader");
        info(sv, "Examples of revolving mesh objects constructed by rotating a 2D curve. The glass shader reflects and refracts the environment map. Try ray tracing.");

        // Test map material
        SLGLTexture* tex1 = new SLGLTexture("Testmap_0512_C.png");
        SLMaterial* mat1 = new SLMaterial("mat1", tex1);

        // floor material
        SLGLTexture* tex2 = new SLGLTexture("wood0_0512_C.jpg");
        SLMaterial* mat2 = new SLMaterial("mat2", tex2);
        mat2->specular(SLCol4f::BLACK);

        // Back wall material
        SLGLTexture* tex3 = new SLGLTexture("bricks1_0256_C.jpg");
        SLMaterial* mat3 = new SLMaterial("mat3", tex3);
        mat3->specular(SLCol4f::BLACK);

        // Left wall material
        SLGLTexture* tex4 = new SLGLTexture("wood2_0512_C.jpg");
        SLMaterial* mat4 = new SLMaterial("mat4", tex4);
        mat4->specular(SLCol4f::BLACK);

        // Glass material
        SLGLTexture* tex5 = new SLGLTexture("wood2_0256_C.jpg", "wood2_0256_C.jpg"
                                            ,"gray_0256_C.jpg", "wood0_0256_C.jpg"
                                            ,"gray_0256_C.jpg", "bricks1_0256_C.jpg");
        SLMaterial* mat5 = new SLMaterial("glass", SLCol4f::BLACK, SLCol4f::WHITE, 255, 0.1f, 0.9f, 1.5f);
        mat5->textures().push_back(tex5);
        SLGLProgram* sp1 = new SLGLGenericProgram("RefractReflect.vert", "RefractReflect.frag");
        mat5->program(sp1);

        // Wine material
        SLMaterial* mat6 = new SLMaterial("wine", SLCol4f(0.4f,0.0f,0.2f), SLCol4f::BLACK, 255, 0.2f, 0.7f, 1.3f);
        mat6->textures().push_back(tex5);
        mat6->program(sp1);

        // camera
        SLCamera* cam1 = new SLCamera();
        cam1->name("cam1");
        cam1->translation(0,1,17);
        cam1->lookAt(0,1,0);
        cam1->focalDist(17);
        cam1->setInitialState();

        // light
        SLLightSphere* light1 = new SLLightSphere(0, 4, 0, 0.3f);
        light1->diffuse(SLCol4f(1, 1, 1));
        light1->ambient(SLCol4f(0.2f, 0.2f, 0.2f));
        light1->specular(SLCol4f(1, 1, 1));
        light1->attenuation(1,0,0);
        SLAnimation* anim = SLAnimation::create("light1_anim", 4.0f);
        anim->createEllipticNodeTrack(light1, 6.0f, ZAxis, 6.0f, XAxis);

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
        SLNode* glass = new SLNode(new SLRevolver(revG, SLVec3f(0,1,0), 36, true, false, "GlassRev", mat5));
        glass->translate(0.0f,-3.5f, 0.0f, TS_Object);

        // wine 2D polyline definition for revolution with two sided material
        SLVVec3f revW;
        revW.push_back(SLVec3f(0.00f, 3.82f));
        revW.push_back(SLVec3f(0.20f, 3.80f));
        revW.push_back(SLVec3f(0.80f, 4.00f));
        revW.push_back(SLVec3f(1.30f, 4.30f));
        revW.push_back(SLVec3f(1.70f, 4.80f));
        revW.push_back(SLVec3f(1.95f, 5.40f));
        revW.push_back(SLVec3f(2.05f, 6.00f));
        SLMesh* wineMesh = new SLRevolver(revW, SLVec3f(0,1,0), 36, true, false, "WineRev", mat6);
        wineMesh->matOut = mat5;
        SLNode* wine = new SLNode(wineMesh);
        wine->translate(0.0f,-3.5f, 0.0f, TS_Object);

        // wine fluid top
        SLNode* wineTop = new SLNode(new SLDisk(2.05f, -SLVec3f::AXISY, 36, false, "WineRevTop", mat6));
        wineTop->translate(0.0f, 2.5f, 0.0f, TS_Object);

        // Other revolver objects
        SLNode* sphere = new SLNode(new SLSphere(1,16,16, "sphere", mat1));
        sphere->translate(3,0,0, TS_Object);
        SLNode* cylinder = new SLNode(new SLCylinder(0.1f, 7, 3, 16, true, true, "cylinder", mat1));
        cylinder->translate(0,0.5f,0);
        cylinder->rotate(90,-1,0,0);
        cylinder->rotate(30, 0,1,0);
        SLNode* cone = new SLNode(new SLCone(1, 3, 3, 16, true, "cone", mat1));
        cone->translate(-3,-1,0, TS_Object);
        cone->rotate(90, -1,0,0);

        // Cube dimensions
        SLfloat pL = -9.0f, pR =  9.0f; // left/right
        SLfloat pB = -3.5f, pT = 14.5f; // bottom/top
        SLfloat pN =  9.0f, pF = -9.0f; // near/far

        //// bottom rectangle
        SLNode* b = new SLNode(new SLRectangle(SLVec2f(pL,-pN), SLVec2f(pR,-pF), 10, 10, "PolygonFloor", mat2));
        b->rotate(90, -1,0,0); b->translate(0,0,pB, TS_Object);

        // top rectangle
        SLNode* t = new SLNode(new SLRectangle(SLVec2f(pL,pF), SLVec2f(pR,pN), 10, 10, "top", mat2));
        t->rotate(90, 1,0,0); t->translate(0,0,-pT, TS_Object);

        // far rectangle
        SLNode* f = new SLNode(new SLRectangle(SLVec2f(pL,pB), SLVec2f(pR,pT), 10, 10, "far", mat3));
        f->translate(0,0,pF, TS_Object);

        // left rectangle
        SLNode* l = new SLNode(new SLRectangle(SLVec2f(-pN,pB), SLVec2f(-pF,pT), 10, 10, "left", mat4));
        l->rotate(90, 0,1,0); l->translate(0,0,pL, TS_Object);

        // right rectangle
        SLNode* r = new SLNode(new SLRectangle(SLVec2f(pF,pB), SLVec2f(pN,pT), 10, 10, "right", mat4));
        r->rotate(90, 0,-1,0); r->translate(0,0,-pR, TS_Object);

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

        _background.colors(SLCol4f(0.7f, 0.7f, 0.7f), SLCol4f(0.2f, 0.2f, 0.2f));
        sv->camera(cam1);
        _root3D = scene;
    }
    else
    if (sceneName == cmdSceneLargeModel) //.....................................
    {
        name("Large Model Test");
        info(sv, "Large Model with 7.2 mio. triangles.");
        /*
        SLCamera* cam1 = new SLCamera();
        cam1->name("cam1");
        cam1->translation(10,0,220);
        cam1->lookAt(10,0,0);
        cam1->clipNear(0.1f);
        cam1->clipFar(500.0f);
        cam1->setInitialState();

        SLLightSphere* light1 = new SLLightSphere(120,120,120, 1);
        light1->ambient(SLCol4f(1,1,1));
        light1->diffuse(SLCol4f(1,1,1));
        light1->specular(SLCol4f(1,1,1));
        light1->attenuation(1,0,0);

        SLAssimpImporter importer;
        SLNode* largeModel = importer.load("PLY/xyzrgb_dragon.ply");

        SLNode* scene = new SLNode("Scene");
        scene->addChild(light1);
        if (largeModel) scene->addChild(largeModel);
        scene->addChild(cam1);

        _backColor.set(0.5f,0.5f,0.5f);
        sv->camera(cam1);
        _root3D = scene;
        */
        
        SLCamera* cam1 = new SLCamera();
        cam1->name("cam1");
        cam1->translation(0,0,600000);
        cam1->lookAt(0,0,0);
        cam1->clipNear(20);
        cam1->clipFar(1000000);
        cam1->setInitialState();

        SLLightSphere* light1 = new SLLightSphere(600000,600000,600000, 1);
        light1->ambient(SLCol4f(1,1,1));
        light1->diffuse(SLCol4f(1,1,1));
        light1->specular(SLCol4f(1,1,1));
        light1->attenuation(1,0,0);

        SLAssimpImporter importer;
        SLNode* largeModel = importer.load("PLY/switzerland.ply"
                                            //,SLProcess_JoinIdenticalVertices
                                            //|SLProcess_RemoveRedundantMaterials
                                            //|SLProcess_SortByPType
                                            //|SLProcess_FindDegenerates
                                            //|SLProcess_FindInvalidData
                                            //|SLProcess_SplitLargeMeshes
                                           );
        largeModel->scaleToCenter(100000.0f);

        SLNode* scene = new SLNode("Scene");
        scene->addChild(light1);
        if (largeModel) scene->addChild(largeModel);
        scene->addChild(cam1);

        _background.colors(SLCol4f(0.5f,0.5f,0.5f));
        sv->camera(cam1);
        _root3D = scene;
        
    }
    else
    if (sceneName == cmdSceneChristoffel) //...................................
    {
        name("Christoffel Tower");
        info(sv, "Augmented Reality Christoffel Tower");
        
        SLCamera* cam1 = new SLCamera();
        cam1->name("cam1");
        cam1->translation(0,2,60);
        cam1->lookAt(15,15,0);
        cam1->clipNear(0.1f);
        cam1->clipFar(500.0f);
        cam1->setInitialState();
        cam1->camAnim(walkingYUp);

        SLLightSphere* light1 = new SLLightSphere(120,120,120, 1);
        light1->ambient(SLCol4f(1,1,1));
        light1->diffuse(SLCol4f(1,1,1));
        light1->specular(SLCol4f(1,1,1));
        light1->attenuation(1,0,0);

        SLAssimpImporter importer;
        #if defined(SL_OS_IOS) || defined(SL_OS_ANDROID)
        SLNode* tower = importer.load("christoffelturm.obj");
        #else
        SLNode* tower = importer.load("OBJ/Christoffelturm/christoffelturm.obj");
        #endif
        tower->rotate(90, -1,0,0);

        SLNode* scene = new SLNode("Scene");
        scene->addChild(light1);
        if (tower) scene->addChild(tower);
        scene->addChild(cam1);

        _background.texture(&_videoTexture, true);
        _usesVideoImage = true;

        sv->waitEvents(false); // for constant video feed
        //sv->usesRotation(true);
        sv->camera(cam1);
        _root3D = scene;
    }
    else
    if (sceneName == cmdSceneTextureBlend) //..................................
    {
        name("Blending: Texture Transparency with sorting");
        info(sv, "Texture map blending with depth sorting. Trees in view frustum are rendered back to front.");

        SLGLTexture* t1 = new SLGLTexture("tree1_1024_C.png",
                                          GL_LINEAR_MIPMAP_LINEAR, GL_LINEAR,
                                          ColorMap,
                                          GL_CLAMP_TO_EDGE, GL_CLAMP_TO_EDGE);
        SLGLTexture* t2 = new SLGLTexture("grass0512_C.jpg",
                                          GL_LINEAR_MIPMAP_LINEAR, GL_LINEAR);

        SLMaterial* m1 = new SLMaterial("m1", SLCol4f(1,1,1), SLCol4f(0,0,0), 100);
        SLMaterial* m2 = new SLMaterial("m2", SLCol4f(1,1,1), SLCol4f(0,0,0), 100);
        m1->program(_programs[TextureOnly]);
        m1->textures().push_back(t1);
        m2->textures().push_back(t2);

        SLCamera* cam1 = new SLCamera();
        cam1->name("cam1");
        cam1->translation(0,3,25);
        cam1->lookAt(0,0,10);
        cam1->focalDist(25);
        cam1->setInitialState();

        SLLightSphere* light = new SLLightSphere(0.1f);
        light->translation(5,5,5);
        light->lookAt(0, 0, 0);
        light->attenuation(1,0,0);

        // Build arrays for polygon vertices and texcords for tree
        SLVVec3f pNW, pSE;
        SLVVec2f tNW, tSE;
        pNW.push_back(SLVec3f( 0, 0,0)); tNW.push_back(SLVec2f(0.5f,0.0f));
        pNW.push_back(SLVec3f( 1, 0,0)); tNW.push_back(SLVec2f(1.0f,0.0f));
        pNW.push_back(SLVec3f( 1, 2,0)); tNW.push_back(SLVec2f(1.0f,1.0f));
        pNW.push_back(SLVec3f( 0, 2,0)); tNW.push_back(SLVec2f(0.5f,1.0f));
        pSE.push_back(SLVec3f(-1, 0,0)); tSE.push_back(SLVec2f(0.0f,0.0f));
        pSE.push_back(SLVec3f( 0, 0,0)); tSE.push_back(SLVec2f(0.5f,0.0f));
        pSE.push_back(SLVec3f( 0, 2,0)); tSE.push_back(SLVec2f(0.5f,1.0f));
        pSE.push_back(SLVec3f(-1, 2,0)); tSE.push_back(SLVec2f(0.0f,1.0f));

        // Build tree out of 4 polygons
        SLNode* p1 = new SLNode(new SLPolygon(pNW, tNW, "Tree+X", m1));
        SLNode* p2 = new SLNode(new SLPolygon(pNW, tNW, "Tree-Z", m1));  p2->rotate(90, 0,1,0);
        SLNode* p3 = new SLNode(new SLPolygon(pSE, tSE, "Tree-X", m1));
        SLNode* p4 = new SLNode(new SLPolygon(pSE, tSE, "Tree+Z", m1));  p4->rotate(90, 0,1,0);

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

        // Build arrays for polygon vertices and texcords for ground
        SLVVec3f pG;
        SLVVec2f tG;
        pG.push_back(SLVec3f(-22, 0, 22)); tG.push_back(SLVec2f( 0, 0));
        pG.push_back(SLVec3f( 22, 0, 22)); tG.push_back(SLVec2f(30, 0));
        pG.push_back(SLVec3f( 22, 0,-22)); tG.push_back(SLVec2f(30,30));
        pG.push_back(SLVec3f(-22, 0,-22)); tG.push_back(SLVec2f( 0,30));

        SLNode* scene = new SLNode("grScene");
        scene->addChild(light);
        scene->addChild(tree);
        scene->addChild(new SLNode(new SLPolygon(pG, tG, "Ground", m2)));

        //create 21*21*21-1 references around the center tree
        SLint size = 10;
        for (SLint iZ=-size; iZ<=size; ++iZ)
        {   for (SLint iX=-size; iX<=size; ++iX)
            {  if (iX!=0 || iZ!=0)
                {   SLNode* t = tree->copyRec();
                    t->translate(float(iX)*2+SL_random(0.7f,1.4f),
                                0,
                                float(iZ)*2+SL_random(0.7f,1.4f), TS_Object);
                    t->rotate(SL_random(0, 90), 0,1,0);
                    t->scale(SL_random(0.5f,1.0f));
                    scene->addChild(t);
                }
            }
        }

        scene->addChild(cam1);

        _background.colors(SLCol4f(0.6f,0.6f,1));
        sv->camera(cam1);
        _root3D = scene;
    }
    else
    if (sceneName == cmdSceneTextureFilter) //.................................
    {
        name("Texturing: Filter Compare and 3D texture");
        info(sv, "Texture filters: Bottom: nearest, left: linear, top: linear mipmap, right: anisotropic");
        
        // Create 4 textures with different filter modes
        SLGLTexture* texB = new SLGLTexture("brick0512_C.png"
                                            ,GL_NEAREST
                                            ,GL_NEAREST);
        SLGLTexture* texL = new SLGLTexture("brick0512_C.png"
                                            ,GL_LINEAR
                                            ,GL_LINEAR);
        SLGLTexture* texT = new SLGLTexture("brick0512_C.png"
                                            ,GL_LINEAR_MIPMAP_LINEAR
                                            ,GL_LINEAR);
        SLGLTexture* texR = new SLGLTexture("brick0512_C.png"
                                            ,SL_ANISOTROPY_MAX
                                            ,GL_LINEAR);

        // define materials with textureOnly shader, no light needed
        SLMaterial* matB = new SLMaterial("matB", texB,0,0,0, _programs[TextureOnly]);
        SLMaterial* matL = new SLMaterial("matL", texL,0,0,0, _programs[TextureOnly]);
        SLMaterial* matT = new SLMaterial("matT", texT,0,0,0, _programs[TextureOnly]);
        SLMaterial* matR = new SLMaterial("matR", texR,0,0,0, _programs[TextureOnly]);

        // build polygons for bottom, left, top & right side
        SLVVec3f VB;
        VB.push_back(SLVec3f(-0.5f,-0.5f, 1.0f));
        VB.push_back(SLVec3f( 0.5f,-0.5f, 1.0f));
        VB.push_back(SLVec3f( 0.5f,-0.5f,-2.0f));
        VB.push_back(SLVec3f(-0.5f,-0.5f,-2.0f));
        SLVVec2f T;
        T.push_back(SLVec2f( 0.0f, 2.0f));
        T.push_back(SLVec2f( 0.0f, 0.0f));
        T.push_back(SLVec2f( 6.0f, 0.0f));
        T.push_back(SLVec2f( 6.0f, 2.0f));
        SLNode* polyB = new SLNode(new SLPolygon(VB, T, "PolygonB", matB));

        SLVVec3f VL;
        VL.push_back(SLVec3f(-0.5f, 0.5f, 1.0f));
        VL.push_back(SLVec3f(-0.5f,-0.5f, 1.0f));
        VL.push_back(SLVec3f(-0.5f,-0.5f,-2.0f));
        VL.push_back(SLVec3f(-0.5f, 0.5f,-2.0f));
        SLNode* polyL = new SLNode(new SLPolygon(VL, T, "PolygonL", matL));

        SLVVec3f VT;
        VT.push_back(SLVec3f( 0.5f, 0.5f, 1.0f));
        VT.push_back(SLVec3f(-0.5f, 0.5f, 1.0f));
        VT.push_back(SLVec3f(-0.5f, 0.5f,-2.0f));
        VT.push_back(SLVec3f( 0.5f, 0.5f,-2.0f));
        SLNode* polyT = new SLNode(new SLPolygon(VT, T, "PolygonT", matT));

        SLVVec3f VR;
        VR.push_back(SLVec3f( 0.5f,-0.5f, 1.0f));
        VR.push_back(SLVec3f( 0.5f, 0.5f, 1.0f));
        VR.push_back(SLVec3f( 0.5f, 0.5f,-2.0f));
        VR.push_back(SLVec3f( 0.5f,-0.5f,-2.0f));
        SLNode* polyR = new SLNode(new SLPolygon(VR, T, "PolygonR", matR));

        
        #ifdef SL_GLES2
        // Create 3D textured sphere mesh and node
        SLNode* sphere = new SLNode(new SLSphere(0.2f, 16, 16, "Sphere", matL));
        #else
        // 3D Texture Mapping on a pyramid
        SLVstring tex3DFiles;
        for (SLint i=0; i<256; ++i) tex3DFiles.push_back("Wave_radial10_256C.jpg");
        SLGLTexture* tex3D = new SLGLTexture(tex3DFiles);
        SLGLProgram* spr3D = new SLGLGenericProgram("TextureOnly3D.vert", "TextureOnly3D.frag");
        SLMaterial*  mat3D = new SLMaterial("mat3D", tex3D ,0,0,0, spr3D);

        // Create 3D textured pyramid mesh and node
        SLMesh* pyramid = new SLMesh("Pyramid");
        pyramid->mat = mat3D;
        pyramid->P = new SLVec3f[5]{{-1,-1,1},{1,-1,1},{1,-1,-1},{-1,-1,-1},{0,2,0}};
        pyramid->numV = 5;
        pyramid->I16 = new SLushort[18]{0,3,1, 1,3,2, 4,0,1, 4,1,2, 4,2,3, 4,3,0};
        pyramid->numI = 18;
        SLNode* pyramidNode = new SLNode(pyramid, "Pyramid");
        pyramidNode->scale(0.2f);
        pyramidNode->translate(0, 0, -3);

        // Create 3D textured sphere mesh and node
        SLNode* sphere = new SLNode(new SLSphere(0.2f, 16, 16, "Sphere", mat3D));
        #endif

        SLCamera* cam1 = new SLCamera;
        cam1->translation(0,0,2.2f);
        cam1->lookAt(0, 0, 0);
        cam1->focalDist(2.2f);
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
        
        _background.colors(SLCol4f(0.2f,0.2f,0.2f));
        sv->camera(cam1);
        _root3D = scene;
    }
    else
    if (sceneName == cmdSceneTextureVideo) //..................................
    {
        // Set scene name and info string
        name("Live Video Texture Example");
        info(sv, "Minimal texture mapping example with live video source.");

        // Back wall material with live video texture
        SLMaterial* m1 = new SLMaterial("mat3", &_videoTexture);
        _usesVideoImage = true;

        // Create a camera node
        SLCamera* cam1 = new SLCamera();
        cam1->name("camera node");
        cam1->translation(0,0,20);
        cam1->lookAt(0, 0, 0);
        cam1->setInitialState();

        // Create a light source node
        SLLightSphere* light1 = new SLLightSphere(0.3f);
        light1->translation(0,0,5);
        light1->lookAt(0, 0, 0);
        light1->name("light node");

        // Create meshes and nodes
        SLMesh* rectMesh = new SLRectangle(SLVec2f(-5,-5), SLVec2f(5,5), 1,1, "rect mesh", m1);
        SLNode* rectNode = new SLNode(rectMesh, "rect node");

        // Create a scene group and add all nodes
        SLNode* scene = new SLNode("scene node");
        scene->addChild(light1);
        scene->addChild(cam1);
        scene->addChild(rectNode);

        // Set background color and the root scene node
        _background.colors(SLCol4f(0.7f,0.7f,0.7f), SLCol4f(0.2f,0.2f,0.2f));
        _root3D = scene;

        // Set active camera
        sv->camera(cam1);
    }
    if (sceneName == cmdSceneFrustumCull1) //...................................
    {  
        name("Frustum Culling Test 1");
        info(sv, "View frustum culling: Only objects in view frustum are rendered. You can turn view culling off in the render flags.");

        // create texture
        SLGLTexture* tex = new SLGLTexture("earth1024_C.jpg");
        SLMaterial* mat1 = new SLMaterial("mat1", tex);

        SLCamera* cam1 = new SLCamera();
        cam1->name("cam1");
        cam1->clipNear(0.1f);
        cam1->clipFar(100);
        cam1->translation(0,0,5);
        cam1->lookAt(0, 0, 0);
        cam1->focalDist(5);
        cam1->setInitialState();

        SLLightSphere* light1 = new SLLightSphere(10, 10, 10, 0.3f);
        light1->ambient(SLCol4f(0.2f, 0.2f, 0.2f));
        light1->diffuse(SLCol4f(0.8f, 0.8f, 0.8f));
        light1->specular(SLCol4f(1, 1, 1));
        light1->attenuation(1,0,0);

        SLNode* scene = new SLNode;
        scene->addChild(cam1);
        scene->addChild(light1);

        // add one single sphere in the center
        SLint resolution = 16;
        SLNode* sphere = new SLNode(new SLSphere(0.15f, resolution, resolution, "mySphere", mat1));
        scene->addChild(sphere);

        // create spheres around the center sphere
        SLint size = 10;
        for (SLint iZ=-size; iZ<=size; ++iZ)
        {   for (SLint iY=-size; iY<=size; ++iY)
            {   for (SLint iX=-size; iX<=size; ++iX)
                {   if (iX!=0 || iY!=0 || iZ !=0)
                    {
                        SLNode* s = sphere->copyRec();
                        s->translate(float(iX), float(iY), float(iZ), TS_Object);
                        scene->addChild(s);
                    }
                }
            }
        }

        SLint num = size + size + 1;
        SL_LOG("Triangles in scene: %d\n", resolution*resolution*2*num*num*num);

        _background.colors(SLCol4f(0.1f,0.1f,0.1f));
        sv->camera(cam1);
        _root3D = scene;
    }
    else
    if (sceneName == cmdSceneFrustumCull2) //...................................
    {
        name("Frustum Culling Test 2");
        info(sv, "View frustum culling: Only objects in view frustum are rendered. You can turn view culling off in the render flags.");
        // Create textures and materials
        SLGLTexture* t1 = new SLGLTexture("grass0512_C.jpg", GL_LINEAR_MIPMAP_LINEAR, GL_LINEAR);
        SLMaterial* m1 = new SLMaterial("m1", t1); m1->specular(SLCol4f::BLACK);
        SLMaterial* m2 = new SLMaterial("m2", SLCol4f::WHITE*0.5, SLCol4f::WHITE,128, 0.5f, 0.0f, 1.0f);

        // Define a light
        SLLightSphere* light1 = new SLLightSphere(0, 20, 0, 0.5f);
        light1->ambient (SLCol4f(0.2f,0.2f,0.2f));
        light1->diffuse (SLCol4f(0.9f,0.9f,0.9f));
        light1->specular(SLCol4f(0.9f,0.9f,0.9f));
        light1->attenuation(1,0,0);

        // Define camera
        SLCamera* cam1 = new SLCamera;
        cam1->translation(0,100,180);
        cam1->lookAt(0, 0, 0);
        cam1->setInitialState();

        // Floor rectangle
        SLNode* rect = new SLNode(new SLRectangle(SLVec2f(-100,-100), 
                                                  SLVec2f( 100, 100), 
                                                  SLVec2f(   0,   0), 
                                                  SLVec2f(  50,  50), 50, 50, "Floor", m1));
        rect->rotate(90, -1,0,0);
        rect->translate(0,0,-5.5f, TS_Object);

        SLNode* figure = BuildFigureGroup(m2);

        // Add animation for light 1
        SLAnimation* anim = SLAnimation::create("light1_anim", 4.0f);
        anim->createEllipticNodeTrack(light1, 12.0f, ZAxis, 12.0f, XAxis);


        // Assemble scene
        SLNode* scene = new SLNode("scene group");
        scene->addChild(light1);
        scene->addChild(rect);
        scene->addChild(figure);
        scene->addChild(cam1);

        // create spheres around the center sphere
        SLint size = 15;
        for (SLint iZ=-size; iZ<=size; ++iZ)
        {   for (SLint iX=-size; iX<=size; ++iX)
            {   if (iX!=0 || iZ!=0)
                {   SLNode* f = figure->copyRec();
                    f->translate(float(iX)*5, float(iZ)*5, 0, TS_Object);
                    scene->addChild(f);
                }
            }
        }

        // Set backround color, active camera & the root pointer
        _background.colors(SLCol4f(0.1f,0.4f,0.8f));
        sv->camera(cam1);
        _root3D = scene;
    }
    else
    if (sceneName == cmdScenePerVertexBlinn) //.................................
    {
        name("Blinn-Phong per vertex lighting");
        info(sv, "Per-vertex lighting with Blinn-Phong lightmodel. The reflection of 4 light sources is calculated per vertex and is then interpolated over the triangles.");

        // create material
        SLMaterial* m1 = new SLMaterial("m1", 0,0,0,0, _programs[PerVrtBlinn]);
        m1->shininess(500);

        SLCamera* cam1 = new SLCamera;
        cam1->translation(0,1,8);
        cam1->lookAt(0,1,0);
        cam1->focalDist(8);
        cam1->setInitialState();

        // define 4 light sources
        SLLightRect* light0 = new SLLightRect(2.0f,1.0f);
        light0->ambient(SLCol4f(0,0,0));
        light0->diffuse(SLCol4f(1,1,1));
        light0->translation(0,3,0);
        light0->lookAt(0,0,0, 0,0,-1);
        light0->attenuation(0,0,1);

        SLLightSphere* light1 = new SLLightSphere(0.1f);
        light1->ambient(SLCol4f(0,0,0));
        light1->diffuse(SLCol4f(1,0,0));
        light1->specular(SLCol4f(1,0,0));
        light1->translation(0, 0, 2);
        light1->lookAt(0, 0, 0);
        light1->attenuation(0,0,1);

        SLLightSphere* light2 = new SLLightSphere(0.1f);
        light2->ambient(SLCol4f(0,0,0));
        light2->diffuse(SLCol4f(0,1,0));
        light2->specular(SLCol4f(0,1,0));
        light2->translation(1.5, 1.5, 1.5);
        light2->lookAt(0, 0, 0);
        light2->spotCutoff(20);
        light2->attenuation(0,0,1);

        SLLightSphere* light3 = new SLLightSphere(0.1f);
        light3->ambient(SLCol4f(0,0,0));
        light3->diffuse(SLCol4f(0,0,1));
        light3->specular(SLCol4f(0,0,1));
        light3->translation(-1.5, 1.5, 1.5);
        light3->lookAt(0, 0, 0);
        light3->spotCutoff(20);
        light3->attenuation(0,0,1);

        // Assemble scene graph
        SLNode* scene = new SLNode;
        scene->addChild(cam1);
        scene->addChild(light0);
        scene->addChild(light1);
        scene->addChild(light2);
        scene->addChild(light3);
        scene->addChild(new SLNode(new SLSphere(1.0f, 20, 20, "Sphere", m1)));
        scene->addChild(new SLNode(new SLBox(1,-1,-1, 2,1,1, "Box", m1)));

        _background.colors(SLCol4f(0.1f,0.1f,0.1f));
        sv->camera(cam1);
        _root3D = scene;
    }
    else
    if (sceneName == cmdScenePerPixelBlinn) //..................................
    {
        name("Blinn-Phong per pixel lighting");
        info(sv, "Per-pixel lighting with Blinn-Phong lightmodel. The reflection of 4 light sources is calculated per pixel.");

        // create material
        SLMaterial* m1 = new SLMaterial("m1", 0,0,0,0, _programs[PerPixBlinn]);
        m1->shininess(500);

        SLCamera* cam1 = new SLCamera;
        cam1->translation(0,1,8);
        cam1->lookAt(0,1,0);
        cam1->focalDist(8);
        cam1->setInitialState();

        // define 4 light sources
        SLLightRect* light0 = new SLLightRect(2.0f,1.0f);
        light0->ambient(SLCol4f(0,0,0));
        light0->diffuse(SLCol4f(1,1,1));
        light0->translation(0,3,0);
        light0->lookAt(0,0,0, 0,0,-1);
        light0->attenuation(0,0,1);

        SLLightSphere* light1 = new SLLightSphere(0.1f);
        light1->ambient(SLCol4f(0,0,0));
        light1->diffuse(SLCol4f(1,0,0));
        light1->specular(SLCol4f(1,0,0));
        light1->translation(0, 0, 2);
        light1->lookAt(0, 0, 0);
        light1->attenuation(0,0,1);

        SLLightSphere* light2 = new SLLightSphere(0.1f);
        light2->ambient(SLCol4f(0,0,0));
        light2->diffuse(SLCol4f(0,1,0));
        light2->specular(SLCol4f(0,1,0));
        light2->translation(1.5, 1.5, 1.5);
        light2->lookAt(0, 0, 0);
        light2->spotCutoff(20);
        light2->attenuation(0,0,1);

        SLLightSphere* light3 = new SLLightSphere(0.1f);
        light3->ambient(SLCol4f(0,0,0));
        light3->diffuse(SLCol4f(0,0,1));
        light3->specular(SLCol4f(0,0,1));
        light3->translation(-1.5, 1.5, 1.5);
        light3->lookAt(0, 0, 0);
        light3->spotCutoff(20);
        light3->attenuation(0,0,1);

        // Assemble scene graph
        SLNode* scene = new SLNode;
        scene->addChild(cam1);
        scene->addChild(light0);
        scene->addChild(light1);
        scene->addChild(light2);
        scene->addChild(light3);
        scene->addChild(new SLNode(new SLSphere(1.0f, 20, 20, "Sphere", m1)));
        scene->addChild(new SLNode(new SLBox(1,-1,-1, 2,1,1, "Box", m1)));

        _background.colors(SLCol4f(0.1f,0.1f,0.1f));
        sv->camera(cam1);
        _root3D = scene;
    }
    else
    if (sceneName == cmdScenePerVertexWave) //..................................
    {
        name("Wave Shader");
        info(sv, "Vertex Shader with wave displacment.");
        cout << "Use H-Key to increment (decrement w. shift) the wave height.\n\n";

        SLCamera* cam1 = new SLCamera;
        cam1->translation(0,3,8);
        cam1->lookAt(0, 0, 0);
        cam1->focalDist(8);
        cam1->setInitialState();

        // Create generic shader program with 4 custom uniforms
        SLGLProgram* sp = new SLGLGenericProgram("Wave.vert", "Wave.frag");
        SLGLUniform1f* u_h = new SLGLUniform1f(UF1Const, "u_h", 0.1f, 0.05f, 0.0f, 0.5f, (SLKey)'H');
        _eventHandlers.push_back(u_h);
        sp->addUniform1f(u_h);
        sp->addUniform1f(new SLGLUniform1f(UF1Inc,    "u_t", 0.0f, 0.06f));
        sp->addUniform1f(new SLGLUniform1f(UF1Const,  "u_a", 2.5f));
        sp->addUniform1f(new SLGLUniform1f(UF1IncDec, "u_b", 2.2f, 0.01f, 2.0f, 2.5f));

        // Create materials
        SLMaterial* matWater = new SLMaterial("matWater", SLCol4f(0.45f,0.65f,0.70f),
                                                        SLCol4f::WHITE, 300);
        matWater->program(sp);
        SLMaterial* matRed  = new SLMaterial("matRed", SLCol4f(1.00f,0.00f,0.00f));

        // water rectangle in the y=0 plane
        SLNode* wave = new SLNode(new SLRectangle(SLVec2f(-SL_PI,-SL_PI), SLVec2f( SL_PI, SL_PI),
                                                    40, 40, "WaterRect", matWater));
        wave->rotate(90, -1,0,0);

        SLLightSphere* light0 = new SLLightSphere();
        light0->ambient(SLCol4f(0,0,0));
        light0->diffuse(SLCol4f(1,1,1));
        light0->translate(0,4,-4, TS_Object);
        light0->attenuation(1,0,0);

        SLNode* scene = new SLNode;
        scene->addChild(light0);
        scene->addChild(wave);
        scene->addChild(new SLNode(new SLSphere(1, 32, 32, "Red Sphere", matRed)));
        scene->addChild(cam1);

        _background.colors(SLCol4f(0.1f,0.4f,0.8f));
        sv->camera(cam1);
        _root3D = scene;
        sv->waitEvents(false);
    }
    else
    if (sceneName == cmdSceneWater) //..........................................
    {
        name("Water Shader");
        info(sv, "Water Shader with reflection & refraction mapping.");
        cout << "Use H-Key to increment (decrement w. shift) the wave height.\n\n";

        SLCamera* cam1 = new SLCamera;
        cam1->translation(0,3,8);
        cam1->lookAt(0, 0, 0);
        cam1->focalDist(8);

        // create texture
        SLGLTexture* tex1 = new SLGLTexture("Pool+X0512_C.png","Pool-X0512_C.png"
                                            ,"Pool+Y0512_C.png","Pool-Y0512_C.png"
                                            ,"Pool+Z0512_C.png","Pool-Z0512_C.png");
        SLGLTexture* tex2 = new SLGLTexture("tile1_0256_C.jpg");

        // Create generic shader program with 4 custom uniforms
        SLGLProgram* sp = new SLGLGenericProgram("WaveRefractReflect.vert",
                                                       "RefractReflect.frag");
        SLGLUniform1f* u_h = new SLGLUniform1f(UF1Const, "u_h", 0.1f, 0.05f, 0.0f, 0.5f, (SLKey)'H');
        _eventHandlers.push_back(u_h);
        sp->addUniform1f(u_h);
        sp->addUniform1f(new SLGLUniform1f(UF1Inc,    "u_t", 0.0f, 0.06f));
        sp->addUniform1f(new SLGLUniform1f(UF1Const,  "u_a", 2.5f));
        sp->addUniform1f(new SLGLUniform1f(UF1IncDec, "u_b", 2.2f, 0.01f, 2.0f, 2.5f));

        // Create materials
        SLMaterial* matWater = new SLMaterial("matWater", SLCol4f(0.45f,0.65f,0.70f),
                                                        SLCol4f::WHITE, 100, 0.1f, 0.9f, 1.5f);
        matWater->program(sp);
        matWater->textures().push_back(tex1);
        SLMaterial* matRed  = new SLMaterial("matRed", SLCol4f(1.00f,0.00f,0.00f));
        SLMaterial* matTile = new SLMaterial("matTile");
        matTile->textures().push_back(tex2);

        // water rectangle in the y=0 plane
        SLNode* rect = new SLNode(new SLRectangle(SLVec2f(-SL_PI,-SL_PI), 
                                                  SLVec2f( SL_PI, SL_PI),
                                                  40, 40, "WaterRect", matWater));
        rect->rotate(90, -1,0,0);

        // Pool rectangles
        SLNode* rectF = new SLNode(new SLRectangle(SLVec2f(-SL_PI,-SL_PI/6), SLVec2f( SL_PI, SL_PI/6),
                                                   SLVec2f(0,0), SLVec2f(10, 2.5f), 10, 10, "rectF", matTile));
        SLNode* rectN = new SLNode(new SLRectangle(SLVec2f(-SL_PI,-SL_PI/6), SLVec2f( SL_PI, SL_PI/6),
                                                   SLVec2f(0,0), SLVec2f(10, 2.5f), 10, 10, "rectN", matTile));
        SLNode* rectL = new SLNode(new SLRectangle(SLVec2f(-SL_PI,-SL_PI/6), SLVec2f( SL_PI, SL_PI/6),
                                                   SLVec2f(0,0), SLVec2f(10, 2.5f), 10, 10, "rectL", matTile));
        SLNode* rectR = new SLNode(new SLRectangle(SLVec2f(-SL_PI,-SL_PI/6), SLVec2f( SL_PI, SL_PI/6),
                                                   SLVec2f(0,0), SLVec2f(10, 2.5f), 10, 10, "rectR", matTile));
        SLNode* rectB = new SLNode(new SLRectangle(SLVec2f(-SL_PI,-SL_PI  ), SLVec2f( SL_PI, SL_PI  ),
                                                   SLVec2f(0,0), SLVec2f(10, 10  ), 10, 10, "rectB", matTile));
        rectF->translate(0,0,-SL_PI, TS_Object);
        rectL->rotate( 90, 0,1,0); rectL->translate(0,0,-SL_PI, TS_Object);
        rectN->rotate(180, 0,1,0); rectN->translate(0,0,-SL_PI, TS_Object);
        rectR->rotate(270, 0,1,0); rectR->translate(0,0,-SL_PI, TS_Object);
        rectB->rotate( 90,-1,0,0); rectB->translate(0,0,-SL_PI/6, TS_Object);

        SLLightSphere* light0 = new SLLightSphere();
        light0->ambient(SLCol4f(0,0,0));
        light0->diffuse(SLCol4f(1,1,1));
        light0->translate(0,4,-4, TS_Object);
        light0->attenuation(1,0,0);

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

        _background.colors(SLCol4f(0.1f,0.4f,0.8f));
        sv->camera(cam1);
        _root3D = scene;
        sv->waitEvents(false);
    }
    else
    if (sceneName == cmdSceneBumpNormal) //.....................................
    {
        name("Normal Map Bump Mapping");
        info(sv, "Normal map bump mapping combined with a per pixel spot lighting.");

        // Create textures
        SLGLTexture* texC = new SLGLTexture("brickwall0512_C.jpg");
        SLGLTexture* texN = new SLGLTexture("brickwall0512_N.jpg");

        // Create materials
        SLMaterial* m1 = new SLMaterial("m1", texC, texN, 0, 0, _programs[BumpNormal]);

        SLCamera* cam1 = new SLCamera();
        cam1->name("cam1");
        cam1->translation(0,0,20);
        cam1->lookAt(0, 0, 0);
        cam1->focalDist(20);

        SLLightSphere* light1 = new SLLightSphere(0.3f);
        light1->ambient(SLCol4f(0.1f, 0.1f, 0.1f));
        light1->diffuse(SLCol4f(1, 1, 1));
        light1->specular(SLCol4f(1, 1, 1));
        light1->attenuation(1,0,0);
        light1->translation(0,0,5);
        light1->lookAt(0, 0, 0);
        light1->spotCutoff(40);

        SLAnimation* anim = SLAnimation::create("light1_anim", 2.0f);
        anim->createEllipticNodeTrack(light1, 2.0f, XAxis, 2.0f, YAxis);

        SLNode* scene = new SLNode;
        scene->addChild(light1);
        scene->addChild(new SLNode(new SLRectangle(SLVec2f(-5,-5),SLVec2f(5,5),1,1,"Rect", m1)));
        scene->addChild(cam1);

        _background.colors(SLCol4f(0.5f,0.5f,0.5f));
        sv->camera(cam1);
        _root3D = scene;
    }
    else
    if (sceneName == cmdSceneBumpParallax) //...................................
    {
        name("Parallax Bump Mapping");
        cout << "Demo application for parallax bump mapping.\n";
        cout << "Use S-Key to increment (decrement w. shift) parallax scale.\n";
        cout << "Use O-Key to increment (decrement w. shift) parallax offset.\n\n";
        info(sv, "Normal map parallax mapping.");

        // Create shader program with 4 uniforms
        SLGLProgram* sp = new SLGLGenericProgram("BumpNormal.vert", "BumpNormalParallax.frag");
        SLGLUniform1f* scale = new SLGLUniform1f(UF1Const, "u_scale", 0.04f, 0.002f, 0, 1, (SLKey)'X');
        SLGLUniform1f* offset = new SLGLUniform1f(UF1Const, "u_offset", -0.03f, 0.002f,-1, 1, (SLKey)'O');
        _eventHandlers.push_back(scale);
        _eventHandlers.push_back(offset);
        sp->addUniform1f(scale);
        sp->addUniform1f(offset);

        // Create textures
        SLGLTexture* texC = new SLGLTexture("brickwall0512_C.jpg");
        SLGLTexture* texN = new SLGLTexture("brickwall0512_N.jpg");
        SLGLTexture* texH = new SLGLTexture("brickwall0512_H.jpg");

        // Create materials
        SLMaterial* m1 = new SLMaterial("mat1", texC, texN, texH, 0, sp);

        SLCamera* cam1 = new SLCamera();
        cam1->name("cam1");
        cam1->translation(0,0,20);
        cam1->lookAt(0, 0, 0);
        cam1->focalDist(20);

        SLLightSphere* light1 = new SLLightSphere(0.3f);
        light1->ambient(SLCol4f(0.1f, 0.1f, 0.1f));
        light1->diffuse(SLCol4f(1, 1, 1));
        light1->specular(SLCol4f(1, 1, 1));
        light1->attenuation(1,0,0);
        light1->translation(0,0,5);
        light1->lookAt(0, 0, 0);
        light1->spotCutoff(50);

        SLAnimation* anim = SLAnimation::create("light1_anim", 2.0f);
        anim->createEllipticNodeTrack(light1, 2.0f, XAxis, 2.0f, YAxis);

        SLNode* scene = new SLNode;
        scene->addChild(light1);
        scene->addChild(new SLNode(new SLRectangle(SLVec2f(-5,-5),SLVec2f(5,5),1,1,"Rect", m1)));
        scene->addChild(cam1);

        _background.colors(SLCol4f(0.5f,0.5f,0.5f));
        sv->camera(cam1);
        _root3D = scene;
    }
    else
    if (sceneName == cmdSceneEarth) //..........................................
    {
        name("Earth Shader from Markus Knecht");
        cout << "Earth Shader from Markus Knecht\n";
        cout << "Use (SHIFT) & key Y to change scale of the parallax mapping\n";
        cout << "Use (SHIFT) & key X to change bias of the parallax mapping\n";
        cout << "Use (SHIFT) & key C to change cloud height\n";
        info(sv, "Complex earth shader with 7 textures: daycolor, nightcolor, normal, height & gloss map of earth, color & alphamap of clouds");

        // Create shader program with 4 uniforms
        SLGLProgram* sp = new SLGLGenericProgram("BumpNormal.vert", "BumpNormalEarth.frag");
        SLGLUniform1f* scale = new SLGLUniform1f(UF1Const, "u_scale", 0.02f, 0.002f, 0, 1, (SLKey)'X');
        SLGLUniform1f* offset = new SLGLUniform1f(UF1Const, "u_offset", -0.02f, 0.002f,-1, 1, (SLKey)'O');
        _eventHandlers.push_back(scale);
        _eventHandlers.push_back(offset);
        sp->addUniform1f(scale);
        sp->addUniform1f(offset);

        // Create textures
        #ifndef SL_GLES2
        SLGLTexture* texC   = new SLGLTexture("earth2048_C.jpg"); // color map
        SLGLTexture* texN   = new SLGLTexture("earth2048_N.jpg"); // normal map
        SLGLTexture* texH   = new SLGLTexture("earth2048_H.jpg"); // height map
        SLGLTexture* texG   = new SLGLTexture("earth2048_G.jpg"); // gloss map
        SLGLTexture* texNC  = new SLGLTexture("earthNight2048_C.jpg"); // night color  map
        #else
        SLGLTexture* texC   = new SLGLTexture("earth1024_C.jpg"); // color map
        SLGLTexture* texN   = new SLGLTexture("earth1024_N.jpg"); // normal map
        SLGLTexture* texH   = new SLGLTexture("earth1024_H.jpg"); // height map
        SLGLTexture* texG   = new SLGLTexture("earth1024_G.jpg"); // gloss map
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

        SLCamera* cam1 = new SLCamera;
        cam1->translation(0,0,4);
        cam1->lookAt(0, 0, 0);
        cam1->focalDist(4);

        SLLightSphere* sun = new SLLightSphere();
        sun->ambient(SLCol4f(0,0,0));
        sun->diffuse(SLCol4f(1,1,1));
        sun->specular(SLCol4f(0.2f,0.2f,0.2f));
        sun->attenuation(1,0,0);
        
        SLAnimation* anim = SLAnimation::create("light1_anim", 24.0f);
        anim->createEllipticNodeTrack(sun, 50.0f, XAxis, 50.0f, ZAxis);

        SLNode* earth = new SLNode(new SLSphere(1, 36, 36, "Earth", matEarth));
        earth->rotate(90,-1,0,0);

        SLNode* scene = new SLNode;
        scene->addChild(sun);
        scene->addChild(earth);
        scene->addChild(cam1);

        _background.colors(SLCol4f(0,0,0));
        sv->camera(cam1);
        _root3D = scene;
    }
    else
    if (sceneName == cmdSceneSkeletalAnimation) //..............................
    {
        name("Skeletal Animation Test");
        info(sv, "Skeletal Animation Test Scene");

        SLAssimpImporter importer;
        #if defined(SL_OS_IOS) || defined(SL_OS_ANDROID)
        SLNode* character = importer.load("AstroBoy.dae");
        #else
        SLNode* character = importer.load("DAE/AstroBoy/AstroBoy.dae");
        #endif
        SLAnimPlayback* charAnim = importer.skeleton()->getAnimPlayback("unnamed_anim_0");
        
        #if defined(SL_OS_IOS) || defined(SL_OS_ANDROID)
        SLNode* box1 = importer.load("skinnedcube2.dae");
        #else
        SLNode* box1 = importer.load("DAE/SkinnedCube/skinnedcube2.dae");
        #endif
        SLAnimPlayback* box1Anim = importer.skeleton()->getAnimPlayback("unnamed_anim_0");

        #if defined(SL_OS_IOS) || defined(SL_OS_ANDROID)
        SLNode* box2 = importer.load("skinnedcube4.dae");
        #else
        SLNode* box2 = importer.load("DAE/SkinnedCube/skinnedcube4.dae");
        #endif
        SLAnimPlayback* box2Anim = importer.skeleton()->getAnimPlayback("unnamed_anim_0");

        #if defined(SL_OS_IOS) || defined(SL_OS_ANDROID)
        SLNode* box3 = importer.load("skinnedcube5.dae");
        #else
        SLNode* box3 = importer.load("DAE/SkinnedCube/skinnedcube5.dae");
        #endif
        SLAnimPlayback* box3Anim = importer.skeleton()->getAnimPlayback("unnamed_anim_0");

        box1->translate(3, 0, 0);
        box2->translate(-3, 0, 0);
        box3->translate(0, 3, 0);

        box1Anim->easing(EC_inOutSine);
        box2Anim->easing(EC_inOutSine);
        box3Anim->loop(AL_pingPongLoop);
        box3Anim->easing(EC_inOutCubic);

        charAnim->playForward();
        box1Anim->playForward();
        box2Anim->playForward();
        box3Anim->playForward();

        // Define camera
        SLCamera* cam1 = new SLCamera();
        cam1->translation(0,2,10);
        cam1->lookAt(0, 2, 0);
        cam1->setInitialState();

        // Define a light
        SLLightSphere* light1 = new SLLightSphere(10, 10, 5, 0.5f);
        light1->ambient (SLCol4f(0.2f,0.2f,0.2f));
        light1->diffuse (SLCol4f(0.9f,0.9f,0.9f));
        light1->specular(SLCol4f(0.9f,0.9f,0.9f));
        light1->attenuation(1,0,0);


        SLMaterial* m2 = new SLMaterial(SLCol4f::WHITE);
        SLGrid* grid = new SLGrid(SLVec3f(-5,0,-5), SLVec3f(5,0,5), 20, 20, "Grid", m2);

        // Assemble scene
        SLNode* scene = new SLNode("scene group");
        scene->addChild(cam1);
        scene->addChild(light1);
        scene->addChild(character);
        scene->addChild(box1);
        scene->addChild(box2);
        scene->addChild(box3);
        scene->addChild(new SLNode(grid, "grid"));

        // Set backround color, active camera & the root pointer
        _background.colors(SLCol4f(0.1f,0.4f,0.8f));
        sv->camera(cam1);
        _root3D = scene;
    }
    else
    if (sceneName == cmdSceneNodeAnimation) //..................................
    {
        name("Node Animations");
        info(sv, "Node animations with different easing curves.");

        // Create textures and materials
        SLGLTexture* tex1 = new SLGLTexture("Checkerboard0512_C.png");
        SLMaterial* m1 = new SLMaterial("m1", tex1); m1->kr(0.5f);
        SLMaterial* m2 = new SLMaterial("m2", SLCol4f::WHITE*0.5, SLCol4f::WHITE,128, 0.5f, 0.0f, 1.0f);

        SLMesh* floorMesh = new SLRectangle(SLVec2f(-5,-5), SLVec2f(5,5), 20, 20, "FloorMesh", m1);
        SLNode* floorRect = new SLNode(floorMesh);
        floorRect->rotate(90, -1,0,0);
        floorRect->translate(0,0,-5.5f);

        // Bouncing balls
        SLNode* ball1 = new SLNode(new SLSphere(0.3f, 16, 16, "Ball1", m2));
        ball1->translate(0,0,4, TS_Object);
        SLAnimation* ball1Anim = SLAnimation::create("Ball1_anim", 1.0f, true, EC_linear, AL_pingPongLoop);
        ball1Anim->createSimpleTranslationNodeTrack(ball1, SLVec3f(0.0f, -5.2f, 0.0f));

        SLNode* ball2 = new SLNode(new SLSphere(0.3f, 16, 16, "Ball2", m2));
        ball2->translate(-1.5f,0,4, TS_Object);
        SLAnimation* ball2Anim = SLAnimation::create("Ball2_anim", 1.0f, true, EC_inQuad, AL_pingPongLoop);
        ball2Anim->createSimpleTranslationNodeTrack(ball2, SLVec3f(0.0f, -5.2f, 0.0f));

        SLNode* ball3 = new SLNode(new SLSphere(0.3f, 16, 16, "Ball3", m2));
        ball3->translate(-2.5f,0,4, TS_Object);
        SLAnimation* ball3Anim = SLAnimation::create("Ball3_anim", 1.0f, true, EC_outQuad, AL_pingPongLoop);
        ball3Anim->createSimpleTranslationNodeTrack(ball3, SLVec3f(0.0f, -5.2f, 0.0f));

        SLNode* ball4 = new SLNode(new SLSphere(0.3f, 16, 16, "Ball4", m2));
        ball4->translate( 1.5f,0,4, TS_Object);
        SLAnimation* ball4Anim = SLAnimation::create("Ball4_anim", 1.0f, true, EC_inOutQuad, AL_pingPongLoop);
        ball4Anim->createSimpleTranslationNodeTrack(ball4, SLVec3f(0.0f, -5.2f, 0.0f));

        SLNode* ball5 = new SLNode(new SLSphere(0.3f, 16, 16, "Ball5", m2));
        ball5->translate( 2.5f,0,4, TS_Object);
        SLAnimation* ball5Anim = SLAnimation::create("Ball5_anim", 1.0f, true, EC_outInQuad, AL_pingPongLoop);
        ball5Anim->createSimpleTranslationNodeTrack(ball5, SLVec3f(0.0f, -5.2f, 0.0f));

        SLCamera* cam1 = new SLCamera();
        cam1->translation(0, 0, 22);
        cam1->lookAt(0, 0, 0);
        cam1->focalDist(22);
        cam1->setInitialState();
        SLCamera* cam2 = new SLCamera;
        cam2->translation(5, 0, 0);
        cam2->lookAt(0, 0, 0);
        cam2->focalDist(5);
        cam2->setInitialState();

        SLLightSphere* light1 = new SLLightSphere(0, 2, 0, 0.5f);
        light1->ambient (SLCol4f(0.2f,0.2f,0.2f));
        light1->diffuse (SLCol4f(0.9f,0.9f,0.9f));
        light1->specular(SLCol4f(0.9f,0.9f,0.9f));
        light1->attenuation(1,0,0);
        SLAnimation* light1Anim = SLAnimation::create("Light1_anim", 4.0f);
        light1Anim->createEllipticNodeTrack(light1, 6, ZAxis, 6, XAxis);

        SLLightSphere* light2 = new SLLightSphere(0, 0, 0, 0.2f);
        light2->ambient (SLCol4f(0.2f,0.0f,0.0f));
        light2->diffuse (SLCol4f(0.9f,0.0f,0.0f));
        light2->specular(SLCol4f(0.9f,0.9f,0.9f));
        light2->attenuation(1,0,0);
        light2->translate(-8, -4, 0, TS_World);
        light2->setInitialState();

        SLAnimation* light2Anim = SLAnimation::create("light2_anim", 2.0f, true, EC_linear, AL_pingPongLoop);
        SLNodeAnimTrack* track = light2Anim->createNodeAnimationTrack();
        track->animatedNode(light2);
        track->createNodeKeyframe(0.0f);
        track->createNodeKeyframe(1.0f)->translation(SLVec3f(8, 8, 0));
        track->createNodeKeyframe(2.0f)->translation(SLVec3f(16, 0, 0));
        track->translationInterpolation(AI_Bezier);

        SLNode* figure = BuildFigureGroup(m2, true);

        SLNode* scene = new SLNode("Scene");
        scene->addChild(light1);
        scene->addChild(light2);
        scene->addChild(cam1);
        scene->addChild(cam2);
        scene->addChild(floorRect);
        scene->addChild(ball1);
        scene->addChild(ball2);
        scene->addChild(ball3);
        scene->addChild(ball4);
        scene->addChild(ball5);
        scene->addChild(figure);

        // Set backround color, active camera & the root pointer
        _background.colors(SLCol4f(0.1f,0.4f,0.8f));
        sv->camera(cam1);
        _root3D = scene;
    }
    else
    if (sceneName == cmdSceneMassAnimation) //..................................
    {
        name("Mass Animation");
        info(sv, "Performance test for transform updates from many animations.");

        init();

        SLLightSphere* light1 = new SLLightSphere(7,7,0, 0.1f, 5, 10);
        light1->attenuation(0,0,1);
        light1->translate(-3, 5, 2, TS_Object);

        // build a basic scene to have a reference for the occuring rotations
        SLMaterial* genericMat = new SLMaterial("some material");

        // we use the same mesh to viasualize all the nodes
        SLBox* box = new SLBox(-0.5f, -0.5f, -0.5f, 0.5f, 0.5f, 0.5f, "box", genericMat);

        _root3D = new SLNode;
        _root3D->addChild(light1);

        // we build a stack of levels, each level has a grid of boxes on it
        // each box on this grid has an other grid above it with child nodes
        // best results are achieved if gridSize is an uneven number.
        // (gridSize^2)^levels = num nodes. handle with care.
        const SLint levels =3;
        const SLint gridSize = 3;
        const SLint gridHalf = gridSize / 2;
        const SLint nodesPerLvl = gridSize * gridSize;


        // node spacing per level
        // nodes are 1^3 in size, we want to space the levels so that the densest levels meet
        // (so exactly 1 unit spacing between blocks)
        SLfloat nodeSpacing[levels];
        for(SLint i = 0; i < levels; ++i)
            nodeSpacing[(levels-1)-i] = (SLfloat)pow((SLfloat)gridSize, (SLfloat)i);

        // lists to keep track of previous grid level to set parents correctly
        vector<SLNode*> parents;
        vector<SLNode*> curParentsVector;

        // first parent is the scene root
        parents.push_back(_root3D);

        SLint nodeIndex = 0;
        for(SLint lvl = 0; lvl < levels; ++lvl)
        {   curParentsVector = parents;
            parents.clear();

            // for each parent in the previous level, add a completely new grid
            for(auto parent : curParentsVector)
            {   for(SLint i = 0; i < nodesPerLvl; ++i)
                {   SLNode* node = new SLNode("MassAnimNode");
                    node->addMesh(box);
                    parent->addChild(node);
                    parents.push_back(node);

                    // position
                    SLfloat x = (SLfloat)(i % gridSize - gridHalf);
                    SLfloat z = (SLfloat)((i > 0) ? i / gridSize - gridHalf : -gridHalf);
                    SLVec3f pos(x*nodeSpacing[lvl] *1.1f, 1.5f, z*nodeSpacing[lvl]*1.1f);

                    node->translate(pos, TS_Object);
                    //node->scale(1.1f);

                    SLfloat duration = 1.0f + 5.0f * ((SLfloat)i/(SLfloat)nodesPerLvl);
                    ostringstream oss;;
                    oss << "random anim " << nodeIndex++;
                    SLAnimation* anim = SLAnimation::create(oss.str(), duration, true, EC_inOutSine, AL_pingPongLoop);
                    anim->createSimpleTranslationNodeTrack(node, SLVec3f(0.0f, 1.0f, 0.0f));
                }
            }
        }
    }
    else
    if (sceneName == cmdSceneAstroboyArmyCPU ||
        sceneName == cmdSceneAstroboyArmyGPU) //................................
    {
        info(sv, "Mass animation scene of identitcal Astroboy models");

        // Create materials
        SLMaterial* m1 = new SLMaterial("m1", SLCol4f::GRAY); m1->specular(SLCol4f::BLACK);

        // Define a light
        SLLightSphere* light1 = new SLLightSphere(100, 40, 100, 1);
        light1->ambient (SLCol4f(0.2f,0.2f,0.2f));
        light1->diffuse (SLCol4f(0.9f,0.9f,0.9f));
        light1->specular(SLCol4f(0.9f,0.9f,0.9f));
        light1->attenuation(1,0,0);

        // Define camera
        SLCamera* cam1 = new SLCamera;
        cam1->translation(0, 20, 20);
        cam1->lookAt(0, 0, 0);
        cam1->setInitialState();

        // Floor rectangle
        SLNode* rect = new SLNode(new SLRectangle(SLVec2f(-20,-20),
                                                  SLVec2f( 20, 20),
                                                  SLVec2f(   0,   0),
                                                  SLVec2f(  50,  50), 50, 50, "Floor", m1));
        rect->rotate(90, -1,0,0);

        SLAssimpImporter importer;
        #if defined(SL_OS_IOS) || defined(SL_OS_ANDROID)
        SLNode* center = importer.load("AstroBoy.dae");
        #else
        SLNode* center = importer.load("DAE/AstroBoy/AstroBoy.dae");
        #endif
        //center->scale(100);
        importer.skeleton()->getAnimPlayback("unnamed_anim_0")->playForward();

        // set the skinning method of the loaded meshes
        // @note RT currently only works with software skinning
        if (sceneName == cmdSceneAstroboyArmyGPU)
        {   name("Astroboy army skinned on GPU");
            for (auto m : importer.meshes())
               m->skinMethod(SM_HardwareSkinning);
        } else
            name("Astroboy army skinned on CPU");


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
        SLint size = 8;
        #endif
        for (SLint iZ=-size; iZ<=size; ++iZ)
        {   for (SLint iX=-size; iX<=size; ++iX)
            {   SLbool shift = iX%2 != 0;
                if (iX!=0 || iZ!=0)
                {   SLNode* n = new SLNode;
                    float xt = float(iX) * 1.0f;
                    float zt = float(iZ) * 1.0f + ((shift) ? 0.5f : 0.0f);
                    n->translate(xt, 0, zt, TS_Object);
                    for (auto m : importer.meshes())
                        n->addMesh(m);
                    scene->addChild(n);
                }
            }
        }

        // Set backround color, active camera & the root pointer
        _background.colors(SLCol4f(0.1f,0.4f,0.8f));
        sv->camera(cam1);
        _root3D = scene;
    }
    else
    if (sceneName == cmdSceneRTMuttenzerBox) //.................................
    {
        name("Muttenzer Box (RT)");
        info(sv, "Muttenzer Box with environment mapped reflective sphere and transparenz refractive glass sphere. Try ray tracing for real reflections and soft shadows.",
            SLCol4f::GRAY);
      
        // Create reflection & glass shaders
        SLGLProgram* sp1 = new SLGLGenericProgram("Reflect.vert", "Reflect.frag");
        SLGLProgram* sp2 = new SLGLGenericProgram("RefractReflect.vert", "RefractReflect.frag");
   
        // Create cube mapping texture
        SLGLTexture* tex1 = new SLGLTexture("MuttenzerBox+X0512_C.png", "MuttenzerBox-X0512_C.png"
                                            ,"MuttenzerBox+Y0512_C.png", "MuttenzerBox-Y0512_C.png"
                                            ,"MuttenzerBox+Z0512_C.png", "MuttenzerBox-Z0512_C.png");
      
        SLCol4f  lightEmisRGB(7.0f, 7.0f, 7.0f);
        SLCol4f  grayRGB  (0.75f, 0.75f, 0.75f);
        SLCol4f  redRGB   (0.75f, 0.25f, 0.25f);
        SLCol4f  blueRGB  (0.25f, 0.25f, 0.75f);
        SLCol4f  blackRGB (0.00f, 0.00f, 0.00f);

        // create materials
        SLMaterial* cream = new SLMaterial("cream", grayRGB,  SLCol4f::BLACK, 0);
        SLMaterial* red   = new SLMaterial("red",   redRGB ,  SLCol4f::BLACK, 0);
        SLMaterial* blue  = new SLMaterial("blue",  blueRGB,  SLCol4f::BLACK, 0);

        // Material for mirror sphere
        SLMaterial* refl=new SLMaterial("refl", blackRGB, SLCol4f::WHITE, 1000, 1.0f);
        refl->textures().push_back(tex1);
        refl->program(sp1);

        // Material for glass sphere
        SLMaterial* refr=new SLMaterial("refr", blackRGB, blackRGB, 100, 0.05f, 0.95f, 1.5f);
        refr->translucency(1000);
        refr->transmission(SLCol4f::WHITE);
        refr->textures().push_back(tex1);
        refr->program(sp2);
   
        SLNode* sphere1 = new SLNode(new SLSphere(0.5f, 32, 32, "Sphere1", refl));
        sphere1->translate(-0.65f, -0.75f, -0.55f, TS_Object);

        SLNode* sphere2 = new SLNode(new SLSphere(0.45f, 32, 32, "Sphere2", refr));
        sphere2->translate( 0.73f, -0.8f, 0.10f, TS_Object);

        SLNode* balls = new SLNode;
        balls->addChild(sphere1);
        balls->addChild(sphere2);

        // Rectangular light 
        SLLightRect* lightRect = new SLLightRect(1, 0.65f);
        lightRect->rotate(90, -1.0f, 0.0f, 0.0f);
        lightRect->translate(0.0f, -0.25f, 1.18f, TS_Object);
        lightRect->spotCutoff(90);
        lightRect->spotExponent(1.0);
        lightRect->diffuse(lightEmisRGB);
        lightRect->attenuation(0,0,1);
        lightRect->samplesXY(11, 7);

        _globalAmbiLight.set(lightEmisRGB*0.05f);

        // create camera
        SLCamera* cam1 = new SLCamera();
        cam1->translation(0.0f, 0.40f, 6.35f);
        cam1->lookAt(0.0f,-0.05f, 0.0f);
        cam1->fov(27);
        cam1->focalDist(6.35f);
      
        // assemble scene
        SLNode* scene = new SLNode;
        scene->addChild(cam1);
        scene->addChild(lightRect);
      
        // create wall polygons    
        SLfloat pL = -1.48f, pR = 1.48f; // left/right
        SLfloat pB = -1.25f, pT = 1.19f; // bottom/top
        SLfloat pN =  1.79f, pF =-1.55f; // near/far
      
        // bottom plane
        SLNode* b = new SLNode(new SLRectangle(SLVec2f(pL,-pN), SLVec2f(pR,-pF), 6, 6, "bottom", cream)); 
        b->rotate(90, -1,0,0); b->translate(0,0,pB,TS_Object); scene->addChild(b);
   
        // top plane
        SLNode* t = new SLNode(new SLRectangle(SLVec2f(pL,pF), SLVec2f(pR,pN), 6, 6, "top", cream)); 
        t->rotate(90, 1,0,0); t->translate(0,0,-pT,TS_Object); scene->addChild(t);
   
        // far plane
        SLNode* f = new SLNode(new SLRectangle(SLVec2f(pL,pB), SLVec2f(pR,pT), 6, 6, "far", cream)); 
        f->translate(0,0,pF,TS_Object); scene->addChild(f);
   
        // left plane
        SLNode* l = new SLNode(new SLRectangle(SLVec2f(-pN,pB), SLVec2f(-pF,pT), 6, 6, "left", red)); 
        l->rotate(90, 0,1,0); l->translate(0,0,pL,TS_Object); scene->addChild(l);
   
        // right plane
        SLNode* r = new SLNode(new SLRectangle(SLVec2f(pF,pB), SLVec2f(pN,pT), 6, 6, "right", blue)); 
        r->rotate(90, 0,-1,0); r->translate(0,0,-pR,TS_Object); scene->addChild(r);
      
        scene->addChild(balls);

        _background.colors(SLCol4f(0.0f,0.0f,0.0f));
        sv->camera(cam1);
        _root3D = scene;
    }
    else
    if (sceneName == cmdSceneRTSpheres) //......................................
    {
        name("Ray tracing Spheres");
        info(sv, "Classic ray tracing scene with transparent and reflective spheres. Be patient on mobile devices.");

        // define materials
        SLMaterial* matGla = new SLMaterial("Glass", SLCol4f(0.0f, 0.0f, 0.0f),
                                                     SLCol4f(0.5f, 0.5f, 0.5f),
                                                     100, 0.4f, 0.6f, 1.5f);
        SLMaterial* matRed = new SLMaterial("Red",   SLCol4f(0.5f, 0.0f, 0.0f),
                                                     SLCol4f(0.5f, 0.5f, 0.5f),
                                                     100, 0.5f, 0.0f, 1.0f);
        SLMaterial* matYel = new SLMaterial("Floor", SLCol4f(0.8f, 0.6f, 0.2f),
                                                     SLCol4f(0.8f, 0.8f, 0.8f),
                                                     100, 0.5f, 0.0f, 1.0f);

        SLCamera* cam1 = new SLCamera();
        cam1->translation(0, 0.1f, 2.5f);
        cam1->lookAt(0, 0, 0);
        cam1->focalDist(4);

        SLNode *rect = new SLNode(new SLRectangle(SLVec2f(-3,-3), SLVec2f(5,4), 20, 20, "Floor", matYel));
        rect->rotate(90, -1,0,0);
        rect->translate(0, -1, -0.5f, TS_Object);

        SLLightSphere* light1 = new SLLightSphere(2, 2, 2, 0.1f);
        light1->ambient(SLCol4f(1, 1, 1));
        light1->diffuse(SLCol4f(7, 7, 7));
        light1->specular(SLCol4f(7, 7, 7));
        light1->attenuation(0,0,1);

        SLLightSphere* light2 = new SLLightSphere(2, 2, -2, 0.1f);
        light2->ambient(SLCol4f(1, 1, 1));
        light2->diffuse(SLCol4f(7, 7, 7));
        light2->specular(SLCol4f(7, 7, 7));
        light2->attenuation(0,0,1);

        SLNode* scene  = new SLNode;
        scene->addChild(light1);
        scene->addChild(light2);
        scene->addChild(SphereGroup(2, 0,0,0, 1, 32, matGla, matRed));
        scene->addChild(rect);
        scene->addChild(cam1);

        _background.colors(SLCol4f(0.1f,0.4f,0.8f));
        _root3D = scene;
        sv->camera(cam1);
    }
    else
    if (sceneName == cmdSceneRTSoftShadows) //..................................
    {
        name("Ray tracing Softshadows");
        info(sv, "Ray tracing with soft shadow light sampling. Each light source is sampled 64x per pixel. Be patient on mobile devices.");

        // define materials
        SLCol4f spec(0.8f, 0.8f, 0.8f);
        SLMaterial* matBlk = new SLMaterial("Glass", SLCol4f(0.0f, 0.0f, 0.0f), SLCol4f(0.5f, 0.5f, 0.5f), 100, 0.5f, 0.5f, 1.5f);
        SLMaterial* matRed = new SLMaterial("Red",   SLCol4f(0.5f, 0.0f, 0.0f), SLCol4f(0.5f, 0.5f, 0.5f), 100, 0.5f, 0.0f, 1.0f);
        SLMaterial* matYel = new SLMaterial("Floor", SLCol4f(0.8f, 0.6f, 0.2f), SLCol4f(0.8f, 0.8f, 0.8f), 100, 0.0f, 0.0f, 1.0f);

        SLCamera* cam1 = new SLCamera;
        cam1->translation(0, 0.1f, 6);
        cam1->lookAt(0, 0, 0);
        cam1->focalDist(6);

        SLNode* rect = new SLNode(new SLRectangle(SLVec2f(-3,-3), SLVec2f(5,4), 32, 32, "Rect", matYel));
        rect->rotate(90, -1,0,0);
        rect->translate(0, -1, -0.5f, TS_Object);

        SLLightSphere* light1 = new SLLightSphere(3, 3, 3, 0.3f);
        #ifndef SL_GLES2
        SLint numSamples = 6;
        #else
        SLint numSamples = 8;
        #endif
        light1->samples(numSamples, numSamples);
        light1->attenuation(0,0,1);
        //light1->lightAt(2,2,2, 0,0,0);
        //light1->spotCutoff(15);
        light1->translation(2, 2, 2);
        light1->lookAt(0, 0, 0);

        SLLightSphere* light2 = new SLLightSphere(0, 1.5, -1.5, 0.3f);
        light2->samples(8,8);
        light2->attenuation(0,0,1);

        SLNode* scene  = new SLNode;
        scene->addChild(light1);
        scene->addChild(light2);
        scene->addChild(SphereGroup(1, 0,0,0, 1, 32, matBlk, matRed));
        scene->addChild(rect);
        scene->addChild(cam1);

        _background.colors(SLCol4f(0.1f,0.4f,0.8f));
        sv->camera(cam1);
        _root3D = scene;
    }
    else
    if (sceneName == cmdSceneRTDoF) //..........................................
    {
        name("Ray tracing: Depth of Field");
        info(sv, "Ray tracing with depth of field blur. Each pixel is sampled 100x from a lens. Be patient on mobile devices.");

        // Create textures and materials
        SLGLTexture* texC = new SLGLTexture("Checkerboard0512_C.png");
        SLMaterial* mT = new SLMaterial("mT", texC, 0, 0, 0); mT->kr(0.5f);
        SLMaterial* mW = new SLMaterial("mW", SLCol4f::WHITE);
        SLMaterial* mB = new SLMaterial("mB", SLCol4f::GRAY);
        SLMaterial* mY = new SLMaterial("mY", SLCol4f::YELLOW);
        SLMaterial* mR = new SLMaterial("mR", SLCol4f::RED);
        SLMaterial* mG = new SLMaterial("mG", SLCol4f::GREEN);
        SLMaterial* mM = new SLMaterial("mM", SLCol4f::MAGENTA);

        #ifndef SL_GLES2
        SLint numSamples = 6;
        #else
        SLint numSamples = 10;
        #endif

        SLCamera* cam1 = new SLCamera;
        cam1->translation(0, 2, 7);
        cam1->lookAt(0, 0, 0);
        cam1->focalDist(7);
        cam1->lensDiameter(0.4f);
        cam1->lensSamples()->samples(numSamples, numSamples);
        cam1->setInitialState();

        SLNode* rect = new SLNode(new SLRectangle(SLVec2f(-5,-5), SLVec2f(5,5), 20, 20, "Rect", mT));
        rect->rotate(90, -1,0,0);
        rect->translate(0,0,-0.5f, TS_Object);

        SLLightSphere* light1 = new SLLightSphere(2,2,0, 0.1f);
        light1->attenuation(0,0,1);

        SLNode* balls = new SLNode;
        SLNode* s;
        s = new SLNode(new SLSphere(0.5f,32,32,"S1",mW)); s->translate( 2.0,0,-4,TS_Object);  balls->addChild(s);
        s = new SLNode(new SLSphere(0.5f,32,32,"S2",mB)); s->translate( 1.5,0,-3,TS_Object);  balls->addChild(s);
        s = new SLNode(new SLSphere(0.5f,32,32,"S3",mY)); s->translate( 1.0,0,-2,TS_Object);  balls->addChild(s);
        s = new SLNode(new SLSphere(0.5f,32,32,"S4",mR)); s->translate( 0.5,0,-1,TS_Object);  balls->addChild(s);
        s = new SLNode(new SLSphere(0.5f,32,32,"S5",mG)); s->translate( 0.0,0, 0,TS_Object);  balls->addChild(s);
        s = new SLNode(new SLSphere(0.5f,32,32,"S6",mM)); s->translate(-0.5,0, 1,TS_Object);  balls->addChild(s);
        s = new SLNode(new SLSphere(0.5f,32,32,"S7",mW)); s->translate(-1.0,0, 2,TS_Object);  balls->addChild(s);

        SLNode* scene  = new SLNode;
        scene->addChild(light1);
        scene->addChild(balls);
        scene->addChild(rect);
        scene->addChild(cam1);

        _background.colors(SLCol4f(0.1f,0.4f,0.8f));
        sv->camera(cam1);
        _root3D = scene;
    }
    else
    if (sceneName == cmdSceneRTLens) //.........................................
	{
        name("Ray tracing: Lens test");
        info(sv,"Ray tracing lens test scene.");

        // Create textures and materials
        SLGLTexture* texC = new SLGLTexture("VisionExample.png");
        //SLGLTexture* texC = new SLGLTexture("Checkerboard0512_C.png");
        
        SLMaterial* mT = new SLMaterial("mT", texC, 0, 0, 0); mT->kr(0.5f);

        // Glass material
        // name, ambient, specular,	shininess, kr(reflectivity), kt(transparency), kn(refraction)
        SLMaterial* matLens = new SLMaterial("lens", SLCol4f(0.0f, 0.0f, 0.0f), SLCol4f(0.5f, 0.5f, 0.5f), 100, 0.5f, 0.5f, 1.5f);
        //SLGLShaderProg* sp1 = new SLGLShaderProgGeneric("RefractReflect.vert", "RefractReflect.frag");
        //matLens->shaderProg(sp1);

        #ifndef SL_GLES2
            SLint numSamples = 10;
        #else
            SLint numSamples = 6;
        #endif

        // Scene
        SLCamera* cam1 = new SLCamera;
        cam1->translation(0, 8, 0);
        cam1->lookAt(0, 0, 0);
        cam1->focalDist(6);
        cam1->lensDiameter(0.4f);
        cam1->lensSamples()->samples(numSamples, numSamples);
        cam1->setInitialState();

        // Light
        //SLLightSphere* light1 = new SLLightSphere(15, 20, 15, 0.1f);
        //light1->attenuation(0, 0, 1);

        // Plane
        //SLNode* rect = new SLNode(new SLRectangle(SLVec2f(-20, -20), SLVec2f(20, 20), 50, 20, "Rect", mT));
        //rect->translate(0, 0, 0, TS_Object);
        //rect->rotate(90, -1, 0, 0);

        SLLightSphere* light1 = new SLLightSphere(1, 6, 1, 0.1f);
        light1->attenuation(0, 0, 1);
        

        SLNode* rect = new SLNode(new SLRectangle(SLVec2f(-5, -5), SLVec2f(5, 5), 20, 20, "Rect", mT));
        rect->rotate(90, -1, 0, 0);
        rect->translate(0, 0, -0.0f, TS_Object);

        // Lens from eye prescription card   
        //SLNode* lensA = new SLNode(new SLLens(0.50f, -0.50f, 4.0f, 0.0f, 32, 32, "presbyopic", matLens));   // Weitsichtig
        //SLNode* lensB = new SLNode(new SLLens(-0.65f, -0.10f, 4.0f, 0.0f, 32, 32, "myopic", matLens));      // Kurzsichtig
        //lensA->translate(-2, 1, -2, TS_Object);
        //lensB->translate(2, 1, -2, TS_Object);

        // Lens with radius
        //SLNode* lensC = new SLNode(new SLLens(5.0, 4.0, 4.0f, 0.0f, 32, 32, "presbyopic", matLens));        // Weitsichtig
        SLNode* lensD = new SLNode(new SLLens(-15.0f, -15.0f, 1.0f, 0.1f, 32, 32, "myopic", matLens));          // Kurzsichtig
        //lensC->translate(-2, 1, 2, TS_Object);
        lensD->translate(0, 6, 0, TS_Object);

        // Node
        SLNode* scene = new SLNode;
        //scene->addChild(lensA);
        //scene->addChild(lensB);
        //scene->addChild(lensC);
        scene->addChild(lensD);
        scene->addChild(rect);
        scene->addChild(light1);
        scene->addChild(cam1);

        _background.colors(SLCol4f(0.1f, 0.4f, 0.8f));
        sv->camera(cam1);
        _root3D = scene;
    }
    else
    if (sceneName == cmdSceneRTTest) //.........................................
    {
        // Set scene name and info string
        name("RT Test Scene");
        info(sv, "RT Test Scene");

        // Create a camera node
        SLCamera* cam1 = new SLCamera();
        cam1->name("camera node");
        cam1->translation(0, 0, 5);
        cam1->lookAt(0, 0, 0);
        cam1->setInitialState();

        // Create a light source node
        SLLightSphere* light1 = new SLLightSphere(0.3f);
        light1->translation(5, 5, 5);
        light1->lookAt(0, 0, 0);
        light1->name("light node");

        // Material for glass sphere
        SLMaterial* matBox1 = new SLMaterial("matBox1", SLCol4f(0.0f, 0.0f, 0.0f), SLCol4f(0.5f, 0.5f, 0.5f), 100, 0.0f, 0.9f, 1.5f);
        SLMesh* boxMesh1 = new SLBox(-0.8f, -1, 0.02f, 1.2f, 1, 1, "boxMesh1", matBox1);
        SLNode* boxNode1 = new SLNode(boxMesh1, "BoxNode1");
        
        SLMaterial* matBox2 = new SLMaterial("matBox2", SLCol4f(0.0f, 0.0f, 0.0f), SLCol4f(0.5f, 0.5f, 0.5f), 100, 0.0f, 0.9f, 1.3f);
        SLMesh* boxMesh2 = new SLBox(-1.2f, -1, -1, 0.8f, 1,-0.02f, "BoxMesh2", matBox2);
        SLNode* boxNode2 = new SLNode(boxMesh2, "BoxNode2");

        // Create a scene group and add all nodes
        SLNode* scene = new SLNode("scene node");
        scene->addChild(light1);
        scene->addChild(cam1);
        scene->addChild(boxNode1);
        scene->addChild(boxNode2);

        // Set background color and the root scene node
        _background.colors(SLCol4f(0.5f,0.5f,0.5f));
        _root3D = scene;

        // Set active camera
        sv->camera(cam1);
    }

    // call onInitialize on all scene views
    for (auto sv : _sceneViews)
    {   if (sv != nullptr)
        {   sv->onInitialize();
            sv->showLoading(false);
        }
    }

}
//-----------------------------------------------------------------------------
