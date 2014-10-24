//#############################################################################
//  File:      SLScene_onLoad.cpp
//  Author:    Marcus Hudritsch
//  Date:      July 2014
//  Codestyle: https://code.google.com/p/slproject/wiki/CodingStyleGuidelines
//  Copyright: Marcus Hudritsch, Kirchrain 18, 2572 Sutz
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
#include <SLAssImp.h>

#include <SLCamera.h>
#include <SLLightSphere.h>
#include <SLLightRect.h>
#include <SLMesh.h>
#include <SLPolygon.h>
#include <SLBox.h>
#include <SLCone.h>
#include <SLCylinder.h>
#include <SLSphere.h>
#include <SLRectangle.h>
#include <SLGrid.h>

SLNode* SphereGroup(SLint, SLfloat, SLfloat, SLfloat, SLfloat, SLint, SLMaterial*, SLMaterial*);
//-----------------------------------------------------------------------------
//! Creates a recursive sphere group used for the ray tracing scenes
SLNode* SphereGroup(SLint depth,                      // depth of recursion 
                    SLfloat x, SLfloat y, SLfloat z,  // position of group
                    SLfloat scale,                    // scale factor
                    SLint  resolution,       // resolution of spheres
                    SLMaterial* matGlass,    // material for center sphere
                    SLMaterial* matRed)      // material for orbiting spheres
{  
    if (depth==0)
    {   SLNode* s = new SLNode(new SLSphere(0.5f*scale,resolution,resolution,"RedSphere", matRed)); 
        s->translate(x,y,z, TS_Local);
        return s;
    } else
    {   depth--;
        SLNode* sGroup = new SLNode;
        sGroup->translate(x,y,z, TS_Local);
        SLint newRes = max(resolution-8,8);
        sGroup->addChild(new SLNode(new SLSphere(0.5f*scale,resolution,resolution,"RedSphere", matGlass)));
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
SLNode* BuildFigureGroup(SLMaterial* material);
SLNode* BuildFigureGroup(SLMaterial* material)
{
    SLNode* cyl;
   
    // Feet
    SLNode* feet = new SLNode("feet group");
    feet->addMesh(new SLSphere(0.2f, 16, 16, "ankle"));
    SLNode* feetbox = new SLNode(new SLBox(-0.2f,-0.1f, 0.0f, 0.2f, 0.1f, 0.8f, "foot"));
    feetbox->translate(0.0f,-0.25f,-0.15f, TS_Local);
    feet->addChild(feetbox);
    feet->translate(0.0f,0.0f,1.6f, TS_Local);
    feet->rotate(-90.0f, 1.0f, 0.0f, 0.0f);
   
    // Assemble low leg
    SLNode* leglow = new SLNode("leglow group");
    leglow->addMesh(new SLSphere(0.3f, 16, 16, "knee")); 
    cyl = new SLNode(new SLCylinder(0.2f, 1.4f, 1, 16, false, false, "shin"));            
    cyl->translate(0.0f, 0.0f, 0.2f, TS_Local);            
    leglow->addChild(cyl);
    leglow->addChild(feet);
   
    // Assemble leg
    SLNode* leg = new SLNode("leg group");
    leg->addMesh(new SLSphere(0.4f, 16, 16, "hip joint"));     
    cyl = new SLNode(new SLCylinder(0.3f, 1.0f, 1, 16, false, false, "thigh"));            
    cyl->translate(0.0f, 0.0f, 0.27f, TS_Local);           
    leg->addChild(cyl);
    leglow->translate(0.0f, 0.0f, 1.27f, TS_Local);        
    leglow->rotate(5, 1.0f, 0.0f, 0.0f);         
    leg->addChild(leglow);

    // Assemble left & right leg
    SLNode* legLeft = new SLNode("left leg group");
    legLeft->translate(-0.4f, 0.0f, 2.2f, TS_Local);
    legLeft->rotate(-45, 1,0,0);
    legLeft->addChild(leg);               
    SLNode* legRight= new SLNode("right leg group");           
    legRight->translate(0.4f, 0.0f, 2.2f, TS_Local);       
    legRight->rotate(70, -1,0,0);
    legRight->addChild(leg->copyRec());  

    // Assemble low arm
    SLNode* armlow = new SLNode("armLow group");
    armlow->addMesh(new SLSphere(0.2f, 16, 16, "ellbow"));    
    cyl = new SLNode(new SLCylinder(0.15f, 1.0f, 1, 16, true, false, "arm"));           
    cyl->translate(0.0f, 0.0f, 0.14f, TS_Local);           
    armlow->addChild(cyl);

    // Assemble arm
    SLNode* arm = new SLNode("arm group");
    arm->addMesh(new SLSphere(0.3f, 16, 16, "shoulder"));                           
    cyl = new SLNode(new SLCylinder(0.2f, 1.0f, 1, 16, false, false, "upper arm"));            
    cyl->translate(0.0f, 0.0f, 0.2f, TS_Local);            
    arm->addChild(cyl);
    armlow->translate(0.0f, 0.0f, 1.2f, TS_Local);         
    armlow->rotate(45, -1.0f, 0.0f, 0.0f);       
    arm->addChild(armlow);

    // Assemble left & right arm
    SLNode* armLeft = new SLNode("left arm group");
    armLeft->translate(-1.1f, 0.0f, 0.3f, TS_Local);       
    armLeft->rotate(10, -1,0,0);
    armLeft->addChild(arm);
    SLNode* armRight= new SLNode("right arm group");
    armRight->translate(1.1f, 0.0f, 0.3f, TS_Local);       
    armRight->rotate(-60, -1,0,0);
    armRight->addChild(arm->copyRec());

    // Assemble head & neck
    SLNode* head = new SLNode(new SLSphere(0.5f, 16, 16, "Head"));
    head->translate(0.0f, 0.0f,-0.7f, TS_Local);
    SLNode* neck = new SLNode(new SLCylinder(0.25f, 0.3f, 1, 16, false, false, "neck"));
    neck->translate(0.0f, 0.0f,-0.3f, TS_Local);
      
    // Assemble figure Left
    SLNode* figure = new SLNode("figure");
    figure->addChild(new SLNode(new SLBox(-0.8f,-0.4f, 0.0f, 0.8f, 0.4f, 2.0f, "Box")));
    figure->addChild(head);
    figure->addChild(neck);
    figure->addChild(armLeft);
    figure->addChild(armRight);
    figure->addChild(legLeft);
    figure->addChild(legRight);
    figure->rotate(90, 1,0,0);

    // Add animations for left leg
    legLeft = figure->findChild<SLNode>("left leg group");
    legLeft->animation(new SLAnimation(2, 60, SLVec3f(1,0,0), pingPongLoop));
    SLNode* legLowLeft = legLeft->findChild<SLNode>("leglow group");
    legLowLeft->animation(new SLAnimation(2, 40, SLVec3f(1,0,0), pingPongLoop));
    SLNode* feetLeft = legLeft->findChild<SLNode>("feet group");
    feetLeft->animation(new SLAnimation(2, 40, SLVec3f(1,0,0), pingPongLoop));
    legRight = figure->findChild<SLNode>("right leg group");
    legRight->rotate(70, 1,0,0);

    return figure;
}
//-----------------------------------------------------------------------------
//! SLScene::onLoad(int sceneName) builds a scene from source code.
/*! SLScene::onLoad builds a scene from source code.
The paramter sceneName is the scene to choose and corresponds to enumeration 
SLCommand value for the different scenes. The first scene is cmdSceneFigure.
*/
void SLScene::onLoad(SLSceneView* sv, SLCmd sceneName)
{  
    // Initialize all preloaded stuff from SLScene
    cout << "------------------------------------------------------------------" << endl;
    init();

    // @todo in the onload function we only notify the active scene view about the completion
    ///       of the loading process. But all scene views might want to update their editor
    ///       cameras after a new scene has been loaded.
    // Note: we currently reset the cameras of ALL scene views to their respective editor
    //       cameras. 


    // Show once the empty loading screen without scene
    // @todo review this, we still pass in the active scene view with sv. Is it necessary?
    for (SLint i = 0; i < _sceneViews.size(); ++i)
    {
        if (_sceneViews[i] != NULL)
        {   _sceneViews[i]->showLoading(true);
        }
    }

    // @todo we need to repaint all the scene views so that the loading screen is visible!
    //sv->onInitialize();
    //sv->onWndUpdate();

    _currentID = sceneName;

    if (sceneName == cmdSceneSmallTest)
    {
        name("Minimal Texture Example");
        info(sv, "Minimal texture mapping example with one light source.");

        // Create textures
        SLGLTexture* texC = new SLGLTexture("earth1024_C.jpg");

        // Create materials
        SLMaterial* m1 = new SLMaterial("m1", texC);

        // Create a camera at 0,0,20
        SLCamera* cam1 = new SLCamera();
        cam1->name("cam1");
        cam1->position(0,0,20);
        cam1->lookAt(0, 0, 0);
        cam1->setInitialState();

        // Create a spherical light source at 0,0,5
        SLLightSphere* light1 = new SLLightSphere(0.3f);
        light1->name("light1");
        light1->position(0,0,5);
        light1->animation(new SLAnimation(2, 4, XAxis, 4, YAxis, loop));

        // Create ground grid
        SLMaterial* m2 = new SLMaterial(SLCol4f::WHITE);
        SLGrid* grid = new SLGrid(SLVec3f(-5,0,-5), SLVec3f(5,0,5), 10, 10, "Grid", m2);

        // Create a scene group and add all nodes
        SLNode* scene = new SLNode("Scene");
        scene->addChild(new SLNode(new SLRectangle(SLVec2f(-5,-5), SLVec2f(5,5),1,1,"Rect", m1), "Rect"));
        scene->addChild(light1);
        scene->addChild(new SLNode(grid, "grid"));
        scene->addChild(cam1);

        // Set background color, the active camera and the root scene node
        _backColor.set(0.5f,0.5f,0.5f);
        sv->camera(cam1);
        _root3D = scene;
    }
    else
    if (sceneName == cmdSceneFigure)
    {
        name("Hierarchical Figure Scene 2");
        info(sv, "Hierarchical scene structure and animation test. Turn on the bounding boxes to see the animation curves");

        // Create textures and materials
        SLGLTexture* tex1 = new SLGLTexture("Checkerboard0512_C.png");
        SLMaterial* m1 = new SLMaterial("m1", tex1); m1->kr(0.5f);
        SLMaterial* m2 = new SLMaterial("m2", SLCol4f::WHITE*0.5, SLCol4f::WHITE,128, 0.5f, 0.0f, 1.0f);

        SLMesh* floorMesh = new SLRectangle(SLVec2f(-5,-5), SLVec2f(5,5), 20, 20, "FloorMesh", m1);
        SLNode* floorRect = new SLNode(floorMesh);
        floorRect->rotate(90, -1,0,0);
        floorRect->translate(0,0,-5.5f, TS_Local);
      
        // Bouncing balls
        SLNode* ball1 = new SLNode(new SLSphere(0.3f, 16, 16, "Ball1", m2));
        ball1->translate(0,0,4, TS_Local);
        ball1->animation(new SLAnimation(1, SLVec3f(0,-5.2f,0), pingPongLoop, linear));
        SLNode* ball2 = new SLNode(new SLSphere(0.3f, 16, 16, "Ball2", m2));
        ball2->translate(-1.5f,0,4, TS_Local);
        ball2->animation(new SLAnimation(1, SLVec3f(0,-5.2f,0), pingPongLoop, inQuad));
        SLNode* ball3 = new SLNode(new SLSphere(0.3f, 16, 16, "Ball3", m2));
        ball3->translate(-2.5f,0,4, TS_Local);
        ball3->animation(new SLAnimation(1, SLVec3f(0,-5.2f,0), pingPongLoop, outQuad));
        SLNode* ball4 = new SLNode(new SLSphere(0.3f, 16, 16, "Ball4", m2));
        ball4->translate( 1.5f,0,4, TS_Local);
        ball4->animation(new SLAnimation(1, SLVec3f(0,-5.2f,0), pingPongLoop, inOutQuad));
        SLNode* ball5 = new SLNode(new SLSphere(0.3f, 16, 16, "Ball5", m2));
        ball5->translate( 2.5f,0,4, TS_Local);
        ball5->animation(new SLAnimation(1, SLVec3f(0,-5.2f,0), pingPongLoop, outInQuad));

        SLCamera* cam1 = new SLCamera();
        cam1->position(0, 0, 22);
        cam1->lookAt(0, 0, 0);
        cam1->focalDist(22);
        cam1->setInitialState();
        SLCamera* cam2 = new SLCamera;
        cam2->position(5, 0, 0);
        cam2->lookAt(0, 0, 0);
        cam2->focalDist(5);
        cam2->setInitialState();

        SLLightSphere* light1 = new SLLightSphere(0, 2, 0, 0.5f);
        light1->ambient (SLCol4f(0.2f,0.2f,0.2f));
        light1->diffuse (SLCol4f(0.9f,0.9f,0.9f));
        light1->specular(SLCol4f(0.9f,0.9f,0.9f));
        light1->attenuation(1,0,0);
        light1->animation(new SLAnimation(4, 6, ZAxis, 6, XAxis, loop));

        SLLightSphere* light2 = new SLLightSphere(0, 0, 0, 0.2f);
        light2->ambient (SLCol4f(0.2f,0.0f,0.0f));
        light2->diffuse (SLCol4f(0.9f,0.0f,0.0f));
        light2->specular(SLCol4f(0.9f,0.9f,0.9f));
        light2->attenuation(1,0,0);
        SLVKeyframe light2Curve;
        light2Curve.push_back(SLKeyframe(0, SLVec3f(-8,-4, 0)));
        light2Curve.push_back(SLKeyframe(1, SLVec3f( 0, 4, 0)));
        light2Curve.push_back(SLKeyframe(1, SLVec3f( 8,-4, 0)));
        light2->animation(new SLAnimation(light2Curve, 0, pingPongLoop));
     
        SLNode* figure = BuildFigureGroup(m2);

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
        _backColor.set(SLCol4f(0.1f,0.4f,0.8f));
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
        //matBlu->shaderProg(_shaderProgs[PerPixBlinn]);
        //matRed->shaderProg(_shaderProgs[PerPixBlinn]);
        //matGre->shaderProg(_shaderProgs[PerPixBlinn]);
        //matGra->shaderProg(_shaderProgs[PerPixBlinn]);

        SLCamera* cam1 = new SLCamera();
        cam1->name("cam1");
        cam1->clipNear(.1f);
        cam1->clipFar(30);
        cam1->position(0,0,10);
        cam1->lookAt(0, 0, 0);
        cam1->speedLimit(40);
        cam1->focalDist(10);
        cam1->eyeSeparation(cam1->focalDist()/30.0f);
        cam1->setInitialState();

        SLLightSphere* light1 = new SLLightSphere(2.5f, 2.5f, 2.5f, 0.2f);
        light1->ambient(SLCol4f(0.1f, 0.1f, 0.1f));
        light1->diffuse(SLCol4f(1.0f, 1.0f, 1.0f));
        light1->specular(SLCol4f(1.0f, 1.0f, 1.0f));
        light1->attenuation(1,0,0);
        light1->animation(new SLAnimation(2,SLVec3f(0,0,-5), pingPongLoop, inOutCubic));
        //light1->samples(8,8);

        SLLightSphere* light2 = new SLLightSphere(-2.5f, -2.5f, 2.5f, 0.2f);
        light2->ambient(SLCol4f(0.1f, 0.1f, 0.1f));
        light2->diffuse(SLCol4f(1.0f, 1.0f, 1.0f));
        light2->specular(SLCol4f(1.0f, 1.0f, 1.0f));
        light2->attenuation(1,0,0);
        light2->animation(new SLAnimation(2,SLVec3f(0,5,0), pingPongLoop, inOutCubic));


        #if defined(SL_OS_IOS) || defined(SL_OS_ANDROID)
        SLNode* mesh3DS = SLAssImp::load("jackolan.3DS");
        SLNode* meshDAE = SLAssImp::load("AstroBoy.dae");
        SLNode* meshFBX = SLAssImp::load("Duck.fbx");
      
        #else
        SLNode* mesh3DS = SLAssImp::load("3DS/Halloween/Jackolan.3DS");
        SLNode* meshDAE = SLAssImp::load("DAE/AstroBoy/AstroBoy.dae");
        SLNode* meshFBX = SLAssImp::load("FBX/Duck/Duck.fbx");
        #endif

        // Scale to so that the AstroBoy is about 2 (meters) high.
        if (mesh3DS) {mesh3DS->scale(0.1f);  mesh3DS->translate(-22.0f, 1.9f, 3.5f, TS_Local);}
        if (meshDAE) {meshDAE->scale(30.0f); meshDAE->translate(0,-10,0, TS_Local);}
        if (meshFBX) {meshFBX->scale(0.1f);  meshFBX->scale(0.1f); meshFBX->translate(200, 30, -30, TS_Local); meshFBX->rotate(-90,0,1,0);}// define rectangles for the surrounding box

        SLfloat b=3; // edge size of rectangles
        SLNode *rb, *rl, *rr, *rf, *rt;
        SLuint res = 20;
        rb = new SLNode(new SLRectangle(SLVec2f(-b,-b), SLVec2f(b,b), res, res, "rectB", matBlu), "rectBNode");                         rb->translate(0,0,-b, TS_Local);
        rl = new SLNode(new SLRectangle(SLVec2f(-b,-b), SLVec2f(b,b), res, res, "rectL", matRed), "rectLNode"); rl->rotate( 90, 0,1,0); rl->translate(0,0,-b, TS_Local);
        rr = new SLNode(new SLRectangle(SLVec2f(-b,-b), SLVec2f(b,b), res, res, "rectR", matGre), "rectRNode"); rr->rotate(-90, 0,1,0); rr->translate(0,0,-b, TS_Local);
        rf = new SLNode(new SLRectangle(SLVec2f(-b,-b), SLVec2f(b,b), res, res, "rectF", matGra), "rectFNode"); rf->rotate(-90, 1,0,0); rf->translate(0,0,-b, TS_Local);
        rt = new SLNode(new SLRectangle(SLVec2f(-b,-b), SLVec2f(b,b), res, res, "rectT", matGra), "rectTNode"); rt->rotate( 90, 1,0,0); rt->translate(0,0,-b, TS_Local);

        SLNode* scene = new SLNode("Scene");
        scene->addChild(light1);
        scene->addChild(light2);
        scene->addChild(rb);
        scene->addChild(rl);
        scene->addChild(rr);
        scene->addChild(rf);
        scene->addChild(rt);
        if (mesh3DS) scene->addChild(mesh3DS);
        if (meshDAE) scene->addChild(meshDAE);
        if (meshFBX) scene->addChild(meshFBX);
        scene->addChild(cam1);

        _backColor.set(0.5f,0.5f,0.5f);
        //sv->camera(cam1);
        _root3D = scene;
    }
    else
    if (sceneName == cmdSceneLargeModel) //.....................................
    {
        name("Large Model Test");
        info(sv, "Large Model with 7.2 mio. triangles.");

        SLCamera* cam1 = new SLCamera();
        cam1->name("cam1");
        cam1->position(10,0,220);
        cam1->lookAt(10,0,0);
        cam1->clipNear(0.1f);
        cam1->clipFar(500.0f);
        cam1->setInitialState();

        SLLightSphere* light1 = new SLLightSphere(120,120,120, 1);
        light1->ambient(SLCol4f(1,1,1));
        light1->diffuse(SLCol4f(1,1,1));
        light1->specular(SLCol4f(1,1,1));
        light1->attenuation(1,0,0);

        SLNode* largeModel = SLAssImp::load("PLY/xyzrgb_dragon.ply");

        SLNode* scene = new SLNode("Scene");
        scene->addChild(light1);
        if (largeModel) scene->addChild(largeModel);
        scene->addChild(cam1);

        _backColor.set(0.5f,0.5f,0.5f);
        sv->camera(cam1);
        _root3D = scene;
    }
    else
    if (sceneName == cmdSceneTextureBlend) //...................................
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
        m1->shaderProg(_shaderProgs[TextureOnly]);
        m1->textures().push_back(t1);
        m2->textures().push_back(t2);

        SLCamera* cam1 = new SLCamera();
        cam1->name("cam1");
        cam1->position(0,3,25);
        cam1->lookAt(0,0,10);
        cam1->focalDist(25);
        cam1->setInitialState();

        SLLightSphere* light = new SLLightSphere(0.1f);
        light->position(5,5,5);
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
                                float(iZ)*2+SL_random(0.7f,1.4f), TS_Local);
                    t->rotate(SL_random(0, 90), 0,1,0);
                    t->scale(SL_random(0.5f,1.0f));
                    scene->addChild(t);
                }
            }
        }

        scene->addChild(cam1);

        _backColor.set(0.6f,0.6f,1);
        sv->camera(cam1);
        _root3D = scene;
    }
    else
    if (sceneName == cmdSceneRevolver) //.......................................
    {
        name("Revolving Mesh Test w. glass shader");
        info(sv, "Examples of revolving mesh objects constructed by rotating a 2D curve. The glass shader reflects and refracts the environment map. Try ray tracing.");

        // Testmap material
        SLGLTexture* tex1 = new SLGLTexture("Testmap_0512_C.png");
        SLMaterial* mat1 = new SLMaterial("mat1", tex1);

        // floor material
        SLGLTexture* tex2 = new SLGLTexture("wood0_0512_C.jpg");
        SLMaterial* mat2 = new SLMaterial("mat2", tex2);
        mat2->specular(SLCol4f::BLACK);

        // Back wall material
        SLGLTexture* tex3 = new SLGLTexture("bricks1_0512_C.jpg");
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
        SLMaterial* mat5 = new SLMaterial("glass", SLCol4f::BLACK, SLCol4f::WHITE,
                                        100, 0.2f, 0.8f, 1.5f);
        mat5->textures().push_back(tex5);
        SLGLShaderProg* sp1 = new SLGLShaderProgGeneric("RefractReflect.vert",
                                                        "RefractReflect.frag");
        mat5->shaderProg(sp1);

        // camera
        SLCamera* cam1 = new SLCamera();
        cam1->name("cam1");
        cam1->position(0,0,17);
        cam1->lookAt(0, 0, 0);
        cam1->focalDist(17);
        cam1->setInitialState();

        // light
        SLLightSphere* light1 = new SLLightSphere(0, 4, 0, 0.3f);
        light1->diffuse(SLCol4f(1, 1, 1));
        light1->ambient(SLCol4f(0.2f, 0.2f, 0.2f));
        light1->specular(SLCol4f(1, 1, 1));
        light1->attenuation(1,0,0);
        light1->animation(new SLAnimation(4, 6, ZAxis, 6, XAxis, loop));

        // wine glass
        SLVVec3f revP;
        revP.push_back(SLVec3f(0.00f, 0.00f));
        revP.push_back(SLVec3f(2.00f, 0.00f));
        revP.push_back(SLVec3f(2.00f, 0.00f));
        revP.push_back(SLVec3f(2.00f, 0.10f));
        revP.push_back(SLVec3f(1.95f, 0.15f));

        revP.push_back(SLVec3f(0.40f, 0.50f));
        revP.push_back(SLVec3f(0.25f, 0.60f));
        revP.push_back(SLVec3f(0.20f, 0.70f));
        revP.push_back(SLVec3f(0.30f, 3.00f));

        revP.push_back(SLVec3f(0.30f, 3.00f));
        revP.push_back(SLVec3f(0.20f, 3.10f));
        revP.push_back(SLVec3f(0.20f, 3.10f));

        revP.push_back(SLVec3f(1.20f, 3.90f));
        revP.push_back(SLVec3f(1.60f, 4.30f));
        revP.push_back(SLVec3f(1.95f, 4.80f));
        revP.push_back(SLVec3f(2.15f, 5.40f));
        revP.push_back(SLVec3f(2.20f, 6.20f));
        revP.push_back(SLVec3f(2.10f, 7.10f));
        revP.push_back(SLVec3f(2.05f, 7.15f));

        revP.push_back(SLVec3f(2.00f, 7.10f));
        revP.push_back(SLVec3f(2.05f, 6.00f));
        revP.push_back(SLVec3f(1.95f, 5.40f));
        revP.push_back(SLVec3f(1.70f, 4.80f));
        revP.push_back(SLVec3f(1.30f, 4.30f));
        revP.push_back(SLVec3f(0.80f, 4.00f));
        revP.push_back(SLVec3f(0.20f, 3.80f));
        revP.push_back(SLVec3f(0.00f, 3.82f));
        SLNode* glass = new SLNode(new SLRevolver(revP, SLVec3f(0,1,0), 36, true, true, "Revolver", mat5));
        glass->translate(0.0f,-3.5f, 0.0f, TS_Local);
      
        SLNode* sphere = new SLNode(new SLSphere(1,16,16, "mySphere", mat1));
        sphere->translate(3,0,0, TS_Local);

        SLNode* cylinder = new SLNode(new SLCylinder(1, 2, 3, 16, true, true, "myCylinder", mat1));
        cylinder->translate(-3,0,-1, TS_Local);

        SLNode* cone = new SLNode(new SLCone(1, 3, 3, 16, true, "myCone", mat1));
        cone->rotate(90, -1,0,0);
        cone->translate(0,0,2.5f, TS_Local);

        // Cube dimensions
        SLfloat pL = -9.0f, pR = 9.0f; // left/right
        SLfloat pB = -3.5f, pT =14.5f; // bottom/top
        SLfloat pN =  9.0f, pF =-9.0f; // near/far

        //// bottom rectangle
        SLNode* b = new SLNode(new SLRectangle(SLVec2f(pL,-pN), SLVec2f(pR,-pF), 10, 10, "PolygonFloor", mat2));
        b->rotate(90, -1,0,0); b->translate(0,0,pB, TS_Local);

        // top rectangle
        SLNode* t = new SLNode(new SLRectangle(SLVec2f(pL,pF), SLVec2f(pR,pN), 10, 10, "top", mat2));
        t->rotate(90, 1,0,0); t->translate(0,0,-pT, TS_Local);

        // far rectangle
        SLNode* f = new SLNode(new SLRectangle(SLVec2f(pL,pB), SLVec2f(pR,pT), 10, 10, "far", mat3));
        f->translate(0,0,pF, TS_Local);

        // left rectangle
        SLNode* l = new SLNode(new SLRectangle(SLVec2f(-pN,pB), SLVec2f(-pF,pT), 10, 10, "left", mat4));
        l->rotate(90, 0,1,0); l->translate(0,0,pL, TS_Local);

        // right rectangle
        SLNode* r = new SLNode(new SLRectangle(SLVec2f(pF,pB), SLVec2f(pN,pT), 10, 10, "right", mat4));
        r->rotate(90, 0,-1,0); r->translate(0,0,-pR, TS_Local);

        SLNode* scene = new SLNode;
        scene->addChild(light1);
        scene->addChild(glass);
        scene->addChild(sphere);
        scene->addChild(cylinder);
        scene->addChild(cone);
        scene->addChild(b);
        scene->addChild(f);
        scene->addChild(t);
        scene->addChild(l);
        scene->addChild(r);
        scene->addChild(cam1);

        _backColor.set(0.5f,0.5f,0.5f);
        sv->camera(cam1);
        _root3D = scene;
    }
    else
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
        cam1->position(0,0,5);
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
                        s->translate(float(iX), float(iY), float(iZ), TS_Local);
                        scene->addChild(s);
                    }
                }
            }
        }

        SLint num = size + size + 1;
        SL_LOG("Triangles in scene: %d\n", resolution*resolution*2*num*num*num);

        _backColor.set(0.1f,0.1f,0.1f);
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
        cam1->position(0,100,180);
        cam1->lookAt(0, 0, 0);
        cam1->setInitialState();

        // Floor rectangle
        SLNode* rect = new SLNode(new SLRectangle(SLVec2f(-100,-100), 
                                                  SLVec2f( 100, 100), 
                                                  SLVec2f(   0,   0), 
                                                  SLVec2f(  50,  50), 50, 50, "Floor", m1));
        rect->rotate(90, -1,0,0);
        rect->translate(0,0,-5.5f, TS_Local);

        SLNode* figure = BuildFigureGroup(m2);

        // Add animation for light 1
        light1->animation(new SLAnimation(4, 12, ZAxis, 12, XAxis, loop));

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
                    f->translate(float(iX)*5, float(iZ)*5, 0, TS_Local);
                    scene->addChild(f);
                }
            }
        }

        // Set backround color, active camera & the root pointer
        _backColor.set(SLCol4f(0.1f,0.4f,0.8f));
        sv->camera(cam1);
        _root3D = scene;
    }
    else
    if (sceneName == cmdSceneTextureFilter) //..................................
    {
        name("Texturing: Filter Compare");
        info(sv, "Texture filter comparison: Bottom: nearest neighbour, left: linear, top: linear mipmap, right: anisotropic");

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
        SLMaterial* matB = new SLMaterial("matB", texB,0,0,0, _shaderProgs[TextureOnly]);
        SLMaterial* matL = new SLMaterial("matL", texL,0,0,0, _shaderProgs[TextureOnly]);
        SLMaterial* matT = new SLMaterial("matT", texT,0,0,0, _shaderProgs[TextureOnly]);
        SLMaterial* matR = new SLMaterial("matR", texR,0,0,0, _shaderProgs[TextureOnly]);

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

        SLNode* sphere = new SLNode(new SLSphere(0.2f,16,16,"Sphere", matR));
        sphere->rotate(90, 1,0,0);

        SLCamera* cam1 = new SLCamera;
        cam1->position(0,0,2.2f);
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

        _backColor.set(SLCol4f(0.2f,0.2f,0.2f));
        sv->camera(cam1);
        _root3D = scene;
    }
    else
    if (sceneName == cmdSceneMassAnimation) //..........................................
    {   
        name("Mass Animation");
        info(sv, "Performance test for transform updates from many animations.");

        init();
   
        SLLightSphere* light1 = new SLLightSphere(7,7,0, 0.1f, 5, 10);
        light1->attenuation(0,0,1);
        light1->translate(-3, 5, 2, TS_Local);
   
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

        for(SLint lvl = 0; lvl < levels; ++lvl) 
        {   curParentsVector = parents;
            parents.clear();
            // for each parent in the previous level, add a completely new grid
            for(SLint p=0; p < curParentsVector.size(); ++p)
            {   for(SLint i = 0; i < nodesPerLvl; ++i) 
                {   SLNode* node = new SLNode("MassAnimNode");
                    node->addMesh(box);
                    curParentsVector[p]->addChild(node);
                    parents.push_back(node);

                    // position
                    SLfloat x = (SLfloat)(i % gridSize - gridHalf);
                    SLfloat z = (SLfloat)((i > 0) ? i / gridSize - gridHalf : -gridHalf);
                    SLVec3f pos(x*nodeSpacing[lvl] *1.1f, 1.5f, z*nodeSpacing[lvl]*1.1f);

                    node->translate(pos, TS_Local);
                    //node->scale(1.1f);

                    if (lvl != 0) 
                    {   SLfloat duration = 1.0f + 5.0f * ((SLfloat)i/(SLfloat)nodesPerLvl);
                    SLAnimation* anim = new SLAnimation(duration, SLVec3f(0, 1.0f, 0), pingPongLoop, inOutSine, "randomAnim");
                    node->animation(anim);
                }   
            }
        }
        }
    }
    else
    if (sceneName == cmdScenePerVertexBlinn) //.................................
    {
        name("Blinn-Phong per vertex lighting");
        info(sv, "Per-vertex lighting with Blinn-Phong lightmodel. The reflection of 4 light sources is calculated per vertex and is then interpolated over the triangles.");

        // create material
        SLMaterial* m1 = new SLMaterial("m1", 0,0,0,0, _shaderProgs[PerVrtBlinn]);
        m1->shininess(500);

        SLCamera* cam1 = new SLCamera;
        cam1->position(0,1,8);
        cam1->lookAt(0,1,0);
        cam1->focalDist(8);
        cam1->setInitialState();

        // define 4 light sources
        SLLightRect* light0 = new SLLightRect(2.0f,1.0f);
        light0->ambient(SLCol4f(0,0,0));
        light0->diffuse(SLCol4f(1,1,1));
        light0->position(0,3,0);
        light0->lookAt(0,0,0, 0,0,-1);
        light0->attenuation(0,0,1);

        SLLightSphere* light1 = new SLLightSphere(0.1f);
        light1->ambient(SLCol4f(0,0,0));
        light1->diffuse(SLCol4f(1,0,0));
        light1->specular(SLCol4f(1,0,0));
        light1->position(0, 0, 2);
        light1->lookAt(0, 0, 0);
        light1->attenuation(0,0,1);

        SLLightSphere* light2 = new SLLightSphere(0.1f);
        light2->ambient(SLCol4f(0,0,0));
        light2->diffuse(SLCol4f(0,1,0));
        light2->specular(SLCol4f(0,1,0));
        light2->position(1.5, 1.5, 1.5);
        light2->lookAt(0, 0, 0);
        light2->spotCutoff(20);
        light2->attenuation(0,0,1);

        SLLightSphere* light3 = new SLLightSphere(0.1f);
        light3->ambient(SLCol4f(0,0,0));
        light3->diffuse(SLCol4f(0,0,1));
        light3->specular(SLCol4f(0,0,1));
        light3->position(-1.5, 1.5, 1.5);
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

        _backColor.set(SLCol4f(0.1f,0.1f,0.1f));
        sv->camera(cam1);
        _root3D = scene;
    }
    else
    if (sceneName == cmdScenePerPixelBlinn) //..................................
    {
        name("Blinn-Phong per pixel lighting");
        info(sv, "Per-pixel lighting with Blinn-Phong lightmodel. The reflection of 4 light sources is calculated per pixel.");

        // create material
        SLMaterial* m1 = new SLMaterial("m1", 0,0,0,0, _shaderProgs[PerPixBlinn]);
        m1->shininess(500);

        SLCamera* cam1 = new SLCamera;
        cam1->position(0,1,8);
        cam1->lookAt(0,1,0);
        cam1->focalDist(8);
        cam1->setInitialState();

        // define 4 light sources
        SLLightRect* light0 = new SLLightRect(2.0f,1.0f);
        light0->ambient(SLCol4f(0,0,0));
        light0->diffuse(SLCol4f(1,1,1));
        light0->position(0,3,0);
        light0->lookAt(0,0,0, 0,0,-1);
        light0->attenuation(0,0,1);

        SLLightSphere* light1 = new SLLightSphere(0.1f);
        light1->ambient(SLCol4f(0,0,0));
        light1->diffuse(SLCol4f(1,0,0));
        light1->specular(SLCol4f(1,0,0));
        light1->position(0, 0, 2);
        light1->lookAt(0, 0, 0);
        light1->attenuation(0,0,1);

        SLLightSphere* light2 = new SLLightSphere(0.1f);
        light2->ambient(SLCol4f(0,0,0));
        light2->diffuse(SLCol4f(0,1,0));
        light2->specular(SLCol4f(0,1,0));
        light2->position(1.5, 1.5, 1.5);
        light2->lookAt(0, 0, 0);
        light2->spotCutoff(20);
        light2->attenuation(0,0,1);

        SLLightSphere* light3 = new SLLightSphere(0.1f);
        light3->ambient(SLCol4f(0,0,0));
        light3->diffuse(SLCol4f(0,0,1));
        light3->specular(SLCol4f(0,0,1));
        light3->position(-1.5, 1.5, 1.5);
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

        _backColor.set(SLCol4f(0.1f,0.1f,0.1f));
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
        cam1->position(0,3,8);
        cam1->lookAt(0, 0, 0);
        cam1->focalDist(8);
        cam1->setInitialState();

        // Create generic shader program with 4 custom uniforms
        SLGLShaderProg* sp = new SLGLShaderProgGeneric("Wave.vert", "Wave.frag");
        SLGLShaderUniform1f* u_h = new SLGLShaderUniform1f(UF1Const, "u_h", 0.1f, 0.05f, 0.0f, 0.5f, (SLKey)'H');
        _eventHandlers.push_back(u_h);
        sp->addUniform1f(u_h);
        sp->addUniform1f(new SLGLShaderUniform1f(UF1Inc,    "u_t", 0.0f, 0.06f));
        sp->addUniform1f(new SLGLShaderUniform1f(UF1Const,  "u_a", 2.5f));
        sp->addUniform1f(new SLGLShaderUniform1f(UF1IncDec, "u_b", 2.2f, 0.01f, 2.0f, 2.5f));

        // Create materials
        SLMaterial* matWater = new SLMaterial("matWater", SLCol4f(0.45f,0.65f,0.70f),
                                                        SLCol4f::WHITE, 300);
        matWater->shaderProg(sp);
        SLMaterial* matRed  = new SLMaterial("matRed", SLCol4f(1.00f,0.00f,0.00f));

        // water rectangle in the y=0 plane
        SLNode* wave = new SLNode(new SLRectangle(SLVec2f(-SL_PI,-SL_PI), SLVec2f( SL_PI, SL_PI),
                                                    40, 40, "WaterRect", matWater));
        wave->rotate(90, -1,0,0);

        SLLightSphere* light0 = new SLLightSphere();
        light0->ambient(SLCol4f(0,0,0));
        light0->diffuse(SLCol4f(1,1,1));
        light0->translate(0,4,-4, TS_Local);
        light0->attenuation(1,0,0);

        SLNode* scene = new SLNode;
        scene->addChild(light0);
        scene->addChild(wave);
        scene->addChild(new SLNode(new SLSphere(1, 32, 32, "Red Sphere", matRed)));
        scene->addChild(cam1);

        _backColor.set(SLCol4f(0.1f,0.4f,0.8f));
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

        _backColor.set(.5f,.5f,1);

        SLCamera* cam1 = new SLCamera;
        cam1->position(0,3,8);
        cam1->lookAt(0, 0, 0);
        cam1->focalDist(8);

        // create texture
        SLGLTexture* tex1 = new SLGLTexture("Pool+X0512_C.png","Pool-X0512_C.png"
                                            ,"Pool+Y0512_C.png","Pool-Y0512_C.png"
                                            ,"Pool+Z0512_C.png","Pool-Z0512_C.png");
        SLGLTexture* tex2 = new SLGLTexture("tile1_0256_C.jpg");

        // Create generic shader program with 4 custom uniforms
        SLGLShaderProg* sp = new SLGLShaderProgGeneric("WaveRefractReflect.vert",
                                                       "RefractReflect.frag");
        SLGLShaderUniform1f* u_h = new SLGLShaderUniform1f(UF1Const, "u_h", 0.1f, 0.05f, 0.0f, 0.5f, (SLKey)'H');
        _eventHandlers.push_back(u_h);
        sp->addUniform1f(u_h);
        sp->addUniform1f(new SLGLShaderUniform1f(UF1Inc,    "u_t", 0.0f, 0.06f));
        sp->addUniform1f(new SLGLShaderUniform1f(UF1Const,  "u_a", 2.5f));
        sp->addUniform1f(new SLGLShaderUniform1f(UF1IncDec, "u_b", 2.2f, 0.01f, 2.0f, 2.5f));

        // Create materials
        SLMaterial* matWater = new SLMaterial("matWater", SLCol4f(0.45f,0.65f,0.70f),
                                                        SLCol4f::WHITE, 100, 0.1f, 0.9f, 1.5f);
        matWater->shaderProg(sp);
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
        rectF->translate(0,0,-SL_PI, TS_Local);
        rectL->rotate( 90, 0,1,0); rectL->translate(0,0,-SL_PI, TS_Local);
        rectN->rotate(180, 0,1,0); rectN->translate(0,0,-SL_PI, TS_Local);
        rectR->rotate(270, 0,1,0); rectR->translate(0,0,-SL_PI, TS_Local);
        rectB->rotate( 90,-1,0,0); rectB->translate(0,0,-SL_PI/6, TS_Local);

        SLLightSphere* light0 = new SLLightSphere();
        light0->ambient(SLCol4f(0,0,0));
        light0->diffuse(SLCol4f(1,1,1));
        light0->translate(0,4,-4, TS_Local);
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

        _backColor.set(SLCol4f(0.1f,0.4f,0.8f));
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
        SLMaterial* m1 = new SLMaterial("m1", texC, texN, 0, 0, _shaderProgs[BumpNormal]);

        SLCamera* cam1 = new SLCamera();
        cam1->name("cam1");
        cam1->position(0,0,20);
        cam1->lookAt(0, 0, 0);
        cam1->focalDist(20);

        SLLightSphere* light1 = new SLLightSphere(0.3f);
        light1->ambient(SLCol4f(0.1f, 0.1f, 0.1f));
        light1->diffuse(SLCol4f(1, 1, 1));
        light1->specular(SLCol4f(1, 1, 1));
        light1->attenuation(1,0,0);
        light1->position(0,0,5);
        light1->lookAt(0, 0, 0);
        light1->spotCutoff(40);
        light1->animation(new SLAnimation(2, 2, XAxis, 2, YAxis, loop));

        SLNode* scene = new SLNode;
        scene->addChild(light1);
        scene->addChild(new SLNode(new SLRectangle(SLVec2f(-5,-5),SLVec2f(5,5),1,1,"Rect", m1)));
        scene->addChild(cam1);

        _backColor.set(0.5f,0.5f,0.5f);
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
        SLGLShaderProg* sp = new SLGLShaderProgGeneric("BumpNormal.vert", "BumpNormalParallax.frag");
        SLGLShaderUniform1f* scale = new SLGLShaderUniform1f(UF1Const, "u_scale", 0.04f, 0.002f, 0, 1, (SLKey)'X');
        SLGLShaderUniform1f* offset = new SLGLShaderUniform1f(UF1Const, "u_offset", -0.03f, 0.002f,-1, 1, (SLKey)'O');
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
        cam1->position(0,0,20);
        cam1->lookAt(0, 0, 0);
        cam1->focalDist(20);

        SLLightSphere* light1 = new SLLightSphere(0.3f);
        light1->ambient(SLCol4f(0.1f, 0.1f, 0.1f));
        light1->diffuse(SLCol4f(1, 1, 1));
        light1->specular(SLCol4f(1, 1, 1));
        light1->attenuation(1,0,0);
        light1->position(0,0,5);
        light1->lookAt(0, 0, 0);
        light1->spotCutoff(50);
        light1->animation(new SLAnimation(2, 2, XAxis, 2, YAxis, loop));

        SLNode* scene = new SLNode;
        scene->addChild(light1);
        scene->addChild(new SLNode(new SLRectangle(SLVec2f(-5,-5),SLVec2f(5,5),1,1,"Rect", m1)));
        scene->addChild(cam1);

        _backColor.set(0.5f,0.5f,0.5f);
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
        SLGLShaderProg* sp = new SLGLShaderProgGeneric("BumpNormal.vert", "BumpNormalEarth.frag");
        SLGLShaderUniform1f* scale = new SLGLShaderUniform1f(UF1Const, "u_scale", 0.02f, 0.002f, 0, 1, (SLKey)'X');
        SLGLShaderUniform1f* offset = new SLGLShaderUniform1f(UF1Const, "u_offset", -0.02f, 0.002f,-1, 1, (SLKey)'O');
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
        matEarth->shaderProg(sp);

        SLCamera* cam1 = new SLCamera;
        cam1->position(0,0,4);
        cam1->lookAt(0, 0, 0);
        cam1->focalDist(4);

        SLLightSphere* sun = new SLLightSphere();
        sun->ambient(SLCol4f(0,0,0));
        sun->diffuse(SLCol4f(1,1,1));
        sun->specular(SLCol4f(0.2f,0.2f,0.2f));
        sun->attenuation(1,0,0);
        sun->animation(new SLAnimation(24, 50, XAxis, 50, ZAxis, loop));

        SLNode* earth = new SLNode(new SLSphere(1, 36, 36, "Earth", matEarth));
        earth->rotate(90,-1,0,0);

        SLNode* scene = new SLNode;
        scene->addChild(sun);
        scene->addChild(earth);
        scene->addChild(cam1);

        _backColor.set(SLCol4f(0,0,0));
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
        SLGLShaderProg* sp1 = new SLGLShaderProgGeneric("Reflect.vert", "Reflect.frag");
        SLGLShaderProg* sp2 = new SLGLShaderProgGeneric("RefractReflect.vert", "RefractReflect.frag");
   
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
        SLCol4f refrSpec1 (1.0f, 1.0f, 1.0f); // Tests only
        SLMaterial* refl=new SLMaterial("refl", blackRGB, refrSpec1, 1000, 1.0f);
        refl->textures().push_back(tex1);
        refl->shaderProg(sp1);

        // Material for glass sphere
        SLCol4f refrDiff (0.0f, 0.0f, 0.0f, 0.01f);
        SLCol4f refrSpec (0.05f, 0.05f, 0.05f);
        SLMaterial* refr=new SLMaterial("refr", blackRGB, blackRGB, 1000, 0.05f, 0.95f, 1.5f);
        refr->translucency(1000);
        refr->transmission(SLCol4f::WHITE);
        refr->textures().push_back(tex1);
        refr->shaderProg(sp2);
   
        SLNode* sphere1 = new SLNode(new SLSphere(0.5f, 32, 32, "Sphere1", refl));
        sphere1->translate(-0.65f, -0.75f, -0.55f, TS_Local);

        SLNode* sphere2 = new SLNode(new SLSphere(0.45f, 32, 32, "Sphere2", refr));
        sphere2->translate( 0.73f, -0.8f, 0.10f, TS_Local);

        SLNode* balls = new SLNode;
        balls->addChild(sphere1);
        balls->addChild(sphere2);

        // Rectangular light 
        SLLightRect* lightRect = new SLLightRect(1, 0.65f);
        lightRect->rotate(90, -1.0f, 0.0f, 0.0f);
        lightRect->translate(0.0f, -0.25f, 1.18f, TS_Local);
        lightRect->spotCutoff(90);
        lightRect->spotExponent(1.0);
        lightRect->diffuse(lightEmisRGB);
        lightRect->attenuation(0,0,1);
        lightRect->samplesXY(5, 3);

        _globalAmbiLight.set(lightEmisRGB*0.05f);

        // create camera
        SLCamera* cam1 = new SLCamera();
        cam1->position(0.0f, 0.40f, 6.35f);
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
        b->rotate(90, -1,0,0); b->translate(0,0,pB,TS_Local); scene->addChild(b);
   
        // top plane
        SLNode* t = new SLNode(new SLRectangle(SLVec2f(pL,pF), SLVec2f(pR,pN), 6, 6, "top", cream)); 
        t->rotate(90, 1,0,0); t->translate(0,0,-pT,TS_Local); scene->addChild(t);
   
        // far plane
        SLNode* f = new SLNode(new SLRectangle(SLVec2f(pL,pB), SLVec2f(pR,pT), 6, 6, "far", cream)); 
        f->translate(0,0,pF,TS_Local); scene->addChild(f);
   
        // left plane
        SLNode* l = new SLNode(new SLRectangle(SLVec2f(-pN,pB), SLVec2f(-pF,pT), 6, 6, "left", red)); 
        l->rotate(90, 0,1,0); l->translate(0,0,pL,TS_Local); scene->addChild(l);
   
        // right plane
        SLNode* r = new SLNode(new SLRectangle(SLVec2f(pF,pB), SLVec2f(pN,pT), 6, 6, "right", blue)); 
        r->rotate(90, 0,-1,0); r->translate(0,0,-pR,TS_Local); scene->addChild(r);
      
        scene->addChild(balls);

        _backColor.set(SLCol4f(0.0f,0.0f,0.0f));
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
                                                     100, 0.0f, 0.0f, 1.0f);

        SLCamera* cam1 = new SLCamera();
        cam1->position(0, 0.1f, 2.5f);
        cam1->lookAt(0, 0, 0);
        cam1->focalDist(4);

        SLNode *rect = new SLNode(new SLRectangle(SLVec2f(-3,-3), SLVec2f(5,4), 20, 20, "Floor", matYel));
        rect->rotate(90, -1,0,0);
        rect->translate(0, -1, -0.5f, TS_Local);

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
        scene->addChild(SphereGroup(1, 0,0,0, 1, 32, matGla, matRed));
        scene->addChild(rect);
        scene->addChild(cam1);

        _backColor.set(SLCol4f(0.1f,0.4f,0.8f));
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
        cam1->position(0, 0.1f, 6);
        cam1->lookAt(0, 0, 0);
        cam1->focalDist(6);

        SLNode* rect = new SLNode(new SLRectangle(SLVec2f(-3,-3), SLVec2f(5,4), 32, 32, "Rect", matYel));
        rect->rotate(90, -1,0,0);
        rect->translate(0, -1, -0.5f, TS_Local);

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
        light1->position(2, 2, 2);
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

        _backColor.set(SLCol4f(0.1f,0.4f,0.8f));
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
        cam1->position(0, 2, 7);
        cam1->lookAt(0, 0, 0);
        cam1->focalDist(7);
        cam1->lensDiameter(0.4f);
        cam1->lensSamples()->samples(numSamples, numSamples);
        cam1->setInitialState();

        SLNode* rect = new SLNode(new SLRectangle(SLVec2f(-5,-5), SLVec2f(5,5), 20, 20, "Rect", mT));
        rect->rotate(90, -1,0,0);
        rect->translate(0,0,-0.5f, TS_Local);

        SLLightSphere* light1 = new SLLightSphere(2,2,0, 0.1f);
        light1->attenuation(0,0,1);

        SLNode* balls = new SLNode;
        SLNode* s;
        s = new SLNode(new SLSphere(0.5f,32,32,"S1",mW)); s->translate( 2.0,0,-4,TS_Local);  balls->addChild(s);
        s = new SLNode(new SLSphere(0.5f,32,32,"S2",mB)); s->translate( 1.5,0,-3,TS_Local);  balls->addChild(s);
        s = new SLNode(new SLSphere(0.5f,32,32,"S3",mY)); s->translate( 1.0,0,-2,TS_Local);  balls->addChild(s);
        s = new SLNode(new SLSphere(0.5f,32,32,"S4",mR)); s->translate( 0.5,0,-1,TS_Local);  balls->addChild(s);
        s = new SLNode(new SLSphere(0.5f,32,32,"S5",mG)); s->translate( 0.0,0, 0,TS_Local);  balls->addChild(s);
        s = new SLNode(new SLSphere(0.5f,32,32,"S6",mM)); s->translate(-0.5,0, 1,TS_Local);  balls->addChild(s);
        s = new SLNode(new SLSphere(0.5f,32,32,"S7",mW)); s->translate(-1.0,0, 2,TS_Local);  balls->addChild(s);

        SLNode* scene  = new SLNode;
        scene->addChild(light1);
        scene->addChild(balls);
        scene->addChild(rect);
        scene->addChild(cam1);

        _backColor.set(SLCol4f(0.1f,0.4f,0.8f));
        sv->camera(cam1);
        _root3D = scene;
    }

    // call onInitialize on all scene views
    // @todo The parameter sv now marks the active scene view (which will recieve the
    //       cameras of the loaded scene, if any) this needs review and needs to be
    //       implemented better.

    for (SLint i = 0; i < _sceneViews.size(); ++i)
    {
        if (_sceneViews[i] != NULL)
        {   _sceneViews[i]->onInitialize();
            _sceneViews[i]->showLoading(false);
        }
    }

}
//-----------------------------------------------------------------------------
