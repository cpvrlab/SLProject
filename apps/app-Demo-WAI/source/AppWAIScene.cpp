#include <AppWAIScene.h>
#include <SLBox.h>
#include <SLLightDirect.h>
#include <SLLightSpot.h>
#include <SLCoordAxis.h>
#include <SLPoints.h>
#include <SLAssimpImporter.h>

AppWAIScene::AppWAIScene()
{
}

void AppWAIScene::rebuild(std::string location, std::string area)
{
    rootNode          = new SLNode("scene");
    cameraNode        = new SLCamera("Camera 1");
    mapNode           = new SLNode("map");
    mapPC             = new SLNode("MapPC");
    mapMatchedPC      = new SLNode("MapMatchedPC");
    mapLocalPC        = new SLNode("MapLocalPC");
    mapMarkerCornerPC = new SLNode("MapMarkerCornerPC");
    keyFrameNode      = new SLNode("KeyFrames");
    covisibilityGraph = new SLNode("CovisibilityGraph");
    spanningTree      = new SLNode("SpanningTree");
    loopEdges         = new SLNode("LoopEdges");

    redMat = new SLMaterial(SLCol4f::RED, "Red");
    redMat->program(new SLGLGenericProgram("ColorUniformPoint.vert", "Color.frag"));
    redMat->program()->addUniform1f(new SLGLUniform1f(UT_const, "u_pointSize", 3.0f));
    greenMat = new SLMaterial(SLCol4f::GREEN, "Green");
    greenMat->program(new SLGLGenericProgram("ColorUniformPoint.vert", "Color.frag"));
    greenMat->program()->addUniform1f(new SLGLUniform1f(UT_const, "u_pointSize", 5.0f));
    blueMat = new SLMaterial(SLCol4f::BLUE, "Blue");
    blueMat->program(new SLGLGenericProgram("ColorUniformPoint.vert", "Color.frag"));
    blueMat->program()->addUniform1f(new SLGLUniform1f(UT_const, "u_pointSize", 4.0f));
    yellowMat = new SLMaterial("mY", SLCol4f(1, 1, 0, 0.5f));

    if (location == "avenches")
    {
        if (area == "entrance")
        {
            SLAssimpImporter importer;
            augmentationRoot = importer.load("GLTF/Avenches/AvenchesEntrance.gltf",
                                             true,
                                             nullptr,
                                             0.4f);

            // Set some ambient light
            for (auto child : augmentationRoot->children())
            {
                for (auto mesh : child->meshes())
                {
                    mesh->mat()->ambient(SLCol4f(0.5f, 0.5f, 0.5f));
                    mesh->mat()->diffuse(SLCol4f(0.5f, 0.5f, 0.5f));
                    mesh->mat()->specular(SLCol4f(0.5f, 0.5f, 0.5f));
                }
            }

            SLNode* n = augmentationRoot->findChild<SLNode>("TexturedMesh", true);
            if (n)
            {
                n->drawBits()->set(SL_DB_CULLOFF, true);
            }

            // Create directional light for the sun light
            SLLightDirect* light = new SLLightDirect(1.0f);
            light->ambient(SLCol4f(1, 1, 1));
            light->diffuse(SLCol4f(1, 1, 1));
            light->specular(SLCol4f(1, 1, 1));
            light->attenuation(1, 0, 0);
            light->translation(0, 10, 0);
            light->lookAt(10, 0, 10);

            rootNode->addChild(augmentationRoot);
            rootNode->addChild(light);
        }
    }
    else if (location == "augst")
    {
        if (area == "templeHill-marker")
        {
            SLAssimpImporter importer;
            augmentationRoot = importer.load("GLTF/AugustaRaurica/Tempel-Theater-02.gltf",
                                             true,
                                             nullptr,
                                             0.4f);

            SLNode* portikusSockel = augmentationRoot->findChild<SLNode>("Tmp-Portikus-Sockel", true);
            if (portikusSockel)
            {
                portikusSockel->drawBits()->set(SL_DB_HIDDEN, true);
            }

            SLNode* boden = augmentationRoot->findChild<SLNode>("Tmp-Boden", true);
            if (boden)
            {
                boden->drawBits()->set(SL_DB_HIDDEN, true);
            }

            // Create directional light for the sun light
            SLLightDirect* light = new SLLightDirect(5.0f);
            light->ambient(SLCol4f(1, 1, 1));
            light->diffuse(SLCol4f(1, 1, 1));
            light->specular(SLCol4f(1, 1, 1));
            light->attenuation(1, 0, 0);
            light->translation(0, 10, 0);
            light->lookAt(10, 0, 10);

            rootNode->addChild(augmentationRoot);
            rootNode->addChild(light);
        }
        else if (area == "templeHillTheaterBottom")
        {
            SLAssimpImporter importer;
            augmentationRoot = importer.load("GLTF/AugustaRaurica/Tempel-Theater-02.gltf",
                                             true,
                                             nullptr,
                                             0.4f);

            SLNode* portikusSockel = augmentationRoot->findChild<SLNode>("Tmp-Portikus-Sockel", true);
            if (portikusSockel)
            {
                portikusSockel->drawBits()->set(SL_DB_HIDDEN, true);
            }

            SLNode* boden = augmentationRoot->findChild<SLNode>("Tmp-Boden", true);
            if (boden)
            {
                boden->drawBits()->set(SL_DB_HIDDEN, true);
            }

            // Create directional light for the sun light
            SLLightDirect* light = new SLLightDirect(5.0f);
            light->ambient(SLCol4f(1, 1, 1));
            light->diffuse(SLCol4f(1, 1, 1));
            light->specular(SLCol4f(1, 1, 1));
            light->attenuation(1, 0, 0);
            light->translation(0, 10, 0);
            light->lookAt(10, 0, 10);

            rootNode->addChild(augmentationRoot);
            rootNode->addChild(light);
        }
    }

#if 0 // office table boxes scene
    //SLBox*      box1     = new SLBox(0.0f, 0.0f, 0.0f, l, h, b, "Box 1", yellow);
    SLBox* box1 = new SLBox(0.0f, 0.0f, 0.0f, 0.355f, 0.2f, 0.1f, "Box 1", yellow);
    //SLBox*  box1     = new SLBox(0.0f, 0.0f, 0.0f, 10.0f, 5.0f, 3.0f, "Box 1", yellow);
    SLNode* boxNode1 = new SLNode(box1, "boxNode1");
    //boxNode1->rotate(-45.0f, 1.0f, 0.0f, 0.0f);
    //boxNode1->translate(10.0f, -5.0f, 15.0f);
    boxNode1->translate(0.316, -1.497f, -0.1f);
    SLBox*  box2     = new SLBox(0.0f, 0.0f, 0.0f, 0.355f, 0.2f, 0.1f, "Box 2", yellow);
    SLNode* boxNode2 = new SLNode(box2, "boxNode2");
    SLNode* axisNode = new SLNode(new SLCoordAxis(), "axis node");
    SLBox*  box3     = new SLBox(0.0f, 0.0f, 0.0f, 1.745f, 0.745, 0.81, "Box 3", yellow);
    SLNode* boxNode3 = new SLNode(box3, "boxNode3");
    boxNode3->translate(2.561f, -5.147f, -0.06f);

    rootNode->addChild(boxNode1);
    rootNode->addChild(axisNode);
    rootNode->addChild(boxNode2);
    rootNode->addChild(boxNode3);
#endif

#if 0 // locsim scene
    SLBox*  box2     = new SLBox(-0.5f, -0.5f, -0.5f, 0.5f, 0.5f, 0.5f, "Box 2", yellow);
    SLNode* boxNode2 = new SLNode(box2, "boxNode2");
    boxNode2->translation(79.7f, -3.26f, 2.88f);
    boxNode2->scale(1.0f, 8.95f, 1.0f);
    boxNode2->rotate(1.39f, SLVec3f(1.0f, 0.0f, 0.0f), TS_parent);
    boxNode2->rotate(3.88f, SLVec3f(0.0f, 1.0f, 0.0f), TS_parent);
    boxNode2->rotate(-0.1f, SLVec3f(0.0f, 0.0f, 1.0f), TS_parent);

    SLBox*  box3     = new SLBox(-0.5f, -0.5f, -0.5f, 0.5f, 0.5f, 0.5f, "Box 3", yellow);
    SLNode* boxNode3 = new SLNode(box3, "boxNode3");
    boxNode3->translation(83.54f, -3.26f, 23.64f);
    boxNode3->scale(1.0f, 8.95f, 1.0f);
    boxNode3->rotate(1.39f, SLVec3f(1.0f, 0.0f, 0.0f), TS_parent);
    boxNode3->rotate(3.88f, SLVec3f(0.0f, 1.0f, 0.0f), TS_parent);
    boxNode3->rotate(-0.1f, SLVec3f(0.0f, 0.0f, 1.0f), TS_parent);

    SLBox*  box4     = new SLBox(0.0f, 0.0f, 0.0f, 1.0f, 1.0f, 21.11f, "Box 4", yellow);
    SLNode* boxNode4 = new SLNode(box4, "boxNode4");
    boxNode4->translation(79.38f, 0.74f, 3.0f);
    boxNode4->rotate(-0.19f, SLVec3f(1.0f, 0.0f, 0.0f), TS_parent);
    boxNode4->rotate(9.91f, SLVec3f(0.0f, 1.0f, 0.0f), TS_parent);
    boxNode4->rotate(-0.95f, SLVec3f(0.0f, 0.0f, 1.0f), TS_parent);

    rootNode->addChild(boxNode1);
    rootNode->addChild(boxNode2);
    rootNode->addChild(boxNode3);
    rootNode->addChild(boxNode4);
#endif

    //boxNode->addChild(axisNode);

    covisibilityGraphMat = new SLMaterial("YellowLines", SLCol4f::YELLOW);
    spanningTreeMat      = new SLMaterial("GreenLines", SLCol4f::GREEN);
    loopEdgesMat         = new SLMaterial("RedLines", SLCol4f::RED);

    cameraNode->translation(0, 0, 0.1f);
    cameraNode->lookAt(0, 0, 0);
    //for tracking we have to use the field of view from calibration
    cameraNode->clipNear(0.001f);
    cameraNode->clipFar(1000000.0f); // Increase to infinity?
    cameraNode->setInitialState();

    mapNode->addChild(mapPC);
    mapNode->addChild(mapMatchedPC);
    mapNode->addChild(mapLocalPC);
    mapNode->addChild(mapMarkerCornerPC);
    mapNode->addChild(keyFrameNode);
    mapNode->addChild(covisibilityGraph);
    mapNode->addChild(spanningTree);
    mapNode->addChild(loopEdges);
    mapNode->addChild(cameraNode);

    mapNode->rotate(180, 1, 0, 0);

    //setup scene
    rootNode->addChild(mapNode);
}

void AppWAIScene::adjustAugmentationTransparency(float kt)
{
    if (augmentationRoot)
    {
        for (SLNode* child : augmentationRoot->children())
        {
            for (SLMesh* mesh : child->meshes())
            {
                mesh->mat()->kt(kt);
                mesh->mat()->ambient(SLCol4f(0.3f, 0.3f, 0.3f));
                mesh->init(child);
            }
        }
    }
}
