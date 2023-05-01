// #############################################################################
//   File:      AppDemoSceneLoad.cpp
//   Date:      February 2018
//   Codestyle: https://github.com/cpvrlab/SLProject/wiki/SLProject-Coding-Style
//   Authors:   Marcus Hudritsch
//   License:   This software is provided under the GNU General Public License
//              Please visit: http://opensource.org/licenses/GPL-3.0
// #############################################################################

#include <SL.h>
#include <GlobalTimer.h>

#include <CVCapture.h>
#include <CVTrackedAruco.h>
#include <CVTrackedChessboard.h>
#include <CVTrackedFaces.h>
#include <CVTrackedFeatures.h>
#include <CVTrackedMediaPipeHands.h>
#include <CVCalibrationEstimator.h>

#include <SLAlgo.h>
#include <AppDemo.h>
#include <SLAssetManager.h>
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
#include <SLParticleSystem.h>
#include <SLPoints.h>
#include <SLPolygon.h>
#include <SLRectangle.h>
#include <SLSkybox.h>
#include <SLSphere.h>
#include <SLText.h>
#include <SLTexColorLUT.h>
#include <SLScene.h>
#include <SLGLProgramManager.h>
#include <SLGLTexture.h>
#include <Profiler.h>
#include <AppDemoGui.h>
#include <SLDeviceLocation.h>
#include <SLNodeLOD.h>
#include <imgui_color_gradient.h> // For color over life, need to create own color interpolator
#include <SLEntities.h>
#include <SLFileStorage.h>

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
//-----------------------------------------------------------------------------
//! Global pointer to dragon model for threaded loading
SLNode* gDragonModel = nullptr;
//-----------------------------------------------------------------------------
//! Creates a recursive sphere group used for the ray tracing scenes
SLNode* SphereGroupRT(SLAssetManager* am,
                      SLint           depth, // depth of recursion
                      SLfloat         x,
                      SLfloat         y,
                      SLfloat         z,
                      SLfloat         scale,
                      SLuint          resolution,
                      SLMaterial*     matGlass,
                      SLMaterial*     matRed)
{
    PROFILE_FUNCTION();

    SLstring name = matGlass->kt() > 0 ? "GlassSphere" : "RedSphere";
    if (depth == 0)
    {
        SLSphere* sphere  = new SLSphere(am, 0.5f * scale, resolution, resolution, name, matRed);
        SLNode*   sphNode = new SLNode(sphere, "Sphere");
        sphNode->translate(x, y, z, TS_object);
        return sphNode;
    }
    else
    {
        depth--;
        SLNode* sGroup = new SLNode(new SLSphere(am, 0.5f * scale, resolution, resolution, name, matGlass), "SphereGroupRT");
        sGroup->translate(x, y, z, TS_object);
        SLuint newRes = (SLuint)std::max((SLint)resolution - 4, 8);
        sGroup->addChild(SphereGroupRT(am, depth, 0.643951f * scale, 0, 0.172546f * scale, scale / 3, newRes, matRed, matRed));
        sGroup->addChild(SphereGroupRT(am, depth, 0.172546f * scale, 0, 0.643951f * scale, scale / 3, newRes, matRed, matRed));
        sGroup->addChild(SphereGroupRT(am, depth, -0.471405f * scale, 0, 0.471405f * scale, scale / 3, newRes, matRed, matRed));
        sGroup->addChild(SphereGroupRT(am, depth, -0.643951f * scale, 0, -0.172546f * scale, scale / 3, newRes, matRed, matRed));
        sGroup->addChild(SphereGroupRT(am, depth, -0.172546f * scale, 0, -0.643951f * scale, scale / 3, newRes, matRed, matRed));
        sGroup->addChild(SphereGroupRT(am, depth, 0.471405f * scale, 0, -0.471405f * scale, scale / 3, newRes, matRed, matRed));
        sGroup->addChild(SphereGroupRT(am, depth, 0.272166f * scale, 0.544331f * scale, 0.272166f * scale, scale / 3, newRes, matRed, matRed));
        sGroup->addChild(SphereGroupRT(am, depth, -0.371785f * scale, 0.544331f * scale, 0.099619f * scale, scale / 3, newRes, matRed, matRed));
        sGroup->addChild(SphereGroupRT(am, depth, 0.099619f * scale, 0.544331f * scale, -0.371785f * scale, scale / 3, newRes, matRed, matRed));
        return sGroup;
    }
}
//-----------------------------------------------------------------------------
//! Creates a recursive rotating sphere group used for performance test
/*!
 * This performance benchmark is expensive in terms of world matrix updates
 * because all sphere groups rotate. Therefore all children need to update
 * their wm every frame.
 * @param am Pointer to the asset manager
 * @param s Pointer to project scene aka asset manager
 * @param depth Max. allowed recursion depth
 * @param x Position in x direction
 * @param y Position in y direction
 * @param z Position in z direction
 * @param scale Scale factor > 0 and < 1 for the children spheres
 * @param resolution NO. of stack and slices of the spheres
 * @param mat Reference to an vector of materials
 * @return Group node of spheres
 */
SLNode* RotatingSphereGroup(SLAssetManager* am,
                            SLScene*        s,
                            SLint           depth,
                            SLfloat         x,
                            SLfloat         y,
                            SLfloat         z,
                            SLfloat         scale,
                            SLuint          resolution,
                            SLVMaterial&    mat)
{
    PROFILE_FUNCTION();
    assert(depth >= 0);
    assert(scale >= 0.0f && scale <= 1.0f);
    assert(resolution > 0 && resolution < 64);

    // Choose the material index randomly
    SLint iMat = Utils::random(0, (int)mat.size() - 1);

    // Generate unique names for meshes, nodes and animations
    static int sphereNum = 0;
    string     meshName  = "Mesh" + std::to_string(sphereNum);
    string     animName  = "Anim" + std::to_string(sphereNum);
    string     nodeName  = "Node" + std::to_string(sphereNum);
    sphereNum++;

    SLAnimation* nodeAnim = s->animManager().createNodeAnimation(animName,
                                                                 60,
                                                                 true,
                                                                 EC_linear,
                                                                 AL_loop);
    if (depth == 0)
    {
        SLSphere* sphere  = new SLSphere(am, 5.0f * scale, resolution, resolution, meshName, mat[iMat]);
        SLNode*   sphNode = new SLNode(sphere, nodeName);
        sphNode->translate(x, y, z, TS_object);
        nodeAnim->createNodeAnimTrackForRotation360(sphNode, SLVec3f(0, 1, 0));
        return sphNode;
    }
    else
    {
        depth--;

        // decrease resolution to reduce memory consumption
        if (resolution > 8)
            resolution -= 2;

        SLNode* sGroup = new SLNode(new SLSphere(am, 5.0f * scale, resolution, resolution, meshName, mat[iMat]), nodeName);
        sGroup->translate(x, y, z, TS_object);
        nodeAnim->createNodeAnimTrackForRotation360(sGroup, SLVec3f(0, 1, 0));
        sGroup->addChild(RotatingSphereGroup(am, s, depth, 6.43951f * scale, 0, 1.72546f * scale, scale / 3.0f, resolution, mat));
        sGroup->addChild(RotatingSphereGroup(am, s, depth, 1.72546f * scale, 0, 6.43951f * scale, scale / 3.0f, resolution, mat));
        sGroup->addChild(RotatingSphereGroup(am, s, depth, -4.71405f * scale, 0, 4.71405f * scale, scale / 3.0f, resolution, mat));
        sGroup->addChild(RotatingSphereGroup(am, s, depth, -6.43951f * scale, 0, -1.72546f * scale, scale / 3.0f, resolution, mat));
        sGroup->addChild(RotatingSphereGroup(am, s, depth, -1.72546f * scale, 0, -6.43951f * scale, scale / 3.0f, resolution, mat));
        sGroup->addChild(RotatingSphereGroup(am, s, depth, 4.71405f * scale, 0, -4.71405f * scale, scale / 3.0f, resolution, mat));
        sGroup->addChild(RotatingSphereGroup(am, s, depth, 2.72166f * scale, 5.44331f * scale, 2.72166f * scale, scale / 3.0f, resolution, mat));
        sGroup->addChild(RotatingSphereGroup(am, s, depth, -3.71785f * scale, 5.44331f * scale, 0.99619f * scale, scale / 3.0f, resolution, mat));
        sGroup->addChild(RotatingSphereGroup(am, s, depth, 0.99619f * scale, 5.44331f * scale, -3.71785f * scale, scale / 3.0f, resolution, mat));
        return sGroup;
    }
}
//-----------------------------------------------------------------------------
//! Build a hierarchical figurine with arms and legs
SLNode* BuildFigureGroup(SLAssetManager* am,
                         SLScene*        s,
                         SLMaterial*     mat,
                         SLbool          withAnimation)
{
    SLNode* cyl;
    SLuint  res = 16;

    // Feet
    SLNode* feet = new SLNode("feet group (T13,R6)");
    feet->addMesh(new SLSphere(am, 0.2f, 16, 16, "ankle", mat));
    SLNode* feetbox = new SLNode(new SLBox(am,
                                           -0.2f,
                                           -0.1f,
                                           0.0f,
                                           0.2f,
                                           0.1f,
                                           0.8f,
                                           "foot mesh",
                                           mat),
                                 "feet (T14)");
    feetbox->translate(0.0f, -0.25f, -0.15f, TS_object);
    feet->addChild(feetbox);
    feet->translate(0.0f, 0.0f, 1.6f, TS_object);
    feet->rotate(-90.0f, 1.0f, 0.0f, 0.0f);

    // Assemble low leg
    SLNode* leglow = new SLNode("low leg group (T11, R5)");
    leglow->addMesh(new SLSphere(am, 0.3f, res, res, "knee mesh", mat));
    cyl = new SLNode(new SLCylinder(am,
                                    0.2f,
                                    1.4f,
                                    1,
                                    res,
                                    false,
                                    false,
                                    "shin mesh",
                                    mat),
                     "shin (T12)");
    cyl->translate(0.0f, 0.0f, 0.2f, TS_object);
    leglow->addChild(cyl);
    leglow->addChild(feet);
    leglow->translate(0.0f, 0.0f, 1.27f, TS_object);
    leglow->rotate(0, 1.0f, 0.0f, 0.0f);

    // Assemble leg
    SLNode* leg = new SLNode("leg group");
    leg->addMesh(new SLSphere(am, 0.4f, res, res, "hip joint mesh", mat));
    cyl = new SLNode(new SLCylinder(am,
                                    0.3f,
                                    1.0f,
                                    1,
                                    res,
                                    false,
                                    false,
                                    "thigh mesh",
                                    mat),
                     "thigh (T10)");
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
    armlow->addMesh(new SLSphere(am, 0.2f, 16, 16, "elbow mesh", mat));
    cyl = new SLNode(new SLCylinder(am, 0.15f, 1.0f, 1, res, true, false, "low arm mesh", mat), "low arm (T7)");
    cyl->translate(0.0f, 0.0f, 0.14f, TS_object);
    armlow->addChild(cyl);
    armlow->translate(0.0f, 0.0f, 1.2f, TS_object);
    armlow->rotate(45, -1.0f, 0.0f, 0.0f);

    // Assemble arm
    SLNode* arm = new SLNode("arm group");
    arm->addMesh(new SLSphere(am, 0.3f, 16, 16, "shoulder mesh", mat));
    cyl = new SLNode(new SLCylinder(am, 0.2f, 1.0f, 1, res, false, false, "upper arm mesh", mat), "upper arm (T5)");
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
    SLNode* head = new SLNode(new SLSphere(am, 0.5f, res, res, "head mesh", mat), "head (T1)");
    head->translate(0.0f, 0.0f, -0.7f, TS_object);
    SLSphere* eye  = new SLSphere(am, 0.06f, res, res, "eye mesh", mat);
    SLNode*   eyeL = new SLNode(eye, SLVec3f(-0.15f, 0.48f, 0), "eyeL (T1.1)");
    SLNode*   eyeR = new SLNode(eye, SLVec3f(0.15f, 0.48f, 0), "eyeR (T1.2)");
    head->addChild(eyeL);
    head->addChild(eyeR);
    SLNode* neck = new SLNode(new SLCylinder(am, 0.25f, 0.3f, 1, res, false, false, "neck mesh", mat), "neck (T2)");
    neck->translate(0.0f, 0.0f, -0.3f, TS_object);

    // Assemble figure Left
    SLNode* figure = new SLNode("figure group (R1)");
    figure->addChild(new SLNode(new SLBox(am, -0.8f, -0.4f, 0.0f, 0.8f, 0.4f, 2.0f, "chest mesh", mat), "chest"));
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
        anim->createNodeAnimTrackForRotation(legLeft, 60, SLVec3f(1, 0, 0));

        SLNode* legLowLeft = legLeft->findChild<SLNode>("low leg group (T11, R5)");
        anim->createNodeAnimTrackForRotation(legLowLeft, 40, SLVec3f(1, 0, 0));

        SLNode* feetLeft = legLeft->findChild<SLNode>("feet group (T13,R6)");
        anim->createNodeAnimTrackForRotation(feetLeft, 40, SLVec3f(1, 0, 0));

        armLeft = figure->findChild<SLNode>("left arm group (T3,R2)");
        armLeft->rotate(-45, -1, 0, 0);
        anim->createNodeAnimTrackForRotation(armLeft, -60, SLVec3f(1, 0, 0));

        armRight = figure->findChild<SLNode>("right arm group (T4,R3)");
        armRight->rotate(45, -1, 0, 0);
        anim->createNodeAnimTrackForRotation(armRight, 60, SLVec3f(1, 0, 0));
    }

    return figure;
}
//-----------------------------------------------------------------------------
//! Adds another level to Jan's Universe scene
void addUniverseLevel(SLAssetManager* am,
                      SLScene*        s,
                      SLNode*         parent,
                      SLint           parentID,
                      SLuint          currentLevel,
                      SLuint          levels,
                      SLuint          childCount,
                      SLVMaterial&    materials,
                      SLVMesh&        meshes,
                      SLuint&         numNodes)
{
    if (currentLevel >= levels)
        return;

    const float degPerChild = 360.0f / (float)childCount;
    SLuint      mod         = currentLevel % 3;

    float scaleFactor = 0.25f;

    for (SLuint i = 0; i < childCount; i++)
    {
        numNodes++;
        string childName = "Node" + std::to_string(numNodes) +
                           "-L" + std::to_string(currentLevel) +
                           "-C" + std::to_string(i);
        SLNode* child = new SLNode(meshes[numNodes % meshes.size()], childName);

        child->rotate((float)i * degPerChild, 0, 0, 1);
        child->translate(2, 0, 0);
        child->scale(scaleFactor);

        // Node animation on child node
        string       animName  = "Anim" + std::to_string(numNodes);
        SLAnimation* childAnim = s->animManager().createNodeAnimation(animName.c_str(),
                                                                      60,
                                                                      true,
                                                                      EC_linear,
                                                                      AL_loop);
        childAnim->createNodeAnimTrackForRotation360(child, {0, 0, 1});

        parent->addChild(child);

        addUniverseLevel(am,
                         s,
                         child,
                         parentID,
                         currentLevel + 1,
                         levels,
                         childCount,
                         materials,
                         meshes,
                         numNodes);
    }
}
//-----------------------------------------------------------------------------
//! Generates the Jan's Universe scene
void generateUniverse(SLAssetManager* am,
                      SLScene*        s,
                      SLNode*         parent,
                      SLint           parentID,
                      SLuint          levels,
                      SLuint          childCount,
                      SLVMaterial&    materials,
                      SLVMesh&        meshes)
{
    // Point light without mesh
    SLLightSpot* light = new SLLightSpot(am, s, 0, 0, 0, 1.0f, 180, 0, 1000, 1000, true);
    light->attenuation(1, 0, 0);
    light->scale(10, 10, 10);
    light->diffuseColor({1.0f, 1.0f, 0.5f});

    // Node animation on light node
    SLAnimation* lightAnim = s->animManager().createNodeAnimation("anim0",
                                                                  60,
                                                                  true,
                                                                  EC_linear,
                                                                  AL_loop);
    lightAnim->createNodeAnimTrackForRotation360(light, SLVec3f(0, 1, 0));

    parent->addChild(light);

    SLuint numNodes = 1;

    addUniverseLevel(am,
                     s,
                     light,
                     parentID,
                     1,
                     levels,
                     childCount,
                     materials,
                     meshes,
                     numNodes);
}
//-----------------------------------------------------------------------------
//! Creates a complex fire group node
SLNode* createComplexFire(SLAssetManager* am,
                          SLScene*        s,
                          SLbool          withLight,
                          SLGLTexture*    texFireCld,
                          SLGLTexture*    texFireFlm,
                          SLint           flipbookCols,
                          SLint           flipbookRows,
                          SLGLTexture*    texCircle,
                          SLGLTexture*    texSmokeB,
                          SLGLTexture*    texSmokeW)
{
    SLNode* fireComplex = new SLNode("Fire complex node");

    // Fire light node
    if (withLight)
    {
        SLLightSpot* light1 = new SLLightSpot(am, s, 0.1f, 180.0f, false);
        light1->name("Fire light node");
        light1->translate(0, 1.0f, 0);
        light1->diffuseColor(SLCol4f(1, 0.7f, 0.2f));
        light1->diffusePower(15);
        light1->attenuation(0, 0, 1);
        fireComplex->addChild(light1);
    }

    // Fire glow mesh
    {
        SLParticleSystem* fireGlowMesh = new SLParticleSystem(am,
                                                              24,
                                                              SLVec3f(-0.1f, 0.0f, -0.1f),
                                                              SLVec3f(0.1f, 0.0f, 0.1f),
                                                              4.0f,
                                                              texFireCld,
                                                              "Fire glow PS",
                                                              texFireFlm);
        fireGlowMesh->timeToLive(2.0f);
        fireGlowMesh->billboardType(BT_Camera);
        fireGlowMesh->radiusW(0.4f);
        fireGlowMesh->radiusH(0.4f);
        fireGlowMesh->doShape(false);
        fireGlowMesh->doRotation(true);
        fireGlowMesh->doRotRange(true);
        fireGlowMesh->doSizeOverLT(false);
        fireGlowMesh->doAlphaOverLT(false);
        fireGlowMesh->doColorOverLT(false);
        fireGlowMesh->doBlendBrightness(true);
        fireGlowMesh->color(SLCol4f(0.925f, 0.5f, 0.097f, 0.199f));
        fireGlowMesh->doAcceleration(false);
        SLNode* flameGlowNode = new SLNode(fireGlowMesh, "Fire glow node");
        flameGlowNode->translate(0, 0.15f, 0);
        fireComplex->addChild(flameGlowNode);
    }

    // Fire flame mesh
    {
        SLParticleSystem* fireFlameMesh = new SLParticleSystem(am,
                                                               1,
                                                               SLVec3f(0.0f, 0.0f, 0.0f),
                                                               SLVec3f(0.0f, 0.0f, 0.0f),

                                                               1.0f,
                                                               texFireCld,
                                                               "Fire flame PS",
                                                               texFireFlm);
        // Fire flame flipbook settings
        fireFlameMesh->flipbookColumns(flipbookCols);
        fireFlameMesh->flipbookRows(flipbookRows);
        fireFlameMesh->doFlipBookTexture(true);
        fireFlameMesh->doCounterGap(false); // We don't want to have flickering
        fireFlameMesh->changeTexture();     // Switch texture, need to be done, to have flipbook texture as active

        fireFlameMesh->doAlphaOverLT(false);
        fireFlameMesh->doSizeOverLT(false);
        fireFlameMesh->doRotation(false);

        fireFlameMesh->frameRateFB(64);
        fireFlameMesh->radiusW(0.6f);
        fireFlameMesh->radiusH(0.6f);
        fireFlameMesh->scale(1.2f);
        fireFlameMesh->billboardType(BT_Vertical);

        // Fire flame color
        fireFlameMesh->doColor(true);
        fireFlameMesh->color(SLCol4f(0.52f, 0.47f, 0.32f, 1.0f));
        fireFlameMesh->doBlendBrightness(true);

        // Fire flame size
        fireFlameMesh->doSizeOverLTCurve(true);
        float sizeCPArrayFl[4] = {0.0f, 1.25f, 0.0f, 1.0f};
        fireFlameMesh->bezierControlPointSize(sizeCPArrayFl);
        float sizeSEArrayFl[4] = {0.0f, 1.0f, 1.0f, 1.0f};
        fireFlameMesh->bezierStartEndPointSize(sizeSEArrayFl);
        fireFlameMesh->generateBernsteinPSize();

        // Fire flame node
        SLNode* fireFlameNode = new SLNode(fireFlameMesh, "Fire flame node");
        fireFlameNode->translate(0.0f, 0.7f, 0.0f, TS_object);
        fireComplex->addChild(fireFlameNode);
    }

    // Fire smoke black mesh
    {
        SLParticleSystem* fireSmokeB = new SLParticleSystem(am,
                                                            8,
                                                            SLVec3f(0.0f, 1.0f, 0.0f),
                                                            SLVec3f(0.0f, 0.7f, 0.0f),
                                                            2.0f,
                                                            texSmokeB,
                                                            "Fire smoke black PS",
                                                            texFireFlm);
        fireSmokeB->doColor(false);

        // Fire smoke black size
        fireSmokeB->doSizeOverLT(true);
        fireSmokeB->doSizeOverLTCurve(true);
        float sizeCPArraySB[4] = {0.0f, 1.0f, 1.0f, 2.0f};
        fireSmokeB->bezierControlPointSize(sizeCPArraySB);
        float sizeSEArraySB[4] = {0.0f, 1.0f, 1.0f, 2.0f};
        fireSmokeB->bezierStartEndPointSize(sizeSEArraySB);
        fireSmokeB->generateBernsteinPSize();

        // Fire smoke black alpha
        fireSmokeB->doAlphaOverLT(true);
        fireSmokeB->doAlphaOverLTCurve(true);
        float alphaCPArraySB[4] = {0.0f, 0.4f, 1.0f, 0.4f};
        fireSmokeB->bezierControlPointAlpha(alphaCPArraySB);
        float alphaSEArraySB[4] = {0.0f, 0.0f, 1.0f, 0.0f};
        fireSmokeB->bezierStartEndPointAlpha(alphaSEArraySB);
        fireSmokeB->generateBernsteinPAlpha();

        // Fire smoke black acceleration
        fireSmokeB->doAcceleration(true);
        fireSmokeB->doAccDiffDir(true);
        fireSmokeB->acceleration(0.0f, 0.25f, 0.3f);

        SLNode* fireSmokeBlackNode = new SLNode(fireSmokeB, "Fire smoke black node");
        fireSmokeBlackNode->translate(0.0f, 0.9f, 0.0f, TS_object);
        fireComplex->addChild(fireSmokeBlackNode);
    }

    // Fire smoke white mesh
    {
        SLParticleSystem* fireSmokeW = new SLParticleSystem(am,
                                                            40,
                                                            SLVec3f(0.0f, 0.8f, 0.0f),
                                                            SLVec3f(0.0f, 0.6f, 0.0f),
                                                            4.0f,
                                                            texSmokeW,
                                                            "Fire smoke white PS",
                                                            texFireFlm);

        fireSmokeW->doColor(false);

        // Size
        fireSmokeW->doSizeOverLT(true);
        fireSmokeW->doSizeOverLTCurve(true);
        float sizeCPArraySW[4] = {0.0f, 0.5f, 1.0f, 2.0f};
        fireSmokeW->bezierControlPointSize(sizeCPArraySW);
        float sizeSEArraySW[4] = {0.0f, 0.5f, 1.0f, 2.0f};
        fireSmokeW->bezierStartEndPointSize(sizeSEArraySW);
        fireSmokeW->generateBernsteinPSize();

        // Alpha
        fireSmokeW->doAlphaOverLT(true);
        fireSmokeW->doAlphaOverLTCurve(true);
        float alphaCPArraySW[4] = {0.0f, 0.018f, 1.0f, 0.018f};
        fireSmokeW->bezierControlPointAlpha(alphaCPArraySW);
        float alphaSEArraySW[4] = {0.0f, 0.0f, 1.0f, 0.0f};
        fireSmokeW->bezierStartEndPointAlpha(alphaSEArraySW);
        fireSmokeW->generateBernsteinPAlpha();

        // Acceleration
        fireSmokeW->doAcceleration(true);
        fireSmokeW->doAccDiffDir(true);
        fireSmokeW->acceleration(0.0f, 0.25f, 0.3f);

        SLNode* fireSmokeWNode = new SLNode(fireSmokeW, "Fire smoke white node");
        fireSmokeWNode->translate(0.0f, 0.9f, 0.0f, TS_object);
        fireComplex->addChild(fireSmokeWNode);
    }

    // Fire sparks rising mesh
    {
        SLParticleSystem* fireSparksRising = new SLParticleSystem(am,
                                                                  30,
                                                                  SLVec3f(-0.5f, 1, -0.5f),
                                                                  SLVec3f(0.5f, 2, 0.5f),
                                                                  1.2f,
                                                                  texCircle,
                                                                  "Fire sparks rising PS",
                                                                  texFireFlm);
        fireSparksRising->scale(0.05f);
        fireSparksRising->radiusH(0.8f);
        fireSparksRising->radiusW(0.3f);
        fireSparksRising->doShape(true);
        fireSparksRising->doRotation(false);
        fireSparksRising->shapeType(ST_Sphere);
        fireSparksRising->shapeRadius(0.05f);
        fireSparksRising->doAcceleration(true);
        fireSparksRising->acceleration(0, 1.5f, 0);
        fireSparksRising->doColor(true);
        fireSparksRising->doColorOverLT(true);
        fireSparksRising->doBlendBrightness(true);
        fireSparksRising->colorPoints().clear();
        fireSparksRising->colorPoints().push_back(SLColorLUTPoint(SLCol3f::WHITE, 0.0f));
        fireSparksRising->colorPoints().push_back(SLColorLUTPoint(SLCol3f::YELLOW, 0.5f));
        fireSparksRising->colorPoints().push_back(SLColorLUTPoint(SLCol3f::RED, 1.0f));
        ImGradient gradient;
        gradient.getMarks().clear();
        for (auto cp : fireSparksRising->colorPoints())
            gradient.addMark(cp.pos, ImColor(cp.color.r, cp.color.g, cp.color.b));
        fireSparksRising->colorArr(gradient.cachedValues());
        fireSparksRising->doSizeOverLT(false);
        fireSparksRising->doAlphaOverLT(false);
        fireSparksRising->doGravity(false);
        fireComplex->addChild(new SLNode(fireSparksRising, "Fire sparks rising node"));
    }

    return fireComplex;
}
//-----------------------------------------------------------------------------
SLNode* createTorchFire(SLAssetManager* am,
                        SLScene*        s,
                        SLbool          withLight,
                        SLGLTexture*    texFireCld,
                        SLGLTexture*    texFireFlm,
                        SLint           flipbookCols,
                        SLint           flipbookRows)
{

    SLNode* torchFire = new SLNode("Fire torch node");

    // Fire light node
    if (withLight)
    {
        SLLightSpot* light1 = new SLLightSpot(am, s, 0.1f, 180.0f, false);
        light1->name("Fire light node");
        light1->translate(0, 0, 0);
        light1->diffuseColor(SLCol4f(1, 0.4f, 0.0f));
        light1->diffusePower(2);
        light1->attenuation(0, 0, 1);
        torchFire->addChild(light1);
    }

    // Fire glow mesh
    {
        SLParticleSystem* fireGlow = new SLParticleSystem(am,
                                                          40,
                                                          SLVec3f(-0.1f, 0.0f, -0.1f),
                                                          SLVec3f(0.1f, 0.0f, 0.1f),
                                                          1.5f,
                                                          texFireCld,
                                                          "Torch Glow PS",
                                                          texFireFlm);
        fireGlow->color(SLCol4f(0.9f, 0.5f, 0, 0.63f));
        fireGlow->doBlendBrightness(true);
        fireGlow->radiusW(0.15f);
        fireGlow->radiusH(0.15f);
        fireGlow->doSizeOverLT(false);
        SLNode* fireGlowNode = new SLNode(fireGlow, "Torch Glow Node");
        fireGlowNode->translate(0, -0.4f, 0);
        torchFire->addChild(fireGlowNode);
    }

    // Fire torches
    {
        SLParticleSystem* torchFlame = new SLParticleSystem(am,
                                                            1,
                                                            SLVec3f(0.0f, 0.0f, 0.0f),
                                                            SLVec3f(0.0f, 0.0f, 0.0f),
                                                            4.0f,
                                                            texFireCld,
                                                            "Torch Flame PS",
                                                            texFireFlm);
        torchFlame->flipbookColumns(flipbookCols);
        torchFlame->flipbookRows(flipbookRows);
        torchFlame->doFlipBookTexture(true);
        torchFlame->doCounterGap(false); // We don't want to have flickering
        torchFlame->changeTexture();     // Switch texture, need to be done, to have flipbook texture as active
        torchFlame->doAlphaOverLT(false);
        torchFlame->doSizeOverLT(false);
        torchFlame->doRotation(false);
        torchFlame->doColor(false);
        torchFlame->frameRateFB(64);
        torchFlame->radiusW(0.3f);
        torchFlame->radiusH(0.8f);
        torchFlame->billboardType(BT_Vertical);
        SLNode* torchFlameNode = new SLNode(torchFlame, "Torch Flame Node");
        torchFlameNode->translate(0, 0.3f, 0);
        torchFire->addChild(torchFlameNode);
    }

    return torchFire;
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
void appDemoLoadScene(SLAssetManager* am,
                      SLScene*        s,
                      SLSceneView*    sv,
                      SLSceneID       sceneID)
{
    PROFILE_FUNCTION();

    SLfloat startLoadMS = GlobalTimer::timeMS();

    CVCapture::instance()->videoType(VT_NONE); // turn off any video

    // Reset non CVTracked and CVCapture infos
    CVTracked::resetTimes(); // delete all tracker times
    delete tracker;
    tracker = nullptr;

    // Reset asset pointer from previous scenes
    videoTexture = nullptr; // The video texture will be deleted by scene uninit
    trackedNode  = nullptr; // The tracked node will be deleted by scene uninit

    if (sceneID != SID_VolumeRayCastLighted)
        gTexMRI3D = nullptr; // The 3D MRI texture will be deleted by scene uninit

    // reset existing sceneviews
    for (auto* sceneview : AppDemo::sceneViews)
        sceneview->unInit();

    // Clear all data in the asset manager
    am->clear();

    AppDemo::sceneID   = sceneID;
    SLGLState* stateGL = SLGLState::instance();

    SLstring texPath    = AppDemo::texturePath;
    SLstring dataPath   = AppDemo::dataPath;
    SLstring modelPath  = AppDemo::modelPath;
    SLstring shaderPath = AppDemo::shaderPath;
    SLstring configPath = AppDemo::configPath;

    // Initialize all preloaded stuff from SLScene
    s->init(am);

    // Make sure the sv->camera doesn't
    sv->camera(sv->sceneViewCamera());

    // Clear the visible materials from the last scene
    sv->visibleMaterials2D().clear();
    sv->visibleMaterials3D().clear();

    // clear gui stuff that depends on scene and sceneview
    AppDemoGui::clear();

    // Deactivate in general the device sensors
    AppDemo::devRot.init();
    AppDemo::devLoc.init();

#ifdef SL_USE_ENTITIES_DEBUG
    SLScene::entities.dump(true);
#endif

    if (sceneID == SID_Empty) //...................................................................
    {
        s->name("No Scene loaded.");
        s->info("No Scene loaded.");
        s->root3D(nullptr);
        sv->sceneViewCamera()->background().colors(SLCol4f(0.7f, 0.7f, 0.7f),
                                                   SLCol4f(0.2f, 0.2f, 0.2f));
        sv->camera(nullptr);
        sv->doWaitOnIdle(true);
    }
    else if (sceneID == SID_Minimal) //............................................................
    {
        // Set scene name and info string
        s->name("Minimal Scene Test");
        s->info("Minimal scene with a texture mapped rectangle with a point light source.\n"
                "You can find all other test scenes in the menu File > Load Test Scenes."
                "You can jump to the next scene with the Shift-Alt-CursorRight.\n"
                "You can open various info windows under the menu Infos. You can drag, dock and stack them on all sides.\n"
                "You can rotate the scene with click and drag on the left mouse button (LMB).\n"
                "You can zoom in/out with the mousewheel. You can pan with click and drag on the middle mouse button (MMB).\n");

        // Create a scene group node
        SLNode* scene = new SLNode("scene node");
        s->root3D(scene);

        // Create textures and materials
        SLGLTexture* texC = new SLGLTexture(am, texPath + "earth2048_C.png");
        SLMaterial*  m1   = new SLMaterial(am, "m1", texC);

        // Create a light source node
        SLLightSpot* light1 = new SLLightSpot(am, s, 0.3f);
        light1->translation(0, 0, 5);
        light1->name("light node");
        scene->addChild(light1);

        // Create meshes and nodes
        SLMesh* rectMesh = new SLRectangle(am,
                                           SLVec2f(-5, -5),
                                           SLVec2f(5, 5),
                                           25,
                                           25,
                                           "rectangle mesh",
                                           m1);
        SLNode* rectNode = new SLNode(rectMesh, "rectangle node");
        scene->addChild(rectNode);

        // Set background color and the root scene node
        sv->sceneViewCamera()->background().colors(SLCol4f(0.7f, 0.7f, 0.7f),
                                                   SLCol4f(0.2f, 0.2f, 0.2f));
        // Save energy
        sv->doWaitOnIdle(true);
    }
    else if (sceneID == SID_Figure) //.............................................................
    {
        s->name("Hierarchical Figure Test");
        s->info("Hierarchical scenegraph with multiple subgroups in the figure. "
                "The goal is design a figure with hierarchical transforms containing only rotations and translations. "
                "You can see the hierarchy better in the Scenegraph window. In there the nodes are white and the meshes yellow. "
                "You can view the axis aligned bounding boxes with key B and the nodes origin and axis with key X.");

        // Create textures and materials
        SLMaterial* m1 = new SLMaterial(am, "m1", SLCol4f::BLACK, SLCol4f::WHITE, 128, 0.2f, 0.8f, 1.5f);
        SLMaterial* m2 = new SLMaterial(am, "m2", SLCol4f::WHITE * 0.3f, SLCol4f::WHITE, 128, 0.5f, 0.0f, 1.0f);

        SLuint  res       = 20;
        SLMesh* rectangle = new SLRectangle(am, SLVec2f(-5, -5), SLVec2f(5, 5), res, res, "rectangle", m2);
        SLNode* floorRect = new SLNode(rectangle);
        floorRect->rotate(90, -1, 0, 0);
        floorRect->translate(0, 0, -5.5f);

        SLCamera* cam1 = new SLCamera("Camera 1");
        cam1->translation(-7, 2, 7);
        cam1->lookAt(0, -2, 0);
        cam1->focalDist(10);
        cam1->setInitialState();
        cam1->devRotLoc(&AppDemo::devRot, &AppDemo::devLoc);
        cam1->background().colors(SLCol4f(0.7f, 0.6f, 1.0f),
                                  SLCol4f(0.1f, 0.4f, 0.8f));

        SLLightSpot* light1 = new SLLightSpot(am, s, 5, 0, 5, 0.5f);
        light1->powers(0.2f, 0.9f, 0.9f);
        light1->attenuation(1, 0, 0);

        SLNode* figure = BuildFigureGroup(am, s, m1, true);

        SLNode* scene = new SLNode("scene node");
        s->root3D(scene);
        scene->addChild(light1);
        scene->addChild(cam1);
        scene->addChild(floorRect);
        scene->addChild(figure);

        sv->camera(cam1);
    }
    else if (sceneID == SID_MeshLoad) //...........................................................
    {
        s->name("Mesh 3D Loader Test");
        s->info("We use the assimp library to load 3D file formats including materials, skeletons and animations. "
                "You can view the skeleton with key K. You can stop all animations with SPACE key.\n"
                "Switch between perspective and orthographic projection with key 5. "
                "Switch to front view with key 1, to side view with key 3 and to top view with key 7.\n"
                "Try the different stereo rendering modes in the menu Camera.");

        SLMaterial* matBlu = new SLMaterial(am, "Blue", SLCol4f(0, 0, 0.2f), SLCol4f(1, 1, 1), 100, 0.8f, 0);
        SLMaterial* matRed = new SLMaterial(am, "Red", SLCol4f(0.2f, 0, 0), SLCol4f(1, 1, 1), 100, 0.8f, 0);
        SLMaterial* matGre = new SLMaterial(am, "Green", SLCol4f(0, 0.2f, 0), SLCol4f(1, 1, 1), 100, 0.8f, 0);
        SLMaterial* matGra = new SLMaterial(am, "Gray", SLCol4f(0.3f, 0.3f, 0.3f), SLCol4f(1, 1, 1), 100, 0, 0);

        SLCamera* cam1 = new SLCamera("Camera 1");
        cam1->clipNear(.1f);
        cam1->clipFar(30);
        cam1->translation(0, 0, 12);
        cam1->lookAt(0, 0, 0);
        cam1->focalDist(12);
        cam1->stereoEyeSeparation(cam1->focalDist() / 30.0f);
        cam1->background().colors(SLCol4f(0.6f, 0.6f, 0.6f), SLCol4f(0.3f, 0.3f, 0.3f));
        cam1->setInitialState();
        cam1->devRotLoc(&AppDemo::devRot, &AppDemo::devLoc);

        SLLightSpot* light1 = new SLLightSpot(am, s, 2.5f, 2.5f, 2.5f, 0.2f);
        light1->powers(0.1f, 1.0f, 1.0f);
        light1->attenuation(1, 0, 0);
        SLAnimation* anim = s->animManager().createNodeAnimation("anim_light1_backforth", 2.0f, true, EC_inOutQuad, AL_pingPongLoop);
        anim->createNodeAnimTrackForTranslation(light1, SLVec3f(0.0f, 0.0f, -5.0f));

        SLLightSpot* light2 = new SLLightSpot(am, s, -2.5f, -2.5f, 2.5f, 0.2f);
        light2->powers(0.1f, 1.0f, 1.0f);
        light2->attenuation(1, 0, 0);
        anim = s->animManager().createNodeAnimation("anim_light2_updown", 2.0f, true, EC_inOutQuint, AL_pingPongLoop);
        anim->createNodeAnimTrackForTranslation(light2, SLVec3f(0.0f, 5.0f, 0.0f));

        SLAssimpImporter importer;

        SLNode* mesh3DS = importer.load(s->animManager(), am, modelPath + "3DS/Halloween/jackolan.3ds", texPath);
        SLNode* meshFBX = importer.load(s->animManager(), am, modelPath + "FBX/Duck/duck.fbx", texPath);
        SLNode* meshDAE = importer.load(s->animManager(), am, modelPath + "DAE/AstroBoy/AstroBoy.dae", texPath);

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
        rb          = new SLNode(new SLRectangle(am, SLVec2f(-b, -b), SLVec2f(b, b), res, res, "rectB", matBlu), "rectBNode");
        rb->translate(0, 0, -b, TS_object);
        rl = new SLNode(new SLRectangle(am, SLVec2f(-b, -b), SLVec2f(b, b), res, res, "rectL", matRed), "rectLNode");
        rl->rotate(90, 0, 1, 0);
        rl->translate(0, 0, -b, TS_object);
        rr = new SLNode(new SLRectangle(am, SLVec2f(-b, -b), SLVec2f(b, b), res, res, "rectR", matGre), "rectRNode");
        rr->rotate(-90, 0, 1, 0);
        rr->translate(0, 0, -b, TS_object);
        rf = new SLNode(new SLRectangle(am, SLVec2f(-b, -b), SLVec2f(b, b), res, res, "rectF", matGra), "rectFNode");
        rf->rotate(-90, 1, 0, 0);
        rf->translate(0, 0, -b, TS_object);
        rt = new SLNode(new SLRectangle(am, SLVec2f(-b, -b), SLVec2f(b, b), res, res, "rectT", matGra), "rectTNode");
        rt->rotate(90, 1, 0, 0);
        rt->translate(0, 0, -b, TS_object);

        SLNode* scene = new SLNode("Scene");
        s->root3D(scene);
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
    }
    else if (sceneID == SID_Revolver) //...........................................................
    {
        s->name("Revolving Mesh Test");
        s->info("Examples of revolving mesh objects constructed by rotating a 2D curve. "
                "The glass shader reflects and refracts the environment map. "
                "Try ray tracing with key R and come back with the ESC key.");

        // Test map material
        SLGLTexture* tex1C = new SLGLTexture(am, texPath + "Testmap_1024_C.jpg");
        SLGLTexture* tex1N = new SLGLTexture(am, texPath + "Testmap_1024_N.jpg");
        SLMaterial*  mat1  = new SLMaterial(am, "mat1", tex1C, tex1N);

        // floor material
        SLGLTexture* tex2 = new SLGLTexture(am, texPath + "wood0_0512_C.jpg");
        SLMaterial*  mat2 = new SLMaterial(am, "mat2", tex2);
        mat2->specular(SLCol4f::BLACK);

        // Back wall material
        SLGLTexture* tex3 = new SLGLTexture(am, texPath + "bricks1_0256_C.jpg");
        SLMaterial*  mat3 = new SLMaterial(am, "mat3", tex3);
        mat3->specular(SLCol4f::BLACK);

        // Left wall material
        SLGLTexture* tex4 = new SLGLTexture(am, texPath + "wood2_0512_C.jpg");
        SLMaterial*  mat4 = new SLMaterial(am, "mat4", tex4);
        mat4->specular(SLCol4f::BLACK);

        // Glass material
        SLGLTexture* tex5 = new SLGLTexture(am,
                                            texPath + "wood2_0256_C.jpg",
                                            texPath + "wood2_0256_C.jpg",
                                            texPath + "gray_0256_C.jpg",
                                            texPath + "wood0_0256_C.jpg",
                                            texPath + "gray_0256_C.jpg",
                                            texPath + "bricks1_0256_C.jpg");
        SLMaterial*  mat5 = new SLMaterial(am, "glass", SLCol4f::BLACK, SLCol4f::WHITE, 255, 0.1f, 0.9f, 1.5f);
        mat5->addTexture(tex5);
        SLGLProgram* sp1 = new SLGLProgramGeneric(am, shaderPath + "RefractReflect.vert", shaderPath + "RefractReflect.frag");
        mat5->program(sp1);

        // Wine material
        SLMaterial* mat6 = new SLMaterial(am, "wine", SLCol4f(0.4f, 0.0f, 0.2f), SLCol4f::BLACK, 255, 0.2f, 0.7f, 1.3f);
        mat6->addTexture(tex5);
        mat6->program(sp1);

        // camera
        SLCamera* cam1 = new SLCamera();
        cam1->name("cam1");
        cam1->translation(0, 1, 17);
        cam1->lookAt(0, 1, 0);
        cam1->focalDist(17);
        cam1->background().colors(SLCol4f(0.7f, 0.7f, 0.7f), SLCol4f(0.2f, 0.2f, 0.2f));
        cam1->setInitialState();
        cam1->devRotLoc(&AppDemo::devRot, &AppDemo::devLoc);

        // light
        SLLightSpot* light1 = new SLLightSpot(am, s, 0, 4, 0, 0.3f);
        light1->powers(0.2f, 1.0f, 1.0f);
        light1->attenuation(1, 0, 0);
        SLAnimation* anim = s->animManager().createNodeAnimation("light1_anim", 4.0f);
        anim->createNodeAnimTrackForEllipse(light1, 6.0f, A_z, 6.0f, A_x);

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
        SLNode* glass = new SLNode(new SLRevolver(am, revG, SLVec3f(0, 1, 0), res, true, false, "GlassRev", mat5));
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
        SLMesh* wineMesh = new SLRevolver(am, revW, SLVec3f(0, 1, 0), res, true, false, "WineRev", mat6);
        wineMesh->matOut(mat5);
        SLNode* wine = new SLNode(wineMesh);
        wine->translate(0.0f, -3.5f, 0.0f, TS_object);

        // wine fluid top
        SLNode* wineTop = new SLNode(new SLDisk(am, 2.05f, -SLVec3f::AXISY, res, false, "WineRevTop", mat6));
        wineTop->translate(0.0f, 2.5f, 0.0f, TS_object);

        // Other revolver objects
        SLNode* sphere = new SLNode(new SLSphere(am, 1, 16, 16, "sphere", mat1));
        sphere->translate(3, 0, 0, TS_object);
        SLNode* cylinder = new SLNode(new SLCylinder(am, 0.1f, 7, 3, 16, true, true, "cylinder", mat1));
        cylinder->translate(0, 0.5f, 0);
        cylinder->rotate(90, -1, 0, 0);
        cylinder->rotate(30, 0, 1, 0);
        SLNode* cone = new SLNode(new SLCone(am, 1, 3, 3, 16, true, "cone", mat1));
        cone->translate(-3, -1, 0, TS_object);
        cone->rotate(90, -1, 0, 0);

        // Cube dimensions
        SLfloat pL = -9.0f, pR = 9.0f;  // left/right
        SLfloat pB = -3.5f, pT = 14.5f; // bottom/top
        SLfloat pN = 9.0f, pF = -9.0f;  // near/far

        // bottom rectangle
        SLNode* b = new SLNode(new SLRectangle(am, SLVec2f(pL, -pN), SLVec2f(pR, -pF), 10, 10, "PolygonFloor", mat2));
        b->rotate(90, -1, 0, 0);
        b->translate(0, 0, pB, TS_object);

        // top rectangle
        SLNode* t = new SLNode(new SLRectangle(am, SLVec2f(pL, pF), SLVec2f(pR, pN), 10, 10, "top", mat2));
        t->rotate(90, 1, 0, 0);
        t->translate(0, 0, -pT, TS_object);

        // far rectangle
        SLNode* f = new SLNode(new SLRectangle(am, SLVec2f(pL, pB), SLVec2f(pR, pT), 10, 10, "far", mat3));
        f->translate(0, 0, pF, TS_object);

        // left rectangle
        SLNode* l = new SLNode(new SLRectangle(am, SLVec2f(-pN, pB), SLVec2f(-pF, pT), 10, 10, "left", mat4));
        l->rotate(90, 0, 1, 0);
        l->translate(0, 0, pL, TS_object);

        // right rectangle
        SLNode* r = new SLNode(new SLRectangle(am, SLVec2f(pF, pB), SLVec2f(pN, pT), 10, 10, "right", mat4));
        r->rotate(90, 0, -1, 0);
        r->translate(0, 0, -pR, TS_object);

        SLNode* scene = new SLNode;
        s->root3D(scene);
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
    }
    else if (sceneID == SID_TextureBlend) //.......................................................
    {
        s->name("Texture Blending Test");
        s->info("Texture map blending with depth sorting. Transparent tree rectangles in view "
                "frustum are rendered back to front. You can turn on/off alpha sorting in the "
                "menu Preferences of press key J.");

        SLGLTexture* t1 = new SLGLTexture(am,
                                          texPath + "tree1_1024_C.png",
                                          GL_LINEAR_MIPMAP_LINEAR,
                                          GL_LINEAR,
                                          TT_diffuse,
                                          GL_CLAMP_TO_EDGE,
                                          GL_CLAMP_TO_EDGE);
        SLGLTexture* t2 = new SLGLTexture(am,
                                          texPath + "grass0512_C.jpg",
                                          GL_LINEAR_MIPMAP_LINEAR,
                                          GL_LINEAR);

        SLMaterial* m1 = new SLMaterial(am, "m1", SLCol4f(1, 1, 1), SLCol4f(0, 0, 0), 100);
        SLMaterial* m2 = new SLMaterial(am, "m2", SLCol4f(1, 1, 1), SLCol4f(0, 0, 0), 100);

        SLGLProgram* sp = new SLGLProgramGeneric(am,
                                                 shaderPath + "PerVrtTm.vert",
                                                 shaderPath + "PerVrtTm.frag");
        m1->program(sp);
        m1->addTexture(t1);
        m2->addTexture(t2);

        SLCamera* cam1 = new SLCamera("Camera 1");
        cam1->translation(6.5f, 0.5f, -18);
        cam1->lookAt(0, 0, 0);
        cam1->focalDist(18);
        cam1->background().colors(SLCol4f(0.6f, 0.6f, 1));
        cam1->setInitialState();
        cam1->devRotLoc(&AppDemo::devRot, &AppDemo::devLoc);

        SLLightSpot* light = new SLLightSpot(am, s, 0.1f);
        light->translation(5, 5, 5);
        light->lookAt(0, 0, 0);
        light->attenuation(1, 0, 0);

        // Build arrays for polygon vertices and texture coordinates for tree
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
        SLNode* p1 = new SLNode(new SLPolygon(am, pNW, tNW, "Tree+X", m1));
        SLNode* p2 = new SLNode(new SLPolygon(am, pNW, tNW, "Tree-Z", m1));
        p2->rotate(90, 0, 1, 0);
        SLNode* p3 = new SLNode(new SLPolygon(am, pSE, tSE, "Tree-X", m1));
        SLNode* p4 = new SLNode(new SLPolygon(am, pSE, tSE, "Tree+Z", m1));
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
        s->root3D(scene);
        scene->addChild(light);
        scene->addChild(tree);
        scene->addChild(new SLNode(new SLPolygon(am, pG, tG, "Ground", m2)));

        // create 21*21*21-1 references around the center tree
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
                    t->rotate(Utils::random(0.f, 90.f), 0, 1, 0);
                    t->scale(Utils::random(0.5f, 1.0f));
                    scene->addChild(t);
                }
            }
        }

        scene->addChild(cam1);

        sv->camera(cam1);
    }
    else if (sceneID == SID_TextureFilter) //......................................................
    {
        s->name("Texture Filter Test");
        s->info("Texture minification filters: "
                "Bottom: nearest, left: linear, top: linear mipmap, right: anisotropic. "
                "The center sphere uses a 3D texture with linear filtering.");

        // Create 4 textures with different filter modes
        SLGLTexture* texB = new SLGLTexture(am, texPath + "brick0512_C.png", GL_NEAREST, GL_NEAREST);
        SLGLTexture* texL = new SLGLTexture(am, texPath + "brick0512_C.png", GL_LINEAR, GL_LINEAR);
        SLGLTexture* texT = new SLGLTexture(am, texPath + "brick0512_C.png", GL_LINEAR_MIPMAP_LINEAR, GL_LINEAR);
        SLGLTexture* texR = new SLGLTexture(am, texPath + "brick0512_C.png", SL_ANISOTROPY_MAX, GL_LINEAR);

        // define materials with textureOnly shader, no light needed
        SLMaterial* matB = new SLMaterial(am, "matB", texB, nullptr, nullptr, nullptr, SLGLProgramManager::get(SP_TextureOnly));
        SLMaterial* matL = new SLMaterial(am, "matL", texL, nullptr, nullptr, nullptr, SLGLProgramManager::get(SP_TextureOnly));
        SLMaterial* matT = new SLMaterial(am, "matT", texT, nullptr, nullptr, nullptr, SLGLProgramManager::get(SP_TextureOnly));
        SLMaterial* matR = new SLMaterial(am, "matR", texR, nullptr, nullptr, nullptr, SLGLProgramManager::get(SP_TextureOnly));

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
        SLNode* polyB = new SLNode(new SLPolygon(am, VB, T, "PolygonB", matB));

        SLVVec3f VL;
        VL.push_back(SLVec3f(-0.5f, 0.5f, 1.0f));
        VL.push_back(SLVec3f(-0.5f, -0.5f, 1.0f));
        VL.push_back(SLVec3f(-0.5f, -0.5f, -2.0f));
        VL.push_back(SLVec3f(-0.5f, 0.5f, -2.0f));
        SLNode* polyL = new SLNode(new SLPolygon(am, VL, T, "PolygonL", matL));

        SLVVec3f VT;
        VT.push_back(SLVec3f(0.5f, 0.5f, 1.0f));
        VT.push_back(SLVec3f(-0.5f, 0.5f, 1.0f));
        VT.push_back(SLVec3f(-0.5f, 0.5f, -2.0f));
        VT.push_back(SLVec3f(0.5f, 0.5f, -2.0f));
        SLNode* polyT = new SLNode(new SLPolygon(am, VT, T, "PolygonT", matT));

        SLVVec3f VR;
        VR.push_back(SLVec3f(0.5f, -0.5f, 1.0f));
        VR.push_back(SLVec3f(0.5f, 0.5f, 1.0f));
        VR.push_back(SLVec3f(0.5f, 0.5f, -2.0f));
        VR.push_back(SLVec3f(0.5f, -0.5f, -2.0f));
        SLNode* polyR = new SLNode(new SLPolygon(am, VR, T, "PolygonR", matR));

        // 3D Texture Mapping on a pyramid
        SLGLTexture* tex3D = new SLGLTexture(am, 256, texPath + "Wave_radial10_256C.jpg");
        SLGLProgram* spr3D = new SLGLProgramGeneric(am,
                                                    shaderPath + "TextureOnly3D.vert",
                                                    shaderPath + "TextureOnly3D.frag");
        SLMaterial*  mat3D = new SLMaterial(am, "mat3D", tex3D, nullptr, nullptr, nullptr, spr3D);

        // Create 3D textured pyramid mesh and node
        SLMesh* pyramid = new SLMesh(am, "Pyramid");
        pyramid->mat(mat3D);
        pyramid->P          = {{-1, -1, 1}, {1, -1, 1}, {1, -1, -1}, {-1, -1, -1}, {0, 2, 0}};
        pyramid->I16        = {0, 3, 1, 1, 3, 2, 4, 0, 1, 4, 1, 2, 4, 2, 3, 4, 3, 0};
        SLNode* pyramidNode = new SLNode(pyramid, "Pyramid");
        pyramidNode->scale(0.2f);
        pyramidNode->translate(0, 0, -3);

        // Create 3D textured sphere mesh and node
        SLNode*   sphere = new SLNode(new SLSphere(am, 0.2f, 16, 16, "Sphere", mat3D));
        SLCamera* cam1   = new SLCamera("Camera 1");
        cam1->translation(0, 0, 2.6f);
        cam1->lookAt(0, 0, 0);
        cam1->focalDist(2.2f);
        cam1->background().colors(SLCol4f(0.2f, 0.2f, 0.2f));
        cam1->setInitialState();
        cam1->devRotLoc(&AppDemo::devRot, &AppDemo::devLoc);

        SLNode* scene = new SLNode();
        s->root3D(scene);
        scene->addChild(polyB);
        scene->addChild(polyL);
        scene->addChild(polyT);
        scene->addChild(polyR);
        scene->addChild(sphere);
        scene->addChild(cam1);
        sv->camera(cam1);
    }
#ifdef SL_BUILD_WITH_KTX
    else if (sceneID == SID_TextureCompression) //.................................................
    {
        // Set scene name and info string
        s->name("Texture Compression Test Scene");
        s->info(s->name());

        // Create a scene group node
        SLNode* scene = new SLNode("scene node");
        s->root3D(scene);

        // Create a light source node
        SLLightSpot* light1 = new SLLightSpot(am, s, 0.1f);
        light1->translation(5, 5, 5);
        light1->name("light node");
        scene->addChild(light1);

        SLCamera* cam1 = new SLCamera("Camera 1");
        cam1->clipNear(0.1f);
        cam1->clipFar(100);
        cam1->translation(0, 0, 4.2f);
        cam1->lookAt(0, 0, 0);
        cam1->focalDist(4.2f);
        cam1->background().colors(SLCol4f(0.7f, 0.7f, 0.7f), SLCol4f(0.2f, 0.2f, 0.2f));
        cam1->setInitialState();
        scene->addChild(cam1);

        // Position for rectangle and uv out of earth texture
        SLVec2f pMin(-.5f, -.5f), pMax(.5f, .5f);
        SLVec2f tMin(.47f, .69f), tMax(.56f, .81f);
        SLint   minFlt = GL_LINEAR_MIPMAP_LINEAR;
        SLint   magFlt = GL_NEAREST;

        SLGLTexture* texPng      = new SLGLTexture(am, texPath + "earth2048_C.png", minFlt, magFlt);
        SLMaterial*  matPng      = new SLMaterial(am, "matPng", texPng);
        SLMesh*      rectMeshPng = new SLRectangle(am, pMin, pMax, tMin, tMax, 1, 1, "rectMeshPng", matPng);
        SLNode*      rectNodePng = new SLNode(rectMeshPng, "rectNodePng");
        rectNodePng->translate(-1.05f, 1.05f, 0);
        scene->addChild(rectNodePng);

        SLGLTexture* texJpgQ90      = new SLGLTexture(am, texPath + "earth2048_C_Q90.jpg", minFlt, magFlt);
        SLMaterial*  matJpgQ90      = new SLMaterial(am, "matJpgQ90", texJpgQ90);
        SLMesh*      rectMeshJpgQ90 = new SLRectangle(am, pMin, pMax, tMin, tMax, 1, 1, "rectMeshJpgQ90", matJpgQ90);
        SLNode*      rectNodeJpgQ90 = new SLNode(rectMeshJpgQ90, "rectNodeJpgQ90");
        rectNodeJpgQ90->translate(0, 1.05f, 0);
        scene->addChild(rectNodeJpgQ90);

        SLGLTexture* texJpgQ40      = new SLGLTexture(am, texPath + "earth2048_C_Q40.jpg", minFlt, magFlt);
        SLMaterial*  matJpgQ40      = new SLMaterial(am, "matJpgQ40", texJpgQ40);
        SLMesh*      rectMeshJpgQ40 = new SLRectangle(am, pMin, pMax, tMin, tMax, 1, 1, "rectMeshJpgQ40", matJpgQ40);
        SLNode*      rectNodeJpgQ40 = new SLNode(rectMeshJpgQ40, "rectNodeJpgQ40");
        rectNodeJpgQ40->translate(1.05f, 1.05f, 0);
        scene->addChild(rectNodeJpgQ40);

        /* Console commands to generate the following KTX files
        ./../../../externals/prebuilt/mac64_ktx_v4.0.0-beta7-cpvr/release/toktx --automipmap --linear --lower_left_maps_to_s0t0 --bcmp --clevel 4 --qlevel 255 earth2048_C_bcmp_Q255.ktx2 earth2048_C.png
        ./../../../externals/prebuilt/mac64_ktx_v4.0.0-beta7-cpvr/release/toktx --automipmap --linear --lower_left_maps_to_s0t0 --bcmp --clevel 4 --qlevel 128 earth2048_C_bcmp_Q128.ktx2 earth2048_C.png
        ./../../../externals/prebuilt/mac64_ktx_v4.0.0-beta7-cpvr/release/toktx --automipmap --linear --lower_left_maps_to_s0t0 --bcmp --clevel 4 --qlevel   1 earth2048_C_bcmp_Q001.ktx2 earth2048_C.png
        ./../../../externals/prebuilt/mac64_ktx_v4.0.0-beta7-cpvr/release/toktx --automipmap --linear --lower_left_maps_to_s0t0 --uastc 4 --zcmp 19 earth2048_C_uastc4.ktx2 earth2048_C.png
        ./../../../externals/prebuilt/mac64_ktx_v4.0.0-beta7-cpvr/release/toktx --automipmap --linear --lower_left_maps_to_s0t0 --uastc 2 --zcmp 19 earth2048_C_uastc2.ktx2 earth2048_C.png
        ./../../../externals/prebuilt/mac64_ktx_v4.0.0-beta7-cpvr/release/toktx --automipmap --linear --lower_left_maps_to_s0t0 --uastc 0 --zcmp 19 earth2048_C_uastc0.ktx2 earth2048_C.png
        */

        SLGLTexture* texKtxBcmp255      = new SLGLTexture(am, texPath + "earth2048_C_bcmp_Q255.ktx2", minFlt, magFlt);
        SLMaterial*  matKtxBcmp255      = new SLMaterial(am, "matKtxBcmp255", texKtxBcmp255);
        SLMesh*      rectMeshKtxBcmp255 = new SLRectangle(am, pMin, pMax, tMin, tMax, 1, 1, "rectMeshKtxBcmp255", matKtxBcmp255);
        SLNode*      rectNodeKtxBcmp255 = new SLNode(rectMeshKtxBcmp255, "rectNodeKtxBcmp255");
        rectNodeKtxBcmp255->translate(-1.05f, 0, 0);
        scene->addChild(rectNodeKtxBcmp255);

        SLGLTexture* texKtxBcmp128      = new SLGLTexture(am, texPath + "earth2048_C_bcmp_Q128.ktx2", minFlt, magFlt);
        SLMaterial*  matKtxBcmp128      = new SLMaterial(am, "matKtxBcmp128", texKtxBcmp128);
        SLMesh*      rectMeshKtxBcmp128 = new SLRectangle(am, pMin, pMax, tMin, tMax, 1, 1, "rectMeshKtxBcmp128", matKtxBcmp128);
        SLNode*      rectNodeKtxBcmp128 = new SLNode(rectMeshKtxBcmp128, "rectNodeKtxBcmp128");
        rectNodeKtxBcmp128->translate(0, 0, 0);
        scene->addChild(rectNodeKtxBcmp128);

        SLGLTexture* texKtxBcmp001      = new SLGLTexture(am, texPath + "earth2048_C_bcmp_Q001.ktx2", minFlt, magFlt);
        SLMaterial*  matKtxBcmp001      = new SLMaterial(am, "matKtxBcmp001", texKtxBcmp001);
        SLMesh*      rectMeshKtxBcmp001 = new SLRectangle(am, pMin, pMax, tMin, tMax, 1, 1, "rectMeshKtxBcmp001", matKtxBcmp001);
        SLNode*      rectNodeKtxBcmp001 = new SLNode(rectMeshKtxBcmp001, "rectNodeKtxBcmp001");
        rectNodeKtxBcmp001->translate(1.05f, 0, 0);
        scene->addChild(rectNodeKtxBcmp001);

        SLGLTexture* texKtxUastc4      = new SLGLTexture(am, texPath + "earth2048_C_uastc4.ktx2", minFlt, magFlt);
        SLMaterial*  matKtxUastc4      = new SLMaterial(am, "matKtxUastc4", texKtxUastc4);
        SLMesh*      rectMeshKtxUastc4 = new SLRectangle(am, pMin, pMax, tMin, tMax, 1, 1, "rectMeshKtxUastc4", matKtxUastc4);
        SLNode*      rectNodeKtxUastc4 = new SLNode(rectMeshKtxUastc4, "rectNodeKtxUastc4");
        rectNodeKtxUastc4->translate(1.05f, -1.05f, 0);
        scene->addChild(rectNodeKtxUastc4);

        SLGLTexture* texKtxUastc2      = new SLGLTexture(am, texPath + "earth2048_C_uastc2.ktx2", minFlt, magFlt);
        SLMaterial*  matKtxUastc2      = new SLMaterial(am, "matKtxUastc2", texKtxUastc2);
        SLMesh*      rectMeshKtxUastc2 = new SLRectangle(am, pMin, pMax, tMin, tMax, 1, 1, "rectMeshKtxUastc2", matKtxUastc2);
        SLNode*      rectNodeKtxUastc2 = new SLNode(rectMeshKtxUastc2, "rectNodeKtxUastc2");
        rectNodeKtxUastc2->translate(0, -1.05f, 0);
        scene->addChild(rectNodeKtxUastc2);

        SLGLTexture* texKtxUastc0      = new SLGLTexture(am, texPath + "earth2048_C_uastc0.ktx2", minFlt, magFlt);
        SLMaterial*  matKtxUastc0      = new SLMaterial(am, "matKtxUastc0", texKtxUastc0);
        SLMesh*      rectMeshKtxUastc0 = new SLRectangle(am, pMin, pMax, tMin, tMax, 1, 1, "rectMeshKtxUastc0", matKtxUastc0);
        SLNode*      rectNodeKtxUastc0 = new SLNode(rectMeshKtxUastc0, "rectNodeKtxUastc0");
        rectNodeKtxUastc0->translate(-1.05f, -1.05f, 0);
        scene->addChild(rectNodeKtxUastc0);

        // Add active camera
        sv->camera(cam1);

        // Save energy
        sv->doWaitOnIdle(true);
    }
#endif
    else if (sceneID == SID_FrustumCull) //........................................................
    {
        s->name("Frustum Culling Test");
        s->info("View frustum culling: Only objects in view are rendered. "
                "You can turn view frustum culling on/off in the menu Preferences or with the key F.");

        // create texture
        SLGLTexture* tex  = new SLGLTexture(am, texPath + "earth1024_C.jpg");
        SLMaterial*  mat1 = new SLMaterial(am, "mat1", tex);

        SLCamera* cam1 = new SLCamera("Camera 1");
        cam1->clipNear(0.1f);
        cam1->clipFar(100);
        cam1->translation(0, 0, 1);
        cam1->lookAt(0, 0, 0);
        cam1->focalDist(5);
        cam1->background().colors(SLCol4f(0.1f, 0.1f, 0.1f));
        cam1->setInitialState();
        cam1->devRotLoc(&AppDemo::devRot, &AppDemo::devLoc);

        SLLightSpot* light1 = new SLLightSpot(am, s, 10, 10, 10, 0.3f);
        light1->powers(0.2f, 0.8f, 1.0f);
        light1->attenuation(1, 0, 0);

        SLNode* scene = new SLNode;
        s->root3D(scene);
        scene->addChild(cam1);
        scene->addChild(light1);

        // add one single sphere in the center
        SLuint  res    = 16;
        SLNode* sphere = new SLNode(new SLSphere(am, 0.15f, res, res, "mySphere", mat1));
        scene->addChild(sphere);

        // create spheres around the center sphere
        SLint size = 20;
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
    }
    else if (sceneID == SID_2Dand3DText) //........................................................
    {
        s->name("2D & 3D Text Test");
        s->info("All 3D objects are in the _root3D scene and the center text is in the _root2D scene "
                "and rendered in orthographic projection in screen space.");

        SLMaterial* m1 = new SLMaterial(am, "m1", SLCol4f::RED);

        SLCamera* cam1 = new SLCamera("Camera 1");
        cam1->clipNear(0.1f);
        cam1->clipFar(100);
        cam1->translation(0, 0, 5);
        cam1->lookAt(0, 0, 0);
        cam1->focalDist(5);
        cam1->background().colors(SLCol4f(0.1f, 0.1f, 0.1f));
        cam1->setInitialState();
        cam1->devRotLoc(&AppDemo::devRot, &AppDemo::devLoc);

        SLLightSpot* light1 = new SLLightSpot(am, s, 10, 10, 10, 0.3f);
        light1->powers(0.2f, 0.8f, 1.0f);
        light1->attenuation(1, 0, 0);

        // Because all text objects get their sizes in pixels we have to scale them down
        SLfloat  scale = 0.01f;
        SLstring txt   = "This is text in 3D with font07";
        SLVec2f  size  = SLAssetManager::font07->calcTextSize(txt);
        SLNode*  t07   = new SLText(txt, SLAssetManager::font07);
        t07->translate(-size.x * 0.5f * scale, 1.0f, 0);
        t07->scale(scale);

        txt         = "This is text in 3D with font09";
        size        = SLAssetManager::font09->calcTextSize(txt);
        SLNode* t09 = new SLText(txt, SLAssetManager::font09);
        t09->translate(-size.x * 0.5f * scale, 0.8f, 0);
        t09->scale(scale);

        txt         = "This is text in 3D with font12";
        size        = SLAssetManager::font12->calcTextSize(txt);
        SLNode* t12 = new SLText(txt, SLAssetManager::font12);
        t12->translate(-size.x * 0.5f * scale, 0.6f, 0);
        t12->scale(scale);

        txt         = "This is text in 3D with font20";
        size        = SLAssetManager::font20->calcTextSize(txt);
        SLNode* t20 = new SLText(txt, SLAssetManager::font20);
        t20->translate(-size.x * 0.5f * scale, -0.8f, 0);
        t20->scale(scale);

        txt         = "This is text in 3D with font22";
        size        = SLAssetManager::font22->calcTextSize(txt);
        SLNode* t22 = new SLText(txt, SLAssetManager::font22);
        t22->translate(-size.x * 0.5f * scale, -1.2f, 0);
        t22->scale(scale);

        // Now create 2D text but don't scale it (all sizes in pixels)
        txt           = "This is text in 2D with font16";
        size          = SLAssetManager::font16->calcTextSize(txt);
        SLNode* t2D16 = new SLText(txt, SLAssetManager::font16);
        t2D16->translate(-size.x * 0.5f, 0, 0);

        // Assemble 3D scene as usual with camera and light
        SLNode* scene3D = new SLNode("root3D");
        s->root3D(scene3D);
        scene3D->addChild(cam1);
        scene3D->addChild(light1);
        scene3D->addChild(new SLNode(new SLSphere(am, 0.5f, 32, 32, "Sphere", m1)));
        scene3D->addChild(t07);
        scene3D->addChild(t09);
        scene3D->addChild(t12);
        scene3D->addChild(t20);
        scene3D->addChild(t22);

        // Assemble 2D scene
        SLNode* scene2D = new SLNode("root2D");
        s->root2D(scene2D);
        scene2D->addChild(t2D16);

        sv->camera(cam1);
        sv->doWaitOnIdle(true);
    }
    else if (sceneID == SID_PointClouds) //........................................................
    {
        s->name("Point Clouds Test");
        s->info("Point Clouds with normal and uniform distribution. "
                "You can select vertices with rectangle select (CTRL-LMB) "
                "and deselect selected with ALT-LMB.");

        SLCamera* cam1 = new SLCamera("Camera 1");
        cam1->clipNear(0.1f);
        cam1->clipFar(100);
        cam1->translation(0, 0, 15);
        cam1->lookAt(0, 0, 0);
        cam1->focalDist(15);
        cam1->background().colors(SLCol4f(0.1f, 0.1f, 0.1f));
        cam1->setInitialState();
        cam1->devRotLoc(&AppDemo::devRot, &AppDemo::devLoc);

        SLLightSpot* light1 = new SLLightSpot(am, s, 10, 10, 10, 0.3f);
        light1->powers(0.2f, 0.8f, 1.0f);
        light1->attenuation(1, 0, 0);

        SLMaterial* pcMat1 = new SLMaterial(am, "Red", SLCol4f::RED);
        pcMat1->program(new SLGLProgramGeneric(am,
                                               shaderPath + "ColorUniformPoint.vert",
                                               shaderPath + "Color.frag"));
        pcMat1->program()->addUniform1f(new SLGLUniform1f(UT_const, "u_pointSize", 4.0f));
        SLRnd3fNormal rndN(SLVec3f(0, 0, 0), SLVec3f(5, 2, 1));
        SLNode*       pc1 = new SLNode(new SLPoints(am, 1000, rndN, "PC1", pcMat1));
        pc1->translate(-5, 0, 0);

        SLMaterial* pcMat2 = new SLMaterial(am, "Green", SLCol4f::GREEN);
        pcMat2->program(new SLGLProgramGeneric(am,
                                               shaderPath + "ColorUniformPoint.vert",
                                               shaderPath + "Color.frag"));
        pcMat2->program()->addUniform1f(new SLGLUniform1f(UT_const, "u_pointSize", 1.0f));
        SLRnd3fUniform rndU(SLVec3f(0, 0, 0), SLVec3f(2, 3, 5));
        SLNode*        pc2 = new SLNode(new SLPoints(am, 1000, rndU, "PC2", pcMat2));
        pc2->translate(5, 0, 0);

        SLNode* scene = new SLNode("scene");
        s->root3D(scene);
        scene->addChild(cam1);
        scene->addChild(light1);
        scene->addChild(pc1);
        scene->addChild(pc2);

        sv->camera(cam1);
        sv->doWaitOnIdle(false);
    }
    else if (sceneID == SID_ZFighting) //..........................................................
    {
        // Set scene name and info string
        s->name("Z-Fighting Test Scene");
        s->info("The reason for this depth fighting is that the camera's near clipping distance "
                "is almost zero and the far clipping distance is too large. The depth buffer only "
                "has 32-bit precision, which leads to this fighting effect when the distance "
                "between the near and far clipping planes is too large. You can adjust these "
                "values over the menu Camera > Projection");

        // Create a scene group node
        SLNode* scene = new SLNode("scene node");
        s->root3D(scene);

        SLCamera* cam1 = new SLCamera("Camera 1");
        cam1->clipNear(0.0001f);
        cam1->clipFar(1000000);
        cam1->translation(0, 0, 4);
        cam1->lookAt(0, 0, 0);
        cam1->focalDist(4);
        cam1->background().colors(SLCol4f(0.7f, 0.7f, 0.7f),
                                  SLCol4f(0.2f, 0.2f, 0.2f));
        cam1->setInitialState();
        scene->addChild(cam1);

        // Create materials
        SLMaterial* matR = new SLMaterial(am, "matR", SLCol4f::RED);
        SLMaterial* matG = new SLMaterial(am, "matG", SLCol4f::GREEN);

        // Create a light source node
        SLLightSpot* light1 = new SLLightSpot(am, s, 0.3f);
        light1->translation(5, 0, 5);
        light1->name("light node");
        scene->addChild(light1);

        // Create two squares
        SLMesh* rectMeshR = new SLRectangle(am,
                                            SLVec2f(-1, -1),
                                            SLVec2f(1, 1),
                                            1,
                                            1,
                                            "RectR",
                                            matR);
        SLNode* rectNodeR = new SLNode(rectMeshR, "Rect Node Red");
        scene->addChild(rectNodeR);

        SLMesh* rectMeshG = new SLRectangle(am,
                                            SLVec2f(-0.8f, -0.8f),
                                            SLVec2f(0.8f, 0.8f),
                                            1,
                                            1,
                                            "RectG",
                                            matG);
        SLNode* rectNodeG = new SLNode(rectMeshG, "Rect Node Green");
        rectNodeG->rotate(2.0f, 1, 1, 0);
        scene->addChild(rectNodeG);

        // Save energy
        sv->doWaitOnIdle(true);

        sv->camera(cam1);
    }

    else if (sceneID == SID_ShaderPerPixelBlinn ||
             sceneID == SID_ShaderPerVertexBlinn) //...............................................
    {
        SLMaterial*  mL   = nullptr;
        SLMaterial*  mM   = nullptr;
        SLMaterial*  mR   = nullptr;
        SLGLTexture* texC = new SLGLTexture(am, texPath + "earth2048_C_Q95.jpg"); // color map

        if (sceneID == SID_ShaderPerVertexBlinn)
        {
            s->name("Blinn-Phong per vertex lighting");
            s->info("Per-vertex lighting with Blinn-Phong reflection model. "
                    "The reflection of 5 light sources is calculated per vertex. "
                    "The green and the white light are attached to the camera, the others are in the scene. "
                    "The light calculation per vertex is the fastest but leads to artefacts with spot lights");
            SLGLProgram* perVrtTm = new SLGLProgramGeneric(am,
                                                           shaderPath + "PerVrtBlinnTm.vert",
                                                           shaderPath + "PerVrtBlinnTm.frag");
            SLGLProgram* perVrt   = new SLGLProgramGeneric(am,
                                                         shaderPath + "PerVrtBlinn.vert",
                                                         shaderPath + "PerVrtBlinn.frag");
            mL                    = new SLMaterial(am, "mL", texC, nullptr, nullptr, nullptr, perVrtTm);
            mM                    = new SLMaterial(am, "mM", perVrt);
            mR                    = new SLMaterial(am, "mR", texC, nullptr, nullptr, nullptr, perVrtTm);
        }
        else
        {
            s->name("Blinn-Phong per pixel lighting");
            s->info("Per-pixel lighting with Blinn-Phong reflection model. "
                    "The reflection of 5 light sources is calculated per pixel. "
                    "The light calculation is done in the fragment shader.");
            SLGLTexture*   texN   = new SLGLTexture(am, texPath + "earth2048_N.jpg"); // normal map
            SLGLTexture*   texH   = new SLGLTexture(am, texPath + "earth2048_H.jpg"); // height map
            SLGLProgram*   pR     = new SLGLProgramGeneric(am,
                                                     shaderPath + "PerPixBlinnTmNm.vert",
                                                     shaderPath + "PerPixBlinnTmPm.frag");
            SLGLUniform1f* scale  = new SLGLUniform1f(UT_const, "u_scale", 0.02f, 0.002f, 0, 1);
            SLGLUniform1f* offset = new SLGLUniform1f(UT_const, "u_offset", -0.02f, 0.002f, -1, 1);
            pR->addUniform1f(scale);
            pR->addUniform1f(offset);
            mL = new SLMaterial(am, "mL", texC);
            mM = new SLMaterial(am, "mM");
            mR = new SLMaterial(am, "mR", texC, texN, texH, nullptr, pR);
        }

        mM->shininess(500);

        // Base root group node for the scene
        SLNode* scene = new SLNode;
        s->root3D(scene);

        SLCamera* cam1 = new SLCamera("Camera 1");
        cam1->translation(0, 0, 7);
        cam1->lookAt(0, 0, 0);
        cam1->focalDist(7);
        cam1->background().colors(SLCol4f(0.1f, 0.1f, 0.1f));
        cam1->setInitialState();
        scene->addChild(cam1);

        // Define 5 light sources

        // A rectangular white light attached to the camera
        SLLightRect* lightW = new SLLightRect(am, s, 2.0f, 1.0f);
        lightW->ambiDiffPowers(0, 5);
        lightW->translation(0, 2.5f, 0);
        lightW->translation(0, 2.5f, -7);
        lightW->rotate(-90, 1, 0, 0);
        lightW->attenuation(0, 0, 1);
        cam1->addChild(lightW);

        // A red point light from the front attached in the scene
        SLLightSpot* lightR = new SLLightSpot(am, s, 0.1f);
        lightR->ambientColor(SLCol4f(0, 0, 0));
        lightR->diffuseColor(SLCol4f(1, 0, 0));
        lightR->specularColor(SLCol4f(1, 0, 0));
        lightR->translation(0, 0, 2);
        lightR->lookAt(0, 0, 0);
        lightR->attenuation(0, 0, 1);
        scene->addChild(lightR);

        // A green spot head light with 40 deg. spot angle from front right
        SLLightSpot* lightG = new SLLightSpot(am, s, 0.1f, 20, true);
        lightG->ambientColor(SLCol4f(0, 0, 0));
        lightG->diffuseColor(SLCol4f(0, 1, 0));
        lightG->specularColor(SLCol4f(0, 1, 0));
        lightG->translation(1.5f, 1, -5.5f);
        lightG->lookAt(0, 0, -7);
        lightG->attenuation(1, 0, 0);
        cam1->addChild(lightG);

        // A blue spot light with 40 deg. spot angle from front left
        SLLightSpot* lightB = new SLLightSpot(am, s, 0.1f, 20.0f, true);
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
        light3Anim->createNodeAnimTrackForTranslation(lightB, SLVec3f(0, -2, 0));
        scene->addChild(lightB);

        // A yellow directional light from the back-bottom
        // Do constant attenuation for directional lights since it is infinitely far away
        SLLightDirect* lightY = new SLLightDirect(am, s);
        lightY->ambientColor(SLCol4f(0, 0, 0));
        lightY->diffuseColor(SLCol4f(1, 1, 0));
        lightY->specularColor(SLCol4f(1, 1, 0));
        lightY->translation(-1.5f, -1.5f, 1.5f);
        lightY->lookAt(0, 0, 0);
        lightY->attenuation(1, 0, 0);
        scene->addChild(lightY);

        // Add some meshes to be lighted
        SLNode* sphereL = new SLNode(new SLSpheric(am, 1.0f, 0.0f, 180.0f, 36, 36, "Sphere", mL));
        sphereL->translate(-2, 0, 0);
        sphereL->rotate(90, -1, 0, 0);
        SLNode* sphereM = new SLNode(new SLSpheric(am, 1.0f, 0.0f, 180.0f, 36, 36, "Sphere", mM));
        SLNode* sphereR = new SLNode(new SLSpheric(am, 1.0f, 0.0f, 180.0f, 36, 36, "Sphere", mR));
        sphereR->translate(2, 0, 0);
        sphereR->rotate(90, -1, 0, 0);

        scene->addChild(sphereL);
        scene->addChild(sphereM);
        scene->addChild(sphereR);
        sv->camera(cam1);
    }
    else if (sceneID == SID_ShaderPerPixelCook) //.................................................
    {
        s->name("Cook-Torrance Shading");
        s->info("Cook-Torrance reflection model. Left-Right: roughness 0.05-1, Top-Down: metallic: 1-0. "
                "The center sphere has roughness and metallic encoded in textures. "
                "The reflection model produces a more physically based light reflection "
                "than the standard Blinn-Phong reflection model.");

        // Base root group node for the scene
        SLNode* scene = new SLNode;
        s->root3D(scene);

        SLCamera* cam1 = new SLCamera("Camera 1");
        cam1->translation(0, 0, 30);
        cam1->lookAt(0, 0, 0);
        cam1->background().colors(SLCol4f::BLACK);
        cam1->focalDist(30);
        cam1->setInitialState();
        cam1->devRotLoc(&AppDemo::devRot, &AppDemo::devLoc);
        scene->addChild(cam1);

        // Create spheres and materials with roughness & metallic values between 0 and 1
        const SLint nrRows  = 7;
        const SLint nrCols  = 7;
        SLfloat     spacing = 2.5f;
        SLfloat     maxX    = (float)(nrCols - 1) * spacing * 0.5f;
        SLfloat     maxY    = (float)(nrRows - 1) * spacing * 0.5f;
        SLfloat     deltaR  = 1.0f / (float)(nrRows - 1);
        SLfloat     deltaM  = 1.0f / (float)(nrCols - 1);

        SLGLProgram* sp = new SLGLProgramGeneric(am,
                                                 shaderPath + "PerPixCook.vert",
                                                 shaderPath + "PerPixCook.frag");

        SLGLProgram* spTex = new SLGLProgramGeneric(am,
                                                    shaderPath + "PerPixCookTm.vert",
                                                    shaderPath + "PerPixCookTm.frag");

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
                    mat[i] = new SLMaterial(am,
                                            "CookTorranceMatTex",
                                            nullptr,
                                            new SLGLTexture(am, texPath + "rusty-metal_2048_C.jpg"),
                                            new SLGLTexture(am, texPath + "rusty-metal_2048_N.jpg"),
                                            new SLGLTexture(am, texPath + "rusty-metal_2048_M.jpg"),
                                            new SLGLTexture(am, texPath + "rusty-metal_2048_R.jpg"),
                                            nullptr,
                                            spTex);
                }
                else
                {
                    // Cook-Torrance material without textures
                    mat[i] = new SLMaterial(am,
                                            "CookTorranceMat",
                                            nullptr,
                                            SLCol4f::RED * 0.5f,
                                            Utils::clamp((float)r * deltaR, 0.05f, 1.0f),
                                            (float)m * deltaM,
                                            sp);
                }

                SLNode* node = new SLNode(new SLSpheric(am, 1.0f, 0.0f, 180.0f, 32, 32, "Sphere", mat[i]));
                node->translate(x, y, 0);
                scene->addChild(node);
                x += spacing;
                i++;
            }
            y += spacing;
        }

        // Add 5 Lights: 2 point lights, 2 directional lights and 1 spotlight in the center.
        SLLight::gamma      = 2.2f;
        SLLightSpot* light1 = new SLLightSpot(am, s, -maxX, maxY, maxY, 0.2f, 180, 0, 1000, 1000);
        light1->attenuation(0, 0, 1);
        SLLightDirect* light2 = new SLLightDirect(am, s, maxX, maxY, maxY, 0.5f, 0, 10, 10);
        light2->lookAt(0, 0, 0);
        light2->attenuation(0, 0, 1);
        SLLightSpot* light3 = new SLLightSpot(am, s, 0, 0, maxY, 0.2f, 36, 0, 1000, 1000);
        light3->attenuation(0, 0, 1);
        SLLightDirect* light4 = new SLLightDirect(am, s, -maxX, -maxY, maxY, 0.5f, 0, 10, 10);
        light4->lookAt(0, 0, 0);
        light4->attenuation(0, 0, 1);
        SLLightSpot* light5 = new SLLightSpot(am, s, maxX, -maxY, maxY, 0.2f, 180, 0, 1000, 1000);
        light5->attenuation(0, 0, 1);
        scene->addChild(light1);
        scene->addChild(light2);
        scene->addChild(light3);
        scene->addChild(light4);
        scene->addChild(light5);
        sv->camera(cam1);
    }
    else if (sceneID == SID_ShaderIBL) //..........................................................
    {
        // Set scene name and info string
        s->name("Image Based Lighting");
        s->info("Image-based lighting from skybox using high dynamic range images. "
                "It uses the Cook-Torrance reflection model also to calculate the ambient light part from the surrounding HDR skybox.");

        // Create HDR CubeMap and get precalculated textures from it
        SLSkybox* skybox = new SLSkybox(am,
                                        shaderPath,
                                        texPath + "env_barce_rooftop.hdr",
                                        SLVec2i(256, 256),
                                        "HDR Skybox");

        // Create a scene group node
        SLNode* scene = new SLNode("scene node");
        s->root3D(scene);

        // Create camera and initialize its parameters
        SLCamera* cam1 = new SLCamera("Camera 1");
        cam1->translation(0, 0, 30);
        cam1->lookAt(0, 0, 0);
        cam1->background().colors(SLCol4f(0.2f, 0.2f, 0.2f));
        cam1->focalDist(30);
        cam1->setInitialState();
        scene->addChild(cam1);

        // Add directional light with a position that corresponds roughly to the sun direction
        SLLight::gamma        = 2.2f;
        SLLightDirect* light1 = new SLLightDirect(am, s, 4.0f, .3f, 2.0f, 0.5f, 0, 1, 1);
        light1->lookAt(0, 0, 0);
        light1->attenuation(1, 0, 0);
        light1->createsShadows(true);
        light1->createShadowMapAutoSize(cam1, SLVec2i(2048, 2048), 4);
        light1->shadowMap()->cascadesFactor(30.0);
        light1->doSmoothShadows(true);
        light1->castsShadows(false);
        light1->shadowMinBias(0.001f);
        light1->shadowMaxBias(0.003f);
        scene->addChild(light1);

        // Create spheres and materials with roughness & metallic values between 0 and 1
        const SLint nrRows  = 7;
        const SLint nrCols  = 7;
        SLfloat     spacing = 2.5f;
        SLfloat     maxX    = (float)((int)(nrCols / 2) * spacing);
        SLfloat     maxY    = (float)((int)(nrRows / 2) * spacing);
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
                    // and the prefiltered textures for IBL
                    mat[i] = new SLMaterial(am,
                                            "IBLMatTex",
                                            skybox,
                                            new SLGLTexture(am, texPath + "gold-scuffed_2048_C.png"),
                                            new SLGLTexture(am, texPath + "gold-scuffed_2048_N.png"),
                                            new SLGLTexture(am, texPath + "gold-scuffed_2048_M.png"),
                                            new SLGLTexture(am, texPath + "gold-scuffed_2048_R.png"),
                                            new SLGLTexture(am, texPath + "gold-scuffed_2048_A.png"));
                }
                else
                {
                    // Cook-Torrance material with IBL but without textures
                    mat[i] = new SLMaterial(am,
                                            "IBLMat",
                                            skybox,
                                            SLCol4f::WHITE * 0.5f,
                                            Utils::clamp((float)r * deltaR, 0.05f, 1.0f),
                                            (float)m * deltaM);
                }

                SLNode* node = new SLNode(new SLSpheric(am,
                                                        1.0f,
                                                        0.0f,
                                                        180.0f,
                                                        32,
                                                        32,
                                                        "Sphere",
                                                        mat[i]));
                node->translate(x, y, 0);
                scene->addChild(node);
                x += spacing;
                i++;
            }
            y += spacing;
        }

        sv->camera(cam1);
        s->skybox(skybox);

        // Save energy
        sv->doWaitOnIdle(true);
    }
    else if (sceneID == SID_ShaderPerVertexWave) //................................................
    {
        s->name("Wave Shader Test");
        s->info("Vertex Shader with wave displacement.");
        SL_LOG("Use H-Key to increment (decrement w. shift) the wave height.\n");

        SLCamera* cam1 = new SLCamera("Camera 1");
        cam1->translation(0, 3, 8);
        cam1->lookAt(0, 0, 0);
        cam1->focalDist(cam1->translationOS().length());
        cam1->background().colors(SLCol4f(0.1f, 0.4f, 0.8f));
        cam1->setInitialState();
        cam1->devRotLoc(&AppDemo::devRot, &AppDemo::devLoc);

        // Create generic shader program with 4 custom uniforms
        SLGLProgram*   sp  = new SLGLProgramGeneric(am, shaderPath + "Wave.vert", shaderPath + "Wave.frag");
        SLGLUniform1f* u_h = new SLGLUniform1f(UT_const, "u_h", 0.1f, 0.05f, 0.0f, 0.5f, (SLKey)'H');
        s->eventHandlers().push_back(u_h);
        sp->addUniform1f(u_h);
        sp->addUniform1f(new SLGLUniform1f(UT_inc, "u_t", 0.0f, 0.06f));
        sp->addUniform1f(new SLGLUniform1f(UT_const, "u_a", 2.5f));
        sp->addUniform1f(new SLGLUniform1f(UT_incDec, "u_b", 2.2f, 0.01f, 2.0f, 2.5f));

        // Create materials
        SLMaterial* matWater = new SLMaterial(am, "matWater", SLCol4f(0.45f, 0.65f, 0.70f), SLCol4f::WHITE, 300);
        matWater->program(sp);
        SLMaterial* matRed = new SLMaterial(am, "matRed", SLCol4f(1.00f, 0.00f, 0.00f));

        // water rectangle in the y=0 plane
        SLNode* wave = new SLNode(new SLRectangle(am, SLVec2f(-Utils::PI, -Utils::PI), SLVec2f(Utils::PI, Utils::PI), 40, 40, "WaterRect", matWater));
        wave->rotate(90, -1, 0, 0);

        SLLightSpot* light0 = new SLLightSpot(am, s);
        light0->ambiDiffPowers(0, 1);
        light0->translate(0, 4, -4, TS_object);
        light0->attenuation(1, 0, 0);

        SLNode* scene = new SLNode;
        s->root3D(scene);
        scene->addChild(light0);
        scene->addChild(wave);
        scene->addChild(new SLNode(new SLSphere(am, 1, 32, 32, "Red Sphere", matRed)));
        scene->addChild(cam1);

        sv->camera(cam1);
        sv->doWaitOnIdle(false);
    }
    else if (sceneID == SID_ShaderBumpNormal) //...................................................
    {
        s->name("Normal Map Test");
        s->info("Normal map bump mapping combined with a spot and a directional lighting.");

        // Create textures
        SLGLTexture* texC = new SLGLTexture(am, texPath + "brickwall0512_C.jpg");
        SLGLTexture* texN = new SLGLTexture(am, texPath + "brickwall0512_N.jpg");

        // Create materials
        SLMaterial* m1 = new SLMaterial(am, "m1", texC, texN);

        SLCamera* cam1 = new SLCamera("Camera 1");
        cam1->translation(-10, 10, 10);
        cam1->lookAt(0, 0, 0);
        cam1->focalDist(20);
        cam1->background().colors(SLCol4f(0.5f, 0.5f, 0.5f));
        cam1->setInitialState();

        SLLightSpot* light1 = new SLLightSpot(am, s, 0.3f, 40, true);
        light1->powers(0.1f, 1.0f, 1.0f);
        light1->attenuation(1, 0, 0);
        light1->translation(0, 0, 5);
        light1->lookAt(0, 0, 0);

        SLLightDirect* light2 = new SLLightDirect(am, s);
        light2->ambientColor(SLCol4f(0, 0, 0));
        light2->diffuseColor(SLCol4f(1, 1, 0));
        light2->specularColor(SLCol4f(1, 1, 0));
        light2->translation(-5, -5, 5);
        light2->lookAt(0, 0, 0);
        light2->attenuation(1, 0, 0);

        SLAnimation* anim = s->animManager().createNodeAnimation("light1_anim", 2.0f);
        anim->createNodeAnimTrackForEllipse(light1, 2.0f, A_x, 2.0f, A_Y);

        SLNode* scene = new SLNode;
        s->root3D(scene);
        scene->addChild(light1);
        scene->addChild(light2);
        scene->addChild(new SLNode(new SLRectangle(am, SLVec2f(-5, -5), SLVec2f(5, 5), 1, 1, "Rect", m1)));
        scene->addChild(cam1);

        sv->camera(cam1);
    }
    else if (sceneID == SID_ShaderBumpParallax) //.................................................
    {
        s->name("Parallax Map Test");
        s->info("Normal map parallax mapping with a spot and a directional light");
        SL_LOG("Demo application for parallax bump mapping.");
        SL_LOG("Use X-Key to increment (decrement w. shift) parallax scale.");
        SL_LOG("Use O-Key to increment (decrement w. shift) parallax offset.\n");

        // Create shader program with 4 uniforms
        SLGLProgram*   sp     = new SLGLProgramGeneric(am,
                                                 shaderPath + "PerPixBlinnTmNm.vert",
                                                 shaderPath + "PerPixBlinnTmPm.frag");
        SLGLUniform1f* scale  = new SLGLUniform1f(UT_const, "u_scale", 0.04f, 0.002f, 0, 1, (SLKey)'X');
        SLGLUniform1f* offset = new SLGLUniform1f(UT_const, "u_offset", -0.03f, 0.002f, -1, 1, (SLKey)'O');
        s->eventHandlers().push_back(scale);
        s->eventHandlers().push_back(offset);
        sp->addUniform1f(scale);
        sp->addUniform1f(offset);

        // Create textures
        SLGLTexture* texC = new SLGLTexture(am, texPath + "brickwall0512_C.jpg");
        SLGLTexture* texN = new SLGLTexture(am, texPath + "brickwall0512_N.jpg");
        SLGLTexture* texH = new SLGLTexture(am, texPath + "brickwall0512_H.jpg");

        // Create materials
        SLMaterial* m1 = new SLMaterial(am, "mat1", texC, texN, texH, nullptr, sp);

        SLCamera* cam1 = new SLCamera("Camera 1");
        cam1->translation(-10, 10, 10);
        cam1->lookAt(0, 0, 0);
        cam1->focalDist(20);
        cam1->background().colors(SLCol4f(0.5f, 0.5f, 0.5f));
        cam1->setInitialState();

        SLLightSpot* light1 = new SLLightSpot(am, s, 0.3f, 40, true);
        light1->powers(0.1f, 1.0f, 1.0f);
        light1->attenuation(1, 0, 0);
        light1->translation(0, 0, 5);
        light1->lookAt(0, 0, 0);

        SLLightDirect* light2 = new SLLightDirect(am, s);
        light2->ambientColor(SLCol4f(0, 0, 0));
        light2->diffuseColor(SLCol4f(1, 1, 0));
        light2->specularColor(SLCol4f(1, 1, 0));
        light2->translation(-5, -5, 5);
        light2->lookAt(0, 0, 0);
        light2->attenuation(1, 0, 0);

        SLAnimation* anim = s->animManager().createNodeAnimation("light1_anim", 2.0f);
        anim->createNodeAnimTrackForEllipse(light1, 2.0f, A_x, 2.0f, A_Y);

        SLNode* scene = new SLNode;
        s->root3D(scene);
        scene->addChild(light1);
        scene->addChild(light2);
        scene->addChild(new SLNode(new SLRectangle(am, SLVec2f(-5, -5), SLVec2f(5, 5), 1, 1, "Rect", m1)));
        scene->addChild(cam1);

        sv->camera(cam1);
    }
    else if (sceneID == SID_ShaderSkyBox) //.......................................................
    {
        // Set scene name and info string
        s->name("Sky Box Test");
        s->info("Sky box cube with cubemap skybox shader");

        // Create textures and materials
        SLSkybox* skybox = new SLSkybox(am,
                                        shaderPath,
                                        texPath + "Desert+X1024_C.jpg",
                                        texPath + "Desert-X1024_C.jpg",
                                        texPath + "Desert+Y1024_C.jpg",
                                        texPath + "Desert-Y1024_C.jpg",
                                        texPath + "Desert+Z1024_C.jpg",
                                        texPath + "Desert-Z1024_C.jpg");
        // Material for mirror
        SLMaterial* refl = new SLMaterial(am, "refl", SLCol4f::BLACK, SLCol4f::WHITE, 1000, 1.0f);
        refl->addTexture(skybox->environmentCubemap());
        refl->program(new SLGLProgramGeneric(am,
                                             shaderPath + "Reflect.vert",
                                             shaderPath + "Reflect.frag"));
        // Material for glass
        SLMaterial* refr = new SLMaterial(am, "refr", SLCol4f::BLACK, SLCol4f::BLACK, 100, 0.1f, 0.9f, 1.5f);
        refr->translucency(1000);
        refr->transmissive(SLCol4f::WHITE);
        refr->addTexture(skybox->environmentCubemap());
        refr->program(new SLGLProgramGeneric(am,
                                             shaderPath + "RefractReflect.vert",
                                             shaderPath + "RefractReflect.frag"));
        // Create a scene group node
        SLNode* scene = new SLNode("scene node");
        s->root3D(scene);

        // Create camera in the center
        SLCamera* cam1 = new SLCamera("Camera 1");
        cam1->translation(0, 0, 5);
        cam1->setInitialState();
        scene->addChild(cam1);

        // There is no light needed in this scene. All reflections come from cube maps
        // But ray tracing needs light sources
        // Create directional light for the sunlight
        SLLightDirect* light = new SLLightDirect(am, s, 0.5f);
        light->ambientColor(SLCol4f(0.3f, 0.3f, 0.3f));
        light->attenuation(1, 0, 0);
        light->translate(1, 1, -1);
        light->lookAt(-1, -1, 1);
        scene->addChild(light);

        // Center sphere
        SLNode* sphere = new SLNode(new SLSphere(am, 0.5f, 32, 32, "Sphere", refr));
        scene->addChild(sphere);

        // load teapot
        SLAssimpImporter importer;
        SLNode*          teapot = importer.load(s->animManager(),
                                       am,
                                       modelPath + "FBX/Teapot/Teapot.fbx",
                                       texPath,
                                       nullptr,
                                       false,
                                       true,
                                       refl);
        teapot->translate(-1.5f, -0.5f, 0);
        scene->addChild(teapot);

        // load Suzanne
        SLNode* suzanne = importer.load(s->animManager(),
                                        am,
                                        modelPath + "FBX/Suzanne/Suzanne.fbx",
                                        texPath,
                                        nullptr,
                                        false,
                                        true,
                                        refr);
        suzanne->translate(1.5f, -0.5f, 0);
        scene->addChild(suzanne);

        sv->camera(cam1);
        s->skybox(skybox);

        // Save energy
        sv->doWaitOnIdle(true);
    }
    else if (sceneID == SID_ShaderEarth) //........................................................
    {
        s->name("Earth Shader Test");
        s->info("Complex earth shader with 7 textures: day color, night color, normal, height & gloss map of earth, color & alpha-map of clouds");
        SL_LOG("Earth Shader from Markus Knecht");
        SL_LOG("Use (SHIFT) & key X to change scale of the parallax mapping");
        SL_LOG("Use (SHIFT) & key O to change offset of the parallax mapping");

        // Create shader program with 4 uniforms
        SLGLProgram*   sp     = new SLGLProgramGeneric(am,
                                                 shaderPath + "PerPixBlinnTmNm.vert",
                                                 shaderPath + "PerPixBlinnTmNmEarth.frag");
        SLGLUniform1f* scale  = new SLGLUniform1f(UT_const, "u_scale", 0.02f, 0.002f, 0, 1, (SLKey)'X');
        SLGLUniform1f* offset = new SLGLUniform1f(UT_const, "u_offset", -0.02f, 0.002f, -1, 1, (SLKey)'O');
        s->eventHandlers().push_back(scale);
        s->eventHandlers().push_back(offset);
        sp->addUniform1f(scale);
        sp->addUniform1f(offset);

        // Create textures
        SLGLTexture* texC   = new SLGLTexture(am, texPath + "earth2048_C.png");            // color map
        SLGLTexture* texN   = new SLGLTexture(am, texPath + "earth2048_N.jpg");            // normal map
        SLGLTexture* texH   = new SLGLTexture(am, texPath + "earth2048_H.jpg");            // height map
        SLGLTexture* texG   = new SLGLTexture(am, texPath + "earth2048_S.jpg");            // specular map
        SLGLTexture* texNC  = new SLGLTexture(am, texPath + "earthNight2048_C.jpg");       // night color  map
        SLGLTexture* texClC = new SLGLTexture(am, texPath + "earthCloud1024_alpha_C.png"); // cloud color map
        // SLGLTexture* texClA = new SLGLTexture(am, texPath + "earthCloud1024_A.jpg"); // cloud alpha map

        // Create materials
        SLMaterial* matEarth = new SLMaterial(am, "matEarth", texC, texN, texH, texG, sp);
        matEarth->addTexture(texClC);
        // matEarth->addTexture(texClA);
        matEarth->addTexture(texNC);
        matEarth->shininess(4000);
        matEarth->program(sp);

        SLCamera* cam1 = new SLCamera("Camera 1");
        cam1->translation(0, 0, 4);
        cam1->lookAt(0, 0, 0);
        cam1->focalDist(4);
        cam1->background().colors(SLCol4f(0, 0, 0));
        cam1->setInitialState();
        cam1->devRotLoc(&AppDemo::devRot, &AppDemo::devLoc);

        SLLightSpot* sun = new SLLightSpot(am, s);
        sun->powers(0.0f, 1.0f, 0.2f);
        sun->attenuation(1, 0, 0);

        SLAnimation* anim = s->animManager().createNodeAnimation("light1_anim", 24.0f);
        anim->createNodeAnimTrackForEllipse(sun, 50.0f, A_x, 50.0f, A_z);

        SLuint  res   = 30;
        SLNode* earth = new SLNode(new SLSphere(am, 1, res, res, "Earth", matEarth));
        earth->rotate(90, -1, 0, 0);

        SLNode* scene = new SLNode;
        s->root3D(scene);
        scene->addChild(sun);
        scene->addChild(earth);
        scene->addChild(cam1);

        sv->camera(cam1);
    }
    else if (sceneID == SID_ShadowMappingBasicScene) //............................................
    {
        s->name("Shadow Mapping Basic Scene");
        s->info("Shadow Mapping is a technique to render shadows.");

        SLMaterial* matPerPixSM = new SLMaterial(am, "m1");

        // Base root group node for the scene
        SLNode* scene = new SLNode;
        s->root3D(scene);

        SLCamera* cam1 = new SLCamera("Camera 1");
        cam1->translation(0, 7, 12);
        cam1->lookAt(0, 1, 0);
        cam1->focalDist(8);
        cam1->background().colors(SLCol4f(0.1f, 0.1f, 0.1f));
        cam1->setInitialState();
        scene->addChild(cam1);

        // Create light source
        // Do constant attenuation for directional lights since it is infinitely far away
        SLLightDirect* light = new SLLightDirect(am, s);

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
        SLNode* sphereNode = new SLNode(new SLSpheric(am, 1, 0, 180, 20, 20, "Sphere", matPerPixSM));
        sphereNode->translate(0, 2.0, 0);
        sphereNode->castsShadows(true);
        scene->addChild(sphereNode);

        SLAnimation* anim = s->animManager().createNodeAnimation("sphere_anim", 2.0f);
        anim->createNodeAnimTrackForEllipse(sphereNode, 0.5f, A_x, 0.5f, A_z);

        // Add a box which receives shadows
        SLNode* boxNode = new SLNode(new SLBox(am, -5, -1, -5, 5, 0, 5, "Box", matPerPixSM));
        boxNode->castsShadows(false);
        scene->addChild(boxNode);

        sv->camera(cam1);
    }
    else if (sceneID == SID_ShadowMappingLightTypes) //............................................
    {
        s->name("Shadow Mapping light types");
        s->info("Shadow Mapping is implemented for these light types.");

        SLMaterial* mat1 = new SLMaterial(am, "mat1");

        // Base root group node for the scene
        SLNode* scene = new SLNode;
        s->root3D(scene);

        SLCamera* cam1 = new SLCamera("Camera 1");
        cam1->translation(0, 2, 20);
        cam1->lookAt(0, 2, 0);
        cam1->focalDist(8);
        cam1->background().colors(SLCol4f(0.1f, 0.1f, 0.1f));
        cam1->setInitialState();
        scene->addChild(cam1);

        // Create light sources
        vector<SLLight*> lights = {
          new SLLightDirect(am, s),
          new SLLightRect(am, s),
          new SLLightSpot(am, s, 0.3f, 25.0f),
          new SLLightSpot(am, s, 0.1f, 180.0f)};

        for (SLint i = 0; i < lights.size(); ++i)
        {
            SLLight* light = lights[i];
            SLNode*  node  = dynamic_cast<SLNode*>(light);
            SLfloat  x     = ((float)i - ((SLfloat)lights.size() - 1.0f) / 2.0f) * 5;

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
        SLAnimation*     teapotAnim  = s->animManager().createNodeAnimation("teapot_anim",
                                                                       8.0f,
                                                                       true,
                                                                       EC_linear,
                                                                       AL_loop);
        SLNode*          teapotModel = importer.load(s->animManager(),
                                            am,
                                            modelPath + "FBX/Teapot/Teapot.fbx",
                                            texPath,
                                            nullptr,
                                            false,
                                            true,
                                            mat1);

        for (SLLight* light : lights)
        {
            SLNode* teapot = teapotModel->copyRec();

            teapot->translate(light->positionWS().x, 2, 0);
            teapot->children()[0]->castsShadows(true);
            scene->addChild(teapot);

            // Create animation
            SLNodeAnimTrack* track = teapotAnim->createNodeAnimTrack();
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
        SLNode* boxNode = new SLNode(new SLBox(am, minx, -1, -5, maxx, 0, 5, "Box", mat1));
        boxNode->castsShadows(false);
        scene->addChild(boxNode);

        sv->camera(cam1);
    }
    else if (sceneID == SID_ShadowMappingSpotLights) //............................................
    {
        s->name("Shadow Mapping for Spot lights");
        s->info("8 Spot lights use a perspective projection for their light space.");

        // Setup shadow mapping material
        SLMaterial* matPerPixSM = new SLMaterial(am, "m1"); //, SLCol4f::WHITE, SLCol4f::WHITE, 500, 0, 0, 1, progPerPixSM);

        // Base root group node for the scene
        SLNode* scene = new SLNode;
        s->root3D(scene);

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
            SLLightSpot* light = new SLLightSpot(am, s, 0.3f, 45.0f);
            SLCol4f      color;
            color.hsva2rgba(SLVec4f(Utils::TWOPI * (float)i / (float)SL_MAX_LIGHTS, 1.0f, 1.0f));
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
        SLNode* sphereNode = new SLNode(new SLSpheric(am, 1, 0, 180, 20, 20, "Sphere", matPerPixSM));
        sphereNode->translate(0, 2.0, 0);
        sphereNode->castsShadows(true);
        scene->addChild(sphereNode);

        SLAnimation* anim = s->animManager().createNodeAnimation("sphere_anim", 2.0f);
        anim->createNodeAnimTrackForEllipse(sphereNode, 1.0f, A_x, 1.0f, A_z);

        // Add a box which receives shadows
        SLNode* boxNode = new SLNode(new SLBox(am, -5, -1, -5, 5, 0, 5, "Box", matPerPixSM));
        boxNode->castsShadows(false);
        scene->addChild(boxNode);

        sv->camera(cam1);
    }
    else if (sceneID == SID_ShadowMappingPointLights) //...........................................
    {
        s->name("Shadow Mapping for point lights");
        s->info("Point lights use cubemaps to store shadow maps.");

        // Setup shadow mapping material
        SLMaterial* matPerPixSM = new SLMaterial(am, "m1"); //, SLCol4f::WHITE, SLCol4f::WHITE, 500, 0, 0, 1, progPerPixSM);

        // Base root group node for the scene
        SLNode* scene = new SLNode;
        s->root3D(scene);

        // Create camera
        SLCamera* cam1 = new SLCamera;
        cam1->translation(0, 0, 8);
        cam1->lookAt(0, 0, 0);
        cam1->fov(27);
        cam1->focalDist(cam1->translationOS().length());
        cam1->background().colors(SLCol4f(0.1f, 0.1f, 0.1f));
        cam1->setInitialState();
        cam1->devRotLoc(&AppDemo::devRot, &AppDemo::devLoc);

        // Create lights
        SLAnimation* anim = s->animManager().createNodeAnimation("light_anim", 4.0f);

        for (SLint i = 0; i < 3; ++i)
        {
            SLLightSpot* light = new SLLightSpot(am, s, 0.1f);
            light->powers(0.2f, 1.5f, 1.0f, SLCol4f(i == 0, i == 1, i == 2));
            light->attenuation(0, 0, 1);
            light->translate((float)i - 1.0f, (float)i - 1.0f, (float)i - 1.0f);
            light->createsShadows(true);
            light->createShadowMap();
            light->shadowMap()->rayCount(SLVec2i(16, 16));
            scene->addChild(light);
            anim->createNodeAnimTrackForEllipse(light, 0.2f, A_x, 0.2f, A_z);
        }

        // Create wall polygons
        SLfloat pL = -1.48f, pR = 1.48f; // left/right
        SLfloat pB = -1.25f, pT = 1.19f; // bottom/top
        SLfloat pN = 1.79f, pF = -1.55f; // near/far

        // Bottom plane
        SLNode* b = new SLNode(new SLRectangle(am, SLVec2f(pL, -pN), SLVec2f(pR, -pF), 6, 6, "bottom", matPerPixSM));
        b->rotate(90, -1, 0, 0);
        b->translate(0, 0, pB, TS_object);
        scene->addChild(b);

        // Top plane
        SLNode* t = new SLNode(new SLRectangle(am, SLVec2f(pL, pF), SLVec2f(pR, pN), 6, 6, "top", matPerPixSM));
        t->rotate(90, 1, 0, 0);
        t->translate(0, 0, -pT, TS_object);
        scene->addChild(t);

        // Far plane
        SLNode* f = new SLNode(new SLRectangle(am, SLVec2f(pL, pB), SLVec2f(pR, pT), 6, 6, "far", matPerPixSM));
        f->translate(0, 0, pF, TS_object);
        scene->addChild(f);

        // near plane
        SLNode* n = new SLNode(new SLRectangle(am, SLVec2f(pL, pT), SLVec2f(pR, pB), 6, 6, "near", matPerPixSM));
        n->translate(0, 0, pN, TS_object);
        scene->addChild(n);

        // left plane
        SLNode* l = new SLNode(new SLRectangle(am, SLVec2f(-pN, pB), SLVec2f(-pF, pT), 6, 6, "left", matPerPixSM));
        l->rotate(90, 0, 1, 0);
        l->translate(0, 0, pL, TS_object);
        scene->addChild(l);

        // Right plane
        SLNode* r = new SLNode(new SLRectangle(am, SLVec2f(pF, pB), SLVec2f(pN, pT), 6, 6, "right", matPerPixSM));
        r->rotate(90, 0, -1, 0);
        r->translate(0, 0, -pR, TS_object);
        scene->addChild(r);

        // Create cubes which cast shadows
        for (SLint i = 0; i < 64; ++i)
        {
            SLBox* box = new SLBox(am);
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
    }
    else if (sceneID == SID_ShadowMappingCascaded) //..............................................
    {
        s->name("Cascaded Shadow Mapping Test Scene");
        s->info("Cascaded Shadow Mapping uses several cascades of shadow maps to provide higher \
resolution shadows near the camera and lower resolution shadows further away.");

        // Setup shadow mapping material
        SLMaterial* matPerPixSM = new SLMaterial(am, "m1");

        // Base root group node for the scene
        SLNode* scene = new SLNode;
        s->root3D(scene);

        SLCamera* cam1 = new SLCamera("Camera 1");
        cam1->translation(0, 7, 12);
        cam1->lookAt(0, 1, 0);
        cam1->focalDist(8);
        cam1->background().colors(SLCol4f(0.1f, 0.1f, 0.1f));
        cam1->setInitialState();
        scene->addChild(cam1);

        // Create light source
        // Do constant attenuation for directional lights since it is infinitely far away
        SLLightDirect* light = new SLLightDirect(am, s);
        light->powers(0.0f, 1.0f, 1.0f);
        light->translation(0, 5, 0);
        light->lookAt(0, 0, 0);
        light->attenuation(1, 0, 0);
        light->createsShadows(true);
        light->doCascadedShadows(true);
        light->createShadowMapAutoSize(cam1);
        light->shadowMap()->rayCount(SLVec2i(16, 16));
        light->castsShadows(false);
        scene->addChild(light);

        // Add a sphere which casts shadows
        SLNode* sphereNode = new SLNode(new SLSpheric(am, 1, 0, 180, 20, 20, "Sphere", matPerPixSM));
        sphereNode->translate(0, 2.0, 0);
        sphereNode->castsShadows(true);
        scene->addChild(sphereNode);

        SLAnimation* anim = s->animManager().createNodeAnimation("sphere_anim", 2.0f);
        anim->createNodeAnimTrackForEllipse(sphereNode, 0.5f, A_x, 0.5f, A_z);

        // Add a box which receives shadows
        SLNode* boxNode = new SLNode(new SLBox(am, -5, -1, -5, 5, 0, 5, "Box", matPerPixSM));
        boxNode->castsShadows(false);
        scene->addChild(boxNode);

        sv->camera(cam1);
    }
    else if (sceneID >= SID_SuzannePerPixBlinn &&
             sceneID <= SID_SuzannePerPixBlinnTmNmAoSm) //.........................................
    {
        // Set scene name and info string
        switch (sceneID)
        {
            case SID_SuzannePerPixBlinn: s->name("Suzanne with per pixel lighting and reflection colors"); break;
            case SID_SuzannePerPixBlinnTm: s->name("Suzanne with per pixel lighting and texture mapping"); break;
            case SID_SuzannePerPixBlinnNm: s->name("Suzanne with per pixel lighting and normal mapping"); break;
            case SID_SuzannePerPixBlinnAo: s->name("Suzanne with per pixel lighting and ambient occlusion"); break;
            case SID_SuzannePerPixBlinnSm: s->name("Suzanne with per pixel lighting and shadow mapping"); break;
            case SID_SuzannePerPixBlinnTmNm: s->name("Suzanne with per pixel lighting, texture and normal mapping"); break;
            case SID_SuzannePerPixBlinnTmAo: s->name("Suzanne with per pixel lighting, texture mapping and ambient occlusion"); break;
            case SID_SuzannePerPixBlinnNmAo: s->name("Suzanne with per pixel lighting, normal mapping and ambient occlusion"); break;
            case SID_SuzannePerPixBlinnTmSm: s->name("Suzanne with per pixel lighting, texture mapping and shadow mapping"); break;
            case SID_SuzannePerPixBlinnNmSm: s->name("Suzanne with per pixel lighting, normal mapping and shadow mapping"); break;
            case SID_SuzannePerPixBlinnAoSm: s->name("Suzanne with per pixel lighting, ambient occlusion and shadow mapping"); break;
            case SID_SuzannePerPixBlinnTmNmAo: s->name("Suzanne with per pixel lighting and diffuse, normal, ambient occlusion and shadow mapping"); break;
            case SID_SuzannePerPixBlinnTmNmSm: s->name("Suzanne with per pixel lighting and diffuse, normal and shadow mapping "); break;
            case SID_SuzannePerPixBlinnTmNmAoSm: s->name("Suzanne with per pixel lighting and diffuse, normal, ambient occlusion and shadow mapping"); break;
            default: SL_EXIT_MSG("Unknown scene id!");
        }

        s->info(s->name());

        // Create a scene group node
        SLNode* scene = new SLNode("scene node");
        s->root3D(scene);

        // Create camera in the center
        SLCamera* cam1 = new SLCamera("Camera 1");
        cam1->translation(0, 0.5f, 2);
        cam1->lookAt(0, 0.5f, 0);
        cam1->setInitialState();
        cam1->focalDist(2);
        scene->addChild(cam1);

        // Create directional light for the sunlight
        SLLightDirect* light = new SLLightDirect(am, s, 0.1f);
        light->ambientPower(0.6f);
        light->diffusePower(0.6f);
        light->attenuation(1, 0, 0);
        light->translate(0, 0, 0.5);
        light->lookAt(1, -1, 0.5);
        SLAnimation* lightAnim = s->animManager().createNodeAnimation("LightAnim", 4.0f, true, EC_inOutSine, AL_pingPongLoop);
        lightAnim->createNodeAnimTrackForRotation(light, -180, SLVec3f(0, 1, 0));
        scene->addChild(light);

        // Add shadow mapping
        if (sceneID == SID_SuzannePerPixBlinnSm ||
            sceneID == SID_SuzannePerPixBlinnTmSm ||
            sceneID == SID_SuzannePerPixBlinnNmSm ||
            sceneID == SID_SuzannePerPixBlinnAoSm ||
            sceneID == SID_SuzannePerPixBlinnTmNmSm ||
            sceneID == SID_SuzannePerPixBlinnTmNmAoSm)
        {
            light->createsShadows(true);
            light->createShadowMap(-3, 3, SLVec2f(5, 5), SLVec2i(2048, 2048));
            light->doSmoothShadows(true);
        }

        // load teapot
        SLAssimpImporter importer;
        SLNode*          suzanneInCube = importer.load(s->animManager(),
                                              am,
                                              modelPath + "GLTF/AO-Baked-Test/AO-Baked-Test.gltf",
                                              texPath,
                                              nullptr,
                                              false,   // delete tex images after build
                                              true,    // load meshes only
                                              nullptr, // override material
                                              0.5f);   // ambient factor

        SLCol4f stoneColor(0.56f, 0.50f, 0.44f);

        // Remove unwanted textures
        if (sceneID == SID_SuzannePerPixBlinn ||
            sceneID == SID_SuzannePerPixBlinnSm)
        {
            auto removeAllTm = [=](SLMaterial* mat)
            {
                mat->removeTextureType(TT_diffuse);
                mat->removeTextureType(TT_normal);
                mat->removeTextureType(TT_occlusion);
                mat->ambientDiffuse(stoneColor);
            };
            suzanneInCube->updateMeshMat(removeAllTm, true);
        }
        if (sceneID == SID_SuzannePerPixBlinnTm)
        {
            auto removeNmAo = [=](SLMaterial* mat)
            {
                mat->removeTextureType(TT_normal);
                mat->removeTextureType(TT_occlusion);
            };
            suzanneInCube->updateMeshMat(removeNmAo, true);
        }
        if (sceneID == SID_SuzannePerPixBlinnNm)
        {
            auto removeNmAo = [=](SLMaterial* mat)
            {
                mat->ambientDiffuse(stoneColor);
                mat->removeTextureType(TT_diffuse);
                mat->removeTextureType(TT_occlusion);
            };
            suzanneInCube->updateMeshMat(removeNmAo, true);
        }
        if (sceneID == SID_SuzannePerPixBlinnAo)
        {
            auto removeNmAo = [=](SLMaterial* mat)
            {
                mat->ambientDiffuse(stoneColor);
                mat->removeTextureType(TT_diffuse);
                mat->removeTextureType(TT_normal);
            };
            suzanneInCube->updateMeshMat(removeNmAo, true);
        }
        if (sceneID == SID_SuzannePerPixBlinnTmSm)
        {
            auto removeTmNm = [=](SLMaterial* mat)
            {
                mat->removeTextureType(TT_normal);
                mat->removeTextureType(TT_occlusion);
            };
            suzanneInCube->updateMeshMat(removeTmNm, true);
        }
        if (sceneID == SID_SuzannePerPixBlinnNmSm)
        {
            auto removeTmNm = [=](SLMaterial* mat)
            {
                mat->ambientDiffuse(stoneColor);
                mat->removeTextureType(TT_diffuse);
                mat->removeTextureType(TT_occlusion);
            };
            suzanneInCube->updateMeshMat(removeTmNm, true);
        }
        if (sceneID == SID_SuzannePerPixBlinnAoSm)
        {
            auto removeTmNm = [=](SLMaterial* mat)
            {
                mat->ambientDiffuse(stoneColor);
                mat->removeTextureType(TT_diffuse);
                mat->removeTextureType(TT_normal);
            };
            suzanneInCube->updateMeshMat(removeTmNm, true);
        }
        if (sceneID == SID_SuzannePerPixBlinnTmAo)
        {
            auto removeNmAo = [=](SLMaterial* mat)
            {
                mat->removeTextureType(TT_normal);
            };
            suzanneInCube->updateMeshMat(removeNmAo, true);
        }
        if (sceneID == SID_SuzannePerPixBlinnNmAo)
        {
            auto removeNmAo = [=](SLMaterial* mat)
            {
                mat->ambientDiffuse(stoneColor);
                mat->removeTextureType(TT_diffuse);
            };
            suzanneInCube->updateMeshMat(removeNmAo, true);
        }
        if (sceneID == SID_SuzannePerPixBlinnTmNm ||
            sceneID == SID_SuzannePerPixBlinnTmNmSm)
        {
            auto removeAo = [=](SLMaterial* mat)
            { mat->removeTextureType(TT_occlusion); };
            suzanneInCube->updateMeshMat(removeAo, true);
        }

        scene->addChild(suzanneInCube);

        sv->camera(cam1);

        // Save energy
        sv->doWaitOnIdle(true);
    }

    else if (
      sceneID == SID_glTF_DamagedHelmet ||
      sceneID == SID_glTF_FlightHelmet ||
      sceneID == SID_glTF_Sponza ||
      sceneID == SID_glTF_WaterBottle) //..........................................................
    {
        SLstring damagedHelmet = modelPath + "GLTF/glTF-Sample-Models/2.0/DamagedHelmet/glTF/DamagedHelmet.gltf";
        SLstring flightHelmet  = modelPath + "GLTF/glTF-Sample-Models/2.0/FlightHelmet/glTF/FlightHelmet.gltf";
        SLstring sponzaPalace  = modelPath + "GLTF/glTF-Sample-Models/2.0/Sponza/glTF/Sponza.gltf";
        SLstring waterBottle   = modelPath + "GLTF/glTF-Sample-Models/2.0/WaterBottle/glTF/WaterBottle.gltf";

        if ( //(sceneID == SID_glTF_ClearCoatTest && Utils::fileExists(clearCoatTest)) ||
          (sceneID == SID_glTF_DamagedHelmet && SLFileStorage::exists(damagedHelmet, IOK_model)) ||
          (sceneID == SID_glTF_FlightHelmet && SLFileStorage::exists(flightHelmet, IOK_model)) ||
          (sceneID == SID_glTF_Sponza && SLFileStorage::exists(sponzaPalace, IOK_model)) ||
          (sceneID == SID_glTF_WaterBottle && SLFileStorage::exists(waterBottle, IOK_model)))
        {
            SLVec3f  camPos, lookAt;
            SLfloat  camClipFar = 100;
            SLstring modelFile;

            switch (sceneID)
            {
                case SID_glTF_DamagedHelmet:
                {
                    s->name("glTF-Sample-Model: Damaged Helmet");
                    modelFile = damagedHelmet;
                    camPos.set(0, 0, 3);
                    lookAt.set(0, camPos.y, 0);
                    camClipFar = 20;
                    break;
                }
                case SID_glTF_FlightHelmet:
                {
                    s->name("glTF-Sample-Model: Flight Helmet");
                    modelFile = flightHelmet;
                    camPos.set(0, 0.33f, 1.1f);
                    lookAt.set(0, camPos.y, 0);
                    camClipFar = 20;
                    break;
                }
                case SID_glTF_Sponza:
                {
                    s->name("glTF-Sample-Model: Sponza Palace in Dubrovnic");
                    modelFile = sponzaPalace;
                    camPos.set(-8, 1.6f, 0);
                    lookAt.set(0, camPos.y, 0);
                    break;
                }
                case SID_glTF_WaterBottle:
                {
                    s->name("glTF-Sample-Model: WaterBottle");
                    modelFile = waterBottle;
                    camPos.set(0, 0, 0.5f);
                    lookAt.set(0, camPos.y, 0);
                    camClipFar = 20;
                    break;
                }
            }

            s->info("glTF Sample Model with Physically Based Rendering material and Image Based Lighting.");

            // Create HDR CubeMap and get precalculated textures from it
            SLSkybox* skybox = new SLSkybox(am,
                                            shaderPath,
                                            modelPath + "GLTF/glTF-Sample-Models/hdris/envmap_malibu.hdr",
                                            SLVec2i(256, 256),
                                            "HDR Skybox");
            // Create a scene group node
            SLNode* scene = new SLNode("scene node");
            s->root3D(scene);

            // Create camera and initialize its parameters
            SLCamera* cam1 = new SLCamera("Camera 1");
            cam1->translation(camPos);
            cam1->lookAt(lookAt);
            cam1->background().colors(SLCol4f(0.2f, 0.2f, 0.2f));
            cam1->focalDist(camPos.length());
            cam1->clipFar(camClipFar);
            cam1->setInitialState();
            scene->addChild(cam1);

            // Add directional light with a position that corresponds roughly to the sun direction
            SLLight::gamma        = 2.2f;
            SLLightDirect* light1 = new SLLightDirect(am, s, 0.55f, 1.0f, -0.2f, 0.2f, 0, 1, 1);
            light1->lookAt(0, 0, 0);
            light1->attenuation(1, 0, 0);
            light1->createsShadows(true);
            light1->createShadowMapAutoSize(cam1, SLVec2i(2048, 2048), 4);
            light1->shadowMap()->cascadesFactor(1.0);
            light1->doSmoothShadows(true);
            light1->castsShadows(false);
            light1->shadowMinBias(0.001f);
            light1->shadowMaxBias(0.003f);
            scene->addChild(light1);

            // Import main model
            SLAssimpImporter importer;
            SLNode*          pbrGroup = importer.load(s->animManager(),
                                             am,
                                             modelFile,
                                             Utils::getPath(modelFile),
                                             skybox,
                                             false,   // delete tex images after build
                                             true,    // only meshes
                                             nullptr, // no replacement material
                                             0.4f);   // 40% ambient reflection
            scene->addChild(pbrGroup);

            s->skybox(skybox);
            sv->camera(cam1);
            sv->doWaitOnIdle(true); // Saves energy
        }
    }

    else if (sceneID == SID_Robotics_FanucCRX_FK) //...............................................
    {
        SLstring modelFile = modelPath + "GLTF/FanucCRX/Fanuc-CRX.gltf";

        if (SLFileStorage::exists(modelFile, IOK_model))
        {
            s->info("FANUC CRX Robot with Forward Kinematic");

            // Create a scene group node
            SLNode* scene = new SLNode("scene node");
            s->root3D(scene);

            // Create camera and initialize its parameters
            SLCamera* cam1 = new SLCamera("Camera 1");
            cam1->translation(0, 0.5f, 2.0f);
            cam1->lookAt(0, 0.5f, 0);
            cam1->background().colors(SLCol4f(0.7f, 0.7f, 0.7f),
                                      SLCol4f(0.2f, 0.2f, 0.2f));
            cam1->focalDist(2);
            cam1->setInitialState();
            scene->addChild(cam1);

            // Define directional
            SLLightDirect* light1 = new SLLightDirect(am,
                                                      s,
                                                      2,
                                                      2,
                                                      2,
                                                      0.2f,
                                                      1,
                                                      1,
                                                      1);
            light1->lookAt(0, 0, 0);
            light1->attenuation(1, 0, 0);
            light1->createsShadows(true);
            light1->createShadowMap(1, 7, SLVec2f(5, 5), SLVec2i(2048, 2048));
            light1->doSmoothShadows(true);
            light1->castsShadows(false);
            scene->addChild(light1);

            SLMaterial* matFloor = new SLMaterial(am, "matFloor", SLCol4f::WHITE * 0.5f);
            matFloor->ambient(SLCol4f::WHITE * 0.3f);
            SLMesh* rectangle = new SLRectangle(am,
                                                SLVec2f(-2, -2),
                                                SLVec2f(2, 2),
                                                1,
                                                1,
                                                "rectangle",
                                                matFloor);
            SLNode* floorRect = new SLNode(rectangle);
            floorRect->rotate(90, -1, 0, 0);
            scene->addChild(floorRect);

            // Import robot model
            SLAssimpImporter importer;
            SLNode*          robot = importer.load(s->animManager(),
                                          am,
                                          modelFile,
                                          Utils::getPath(modelFile),
                                          nullptr,
                                          false,   // delete tex images after build
                                          true,    // only meshes
                                          nullptr, // no replacement material
                                          0.2f);   // 40% ambient reflection

            // Set missing specular color
            robot->updateMeshMat([](SLMaterial* m)
                                 { m->specular(SLCol4f::WHITE); },
                                 true);

            SLNode* crx_j1 = robot->findChild<SLNode>("crx_j1");
            SLNode* crx_j2 = robot->findChild<SLNode>("crx_j2");
            SLNode* crx_j3 = robot->findChild<SLNode>("crx_j3");
            SLNode* crx_j4 = robot->findChild<SLNode>("crx_j4");
            SLNode* crx_j5 = robot->findChild<SLNode>("crx_j5");
            SLNode* crx_j6 = robot->findChild<SLNode>("crx_j6");

            SLfloat angleDEG    = 45;
            SLfloat durationSEC = 3.0f;

            SLAnimation* j1Anim = s->animManager().createNodeAnimation("j1Anim", durationSEC, true, EC_inOutCubic, AL_pingPongLoop);
            j1Anim->createNodeAnimTrackForRotation3(crx_j1, -angleDEG, 0, angleDEG, crx_j1->axisYOS());

            SLAnimation* j2Anim = s->animManager().createNodeAnimation("j2Anim", durationSEC, true, EC_inOutCubic, AL_pingPongLoop);
            j2Anim->createNodeAnimTrackForRotation3(crx_j2, -angleDEG, 0, angleDEG, -crx_j2->axisZOS());

            SLAnimation* j3Anim = s->animManager().createNodeAnimation("j3Anim", durationSEC, true, EC_inOutCubic, AL_pingPongLoop);
            j3Anim->createNodeAnimTrackForRotation3(crx_j3, angleDEG, 0, -angleDEG, -crx_j3->axisZOS());

            SLAnimation* j4Anim = s->animManager().createNodeAnimation("j4Anim", durationSEC, true, EC_inOutCubic, AL_pingPongLoop);
            j4Anim->createNodeAnimTrackForRotation3(crx_j4, -2 * angleDEG, 0, 2 * angleDEG, crx_j4->axisXOS());

            SLAnimation* j5Anim = s->animManager().createNodeAnimation("j5Anim", durationSEC, true, EC_inOutCubic, AL_pingPongLoop);
            j5Anim->createNodeAnimTrackForRotation3(crx_j5, -2 * angleDEG, 0, 2 * angleDEG, -crx_j5->axisZOS());

            SLAnimation* j6Anim = s->animManager().createNodeAnimation("j6Anim", durationSEC, true, EC_inOutCubic, AL_pingPongLoop);
            j6Anim->createNodeAnimTrackForRotation3(crx_j6, -2 * angleDEG, 0, 2 * angleDEG, crx_j6->axisXOS());

            scene->addChild(robot);

            sv->camera(cam1);
            sv->doWaitOnIdle(true); // Saves energy
        }
    }

    else if (sceneID == SID_VolumeRayCast) //......................................................
    {
        s->name("Volume Ray Cast Test");
        s->info("Volume Rendering of an angiographic MRI scan");

        // Load volume data into 3D texture
        SLVstring mriImages;
        for (SLint i = 0; i < 207; ++i)
            mriImages.push_back(Utils::formatString(texPath + "i%04u_0000b.png", i));

        SLint clamping3D = GL_CLAMP_TO_EDGE;
        if (SLGLState::instance()->getSLVersionNO() > "320")
            clamping3D = 0x812D; // GL_CLAMP_TO_BORDER

        SLGLTexture* texMRI = new SLGLTexture(am,
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
        SLTexColorLUT*   tf       = new SLTexColorLUT(am, tfAlphas, CLUT_BCGYR);

        // Load shader and uniforms for volume size
        SLGLProgram*   sp   = new SLGLProgramGeneric(am,
                                                 shaderPath + "VolumeRenderingRayCast.vert",
                                                 shaderPath + "VolumeRenderingRayCast.frag");
        SLGLUniform1f* volX = new SLGLUniform1f(UT_const, "u_volumeX", (SLfloat)texMRI->images()[0]->width());
        SLGLUniform1f* volY = new SLGLUniform1f(UT_const, "u_volumeY", (SLfloat)texMRI->images()[0]->height());
        SLGLUniform1f* volZ = new SLGLUniform1f(UT_const, "u_volumeZ", (SLfloat)mriImages.size());
        sp->addUniform1f(volX);
        sp->addUniform1f(volY);
        sp->addUniform1f(volZ);

        // Create volume rendering material
        SLMaterial* matVR = new SLMaterial(am, "matVR", texMRI, tf, nullptr, nullptr, sp);

        // Create camera
        SLCamera* cam1 = new SLCamera("Camera 1");
        cam1->translation(0, 0, 3);
        cam1->lookAt(0, 0, 0);
        cam1->focalDist(3);
        cam1->background().colors(SLCol4f(0, 0, 0));
        cam1->setInitialState();
        cam1->devRotLoc(&AppDemo::devRot, &AppDemo::devLoc);

        // Set light
        SLLightSpot* light1 = new SLLightSpot(am, s, 0.3f);
        light1->powers(0.1f, 1.0f, 1.0f);
        light1->attenuation(1, 0, 0);
        light1->translation(5, 5, 5);

        // Assemble scene with box node
        SLNode* scene = new SLNode("Scene");
        s->root3D(scene);
        scene->addChild(light1);
        scene->addChild(new SLNode(new SLBox(am, -1, -1, -1, 1, 1, 1, "Box", matVR)));
        scene->addChild(cam1);

        sv->camera(cam1);
    }
    else if (sceneID == SID_VolumeRayCastLighted) //...............................................
    {
        s->name("Volume Ray Cast Lighted Test");
        s->info("Volume Rendering of an angiographic MRI scan with lighting");

        // The MRI Images got loaded in advance
        if (gTexMRI3D && !gTexMRI3D->images().empty())
        {
            // Add pointer to the global resource vectors for deallocation
            if (am)
                am->textures().push_back(gTexMRI3D);
        }
        else
        {
            // Load volume data into 3D texture
            SLVstring mriImages;
            for (SLint i = 0; i < 207; ++i)
                mriImages.push_back(Utils::formatString(texPath + "i%04u_0000b.png", i));

            gTexMRI3D = new SLGLTexture(am,
                                        mriImages,
                                        GL_LINEAR,
                                        GL_LINEAR,
#ifndef SL_EMSCRIPTEN
                                        0x812D, // GL_CLAMP_TO_BORDER (GLSL 320)
                                        0x812D, // GL_CLAMP_TO_BORDER (GLSL 320)
#else
                                        GL_CLAMP_TO_EDGE,
                                        GL_CLAMP_TO_EDGE,
#endif
                                        "mri_head_front_to_back",
                                        true);

            gTexMRI3D->calc3DGradients(1, [](int progress)
                                       { AppDemo::jobProgressNum(progress); });
            // gTexMRI3D->smooth3DGradients(1, [](int progress) {AppDemo::jobProgressNum(progress);});
        }

        // Create transfer LUT 1D texture
        SLVAlphaLUTPoint tfAlphas = {SLAlphaLUTPoint(0.00f, 0.00f),
                                     SLAlphaLUTPoint(0.01f, 0.75f),
                                     SLAlphaLUTPoint(1.00f, 1.00f)};
        SLTexColorLUT*   tf       = new SLTexColorLUT(am, tfAlphas, CLUT_BCGYR);

        // Load shader and uniforms for volume size
        SLGLProgram*   sp   = new SLGLProgramGeneric(am,
                                                 shaderPath + "VolumeRenderingRayCast.vert",
                                                 shaderPath + "VolumeRenderingRayCastLighted.frag");
        SLGLUniform1f* volX = new SLGLUniform1f(UT_const, "u_volumeX", (SLfloat)gTexMRI3D->images()[0]->width());
        SLGLUniform1f* volY = new SLGLUniform1f(UT_const, "u_volumeY", (SLfloat)gTexMRI3D->images()[0]->height());
        SLGLUniform1f* volZ = new SLGLUniform1f(UT_const, "u_volumeZ", (SLfloat)gTexMRI3D->images().size());
        sp->addUniform1f(volX);
        sp->addUniform1f(volY);
        sp->addUniform1f(volZ);

        // Create volume rendering material
        SLMaterial* matVR = new SLMaterial(am, "matVR", gTexMRI3D, tf, nullptr, nullptr, sp);

        // Create camera
        SLCamera* cam1 = new SLCamera("Camera 1");
        cam1->translation(0, 0, 3);
        cam1->lookAt(0, 0, 0);
        cam1->focalDist(3);
        cam1->background().colors(SLCol4f(0, 0, 0));
        cam1->setInitialState();
        cam1->devRotLoc(&AppDemo::devRot, &AppDemo::devLoc);

        // Set light
        SLLightSpot* light1 = new SLLightSpot(am, s, 0.3f);
        light1->powers(0.1f, 1.0f, 1.0f);
        light1->attenuation(1, 0, 0);
        light1->translation(5, 5, 5);

        // Assemble scene with box node
        SLNode* scene = new SLNode("Scene");
        s->root3D(scene);
        scene->addChild(light1);
        scene->addChild(new SLNode(new SLBox(am, -1, -1, -1, 1, 1, 1, "Box", matVR)));
        scene->addChild(cam1);

        sv->camera(cam1);
    }

    else if (sceneID == SID_AnimationSkeletal) //..................................................
    {
        s->name("Skeletal Animation Test");
        s->info("Skeletal Animation Test Scene");

        SLAssimpImporter importer;

        // Root scene node
        SLNode* scene = new SLNode("scene group");
        s->root3D(scene);

        // camera
        SLCamera* cam1 = new SLCamera("Camera 1");
        cam1->translation(0, 2, 10);
        cam1->lookAt(0, 2, 0);
        cam1->focalDist(10);
        cam1->setInitialState();
        cam1->background().colors(SLCol4f(0.1f, 0.4f, 0.8f));
        cam1->setInitialState();
        cam1->devRotLoc(&AppDemo::devRot, &AppDemo::devLoc);
        scene->addChild(cam1);

        // light
        SLLightSpot* light1 = new SLLightSpot(am, s, 10, 10, 5, 0.5f);
        light1->powers(0.2f, 1.0f, 1.0f);
        light1->attenuation(1, 0, 0);
        scene->addChild(light1);

        // Floor grid
        SLMaterial* m2   = new SLMaterial(am, "m2", SLCol4f::WHITE);
        SLGrid*     grid = new SLGrid(am, SLVec3f(-5, 0, -5), SLVec3f(5, 0, 5), 20, 20, "Grid", m2);
        scene->addChild(new SLNode(grid, "grid"));

        // Astro boy character
        SLNode* char1 = importer.load(s->animManager(), am, modelPath + "DAE/AstroBoy/AstroBoy.dae", texPath);
        char1->translate(-1, 0, 0);
        SLAnimPlayback* char1Anim = s->animManager().lastAnimPlayback();
        char1Anim->playForward();
        scene->addChild(char1);

        // Sintel character
        SLNode* char2 = importer.load(s->animManager(),
                                      am,
                                      modelPath + "DAE/Sintel/SintelLowResOwnRig.dae",
                                      texPath
                                      //,false
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
        SLNode* cube1 = importer.load(s->animManager(),
                                      am,
                                      modelPath + "DAE/SkinnedCube/skinnedcube2.dae",
                                      texPath);
        cube1->translate(3, 0, 0);
        SLAnimPlayback* cube1Anim = s->animManager().lastAnimPlayback();
        cube1Anim->easing(EC_inOutSine);
        cube1Anim->playForward();
        scene->addChild(cube1);

        // Skinned cube 2
        SLNode* cube2 = importer.load(s->animManager(), am, modelPath + "DAE/SkinnedCube/skinnedcube4.dae", texPath);
        cube2->translate(-3, 0, 0);
        SLAnimPlayback* cube2Anim = s->animManager().lastAnimPlayback();
        cube2Anim->easing(EC_inOutSine);
        cube2Anim->playForward();
        scene->addChild(cube2);

        // Skinned cube 3
        SLNode* cube3 = importer.load(s->animManager(), am, modelPath + "DAE/SkinnedCube/skinnedcube5.dae", texPath);
        cube3->translate(0, 3, 0);
        SLAnimPlayback* cube3Anim = s->animManager().lastAnimPlayback();
        cube3Anim->loop(AL_pingPongLoop);
        cube3Anim->easing(EC_inOutCubic);
        cube3Anim->playForward();
        scene->addChild(cube3);

        sv->camera(cam1);
    }
    else if (sceneID == SID_AnimationNode) //......................................................
    {
        s->name("Node Animations Test");
        s->info("Node animations with different easing curves.");

        // Create textures and materials
        SLGLTexture* tex1 = new SLGLTexture(am, texPath + "Checkerboard0512_C.png");
        SLMaterial*  m1   = new SLMaterial(am, "m1", tex1);
        m1->kr(0.5f);
        SLMaterial* m2 = new SLMaterial(am, "m2", SLCol4f::WHITE * 0.5, SLCol4f::WHITE, 128, 0.5f, 0.0f, 1.0f);

        SLMesh* floorMesh = new SLRectangle(am, SLVec2f(-5, -5), SLVec2f(5, 5), 20, 20, "FloorMesh", m1);
        SLNode* floorRect = new SLNode(floorMesh);
        floorRect->rotate(90, -1, 0, 0);
        floorRect->translate(0, 0, -5.5f);

        // Bouncing balls
        SLNode* ball1 = new SLNode(new SLSphere(am, 0.3f, 16, 16, "Ball1", m2));
        ball1->translate(0, 0, 4, TS_object);
        SLAnimation* ball1Anim = s->animManager().createNodeAnimation("Ball1_anim", 1.0f, true, EC_linear, AL_pingPongLoop);
        ball1Anim->createNodeAnimTrackForTranslation(ball1, SLVec3f(0.0f, -5.2f, 0.0f));

        SLNode* ball2 = new SLNode(new SLSphere(am, 0.3f, 16, 16, "Ball2", m2));
        ball2->translate(-1.5f, 0, 4, TS_object);
        SLAnimation* ball2Anim = s->animManager().createNodeAnimation("Ball2_anim", 1.0f, true, EC_inQuad, AL_pingPongLoop);
        ball2Anim->createNodeAnimTrackForTranslation(ball2, SLVec3f(0.0f, -5.2f, 0.0f));

        SLNode* ball3 = new SLNode(new SLSphere(am, 0.3f, 16, 16, "Ball3", m2));
        ball3->translate(-2.5f, 0, 4, TS_object);
        SLAnimation* ball3Anim = s->animManager().createNodeAnimation("Ball3_anim", 1.0f, true, EC_outQuad, AL_pingPongLoop);
        ball3Anim->createNodeAnimTrackForTranslation(ball3, SLVec3f(0.0f, -5.2f, 0.0f));

        SLNode* ball4 = new SLNode(new SLSphere(am, 0.3f, 16, 16, "Ball4", m2));
        ball4->translate(1.5f, 0, 4, TS_object);
        SLAnimation* ball4Anim = s->animManager().createNodeAnimation("Ball4_anim", 1.0f, true, EC_inOutQuad, AL_pingPongLoop);
        ball4Anim->createNodeAnimTrackForTranslation(ball4, SLVec3f(0.0f, -5.2f, 0.0f));

        SLNode* ball5 = new SLNode(new SLSphere(am, 0.3f, 16, 16, "Ball5", m2));
        ball5->translate(2.5f, 0, 4, TS_object);
        SLAnimation* ball5Anim = s->animManager().createNodeAnimation("Ball5_anim", 1.0f, true, EC_outInQuad, AL_pingPongLoop);
        ball5Anim->createNodeAnimTrackForTranslation(ball5, SLVec3f(0.0f, -5.2f, 0.0f));

        SLCamera* cam1 = new SLCamera("Camera 1");
        cam1->translation(0, 0, 22);
        cam1->lookAt(0, 0, 0);
        cam1->focalDist(22);
        cam1->setInitialState();
        cam1->devRotLoc(&AppDemo::devRot, &AppDemo::devLoc);

        SLCamera* cam2 = new SLCamera("Camera 2");
        cam2->translation(5, 0, 0);
        cam2->lookAt(0, 0, 0);
        cam2->focalDist(5);
        cam2->clipFar(10);
        cam2->background().colors(SLCol4f(0, 0, 0.6f), SLCol4f(0, 0, 0.3f));
        cam2->setInitialState();
        cam2->devRotLoc(&AppDemo::devRot, &AppDemo::devLoc);

        SLCamera* cam3 = new SLCamera("Camera 3");
        cam3->translation(-5, -2, 0);
        cam3->lookAt(0, 0, 0);
        cam3->focalDist(5);
        cam3->clipFar(10);
        cam3->background().colors(SLCol4f(0.6f, 0, 0), SLCol4f(0.3f, 0, 0));
        cam3->setInitialState();
        cam3->devRotLoc(&AppDemo::devRot, &AppDemo::devLoc);

        SLLightSpot* light1 = new SLLightSpot(am, s, 0, 2, 0, 0.5f);
        light1->powers(0.2f, 1.0f, 1.0f);
        light1->attenuation(1, 0, 0);
        SLAnimation* light1Anim = s->animManager().createNodeAnimation("Light1_anim", 4.0f);
        light1Anim->createNodeAnimTrackForEllipse(light1, 6, A_z, 6, A_x);

        SLLightSpot* light2 = new SLLightSpot(am, s, 0, 0, 0, 0.2f);
        light2->powers(0.1f, 1.0f, 1.0f);
        light2->attenuation(1, 0, 0);
        light2->translate(-8, -4, 0, TS_world);
        light2->setInitialState();
        SLAnimation*     light2Anim = s->animManager().createNodeAnimation("light2_anim", 2.0f, true, EC_linear, AL_pingPongLoop);
        SLNodeAnimTrack* track      = light2Anim->createNodeAnimTrack();
        track->animatedNode(light2);
        track->createNodeKeyframe(0.0f);
        track->createNodeKeyframe(1.0f)->translation(SLVec3f(8, 8, 0));
        track->createNodeKeyframe(2.0f)->translation(SLVec3f(16, 0, 0));
        track->translationInterpolation(AI_bezier);

        SLNode* figure = BuildFigureGroup(am, s, m2, true);

        SLNode* scene = new SLNode("Scene");
        s->root3D(scene);
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

        sv->camera(cam1);
    }
    else if (sceneID == SID_AnimationMass) //......................................................
    {
        s->name("Mass Animation Test");
        s->info("Performance test for transform updates from many animations.");

        // Create a scene group node
        SLNode* scene = new SLNode("scene node");
        s->root3D(scene);

        // Create and add camera
        SLCamera* cam1 = new SLCamera("Camera 1");
        cam1->translation(0, 20, 40);
        cam1->lookAt(0, 0, 0);
        cam1->focalDist(42);
        scene->addChild(cam1);
        sv->camera(cam1);

        // Add spotlight
        SLLightSpot* light1 = new SLLightSpot(am, s, 0.1f);
        light1->translate(0, 10, 0);
        scene->addChild(light1);

        // build a basic scene to have a reference for the occurring rotations
        SLMaterial* genericMat = new SLMaterial(am, "some material");

        // we use the same mesh to visualize all the nodes
        SLBox* box = new SLBox(am, -0.5f, -0.5f, -0.5f, 0.5f, 0.5f, 0.5f, "box", genericMat);

        // We build a stack of levels, each level has a grid of boxes on it
        // each box on this grid has another grid above it with child nodes.
        // Best results are achieved if gridSize is an uneven number.
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
        parents.push_back(scene);

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
                    // node->scale(1.1f);

                    SLfloat       duration = 1.0f + 5.0f * ((SLfloat)i / (SLfloat)nodesPerLvl);
                    ostringstream oss;

                    oss << "random anim " << nodeIndex++;
                    SLAnimation* anim = s->animManager().createNodeAnimation(oss.str(), duration, true, EC_inOutSine, AL_pingPongLoop);
                    anim->createNodeAnimTrackForTranslation(node, SLVec3f(0.0f, 1.0f, 0.0f));
                }
            }
        }
    }
    else if (sceneID == SID_AnimationAstroboyArmy) //..............................................
    {
        s->name("Astroboy Army Skinned Animation");
        s->info(s->name());

        // Create materials
        SLMaterial* m1 = new SLMaterial(am, "m1", SLCol4f::GRAY);
        m1->specular(SLCol4f::BLACK);

        // Define a light
        SLLightSpot* light1 = new SLLightSpot(am, s, 100, 40, 100, 1);
        light1->powers(0.1f, 1.0f, 1.0f);
        light1->attenuation(1, 0, 0);

        // Define camera
        SLCamera* cam1 = new SLCamera;
        cam1->translation(0, 10, 10);
        cam1->lookAt(0, 0, 0);
        cam1->focalDist(cam1->translationOS().length());
        cam1->background().colors(SLCol4f(0.1f, 0.4f, 0.8f));
        cam1->setInitialState();
        cam1->devRotLoc(&AppDemo::devRot, &AppDemo::devLoc);

        // Floor rectangle
        SLNode* rect = new SLNode(new SLRectangle(am,
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
        SLNode*          center = importer.load(s->animManager(), am, modelPath + "DAE/AstroBoy/AstroBoy.dae", texPath);
        s->animManager().lastAnimPlayback()->playForward();

        // Assemble scene
        SLNode* scene = new SLNode("scene group");
        s->root3D(scene);
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

        sv->camera(cam1);
    }
    else if (sceneID == SID_VideoTextureLive ||
             sceneID == SID_VideoTextureFile) //...................................................
    {
        // Set scene name and info string
        if (sceneID == SID_VideoTextureLive)
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
            CVCapture::instance()->videoFilename = AppDemo::videoPath + "street3.mp4";
            CVCapture::instance()->videoLoops    = true;
        }
        sv->viewportSameAsVideo(true);

        // Create video texture on global pointer updated in AppDemoVideo
        videoTexture   = new SLGLTexture(am, texPath + "LiveVideoError.png", GL_LINEAR, GL_LINEAR);
        SLMaterial* m1 = new SLMaterial(am, "VideoMat", videoTexture);

        // Create a root scene group for all nodes
        SLNode* scene = new SLNode("scene node");
        s->root3D(scene);

        // Create a camera node
        SLCamera* cam1 = new SLCamera("Camera 1");
        cam1->translation(0, 0, 20);
        cam1->focalDist(20);
        cam1->lookAt(0, 0, 0);
        cam1->background().texture(videoTexture);
        cam1->setInitialState();
        cam1->devRotLoc(&AppDemo::devRot, &AppDemo::devLoc);
        scene->addChild(cam1);

        // Create rectangle meshe and nodes
        SLfloat h        = 5.0f;
        SLfloat w        = h * sv->viewportWdivH();
        SLMesh* rectMesh = new SLRectangle(am, SLVec2f(-w, -h), SLVec2f(w, h), 1, 1, "rect mesh", m1);
        SLNode* rectNode = new SLNode(rectMesh, "rect node");
        rectNode->translation(0, 0, -5);
        scene->addChild(rectNode);

        // Center sphere
        SLNode* sphere = new SLNode(new SLSphere(am, 2, 32, 32, "Sphere", m1));
        sphere->rotate(-90, 1, 0, 0);
        scene->addChild(sphere);

        // Create a light source node
        SLLightSpot* light1 = new SLLightSpot(am, s, 0.3f);
        light1->translation(0, 0, 5);
        light1->lookAt(0, 0, 0);
        light1->name("light node");
        scene->addChild(light1);

        sv->camera(cam1);
        sv->doWaitOnIdle(false);
    }
    else if (sceneID == SID_VideoTrackChessMain ||
             sceneID == SID_VideoTrackChessScnd ||
             sceneID == SID_VideoCalibrateMain ||
             sceneID == SID_VideoCalibrateScnd) //.................................................
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
        if (sceneID == SID_VideoTrackChessMain ||
            sceneID == SID_VideoTrackChessScnd)
        {
            if (sceneID == SID_VideoTrackChessMain)
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
        else if (sceneID == SID_VideoCalibrateMain)
        {
            if (AppDemo::calibrationEstimator)
            {
                delete AppDemo::calibrationEstimator;
                AppDemo::calibrationEstimator = nullptr;
            }
            CVCapture::instance()->videoType(VT_MAIN);
            s->name("Calibrate Main Cam.");
        }
        else if (sceneID == SID_VideoCalibrateScnd)
        {
            if (AppDemo::calibrationEstimator)
            {
                delete AppDemo::calibrationEstimator;
                AppDemo::calibrationEstimator = nullptr;
            }
            CVCapture::instance()->videoType(VT_SCND);
            s->name("Calibrate Scnd Cam.");
        }

        // Create video texture on global pointer updated in AppDemoVideo
        videoTexture = new SLGLTexture(am, texPath + "LiveVideoError.png", GL_LINEAR, GL_LINEAR);

        // Material
        SLMaterial* yellow = new SLMaterial(am, "mY", SLCol4f(1, 1, 0, 0.5f));

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
        cam1->devRotLoc(&AppDemo::devRot, &AppDemo::devLoc);
        scene->addChild(cam1);

        // Create a light source node
        SLLightSpot* light1 = new SLLightSpot(am, s, e1 * 0.5f);
        light1->translate(e9, e9, e9);
        light1->name("light node");
        scene->addChild(light1);

        // Build mesh & node
        if (sceneID == SID_VideoTrackChessMain ||
            sceneID == SID_VideoTrackChessScnd)
        {
            SLBox*  box     = new SLBox(am, 0.0f, 0.0f, 0.0f, e3, e3, e3, "Box", yellow);
            SLNode* boxNode = new SLNode(box, "Box Node");
            boxNode->setDrawBitsRec(SL_DB_CULLOFF, true);
            SLNode* axisNode = new SLNode(new SLCoordAxis(am), "Axis Node");
            axisNode->setDrawBitsRec(SL_DB_MESHWIRED, false);
            axisNode->scale(e3);
            boxNode->addChild(axisNode);
            scene->addChild(boxNode);
        }

        // Create OpenCV Tracker for the camera node for AR camera.
        tracker = new CVTrackedChessboard(AppDemo::calibIniPath);
        tracker->drawDetection(true);
        trackedNode = cam1;

        // pass the scene group as root node
        s->root3D(scene);

        // Set active camera
        sv->camera(cam1);
        sv->doWaitOnIdle(false);
    }
    else if (sceneID == SID_VideoTrackArucoMain ||
             sceneID == SID_VideoTrackArucoScnd) //................................................
    {
        /*
        The tracking of markers is done in AppDemoVideo::onUpdateVideo by calling the specific
        CVTracked::track method. If a marker was found it overwrites the linked nodes
        object matrix (SLNode::_om). If the linked node is the active camera the found
        transform is additionally inversed. This would be the standard augmented realtiy
        use case.
        */

        if (sceneID == SID_VideoTrackArucoMain)
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
        videoTexture = new SLGLTexture(am, texPath + "LiveVideoError.png", GL_LINEAR, GL_LINEAR);

        // Material
        SLMaterial* yellow = new SLMaterial(am, "mY", SLCol4f(1, 1, 0, 0.5f));
        SLMaterial* cyan   = new SLMaterial(am, "mY", SLCol4f(0, 1, 1, 0.5f));

        // Create a scene group node
        SLNode* scene = new SLNode("scene node");
        s->root3D(scene);

        // Create a camera node 1
        SLCamera* cam1 = new SLCamera("Camera 1");
        cam1->translation(0, 0, 5);
        cam1->lookAt(0, 0, 0);
        cam1->fov(CVCapture::instance()->activeCamera->calibration.cameraFovVDeg());
        cam1->background().texture(videoTexture);
        cam1->setInitialState();
        cam1->devRotLoc(&AppDemo::devRot, &AppDemo::devLoc);
        scene->addChild(cam1);

        // Create a light source node
        SLLightSpot* light1 = new SLLightSpot(am, s, 0.02f);
        light1->translation(0.12f, 0.12f, 0.12f);
        light1->name("light node");
        scene->addChild(light1);

        // Get the half edge length of the aruco marker
        SLfloat edgeLen = CVTrackedAruco::params.edgeLength;
        SLfloat he      = edgeLen * 0.5f;

        // Build mesh & node that will be tracked by the 1st marker (camera)
        SLBox*  box1      = new SLBox(am, -he, -he, 0.0f, he, he, 2 * he, "Box 1", yellow);
        SLNode* boxNode1  = new SLNode(box1, "Box Node 1");
        SLNode* axisNode1 = new SLNode(new SLCoordAxis(am), "Axis Node 1");
        axisNode1->setDrawBitsRec(SL_DB_MESHWIRED, false);
        axisNode1->scale(edgeLen);
        boxNode1->addChild(axisNode1);
        boxNode1->setDrawBitsRec(SL_DB_CULLOFF, true);
        scene->addChild(boxNode1);

        // Create OpenCV Tracker for the box node
        tracker = new CVTrackedAruco(9, AppDemo::calibIniPath);
        tracker->drawDetection(true);
        trackedNode = boxNode1;

        // Set active camera
        sv->camera(cam1);

        // Turn on constant redraw
        sv->doWaitOnIdle(false);
    }
    else if (sceneID == SID_VideoTrackFeature2DMain) //............................................
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
        videoTexture = new SLGLTexture(am, texPath + "LiveVideoError.png", GL_LINEAR, GL_LINEAR);

        SLCamera* cam1 = new SLCamera("Camera 1");
        cam1->translation(0, 2, 60);
        cam1->lookAt(15, 15, 0);
        cam1->clipNear(0.1f);
        cam1->clipFar(1000.0f); // Increase to infinity?
        cam1->setInitialState();
        cam1->devRotLoc(&AppDemo::devRot, &AppDemo::devLoc);
        cam1->background().texture(videoTexture);
        CVCapture::instance()->videoType(VT_MAIN);

        SLLightSpot* light1 = new SLLightSpot(am, s, 420, 420, 420, 1);
        light1->powers(1.0f, 1.0f, 1.0f);

        SLLightSpot* light2 = new SLLightSpot(am, s, -450, -340, 420, 1);
        light2->powers(1.0f, 1.0f, 1.0f);
        light2->attenuation(1, 0, 0);

        SLLightSpot* light3 = new SLLightSpot(am, s, 450, -370, 0, 1);
        light3->powers(1.0f, 1.0f, 1.0f);
        light3->attenuation(1, 0, 0);

        // Coordinate axis node
        SLNode* axis = new SLNode(new SLCoordAxis(am), "Axis Node");
        axis->setDrawBitsRec(SL_DB_MESHWIRED, false);
        axis->scale(100);
        axis->rotate(-90, 1, 0, 0);

        // Yellow center box
        SLMaterial* yellow = new SLMaterial(am, "mY", SLCol4f(1, 1, 0, 0.5f));
        SLNode*     box    = new SLNode(new SLBox(am, 0, 0, 0, 100, 100, 100, "Box", yellow), "Box Node");
        box->rotate(-90, 1, 0, 0);

        // Scene structure
        SLNode* scene = new SLNode("Scene");
        s->root3D(scene);
        scene->addChild(light1);
        scene->addChild(light2);
        scene->addChild(light3);
        scene->addChild(axis);
        scene->addChild(box);
        scene->addChild(cam1);

        // Create feature tracker and let it pose the camera for AR posing
        tracker = new CVTrackedFeatures(texPath + "features_stones.jpg");
        // tracker = new CVTrackedFeatures("features_abstract.jpg");
        tracker->drawDetection(true);
        trackedNode = cam1;

        sv->doWaitOnIdle(false); // for constant video feed
        sv->camera(cam1);
        AppDemo::devRot.isUsed(true);
    }
#ifndef SL_EMSCRIPTEN
    else if (sceneID == SID_VideoTrackFaceMain ||
             sceneID == SID_VideoTrackFaceScnd) //.................................................
    {
        /*
        The tracking of markers is done in AppDemoVideo::onUpdateVideo by calling the specific
        CVTracked::track method. If a marker was found it overwrites the linked nodes
        object matrix (SLNode::_om). If the linked node is the active camera the found
        transform is additionally inversed. This would be the standard augmented realtiy
        use case.
        */

        if (sceneID == SID_VideoTrackFaceMain)
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
        videoTexture = new SLGLTexture(am,
                                       texPath + "LiveVideoError.png",
                                       GL_LINEAR,
                                       GL_LINEAR);

        SLCamera* cam1 = new SLCamera("Camera 1");
        cam1->translation(0, 0, 0.5f);
        cam1->clipNear(0.1f);
        cam1->clipFar(1000.0f); // Increase to infinity?
        cam1->setInitialState();
        cam1->devRotLoc(&AppDemo::devRot, &AppDemo::devLoc);
        cam1->background().texture(videoTexture);

        SLLightSpot* light1 = new SLLightSpot(am, s, 10, 10, 10, 1);
        light1->powers(1.0f, 1.0f, 1.0f);
        light1->attenuation(1, 0, 0);

        // Load sunglasses
        SLAssimpImporter importer;
        SLNode*          glasses = importer.load(s->animManager(),
                                        am,
                                        modelPath + "FBX/Sunglasses.fbx",
                                        texPath);
        glasses->scale(0.008f);
        glasses->translate(0, 1.5f, 0);

        // Add axis arrows at world center
        SLNode* axis = new SLNode(new SLCoordAxis(am), "Axis Node");
        axis->setDrawBitsRec(SL_DB_MESHWIRED, false);
        axis->scale(0.03f);

        // Scene structure
        SLNode* scene = new SLNode("Scene");
        s->root3D(scene);
        scene->addChild(light1);
        scene->addChild(cam1);
        scene->addChild(glasses);
        scene->addChild(axis);

        // Add a face tracker that moves the camera node
        tracker     = new CVTrackedFaces(AppDemo::calibIniPath + "haarcascade_frontalface_alt2.xml",
                                     AppDemo::calibIniPath + "lbfmodel.yaml",
                                     3);
        trackedNode = cam1;
        tracker->drawDetection(true);

        sv->doWaitOnIdle(false); // for constant video feed
        sv->camera(cam1);
    }
#endif
#ifdef SL_BUILD_WITH_MEDIAPIPE
    else if (sceneID == SID_VideoTrackMediaPipeHandsMain) //.......................................
    {
        CVCapture::instance()->videoType(VT_MAIN);
        s->name("Tracking Hands with MediaPipe (main cam.)");
        s->info("Tracking Hands with MediaPipe. Please be patient for recognition ...");

        videoTexture = new SLGLTexture(am,
                                       texPath + "LiveVideoError.png",
                                       GL_LINEAR,
                                       GL_LINEAR);

        SLCamera* cam1 = new SLCamera("Camera 1");
        cam1->background().texture(videoTexture);

        SLNode* scene = new SLNode("Scene");
        s->root3D(scene);

        tracker     = new CVTrackedMediaPipeHands(AppDemo::dataPath);
        trackedNode = cam1;
        tracker->drawDetection(true);

        sv->doWaitOnIdle(false);
        sv->camera(cam1);
    }
#endif
#ifdef SL_BUILD_WAI
    else if (sceneID == SID_VideoTrackWAI) //......................................................
    {
        CVCapture::instance()->videoType(VT_MAIN);
        s->name("Track WAI (main cam.)");
        s->info("Track the scene with a point cloud built with the WAI (Where Am I) library.");

        // Create video texture on global pointer updated in AppDemoVideo
        videoTexture = new SLGLTexture(am, texPath + "LiveVideoError.png", GL_LINEAR, GL_LINEAR);

        // Material
        SLMaterial* yellow = new SLMaterial(am, "mY", SLCol4f(1, 1, 0, 0.5f));
        SLMaterial* cyan   = new SLMaterial(am, "mY", SLCol4f(0, 1, 1, 0.5f));

        // Create a scene group node
        SLNode* scene = new SLNode("scene node");
        s->root3D(scene);

        // Create a camera node 1
        SLCamera* cam1 = new SLCamera("Camera 1");
        cam1->translation(0, 0, 5);
        cam1->lookAt(0, 0, 0);
        cam1->fov(CVCapture::instance()->activeCamera->calibration.cameraFovVDeg());
        cam1->background().texture(videoTexture);
        cam1->setInitialState();
        scene->addChild(cam1);

        // Create a light source node
        SLLightSpot* light1 = new SLLightSpot(am, s, 0.02f);
        light1->translation(0.12f, 0.12f, 0.12f);
        light1->name("light node");
        scene->addChild(light1);

        // Get the half edge length of the aruco marker
        SLfloat edgeLen = 0.1f;
        SLfloat he      = edgeLen * 0.5f;

        // Build mesh & node that will be tracked by the 1st marker (camera)
        SLBox*  box1      = new SLBox(am, -he, -he, -he, he, he, he, "Box 1", yellow);
        SLNode* boxNode1  = new SLNode(box1, "Box Node 1");
        SLNode* axisNode1 = new SLNode(new SLCoordAxis(am), "Axis Node 1");
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
        tracker = new CVTrackedWAI(Utils::findFile(vocFileName, {AppDemo::calibIniPath, AppDemo::exePath}));
        tracker->drawDetection(true);
        trackedNode = cam1;

        sv->camera(cam1);

        // Turn on constant redraw
        sv->doWaitOnIdle(false);
    }
#endif
    else if (sceneID == SID_VideoSensorAR) //......................................................
    {
        // Set scene name and info string
        s->name("Video Sensor AR");
        s->info("Minimal scene to test the devices IMU and GPS Sensors. See the sensor information. GPS needs a few sec. to improve the accuracy.");

        // Create video texture on global pointer updated in AppDemoVideo
        videoTexture = new SLGLTexture(am, texPath + "LiveVideoError.png", GL_LINEAR, GL_LINEAR);

        SLCamera* cam1 = new SLCamera("Camera 1");
        cam1->translation(0, 0, 60);
        cam1->lookAt(0, 0, 0);
        cam1->fov(CVCapture::instance()->activeCamera->calibration.cameraFovVDeg());
        cam1->clipNear(0.1f);
        cam1->clipFar(10000.0f);
        cam1->setInitialState();
        cam1->devRotLoc(&AppDemo::devRot, &AppDemo::devLoc);
        cam1->background().texture(videoTexture);

        // Turn on main video
        CVCapture::instance()->videoType(VT_MAIN);

        // Create directional light for the sunlight
        SLLightDirect* light = new SLLightDirect(am, s, 1.0f);
        light->powers(1.0f, 1.0f, 1.0f);
        light->attenuation(1, 0, 0);

        // Let the sun be rotated by time and location
        AppDemo::devLoc.sunLightNode(light);

        SLNode* axis = new SLNode(new SLCoordAxis(am), "Axis Node");
        axis->setDrawBitsRec(SL_DB_MESHWIRED, false);
        axis->scale(2);
        axis->rotate(-90, 1, 0, 0);

        // Yellow center box
        SLMaterial* yellow = new SLMaterial(am, "mY", SLCol4f(1, 1, 0, 0.5f));
        SLNode*     box    = new SLNode(new SLBox(am, -.5f, -.5f, -.5f, .5f, .5f, .5f, "Box", yellow), "Box Node");

        // Scene structure
        SLNode* scene = new SLNode("Scene");
        s->root3D(scene);
        scene->addChild(light);
        scene->addChild(cam1);
        scene->addChild(box);
        scene->addChild(axis);

        sv->camera(cam1);

#if defined(SL_OS_MACIOS) || defined(SL_OS_ANDROID)
        // activate rotation and gps sensor
        AppDemo::devRot.isUsed(true);
        AppDemo::devRot.zeroYawAtStart(false);
        AppDemo::devLoc.isUsed(true);
        AppDemo::devLoc.useOriginAltitude(true);
        AppDemo::devLoc.hasOrigin(false);
        cam1->camAnim(SLCamAnim::CA_deviceRotLocYUp);
#else
        cam1->camAnim(SLCamAnim::CA_turntableYUp);
        AppDemo::devRot.zeroYawAtStart(true);
#endif

        sv->doWaitOnIdle(false);                    // for constant video feed
    }
    else if (sceneID == SID_ErlebARBernChristoffel) //.............................................
    {
        s->name("Christoffel Tower AR");
        s->info("Augmented Reality Christoffel Tower");

        // Create video texture on global pointer updated in AppDemoVideo
        videoTexture = new SLGLTexture(am, texPath + "LiveVideoError.png", GL_LINEAR, GL_LINEAR);
        videoTexture->texType(TT_videoBkgd);

        // Create see through video background material without shadow mapping
        SLMaterial* matVideoBkgd = new SLMaterial(am, "matVideoBkgd", videoTexture);
        matVideoBkgd->reflectionModel(RM_Custom);

        // Create see through video background material with shadow mapping
        SLMaterial* matVideoBkgdSM = new SLMaterial(am, "matVideoBkgdSM", videoTexture);
        matVideoBkgdSM->reflectionModel(RM_Custom);
        matVideoBkgdSM->ambient(SLCol4f(0.6f, 0.6f, 0.6f));
        matVideoBkgdSM->getsShadows(true);

        SLCamera* cam1 = new SLCamera("Camera 1");
        cam1->translation(0, 2, 0);
        cam1->lookAt(-10, 2, 0);
        cam1->clipNear(1);
        cam1->clipFar(700);
        cam1->setInitialState();
        cam1->devRotLoc(&AppDemo::devRot, &AppDemo::devLoc);
        cam1->background().texture(videoTexture);

        // Turn on main video
        CVCapture::instance()->videoType(VT_MAIN);

        // Create directional light for the sunlight
        SLLightDirect* sunLight = new SLLightDirect(am, s, 2.0f);
        sunLight->translate(-44.89f, 18.05f, -26.07f);
        sunLight->powers(1.0f, 1.5f, 1.0f);
        sunLight->attenuation(1, 0, 0);
        sunLight->doSunPowerAdaptation(true);
        sunLight->createsShadows(true);
        sunLight->createShadowMapAutoSize(cam1, SLVec2i(2048, 2048), 4);
        sunLight->shadowMap()->cascadesFactor(3.0);
        // sunLight->createShadowMap(-100, 150, SLVec2f(200, 150), SLVec2i(4096, 4096));
        sunLight->doSmoothShadows(true);
        sunLight->castsShadows(false);
        sunLight->shadowMinBias(0.001f);
        sunLight->shadowMaxBias(0.003f);
        AppDemo::devLoc.sunLightNode(sunLight); // Let the sun be rotated by time and location

        // Import the main model
        SLAssimpImporter importer;
        SLNode*          bern = importer.load(s->animManager(),
                                     am,
                                     dataPath + "erleb-AR/models/bern/bern-christoffel.gltf",
                                     texPath,
                                     nullptr,
                                     false,
                                     true,
                                     nullptr,
                                     0.3f); // ambient factor

        // Make city with hard edges and without shadow mapping
        SLNode* Umg = bern->findChild<SLNode>("Umgebung-Swisstopo");
        Umg->setMeshMat(matVideoBkgd, true);
        Umg->setDrawBitsRec(SL_DB_WITHEDGES, true);
        Umg->castsShadows(false);

        // Hide some objects
        bern->findChild<SLNode>("Baldachin-Glas")->drawBits()->set(SL_DB_HIDDEN, true);
        bern->findChild<SLNode>("Baldachin-Stahl")->drawBits()->set(SL_DB_HIDDEN, true);

        // Set the video background shader on the baldachin and the ground with shadow mapping
        bern->findChild<SLNode>("Baldachin-Stahl")->setMeshMat(matVideoBkgdSM, true);
        bern->findChild<SLNode>("Baldachin-Glas")->setMeshMat(matVideoBkgdSM, true);
        bern->findChild<SLNode>("Chr-Alt-Stadtboden")->setMeshMat(matVideoBkgdSM, true);
        bern->findChild<SLNode>("Chr-Neu-Stadtboden")->setMeshMat(matVideoBkgdSM, true);

        // Hide the new (last) version of the Christoffel tower
        bern->findChild<SLNode>("Chr-Neu")->drawBits()->set(SL_DB_HIDDEN, true);

        // Create textures and material for water
        SLGLTexture* cubemap = new SLGLTexture(am,
                                               dataPath + "erleb-AR/models/bern/Sea1+X1024.jpg",
                                               dataPath + "erleb-AR/models/bern/Sea1-X1024.jpg",
                                               dataPath + "erleb-AR/models/bern/Sea1+Y1024.jpg",
                                               dataPath + "erleb-AR/models/bern/Sea1-Y1024.jpg",
                                               dataPath + "erleb-AR/models/bern/Sea1+Z1024.jpg",
                                               dataPath + "erleb-AR/models/bern/Sea1-Z1024.jpg");
        // Material for water
        SLMaterial* matWater = new SLMaterial(am, "water", SLCol4f::BLACK, SLCol4f::BLACK, 100, 0.1f, 0.9f, 1.5f);
        matWater->translucency(1000);
        matWater->transmissive(SLCol4f::WHITE);
        matWater->addTexture(cubemap);
        matWater->program(new SLGLProgramGeneric(am,
                                                 shaderPath + "Reflect.vert",
                                                 shaderPath + "Reflect.frag"));
        bern->findChild<SLNode>("Chr-Wasser")->setMeshMat(matWater, true);

        // Add axis object a world origin (Loeb Ecke)
        SLNode* axis = new SLNode(new SLCoordAxis(am), "Axis Node");
        axis->setDrawBitsRec(SL_DB_MESHWIRED, false);
        axis->rotate(-90, 1, 0, 0);
        axis->castsShadows(false);

        // Bridge rotation animation
        SLNode*      bridge     = bern->findChild<SLNode>("Chr-Alt-Tor");
        SLAnimation* bridgeAnim = s->animManager().createNodeAnimation("Gate animation", 8.0f, true, EC_inOutQuint, AL_pingPongLoop);
        bridgeAnim->createNodeAnimTrackForRotation(bridge, 90, bridge->forwardOS());

        // Gate translation animation
        SLNode*      gate     = bern->findChild<SLNode>("Chr-Alt-Gatter");
        SLAnimation* gateAnim = s->animManager().createNodeAnimation("Gatter Animation", 8.0f, true, EC_inOutQuint, AL_pingPongLoop);
        gateAnim->createNodeAnimTrackForTranslation(gate, SLVec3f(0.0f, -3.6f, 0.0f));

        SLNode* scene = new SLNode("Scene");
        s->root3D(scene);
        scene->addChild(sunLight);
        scene->addChild(axis);
        scene->addChild(bern);
        scene->addChild(cam1);

        // initialize sensor stuff
        AppDemo::devLoc.originLatLonAlt(46.94763, 7.44074, 542.2);        // Loeb Ecken
        AppDemo::devLoc.defaultLatLonAlt(46.94841, 7.43970, 542.2 + 1.7); // Bahnhof Ausgang in Augenhöhe

        AppDemo::devLoc.nameLocations().push_back(SLLocation("Loeb Ecken, Origin", 46, 56, 51.451, 7, 26, 26.676, 542.2));
        AppDemo::devLoc.nameLocations().push_back(SLLocation("Milchgässli, Velomarkierung, (N)", 46, 56, 54.197, 7, 26, 23.366, 541.2 + 1.7));
        AppDemo::devLoc.nameLocations().push_back(SLLocation("Spitalgasse (E)", 46, 56, 51.703, 7, 26, 27.565, 542.1 + 1.7));
        AppDemo::devLoc.nameLocations().push_back(SLLocation("Tramhaltestelle UBS, eckiger Schachtd. (S)", 46, 56, 50.366, 7, 26, 24.544, 542.3 + 1.7));
        AppDemo::devLoc.nameLocations().push_back(SLLocation("Ecke Schauplatz-Christoffelgasse (S)", 46, 56, 50.139, 7, 26, 27.225, 542.1 + 1.7));
        AppDemo::devLoc.nameLocations().push_back(SLLocation("Bubenbergplatz (S)", 46, 56, 50.304, 7, 26, 22.113, 542.4 + 1.7));
        AppDemo::devLoc.nameLocations().push_back(SLLocation("Heiliggeistkirche (Dole, N-W)", 46, 56, 53.500, 7, 26, 25.499, 541.6 + 1.7));
        AppDemo::devLoc.originLatLonAlt(AppDemo::devLoc.nameLocations()[0].posWGS84LatLonAlt);
        AppDemo::devLoc.activeNamedLocation(1);   // This sets the location 1 as defaultENU
        AppDemo::devLoc.locMaxDistanceM(1000.0f); // Max. Distanz. zum Loeb Ecken
        AppDemo::devLoc.improveOrigin(false);     // Keine autom. Verbesserung vom Origin
        AppDemo::devLoc.useOriginAltitude(true);
        AppDemo::devLoc.hasOrigin(true);
        AppDemo::devLoc.offsetMode(LOM_twoFingerY);
        AppDemo::devRot.zeroYawAtStart(false);
        AppDemo::devRot.offsetMode(ROM_oneFingerX);

        // This loads the DEM file and overwrites the altitude of originLatLonAlt and defaultLatLonAlt
        SLstring tif = dataPath + "erleb-AR/models/bern/DEM-Bern-2600_1199-WGS84.tif";
        AppDemo::devLoc.loadGeoTiff(tif);

#if defined(SL_OS_MACIOS) || defined(SL_OS_ANDROID)
        AppDemo::devLoc.isUsed(true);
        AppDemo::devRot.isUsed(true);
        cam1->camAnim(SLCamAnim::CA_deviceRotLocYUp);
#else
        AppDemo::devLoc.isUsed(false);
        AppDemo::devRot.isUsed(false);
        SLVec3d pos_d = AppDemo::devLoc.defaultENU() - AppDemo::devLoc.originENU();
        SLVec3f pos_f((SLfloat)pos_d.x, (SLfloat)pos_d.y, (SLfloat)pos_d.z);
        cam1->translation(pos_f);
        cam1->focalDist(pos_f.length());
        cam1->lookAt(SLVec3f::ZERO);
        cam1->camAnim(SLCamAnim::CA_turntableYUp);
#endif

        sv->doWaitOnIdle(false); // for constant video feed
        sv->camera(cam1);
    }
    else if (sceneID == SID_ErlebARBielBFH) //.....................................................
    {
        s->name("Biel-BFH AR");
        s->info("Augmented Reality at Biel-BFH");

        // Create video texture on global pointer updated in AppDemoVideo
        videoTexture = new SLGLTexture(am, texPath + "LiveVideoError.png", GL_LINEAR, GL_LINEAR);

        // Define shader that shows on all pixels the video background
        SLGLProgram* spVideoBackground = new SLGLProgramGeneric(am,
                                                                shaderPath + "PerPixTmBackground.vert",
                                                                shaderPath + "PerPixTmBackground.frag");
        SLMaterial*  matVideoBkgd      = new SLMaterial(am,
                                                  "matVideoBkgd",
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
        cam1->devRotLoc(&AppDemo::devRot, &AppDemo::devLoc);
        cam1->background().texture(videoTexture);

        // Turn on main video
        CVCapture::instance()->videoType(VT_MAIN);

        // Create directional light for the sunlight
        SLLightDirect* sunLight = new SLLightDirect(am, s, 5.0f);
        sunLight->powers(1.0f, 1.0f, 1.0f);
        sunLight->attenuation(1, 0, 0);
        sunLight->doSunPowerAdaptation(true);
        sunLight->createsShadows(true);
        sunLight->createShadowMap(-100, 150, SLVec2f(150, 150), SLVec2i(4096, 4096));
        sunLight->doSmoothShadows(true);
        sunLight->castsShadows(false);

        // Let the sun be rotated by time and location
        AppDemo::devLoc.sunLightNode(sunLight);

        SLAssimpImporter importer;
        SLNode*          bfh = importer.load(s->animManager(),
                                    am,
                                    dataPath + "erleb-AR/models/biel/Biel-BFH-Rolex.gltf",
                                    texPath);

        bfh->setMeshMat(matVideoBkgd, true);

        // Make terrain a video shine trough
        // bfh->findChild<SLNode>("Terrain")->setMeshMat(matVideoBkgd, true);

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
        SLNode* axis = new SLNode(new SLCoordAxis(am), "Axis Node");
        axis->scale(2);
        axis->rotate(-90, 1, 0, 0);

        SLNode* scene = new SLNode("Scene");
        s->root3D(scene);
        scene->addChild(sunLight);
        scene->addChild(axis);
        scene->addChild(bfh);
        scene->addChild(cam1);

        // initialize sensor stuff
        AppDemo::devLoc.originLatLonAlt(47.14271, 7.24337, 488.2);        // Ecke Giosa
        AppDemo::devLoc.defaultLatLonAlt(47.14260, 7.24310, 488.7 + 1.7); // auf Parkplatz
        AppDemo::devLoc.locMaxDistanceM(1000.0f);
        AppDemo::devLoc.improveOrigin(false);
        AppDemo::devLoc.useOriginAltitude(true);
        AppDemo::devLoc.hasOrigin(true);
        AppDemo::devLoc.offsetMode(LOM_twoFingerY);
        AppDemo::devRot.zeroYawAtStart(false);
        AppDemo::devRot.offsetMode(ROM_oneFingerX);

        // This loads the DEM file and overwrites the altitude of originLatLonAlt and defaultLatLonAlt
        SLstring tif = dataPath + "erleb-AR/models/biel/DEM_Biel-BFH_WGS84.tif";
        AppDemo::devLoc.loadGeoTiff(tif);

#if defined(SL_OS_MACIOS) || defined(SL_OS_ANDROID)
        AppDemo::devLoc.isUsed(true);
        AppDemo::devRot.isUsed(true);
        cam1->camAnim(SLCamAnim::CA_deviceRotLocYUp);
#else
        AppDemo::devLoc.isUsed(false);
        AppDemo::devRot.isUsed(false);
        SLVec3d pos_d = AppDemo::devLoc.defaultENU() - AppDemo::devLoc.originENU();
        SLVec3f pos_f((SLfloat)pos_d.x, (SLfloat)pos_d.y, (SLfloat)pos_d.z);
        cam1->translation(pos_f);
        cam1->focalDist(pos_f.length());
        cam1->lookAt(SLVec3f::ZERO);
        cam1->camAnim(SLCamAnim::CA_turntableYUp);
#endif

        sv->doWaitOnIdle(false); // for constant video feed
        sv->camera(cam1);
        sv->drawBits()->on(SL_DB_ONLYEDGES);
    }
    else if (sceneID == SID_ErlebARAugustaRauricaTmp) //...........................................
    {
        s->name("Augusta Raurica Temple AR");
        s->info(s->name());

        // Create video texture on global pointer updated in AppDemoVideo
        videoTexture = new SLGLTexture(am, texPath + "LiveVideoError.png", GL_LINEAR, GL_LINEAR);
        videoTexture->texType(TT_videoBkgd);

        // Create see through video background material without shadow mapping
        SLMaterial* matVideoBkgd = new SLMaterial(am, "matVideoBkgd", videoTexture);
        matVideoBkgd->reflectionModel(RM_Custom);

        // Create see through video background material with shadow mapping
        SLMaterial* matVideoBkgdSM = new SLMaterial(am, "matVideoBkgdSM", videoTexture);
        matVideoBkgdSM->reflectionModel(RM_Custom);
        matVideoBkgdSM->ambient(SLCol4f(0.6f, 0.6f, 0.6f));
        matVideoBkgdSM->getsShadows(true);

        // Set the camera
        SLCamera* cam1 = new SLCamera("Camera 1");
        cam1->translation(0, 50, -150);
        cam1->lookAt(0, 0, 0);
        cam1->clipNear(1);
        cam1->clipFar(400);
        cam1->focalDist(150);
        cam1->devRotLoc(&AppDemo::devRot, &AppDemo::devLoc);
        cam1->background().texture(videoTexture);

        // Turn on main video
        CVCapture::instance()->videoType(VT_MAIN);

        string shdDir = shaderPath;
        string texDir = texPath;
        string datDir = dataPath + "erleb-AR/models/augst/";

        // Create directional light for the sunlight
        SLLightDirect* sunLight = new SLLightDirect(am, s, 1.0f);
        sunLight->translate(-42, 10, 13);
        sunLight->powers(1.0f, 1.5f, 1.0f);
        sunLight->attenuation(1, 0, 0);
        sunLight->doSunPowerAdaptation(true);
        sunLight->createsShadows(true);
        sunLight->createShadowMapAutoSize(cam1, SLVec2i(2048, 2048), 4);
        sunLight->shadowMap()->cascadesFactor(3.0);
        // sunLight->createShadowMap(-100, 250, SLVec2f(210, 180), SLVec2i(4096, 4096));
        sunLight->doSmoothShadows(true);
        sunLight->castsShadows(false);
        sunLight->shadowMinBias(0.001f);
        sunLight->shadowMaxBias(0.001f);
        AppDemo::devLoc.sunLightNode(sunLight); // Let the sun be rotated by time and location

        // Load main model
        SLAssimpImporter importer; //(LV_diagnostic);
        SLNode*          thtAndTmp = importer.load(s->animManager(),
                                          am,
                                          datDir + "augst-thtL2-tmpL1.gltf",
                                          texDir,
                                          nullptr,
                                          true,    // delete tex images after build
                                          true,    // only meshes
                                          nullptr, // no replacement material
                                          0.4f);   // 40% ambient reflection

        // Rotate to the true geographic rotation
        thtAndTmp->rotate(16.7f, 0, 1, 0, TS_parent);

        // Let the video shine through on some objects without shadow mapping
        SLNode* tmpUnderground = thtAndTmp->findChild<SLNode>("TmpUnderground");
        if (tmpUnderground) tmpUnderground->setMeshMat(matVideoBkgd, true);
        SLNode* thtUnderground = thtAndTmp->findChild<SLNode>("ThtUnderground");
        if (thtUnderground) thtUnderground->setMeshMat(matVideoBkgd, true);

        // Let the video shine through on some objects with shadow mapping
        SLNode* tmpFloor = thtAndTmp->findChild<SLNode>("TmpFloor");
        if (tmpFloor) tmpFloor->setMeshMat(matVideoBkgdSM, true);

        SLNode* terrain = thtAndTmp->findChild<SLNode>("Terrain");
        if (terrain)
        {
            terrain->setMeshMat(matVideoBkgdSM, true);
            terrain->castsShadows(false);
        }
        SLNode* thtFrontTerrain = thtAndTmp->findChild<SLNode>("ThtFrontTerrain");
        if (thtFrontTerrain)
        {
            thtFrontTerrain->setMeshMat(matVideoBkgdSM, true);
            thtFrontTerrain->castsShadows(false);
        }

        // Add axis object a world origin
        SLNode* axis = new SLNode(new SLCoordAxis(am), "Axis Node");
        axis->setDrawBitsRec(SL_DB_MESHWIRED, false);
        axis->rotate(-90, 1, 0, 0);
        axis->castsShadows(false);

        // Set some ambient light
        thtAndTmp->updateMeshMat([](SLMaterial* m)
                                 { m->ambient(SLCol4f(.25f, .25f, .25f)); },
                                 true);
        SLNode* scene = new SLNode("Scene");
        s->root3D(scene);
        scene->addChild(sunLight);
        scene->addChild(axis);
        scene->addChild(thtAndTmp);
        scene->addChild(cam1);

        // initialize sensor stuff
        AppDemo::devLoc.useOriginAltitude(false);
        AppDemo::devLoc.nameLocations().push_back(SLLocation("Center of theatre, Origin", 47, 31, 59.461, 7, 43, 19.446, 282.6));
        AppDemo::devLoc.nameLocations().push_back(SLLocation("Treppe Tempel", 47, 31, 58.933, 7, 43, 16.799, 290.5 + 1.7));
        AppDemo::devLoc.nameLocations().push_back(SLLocation("Abzweigung (Dolendeckel)", 47, 31, 57.969, 7, 43, 17.946, 286.5 + 1.7));
        AppDemo::devLoc.nameLocations().push_back(SLLocation("Marker bei Tempel", 47, 31, 59.235, 7, 43, 15.161, 293.1 + 1.7));
        AppDemo::devLoc.nameLocations().push_back(SLLocation("Theater 1. Rang Zugang Ost", 47, 31, 59.698, 7, 43, 20.518, 291.0 + 1.7));
        AppDemo::devLoc.nameLocations().push_back(SLLocation("Theater 1. Rang Nord", 47, 32, 0.216, 7, 43, 19.173, 291.0 + 1.7));
        AppDemo::devLoc.originLatLonAlt(AppDemo::devLoc.nameLocations()[0].posWGS84LatLonAlt);
        AppDemo::devLoc.activeNamedLocation(1);   // This sets the location 1 as defaultENU
        AppDemo::devLoc.locMaxDistanceM(1000.0f); // Max. allowed distance to origin
        AppDemo::devLoc.improveOrigin(false);     // No autom. origin improvement
        AppDemo::devLoc.hasOrigin(true);
        AppDemo::devLoc.offsetMode(LOM_twoFingerY);
        AppDemo::devRot.zeroYawAtStart(false);
        AppDemo::devRot.offsetMode(ROM_oneFingerX);

        // This loads the DEM file and overwrites the altitude of originLatLonAlt and defaultLatLonAlt
        SLstring tif = datDir + "DTM-Theater-Tempel-WGS84.tif";
        AppDemo::devLoc.loadGeoTiff(tif);

#if defined(SL_OS_MACIOS) || defined(SL_OS_ANDROID)
        AppDemo::devLoc.isUsed(true);
        AppDemo::devRot.isUsed(true);
        cam1->camAnim(SLCamAnim::CA_deviceRotLocYUp);
#else
        AppDemo::devLoc.isUsed(false);
        AppDemo::devRot.isUsed(false);
        SLVec3d pos_d = AppDemo::devLoc.defaultENU() - AppDemo::devLoc.originENU();
        SLVec3f pos_f((SLfloat)pos_d.x, (SLfloat)pos_d.y, (SLfloat)pos_d.z);
        cam1->translation(pos_f);
        cam1->focalDist(pos_f.length());
        cam1->lookAt(SLVec3f::ZERO);
        cam1->camAnim(SLCamAnim::CA_turntableYUp);
#endif

        sv->doWaitOnIdle(false); // for constant video feed
        sv->camera(cam1);
    }
    else if (sceneID == SID_ErlebARAugustaRauricaTht) //...........................................
    {
        s->name("Augusta Raurica Theater AR");
        s->info(s->name());

        // Create video texture on global pointer updated in AppDemoVideo
        videoTexture = new SLGLTexture(am, texPath + "LiveVideoError.png", GL_LINEAR, GL_LINEAR);
        videoTexture->texType(TT_videoBkgd);

        // Create see through video background material without shadow mapping
        SLMaterial* matVideoBkgd = new SLMaterial(am, "matVideoBkgd", videoTexture);
        matVideoBkgd->reflectionModel(RM_Custom);

        // Create see through video background material with shadow mapping
        SLMaterial* matVideoBkgdSM = new SLMaterial(am, "matVideoBkgdSM", videoTexture);
        matVideoBkgdSM->reflectionModel(RM_Custom);
        matVideoBkgdSM->ambient(SLCol4f(0.6f, 0.6f, 0.6f));
        matVideoBkgdSM->getsShadows(true);

        // Setup the camera
        SLCamera* cam1 = new SLCamera("Camera 1");
        cam1->translation(0, 50, -150);
        cam1->lookAt(0, 0, 0);
        cam1->clipNear(1);
        cam1->clipFar(400);
        cam1->focalDist(150);
        cam1->devRotLoc(&AppDemo::devRot, &AppDemo::devLoc);
        cam1->background().texture(videoTexture);

        // Turn on main video
        CVCapture::instance()->videoType(VT_MAIN);

        string shdDir = shaderPath;
        string texDir = texPath;
        string datDir = dataPath + "erleb-AR/models/augst/";

        // Create directional light for the sunlight
        SLLightDirect* sunLight = new SLLightDirect(am, s, 1.0f);
        sunLight->translate(-42, 10, 13);
        sunLight->powers(1.0f, 1.5f, 1.0f);
        sunLight->attenuation(1, 0, 0);
        sunLight->doSunPowerAdaptation(true);
        sunLight->createsShadows(true);
        sunLight->createShadowMapAutoSize(cam1, SLVec2i(2048, 2048), 4);
        sunLight->shadowMap()->cascadesFactor(3.0);
        // sunLight->createShadowMap(-100, 250, SLVec2f(210, 180), SLVec2i(4096, 4096));
        sunLight->doSmoothShadows(true);
        sunLight->castsShadows(false);
        sunLight->shadowMinBias(0.001f);
        sunLight->shadowMaxBias(0.001f);
        AppDemo::devLoc.sunLightNode(sunLight); // Let the sun be rotated by time and location

        // Load main model
        SLAssimpImporter importer; //(LV_diagnostic);
        SLNode*          thtAndTmp = importer.load(s->animManager(),
                                          am,
                                          datDir + "augst-thtL1-tmpL2.gltf",
                                          texDir,
                                          nullptr,
                                          true,    // delete tex images after build
                                          true,    // only meshes
                                          nullptr, // no replacement material
                                          0.4f);   // 40% ambient reflection

        // Rotate to the true geographic rotation
        thtAndTmp->rotate(16.7f, 0, 1, 0, TS_parent);

        // Let the video shine through on some objects without shadow mapping
        SLNode* tmpUnderground = thtAndTmp->findChild<SLNode>("TmpUnderground");
        if (tmpUnderground) tmpUnderground->setMeshMat(matVideoBkgd, true);
        SLNode* thtUnderground = thtAndTmp->findChild<SLNode>("ThtUnderground");
        if (thtUnderground) thtUnderground->setMeshMat(matVideoBkgd, true);

        // Let the video shine through on some objects with shadow mapping
        SLNode* tmpFloor = thtAndTmp->findChild<SLNode>("TmpFloor");
        if (tmpFloor) tmpFloor->setMeshMat(matVideoBkgdSM, true);

        SLNode* terrain = thtAndTmp->findChild<SLNode>("Terrain");
        if (terrain)
        {
            terrain->setMeshMat(matVideoBkgdSM, true);
            terrain->castsShadows(false);
        }
        SLNode* thtFrontTerrain = thtAndTmp->findChild<SLNode>("ThtFrontTerrain");
        if (thtFrontTerrain)
        {
            thtFrontTerrain->setMeshMat(matVideoBkgdSM, true);
            thtFrontTerrain->castsShadows(false);
        }

        // Add axis object a world origin
        SLNode* axis = new SLNode(new SLCoordAxis(am), "Axis Node");
        axis->setDrawBitsRec(SL_DB_MESHWIRED, false);
        axis->rotate(-90, 1, 0, 0);
        axis->castsShadows(false);

        // Set some ambient light
        thtAndTmp->updateMeshMat([](SLMaterial* m)
                                 { m->ambient(SLCol4f(.25f, .25f, .25f)); },
                                 true);
        SLNode* scene = new SLNode("Scene");
        s->root3D(scene);
        scene->addChild(sunLight);
        scene->addChild(axis);
        scene->addChild(thtAndTmp);
        scene->addChild(cam1);

        // initialize sensor stuff
        AppDemo::devLoc.useOriginAltitude(false);
        AppDemo::devLoc.nameLocations().push_back(SLLocation("Center of theatre, Origin", 47, 31, 59.461, 7, 43, 19.446, 282.6));
        AppDemo::devLoc.nameLocations().push_back(SLLocation("Treppe Tempel", 47, 31, 58.933, 7, 43, 16.799, 290.5 + 1.7));
        AppDemo::devLoc.nameLocations().push_back(SLLocation("Abzweigung (Dolendeckel)", 47, 31, 57.969, 7, 43, 17.946, 286.5 + 1.7));
        AppDemo::devLoc.nameLocations().push_back(SLLocation("Marker bei Tempel", 47, 31, 59.235, 7, 43, 15.161, 293.1 + 1.7));
        AppDemo::devLoc.nameLocations().push_back(SLLocation("Theater 1. Rang Zugang Ost", 47, 31, 59.698, 7, 43, 20.518, 291.0 + 1.7));
        AppDemo::devLoc.nameLocations().push_back(SLLocation("Theater 1. Rang Nord", 47, 32, 0.216, 7, 43, 19.173, 291.0 + 1.7));
        AppDemo::devLoc.originLatLonAlt(AppDemo::devLoc.nameLocations()[0].posWGS84LatLonAlt);
        AppDemo::devLoc.activeNamedLocation(1);   // This sets the location 1 as defaultENU
        AppDemo::devLoc.locMaxDistanceM(1000.0f); // Max. allowed distance to origin
        AppDemo::devLoc.improveOrigin(false);     // No autom. origin improvement
        AppDemo::devLoc.hasOrigin(true);
        AppDemo::devLoc.offsetMode(LOM_twoFingerY);
        AppDemo::devRot.zeroYawAtStart(false);
        AppDemo::devRot.offsetMode(ROM_oneFingerX);

        // This loads the DEM file and overwrites the altitude of originLatLonAlt and defaultLatLonAlt
        SLstring tif = datDir + "DTM-Theater-Tempel-WGS84.tif";
        AppDemo::devLoc.loadGeoTiff(tif);

#if defined(SL_OS_MACIOS) || defined(SL_OS_ANDROID)
        AppDemo::devLoc.isUsed(true);
        AppDemo::devRot.isUsed(true);
        cam1->camAnim(SLCamAnim::CA_deviceRotLocYUp);
#else
        AppDemo::devLoc.isUsed(false);
        AppDemo::devRot.isUsed(false);
        SLVec3d pos_d = AppDemo::devLoc.defaultENU() - AppDemo::devLoc.originENU();
        SLVec3f pos_f((SLfloat)pos_d.x, (SLfloat)pos_d.y, (SLfloat)pos_d.z);
        cam1->translation(pos_f);
        cam1->focalDist(pos_f.length());
        cam1->lookAt(SLVec3f::ZERO);
        cam1->camAnim(SLCamAnim::CA_turntableYUp);
#endif

        sv->doWaitOnIdle(false); // for constant video feed
        sv->camera(cam1);
    }
    else if (sceneID == SID_ErlebARAugustaRauricaTmpTht) //........................................
    {
        s->name("Augusta Raurica AR Temple and Theater");
        s->info(s->name());

        // Create video texture on global pointer updated in AppDemoVideo
        videoTexture = new SLGLTexture(am, texPath + "LiveVideoError.png", GL_LINEAR, GL_LINEAR);
        videoTexture->texType(TT_videoBkgd);

        // Create see through video background material without shadow mapping
        SLMaterial* matVideoBkgd = new SLMaterial(am, "matVideoBkgd", videoTexture);
        matVideoBkgd->reflectionModel(RM_Custom);

        // Create see through video background material with shadow mapping
        SLMaterial* matVideoBkgdSM = new SLMaterial(am, "matVideoBkgdSM", videoTexture);
        matVideoBkgdSM->reflectionModel(RM_Custom);
        matVideoBkgdSM->ambient(SLCol4f(0.6f, 0.6f, 0.6f));
        matVideoBkgdSM->getsShadows(true);

        // Setup the camera
        SLCamera* cam1 = new SLCamera("Camera 1");
        cam1->translation(0, 50, -150);
        cam1->lookAt(0, 0, 0);
        cam1->clipNear(1);
        cam1->clipFar(400);
        cam1->focalDist(150);
        cam1->devRotLoc(&AppDemo::devRot, &AppDemo::devLoc);
        cam1->background().texture(videoTexture);

        // Turn on main video
        CVCapture::instance()->videoType(VT_MAIN);

        string shdDir = shaderPath;
        string texDir = texPath;
        string datDir = dataPath + "erleb-AR/models/augst/";

        // Create directional light for the sunlight
        SLLightDirect* sunLight = new SLLightDirect(am, s, 1.0f);
        sunLight->translate(-42, 10, 13);
        sunLight->powers(1.0f, 1.5f, 1.0f);
        sunLight->attenuation(1, 0, 0);
        sunLight->doSunPowerAdaptation(true);
        sunLight->createsShadows(true);
        sunLight->createShadowMapAutoSize(cam1, SLVec2i(2048, 2048), 4);
        sunLight->shadowMap()->cascadesFactor(3.0);
        // Old stanard single map shadow map
        // sunLight->createShadowMap(-100, 250, SLVec2f(210, 180), SLVec2i(4096, 4096));
        sunLight->doSmoothShadows(true);
        sunLight->castsShadows(false);
        sunLight->shadowMinBias(0.001f);
        sunLight->shadowMaxBias(0.001f);
        AppDemo::devLoc.sunLightNode(sunLight); // Let the sun be rotated by time and location

        // Load main model
        SLAssimpImporter importer; //(LV_diagnostic);
        SLNode*          thtAndTmp = importer.load(s->animManager(),
                                          am,
                                          datDir + "augst-thtL1L2-tmpL1L2.gltf",
                                          texDir,
                                          nullptr,
                                          true,    // delete tex images after build
                                          true,    // only meshes
                                          nullptr, // no replacement material
                                          0.4f);   // 40% ambient reflection

        // Rotate to the true geographic rotation
        thtAndTmp->rotate(16.7f, 0, 1, 0, TS_parent);

        // Let the video shine through on some objects without shadow mapping
        SLNode* tmpUnderground = thtAndTmp->findChild<SLNode>("TmpUnderground");
        if (tmpUnderground) tmpUnderground->setMeshMat(matVideoBkgd, true);
        SLNode* thtUnderground = thtAndTmp->findChild<SLNode>("ThtUnderground");
        if (thtUnderground) thtUnderground->setMeshMat(matVideoBkgd, true);

        // Let the video shine through on some objects with shadow mapping
        SLNode* tmpFloor = thtAndTmp->findChild<SLNode>("TmpFloor");
        if (tmpFloor) tmpFloor->setMeshMat(matVideoBkgdSM, true);

        SLNode* terrain = thtAndTmp->findChild<SLNode>("Terrain");
        if (terrain)
        {
            terrain->setMeshMat(matVideoBkgdSM, true);
            terrain->castsShadows(false);
        }
        SLNode* thtFrontTerrain = thtAndTmp->findChild<SLNode>("ThtFrontTerrain");
        if (thtFrontTerrain)
        {
            thtFrontTerrain->setMeshMat(matVideoBkgdSM, true);
            thtFrontTerrain->castsShadows(false);
        }

        // Add axis object a world origin
        SLNode* axis = new SLNode(new SLCoordAxis(am), "Axis Node");
        axis->setDrawBitsRec(SL_DB_MESHWIRED, false);
        axis->rotate(-90, 1, 0, 0);
        axis->castsShadows(false);

        // Set some ambient light
        thtAndTmp->updateMeshMat([](SLMaterial* m)
                                 { m->ambient(SLCol4f(.25f, .25f, .25f)); },
                                 true);
        SLNode* scene = new SLNode("Scene");
        s->root3D(scene);
        scene->addChild(sunLight);
        scene->addChild(axis);
        scene->addChild(thtAndTmp);
        scene->addChild(cam1);

        // initialize sensor stuff
        AppDemo::devLoc.useOriginAltitude(false);
        AppDemo::devLoc.nameLocations().push_back(SLLocation("Center of theatre, Origin", 47, 31, 59.461, 7, 43, 19.446, 282.6));
        AppDemo::devLoc.nameLocations().push_back(SLLocation("Treppe Tempel", 47, 31, 58.933, 7, 43, 16.799, 290.5 + 1.7));
        AppDemo::devLoc.nameLocations().push_back(SLLocation("Abzweigung (Dolendeckel)", 47, 31, 57.969, 7, 43, 17.946, 286.5 + 1.7));
        AppDemo::devLoc.nameLocations().push_back(SLLocation("Marker bei Tempel", 47, 31, 59.235, 7, 43, 15.161, 293.1 + 1.7));
        AppDemo::devLoc.nameLocations().push_back(SLLocation("Theater 1. Rang Zugang Ost", 47, 31, 59.698, 7, 43, 20.518, 291.0 + 1.7));
        AppDemo::devLoc.nameLocations().push_back(SLLocation("Theater 1. Rang Nord", 47, 32, 0.216, 7, 43, 19.173, 291.0 + 1.7));
        AppDemo::devLoc.originLatLonAlt(AppDemo::devLoc.nameLocations()[0].posWGS84LatLonAlt);
        AppDemo::devLoc.activeNamedLocation(1);   // This sets the location 1 as defaultENU
        AppDemo::devLoc.locMaxDistanceM(1000.0f); // Max. allowed distance to origin
        AppDemo::devLoc.improveOrigin(false);     // No autom. origin improvement
        AppDemo::devLoc.hasOrigin(true);
        AppDemo::devLoc.offsetMode(LOM_twoFingerY);
        AppDemo::devRot.zeroYawAtStart(false);
        AppDemo::devRot.offsetMode(ROM_oneFingerX);

        // Level of Detail switch for Temple and Theater
        SLNode* tmpAltar = thtAndTmp->findChild<SLNode>("TmpAltar");
        SLNode* tmpL1    = thtAndTmp->findChild<SLNode>("Tmp-L1");
        SLNode* tmpL2    = thtAndTmp->findChild<SLNode>("Tmp-L2");
        SLNode* thtL1    = thtAndTmp->findChild<SLNode>("Tht-L1");
        SLNode* thtL2    = thtAndTmp->findChild<SLNode>("Tht-L2");
        thtL1->drawBits()->set(SL_DB_HIDDEN, false);
        thtL2->drawBits()->set(SL_DB_HIDDEN, true);
        tmpL1->drawBits()->set(SL_DB_HIDDEN, false);
        tmpL2->drawBits()->set(SL_DB_HIDDEN, true);

        // Add level of detail switch callback lambda
        cam1->onCamUpdateCB([=](SLSceneView* sv)
                            {
            SLVec3f posCam     = sv->camera()->updateAndGetWM().translation();
            SLVec3f posAlt     = tmpAltar->updateAndGetWM().translation();
            SLVec3f distCamAlt = posCam - posAlt;
            float   tmpDist    = distCamAlt.length();
            float   thtDist    = posCam.length();

            // If the temple is closer than the theater activate level 1 and deactivate level 2
            if (tmpDist < thtDist)
            {
                thtL1->drawBits()->set(SL_DB_HIDDEN, true);
                thtL2->drawBits()->set(SL_DB_HIDDEN, false);
                tmpL1->drawBits()->set(SL_DB_HIDDEN, false);
                tmpL2->drawBits()->set(SL_DB_HIDDEN, true);
            }
            else
            {
                thtL1->drawBits()->set(SL_DB_HIDDEN, false);
                thtL2->drawBits()->set(SL_DB_HIDDEN, true);
                tmpL1->drawBits()->set(SL_DB_HIDDEN, true);
                tmpL2->drawBits()->set(SL_DB_HIDDEN, false);
            } });

        // This loads the DEM file and overwrites the altitude of originLatLonAlt and defaultLatLonAlt
        SLstring tif = datDir + "DTM-Theater-Tempel-WGS84.tif";
        AppDemo::devLoc.loadGeoTiff(tif);

#if defined(SL_OS_MACIOS) || defined(SL_OS_ANDROID)
        AppDemo::devLoc.isUsed(true);
        AppDemo::devRot.isUsed(true);
        cam1->camAnim(SLCamAnim::CA_deviceRotLocYUp);
#else
        AppDemo::devLoc.isUsed(false);
        AppDemo::devRot.isUsed(false);
        SLVec3d pos_d = AppDemo::devLoc.defaultENU() - AppDemo::devLoc.originENU();
        SLVec3f pos_f((SLfloat)pos_d.x, (SLfloat)pos_d.y, (SLfloat)pos_d.z);
        cam1->translation(pos_f);
        cam1->focalDist(pos_f.length());
        cam1->lookAt(SLVec3f::ZERO);
        cam1->camAnim(SLCamAnim::CA_turntableYUp);
#endif

        sv->doWaitOnIdle(false); // for constant video feed
        sv->camera(cam1);
    }
    else if (sceneID == SID_ErlebARAventicumAmphiteatre) //........................................
    {
        s->name("Aventicum Amphitheatre AR (AO)");
        s->info("Augmented Reality for Aventicum Amphitheatre (AO)");

        // Create video texture on global pointer updated in AppDemoVideo
        videoTexture = new SLGLTexture(am, texPath + "LiveVideoError.png", GL_LINEAR, GL_LINEAR);
        videoTexture->texType(TT_videoBkgd);

        // Create see through video background material without shadow mapping
        SLMaterial* matVideoBkgd = new SLMaterial(am, "matVideoBkgd", videoTexture);
        matVideoBkgd->reflectionModel(RM_Custom);

        // Create see through video background material with shadow mapping
        SLMaterial* matVideoBkgdSM = new SLMaterial(am, "matVideoBkgdSM", videoTexture);
        matVideoBkgdSM->reflectionModel(RM_Custom);
        matVideoBkgdSM->ambient(SLCol4f(0.6f, 0.6f, 0.6f));
        matVideoBkgdSM->getsShadows(true);

        // Setup the camera
        SLCamera* cam1 = new SLCamera("Camera 1");
        cam1->translation(0, 50, -150);
        cam1->lookAt(0, 0, 0);
        cam1->clipNear(1);
        cam1->clipFar(400);
        cam1->focalDist(150);
        cam1->setInitialState();
        cam1->devRotLoc(&AppDemo::devRot, &AppDemo::devLoc);
        cam1->background().texture(videoTexture);

        // Turn on main video
        CVCapture::instance()->videoType(VT_MAIN);

        // Create directional light for the sunlight
        SLLightDirect* sunLight = new SLLightDirect(am, s, 1.0f);
        sunLight->powers(1.0f, 1.5f, 1.0f);
        sunLight->attenuation(1, 0, 0);
        sunLight->translation(0, 1, 0);
        sunLight->doSunPowerAdaptation(true);
        sunLight->createsShadows(true);
        sunLight->createShadowMapAutoSize(cam1, SLVec2i(2048, 2048), 4);
        sunLight->shadowMap()->cascadesFactor(3.0);
        // sunLight->createShadowMap(-70, 70, SLVec2f(140, 100), SLVec2i(4096, 4096));
        sunLight->doSmoothShadows(true);
        sunLight->castsShadows(false);
        sunLight->shadowMinBias(0.001f);
        sunLight->shadowMaxBias(0.003f);

        // Let the sun be rotated by time and location
        AppDemo::devLoc.sunLightNode(sunLight);

        SLAssimpImporter importer;
        SLNode*          amphiTheatre = importer.load(s->animManager(),
                                             am,
                                             dataPath + "erleb-AR/models/avenches/avenches-amphitheater.gltf",
                                             texPath,
                                             nullptr,
                                             false,   // delete tex images after build
                                             true,    // only meshes
                                             nullptr, // no replacement material
                                             0.4f);   // 40% ambient reflection

        // Rotate to the true geographic rotation
        amphiTheatre->rotate(13.25f, 0, 1, 0, TS_parent);

        // Let the video shine through some objects
        amphiTheatre->findChild<SLNode>("Tht-Aussen-Untergrund")->setMeshMat(matVideoBkgd, true);
        amphiTheatre->findChild<SLNode>("Tht-Eingang-Ost-Boden")->setMeshMat(matVideoBkgdSM, true);
        amphiTheatre->findChild<SLNode>("Tht-Arenaboden")->setMeshMat(matVideoBkgdSM, true);
        // amphiTheatre->findChild<SLNode>("Tht-Aussen-Terrain")->setMeshMat(matVideoBkgdSM, true);

        // Add axis object a world origin
        SLNode* axis = new SLNode(new SLCoordAxis(am), "Axis Node");
        axis->setDrawBitsRec(SL_DB_MESHWIRED, false);
        axis->rotate(-90, 1, 0, 0);
        axis->castsShadows(false);

        SLNode* scene = new SLNode("Scene");
        s->root3D(scene);
        scene->addChild(sunLight);
        scene->addChild(axis);
        scene->addChild(amphiTheatre);
        scene->addChild(cam1);

        // initialize sensor stuff
        AppDemo::devLoc.useOriginAltitude(false);
        AppDemo::devLoc.nameLocations().push_back(SLLocation("Arena Centre, Origin", 46, 52, 51.685, 7, 2, 33.458, 461.4));
        AppDemo::devLoc.nameLocations().push_back(SLLocation("Entrance East, Manhole Cover", 46, 52, 52.344, 7, 2, 37.600, 461.4 + 1.7));
        AppDemo::devLoc.nameLocations().push_back(SLLocation("Arena, Sewer Cover West", 46, 52, 51.484, 7, 2, 32.307, 461.3 + 1.7));
        AppDemo::devLoc.nameLocations().push_back(SLLocation("Arena, Sewer Cover East", 46, 52, 51.870, 7, 2, 34.595, 461.1 + 1.7));
        AppDemo::devLoc.nameLocations().push_back(SLLocation("Stand South, Sewer Cover", 46, 52, 50.635, 7, 2, 34.099, 471.7 + 1.7));
        AppDemo::devLoc.nameLocations().push_back(SLLocation("Stand West, Sewer Cover", 46, 52, 51.889, 7, 2, 31.567, 471.7 + 1.7));
        AppDemo::devLoc.originLatLonAlt(AppDemo::devLoc.nameLocations()[0].posWGS84LatLonAlt);
        AppDemo::devLoc.activeNamedLocation(1);   // This sets the location 1 as defaultENU
        AppDemo::devLoc.locMaxDistanceM(1000.0f); // Max. Distanz. zum Nullpunkt
        AppDemo::devLoc.improveOrigin(false);     // Keine autom. Verbesserung vom Origin
        AppDemo::devLoc.hasOrigin(true);
        AppDemo::devLoc.offsetMode(LOM_twoFingerY);
        AppDemo::devRot.zeroYawAtStart(false);
        AppDemo::devRot.offsetMode(ROM_oneFingerX);

        // This loads the DEM file and overwrites the altitude of originLatLonAlt and defaultLatLonAlt
        SLstring tif = dataPath + "erleb-AR/models/avenches/DTM-Aventicum-WGS84.tif";
        AppDemo::devLoc.loadGeoTiff(tif);

#if defined(SL_OS_MACIOS) || defined(SL_OS_ANDROID)
        AppDemo::devLoc.isUsed(true);
        AppDemo::devRot.isUsed(true);
        cam1->camAnim(SLCamAnim::CA_deviceRotLocYUp);
#else
        AppDemo::devLoc.isUsed(false);
        AppDemo::devRot.isUsed(false);
        SLVec3d pos_d = AppDemo::devLoc.defaultENU() - AppDemo::devLoc.originENU();
        SLVec3f pos_f((SLfloat)pos_d.x, (SLfloat)pos_d.y, (SLfloat)pos_d.z);
        cam1->translation(pos_f);
        cam1->focalDist(pos_f.length());
        cam1->lookAt(SLVec3f::ZERO);
        cam1->camAnim(SLCamAnim::CA_turntableYUp);
#endif

        sv->doWaitOnIdle(false); // for constant video feed
        sv->camera(cam1);
    }
    else if (sceneID == SID_ErlebARAventicumCigognier) //..........................................
    {
        s->name("Aventicum Cigognier AR (AO)");
        s->info("Augmented Reality for Aventicum Cigognier Temple");

        // Create video texture on global pointer updated in AppDemoVideo
        videoTexture = new SLGLTexture(am, texPath + "LiveVideoError.png", GL_LINEAR, GL_LINEAR);
        videoTexture->texType(TT_videoBkgd);

        // Create see through video background material without shadow mapping
        SLMaterial* matVideoBkgd = new SLMaterial(am, "matVideoBkgd", videoTexture);
        matVideoBkgd->reflectionModel(RM_Custom);

        // Create see through video background material with shadow mapping
        SLMaterial* matVideoBkgdSM = new SLMaterial(am, "matVideoBkgdSM", videoTexture);
        matVideoBkgdSM->reflectionModel(RM_Custom);
        matVideoBkgdSM->ambient(SLCol4f(0.6f, 0.6f, 0.6f));
        matVideoBkgdSM->getsShadows(true);

        // Setup the camera
        SLCamera* cam1 = new SLCamera("Camera 1");
        cam1->translation(0, 50, -150);
        cam1->lookAt(0, 0, 0);
        cam1->clipNear(1);
        cam1->clipFar(400);
        cam1->focalDist(150);
        cam1->setInitialState();
        cam1->devRotLoc(&AppDemo::devRot, &AppDemo::devLoc);
        cam1->background().texture(videoTexture);

        // Turn on main video
        CVCapture::instance()->videoType(VT_MAIN);

        // Create directional light for the sunlight
        SLLightDirect* sunLight = new SLLightDirect(am, s, 1.0f);
        sunLight->powers(1.0f, 1.0f, 1.0f);
        sunLight->attenuation(1, 0, 0);
        sunLight->translation(0, 10, 0);
        sunLight->lookAt(10, 0, 10);
        sunLight->doSunPowerAdaptation(true);
        sunLight->createsShadows(true);
        sunLight->createShadowMapAutoSize(cam1, SLVec2i(2048, 2048), 4);
        sunLight->shadowMap()->cascadesFactor(3.0);
        // sunLight->createShadowMap(-70, 120, SLVec2f(150, 150), SLVec2i(2048, 2048));
        sunLight->doSmoothShadows(true);
        sunLight->castsShadows(false);
        sunLight->shadowMinBias(0.001f);
        sunLight->shadowMaxBias(0.003f);

        // Let the sun be rotated by time and location
        AppDemo::devLoc.sunLightNode(sunLight);

        SLAssimpImporter importer;
        SLNode*          cigognier = importer.load(s->animManager(),
                                          am,
                                          dataPath + "erleb-AR/models/avenches/avenches-cigognier.gltf",
                                          texPath,
                                          nullptr,
                                          true,    // delete tex images after build
                                          true,    // only meshes
                                          nullptr, // no replacement material
                                          0.4f);   // 40% ambient reflection

        // Rotate to the true geographic rotation
        cigognier->rotate(-36.52f, 0, 1, 0, TS_parent);

        // Let the video shine through some objects
        cigognier->findChild<SLNode>("Tmp-Sol-Pelouse")->setMeshMat(matVideoBkgdSM, true);
        cigognier->findChild<SLNode>("Tmp-Souterrain")->setMeshMat(matVideoBkgd, true);

        // Add axis object a world origin
        SLNode* axis = new SLNode(new SLCoordAxis(am), "Axis Node");
        axis->setDrawBitsRec(SL_DB_MESHWIRED, false);
        axis->rotate(-90, 1, 0, 0);
        axis->castsShadows(false);

        SLNode* scene = new SLNode("Scene");
        s->root3D(scene);
        scene->addChild(sunLight);
        scene->addChild(axis);
        scene->addChild(cigognier);
        scene->addChild(cam1);

        // initialize sensor stuff
        AppDemo::devLoc.useOriginAltitude(false);
        AppDemo::devLoc.nameLocations().push_back(SLLocation("Center of place, Origin", 46, 52, 53.245, 7, 2, 47.198, 450.9));
        AppDemo::devLoc.nameLocations().push_back(SLLocation("At the altar", 46, 52, 53.107, 7, 2, 47.498, 450.9 + 1.7));
        AppDemo::devLoc.nameLocations().push_back(SLLocation("Old AR viewer", 46, 52, 53.666, 7, 2, 48.316, 451.0 + 1.7));
        AppDemo::devLoc.nameLocations().push_back(SLLocation("Temple Entrance in hall", 46, 52, 54.007, 7, 2, 45.702, 453.0 + 1.7));
        AppDemo::devLoc.originLatLonAlt(AppDemo::devLoc.nameLocations()[0].posWGS84LatLonAlt);
        AppDemo::devLoc.activeNamedLocation(1);   // This sets the location 1 as defaultENU
        AppDemo::devLoc.locMaxDistanceM(1000.0f); // Max. allowed distance from origin
        AppDemo::devLoc.improveOrigin(false);     // No auto improvement from
        AppDemo::devLoc.hasOrigin(true);
        AppDemo::devLoc.offsetMode(LOM_twoFingerY);
        AppDemo::devRot.zeroYawAtStart(false);
        AppDemo::devRot.offsetMode(ROM_oneFingerX);

        // This loads the DEM file and overwrites the altitude of originLatLonAlt and defaultLatLonAlt
        SLstring tif = dataPath + "erleb-AR/models/avenches/DTM-Aventicum-WGS84.tif";
        AppDemo::devLoc.loadGeoTiff(tif);

#if defined(SL_OS_MACIOS) || defined(SL_OS_ANDROID)
        AppDemo::devLoc.isUsed(true);
        AppDemo::devRot.isUsed(true);
        cam1->camAnim(SLCamAnim::CA_deviceRotLocYUp);
#else
        AppDemo::devLoc.isUsed(false);
        AppDemo::devRot.isUsed(false);
        SLVec3d pos_d = AppDemo::devLoc.defaultENU() - AppDemo::devLoc.originENU();
        SLVec3f pos_f((SLfloat)pos_d.x, (SLfloat)pos_d.y, (SLfloat)pos_d.z);
        cam1->translation(pos_f);
        cam1->focalDist(pos_f.length());
        cam1->lookAt(0, cam1->translationWS().y, 0);
        cam1->camAnim(SLCamAnim::CA_turntableYUp);
#endif

        sv->doWaitOnIdle(false); // for constant video feed
        sv->camera(cam1);
    }
    else if (sceneID == SID_ErlebARAventicumTheatre) //............................................
    {
        s->name("Aventicum Theatre AR");
        s->info("Augmented Reality for Aventicum Theatre");

        // Create video texture on global pointer updated in AppDemoVideo
        videoTexture = new SLGLTexture(am, texPath + "LiveVideoError.png", GL_LINEAR, GL_LINEAR);
        videoTexture->texType(TT_videoBkgd);

        // Create see through video background material without shadow mapping
        SLMaterial* matVideoBkgd = new SLMaterial(am, "matVideoBkgd", videoTexture);
        matVideoBkgd->reflectionModel(RM_Custom);

        // Create see through video background material with shadow mapping
        SLMaterial* matVideoBkgdSM = new SLMaterial(am, "matVideoBkgdSM", videoTexture);
        matVideoBkgdSM->reflectionModel(RM_Custom);
        matVideoBkgdSM->ambient(SLCol4f(0.6f, 0.6f, 0.6f));
        matVideoBkgdSM->getsShadows(true);

        // Setup the camera
        SLCamera* cam1 = new SLCamera("Camera 1");
        cam1->translation(0, 50, -150);
        cam1->lookAt(0, 0, 0);
        cam1->clipNear(1);
        cam1->clipFar(300);
        cam1->focalDist(150);
        cam1->setInitialState();
        cam1->devRotLoc(&AppDemo::devRot, &AppDemo::devLoc);
        cam1->background().texture(videoTexture);

        // Turn on main video
        CVCapture::instance()->videoType(VT_MAIN);

        // Create directional light for the sunlight
        SLLightDirect* sunLight = new SLLightDirect(am, s, 1.0f);
        sunLight->powers(1.0f, 1.0f, 1.0f);
        sunLight->attenuation(1, 0, 0);
        sunLight->translation(0, 1, 0);

        sunLight->doSunPowerAdaptation(true);
        sunLight->createsShadows(true);
        sunLight->createShadowMapAutoSize(cam1, SLVec2i(2048, 2048), 4);
        sunLight->shadowMap()->cascadesFactor(3.0);
        // sunLight->createShadowMap(-80, 100, SLVec2f(130, 130), SLVec2i(4096, 4096));
        sunLight->doSmoothShadows(true);
        sunLight->castsShadows(false);
        sunLight->shadowMinBias(0.001f);
        sunLight->shadowMaxBias(0.001f);

        // Let the sun be rotated by time and location
        AppDemo::devLoc.sunLightNode(sunLight);

        SLAssimpImporter importer;
        SLNode*          theatre = importer.load(s->animManager(),
                                        am,
                                        dataPath + "erleb-AR/models/avenches/avenches-theater.gltf",
                                        texPath,
                                        nullptr,
                                        true,    // delete tex images after build
                                        true,    // only meshes
                                        nullptr, // no replacement material
                                        0.4f);   // 40% ambient reflection

        // Rotate to the true geographic rotation
        theatre->rotate(-36.7f, 0, 1, 0, TS_parent);

        // Let the video shine through some objects
        theatre->findChild<SLNode>("Tht-Rasen")->setMeshMat(matVideoBkgdSM, true);
        theatre->findChild<SLNode>("Tht-Untergrund")->setMeshMat(matVideoBkgd, true);
        theatre->findChild<SLNode>("Tht-Boden")->setMeshMat(matVideoBkgdSM, true);
        theatre->findChild<SLNode>("Tht-Boden")->setDrawBitsRec(SL_DB_WITHEDGES, true);

        // Add axis object a world origin
        SLNode* axis = new SLNode(new SLCoordAxis(am), "Axis Node");
        axis->setDrawBitsRec(SL_DB_MESHWIRED, false);
        axis->rotate(-90, 1, 0, 0);
        axis->castsShadows(false);

        SLNode* scene = new SLNode("Scene");
        s->root3D(scene);
        scene->addChild(sunLight);
        scene->addChild(axis);
        scene->addChild(theatre);
        scene->addChild(cam1);

        // initialize sensor stuff
        // https://map.geo.admin.ch/?lang=de&topic=ech&bgLayer=ch.swisstopo.swissimage&layers=ch.swisstopo.zeitreihen,ch.bfs.gebaeude_wohnungs_register,ch.bav.haltestellen-oev,ch.swisstopo.swisstlm3d-wanderwege&layers_opacity=1,1,1,0.8&layers_visibility=false,false,false,false&layers_timestamp=18641231,,,&E=2570281&N=1192204&zoom=13&crosshair=marker
        AppDemo::devLoc.useOriginAltitude(false);
        AppDemo::devLoc.nameLocations().push_back(SLLocation("Center of theatre, Origin", 46, 52, 49.041, 7, 2, 55.543, 454.9));
        AppDemo::devLoc.nameLocations().push_back(SLLocation("On the stage", 46, 52, 49.221, 7, 2, 55.206, 455.5 + 1.7));
        AppDemo::devLoc.nameLocations().push_back(SLLocation("At the tree (N-E)", 46, 52, 50.791, 7, 2, 55.960, 455.5 + 1.7));
        AppDemo::devLoc.nameLocations().push_back(SLLocation("Over the entrance (S)", 46, 52, 48.162, 7, 2, 56.097, 464.0 + 1.7));
        AppDemo::devLoc.nameLocations().push_back(SLLocation("At the 3rd tree (S-W)", 46, 52, 48.140, 7, 2, 51.506, 455.0 + 1.7));
        AppDemo::devLoc.originLatLonAlt(AppDemo::devLoc.nameLocations()[0].posWGS84LatLonAlt);
        AppDemo::devLoc.activeNamedLocation(1);   // This sets the location 1 as defaultENU
        AppDemo::devLoc.locMaxDistanceM(1000.0f); // Max. Distanz. zum Nullpunkt
        AppDemo::devLoc.improveOrigin(false);     // Keine autom. Verbesserung vom Origin
        AppDemo::devLoc.hasOrigin(true);
        AppDemo::devLoc.offsetMode(LOM_twoFingerY);
        AppDemo::devRot.zeroYawAtStart(false);
        AppDemo::devRot.offsetMode(ROM_oneFingerX);

        // This loads the DEM file and overwrites the altitude of originLatLonAlt and defaultLatLonAlt
        SLstring tif = dataPath + "erleb-AR/models/avenches/DTM-Aventicum-WGS84.tif";
        AppDemo::devLoc.loadGeoTiff(tif);

#if defined(SL_OS_MACIOS) || defined(SL_OS_ANDROID)
        AppDemo::devLoc.isUsed(true);
        AppDemo::devRot.isUsed(true);
        cam1->camAnim(SLCamAnim::CA_deviceRotLocYUp);
#else
        AppDemo::devLoc.isUsed(false);
        AppDemo::devRot.isUsed(false);
        SLVec3d pos_d = AppDemo::devLoc.defaultENU() - AppDemo::devLoc.originENU();
        SLVec3f pos_f((SLfloat)pos_d.x, (SLfloat)pos_d.y, (SLfloat)pos_d.z);
        cam1->translation(pos_f);
        cam1->focalDist(pos_f.length());
        cam1->lookAt(SLVec3f::ZERO);
        cam1->camAnim(SLCamAnim::CA_turntableYUp);
#endif

        sv->doWaitOnIdle(false); // for constant video feed
        sv->camera(cam1);
    }
    else if (sceneID == SID_ErlebARSutzKirchrain18) //.............................................
    {
        s->name("Sutz, Kirchrain 18");
        s->info("Augmented Reality for Sutz, Kirchrain 18");

        // Create video texture on global pointer updated in AppDemoVideo
        videoTexture = new SLGLTexture(am, texPath + "LiveVideoError.png", GL_LINEAR, GL_LINEAR);
        videoTexture->texType(TT_videoBkgd);

        // Create see through video background material without shadow mapping
        SLMaterial* matVideoBkgd = new SLMaterial(am, "matVideoBkgd", videoTexture);
        matVideoBkgd->reflectionModel(RM_Custom);

        // Create see through video background material with shadow mapping
        SLMaterial* matVideoBkgdSM = new SLMaterial(am, "matVideoBkgdSM", videoTexture);
        matVideoBkgdSM->reflectionModel(RM_Custom);
        matVideoBkgdSM->ambient(SLCol4f(0.6f, 0.6f, 0.6f));
        matVideoBkgdSM->getsShadows(true);

        // Create directional light for the sunlight
        SLLightDirect* sunLight = new SLLightDirect(am, s, 5.0f);
        sunLight->powers(1.0f, 1.0f, 1.0f);
        sunLight->attenuation(1, 0, 0);
        sunLight->translation(0, 10, 0);
        sunLight->lookAt(10, 0, 10);
        sunLight->doSunPowerAdaptation(true);
        sunLight->createsShadows(true);
        sunLight->createShadowMap(-100, 150, SLVec2f(150, 150), SLVec2i(4096, 4096));
        sunLight->doSmoothShadows(true);
        sunLight->castsShadows(false);

        // Setup the camera
        SLCamera* cam1 = new SLCamera("Camera 1");
        cam1->translation(0, 50, -150);
        cam1->lookAt(0, 0, 0);
        cam1->clipNear(1);
        cam1->clipFar(300);
        cam1->focalDist(150);
        cam1->setInitialState();
        cam1->devRotLoc(&AppDemo::devRot, &AppDemo::devLoc);
        cam1->background().texture(videoTexture);

        // Turn on main video
        CVCapture::instance()->videoType(VT_MAIN);

        // Let the sun be rotated by time and location
        AppDemo::devLoc.sunLightNode(sunLight);

        // Import main model
        SLAssimpImporter importer;
        SLNode*          sutzK18 = importer.load(s->animManager(),
                                        am,
                                        dataPath + "erleb-AR/models/sutzKirchrain18/Sutz-Kirchrain18.gltf",
                                        texPath,
                                        nullptr,
                                        true,    // delete tex images after build
                                        true,    // only meshes
                                        nullptr, // no replacement material
                                        0.4f);   // 40% ambient reflection

        // Rotate to the true geographic rotation
        // Nothing to do here because the model is north up

        // Let the video shine through some objects
        sutzK18->findChild<SLNode>("Terrain")->setMeshMat(matVideoBkgdSM, true);

        // Make buildings transparent with edges
        SLNode* buildings = sutzK18->findChild<SLNode>("Buildings");
        buildings->setMeshMat(matVideoBkgd, true);
        buildings->setDrawBitsRec(SL_DB_WITHEDGES, true);

        // Add axis object a world origin
        SLNode* axis = new SLNode(new SLCoordAxis(am), "Axis Node");
        axis->setDrawBitsRec(SL_DB_MESHWIRED, false);
        axis->rotate(-90, 1, 0, 0);
        axis->castsShadows(false);

        SLNode* scene = new SLNode("Scene");
        s->root3D(scene);
        scene->addChild(sunLight);
        scene->addChild(axis);
        scene->addChild(sutzK18);
        scene->addChild(cam1);

        // initialize sensor stuff
        // Go to https://map.geo.admin.ch and choose your origin and default point
        AppDemo::devLoc.useOriginAltitude(false);
        AppDemo::devLoc.originLatLonAlt(47.10600, 7.21772, 434.4f);        // Corner Carport
        AppDemo::devLoc.defaultLatLonAlt(47.10598, 7.21757, 433.9f + 1.7); // In the street

        AppDemo::devLoc.nameLocations().push_back(SLLocation("Corner Carport, Origin", 47, 6, 21.609, 7, 13, 3.788, 434.4));
        AppDemo::devLoc.nameLocations().push_back(SLLocation("Einfahrt (Dolendeckel)", 47, 6, 21.639, 7, 13, 2.764, 433.6 + 1.7));
        AppDemo::devLoc.nameLocations().push_back(SLLocation("Elektrokasten, Brunnenweg", 47, 6, 21.044, 7, 13, 4.920, 438.4 + 1.7));
        AppDemo::devLoc.nameLocations().push_back(SLLocation("Sitzbänkli am See", 47, 6, 24.537, 7, 13, 2.766, 431.2 + 1.7));
        AppDemo::devLoc.originLatLonAlt(AppDemo::devLoc.nameLocations()[0].posWGS84LatLonAlt);
        AppDemo::devLoc.activeNamedLocation(1);   // This sets the location 1 as defaultENU
        AppDemo::devLoc.locMaxDistanceM(1000.0f); // Max. Distanz. zum Nullpunkt
        AppDemo::devLoc.improveOrigin(false);     // Keine autom. Verbesserung vom Origin
        AppDemo::devLoc.hasOrigin(true);
        AppDemo::devLoc.offsetMode(LOM_twoFingerY);
        AppDemo::devRot.zeroYawAtStart(false);
        AppDemo::devRot.offsetMode(ROM_oneFingerX);

        // This loads the DEM file and overwrites the altitude of originLatLonAlt and defaultLatLonAlt
        SLstring tif = dataPath + "erleb-AR/models/sutzKirchrain18/Sutz-Kirchrain18-DEM-WGS84.tif";
        AppDemo::devLoc.loadGeoTiff(tif);

#if defined(SL_OS_MACIOS) || defined(SL_OS_ANDROID)
        AppDemo::devLoc.isUsed(true);
        AppDemo::devRot.isUsed(true);
        cam1->camAnim(SLCamAnim::CA_deviceRotLocYUp);
#else
        AppDemo::devLoc.isUsed(false);
        AppDemo::devRot.isUsed(false);
        SLVec3d pos_d = AppDemo::devLoc.defaultENU() - AppDemo::devLoc.originENU();
        SLVec3f pos_f((SLfloat)pos_d.x, (SLfloat)pos_d.y, (SLfloat)pos_d.z);
        cam1->translation(pos_f);
        cam1->focalDist(pos_f.length());
        cam1->lookAt(SLVec3f::ZERO);
        cam1->camAnim(SLCamAnim::CA_turntableYUp);
#endif

        sv->doWaitOnIdle(false); // for constant video feed
        sv->camera(cam1);
    }
    else if (sceneID == SID_RTMuttenzerBox) //.....................................................
    {
        s->name("Muttenzer Box");
        s->info("Muttenzer Box with environment mapped reflective sphere and transparent "
                "refractive glass sphere. Try ray tracing for real reflections and soft shadows.");

        // Create reflection & glass shaders
        SLGLProgram* sp1 = new SLGLProgramGeneric(am, shaderPath + "Reflect.vert", shaderPath + "Reflect.frag");
        SLGLProgram* sp2 = new SLGLProgramGeneric(am, shaderPath + "RefractReflect.vert", shaderPath + "RefractReflect.frag");

        // Create cube mapping texture
        SLGLTexture* tex1 = new SLGLTexture(am,
                                            texPath + "MuttenzerBox+X0512_C.png",
                                            texPath + "MuttenzerBox-X0512_C.png",
                                            texPath + "MuttenzerBox+Y0512_C.png",
                                            texPath + "MuttenzerBox-Y0512_C.png",
                                            texPath + "MuttenzerBox+Z0512_C.png",
                                            texPath + "MuttenzerBox-Z0512_C.png");

        SLCol4f lightEmisRGB(7.0f, 7.0f, 7.0f);
        SLCol4f grayRGB(0.75f, 0.75f, 0.75f);
        SLCol4f redRGB(0.75f, 0.25f, 0.25f);
        SLCol4f blueRGB(0.25f, 0.25f, 0.75f);
        SLCol4f blackRGB(0.00f, 0.00f, 0.00f);

        // create materials
        SLMaterial* cream = new SLMaterial(am, "cream", grayRGB, SLCol4f::BLACK, 0);
        SLMaterial* red   = new SLMaterial(am, "red", redRGB, SLCol4f::BLACK, 0);
        SLMaterial* blue  = new SLMaterial(am, "blue", blueRGB, SLCol4f::BLACK, 0);

        // Material for mirror sphere
        SLMaterial* refl = new SLMaterial(am, "refl", blackRGB, SLCol4f::WHITE, 1000, 1.0f);
        refl->addTexture(tex1);
        refl->program(sp1);

        // Material for glass sphere
        SLMaterial* refr = new SLMaterial(am, "refr", blackRGB, blackRGB, 100, 0.05f, 0.95f, 1.5f);
        refr->translucency(1000);
        refr->transmissive(SLCol4f::WHITE);
        refr->addTexture(tex1);
        refr->program(sp2);

        SLNode* sphere1 = new SLNode(new SLSphere(am, 0.5f, 32, 32, "Sphere1", refl));
        sphere1->translate(-0.65f, -0.75f, -0.55f, TS_object);

        SLNode* sphere2 = new SLNode(new SLSphere(am, 0.45f, 32, 32, "Sphere2", refr));
        sphere2->translate(0.73f, -0.8f, 0.10f, TS_object);

        SLNode* balls = new SLNode;
        balls->addChild(sphere1);
        balls->addChild(sphere2);

        // Rectangular light
        SLLightRect* lightRect = new SLLightRect(am, s, 1, 0.65f);
        lightRect->rotate(90, -1.0f, 0.0f, 0.0f);
        lightRect->translate(0.0f, -0.25f, 1.18f, TS_object);
        lightRect->spotCutOffDEG(90);
        lightRect->spotExponent(1.0);
        lightRect->ambientColor(SLCol4f::WHITE);
        lightRect->ambientPower(0.25f);
        lightRect->diffuseColor(lightEmisRGB);
        lightRect->attenuation(0, 0, 1);
        lightRect->samplesXY(11, 7);
        lightRect->createsShadows(true);
        lightRect->createShadowMap();

        SLLight::globalAmbient.set(lightEmisRGB * 0.01f);

        // create camera
        SLCamera* cam1 = new SLCamera();
        cam1->translation(0.0f, 0.40f, 6.35f);
        cam1->lookAt(0.0f, -0.05f, 0.0f);
        cam1->fov(27);
        cam1->focalDist(cam1->translationOS().length());
        cam1->background().colors(SLCol4f(0.0f, 0.0f, 0.0f));
        cam1->setInitialState();
        cam1->devRotLoc(&AppDemo::devRot, &AppDemo::devLoc);

        // assemble scene
        SLNode* scene = new SLNode;
        s->root3D(scene);
        scene->addChild(cam1);
        scene->addChild(lightRect);

        // create wall polygons
        SLfloat pL = -1.48f, pR = 1.48f; // left/right
        SLfloat pB = -1.25f, pT = 1.19f; // bottom/top
        SLfloat pN = 1.79f, pF = -1.55f; // near/far

        // bottom plane
        SLNode* b = new SLNode(new SLRectangle(am, SLVec2f(pL, -pN), SLVec2f(pR, -pF), 6, 6, "bottom", cream));
        b->rotate(90, -1, 0, 0);
        b->translate(0, 0, pB, TS_object);
        scene->addChild(b);

        // top plane
        SLNode* t = new SLNode(new SLRectangle(am, SLVec2f(pL, pF), SLVec2f(pR, pN), 6, 6, "top", cream));
        t->rotate(90, 1, 0, 0);
        t->translate(0, 0, -pT, TS_object);
        scene->addChild(t);

        // far plane
        SLNode* f = new SLNode(new SLRectangle(am, SLVec2f(pL, pB), SLVec2f(pR, pT), 6, 6, "far", cream));
        f->translate(0, 0, pF, TS_object);
        scene->addChild(f);

        // left plane
        SLNode* l = new SLNode(new SLRectangle(am, SLVec2f(-pN, pB), SLVec2f(-pF, pT), 6, 6, "left", red));
        l->rotate(90, 0, 1, 0);
        l->translate(0, 0, pL, TS_object);
        scene->addChild(l);

        // right plane
        SLNode* r = new SLNode(new SLRectangle(am, SLVec2f(pF, pB), SLVec2f(pN, pT), 6, 6, "right", blue));
        r->rotate(90, 0, -1, 0);
        r->translate(0, 0, -pR, TS_object);
        scene->addChild(r);

        scene->addChild(balls);

        sv->camera(cam1);
    }
    else if (sceneID == SID_RTSpheres) //..........................................................
    {
        s->name("Ray tracing Spheres");
        s->info("Classic ray tracing scene with transparent and reflective spheres. Be patient on mobile devices.");

        // define materials
        SLMaterial* matGla = new SLMaterial(am, "Glass", SLCol4f(0.0f, 0.0f, 0.0f), SLCol4f(0.5f, 0.5f, 0.5f), 100, 0.4f, 0.6f, 1.5f);
        SLMaterial* matRed = new SLMaterial(am, "Red", SLCol4f(0.5f, 0.0f, 0.0f), SLCol4f(0.5f, 0.5f, 0.5f), 100, 0.5f, 0.0f, 1.0f);
        SLMaterial* matYel = new SLMaterial(am, "Floor", SLCol4f(0.8f, 0.6f, 0.2f), SLCol4f(0.8f, 0.8f, 0.8f), 100, 0.5f, 0.0f, 1.0f);

        SLCamera* cam1 = new SLCamera();
        cam1->translation(0, 0.1f, 2.5f);
        cam1->lookAt(0, 0, 0);
        cam1->focalDist(cam1->translationOS().length());
        cam1->background().colors(SLCol4f(0.1f, 0.4f, 0.8f));
        cam1->setInitialState();
        cam1->devRotLoc(&AppDemo::devRot, &AppDemo::devLoc);

        SLNode* rect = new SLNode(new SLRectangle(am, SLVec2f(-3, -3), SLVec2f(5, 4), 20, 20, "Floor", matYel));
        rect->rotate(90, -1, 0, 0);
        rect->translate(0, -1, -0.5f, TS_object);

        SLLightSpot* light1 = new SLLightSpot(am, s, 2, 2, 2, 0.1f);
        light1->powers(1, 7, 7);
        light1->attenuation(0, 0, 1);

        SLLightSpot* light2 = new SLLightSpot(am, s, 2, 2, -2, 0.1f);
        light2->powers(1, 7, 7);
        light2->attenuation(0, 0, 1);

        SLNode* scene = new SLNode;
        sv->camera(cam1);
        scene->addChild(light1);
        scene->addChild(light2);
        scene->addChild(SphereGroupRT(am, 3, 0, 0, 0, 1, 30, matGla, matRed));
        scene->addChild(rect);
        scene->addChild(cam1);

        s->root3D(scene);
    }
    else if (sceneID == SID_RTSoftShadows) //......................................................
    {
        s->name("Ray tracing soft shadows");
        s->info("Ray tracing with soft shadow light sampling. Each light source is sampled 64x per pixel. Be patient on mobile devices.");

        // Create root node
        SLNode* scene = new SLNode;
        s->root3D(scene);

        // define materials
        SLCol4f     spec(0.8f, 0.8f, 0.8f);
        SLMaterial* matBlk = new SLMaterial(am, "Glass", SLCol4f(0.0f, 0.0f, 0.0f), SLCol4f(0.5f, 0.5f, 0.5f), 100, 0.5f, 0.5f, 1.5f);
        SLMaterial* matRed = new SLMaterial(am, "Red", SLCol4f(0.5f, 0.0f, 0.0f), SLCol4f(0.5f, 0.5f, 0.5f), 100, 0.5f, 0.0f, 1.0f);
        SLMaterial* matYel = new SLMaterial(am, "Floor", SLCol4f(0.8f, 0.6f, 0.2f), SLCol4f(0.8f, 0.8f, 0.8f), 100, 0.0f, 0.0f, 1.0f);

        SLCamera* cam1 = new SLCamera;
        cam1->translation(0, 0.1f, 4);
        cam1->lookAt(0, 0, 0);
        cam1->focalDist(cam1->translationOS().length());
        cam1->background().colors(SLCol4f(0.1f, 0.4f, 0.8f));
        cam1->setInitialState();
        cam1->devRotLoc(&AppDemo::devRot, &AppDemo::devLoc);
        scene->addChild(cam1);

        SLNode* rect = new SLNode(new SLRectangle(am, SLVec2f(-5, -5), SLVec2f(5, 5), 1, 1, "Rect", matYel));
        rect->rotate(90, -1, 0, 0);
        rect->translate(0, 0, -0.5f);
        rect->castsShadows(false);
        scene->addChild(rect);

        SLLightSpot* light1 = new SLLightSpot(am, s, 2, 2, 2, 0.3f);
        light1->samples(8, 8);
        light1->attenuation(0, 0, 1);
        light1->createsShadows(true);
        light1->createShadowMap();
        scene->addChild(light1);

        SLLightSpot* light2 = new SLLightSpot(am, s, 2, 2, -2, 0.3f);
        light2->samples(8, 8);
        light2->attenuation(0, 0, 1);
        light2->createsShadows(true);
        light2->createShadowMap();
        scene->addChild(light2);

        scene->addChild(SphereGroupRT(am, 1, 0, 0, 0, 1, 32, matBlk, matRed));

        sv->camera(cam1);
    }
    else if (sceneID == SID_RTDoF) //..............................................................
    {
        s->name("Ray tracing depth of field");

        // Create root node
        SLNode* scene = new SLNode;
        s->root3D(scene);

        // Create textures and materials
        SLGLTexture* texC = new SLGLTexture(am, texPath + "Checkerboard0512_C.png", SL_ANISOTROPY_MAX, GL_LINEAR);
        SLMaterial*  mT   = new SLMaterial(am, "mT", texC);
        mT->kr(0.5f);
        SLMaterial* mW = new SLMaterial(am, "mW", SLCol4f::WHITE);
        SLMaterial* mB = new SLMaterial(am, "mB", SLCol4f::GRAY);
        SLMaterial* mY = new SLMaterial(am, "mY", SLCol4f::YELLOW);
        SLMaterial* mR = new SLMaterial(am, "mR", SLCol4f::RED);
        SLMaterial* mG = new SLMaterial(am, "mG", SLCol4f::GREEN);
        SLMaterial* mM = new SLMaterial(am, "mM", SLCol4f::MAGENTA);

#ifndef SL_GLES
        SLuint numSamples = 10;
#else
        SLuint       numSamples   = 4;
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
        scene->addChild(cam1);

        SLuint  res  = 36;
        SLNode* rect = new SLNode(new SLRectangle(am,
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
        scene->addChild(rect);

        SLLightSpot* light1 = new SLLightSpot(am, s, 2, 2, 0, 0.1f);
        light1->ambiDiffPowers(0.1f, 1);
        light1->attenuation(1, 0, 0);
        scene->addChild(light1);

        SLNode* balls = new SLNode;
        SLNode* sp;
        sp = new SLNode(new SLSphere(am, 0.5f, res, res, "S1", mW));
        sp->translate(2.0, 0, -4, TS_object);
        balls->addChild(sp);
        sp = new SLNode(new SLSphere(am, 0.5f, res, res, "S2", mB));
        sp->translate(1.5, 0, -3, TS_object);
        balls->addChild(sp);
        sp = new SLNode(new SLSphere(am, 0.5f, res, res, "S3", mY));
        sp->translate(1.0, 0, -2, TS_object);
        balls->addChild(sp);
        sp = new SLNode(new SLSphere(am, 0.5f, res, res, "S4", mR));
        sp->translate(0.5, 0, -1, TS_object);
        balls->addChild(sp);
        sp = new SLNode(new SLSphere(am, 0.5f, res, res, "S5", mG));
        sp->translate(0.0, 0, 0, TS_object);
        balls->addChild(sp);
        sp = new SLNode(new SLSphere(am, 0.5f, res, res, "S6", mM));
        sp->translate(-0.5, 0, 1, TS_object);
        balls->addChild(sp);
        sp = new SLNode(new SLSphere(am, 0.5f, res, res, "S7", mW));
        sp->translate(-1.0, 0, 2, TS_object);
        balls->addChild(sp);
        scene->addChild(balls);

        sv->camera(cam1);
    }
    else if (sceneID == SID_RTLens) //.............................................................
    {
        s->name("Ray tracing lens test");
        s->info("Ray tracing lens test scene.");

        // Create root node
        SLNode* scene = new SLNode;
        s->root3D(scene);

        // Create textures and materials
        SLGLTexture* texC = new SLGLTexture(am, texPath + "VisionExample.jpg");
        // SLGLTexture* texC = new SLGLTexture(am, texPath + "Checkerboard0512_C.png");

        SLMaterial* mT = new SLMaterial(am, "mT", texC, nullptr, nullptr, nullptr);
        mT->kr(0.5f);

        // Glass material
        // name, ambient, specular,	shininess, kr(reflectivity), kt(transparency), kn(refraction)
        SLMaterial* matLens = new SLMaterial(am, "lens", SLCol4f(0.0f, 0.0f, 0.0f), SLCol4f(0.5f, 0.5f, 0.5f), 100, 0.5f, 0.5f, 1.5f);
        // SLGLShaderProg* sp1 = new SLGLShaderProgGeneric("RefractReflect.vert", "RefractReflect.frag");
        // matLens->shaderProg(sp1);

#ifndef APP_USES_GLES
        SLuint numSamples = 10;
#else
        SLuint       numSamples   = 6;
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
        cam1->devRotLoc(&AppDemo::devRot, &AppDemo::devLoc);
        scene->addChild(cam1);

        // Plane
        // SLNode* rect = new SLNode(new SLRectangle(SLVec2f(-20, -20), SLVec2f(20, 20), 50, 20, "Rect", mT));
        // rect->translate(0, 0, 0, TS_Object);
        // rect->rotate(90, -1, 0, 0);

        SLLightSpot* light1 = new SLLightSpot(am, s, 1, 6, 1, 0.1f);
        light1->attenuation(0, 0, 1);
        scene->addChild(light1);

        SLuint  res  = 20;
        SLNode* rect = new SLNode(new SLRectangle(am, SLVec2f(-5, -5), SLVec2f(5, 5), res, res, "Rect", mT));
        rect->rotate(90, -1, 0, 0);
        rect->translate(0, 0, -0.0f, TS_object);
        scene->addChild(rect);

        // Lens from eye prescription card
        // SLNode* lensA = new SLNode(new SLLens(s, 0.50f, -0.50f, 4.0f, 0.0f, 32, 32, "presbyopic", matLens));   // Weitsichtig
        // lensA->translate(-2, 1, -2);
        // scene->addChild(lensA);

        // SLNode* lensB = new SLNode(new SLLens(s, -0.65f, -0.10f, 4.0f, 0.0f, 32, 32, "myopic", matLens));      // Kurzsichtig
        // lensB->translate(2, 1, -2);
        // scene->addChild(lensB);

        // Lens with radius
        // SLNode* lensC = new SLNode(new SLLens(s, 5.0, 4.0, 4.0f, 0.0f, 32, 32, "presbyopic", matLens));        // Weitsichtig
        // lensC->translate(-2, 1, 2);
        // scene->addChild(lensC);

        SLNode* lensD = new SLNode(new SLLens(am,
                                              -15.0f,
                                              -15.0f,
                                              1.0f,
                                              0.1f,
                                              res,
                                              res,
                                              "myopic",
                                              matLens)); // Kurzsichtig
        lensD->translate(0, 6, 0);
        scene->addChild(lensD);

        sv->camera(cam1);
    }
    else if (sceneID == SID_RTTest) //.............................................................
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
        cam1->devRotLoc(&AppDemo::devRot, &AppDemo::devLoc);

        // Create a light source node
        SLLightSpot* light1 = new SLLightSpot(am, s, 0.3f);
        light1->translation(5, 5, 5);
        light1->lookAt(0, 0, 0);
        light1->name("light node");

        // Material for glass sphere
        SLMaterial* matBox1  = new SLMaterial(am, "matBox1", SLCol4f(0.0f, 0.0f, 0.0f), SLCol4f(0.5f, 0.5f, 0.5f), 100, 0.0f, 0.9f, 1.5f);
        SLMesh*     boxMesh1 = new SLBox(am, -0.8f, -1, 0.02f, 1.2f, 1, 1, "boxMesh1", matBox1);
        SLNode*     boxNode1 = new SLNode(boxMesh1, "BoxNode1");

        SLMaterial* matBox2  = new SLMaterial(am, "matBox2", SLCol4f(0.0f, 0.0f, 0.0f), SLCol4f(0.5f, 0.5f, 0.5f), 100, 0.0f, 0.9f, 1.3f);
        SLMesh*     boxMesh2 = new SLBox(am, -1.2f, -1, -1, 0.8f, 1, -0.02f, "BoxMesh2", matBox2);
        SLNode*     boxNode2 = new SLNode(boxMesh2, "BoxNode2");

        // Create a scene group and add all nodes
        SLNode* scene = new SLNode("scene node");
        s->root3D(scene);
        scene->addChild(light1);
        scene->addChild(cam1);
        scene->addChild(boxNode1);
        scene->addChild(boxNode2);

        sv->camera(cam1);
    }
    if (sceneID == SID_ParticleSystem_First) //...............................................
    {
        if (stateGL->glHasGeometryShaders())
        {
            // Set scene name and info string
            s->name("First particle system");
            s->info("First scene with a particle system");

            // Create a scene group node
            SLNode* scene = new SLNode("scene node");
            s->root3D(scene);

            // Create and add camera
            SLCamera* cam1 = new SLCamera("Camera 1");
            cam1->translation(0, 1.5f, 4);
            cam1->lookAt(0, 1.5f, 0);
            scene->addChild(cam1);

            // Create textures and materials
            SLGLTexture* texC        = new SLGLTexture(am, texPath + "ParticleSmoke_08_C.png");
            SLGLTexture* texFlipbook = new SLGLTexture(am, texPath + "ParticleSmoke_03_8x8_C.png");

            // Create a light source node
            SLLightSpot* light1 = new SLLightSpot(am, s, 0.3f);
            light1->translation(5, 5, 5);
            light1->name("light node");
            scene->addChild(light1);

            // Create meshes and nodes
            SLParticleSystem* ps     = new SLParticleSystem(am,
                                                        50,
                                                        SLVec3f(0.04f, 0.4f, 0.1f),
                                                        SLVec3f(-0.11f, 0.7f, -0.1f),
                                                        4.0f,
                                                        texC,
                                                        "Particle System",
                                                        texFlipbook);
            SLNode*           pSNode = new SLNode(ps, "Particle system node");
            scene->addChild(pSNode);

            // Set background color and the root scene node
            sv->sceneViewCamera()->background().colors(SLCol4f(0.8f, 0.8f, 0.8f),
                                                       SLCol4f(0.2f, 0.2f, 0.2f));
            sv->camera(cam1);
            sv->doWaitOnIdle(false);
        }
    }
    else if (sceneID == SID_ParticleSystem_Demo) //................................................
    {
        if (stateGL->glHasGeometryShaders())
        {
            // Set scene name and info string
            s->name("Simple Demo Particle System");
            s->info("This most simple single particle system is meant to be improved by adding more and more features in the properties list.");

            // Create a scene group node
            SLNode* scene = new SLNode("scene node");
            s->root3D(scene);

            // Create textures and materials
            SLGLTexture* texC        = new SLGLTexture(am, texPath + "ParticleSmoke_08_C.png");
            SLGLTexture* texFlipbook = new SLGLTexture(am, texPath + "ParticleSmoke_03_8x8_C.png");

            // Create meshes and nodes
            SLParticleSystem* ps = new SLParticleSystem(am,
                                                        1,
                                                        SLVec3f(0.04f, 0.4f, 0.1f),
                                                        SLVec3f(-0.11f, 0.7f, -0.1f),
                                                        4.0f,
                                                        texC,
                                                        "Particle System",
                                                        texFlipbook);
            ps->doAlphaOverLT(false);
            ps->doSizeOverLT(false);
            ps->doRotation(false);
            ps->doColor(false);
            ps->acceleration(-0.5, 0.0, 0.0);
            ps->timeToLive(2.0f);
            SLMesh* pSMesh = ps;
            SLNode* pSNode = new SLNode(pSMesh, "Particle system node");
            scene->addChild(pSNode);

            // Set background color and the root scene node
            sv->sceneViewCamera()->background().colors(SLCol4f(0.8f, 0.8f, 0.8f),
                                                       SLCol4f(0.2f, 0.2f, 0.2f));
            // Save energy
            sv->doWaitOnIdle(false);
        }
    }
    else if (sceneID == SID_ParticleSystem_DustStorm) //...........................................
    {
        if (stateGL->glHasGeometryShaders())
        {
            // Set scene name and info string
            s->name("Dust storm particle system");
            s->info("This dust storm particle system uses the box shape type for distribution.\n"
                    "See the properties window for the detailed settings of the particles system");

            // Create a scene group node
            SLNode* scene = new SLNode("scene node");
            s->root3D(scene);

            // Create and add camera
            SLCamera* cam1 = new SLCamera("Camera 1");
            cam1->translation(0, 0, 55);
            cam1->lookAt(0, 0, 0);
            cam1->focalDist(55);
            scene->addChild(cam1);
            sv->camera(cam1);

            // Create textures and materials
            SLGLTexture* texC             = new SLGLTexture(am, texPath + "ParticleSmoke_08_C.png");
            SLGLTexture* texFlipbookSmoke = new SLGLTexture(am, texPath + "ParticleSmoke_03_8x8_C.png");

            // Create meshes and nodes
            // Dust storm
            SLParticleSystem* ps = new SLParticleSystem(am,
                                                        500,
                                                        SLVec3f(-0.1f, -0.5f, -5.0f),
                                                        SLVec3f(0.1f, 0.5f, -2.5f),
                                                        3.5f,
                                                        texC,
                                                        "DustStorm",
                                                        texFlipbookSmoke);
            ps->doShape(true);
            ps->shapeType(ST_Box);
            ps->shapeScale(50.0f, 1.0f, 50.0f);
            ps->scale(15.0f);
            ps->doSizeOverLT(false);
            ps->doAlphaOverLT(true);
            ps->doAlphaOverLTCurve(true);
            ps->bezierStartEndPointAlpha()[1] = 0.0f;
            ps->bezierControlPointAlpha()[1]  = 0.5f;
            ps->bezierControlPointAlpha()[2]  = 0.5f;
            ps->generateBernsteinPAlpha();
            ps->doRotRange(true);
            ps->color(SLCol4f(1.0f, 1.0f, 1.0f, 1.0f));
            ps->doBlendBrightness(false);
            ps->frameRateFB(16);

            SLMesh* pSMesh = ps;
            SLNode* pSNode = new SLNode(pSMesh, "Particle system node fire2");
            pSNode->translate(3.0f, -0.8f, 0.0f, TS_object);

            scene->addChild(pSNode);

            // Set background color and the root scene node
            sv->sceneViewCamera()->background().colors(SLCol4f(0.8f, 0.8f, 0.8f),
                                                       SLCol4f(0.2f, 0.2f, 0.2f));
            // Save energy
            sv->doWaitOnIdle(false);
        }
    }
    else if (sceneID == SID_ParticleSystem_Fountain) //............................................
    {
        if (stateGL->glHasGeometryShaders())
        {
            // Set scene name and info string
            s->name("Fountain particle system");
            s->info("This fountain particle system uses acceleration and gravity.\n"
                    "See the properties window for the detailed settings of the particles system");

            // Create a scene group node
            SLNode* scene = new SLNode("scene node");
            s->root3D(scene);

            // Create and add camera
            SLCamera* cam1 = new SLCamera("Camera 1");
            cam1->translation(0, -1, 55);
            cam1->lookAt(0, -1, 0);
            cam1->focalDist(55);
            scene->addChild(cam1);
            sv->camera(cam1);

            // Create textures and materials
            SLGLTexture* texC        = new SLGLTexture(am, texPath + "ParticleCircle_05_C.png");
            SLGLTexture* texFlipbook = new SLGLTexture(am, texPath + "ParticleSmoke_03_8x8_C.png");
            // SLGLTexture* texFlipbook = new SLGLTexture(am, texPath + "ParticleSmoke_04_8x8_C.png");

            // Create a light source node
            SLLightSpot* light1 = new SLLightSpot(am, s, 0.3f);
            light1->translation(0, -1, 2);
            light1->name("light node");
            scene->addChild(light1);

            // Create meshes and nodes
            SLParticleSystem* ps     = new SLParticleSystem(am,
                                                        5000,
                                                        SLVec3f(5.0f, 15.0f, 5.0f),
                                                        SLVec3f(-5.0f, 17.0f, -5.0f),
                                                        5.0f,
                                                        texC,
                                                        "Fountain",
                                                        texFlipbook);
            SLMesh*           pSMesh = ps;
            ps->doGravity(true);
            ps->color(SLCol4f(0.0039f, 0.14f, 0.86f, 0.33f));
            ps->doSizeOverLT(false);
            ps->doAlphaOverLT(false);
            SLNode* pSNode = new SLNode(pSMesh, "Particle system node");
            scene->addChild(pSNode);

            // Set background color and the root scene node
            sv->sceneViewCamera()->background().colors(SLCol4f(0.8f, 0.8f, 0.8f),
                                                       SLCol4f(0.2f, 0.2f, 0.2f));
            // Save energy
            sv->doWaitOnIdle(false);
        }
    }
    else if (sceneID == SID_ParticleSystem_Sun) //.................................................
    {
        if (stateGL->glHasGeometryShaders())
        {
            // Set scene name and info string
            s->name("Sun particle system");
            s->info("This sun particle system uses the sphere shape type for distribution.\n"
                    "See the properties window for the detailed settings of the particles system");

            // Create a scene group node
            SLNode* scene = new SLNode("scene node");
            s->root3D(scene);

            // Create textures and materials
            SLGLTexture* texC        = new SLGLTexture(am, texPath + "ParticleSmoke_08_C.png");
            SLGLTexture* texFlipbook = new SLGLTexture(am, texPath + "ParticleSmoke_03_8x8_C.png");

            // Create meshes and nodes
            SLParticleSystem* ps = new SLParticleSystem(am,
                                                        10000,
                                                        SLVec3f(0.0f, 0.0f, 0.0f),
                                                        SLVec3f(0.0f, 0.0f, 0.0f),
                                                        4.0f,
                                                        texC,
                                                        "Sun Particle System",
                                                        texFlipbook);

            ps->doShape(true);
            ps->shapeType(ST_Sphere);
            ps->shapeRadius(3.0f);
            ps->doBlendBrightness(true);
            ps->color(SLCol4f(0.925f, 0.238f, 0.097f, 0.199f));

            SLMesh* pSMesh = ps;
            SLNode* pSNode = new SLNode(pSMesh, "Particle Sun node");
            scene->addChild(pSNode);

            // Set background color and the root scene node
            sv->sceneViewCamera()->background().colors(SLCol4f(0.8f, 0.8f, 0.8f),
                                                       SLCol4f(0.2f, 0.2f, 0.2f));
            // Save energy
            sv->doWaitOnIdle(false);
        }
    }
    else if (sceneID == SID_ParticleSystem_RingOfFire) //..........................................
    {
        if (stateGL->glHasGeometryShaders())
        {
            // Set scene name and info string
            s->name("Ring of fire particle system");
            s->info("This ring particle system uses the cone shape type for distribution.\n"
                    "See the properties window for the settings of the particles system");

            // Create a scene group node
            SLNode* scene = new SLNode("scene node");
            s->root3D(scene);

            // Create textures and materials
            SLGLTexture* texC        = new SLGLTexture(am, texPath + "ParticleSmoke_08_C.png");
            SLGLTexture* texFlipbook = new SLGLTexture(am, texPath + "ParticleSmoke_03_8x8_C.png");

            // Create meshes and nodes
            SLParticleSystem* ps = new SLParticleSystem(am,
                                                        1000,
                                                        SLVec3f(0.0f, 0.0f, 0.0f),
                                                        SLVec3f(0.0f, 0.0f, 0.0f),
                                                        4.0f,
                                                        texC,
                                                        "Ring of fire Particle System",
                                                        texFlipbook);

            ps->doShape(true);
            ps->shapeType(ST_Cone);
            ps->doShapeSpawnBase(true);
            ps->doShapeSurface(true);
            ps->shapeRadius(1.0f);
            ps->doBlendBrightness(true);
            ps->color(SLCol4f(0.925f, 0.238f, 0.097f, 0.503f));

            SLMesh* pSMesh = ps;
            SLNode* pSNode = new SLNode(pSMesh, "Particle Ring Fire node");
            pSNode->rotate(90, 1, 0, 0);
            scene->addChild(pSNode);

            // Set background color and the root scene node
            sv->sceneViewCamera()->background().colors(SLCol4f(0.8f, 0.8f, 0.8f),
                                                       SLCol4f(0.2f, 0.2f, 0.2f));
            // Save energy
            sv->doWaitOnIdle(false);
        }
    }
    else if (sceneID == SID_ParticleSystem_FireComplex) //.........................................
    {
        if (stateGL->glHasGeometryShaders())
        {
            // Set scene name and info string
            s->name("Fire Complex particle system");
            s->info("The fire particle systems contain each multiple sub particle systems.\n"
                    "See the scenegraph window for the sub particles systems. "
                    "See the properties window for the settings of the particles systems");

            // Create a scene group node
            SLNode* scene = new SLNode("scene node");
            s->root3D(scene);

            // Create and add camera
            SLCamera* cam1 = new SLCamera("Camera 1");
            cam1->translation(0, 1.2f, 4.0f);
            cam1->lookAt(0, 1.2f, 0);
            cam1->focalDist(4.5f);
            cam1->setInitialState();
            scene->addChild(cam1);
            sv->camera(cam1);

            // Create textures and materials
            SLGLTexture* texFireCld  = new SLGLTexture(am, texPath + "ParticleFirecloudTransparent_C.png");
            SLGLTexture* texFireFlm  = new SLGLTexture(am, texPath + "ParticleFlames_06_8x8_C.png");
            SLGLTexture* texCircle   = new SLGLTexture(am, texPath + "ParticleCircle_05_C.png");
            SLGLTexture* texSmokeB   = new SLGLTexture(am, texPath + "ParticleCloudBlack_C.png");
            SLGLTexture* texSmokeW   = new SLGLTexture(am, texPath + "ParticleCloudWhite_C.png");
            SLGLTexture* texTorchFlm = new SLGLTexture(am, texPath + "ParticleFlames_04_16x4_C.png");
            SLGLTexture* texTorchSmk = new SLGLTexture(am, texPath + "ParticleSmoke_08_C.png");

            SLNode* complexFire = createComplexFire(am,
                                                    s,
                                                    true,
                                                    texTorchSmk,
                                                    texFireFlm,
                                                    8,
                                                    8,
                                                    texCircle,
                                                    texSmokeB,
                                                    texSmokeW);
            scene->addChild(complexFire);

            // Room around
            {
                // Room parent node
                SLNode* room = new SLNode("Room");
                scene->addChild(room);

                // Back wall material
                SLGLTexture* texWallDIF = new SLGLTexture(am, texPath + "BrickLimestoneGray_1K_DIF.jpg", SL_ANISOTROPY_MAX, GL_LINEAR);
                SLGLTexture* texWallNRM = new SLGLTexture(am, texPath + "BrickLimestoneGray_1K_NRM.jpg", SL_ANISOTROPY_MAX, GL_LINEAR);
                SLMaterial*  matWall    = new SLMaterial(am, "mat3", texWallDIF, texWallNRM);
                matWall->specular(SLCol4f::BLACK);
                matWall->metalness(0);
                matWall->roughness(1);
                matWall->reflectionModel(RM_CookTorrance);

                // Room dimensions
                SLfloat pL = -2.0f, pR = 2.0f;  // left/right
                SLfloat pB = -0.01f, pT = 4.0f; // bottom/top
                SLfloat pN = 2.0f, pF = -2.0f;  // near/far

                // bottom rectangle
                SLNode* b = new SLNode(new SLRectangle(am, SLVec2f(pL, -pN), SLVec2f(pR, -pF), 10, 10, "Floor", matWall));
                b->rotate(90, -1, 0, 0);
                b->translate(0, 0, pB, TS_object);
                room->addChild(b);

                // far rectangle
                SLNode* f = new SLNode(new SLRectangle(am, SLVec2f(pL, pB), SLVec2f(pR, pT), 10, 10, "Wall far", matWall));
                f->translate(0, 0, pF, TS_object);
                room->addChild(f);

                // near rectangle
                SLNode* n = new SLNode(new SLRectangle(am, SLVec2f(pL, pB), SLVec2f(pR, pT), 10, 10, "Wall near", matWall));
                n->rotate(180, 0, 1, 0);
                n->translate(0, 0, pF, TS_object);
                room->addChild(n);

                // left rectangle
                SLNode* l = new SLNode(new SLRectangle(am, SLVec2f(-pN, pB), SLVec2f(-pF, pT), 10, 10, "Wall left", matWall));
                l->rotate(90, 0, 1, 0);
                l->translate(0, 0, pL, TS_object);
                room->addChild(l);

                // right rectangle
                SLNode* r = new SLNode(new SLRectangle(am, SLVec2f(pF, pB), SLVec2f(pN, pT), 10, 10, "Wall right", matWall));
                r->rotate(90, 0, -1, 0);
                r->translate(0, 0, -pR, TS_object);
                room->addChild(r);
            }

            // Firewood
            SLAssimpImporter importer;
            SLNode*          firewood = importer.load(s->animManager(),
                                             am,
                                             modelPath + "GLTF/Firewood/Firewood1.gltf",
                                             texPath,
                                             nullptr,
                                             false,
                                             true,
                                             nullptr,
                                             0.3f,
                                             true);
            firewood->scale(2);
            scene->addChild(firewood);

            // Torch
            SLNode* torchL = importer.load(s->animManager(),
                                           am,
                                           modelPath + "GLTF/Torch/Torch.gltf",
                                           texPath,
                                           nullptr,
                                           false,
                                           true,
                                           nullptr,
                                           0.3f,
                                           true);
            torchL->name("Torch Left");
            SLNode* torchR = torchL->copyRec();
            torchR->name("Torch Right");
            torchL->translate(-2, 1.5f, 0);
            torchL->rotate(90, 0, 1, 0);
            torchL->scale(2);
            scene->addChild(torchL);
            torchR->translate(2, 1.5f, 0);
            torchR->rotate(-90, 0, 1, 0);
            torchR->scale(2);
            scene->addChild(torchR);

            // Torch flame left
            SLNode* torchFlameNodeL = createTorchFire(am,
                                                      s,
                                                      true,
                                                      texTorchSmk,
                                                      texTorchFlm,
                                                      16,
                                                      4);
            torchFlameNodeL->translate(-1.6f, 2.25f, 0);
            torchFlameNodeL->name("Torch Fire Left");
            scene->addChild(torchFlameNodeL);

            // Torch flame right
            SLNode* torchFlameNodeR = createTorchFire(am,
                                                      s,
                                                      true,
                                                      texTorchSmk,
                                                      texTorchFlm,
                                                      16,
                                                      4);
            torchFlameNodeR->translate(1.6f, 2.25f, 0);
            torchFlameNodeR->name("Torch Fire Right");
            scene->addChild(torchFlameNodeR);

            // Set background color and the root scene node
            sv->sceneViewCamera()->background().colors(SLCol4f(0.8f, 0.8f, 0.8f),
                                                       SLCol4f(0.2f, 0.2f, 0.2f));
            // Save energy
            sv->doWaitOnIdle(false);
        }
    }

    else if (sceneID == SID_Benchmark1_LargeModel) //..............................................
    {
        SLstring largeFile = modelPath + "PLY/xyzrgb_dragon/xyzrgb_dragon.ply";

        if (SLFileStorage::exists(largeFile, IOK_model))
        {
            s->name("Large Model Benchmark Scene");
            s->info("Large Model with 7.2 mio. triangles.");

            // Material for glass
            SLMaterial* diffuseMat = new SLMaterial(am, "diffuseMat", SLCol4f::WHITE, SLCol4f::WHITE);

            SLCamera* cam1 = new SLCamera("Camera 1");
            cam1->translation(0, 0, 220);
            cam1->lookAt(0, 0, 0);
            cam1->clipNear(1);
            cam1->clipFar(10000);
            cam1->focalDist(220);
            cam1->background().colors(SLCol4f(0.7f, 0.7f, 0.7f), SLCol4f(0.2f, 0.2f, 0.2f));
            cam1->setInitialState();
            cam1->devRotLoc(&AppDemo::devRot, &AppDemo::devLoc);

            SLLightSpot* light1 = new SLLightSpot(am, s, 200, 200, 200, 1);
            light1->powers(0.1f, 1.0f, 1.0f);
            light1->attenuation(1, 0, 0);

            SLAssimpImporter importer;
            gDragonModel = importer.load(s->animManager(),
                                         am,
                                         largeFile,
                                         texPath,
                                         nullptr,
                                         false, // delete tex images after build
                                         true,
                                         diffuseMat,
                                         0.2f,
                                         false,
                                         nullptr,
                                         SLProcess_Triangulate | SLProcess_JoinIdenticalVertices);

            SLNode* scene = new SLNode("Scene");
            s->root3D(scene);
            scene->addChild(light1);
            scene->addChild(gDragonModel);
            scene->addChild(cam1);

            sv->camera(cam1);
        }
    }
    else if (sceneID == SID_Benchmark2_MassiveNodes) //............................................
    {
        s->name("Massive Data Benchmark Scene");
        s->info(s->name());

        SLCamera* cam1 = new SLCamera("Camera 1");
        cam1->clipNear(0.1f);
        cam1->clipFar(100);
        cam1->translation(0, 0, 50);
        cam1->lookAt(0, 0, 0);
        cam1->focalDist(50);
        cam1->background().colors(SLCol4f(0.1f, 0.1f, 0.1f));
        cam1->setInitialState();

        SLLightSpot* light1 = new SLLightSpot(am, s, 15, 15, 15, 0.3f);
        light1->powers(0.2f, 0.8f, 1.0f);
        light1->attenuation(1, 0, 0);

        SLNode* scene = new SLNode;
        s->root3D(scene);
        scene->addChild(cam1);
        scene->addChild(light1);

        // Generate NUM_MAT materials
        const int   NUM_MAT = 20;
        SLVMaterial mat;
        for (int i = 0; i < NUM_MAT; ++i)
        {
            SLGLTexture* texC    = new SLGLTexture(am, texPath + "earth2048_C_Q95.jpg");
            SLstring     matName = "mat-" + std::to_string(i);
            mat.push_back(new SLMaterial(am, matName.c_str(), texC));
            SLCol4f color;
            color.hsva2rgba(SLVec4f(Utils::TWOPI * (float)i / (float)NUM_MAT, 1.0f, 1.0f));
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
                    SLSphere* earth  = new SLSphere(am, 0.3f, res, res, nodeName, mat[iMat]);
                    SLNode*   sphere = new SLNode(earth);
                    sphere->translate(float(iX), float(iY), float(iZ), TS_object);
                    scene->addChild(sphere);
                    // SL_LOG("Earth: %000d (Mat: %00d)", n, iMat);
                    n++;
                }
            }
        }

        sv->camera(cam1);
        sv->doWaitOnIdle(false);
    }
    else if (sceneID == SID_Benchmark3_NodeAnimations) //..........................................
    {
        s->name("Massive Node Animation Benchmark Scene");
        s->info(s->name());

        SLCamera* cam1 = new SLCamera("Camera 1");
        cam1->clipNear(0.1f);
        cam1->clipFar(100);
        cam1->translation(0, 2.5f, 20);
        cam1->focalDist(20);
        cam1->lookAt(0, 2.5f, 0);
        cam1->background().colors(SLCol4f(0.1f, 0.1f, 0.1f));
        cam1->setInitialState();

        SLLightSpot* light1 = new SLLightSpot(am, s, 15, 15, 15, 0.3f);
        light1->powers(0.2f, 0.8f, 1.0f);
        light1->attenuation(1, 0, 0);

        SLNode* scene = new SLNode;
        s->root3D(scene);
        scene->addChild(cam1);
        scene->addChild(light1);

        // Generate NUM_MAT materials
        const int   NUM_MAT = 20;
        SLVMaterial mat;
        for (int i = 0; i < NUM_MAT; ++i)
        {
            SLGLTexture* texC    = new SLGLTexture(am, texPath + "earth2048_C_Q95.jpg"); // color map
            SLstring     matName = "mat-" + std::to_string(i);
            mat.push_back(new SLMaterial(am, matName.c_str(), texC));
            SLCol4f color;
            color.hsva2rgba(SLVec4f(Utils::TWOPI * (float)i / (float)NUM_MAT, 1.0f, 1.0f));
            mat[i]->diffuse(color);
        }

        // create rotating sphere group
        SLint maxDepth = 5;

        SLint resolution = 18;
        scene->addChild(RotatingSphereGroup(am,
                                            s,
                                            maxDepth,
                                            0,
                                            0,
                                            0,
                                            1,
                                            resolution,
                                            mat));

        sv->camera(cam1);
        sv->doWaitOnIdle(false);
    }
    else if (sceneID == SID_Benchmark4_SkinnedAnimations) //.......................................
    {
        SLint  size         = 20;
        SLint  numAstroboys = size * size;
        SLchar name[512];
        snprintf(name, sizeof(name), "Massive Skinned Animation Benchmark w. %d individual Astroboys", numAstroboys);
        s->name(name);
        s->info(s->name());

        // Create materials
        SLMaterial* m1 = new SLMaterial(am, "m1", SLCol4f::GRAY);
        m1->specular(SLCol4f::BLACK);

        // Define a light
        SLLightSpot* light1 = new SLLightSpot(am, s, 100, 40, 100, 1);
        light1->powers(0.1f, 1.0f, 1.0f);
        light1->attenuation(1, 0, 0);

        // Define camera
        SLCamera* cam1 = new SLCamera;
        cam1->translation(0, 30, 0);
        cam1->lookAt(0, 0, 0);
        cam1->focalDist(cam1->translationOS().length());
        cam1->background().colors(SLCol4f(0.1f, 0.4f, 0.8f));
        cam1->setInitialState();
        cam1->devRotLoc(&AppDemo::devRot, &AppDemo::devLoc);

        // Floor rectangle
        SLNode* rect = new SLNode(new SLRectangle(am,
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

        // Assemble scene
        SLNode* scene = new SLNode("scene group");
        s->root3D(scene);
        scene->addChild(light1);
        scene->addChild(rect);
        scene->addChild(cam1);

        // create army with individual astroboys
        SLfloat offset = 1.0f;
        SLfloat z      = (float)(size - 1) * offset * 0.5f;

        for (SLint iZ = 0; iZ < size; ++iZ)
        {
            SLfloat x = -(float)(size - 1) * offset * 0.5f;

            for (SLint iX = 0; iX < size; ++iX)
            {
                SLNode* astroboy = importer.load(s->animManager(),
                                                 am,
                                                 modelPath + "DAE/AstroBoy/AstroBoy.dae",
                                                 texPath);

                s->animManager().lastAnimPlayback()->playForward();
                s->animManager().lastAnimPlayback()->playbackRate(Utils::random(0.5f, 1.5f));
                astroboy->translate(x, 0, z, TS_object);
                scene->addChild(astroboy);
                x += offset;
            }
            z -= offset;
        }

        sv->camera(cam1);
    }
    else if (sceneID == SID_Benchmark5_ColumnsNoLOD ||
             sceneID == SID_Benchmark6_ColumnsLOD) //..............................................
    {
        SLstring modelFile = modelPath + "GLTF/CorinthianColumn/Corinthian-Column-Round-LOD.gltf";
        SLstring texCFile  = modelPath + "GLTF/CorinthianColumn/PavementSlateSquare2_2K_DIF.jpg";
        SLstring texNFile  = modelPath + "GLTF/CorinthianColumn/PavementSlateSquare2_2K_NRM.jpg";

        if (SLFileStorage::exists(modelFile, IOK_model) &&
            SLFileStorage::exists(texCFile, IOK_image) &&
            SLFileStorage::exists(texNFile, IOK_image))
        {
            SLchar name[512];
            SLint  size;
            if (sceneID == SID_Benchmark5_ColumnsNoLOD)
            {
                size = 10;
                snprintf(name, sizeof(name), "%d corinthian columns without LOD", size * size);
                s->name(name);
            }
            else
            {
                size = 50;
                snprintf(name, sizeof(name), "%d corinthian columns with LOD", size * size);
                s->name(name);
            }
            s->info(s->name() + " with cascaded shadow mapping. In the Day-Time dialogue you can change the sun angle.");

            // Create ground material
            SLGLTexture* texFloorDif = new SLGLTexture(am, texCFile, SL_ANISOTROPY_MAX, GL_LINEAR);
            SLGLTexture* texFloorNrm = new SLGLTexture(am, texNFile, SL_ANISOTROPY_MAX, GL_LINEAR);
            SLMaterial*  matFloor    = new SLMaterial(am, "matFloor", texFloorDif, texFloorNrm);

            // Define camera
            SLCamera* cam1 = new SLCamera;
            cam1->translation(0, 1.7f, 20);
            cam1->lookAt(0, 1.7f, 0);
            cam1->focalDist(cam1->translationOS().length());
            cam1->clipFar(600);
            cam1->background().colors(SLCol4f(0.1f, 0.4f, 0.8f));
            cam1->setInitialState();

            // Create directional light for the sunlight
            SLLightDirect* sunLight = new SLLightDirect(am, s, 1.0f);
            sunLight->powers(0.25f, 1.0f, 1.0f);
            sunLight->attenuation(1, 0, 0);
            sunLight->translation(0, 1.7f, 0);
            sunLight->lookAt(-1, 0, -1);
            sunLight->doSunPowerAdaptation(true);

            // Add cascaded shadow mapping
            sunLight->createsShadows(true);
            sunLight->createShadowMapAutoSize(cam1);
            sunLight->doSmoothShadows(true);
            sunLight->castsShadows(false);
            sunLight->shadowMinBias(0.003f);
            sunLight->shadowMaxBias(0.012f);

            // Let the sun be rotated by time and location
            AppDemo::devLoc.sunLightNode(sunLight);
            AppDemo::devLoc.originLatLonAlt(47.14271, 7.24337, 488.2);        // Ecke Giosa
            AppDemo::devLoc.defaultLatLonAlt(47.14260, 7.24310, 488.7 + 1.7); // auf Parkplatz

            // Floor rectangle
            SLNode* rect = new SLNode(new SLRectangle(am,
                                                      SLVec2f(-200, -200),
                                                      SLVec2f(200, 200),
                                                      SLVec2f(0, 0),
                                                      SLVec2f(50, 50),
                                                      50,
                                                      50,
                                                      "Floor",
                                                      matFloor));
            rect->rotate(90, -1, 0, 0);
            rect->castsShadows(false);

            // Load the corinthian column
            SLAssimpImporter importer;
            SLNode*          columnLOD = importer.load(s->animManager(),
                                              am,
                                              modelFile,
                                              texPath,
                                              nullptr,
                                              false,   // delete tex images after build
                                              true,    // only meshes
                                              nullptr, // no replacement material
                                              1.0f);   // 40% ambient reflection

            SLNode* columnL0 = columnLOD->findChild<SLNode>("Corinthian-Column-Round-L0");
            SLNode* columnL1 = columnLOD->findChild<SLNode>("Corinthian-Column-Round-L1");
            SLNode* columnL2 = columnLOD->findChild<SLNode>("Corinthian-Column-Round-L2");
            SLNode* columnL3 = columnLOD->findChild<SLNode>("Corinthian-Column-Round-L3");

            // Assemble scene
            SLNode* scene = new SLNode("Scene");
            s->root3D(scene);
            scene->addChild(sunLight);
            scene->addChild(rect);
            scene->addChild(cam1);

            // create loads of pillars
            SLint   numColumns = size * size;
            SLfloat offset     = 5.0f;
            SLfloat z          = (float)(size - 1) * offset * 0.5f;

            for (SLint iZ = 0; iZ < size; ++iZ)
            {
                SLfloat x = -(float)(size - 1) * offset * 0.5f;

                for (SLint iX = 0; iX < size; ++iX)
                {
                    SLint iZX = iZ * size + iX;

                    if (sceneID == SID_Benchmark5_ColumnsNoLOD)
                    {
                        // Without just the level 0 node
                        string  strNode = "Node" + std::to_string(iZX);
                        SLNode* column  = new SLNode(columnL1->mesh(), strNode + "-L0");
                        column->translate(x, 0, z, TS_object);
                        scene->addChild(column);
                    }
                    else
                    {
                        // With LOD parent node and 3 levels
                        string     strLOD    = "LOD" + std::to_string(iZX);
                        SLNodeLOD* lod_group = new SLNodeLOD(strLOD);
                        lod_group->translate(x, 0, z, TS_object);
                        lod_group->addChildLOD(new SLNode(columnL1->mesh(), strLOD + "-L0"), 0.1f, 3);
                        lod_group->addChildLOD(new SLNode(columnL2->mesh(), strLOD + "-L1"), 0.01f, 3);
                        lod_group->addChildLOD(new SLNode(columnL3->mesh(), strLOD + "-L2"), 0.0001f, 3);
                        scene->addChild(lod_group);
                    }
                    x += offset;
                }
                z -= offset;
            }

            // Set active camera & the root pointer
            sv->camera(cam1);
            sv->doWaitOnIdle(false);
        }
    }
    else if (sceneID == SID_Benchmark7_JansUniverse) //............................................
    {
        s->name("Jan's Universe Test Scene");
        s->info(s->name());

        SLCamera* cam1 = new SLCamera("Camera 1");
        cam1->clipNear(0.1f);
        cam1->clipFar(1000);
        cam1->translation(0, 0, 75);
        cam1->focalDist(75);
        cam1->lookAt(0, 0, 0);
        cam1->background().colors(SLCol4f(0.3f, 0.3f, 0.3f));
        cam1->setInitialState();

        // Root scene node
        SLNode* scene = new SLNode;
        s->root3D(scene);
        scene->addChild(cam1);

        // Generate NUM_MAT cook-torrance materials
#ifndef SL_GLES
        const int    NUM_MAT_MESH = 100;
        SLuint const levels       = 6;
        SLuint const childCount   = 8;
#else
        const int    NUM_MAT_MESH = 20;
        SLuint const levels       = 6;
        SLuint const childCount   = 8;
#endif
        SLVMaterial materials(NUM_MAT_MESH);
        for (int i = 0; i < NUM_MAT_MESH; ++i)
        {
            /*
            SLGLProgram* spTex   = new SLGLProgramGeneric(am,
                                                        shaderPath + "PerPixCookTm.vert",
                                                        shaderPath + "PerPixCookTm.frag");*/
            SLstring matName = "mat-" + std::to_string(i);
            materials[i]     = new SLMaterial(am,
                                          matName.c_str(),
                                          nullptr,
                                          new SLGLTexture(am, texPath + "rusty-metal_2048_C.jpg"),
                                          nullptr, // new SLGLTexture(am, texPath + "rusty-metal_2048_N.jpg"),
                                          new SLGLTexture(am, texPath + "rusty-metal_2048_M.jpg"),
                                          new SLGLTexture(am, texPath + "rusty-metal_2048_R.jpg"),
                                          nullptr,
                                          nullptr);
            SLCol4f color;
            color.hsva2rgba(SLVec4f(Utils::TWOPI * (float)i / (float)NUM_MAT_MESH, 1.0f, 1.0f));
            materials[i]->diffuse(color);
        }

        // Generate NUM_MESH sphere meshes
        SLVMesh meshes(NUM_MAT_MESH);
        for (int i = 0; i < NUM_MAT_MESH; ++i)
        {
            SLstring meshName = "mesh-" + std::to_string(i);
            meshes[i]         = new SLSphere(am, 1.0f, 32, 32, meshName.c_str(), materials[i % NUM_MAT_MESH]);
        }

        // Create universe
        generateUniverse(am, s, scene, 0, levels, childCount, materials, meshes);

        sv->camera(cam1);
        sv->doWaitOnIdle(false);
    }
    else if (sceneID == SID_Benchmark8_ParticleSystemFireComplex) //...............................
    {
        if (stateGL->glHasGeometryShaders())
        {
            s->name("Fire Complex Test Scene");
            s->info(s->name());

            SLCamera* cam1 = new SLCamera("Camera 1");
            cam1->clipNear(0.1f);
            cam1->clipFar(1000);
            cam1->translation(0, 10, 40);
            cam1->focalDist(100);
            cam1->lookAt(0, 0, 0);
            cam1->background().colors(SLCol4f(0.3f, 0.3f, 0.3f));
            cam1->setInitialState();

            // Root scene node
            SLNode* root = new SLNode;
            s->root3D(root);
            root->addChild(cam1);
            const int NUM_NODES = 250;

            // Create textures and materials
            SLGLTexture* texFireCld = new SLGLTexture(am, texPath + "ParticleFirecloudTransparent_C.png");
            SLGLTexture* texFireFlm = new SLGLTexture(am, texPath + "ParticleFlames_00_8x4_C.png");
            SLGLTexture* texCircle  = new SLGLTexture(am, texPath + "ParticleCircle_05_C.png");
            SLGLTexture* texSmokeB  = new SLGLTexture(am, texPath + "ParticleCloudBlack_C.png");
            SLGLTexture* texSmokeW  = new SLGLTexture(am, texPath + "ParticleCloudWhite_C.png");

            SLVNode nodes(NUM_NODES);
            for (int i = 0; i < NUM_NODES; ++i)
            {
                SLNode* fireComplex = createComplexFire(am,
                                                        s,
                                                        false,
                                                        texFireCld,
                                                        texFireFlm,
                                                        8,
                                                        4,
                                                        texCircle,
                                                        texSmokeB,
                                                        texSmokeW);
                fireComplex->translate(-20.0f + (float)(i % 20) * 2,
                                       0.0f,
                                       -(float)((i - (i % 20)) / 20) * 4,
                                       TS_object);
                root->addChild(fireComplex);
            }

            sv->camera(cam1);
            sv->doWaitOnIdle(false);
        }
    }
    else if (sceneID == SID_Benchmark9_ParticleSystemManyParticles) //.............................
    {
        if (stateGL->glHasGeometryShaders())
        {
            s->name("Particle System number Scene");
            s->info(s->name());

            SLCamera* cam1 = new SLCamera("Camera 1");
            cam1->clipNear(0.1f);
            cam1->clipFar(1000);
            cam1->translation(0, 0, 400);
            cam1->focalDist(400);
            cam1->lookAt(0, 0, 0);
            cam1->background().colors(SLCol4f(0.3f, 0.3f, 0.3f));
            cam1->setInitialState();

            // Root scene node
            SLNode* root = new SLNode;
            root->addChild(cam1);

            // Create textures and materials
            SLGLTexture* texC        = new SLGLTexture(am, texPath + "ParticleSmoke_08_C.png");
            SLGLTexture* texFlipbook = new SLGLTexture(am, texPath + "ParticleSmoke_03_8x8_C.png");

            // Create meshes and nodes
            SLParticleSystem* ps = new SLParticleSystem(am,
                                                        1000000,
                                                        SLVec3f(-10.0f, -10.0f, -10.0f),
                                                        SLVec3f(10.0f, 10.0f, 10.0f),
                                                        4.0f,
                                                        texC,
                                                        "Particle System",
                                                        texFlipbook);
            ps->doAlphaOverLT(false);
            ps->doSizeOverLT(false);
            ps->doRotation(false);
            ps->doShape(true);
            ps->shapeType(ST_Box);
            ps->shapeScale(100.0f, 100.0f, 100.0f);
            ps->doDirectionSpeed(true);
            ps->doBlendBrightness(true);
            ps->doColor(true);
            ps->color(SLCol4f(0.875f, 0.156f, 0.875f, 1.0f));
            ps->speed(0.0f);
            SLMesh* pSMesh = ps;
            SLNode* pSNode = new SLNode(pSMesh, "Particle system node");
            root->addChild(pSNode);

            sv->camera(cam1);
            sv->doWaitOnIdle(false);
            s->root3D(root);
        }
    }

    ////////////////////////////////////////////////////////////////////////////
    // call onInitialize on all scene views to init the scenegraph and stats
    for (auto* sceneView : AppDemo::sceneViews)
        if (sceneView != nullptr)
            sceneView->onInitialize();

    if (CVCapture::instance()->videoType() != VT_NONE)
    {
        if (sv->viewportSameAsVideo())
        {
            // Pass a negative value to the start function, so that the
            // viewport aspect ratio can be adapted later to the video aspect.
            // This will be known after start.
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

#ifdef SL_USE_ENTITIES_DEBUG
    SLScene::entities.dump(true);
#endif
}
//-----------------------------------------------------------------------------
