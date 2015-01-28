
#include <SLSceneView.h>
#include "Leap.h"
#include <SLLeapController.h>
#include <SLBox.h>
#include <SLSphere.h>
#include <SLCylinder.h>

#pragma warning(disable:4996)



class SampleListener : public Leap::Listener {
  public:
    virtual void onInit(const Leap::Controller&);
    virtual void onConnect(const Leap::Controller&);
    virtual void onDisconnect(const Leap::Controller&);
    virtual void onExit(const Leap::Controller&);
    virtual void onFrame(const Leap::Controller&);
    virtual void onFocusGained(const Leap::Controller&);
    virtual void onFocusLost(const Leap::Controller&);
    virtual void onDeviceChange(const Leap::Controller&);
    virtual void onServiceConnect(const Leap::Controller&);
    virtual void onServiceDisconnect(const Leap::Controller&);

    SLVec3f posLeft;
    SLVec3f posRight;

    SLVec3f positionsLeft[26];
    SLVec3f positionsRight[26];
};


class SampleHandListener : public SLLeapHandListener
{
public:
    void init()
    {
        SLScene* s = SLScene::current;
        
        SLMesh* palmMesh = new SLBox(-0.15f, -0.05f, -0.15f, 0.15f, 0.05f, 0.15f);
        SLMesh* jointMesh = new SLSphere(0.05f);
        SLMesh* jointMeshBig = new SLSphere(0.1f);
        SLMesh* boneMesh = new SLCylinder(0.015f, 1.0f);
        
        leftHand = new SLNode(palmMesh);
        rightHand = new SLNode(palmMesh);
        
        leftArm = new SLNode;
        rightArm = new SLNode;
        SLNode* tempCont = new SLNode(boneMesh);
        tempCont->position(0, 0, -1.4f);
        tempCont->scale(2.8f);
        leftArm->addChild(tempCont);
        tempCont = new SLNode(boneMesh);
        tempCont->position(0, 0, -1.4f);
        tempCont->scale(2.8f);
        rightArm->addChild(tempCont);

        leftWrist = new SLNode(jointMeshBig);
        rightWrist = new SLNode(jointMeshBig);
        leftElbow = new SLNode(jointMeshBig);
        rightElbow = new SLNode(jointMeshBig);
        
        // generate joints and bones for fingers
        SLfloat boneScales[5][4] = { 
            { 0.01f, 0.36f, 0.3f, 0.15f},  // thumb
            { 0.6f, 0.36f, 0.3f, 0.12f},  // index
            { 0.6f, 0.36f, 0.33f, 0.12f},  // middle
            { 0.5f, 0.36f, 0.3f, 0.12f},  // ring
            { 0.5f, 0.3f, 0.25f, 0.12f}   // pinky
        };

        for (SLint i = 0; i < 5; ++i) {
            // joints
            for (SLint j = 0; j < 5; ++j) {
                leftJoints[i][j] = new SLNode(jointMesh);
                rightJoints[i][j] = new SLNode(jointMesh);

                s->root3D()->addChild(leftJoints[i][j]);
                s->root3D()->addChild(rightJoints[i][j]);
            }
            // bones
            for (SLint j = 0; j < 4; ++j) {
                SLNode* meshCont = new SLNode(boneMesh);
                meshCont->position(0, 0, -0.5 * boneScales[i][j]);
                meshCont->scale(1, 1, boneScales[i][j]);
                
                leftBones[i][j] = new SLNode;
                leftBones[i][j]->addChild(meshCont);
                s->root3D()->addChild(leftBones[i][j]);

                
                meshCont = new SLNode(boneMesh);
                meshCont->position(0, 0, -0.5 * boneScales[i][j]);
                meshCont->scale(1, 1, boneScales[i][j]);
                
                rightBones[i][j] = new SLNode;
                rightBones[i][j]->addChild(meshCont);
                s->root3D()->addChild(rightBones[i][j]);
            }
        }


        
        s->root3D()->addChild(leftHand);
        s->root3D()->addChild(rightHand);
        
        s->root3D()->addChild(leftArm);
        s->root3D()->addChild(rightArm);
        s->root3D()->addChild(leftElbow);
        s->root3D()->addChild(rightElbow);
        s->root3D()->addChild(leftWrist);
        s->root3D()->addChild(rightWrist);
    }

protected:
    SLNode* leftHand;
    SLNode* rightHand;
    SLint leftId;
    SLint rightId;
    
    SLNode* leftJoints[5][5];
    SLNode* leftBones[5][4];
    SLNode* rightJoints[5][5];
    SLNode* rightBones[5][4];
    
    SLNode* leftArm;
    SLNode* rightArm;
    SLNode* leftElbow;
    SLNode* rightElbow;
    SLNode* leftWrist;
    SLNode* rightWrist;

    virtual void onLeapHandChange(const vector<SLLeapHand>& hands)
    {
        for (SLint i = 0; i < hands.size(); ++i)
        {
            SLNode* hand = (hands[i].isLeft()) ? leftHand : rightHand;
            SLNode* elbow = (hands[i].isLeft()) ? leftElbow : rightElbow;
            SLNode* wrist = (hands[i].isLeft()) ? leftWrist : rightWrist;
            SLNode* arm = (hands[i].isLeft()) ? leftArm : rightArm;
            
            hand->position(hands[i].palmPosition());            
            hand->rotation(hands[i].palmRotation(), TS_World);
            
            SLQuat4f test = hands[i].palmRotation();
            hand->rotation(hands[i].palmRotation(), TS_World);

            elbow->position(hands[i].elbowPosition());
            wrist->position(hands[i].wristPosition());
            
            arm->position(hands[i].armCenter());
            arm->rotation(hands[i].armRotation(), TS_World);


            for (SLint j = 0; j < hands[i].fingers().size(); ++j)
            {
                // set joint positions
                for (SLint k = 0; k < 5; ++k) {
                    SLNode* joint = (hands[i].isLeft()) ? leftJoints[j][k] : rightJoints[j][k];
                    joint->position(hands[i].fingers()[j].jointPosition(k));
                }
                
                // set bone positions
                for (SLint k = 0; k < 4; ++k) {                    
                    SLNode* bone = (hands[i].isLeft()) ? leftBones[j][k] : rightBones[j][k];
                    bone->position(hands[i].fingers()[j].boneCenter(k));
                    bone->rotation(hands[i].fingers()[j].boneRotation(k), TS_World);
                }
            }
        }

    }
};

class SampleToolListener : public SLLeapToolListener
{
public:
    void init()
    {
        SLScene* s = SLScene::current;
        
        SLMesh* toolTipMesh = new SLSphere(0.03f);
        SLMesh* toolMesh = new SLCylinder(0.015f, 5.0f);
        
        _toolNode = new SLNode;
        _toolNode->addMesh(toolTipMesh);
        _toolNode->addMesh(toolMesh);

        s->root3D()->addChild(_toolNode);
    }

protected:
    // only one tool allowed currently
    SLNode* _toolNode;

    virtual void onLeapToolChange(const vector<SLLeapTool>& tools)
    {
        if (tools.size() == 0)
            return;

        const SLLeapTool& tool = tools[0];

        _toolNode->position(tool.toolTipPosition());
        _toolNode->rotation(tool.toolRotation());
    }
};

class SampleGestureListener : public SLLeapGestureListener
{
protected:
    virtual void onLeapGesture(const SLLeapGesture& gesture)
    {/*
        switch (gesture.type())
        {
        case SLLeapGesture::Swipe: SL_LOG("SWIPE\n") break;
        case SLLeapGesture::KeyTap: SL_LOG("KEY TAP\n") break;
        case SLLeapGesture::ScreenTap: SL_LOG("SCREEN TAP\n") break;
        case SLLeapGesture::Circle: SL_LOG("CIRCLE\n") break;
        }*/
    }
};


// @note this is just temporary test code here! Clean it up and implement it better for the final product
class SLRiggedLeapHandListener : public SLLeapHandListener
{
public:
    SLRiggedLeapHandListener()
    {
        for (SLint i = 0; i < 5; ++i) {
            for (SLint j = 0; j < 4; ++j) {
                _leftFingers[i][j] = _rightFingers[i][j] = NULL;
            }
        }
        axis1 = SLVec3f(1, 0, 0);
        axis2 = SLVec3f(0, 1, 0);
        axis2 = SLVec3f(0, 0, 1);
        dir = 1;
    }

    void setSkeleton(SLSkeleton* skel){
        _skeleton = skel;
    }
    
    void setLWrist(const SLstring& name)
    {
        _leftWrist = _skeleton->getJoint(name);
    }
    void setRWrist(const SLstring& name)
    {
        _rightWrist = _skeleton->getJoint(name);
    }
    void setLArm(const SLstring& name)
    {

    }
    void setRArm(const SLstring& name)
    {

    }
    // @todo provide enums for finger type and bone type
    void setLFingerJoint(SLint fingerType, SLint boneType, const SLstring& name)
    {
        if (!_skeleton) return;

        _leftFingers[fingerType][boneType] = _skeleton->getJoint(name);
    }
    void setRFingerJoint(SLint fingerType, SLint boneType, const SLstring& name)
    {
        if (!_skeleton) return;

        if (_skeleton->getJoint(name))
            _skeleton->getJoint(name)->setInitialState();
        _rightFingers[fingerType][boneType] = _skeleton->getJoint(name);
    }

    /*
    leap bone types
        TYPE_METACARPAL = 0     Bone connected to the wrist inside the palm.
        TYPE_PROXIMAL = 1       Bone connecting to the palm.
        TYPE_INTERMEDIATE = 2   Bone between the tip and the base.
        TYPE_DISTAL = 3         Bone at the tip of the finger.
    */

    SLQuat4f correction1;
    SLQuat4f correction2;
    SLVec3f axis1;
    SLVec3f axis2;
    SLVec3f axis3;
    int dir;

protected:
    
    SLSkeleton* _skeleton;
    SLJoint* _leftFingers[5][4];
    SLJoint* _rightFingers[5][4];
    SLJoint* _leftWrist;
    SLJoint* _rightWrist;

    virtual void onLeapHandChange(const vector<SLLeapHand>& hands)
    {
        for (SLint i = 0; i < hands.size(); ++i)
        {
            SLQuat4f rot = hands[i].palmRotation();
            SLJoint* jnt = (hands[i].isLeft()) ? _leftWrist : _rightWrist;

            jnt->rotation(rot, TS_World);
            jnt->position(hands[i].palmPosition()*7.5, TS_World);
            
            for (SLint j = 0; j < hands[i].fingers().size(); ++j)
            {                
                for (SLint k = 0; k < 4; ++k) {                    
                    SLJoint* bone = (hands[i].isLeft()) ? _leftFingers[j][k] : _rightFingers[j][k];
                    if (bone == NULL)
                        continue;

                    bone->rotation(hands[i].fingers()[j].boneRotation(k), TS_World);
                }
            }
        }
    }
};

class SampleObjectMover : public SLLeapHandListener
{
public:
    SampleObjectMover()
    : grabThreshold(0.9f)
    { }

    void setGrabThreshold(SLfloat t) { grabThreshold = t; }
    void setGrabCallback(std::function<void(SLVec3f&, SLQuat4f&)> cb) { grabCallback = cb; }
    void setReleaseCallback(std::function<void()> cb) { releaseCallback = cb; }
    void setMoveCallback(std::function<void(SLVec3f&, SLQuat4f&)> cb) { moveCallback = cb; }

protected:
    SLfloat grabThreshold;
    SLbool grabbing;

    std::function<void(SLVec3f&, SLQuat4f&)> grabCallback;
    std::function<void()> releaseCallback;
    std::function<void(SLVec3f&, SLQuat4f&)> moveCallback;

    virtual void onLeapHandChange(const vector<SLLeapHand>& hands)
    {
        for (SLint i = 0; i < hands.size(); ++i)
        {
            SLLeapHand hand = hands[i];

            // just use the right hand for a first test
            if (hand.isLeft())
                return;

            // For now just the intermediate position between thumb and index finger
            // @note  a pinch can also be between any other finger and the tumb, so this
            //        currently only works for index and thumb pinches
            SLVec3f grabPosition = hand.fingers()[0].tipPosition() + 
                                   hand.fingers()[1].tipPosition();
            grabPosition *= 0.5f;
            SLQuat4f palmRotation = hand.palmRotation();

            if (hand.pinchStrength() > grabThreshold) {
                if (grabbing) {
                    moveCallback(grabPosition, palmRotation);
                }
                else {
                    grabCallback(grabPosition, palmRotation);
                    grabbing = true;
                }
            }
            else {
                if (grabbing) {
                    releaseCallback();
                    grabbing = false;
                }
            }

        }
    }
};

class CustomSceneView : public SLSceneView
{
public:   
    CustomSceneView()
    { }
    ~CustomSceneView();

    void                preDraw();
    void                postDraw();
                        
    void                postSceneLoad();
    
    SLbool              onKeyPress(const SLKey key, const SLKey mod);
    SLbool              onKeyRelease(const SLKey key, const SLKey mod);

    SLbool              update();

private:
    SLLeapController            _leapController;
    SampleHandListener          _slHandListener;
    SLRiggedLeapHandListener    _riggedListener;
    SampleToolListener          _slToolListener;
    SampleGestureListener       _gestureListener;
    SampleObjectMover           _objectMover;
    vector<SLNode*>             _movableBoxes;
    SLNode*                     _currentGrabbedObject;
    SLQuat4f                    _prevRotation;
    SLVec3f                     _prevPosition;
    SLQuat4f                    _initialRotation;
    SLVec3f                     _initialPosition;

    void grabCallback(SLVec3f&, SLQuat4f&);
    void moveCallback(SLVec3f&, SLQuat4f&);
    void releaseCallback();
};