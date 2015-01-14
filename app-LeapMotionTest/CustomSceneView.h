
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
            
            SLfloat angle;
            SLVec3f axis;
            hands[i].palmRotation().toAngleAxis(angle, axis);
            hand->rotation(angle, axis, TS_Parent);
            
            SLQuat4f test = hands[i].palmRotation();
            hand->rotation(hands[i].palmRotation(), TS_Parent);

            elbow->position(hands[i].elbowPosition());
            wrist->position(hands[i].wristPosition());
            
            arm->position(hands[i].armCenter());
            hands[i].armRotation().toAngleAxis(angle, axis);
            arm->rotation(angle, axis, TS_Parent);


            for (SLint j = 0; j < hands[i].fingers().size(); ++j)
            {
                for (SLint k = 0; k < 5; ++k) {
                    SLNode* joint = (hands[i].isLeft()) ? leftJoints[j][k] : rightJoints[j][k];

                    joint->position(hands[i].fingers()[j].jointPosition(k));
                }
                
                for (SLint k = 0; k < 4; ++k) {
                    
                    SLNode* bone = (hands[i].isLeft()) ? leftBones[j][k] : rightBones[j][k];
                    bone->position(hands[i].fingers()[j].boneCenter(k));
                    
                    SLfloat angle;
                    SLVec3f axis;
                    hands[i].fingers()[j].boneRotation(k).toAngleAxis(angle, axis);
                    bone->rotation(angle, axis, TS_Parent);
                }
            }
        }

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

        _rightFingers[fingerType][boneType] = _skeleton->getJoint(name);
    }

    /*
    leap bone types
        TYPE_METACARPAL = 0     Bone connected to the wrist inside the palm.
        TYPE_PROXIMAL = 1       Bone connecting to the palm.
        TYPE_INTERMEDIATE = 2   Bone between the tip and the base.
        TYPE_DISTAL = 3         Bone at the tip of the finger.
    */

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
            if (!hands[i].isLeft()) {
                SLQuat4f rot = hands[i].palmRotation();
                SLJoint* jnt = _rightWrist;

                SLQuat4f corrected = SLQuat4f(90.0f, SLVec3f(-1, 0, 0)) * rot * SLQuat4f(90.0f, SLVec3f(1, 0, 0));
                jnt->rotation(corrected);
            }
            
            for (SLint j = 0; j < hands[i].fingers().size(); ++j)
            {                
                for (SLint k = 0; k < 4; ++k) {
                    
                    SLJoint* bone = (hands[i].isLeft()) ? _leftFingers[j][k] : _rightFingers[j][k];
                    if (bone == NULL)
                        continue;

                    SLQuat4f relRot = hands[i].fingers()[j].boneRotation(k) * hands[i].fingers()[j].boneRotation(k-1).inverted();
                    SLQuat4f correctedRot = relRot * SLQuat4f(90.0f, SLVec3f(1, 0, 0));

                    bone->rotation(correctedRot, TS_Parent);
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
    //SampleListener      _leapListener;
    //Leap::Controller    _leapController;

    SLJoint*            _root;
    SLLeapController    _leapController;
    SampleHandListener  _slHandListener;
    SLRiggedLeapHandListener _riggedListener;

};