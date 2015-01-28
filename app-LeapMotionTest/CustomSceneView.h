
#include <SLSceneView.h>
#include "Leap.h"
#include <SLLeapController.h>
#include <SLBox.h>
#include <SLSphere.h>
#include <SLCylinder.h>

#include <SampleListeners.h>

#pragma warning(disable:4996)


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

    void grabCallback(SLVec3f&, SLQuat4f&);
    void moveCallback(SLVec3f&, SLQuat4f&);
    void releaseCallback();
};