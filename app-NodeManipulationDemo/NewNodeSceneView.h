
#include <SLSceneView.h>
#pragma warning(disable:4996)
/*
This project serves as an isolated test case for the new SLNode implementation
What is working? We have an almost finished interface we want to implement
Some of the functions are already implemented but most of them are still unfinished.

Severe issues here:
    - 
    - 
    - 
*/
enum TestMovementModes
{
    TranslationMode,
    RotationMode,
    RotationAroundMode,
    LookAtMode
};

class NewNodeSceneView : public SLSceneView
{
public:   
    NewNodeSceneView()
        : _infoText(NULL), _curMode(TranslationMode),
        _curSpace(TS_Parent), _curObject(NULL),
        _continuousInput(true)
    { }

    void                preDraw();
    void                postDraw();
                        
    void                postSceneLoad();

    void                reset();
    void                translateObject(SLVec3f vec);
    void                rotateObject(SLVec3f val);
    void                rotateObjectAroundPivot(SLVec3f val);
    void                translatePivot(SLVec3f vec);

    SLbool              onContinuousKeyPress(SLKey key);
    SLbool              onKeyPress(const SLKey key, const SLKey mod);
    SLbool              onKeyRelease(const SLKey key, const SLKey mod);

    SLbool              update();
    void                updateCurOrigin();
    void                updateInfoText();
    void                renderText();

    SLMat4f             _curOrigin; //!< current origin of relative space (orientation and position of axes)

    SLNode*             _moveBox;
    SLNode*             _moveBoxChild;
    SLVec3f             _pivotPos;

    SLNode*             _axesMesh;

    SLText*             _infoText;

    bool                _buttonStates[65536];
    bool                _continuousInput;

    SLfloat             _deltaTime;

    SLuint              _curMode;
    SLNode*             _curObject;
    SLTransformSpace    _curSpace;
};