#ifndef APP_WAI_SCENEVIEW_H
#define APP_WAI_SCENEVIEW_H

#include <SLSceneView.h>

struct WAIEvent;

enum WAINodeEditMode
{
    WAINodeEditMode_None,
    WAINodeEditMode_Translate,
    WAINodeEditMode_Scale,
};

class WAISceneView : public SLSceneView
{
public:
    WAISceneView(std::queue<WAIEvent*>* eventQueue);
    void toggleEditMode();

    virtual SLbool onMouseDown(SLMouseButton button, SLint scrX, SLint scrY, SLKey mod);
    virtual SLbool onMouseUp(SLMouseButton button, SLint scrX, SLint scrY, SLKey mod);
    virtual SLbool onMouseMove(SLint scrX, SLint scrY);

private:
    WAINodeEditMode _editMode;

    bool    _mouseIsDown;
    SLVec3f _hitCoordinate;

    SLNode* _editGizmos = nullptr;

    // Translation stuff
    SLNode* _xAxisNode = nullptr;
    SLNode* _yAxisNode = nullptr;
    SLNode* _zAxisNode = nullptr;

    SLVec3f _axisRayO;
    SLVec3f _axisRayDir;

    // Scale stuff
    SLNode* _scaleSphere;

    bool getClosestPointOnAxis(const SLVec3f& pickRayO,
                               const SLVec3f& pickRayDir,
                               const SLVec3f& axisRayO,
                               const SLVec3f& axisRayDir,
                               SLVec3f&       axisPoint);
};

#endif