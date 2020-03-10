#ifndef APP_WAI_SCENEVIEW_H
#define APP_WAI_SCENEVIEW_H

#include <SLSceneView.h>

struct WAIEvent;

class WAISceneView : public SLSceneView
{
public:
    WAISceneView(std::queue<WAIEvent*>* eventQueue);
    void toggleEditMode();

    virtual SLbool onMouseDown(SLMouseButton button, SLint scrX, SLint scrY, SLKey mod);
    virtual SLbool onMouseUp(SLMouseButton button, SLint scrX, SLint scrY, SLKey mod);
    virtual SLbool onMouseMove(SLint scrX, SLint scrY);

protected:
    std::queue<WAIEvent*>* _eventQueue;

    bool _editMode;
    bool _mouseIsDown;

    SLVec2i _mouseDownCoordinates;
    SLVec3f _hitCoordinate;

    SLNode* _editGizmos = nullptr;
    SLNode* _mapNode    = nullptr;
};

#endif