#ifndef APP_WAI_SCENEVIEW_H
#define APP_WAI_SCENEVIEW_H

#include <SLSceneView.h>
#include <SLCircle.h>

struct WAIEvent;

enum WAINodeEditMode
{
    WAINodeEditMode_None,
    WAINodeEditMode_Translate,
    WAINodeEditMode_Scale,
    WAINodeEditMode_Rotate
};

class WAISceneView : public SLSceneView
{
public:
    WAISceneView(std::queue<WAIEvent*>* eventQueue);
    void toggleEditMode(WAINodeEditMode editMode);

    virtual SLbool onMouseDown(SLMouseButton button, SLint scrX, SLint scrY, SLKey mod);
    virtual SLbool onMouseUp(SLMouseButton button, SLint scrX, SLint scrY, SLKey mod);
    virtual SLbool onMouseMove(SLint scrX, SLint scrY);

private:
    WAINodeEditMode _editMode;

    bool    _mouseIsDown;
    SLVec3f _hitCoordinate;

    SLNode* _editGizmos = nullptr;
    float   _gizmoScale;

    // Translation stuff
    SLNode* _translationAxisX = nullptr;
    SLNode* _translationAxisY = nullptr;
    SLNode* _translationAxisZ = nullptr;

    SLNode* _translationLineX = nullptr;
    SLNode* _translationLineY = nullptr;
    SLNode* _translationLineZ = nullptr;

    SLNode* _selectedTranslationAxis = nullptr;

    // Scale stuff
    SLNode* _scaleGizmos;
    SLNode* _scaleDisk;
    SLNode* _scaleCircle;
    float   _oldScaleRadius;

    // Rotation stuff
    SLNode* _rotationGizmos;

    SLNode* _rotationGizmosX;
    SLNode* _rotationGizmosY;
    SLNode* _rotationGizmosZ;

    SLNode* _rotationCircleX;
    SLNode* _rotationDiskX;
    SLNode* _rotationCircleY;
    SLNode* _rotationDiskY;
    SLNode* _rotationCircleZ;
    SLNode* _rotationDiskZ;

    SLNode* _rotationCircleNode;
    SLVec3f _rotationAxis;
    SLVec3f _rotationStartPoint;
    SLVec3f _rotationStartVec;

    SLVec2f _oldMouseCoords;

    bool getClosestPointsBetweenRays(const SLVec3f& ray1O,
                                     const SLVec3f& ray1Dir,
                                     const SLVec3f& ray2O,
                                     const SLVec3f& ray2Dir,
                                     SLVec3f&       ray1P,
                                     SLVec3f&       ray2P);
    bool getClosestPointOnAxis(const SLVec3f& pickRayO,
                               const SLVec3f& pickRayDir,
                               const SLVec3f& axisRayO,
                               const SLVec3f& axisRayDir,
                               SLVec3f&       axisPoint);
    bool rayDiscIntersect(const SLVec3f& rayO,
                          const SLVec3f& rayDir,
                          const SLVec3f& discO,
                          const SLVec3f& discN,
                          const float&   distR,
                          float&         t);
    bool rayPlaneIntersect(const SLVec3f& rayO,
                           const SLVec3f& rayDir,
                           const SLVec3f& discO,
                           const SLVec3f& discN,
                           float&         t);
    bool isCCW(SLVec2f a, SLVec2f b, SLVec2f c);
    void toggleHideRecursive(SLNode* node, bool hidden);
    void lookAt(SLNode* node, SLCamera* camera);
};

#endif