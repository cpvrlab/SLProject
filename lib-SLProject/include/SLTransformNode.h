//#############################################################################
//  File:      SLTransformNode.h
//  Author:    Jan Dellsperger
//  Date:      July 2016
//  Codestyle: https://github.com/cpvrlab/SLProject/wiki/SLProject-Coding-Style
//  Copyright: Jan Dellsperger
//             This software is provide under the GNU General Public License
//             Please visit: http://opensource.org/licenses/GPL-3.0
//#############################################################################

#ifndef SL_TRANSFORMATION_NODE_H
#define SL_TRANSFORMATION_NODE_H

#include <SLSceneView.h>

//-----------------------------------------------------------------------------
enum SLNodeEditMode
{
    NodeEditMode_None,
    NodeEditMode_Translate,
    NodeEditMode_Scale,
    NodeEditMode_Rotate
};
//-----------------------------------------------------------------------------
class SLTransformNode : public SLNode
{
public:
    SLTransformNode(SLAssetManager* assetMgr,
                    SLSceneView*    sv,
                    SLNode*         targetNode);
    ~SLTransformNode() override;

    SLbool onMouseDown(SLMouseButton button,
                       SLint         x,
                       SLint         y,
                       SLKey         mod) override;
    SLbool onMouseUp(SLMouseButton button,
                     SLint         x,
                     SLint         y,
                     SLKey         mod) override;
    SLbool onMouseMove(SLMouseButton button,
                       SLint         x,
                       SLint         y,
                       SLKey         mod) override;

    void editMode(SLNodeEditMode editMode);

    SLNode*        targetNode() { return _targetNode; }
    SLNodeEditMode editMode() { return _editMode; }

private:
    SLSceneView* _sv         = nullptr;
    SLNode*      _targetNode = nullptr;

    SLNodeEditMode _editMode;

    //SLNode* _editGizmos = nullptr;

    bool    _mouseIsDown;
    float   _gizmoScale;
    SLVec3f _hitCoordinate;
    SLNode* _selectedGizmo = nullptr;

    // Translation stuff
    SLNode* _translationAxisX = nullptr;
    SLNode* _translationAxisY = nullptr;
    SLNode* _translationAxisZ = nullptr;
    SLNode* _translationLineX = nullptr;
    SLNode* _translationLineY = nullptr;
    SLNode* _translationLineZ = nullptr;

    // Scale stuff
    SLNode* _scaleGizmos;
    SLNode* _scaleDisk;
    SLNode* _scaleCircle;

    // Rotation stuff
    SLNode* _rotationCircleX;
    SLNode* _rotationDiskX;
    SLNode* _rotationCircleY;
    SLNode* _rotationDiskY;
    SLNode* _rotationCircleZ;
    SLNode* _rotationDiskZ;

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
    bool isCCW(const SLVec2f& a, const SLVec2f& b, const SLVec2f& c);
    void setDrawBitRecursive(SLuint bit, SLNode* node, bool value);
    void lookAt(SLNode* node, SLCamera* camera);
};
//-----------------------------------------------------------------------------
#endif