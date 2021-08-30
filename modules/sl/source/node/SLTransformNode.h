//#############################################################################
//  File:      SLTransformNode.h
//  Authors:   Jan Dellsperger
//  Date:      July 2016
//  Codestyle: https://github.com/cpvrlab/SLProject/wiki/SLProject-Coding-Style
//  Authors:   Jan Dellsperger
//  License:   This software is provided under the GNU General Public License
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
//! Class that holds all visible gizmo node during mouse transforms
/*!
 * An SLTransformNode is added to the scenegraph during edit mode. Depending
 on the transform type (translation, rotation or scaling) arrows or disks and
 circles get shown or hidden. The transform node gets added and removed on the
 fly. So the assets have to be delete in the destructor.
 */
class SLTransformNode : public SLNode
{
public:
    SLTransformNode(SLSceneView* sv,
                    SLNode*      targetNode,
                    SLstring     shaderDir);
    ~SLTransformNode() override;

    virtual SLbool onMouseDown(SLMouseButton button,
                               SLint         x,
                               SLint         y,
                               SLKey         mod) override;
    virtual SLbool onMouseUp(SLMouseButton button,
                             SLint         x,
                             SLint         y,
                             SLKey         mod) override;
    virtual SLbool onMouseMove(SLMouseButton button,
                               SLint         x,
                               SLint         y,
                               SLKey         mod) override;

    virtual void editMode(SLNodeEditMode editMode);

    SLNode*        targetNode() { return _targetNode; }
    SLNodeEditMode editMode() { return _editMode; }

private:
    SLSceneView* _sv         = nullptr;
    SLNode*      _targetNode = nullptr;

    SLNodeEditMode _editMode;

    SLGLProgram* _prog  = nullptr;
    SLMaterial*  _matR  = nullptr;
    SLMaterial*  _matG  = nullptr;
    SLMaterial*  _matB  = nullptr;
    SLMaterial*  _matY  = nullptr;
    SLMaterial*  _matRT = nullptr;
    SLMaterial*  _matGT = nullptr;
    SLMaterial*  _matBT = nullptr;
    SLMaterial*  _matYT = nullptr;
    SLMesh*      _axisR = nullptr;
    SLMesh*      _axisG = nullptr;
    SLMesh*      _axisB = nullptr;
    SLMesh*      _lineR = nullptr;
    SLMesh*      _lineG = nullptr;
    SLMesh*      _lineB = nullptr;
    SLMesh*      _circR = nullptr;
    SLMesh*      _circG = nullptr;
    SLMesh*      _circB = nullptr;
    SLMesh*      _circY = nullptr;
    SLMesh*      _diskR = nullptr;
    SLMesh*      _diskG = nullptr;
    SLMesh*      _diskB = nullptr;
    SLMesh*      _diskY = nullptr;

    bool    _mouseIsDown;
    float   _gizmoScale;
    SLVec3f _hitCoordinate;
    SLNode* _selectedGizmo = nullptr;

    // Translation stuff
    SLNode* _transGizmos = nullptr;
    SLNode* _transLineX  = nullptr;
    SLNode* _transLineY  = nullptr;
    SLNode* _transLineZ  = nullptr;

    // Scale stuff
    SLNode* _scaleGizmos = nullptr;
    SLNode* _scaleDisk   = nullptr;
    SLNode* _scaleCirc   = nullptr;

    // Rotation stuff
    SLNode* _rotGizmos = nullptr;
    SLNode* _rotCircX  = nullptr;
    SLNode* _rotDiskX  = nullptr;
    SLNode* _rotCircY  = nullptr;
    SLNode* _rotDiskY  = nullptr;
    SLNode* _rotCircZ  = nullptr;
    SLNode* _rotDiskZ  = nullptr;

    // Node that contains all gizmos
    SLNode* _gizmosNode = nullptr;

    bool getClosestPointsBetweenRays(const SLVec3f& ray1O,
                                     const SLVec3f& ray1Dir,
                                     const SLVec3f& ray2O,
                                     const SLVec3f& ray2Dir,
                                     SLVec3f&       ray1P,
                                     float&         t1,
                                     SLVec3f&       ray2P,
                                     float&         t2);
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
