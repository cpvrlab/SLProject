//#############################################################################
//  File:      SLTransformNode.cpp
//  Authors:   Jan Dellsperger
//  Date:      April 2020
//  Codestyle: https://github.com/cpvrlab/SLProject/wiki/SLProject-Coding-Style
//  Authors:   Jan Dellsperger, Marcus Hudritsch
//  License:   This software is provided under the GNU General Public License
//             Please visit: http://opensource.org/licenses/GPL-3.0
//#############################################################################

#include <SLTransformNode.h>
#include <SLVec3.h>
#include <SLCoordAxisArrow.h>
#include <SLCircle.h>
#include <SLDisk.h>

//-----------------------------------------------------------------------------
/*!
 Constructor for a transform node.
 Because a transform node will be added and removed on the fly to the
 scenegraph it well be the owner of its meshes (SLMesh), materials (SLMaterial)
 and shader programs (SLGLProgram). It has to delete them in the destructor.
 @param sv Pointer to the SLSceneView
 @param targetNode Pointer to the node that should be transformed.
 @param shaderDir Path to the shader files
 */
SLTransformNode::SLTransformNode(SLSceneView* sv,
                                 SLNode*      targetNode,
                                 SLstring     shaderDir)
  : SLNode("Edit Gizmos"),
    _sv(sv),
    _targetNode(targetNode),
    _editMode(NodeEditMode_None),
    _mouseIsDown(false),
    _gizmoScale(1.0f)
{
    _prog = new SLGLProgramGeneric(nullptr, shaderDir + "ColorUniformPoint.vert", shaderDir + "Color.frag");
    _prog->addUniform1f(new SLGLUniform1f(UT_const, "u_pointSize", 3.0f));

    _matR  = new SLMaterial(nullptr, "Red Opaque", SLCol4f::RED, SLVec4f::WHITE, 100.0f, 0.0f, 0.0f, 0.0f, _prog);
    _matRT = new SLMaterial(nullptr, "Red Transp", SLCol4f::RED, SLVec4f::WHITE, 100.0f, 0.0f, 0.5f, 0.0f, _prog);
    _matG  = new SLMaterial(nullptr, "Green Opaque", SLCol4f::GREEN, SLVec4f::WHITE, 100.0f, 0.0f, 0.0f, 0.0f, _prog);
    _matGT = new SLMaterial(nullptr, "Green Transp", SLCol4f::GREEN, SLVec4f::WHITE, 100.0f, 0.0f, 0.5f, 0.0f, _prog);
    _matB  = new SLMaterial(nullptr, "Blue Opaque", SLCol4f::BLUE, SLVec4f::WHITE, 100.0f, 0.0f, 0.0f, 0.0f, _prog);
    _matBT = new SLMaterial(nullptr, "Blue Transp", SLCol4f::BLUE, SLVec4f::WHITE, 100.0f, 0.0f, 0.5f, 0.0f, _prog);
    _matY  = new SLMaterial(nullptr, "Yellow Opaque", SLCol4f::YELLOW, SLVec4f::WHITE, 100.0f, 0.0f, 0.0f, 0.0f, _prog);
    _matYT = new SLMaterial(nullptr, "Yellow Transp", SLCol4f::YELLOW, SLVec4f::WHITE, 100.0f, 0.0f, 0.5f, 0.0f, _prog);

    _axisR             = new SLCoordAxisArrow(nullptr, _matRT);
    _axisG             = new SLCoordAxisArrow(nullptr, _matGT);
    _axisB             = new SLCoordAxisArrow(nullptr, _matBT);
    SLNode* transAxisX = new SLNode(_axisR, "x-axis node");
    SLNode* transAxisY = new SLNode(_axisG, "y-axis node");
    SLNode* transAxisZ = new SLNode(_axisB, "z-axis node");
    transAxisX->rotate(-90.0f, SLVec3f(0.0f, 0.0f, 1.0f));
    transAxisZ->rotate(90.0f, SLVec3f(1.0f, 0.0f, 0.0f));
    transAxisX->castsShadows(false);
    transAxisY->castsShadows(false);
    transAxisZ->castsShadows(false);

    SLVec3f  startPoint = SLVec3f(0.0f, 0.0f, -1.0f);
    SLVec3f  endPoint   = SLVec3f(0.0f, 0.0f, 1.0f);
    SLVVec3f points;
    points.push_back(startPoint);
    points.push_back(endPoint);

    _lineR      = new SLPolyline(nullptr, points, false, "Translation Line Mesh X", _matR);
    _transLineX = new SLNode(_lineR);
    _transLineX->rotate(90.0f, SLVec3f(0.0f, 1.0f, 0.0f));
    _transLineX->scale(1000.0f);
    _transLineX->castsShadows(false);

    _lineG      = new SLPolyline(nullptr, points, false, "Translation Line Mesh Y", _matG);
    _transLineY = new SLNode(_lineG);
    _transLineY->rotate(-90.0f, SLVec3f(1.0f, 0.0f, 0.0f));
    _transLineY->scale(1000.0f);
    _transLineY->castsShadows(false);

    _lineB      = new SLPolyline(nullptr, points, false, "Translation Line Mesh Z", _matB);
    _transLineZ = new SLNode(_lineB);
    _transLineZ->scale(1000.0f);
    _transLineZ->castsShadows(false);

    _circY     = new SLCircle(nullptr, "Scale Circle Mesh", _matY);
    _scaleCirc = new SLNode(_circY, "Scale Circle");
    _scaleCirc->castsShadows(false);
    _diskY     = new SLDisk(nullptr, 1.0f, SLVec3f::AXISZ, 36U, true, "Scale Disk", _matYT);
    _scaleDisk = new SLNode(_diskY, "Scale Disk");
    _scaleDisk->castsShadows(false);

    _circR    = new SLCircle(nullptr, "Rotation Circle Mesh X", _matR);
    _rotCircX = new SLNode(_circR, "Rotation Circle X");
    _rotCircX->castsShadows(false);
    _diskR    = new SLDisk(nullptr, 1.0f, SLVec3f::AXISZ, 36U, true, "Rotation Disk X", _matRT);
    _rotDiskX = new SLNode(_diskR, "Rotation Disk X");
    _rotDiskX->castsShadows(false);
    _circG    = new SLCircle(nullptr, "Rotation Circle Mesh Y", _matG);
    _rotCircY = new SLNode(_circG, "Rotation Circle Y");
    _rotCircY->castsShadows(false);
    _diskG    = new SLDisk(nullptr, 1.0f, SLVec3f::AXISZ, 36U, true, "Rotation Disk Y", _matGT);
    _rotDiskY = new SLNode(_diskG, "Rotation Disk Y");
    _rotDiskY->castsShadows(false);
    _circB    = new SLCircle(nullptr, "Rotation Circle Mesh Z", _matB);
    _rotCircZ = new SLNode(_circB, "Rotation Circle Z");
    _rotCircZ->castsShadows(false);
    _diskB    = new SLDisk(nullptr, 1.0f, SLVec3f::AXISZ, 36U, true, "Rotation Disk Z", _matBT);
    _rotDiskZ = new SLNode(_diskB, "Rotation Disk Z");
    _rotDiskZ->castsShadows(false);

    SLNode* rotationGizmosX = new SLNode("Rotation Gizmos X");
    rotationGizmosX->addChild(_rotCircX);
    rotationGizmosX->addChild(_rotDiskX);

    SLNode* rotationGizmosY = new SLNode("Rotation Gizmos Y");
    rotationGizmosY->addChild(_rotCircY);
    rotationGizmosY->addChild(_rotDiskY);

    SLNode* rotationGizmosZ = new SLNode("Rotation Gizmos Z");
    rotationGizmosZ->addChild(_rotCircZ);
    rotationGizmosZ->addChild(_rotDiskZ);

    _selectedGizmo = nullptr;

    rotationGizmosX->rotate(90.0f, SLVec3f(0.0f, 1.0f, 0.0f));
    rotationGizmosY->rotate(-90.0f, SLVec3f(1.0f, 0.0f, 0.0f));

    _transGizmos = new SLNode("Translation Gizmos");
    _transGizmos->addChild(transAxisX);
    _transGizmos->addChild(_transLineX);
    _transGizmos->addChild(transAxisY);
    _transGizmos->addChild(_transLineY);
    _transGizmos->addChild(transAxisZ);
    _transGizmos->addChild(_transLineZ);

    _scaleGizmos = new SLNode("Scale Gizmos");
    _scaleGizmos->addChild(_scaleCirc);
    _scaleGizmos->addChild(_scaleDisk);

    _rotGizmos = new SLNode("Rotation Gizmos");
    _rotGizmos->addChild(rotationGizmosX);
    _rotGizmos->addChild(rotationGizmosY);
    _rotGizmos->addChild(rotationGizmosZ);

    _gizmosNode = new SLNode("Gizmos");
    _gizmosNode->addChild(_transGizmos);
    _gizmosNode->addChild(_scaleGizmos);
    _gizmosNode->addChild(_rotGizmos);
    this->addChild(_gizmosNode);

    this->updateAABBRec(true);

    setDrawBitRecursive(SL_DB_OVERDRAW, _gizmosNode, true);

    _sv->s()->eventHandlers().push_back(this);
}
//-----------------------------------------------------------------------------
/*!
 * Destructor for a transform node.
 * Because a transform node will be added and removed on the fly to the
 * scenegraph it is the owner of its meshes (SLMesh), materials (SLMaterial)
 * and shader programs (SLGLProgram). It has to delete them in here.
 */
SLTransformNode::~SLTransformNode()
{
    // delete gizmos
    _gizmosNode->deleteChildren();
    // delete _gizmosNode;
    this->deleteChild(_gizmosNode);
    this->deleteChildren();
    // delete all programs, materials and meshes
    delete _prog;
    delete _matR;
    delete _matG;
    delete _matB;
    delete _matY;
    delete _matRT;
    delete _matGT;
    delete _matBT;
    delete _matYT;
    delete _axisR;
    delete _axisG;
    delete _axisB;
    delete _lineR;
    delete _lineG;
    delete _lineB;
    delete _circR;
    delete _circG;
    delete _circB;
    delete _circY;
    delete _diskR;
    delete _diskG;
    delete _diskB;
    delete _diskY;
}
//-----------------------------------------------------------------------------
/*!
 * Setter function for the edit mode. It shows or hides the appropriate gizmo
 * meshes for the mouse interaction.
 * @param editMode New edit mode to switch to.
 */
void SLTransformNode::editMode(SLNodeEditMode editMode)
{
    _editMode = editMode;

    if (_editMode)
    {
        if (_targetNode)
        {
            _gizmosNode->translation(_targetNode->updateAndGetWM().translation());

            SLVec2f p1 = _sv->camera()->projectWorldToNDC(SLVec4f(_gizmosNode->translationWS()));
            SLVec2f p2 = _sv->camera()->projectWorldToNDC(SLVec4f(_gizmosNode->translationWS() +
                                                                  _sv->camera()->upWS().normalize()));

            float actualHeight = (p1 - p2).length();
            float targetHeight = 0.2f; // % of screen that gizmos should occupy
            float scaleFactor  = targetHeight / actualHeight;

            _gizmosNode->scale(scaleFactor / _gizmoScale);
            _gizmoScale = scaleFactor;

            setDrawBitRecursive(SL_DB_HIDDEN, _gizmosNode, true);
            _gizmosNode->drawBits()->set(SL_DB_HIDDEN, false);

            switch (_editMode)
            {
                case NodeEditMode_Translate:
                {
                    setDrawBitRecursive(SL_DB_HIDDEN, _transGizmos, false);
                }
                break;

                case NodeEditMode_Scale:
                {
                    if (_sv->camera())
                    {
                        // TODO(dgj1): this behaviour is that of a billboard... introduce in SLProject?
                        lookAt(_scaleGizmos, _sv->camera());
                    }

                    setDrawBitRecursive(SL_DB_HIDDEN, _scaleGizmos, false);
                }
                break;

                case NodeEditMode_Rotate:
                {
                    setDrawBitRecursive(SL_DB_HIDDEN, _rotGizmos, false);
                }
                break;

                case NodeEditMode_None:
                default:
                {
                }
                break;
            }
        }
    }
    else
    {
        setDrawBitRecursive(SL_DB_HIDDEN, _gizmosNode, true);
        _editMode = NodeEditMode_None;
    }
}
//-----------------------------------------------------------------------------
//! onMouseDown event handler during editing interaction
SLbool SLTransformNode::onMouseDown(SLMouseButton button,
                                    SLint         x,
                                    SLint         y,
                                    SLKey         mod)
{
    bool result = false;

    if (_editMode && _selectedGizmo)
    {
        result       = true;
        _mouseIsDown = true;
    }

    return result;
}
//-----------------------------------------------------------------------------
//! onMouseUp event handler during editing interaction
SLbool SLTransformNode::onMouseUp(SLMouseButton button,
                                  SLint         x,
                                  SLint         y,
                                  SLKey         mod)
{
    bool result = false;

    if (_editMode && _mouseIsDown)
    {
        result = true;

        if (_targetNode)
        {
            _gizmosNode->translation(_targetNode->updateAndGetWM().translation());
        }

        _selectedGizmo = nullptr;
        _mouseIsDown   = false;
    }

    return result;
}
//-----------------------------------------------------------------------------
//! onMouseMove event handler during editing interaction
SLbool SLTransformNode::onMouseMove(const SLMouseButton button,
                                    SLint               x,
                                    SLint               y,
                                    const SLKey         mod)
{
    bool result = false;

    if (_editMode)
    {
        if (_sv->camera())
        {
            switch (_editMode)
            {
                case NodeEditMode_Translate:
                {
                    if (_targetNode)
                    {
                        SLRay pickRay(_sv);
                        _sv->camera()->eyeToPixelRay((SLfloat)x, (SLfloat)y, &pickRay);

                        SLVec3f pickRayPoint;
                        SLVec3f axisPoint;
                        if (_mouseIsDown)
                        {
                            float t1, t2;
                            if (getClosestPointsBetweenRays(pickRay.origin,
                                                            pickRay.dir,
                                                            _selectedGizmo->translationWS(),
                                                            _selectedGizmo->forwardWS(),
                                                            pickRayPoint,
                                                            t1,
                                                            axisPoint,
                                                            t2))
                            {
                                SLVec3f translationDiff = axisPoint - _hitCoordinate;
                                _targetNode->translate(translationDiff, TS_world);
                                _gizmosNode->translation(_targetNode->updateAndGetWM().translation());
                                _hitCoordinate = axisPoint;
                            }

                            result = true;
                        }
                        else
                        {
                            _selectedGizmo = nullptr;

                            _transLineX->drawBits()->set(SL_DB_HIDDEN, true);
                            _transLineY->drawBits()->set(SL_DB_HIDDEN, true);
                            _transLineZ->drawBits()->set(SL_DB_HIDDEN, true);

                            float nodeToCameraDist = (pickRay.origin - _transLineX->translationWS()).length();

                            float   dist = FLT_MAX;
                            SLVec3f axisPointCand;
                            float   t1, t2;
                            float   minDistToOrigin = std::min(nodeToCameraDist * 0.1f, 1.0f);
                            if (getClosestPointsBetweenRays(pickRay.origin,
                                                            pickRay.dir,
                                                            _transLineX->translationWS(),
                                                            _transLineX->forwardWS(),
                                                            pickRayPoint,
                                                            t1,
                                                            axisPointCand,
                                                            t2))
                            {
                                float distCand = (axisPointCand - pickRayPoint).length();

                                if (distCand < dist && distCand < (_gizmoScale * 0.1f) && t1 > minDistToOrigin)
                                {
                                    dist           = distCand;
                                    _selectedGizmo = _transLineX;
                                    axisPoint      = axisPointCand;
                                }
                            }

                            if (getClosestPointsBetweenRays(pickRay.origin,
                                                            pickRay.dir,
                                                            _transLineY->translationWS(),
                                                            _transLineY->forwardWS(),
                                                            pickRayPoint,
                                                            t1,
                                                            axisPointCand,
                                                            t2))
                            {
                                float distCand = (axisPointCand - pickRayPoint).length();

                                if (distCand < dist && distCand < (_gizmoScale * 0.1f) && t1 > minDistToOrigin)
                                {
                                    dist           = distCand;
                                    _selectedGizmo = _transLineY;
                                    axisPoint      = axisPointCand;
                                }
                            }

                            if (getClosestPointsBetweenRays(pickRay.origin,
                                                            pickRay.dir,
                                                            _transLineZ->translationWS(),
                                                            _transLineZ->forwardWS(),
                                                            pickRayPoint,
                                                            t1,
                                                            axisPointCand,
                                                            t2))
                            {
                                float distCand = (axisPointCand - pickRayPoint).length();

                                if (distCand < dist && distCand < (_gizmoScale * 0.1f) && t1 > minDistToOrigin)
                                {
                                    dist           = distCand;
                                    _selectedGizmo = _transLineZ;
                                    axisPoint      = axisPointCand;
                                }
                            }

                            if (_selectedGizmo)
                            {
                                // printf("Selected gizmo %s with dist %f\n", _selectedGizmo->name().c_str(), dist);
                                _selectedGizmo->drawBits()->set(SL_DB_HIDDEN, false);
                                _hitCoordinate = axisPoint;
                            }
                        }
                    }
                }
                break;

                case NodeEditMode_Scale:
                {
                    // TODO(dgj1): this behaviour is that of a billboard... introduce in SLProject?
                    lookAt(_scaleGizmos, _sv->camera());

                    SLRay pickRay(_sv);
                    _sv->camera()->eyeToPixelRay((SLfloat)x, (SLfloat)y, &pickRay);

                    if (_targetNode)
                    {
                        if (_mouseIsDown)
                        {
                            float t = FLT_MAX;
                            if (rayPlaneIntersect(pickRay.origin,
                                                  pickRay.dir,
                                                  _selectedGizmo->translationWS(),
                                                  _selectedGizmo->forwardWS(),
                                                  t))
                            {
                                SLVec3f intersectionPointWS = pickRay.origin + pickRay.dir * t;
                                SLVec3f intersectionPoint   = _selectedGizmo->updateAndGetWMI() * intersectionPointWS;

                                float newRadius   = (intersectionPoint - _selectedGizmo->translationOS()).length();
                                float oldRadius   = (_hitCoordinate - _selectedGizmo->translationOS()).length();
                                float scaleFactor = newRadius / oldRadius;

                                _targetNode->scale(scaleFactor);
                                _gizmoScale *= scaleFactor;
                                _gizmosNode->scale(scaleFactor);
                            }

                            result = true;
                        }
                        else
                        {
                            _selectedGizmo = nullptr;

                            float t = FLT_MAX;
                            if (rayDiscIntersect(pickRay.origin,
                                                 pickRay.dir,
                                                 _scaleCirc->translationWS(),
                                                 _scaleCirc->forwardWS(),
                                                 _gizmoScale,
                                                 t))
                            {
                                _selectedGizmo = _scaleCirc;

                                SLVec3f intersectionPointWS = pickRay.origin + pickRay.dir * t;
                                _hitCoordinate              = _scaleCirc->updateAndGetWMI() * intersectionPointWS;

                                _scaleDisk->drawBits()->set(SL_DB_HIDDEN, false);
                            }
                            else
                            {
                                _scaleDisk->drawBits()->set(SL_DB_HIDDEN, true);
                            }
                        }
                    }
                }
                break;

                case NodeEditMode_Rotate:
                {
                    if (_targetNode)
                    {
                        SLRay pickRay(_sv);
                        _sv->camera()->eyeToPixelRay((SLfloat)x, (SLfloat)y, &pickRay);

                        if (_mouseIsDown)
                        {
                            float t = FLT_MAX;
                            if (rayPlaneIntersect(pickRay.origin, pickRay.dir, _selectedGizmo->translationWS(), _selectedGizmo->forwardWS(), t))
                            {
                                SLVec3f intersectionPointWS = pickRay.origin + pickRay.dir * t;
                                SLVec3f intersectionPoint   = _selectedGizmo->updateAndGetWMI() * intersectionPointWS;
                                SLVec3f rotationStartVec    = (_hitCoordinate - _selectedGizmo->translationOS()).normalize();
                                SLVec3f rotationVec         = (intersectionPoint - _selectedGizmo->translationOS()).normalize();

                                float angle = RAD2DEG * acos(rotationVec * rotationStartVec);

                                if (angle > FLT_EPSILON || angle < -FLT_EPSILON)
                                {
                                    // determine if we have to rotate ccw or cw
                                    if (!isCCW(SLVec2f(_selectedGizmo->translationOS().x, _selectedGizmo->translationOS().y),
                                               SLVec2f(_hitCoordinate.x, _hitCoordinate.y),
                                               SLVec2f(intersectionPoint.x, intersectionPoint.y)))
                                    {
                                        angle = -angle;
                                    }

                                    _targetNode->rotate(angle, _selectedGizmo->forwardWS().normalize(), TS_world);

                                    _hitCoordinate = intersectionPoint;
                                }
                            }

                            result = true;
                        }
                        else
                        {
                            _rotDiskX->drawBits()->set(SL_DB_HIDDEN, true);
                            _rotDiskY->drawBits()->set(SL_DB_HIDDEN, true);
                            _rotDiskZ->drawBits()->set(SL_DB_HIDDEN, true);

                            _selectedGizmo = nullptr;

                            float t     = FLT_MAX;
                            float tCand = FLT_MAX;
                            if (rayDiscIntersect(pickRay.origin,
                                                 pickRay.dir,
                                                 _rotDiskX->translationWS(),
                                                 _rotDiskX->forwardWS(),
                                                 _gizmoScale,
                                                 tCand))
                            {
                                _selectedGizmo = _rotDiskX;
                                t              = tCand;
                            }

                            if (rayDiscIntersect(pickRay.origin,
                                                 pickRay.dir,
                                                 _rotDiskY->translationWS(),
                                                 _rotDiskY->forwardWS(),
                                                 _gizmoScale,
                                                 tCand))
                            {
                                if (tCand < t)
                                {
                                    _selectedGizmo = _rotDiskY;
                                    t              = tCand;
                                }
                            }

                            if (rayDiscIntersect(pickRay.origin,
                                                 pickRay.dir,
                                                 _rotDiskZ->translationWS(),
                                                 _rotDiskZ->forwardWS(),
                                                 _gizmoScale,
                                                 tCand))
                            {
                                if (tCand < t)
                                {
                                    _selectedGizmo = _rotDiskZ;
                                    t              = tCand;
                                }
                            }

                            if (_selectedGizmo)
                            {
                                SLVec3f intersectionPointWS = pickRay.origin + pickRay.dir * t;
                                _hitCoordinate              = _selectedGizmo->updateAndGetWMI() * intersectionPointWS;

                                _selectedGizmo->drawBits()->set(SL_DB_HIDDEN, false);
                            }
                        }
                    }
                }
                break;

                case NodeEditMode_None:
                default:
                {
                }
                break;
            }
        }
    }

    return result;
}
//-----------------------------------------------------------------------------
bool SLTransformNode::getClosestPointsBetweenRays(const SLVec3f& ray1O,
                                                  const SLVec3f& ray1Dir,
                                                  const SLVec3f& ray2O,
                                                  const SLVec3f& ray2Dir,
                                                  SLVec3f&       ray1P,
                                                  float&         t1,
                                                  SLVec3f&       ray2P,
                                                  float&         t2)
{
    bool result = false;

    // Check if lines are parallel
    SLVec3f cross;
    cross.cross(ray1Dir, ray2Dir);
    float den = cross.lengthSqr();

    // printf("den: %f, sqrt(den): %f\n", den, cross.length());

    if (den > FLT_EPSILON)
    {
        SLVec3f diffO = ray2O - ray1O;

        // clang-format off
        SLMat3f m1   = SLMat3f(diffO.x, ray2Dir.x, cross.x,
                               diffO.y, ray2Dir.y, cross.y,
                               diffO.z, ray2Dir.z, cross.z);
        float   det1 = m1.det();
        t1   = det1 / den;
        ray1P        = ray1O + (ray1Dir * t1);

        SLMat3f m2   = SLMat3f(diffO.x, ray1Dir.x, cross.x,
                               diffO.y, ray1Dir.y, cross.y,
                               diffO.z, ray1Dir.z, cross.z);
        float   det2 = m2.det();
        t2   = det2 / den;
        ray2P        = ray2O + (ray2Dir * t2);
        // clang-format on

        result = true;
    }

    return result;
}
//-----------------------------------------------------------------------------
bool SLTransformNode::getClosestPointOnAxis(const SLVec3f& pickRayO,
                                            const SLVec3f& pickRayDir,
                                            const SLVec3f& axisRayO,
                                            const SLVec3f& axisRayDir,
                                            SLVec3f&       axisPoint)
{
    bool result = false;

    // Check if lines are parallel
    SLVec3f cross;
    cross.cross(pickRayDir, axisRayDir);
    float den = cross.lengthSqr();

    if (den > FLT_EPSILON)
    {
        SLVec3f diffO = axisRayO - pickRayO;

        // clang-format off
        SLMat3f m = SLMat3f(diffO.x, pickRayDir.x, cross.x,
                            diffO.y, pickRayDir.y, cross.y,
                            diffO.z, pickRayDir.z, cross.z);
        // clang-format on

        float det = m.det();
        float t   = det / den;

        axisPoint = axisRayO + (axisRayDir * t);

        result = true;
    }

    return result;
}
//-----------------------------------------------------------------------------
bool SLTransformNode::rayDiscIntersect(const SLVec3f& rayO,
                                       const SLVec3f& rayDir,
                                       const SLVec3f& discO,
                                       const SLVec3f& discN,
                                       const float&   discR,
                                       float&         t)
{
    bool result = false;

    if (rayPlaneIntersect(rayO, rayDir, discO, discN, t))
    {
        SLVec3f intersectPoint = rayO + rayDir * t;
        SLVec3f discPointDist  = intersectPoint - discO;

        result = (discPointDist.length() <= discR);
    }

    return result;
}
//-----------------------------------------------------------------------------
bool SLTransformNode::rayPlaneIntersect(const SLVec3f& rayO,
                                        const SLVec3f& rayDir,
                                        const SLVec3f& planeO,
                                        const SLVec3f& planeN,
                                        float&         t)
{
    bool result = false;

    float den = planeN * rayDir;
    if (den > FLT_EPSILON || den < -FLT_EPSILON)
    {
        SLVec3f oDiff = planeO - rayO;
        t             = (oDiff * planeN) / den;

        result = (t >= 0);
    }

    return result;
}
//-----------------------------------------------------------------------------
// uses signed area to determine winding order
// returns true if a,b,c are wound in ccw order, false otherwise
// https://www.quora.com/What-is-the-signed-Area-of-the-triangle
bool SLTransformNode::isCCW(const SLVec2f& a,
                            const SLVec2f& b,
                            const SLVec2f& c)
{
    SLVec2f ac = a - c;
    SLVec2f bc = b - c;

    float signedArea = 0.5f * (bc.x * ac.y - ac.x * bc.y);
    bool  result     = (signedArea > 0.0f);

    return result;
}
//-----------------------------------------------------------------------------
void SLTransformNode::setDrawBitRecursive(SLuint bit, SLNode* node, bool value)
{
    node->drawBits()->set(bit, value);

    for (SLNode* child : node->children())
    {
        setDrawBitRecursive(bit, child, value);
    }
}
//-----------------------------------------------------------------------------
void SLTransformNode::lookAt(SLNode* node, SLCamera* camera)
{
    // TODO(dgj1): this is a lookat function, because the one in SLNode doesn't work
    // or maybe I don't understand how to use it
    // TODO(dgj1): this is only correct for the case that the node doesn't have a scale
    SLVec3f nodePos    = node->translationWS();
    SLVec3f nodeTarget = camera->translationWS();
    SLVec3f nodeDir    = (nodePos - nodeTarget).normalize();
    SLVec3f up         = SLVec3f(0.0f, 1.0f, 0.0f);
    SLVec3f nodeRight  = (up ^ nodeDir).normalize();
    SLVec3f nodeUp     = (nodeDir ^ nodeRight).normalize();

    SLVec3f nodeTranslation = node->om().translation();

    // clang-format off
    SLMat4f updatedOm = SLMat4f(nodeRight.x, nodeUp.x, nodeDir.x, nodeTranslation.x,
                                nodeRight.y, nodeUp.y, nodeDir.y, nodeTranslation.y,
                                nodeRight.z, nodeUp.z, nodeDir.z, nodeTranslation.z,
                                       0.0f,     0.0f,      0.0f,       1.0f);
    // clang-format on

    node->om(updatedOm);
}
//-----------------------------------------------------------------------------
