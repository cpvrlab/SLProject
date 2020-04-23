//#############################################################################
//  File:      SLTransformNode.cpp
//  Author:    Jan Dellsperger
//  Date:      July 2016
//  Codestyle: https://github.com/cpvrlab/SLProject/wiki/SLProject-Coding-Style
//  Copyright: Jan Dellsperger, Marcus Hudritsch
//             This software is provide under the GNU General Public License
//             Please visit: http://opensource.org/licenses/GPL-3.0
//#############################################################################

#include <SLTransformNode.h>
#include <SLVec3.h>
#include <SLCoordAxisArrow.h>
#include <SLCircle.h>
#include <SLDisk.h>

//-----------------------------------------------------------------------------
SLTransformNode::SLTransformNode(SLAssetManager* assetMgr,
                                 SLSceneView*    sv,
                                 SLNode*         targetNode)
  : SLNode("Edit Gizmos"),
    _sv(sv),
    _targetNode(targetNode),
    _editMode(NodeEditMode_None),
    _mouseIsDown(false),
    _gizmoScale(1.0f)
{
    SLMaterial* redMat = new SLMaterial(assetMgr, "Red", SLCol4f::RED);
    redMat->program(new SLGLGenericProgram(assetMgr, "ColorUniformPoint.vert", "Color.frag"));
    redMat->program()->addUniform1f(new SLGLUniform1f(UT_const, "u_pointSize", 3.0f));

    SLMaterial* redTransparentMat = new SLMaterial(assetMgr, "Red Transparent", SLCol4f::RED, SLVec4f::WHITE, 100.0f, 0.0f, 0.5f, 0.0f, new SLGLGenericProgram(assetMgr, "ColorUniformPoint.vert", "Color.frag"));
    redTransparentMat->program()->addUniform1f(new SLGLUniform1f(UT_const, "u_pointSize", 4.0f));

    SLMaterial* greenMat = new SLMaterial(assetMgr, "Green", SLCol4f::GREEN);
    greenMat->program(new SLGLGenericProgram(assetMgr, "ColorUniformPoint.vert", "Color.frag"));
    greenMat->program()->addUniform1f(new SLGLUniform1f(UT_const, "u_pointSize", 5.0f));

    SLMaterial* greenTransparentMat = new SLMaterial(assetMgr, "Green Transparent", SLCol4f::GREEN, SLVec4f::WHITE, 100.0f, 0.0f, 0.5f, 0.0f, new SLGLGenericProgram(assetMgr, "ColorUniformPoint.vert", "Color.frag"));
    greenTransparentMat->program()->addUniform1f(new SLGLUniform1f(UT_const, "u_pointSize", 4.0f));

    SLMaterial* blueMat = new SLMaterial(assetMgr, "Blue", SLCol4f::BLUE);
    blueMat->program(new SLGLGenericProgram(assetMgr, "ColorUniformPoint.vert", "Color.frag"));
    blueMat->program()->addUniform1f(new SLGLUniform1f(UT_const, "u_pointSize", 4.0f));

    SLMaterial* blueTransparentMat = new SLMaterial(assetMgr, "Blue Transparent", SLCol4f::BLUE, SLVec4f::WHITE, 100.0f, 0.0f, 0.5f, 0.0f, new SLGLGenericProgram(assetMgr, "ColorUniformPoint.vert", "Color.frag"));
    blueTransparentMat->program()->addUniform1f(new SLGLUniform1f(UT_const, "u_pointSize", 4.0f));

    SLMaterial* yellowMat = new SLMaterial(assetMgr, "Yellow", SLCol4f::YELLOW);
    yellowMat->program(new SLGLGenericProgram(assetMgr, "ColorUniformPoint.vert", "Color.frag"));
    yellowMat->program()->addUniform1f(new SLGLUniform1f(UT_const, "u_pointSize", 4.0f));

    SLMaterial* yellowTransparentMat = new SLMaterial(assetMgr, "Yellow Transparent", SLCol4f::YELLOW, SLVec4f::WHITE, 100.0f, 0.0f, 0.5f, 0.0f, new SLGLGenericProgram(assetMgr, "ColorUniformPoint.vert", "Color.frag"));
    yellowTransparentMat->program()->addUniform1f(new SLGLUniform1f(UT_const, "u_pointSize", 4.0f));

    _translationAxisX = new SLNode(new SLCoordAxisArrow(assetMgr, redTransparentMat), "x-axis node");
    _translationAxisY = new SLNode(new SLCoordAxisArrow(assetMgr, greenTransparentMat), "y-axis node");
    _translationAxisZ = new SLNode(new SLCoordAxisArrow(assetMgr, blueTransparentMat), "z-axis node");

    _translationAxisX->rotate(-90.0f, SLVec3f(0.0f, 0.0f, 1.0f));
    _translationAxisZ->rotate(90.0f, SLVec3f(1.0f, 0.0f, 0.0f));

    SLVec3f  startPoint = SLVec3f(0.0f, 0.0f, -1.0f);
    SLVec3f  endPoint   = SLVec3f(0.0f, 0.0f, 1.0f);
    SLVVec3f points;
    points.push_back(startPoint);
    points.push_back(endPoint);

    _translationLineX = new SLNode(new SLPolyline(assetMgr, points, false, "Translation Line Mesh X", redMat));
    _translationLineX->rotate(90.0f, SLVec3f(0.0f, 1.0f, 0.0f));
    _translationLineX->scale(1000.0f);

    _translationLineY = new SLNode(new SLPolyline(assetMgr, points, false, "Translation Line Mesh Y", greenMat));
    _translationLineY->rotate(-90.0f, SLVec3f(1.0f, 0.0f, 0.0f));
    _translationLineY->scale(1000.0f);

    _translationLineZ = new SLNode(new SLPolyline(assetMgr, points, false, "Translation Line Mesh Z", blueMat));
    _translationLineZ->scale(1000.0f);

    _scaleCircle = new SLNode(new SLCircle(assetMgr, "Scale Circle Mesh", yellowMat), "Scale Circle");
    _scaleDisk   = new SLNode(new SLDisk(assetMgr, 1.0f, SLVec3f::AXISZ, 36U, true, "Scale Disk", yellowTransparentMat), "Scale Disk");

    _scaleGizmos = new SLNode("Scale Gizmos");
    _scaleGizmos->addChild(_scaleCircle);
    _scaleGizmos->addChild(_scaleDisk);

    _rotationCircleX = new SLNode(new SLCircle(assetMgr, "Rotation Circle Mesh X", redMat), "Rotation Circle X");
    _rotationDiskX   = new SLNode(new SLDisk(assetMgr, 1.0f, SLVec3f::AXISZ, 36U, true, "Rotation Disk X", redTransparentMat), "Rotation Disk X");
    _rotationCircleY = new SLNode(new SLCircle(assetMgr, "Rotation Circle Mesh Y", greenMat), "Rotation Circle Y");
    _rotationDiskY   = new SLNode(new SLDisk(assetMgr, 1.0f, SLVec3f::AXISZ, 36U, true, "Rotation Disk Y", greenTransparentMat), "Rotation Disk Y");
    _rotationCircleZ = new SLNode(new SLCircle(assetMgr, "Rotation Circle Mesh Z", blueMat), "Rotation Circle Z");
    _rotationDiskZ   = new SLNode(new SLDisk(assetMgr, 1.0f, SLVec3f::AXISZ, 36U, true, "Rotation Disk Z", blueTransparentMat), "Rotation Disk Z");

    SLNode* rotationGizmosX = new SLNode("Rotation Gizmos X");
    rotationGizmosX->addChild(_rotationCircleX);
    rotationGizmosX->addChild(_rotationDiskX);

    SLNode* rotationGizmosY = new SLNode("Rotation Gizmos Y");
    rotationGizmosY->addChild(_rotationCircleY);
    rotationGizmosY->addChild(_rotationDiskY);

    SLNode* rotationGizmosZ = new SLNode("Rotation Gizmos Z");
    rotationGizmosZ->addChild(_rotationCircleZ);
    rotationGizmosZ->addChild(_rotationDiskZ);

    _selectedGizmo = nullptr;

    rotationGizmosX->rotate(90.0f, SLVec3f(0.0f, 1.0f, 0.0f));
    rotationGizmosY->rotate(-90.0f, SLVec3f(1.0f, 0.0f, 0.0f));

    SLNode* rotationGizmos = new SLNode("Rotation Gizmos");
    rotationGizmos->addChild(rotationGizmosX);
    rotationGizmos->addChild(rotationGizmosY);
    rotationGizmos->addChild(rotationGizmosZ);

    this->addChild(_translationAxisX);
    this->addChild(_translationLineX);
    this->addChild(_translationAxisY);
    this->addChild(_translationLineY);
    this->addChild(_translationAxisZ);
    this->addChild(_translationLineZ);
    this->addChild(_scaleGizmos);
    this->addChild(rotationGizmos);

    this->updateAABBRec();

    setDrawBitRecursive(SL_DB_OVERDRAW, this, true);

    _sv->s().eventHandlers().push_back(this);
}
//-----------------------------------------------------------------------------
SLTransformNode::~SLTransformNode()
{
    vector<SLEventHandler*>::iterator it = std::find(_sv->s().eventHandlers().begin(),
                                                     _sv->s().eventHandlers().end(),
                                                     this);

    if (it != _sv->s().eventHandlers().end())
    {
        _sv->s().eventHandlers().erase(it);
    }
}
//-----------------------------------------------------------------------------
void SLTransformNode::toggleEditMode(SLNodeEditMode editMode)
{
    _editMode = editMode;

    if (_editMode)
    {
        if (_targetNode)
        {
            this->translation(_targetNode->updateAndGetWM().translation());

            SLVec2f p1 = _sv->camera()->projectWorldToNDC(this->translationWS());
            SLVec2f p2 = _sv->camera()->projectWorldToNDC(this->translationWS() +
                                                          _sv->camera()->upWS().normalize());

            float actualHeight = (p1 - p2).length();
            float targetHeight = 0.2f; // % of screen that gizmos should occupy
            float scaleFactor  = targetHeight / actualHeight;

            this->scale(scaleFactor / _gizmoScale);
            _gizmoScale = scaleFactor;

            setDrawBitRecursive(SL_DB_HIDDEN, this, true);
            this->drawBits()->set(SL_DB_HIDDEN, false);

            switch (_editMode)
            {
                case NodeEditMode_Translate: {
                    _translationAxisX->drawBits()->set(SL_DB_HIDDEN, false);
                    _translationAxisY->drawBits()->set(SL_DB_HIDDEN, false);
                    _translationAxisZ->drawBits()->set(SL_DB_HIDDEN, false);
                }
                break;

                case NodeEditMode_Scale: {
                    if (_sv->camera())
                    {
                        // TODO(dgj1): this behaviour is that of a billboard... introduce in SLProject?
                        lookAt(_scaleGizmos, _sv->camera());
                    }

                    _scaleCircle->drawBits()->set(SL_DB_HIDDEN, false);
                }
                break;

                case NodeEditMode_Rotate: {
                    _rotationCircleX->drawBits()->set(SL_DB_HIDDEN, false);
                    _rotationCircleY->drawBits()->set(SL_DB_HIDDEN, false);
                    _rotationCircleZ->drawBits()->set(SL_DB_HIDDEN, false);
                }
                break;

                case NodeEditMode_None:
                default: {
                }
                break;
            }
        }
    }
    else
    {
        setDrawBitRecursive(SL_DB_HIDDEN, this, true);
        this->drawBits()->set(SL_DB_HIDDEN, false);

        _editMode = NodeEditMode_None;
    }
}
//-----------------------------------------------------------------------------
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
            this->translation(_targetNode->updateAndGetWM().translation());
        }

        _selectedGizmo = nullptr;
        _mouseIsDown   = false;
    }

    return result;
}
//-----------------------------------------------------------------------------
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
                case NodeEditMode_Translate: {
                    if (_targetNode)
                    {
                        SLRay pickRay(_sv);
                        _sv->camera()->eyeToPixelRay((SLfloat)x, (SLfloat)y, &pickRay);

                        SLVec3f pickRayPoint;
                        SLVec3f axisPoint;
                        if (_mouseIsDown)
                        {
                            if (getClosestPointsBetweenRays(pickRay.origin,
                                                            pickRay.dir,
                                                            _selectedGizmo->translationWS(),
                                                            _selectedGizmo->forwardWS(),
                                                            pickRayPoint,
                                                            axisPoint))
                            {
                                SLVec3f translationDiff = axisPoint - _hitCoordinate;

                                _targetNode->translate(translationDiff, TS_world);

                                this->translation(_targetNode->updateAndGetWM().translation());

                                _hitCoordinate = axisPoint;
                            }

                            result = true;
                        }
                        else
                        {
                            _selectedGizmo = nullptr;

                            _translationLineX->drawBits()->set(SL_DB_HIDDEN, true);
                            _translationLineY->drawBits()->set(SL_DB_HIDDEN, true);
                            _translationLineZ->drawBits()->set(SL_DB_HIDDEN, true);

                            float   dist = FLT_MAX;
                            SLVec3f axisPointCand;
                            if (getClosestPointsBetweenRays(pickRay.origin,
                                                            pickRay.dir,
                                                            _translationLineX->translationWS(),
                                                            _translationLineX->forwardWS(),
                                                            pickRayPoint,
                                                            axisPointCand))
                            {
                                float distCand = (axisPointCand - pickRayPoint).length();

                                if (distCand < dist && distCand < (_gizmoScale * 0.1f))
                                {
                                    dist           = distCand;
                                    _selectedGizmo = _translationLineX;
                                    axisPoint      = axisPointCand;
                                }
                            }

                            if (getClosestPointsBetweenRays(pickRay.origin,
                                                            pickRay.dir,
                                                            _translationLineY->translationWS(),
                                                            _translationLineY->forwardWS(),
                                                            pickRayPoint,
                                                            axisPointCand))
                            {
                                float distCand = (axisPointCand - pickRayPoint).length();

                                if (distCand < dist && distCand < (_gizmoScale * 0.1f))
                                {
                                    dist           = distCand;
                                    _selectedGizmo = _translationLineY;
                                    axisPoint      = axisPointCand;
                                }
                            }

                            if (getClosestPointsBetweenRays(pickRay.origin,
                                                            pickRay.dir,
                                                            _translationLineZ->translationWS(),
                                                            _translationLineZ->forwardWS(),
                                                            pickRayPoint,
                                                            axisPointCand))
                            {
                                float distCand = (axisPointCand - pickRayPoint).length();

                                if (distCand < dist && distCand < (_gizmoScale * 0.1f))
                                {
                                    dist           = distCand;
                                    _selectedGizmo = _translationLineZ;
                                    axisPoint      = axisPointCand;
                                }
                            }

                            if (_selectedGizmo)
                            {
                                _selectedGizmo->drawBits()->set(SL_DB_HIDDEN, false);
                                _hitCoordinate = axisPoint;
                            }
                        }
                    }
                }
                break;

                case NodeEditMode_Scale: {
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
                                this->scale(scaleFactor);
                            }

                            result = true;
                        }
                        else
                        {
                            _selectedGizmo = nullptr;

                            float t = FLT_MAX;
                            if (rayDiscIntersect(pickRay.origin,
                                                 pickRay.dir,
                                                 _scaleCircle->translationWS(),
                                                 _scaleCircle->forwardWS(),
                                                 _gizmoScale,
                                                 t))
                            {
                                _selectedGizmo = _scaleCircle;

                                SLVec3f intersectionPointWS = pickRay.origin + pickRay.dir * t;
                                _hitCoordinate              = _scaleCircle->updateAndGetWMI() * intersectionPointWS;

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

                case NodeEditMode_Rotate: {
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
                            _rotationDiskX->drawBits()->set(SL_DB_HIDDEN, true);
                            _rotationDiskY->drawBits()->set(SL_DB_HIDDEN, true);
                            _rotationDiskZ->drawBits()->set(SL_DB_HIDDEN, true);

                            _selectedGizmo = nullptr;

                            float t     = FLT_MAX;
                            float tCand = FLT_MAX;
                            if (rayDiscIntersect(pickRay.origin, pickRay.dir, _rotationDiskX->translationWS(), _rotationDiskX->forwardWS(), _gizmoScale, tCand))
                            {
                                _selectedGizmo = _rotationDiskX;
                                t              = tCand;
                            }

                            if (rayDiscIntersect(pickRay.origin, pickRay.dir, _rotationDiskY->translationWS(), _rotationDiskY->forwardWS(), _gizmoScale, tCand))
                            {
                                if (tCand < t)
                                {
                                    _selectedGizmo = _rotationDiskY;
                                    t              = tCand;
                                }
                            }

                            if (rayDiscIntersect(pickRay.origin, pickRay.dir, _rotationDiskZ->translationWS(), _rotationDiskZ->forwardWS(), _gizmoScale, tCand))
                            {
                                if (tCand < t)
                                {
                                    _selectedGizmo = _rotationDiskZ;
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
                default: {
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
                                                  SLVec3f&       ray2P)
{
    bool result = false;

    // Check if lines are parallel
    SLVec3f cross;
    cross.cross(ray1Dir, ray2Dir);
    float den = cross.lengthSqr();

    if (den > FLT_EPSILON)
    {
        SLVec3f diffO = ray2O - ray1O;

        // clang-format off
        SLMat3f m1   = SLMat3f(diffO.x, ray2Dir.x, cross.x,
                               diffO.y, ray2Dir.y, cross.y,
                               diffO.z, ray2Dir.z, cross.z);
        float   det1 = m1.det();
        float   t1   = det1 / den;
        ray1P        = ray1O + (ray1Dir * t1);

        SLMat3f m2   = SLMat3f(diffO.x, ray1Dir.x, cross.x,
                               diffO.y, ray1Dir.y, cross.y,
                               diffO.z, ray1Dir.z, cross.z);
        float   det2 = m2.det();
        float   t2   = det2 / den;
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
