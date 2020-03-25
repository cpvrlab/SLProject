#include <AppWAISceneView.h>

#include <SLApplication.h>
#include <SLVec3.h>
#include <SLCoordAxisArrow.h>
#include <SLSphere.h>
#include <SLCircle.h>

#include <WAIApp.h>

WAISceneView::WAISceneView(std::queue<WAIEvent*>* eventQueue) : SLSceneView(),
                                                                _editMode(WAINodeEditMode_None),
                                                                _mouseIsDown(false)
{
}

void WAISceneView::toggleEditMode(WAINodeEditMode editMode)
{
    _editMode = editMode;

    if (_editMode)
    {
        SLScene* s = SLApplication::scene;
        if (s->root3D())
        {
            SLNode* mapNode = s->root3D()->findChild<SLNode>("map");
            if (mapNode)
            {
                if (!_editGizmos)
                {
                    SLScene* s = SLApplication::scene;

                    _editGizmos = new SLNode("Gizmos");

                    _xAxisNode = new SLNode(new SLCoordAxisArrow(SLVec4f::RED), "x-axis node");
                    _xAxisNode->rotate(-90.0f, SLVec3f(0.0f, 0.0f, 1.0f));
                    _yAxisNode = new SLNode(new SLCoordAxisArrow(SLVec4f::GREEN), "y-axis node");
                    _zAxisNode = new SLNode(new SLCoordAxisArrow(SLVec4f::BLUE), "z-axis node");
                    _zAxisNode->rotate(90.0f, SLVec3f(1.0f, 0.0f, 0.0f));

                    _scaleSphere = new SLCircle(200.0f);

                    SLMaterial* redMat = new SLMaterial(SLCol4f::RED, "Red");
                    redMat->program(new SLGLGenericProgram("ColorUniformPoint.vert", "Color.frag"));
                    redMat->program()->addUniform1f(new SLGLUniform1f(UT_const, "u_pointSize", 3.0f));
                    SLMaterial* greenMat = new SLMaterial(SLCol4f::GREEN, "Green");
                    greenMat->program(new SLGLGenericProgram("ColorUniformPoint.vert", "Color.frag"));
                    greenMat->program()->addUniform1f(new SLGLUniform1f(UT_const, "u_pointSize", 5.0f));
                    SLMaterial* blueMat = new SLMaterial(SLCol4f::BLUE, "Blue");
                    blueMat->program(new SLGLGenericProgram("ColorUniformPoint.vert", "Color.frag"));
                    blueMat->program()->addUniform1f(new SLGLUniform1f(UT_const, "u_pointSize", 4.0f));

                    _rotationCircleMeshX = new SLCircleMesh("Rotation Circle X", redMat);
                    _rotationCircleMeshY = new SLCircleMesh("Rotation Circle Y", greenMat);
                    _rotationCircleMeshZ = new SLCircleMesh("Rotation Circle Z", blueMat);

                    _rotationCircleX = new SLNode(_rotationCircleMeshX, "Rotation Circle X");
                    _rotationCircleY = new SLNode(_rotationCircleMeshY, "Rotation Circle Y");
                    _rotationCircleZ = new SLNode(_rotationCircleMeshZ, "Rotation Circle Z");

                    _rotationCircleX->translation(mapNode->translationWS(), TS_world);
                    _rotationCircleY->translation(mapNode->translationWS(), TS_world);
                    _rotationCircleZ->translation(mapNode->translationWS(), TS_world);

                    float scaleFactor     = mapNode->aabb()->radiusOS() * 0.5f;
                    _rotationCircleRadius = scaleFactor;
                    _rotationCircleX->scale(scaleFactor);
                    _rotationCircleY->scale(scaleFactor);
                    _rotationCircleZ->scale(scaleFactor);

                    _rotationCircleNode = nullptr;

                    _rotationCircleX->rotate(90.0f, SLVec3f(0.0f, 1.0f, 0.0f));
                    _rotationCircleY->rotate(-90.0f, SLVec3f(1.0f, 0.0f, 0.0f));

                    _editGizmos->addChild(_xAxisNode);
                    _editGizmos->addChild(_yAxisNode);
                    _editGizmos->addChild(_zAxisNode);
                    _editGizmos->addChild(_rotationCircleX);
                    _editGizmos->addChild(_rotationCircleY);
                    _editGizmos->addChild(_rotationCircleZ);

                    s->root3D()->addChild(_editGizmos);

                    s->root2D()->addChild(_scaleSphere);

                    s->root3D()->updateAABBRec();
                }

                _editGizmos->translation(mapNode->updateAndGetWM().translation());

                for (SLNode* child : _editGizmos->children())
                {
                    child->drawBits()->set(SL_DB_HIDDEN, true);
                }
                _scaleSphere->drawBits()->set(SL_DB_HIDDEN, true);

                switch (_editMode)
                {
                    case WAINodeEditMode_Translate: {
                        _xAxisNode->drawBits()->set(SL_DB_HIDDEN, false);
                        _yAxisNode->drawBits()->set(SL_DB_HIDDEN, false);
                        _zAxisNode->drawBits()->set(SL_DB_HIDDEN, false);
                    }
                    break;

                    case WAINodeEditMode_Scale: {
                        _camera->setView(this, ET_center);

                        SLVec4f worldPos     = SLVec4f(mapNode->translationWS().x,
                                                   mapNode->translationWS().y,
                                                   mapNode->translationWS().z,
                                                   1.0f);
                        SLVec2f mapNodeProj  = _camera->projectWorldToNDC(worldPos);
                        SLVec2f screenOffset = SLVec2f(mapNodeProj.x * 0.5f * (float)_viewportRect.width,
                                                       mapNodeProj.y * 0.5f * (float)_viewportRect.height);
                        _scaleSphere->screenOffset(screenOffset);

                        _scaleSphere->drawBits()->set(SL_DB_HIDDEN, false);
                    }
                    break;

                    case WAINodeEditMode_Rotate: {
                        _rotationCircleX->drawBits()->set(SL_DB_HIDDEN, false);
                        _rotationCircleY->drawBits()->set(SL_DB_HIDDEN, false);
                        _rotationCircleZ->drawBits()->set(SL_DB_HIDDEN, false);
                    }
                    break;

                    case WAINodeEditMode_None:
                    default: {
                    }
                    break;
                }
            }
        }
    }
    else
    {
        _xAxisNode->drawBits()->set(SL_DB_HIDDEN, true);
        _yAxisNode->drawBits()->set(SL_DB_HIDDEN, true);
        _zAxisNode->drawBits()->set(SL_DB_HIDDEN, true);
        _rotationCircleX->drawBits()->set(SL_DB_HIDDEN, true);
        _rotationCircleY->drawBits()->set(SL_DB_HIDDEN, true);
        _rotationCircleZ->drawBits()->set(SL_DB_HIDDEN, true);
        _scaleSphere->drawBits()->set(SL_DB_HIDDEN, true);

        _editMode = WAINodeEditMode_None;
    }
}

SLbool WAISceneView::onMouseDown(SLMouseButton button, SLint scrX, SLint scrY, SLKey mod)
{
    bool result = false;

    if (!_editMode)
    {
        result = SLSceneView::onMouseDown(button, scrX, scrY, mod);
    }
    else
    {
        result = true;

        // Correct viewport offset
        // mouse corrds are top-left, viewport is bottom-left)
        SLint x = scrX - _viewportRect.x;
        SLint y = scrY - ((_scrH - _viewportRect.height) - _viewportRect.y);

        // Pass the event to imgui
        if (_gui)
        {
            _gui->onMouseDown(button, x, y);

#ifdef SL_GLES
            // Touch devices on iOS or Android have no mouse move event when the
            // finger isn't touching the screen. Therefore imgui can not detect hovering
            // over an imgui window. Without this extra frame you would have to touch
            // the display twice to open e.g. a menu.
            _gui->renderExtraFrame(s, this, x, y);
#endif
        }

        switch (_editMode)
        {
            case WAINodeEditMode_Translate: {
                bool  axisHit = false;
                SLRay pickRay(this);
                if (_camera)
                {
                    _camera->eyeToPixelRay((SLfloat)x, (SLfloat)y, &pickRay);
                    _editGizmos->hitRec(&pickRay);

                    if (pickRay.hitNode)
                    {
                        SLMat4f gizmoMat = _editGizmos->updateAndGetWM();
                        _axisRayO        = gizmoMat.translation();

                        if (pickRay.hitNode == _xAxisNode)
                        {
                            _axisRayDir = gizmoMat.axisX();

                            axisHit = true;
                        }
                        else if (pickRay.hitNode == _yAxisNode)
                        {
                            _axisRayDir = gizmoMat.axisY();

                            axisHit = true;
                        }
                        else if (pickRay.hitNode == _zAxisNode)
                        {
                            _axisRayDir = gizmoMat.axisZ();

                            axisHit = true;
                        }

                        if (axisHit)
                        {
                            SLVec3f axisPoint;
                            if (getClosestPointOnAxis(pickRay.origin, pickRay.dir, _axisRayO, _axisRayDir, axisPoint))
                            {
                                _hitCoordinate = axisPoint;

                                _mouseIsDown = true;
                            }
                        }
                    }
                }

                if (!axisHit)
                {
                    result = SLSceneView::onMouseDown(button, scrX, scrY, mod);
                }
            }
            break;

            case WAINodeEditMode_Scale: {
                _oldMouseCoords = SLVec2f(x, y);
                _mouseIsDown    = true;
            }
            break;

            case WAINodeEditMode_Rotate: {
                SLRay pickRay(this);
                _camera->eyeToPixelRay((SLfloat)x, (SLfloat)y, &pickRay);

                _rotationCircleNode = nullptr;

                float t     = FLT_MAX;
                float tCand = FLT_MAX;
                if (rayDiscIntersect(pickRay.origin, pickRay.dir, _rotationCircleX->translationWS(), _rotationCircleX->forwardWS(), _rotationCircleRadius, tCand))
                {
                    _rotationCircleNode = _rotationCircleX;
                    _rotationAxis       = SLVec3f(1.0f, 0.0f, 0.0f);
                    t                   = tCand;
                }

                if (rayDiscIntersect(pickRay.origin, pickRay.dir, _rotationCircleY->translationWS(), _rotationCircleY->forwardWS(), _rotationCircleRadius, tCand))
                {
                    if (tCand < t)
                    {
                        _rotationCircleNode = _rotationCircleY;
                        _rotationAxis       = SLVec3f(0.0f, 1.0f, 0.0f);
                        t                   = tCand;
                    }
                }

                if (rayDiscIntersect(pickRay.origin, pickRay.dir, _rotationCircleZ->translationWS(), _rotationCircleZ->forwardWS(), _rotationCircleRadius, tCand))
                {
                    if (tCand < t)
                    {
                        _rotationCircleNode = _rotationCircleZ;
                        _rotationAxis       = SLVec3f(0.0f, 0.0f, 1.0f);
                        t                   = tCand;
                    }
                }

                if (_rotationCircleNode)
                {
                    SLVec3f intersectionPointWS = pickRay.origin + pickRay.dir * t;
                    _rotationStartPoint         = _rotationCircleNode->updateAndGetWMI() * intersectionPointWS;
                    _rotationStartVec           = (_rotationStartPoint - _rotationCircleNode->translationOS()).normalize();

                    _mouseIsDown = true;
                }
                else
                {
                    result = SLSceneView::onMouseDown(button, scrX, scrY, mod);
                }
            }
            break;

            case WAINodeEditMode_None:
            default: {
            }
            break;
        }
    }

    return result;
}

SLbool WAISceneView::onMouseUp(SLMouseButton button, SLint scrX, SLint scrY, SLKey mod)
{
    bool result = false;

    if (!_editMode || !_mouseIsDown)
    {
        result = SLSceneView::onMouseUp(button, scrX, scrY, mod);
    }
    else
    {
        result = true;

        // Correct viewport offset
        // mouse corrds are top-left, viewport is bottom-left)
        SLint x = scrX - _viewportRect.x;
        SLint y = scrY - ((_scrH - _viewportRect.height) - _viewportRect.y);

        // Pass the event to imgui
        if (_gui)
        {
            _gui->onMouseUp(button, x, y);
        }

        if (_mouseIsDown)
        {
            SLScene* s       = SLApplication::scene;
            SLNode*  mapNode = s->root3D()->findChild<SLNode>("map");
            _editGizmos->translation(mapNode->updateAndGetWM().translation());

            _rotationCircleNode = nullptr;
            _mouseIsDown        = false;
        }
    }

    return result;
}

SLbool WAISceneView::onMouseMove(SLint scrX, SLint scrY)
{
    bool result = false;

    if (!_editMode || !_mouseIsDown)
    {
        result = SLSceneView::onMouseMove(scrX, scrY);
    }
    else
    {
        // Correct viewport offset
        // mouse corrds are top-left, viewport is bottom-left)
        SLint x = scrX - _viewportRect.x;
        SLint y = scrY - ((_scrH - _viewportRect.height) - _viewportRect.y);

        // Pass the event to imgui
        if (_gui)
        {
            _gui->onMouseMove(x, y);
        }

        if (_camera)
        {
            switch (_editMode)
            {
                case WAINodeEditMode_Translate: {
                    if (_mouseIsDown)
                    {
                        SLRay pickRay(this);
                        _camera->eyeToPixelRay((SLfloat)x, (SLfloat)y, &pickRay);

                        SLVec3f axisPoint;
                        if (getClosestPointOnAxis(pickRay.origin, pickRay.dir, _axisRayO, _axisRayDir, axisPoint))
                        {
                            SLVec3f translationDiff = axisPoint - _hitCoordinate;

                            SLScene* s       = SLApplication::scene;
                            SLNode*  mapNode = s->root3D()->findChild<SLNode>("map");
                            mapNode->translate(translationDiff, TS_world);

                            _editGizmos->translation(mapNode->updateAndGetWM().translation());

                            _hitCoordinate = axisPoint;
                        }
                    }
                }
                break;

                case WAINodeEditMode_Scale: {
                    if (_mouseIsDown)
                    {
                        SLScene* s = SLApplication::scene;
                        if (s->root3D())
                        {
                            SLNode* mapNode = s->root3D()->findChild<SLNode>("map");

                            if (mapNode)
                            {
                                SLVec4f worldPos     = SLVec4f(mapNode->translationWS().x,
                                                           mapNode->translationWS().y,
                                                           mapNode->translationWS().z,
                                                           1.0f);
                                SLVec2f mapNodeProj  = _camera->projectWorldToNDC(worldPos);
                                SLVec2f screenOffset = SLVec2f((mapNodeProj.x + 1.0f) * 0.5f * (float)_viewportRect.width,
                                                               (mapNodeProj.y + 1.0f) * 0.5f * (float)_viewportRect.height);

                                SLVec2f newMouseCoords = SLVec2f(x, y);

                                float oldRadius = (_oldMouseCoords - screenOffset).length();
                                float newRadius = (newMouseCoords - screenOffset).length();

                                float scaleFactor = newRadius / oldRadius;

                                mapNode->scale(scaleFactor);
                                _scaleSphere->scaleRadius(scaleFactor);
                                //_scaleSphere->scale(scaleFactor);

                                _oldMouseCoords = newMouseCoords;
                            }
                        }
                    }
                }
                break;

                case WAINodeEditMode_Rotate: {
                    SLScene* s = SLApplication::scene;
                    if (s->root3D())
                    {
                        SLNode* mapNode = s->root3D()->findChild<SLNode>("map");

                        if (mapNode)
                        {
                            if (_rotationCircleNode)
                            {
                                SLRay pickRay(this);
                                _camera->eyeToPixelRay((SLfloat)x, (SLfloat)y, &pickRay);

                                float t = FLT_MAX;
                                if (rayPlaneIntersect(pickRay.origin, pickRay.dir, _rotationCircleNode->translationWS(), _rotationCircleNode->forwardWS(), t))
                                {
                                    SLVec3f intersectionPointWS = pickRay.origin + pickRay.dir * t;
                                    SLVec3f intersectionPoint   = _rotationCircleNode->updateAndGetWMI() * intersectionPointWS;
                                    SLVec3f rotationVec         = (intersectionPoint - _rotationCircleNode->translationOS()).normalize();

                                    float angle = RAD2DEG * acos(rotationVec * _rotationStartVec);

                                    if (angle > FLT_EPSILON || angle < -FLT_EPSILON)
                                    {
                                        // determine if we have to rotate ccw or cw
                                        if (isCCW(SLVec2f(_rotationCircleNode->translationOS().x, _rotationCircleNode->translationOS().y),
                                                  SLVec2f(_rotationStartPoint.x, _rotationStartPoint.y),
                                                  SLVec2f(intersectionPoint.x, intersectionPoint.y)))
                                        {
                                            angle = -angle;
                                        }

                                        mapNode->rotate(angle, _rotationAxis, TS_world);

                                        _rotationStartPoint = intersectionPoint;
                                        _rotationStartVec   = rotationVec;
                                    }
                                }
                            }
                        }
                    }
                }
                break;

                case WAINodeEditMode_None:
                default: {
                }
                break;
            }
        }

        result = true;
    }

    return result;
}

bool WAISceneView::getClosestPointOnAxis(const SLVec3f& pickRayO,
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

        SLMat3f m = SLMat3f(diffO.x, pickRayDir.x, cross.x, diffO.y, pickRayDir.y, cross.y, diffO.z, pickRayDir.z, cross.z);

        float det = m.det();
        float t   = det / den;

        axisPoint = axisRayO + (axisRayDir * t);
        result    = true;
    }

    return result;
}

bool WAISceneView::rayDiscIntersect(const SLVec3f& rayO,
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

bool WAISceneView::rayPlaneIntersect(const SLVec3f& rayO,
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

// uses signed area to determine winding order
// returns true if a,b,c are wound in ccw order, false otherwise
// https://www.quora.com/What-is-the-signed-Area-of-the-triangle
bool WAISceneView::isCCW(SLVec2f a, SLVec2f b, SLVec2f c)
{
    SLVec2f ac = a - c;
    SLVec2f bc = b - c;

    float signedArea = 0.5f * (bc.x * ac.y - ac.x * bc.y);
    bool  result     = (signedArea > 0.0f);

    return result;
}
