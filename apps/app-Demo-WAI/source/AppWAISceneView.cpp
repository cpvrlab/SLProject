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

void WAISceneView::toggleEditMode()
{
    if (!_editMode)
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

                    _editGizmos->addChild(_xAxisNode);
                    _editGizmos->addChild(_yAxisNode);
                    _editGizmos->addChild(_zAxisNode);

                    //_scaleSphere = new SLNode(new SLSphere(1.0f), "Scale sphere");
                    _scaleSphere = new SLCircle(200.0f);

                    //_editGizmos->addChild(_scaleSphere);

                    s->root3D()->addChild(_editGizmos);
                    s->root2D()->addChild(_scaleSphere);

                    s->root3D()->updateAABBRec();
                }

                _editGizmos->translation(mapNode->updateAndGetWM().translation());

                for (SLNode* child : _editGizmos->children())
                {
                    child->drawBits()->set(SL_DB_HIDDEN, true);
                }

                _editMode = WAINodeEditMode_Scale;
                switch (_editMode)
                {
                    case WAINodeEditMode_Translate: {
                        _xAxisNode->drawBits()->set(SL_DB_HIDDEN, false);
                        _yAxisNode->drawBits()->set(SL_DB_HIDDEN, false);
                        _zAxisNode->drawBits()->set(SL_DB_HIDDEN, false);
                    }
                    break;

                    case WAINodeEditMode_Scale: {
                        _scaleSphere->drawBits()->set(SL_DB_HIDDEN, false);
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
        _editGizmos->drawBits()->set(SL_DB_HIDDEN, true);
        for (SLNode* child : _editGizmos->children())
        {
            child->drawBits()->set(SL_DB_HIDDEN, true);
        }

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
                        }
                        else if (pickRay.hitNode == _yAxisNode)
                        {
                            _axisRayDir = gizmoMat.axisY();
                        }
                        else if (pickRay.hitNode == _zAxisNode)
                        {
                            _axisRayDir = gizmoMat.axisZ();
                        }

                        SLVec3f axisPoint;
                        if (getClosestPointOnAxis(pickRay.origin, pickRay.dir, _axisRayO, _axisRayDir, axisPoint))
                        {
                            _hitCoordinate = axisPoint;

                            _mouseIsDown = true;
                        }
                    }
                }
            }
            break;

            case WAINodeEditMode_Scale: {
                SLRay pickRay(this);
                if (_camera)
                {
                    _camera->eyeToPixelRay((SLfloat)x, (SLfloat)y, &pickRay);

                    SLMat4f sphereMat = _scaleSphere->updateAndGetWM();
                    SLVec3f sphereO   = sphereMat.translation();
                    SLVec3f rayO      = pickRay.origin;
                    SLVec3f rayDir    = pickRay.dir.normalize();

                    _sphereRayDist = raySphereDist(rayO, rayDir, sphereO);

                    _mouseIsDown = true;
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

    if (!_editMode)
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

            _mouseIsDown = false;
        }
    }

    return result;
}

SLbool WAISceneView::onMouseMove(SLint scrX, SLint scrY)
{
    bool result = false;

    if (!_editMode)
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

        if (_mouseIsDown)
        {
            if (_camera)
            {
                switch (_editMode)
                {
                    case WAINodeEditMode_Translate: {
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
                    break;

                    case WAINodeEditMode_Scale: {
                        SLRay pickRay(this);
                        if (_camera)
                        {
                            _camera->eyeToPixelRay((SLfloat)x, (SLfloat)y, &pickRay);

                            SLMat4f sphereMat = _scaleSphere->updateAndGetWM();
                            SLVec3f sphereO   = sphereMat.translation();
                            SLVec3f rayO      = pickRay.origin;
                            SLVec3f rayDir    = pickRay.dir.normalize();

                            float newSphereRayDist = raySphereDist(rayO, rayDir, sphereO);

                            float scaleFactor = newSphereRayDist / _sphereRayDist;
                            _sphereRayDist    = newSphereRayDist;

                            printf("scalefactor %f\n", scaleFactor);

                            SLScene* s       = SLApplication::scene;
                            SLNode*  mapNode = s->root3D()->findChild<SLNode>("map");
                            mapNode->scale(scaleFactor);

                            _scaleSphere->scale(scaleFactor);
                        }
                    }
                    break;

                    case WAINodeEditMode_None:
                    default: {
                    }
                    break;
                }
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

float WAISceneView::raySphereDist(const SLVec3f& rayO, const SLVec3f& rayDir, const SLVec3f& sphereO)
{
    SLVec3f diffV = sphereO - rayO;
    float   t     = rayDir * diffV;

    SLVec3f scalePoint = rayO + (rayDir * t);

    float result = (sphereO - scalePoint).length();
    return result;
}
