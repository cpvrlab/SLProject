#include <AppWAISceneView.h>

#include <SLApplication.h>
#include <SLCoordAxisArrow.h>
#include <SLVec3.h>

#include <WAIApp.h>

WAISceneView::WAISceneView(std::queue<WAIEvent*>* eventQueue) : SLSceneView(),
                                                                _eventQueue(eventQueue),
                                                                _editMode(WAINodeEditMode_None),
                                                                _mouseIsDown(false)
{
}

void WAISceneView::toggleEditMode()
{
    SLScene* s = SLApplication::scene;

    if (!_editMode)
    {
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
                    s->root3D()->addChild(_editGizmos);

                    s->root3D()->updateAABBRec();
                }

                _editGizmos->translation(mapNode->updateAndGetWM().translation());

                _editGizmos->drawBits()->set(SL_DB_HIDDEN, false);
                for (SLNode* child : _editGizmos->children())
                {
                    child->drawBits()->set(SL_DB_HIDDEN, false);
                }

                _editMode = WAINodeEditMode_Translate;
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

        if (_editGizmos)
        {
            SLRay pickRay(this);
            if (_camera)
            {
                _camera->eyeToPixelRay((SLfloat)x, (SLfloat)y, &pickRay);
                _editGizmos->hitRec(&pickRay);
                if (pickRay.hitNode)
                {
                    if (pickRay.hitNode == _xAxisNode)
                    {
                        _editMode = WAINodeEditMode_TranslateX;
                    }
                    else if (pickRay.hitNode == _yAxisNode)
                    {
                        _editMode = WAINodeEditMode_TranslateY;
                    }
                    else if (pickRay.hitNode == _zAxisNode)
                    {
                        _editMode = WAINodeEditMode_TranslateZ;
                    }

                    _mouseIsDown = true;

                    _hitCoordinate = pickRay.origin + (pickRay.dir * pickRay.length);
                }
            }
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
                SLRay pickRay(this);
                _camera->eyeToPixelRay((SLfloat)x, (SLfloat)y, &pickRay);

                SLMat4f gizmoMat = _editGizmos->updateAndGetWM();

                // NOTE(dgj1): ray-ray-intersection according to
                // http://www.realtimerendering.com/intersections.html
                SLVec3f pickRayO   = pickRay.origin;
                SLVec3f pickRayDir = pickRay.dir;
                SLVec3f axisRayO   = gizmoMat.translation();
                SLVec3f axisRayDir;

                switch (_editMode)
                {
                    case WAINodeEditMode_TranslateX: {
                        axisRayDir = gizmoMat.axisX();
                    }
                    break;

                    case WAINodeEditMode_TranslateY: {
                        axisRayDir = gizmoMat.axisY();
                    }
                    break;

                    case WAINodeEditMode_TranslateZ: {
                        axisRayDir = gizmoMat.axisZ();
                    }
                    break;
                }

                // Check if lines are parallel
                SLVec3f cross;
                cross.cross(pickRayDir, axisRayDir);
                float den = cross.lengthSqr();

                if (den > FLT_EPSILON)
                {
                    SLVec3f diffO = axisRayO - pickRayO;

                    SLMat3f mX = SLMat3f(diffO.x, pickRayDir.x, cross.x, diffO.y, pickRayDir.y, cross.y, diffO.z, pickRayDir.z, cross.z);

                    float detX = mX.det();
                    float tX   = detX / den;

                    SLVec3f xPoint = axisRayO + (axisRayDir * tX);

                    SLVec3f translationDiff = xPoint - _hitCoordinate;

                    WAIEventMapNodeTransform* event = new WAIEventMapNodeTransform();
                    event->translation              = translationDiff;
                    event->rotation                 = SLVec3f(0, 0, 0);
                    event->scale                    = 1.0f;
                    event->tSpace                   = TS_world;

                    _eventQueue->push(event);

                    _hitCoordinate = xPoint;
                }
            }
        }

        result = true;
    }

    return result;
}
