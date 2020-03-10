#include <AppWAISceneView.h>

#include <SLApplication.h>
#include <SLCoordAxis.h>
#include <SLVec3.h>

#include <WAIApp.h>

WAISceneView::WAISceneView(std::queue<WAIEvent*>* eventQueue) : SLSceneView(),
                                                                _eventQueue(eventQueue),
                                                                _editMode(false),
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
            //_mapNode = s->root3D()->findChild<SLNode>("map");

            SLNode* mapNode = s->root3D()->findChild<SLNode>("map");
            if (mapNode)
            {
                if (!_editGizmos)
                {
                    _editGizmos = new SLNode("Gizmos");
                    _editGizmos->translation(mapNode->updateAndGetWM().translation());

                    SLNode* axisNode = new SLNode(new SLCoordAxis(), "axis node");
                    _editGizmos->addChild(axisNode);
                    s->root3D()->addChild(_editGizmos);

                    //_mapNode->updateAABBRec();
                    s->root3D()->updateAABBRec();
                }

                _editMode = true;
            }
        }
    }
    else
    {
        //_mapNode->deleteChild(_editGizmos);
        s->root3D()->deleteChild(_editGizmos);
        _editGizmos = nullptr;

        _editMode = false;
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
                    _mouseIsDown          = true;
                    _mouseDownCoordinates = SLVec2i(x, y);

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
                SLVec3f xRayO      = gizmoMat.translation();
                SLVec3f xRayDir    = gizmoMat.axisX();

                // Check if lines are parallel
                SLVec3f cross;
                cross.cross(pickRayDir, xRayDir);
                float den = cross.lengthSqr();

                if (den > FLT_EPSILON)
                {
                    SLVec3f diffO = xRayO - pickRayO;

                    SLMat3f mX = SLMat3f(diffO.x, pickRayDir.x, cross.x, diffO.y, pickRayDir.y, cross.y, diffO.z, pickRayDir.z, cross.z);

                    float detX = mX.det();
                    float tX   = detX / den;

                    SLVec3f xPoint = xRayO + (xRayDir * tX);

                    SLVec3f translationDiff = xPoint - _hitCoordinate;

                    WAIEventMapNodeTransform* event = new WAIEventMapNodeTransform();

                    SLVec3f translation = SLVec3f(0, 0, 0);
                    translation.x       = translationDiff.x;

                    event->translation = translation;
                    event->scale       = 1.0f;
                    event->tSpace      = TS_world;

                    _eventQueue->push(event);

                    _hitCoordinate = xPoint;
                }
            }

            _mouseDownCoordinates = SLVec2i(x, y);
        }

        result = true;
    }

    return result;
}
