//#############################################################################
//  File:      SLInputManager.cpp
//  Author:    Marc Wacker
//  Date:      January 2015
//  Codestyle: https://github.com/cpvrlab/SLProject/wiki/Coding-Style-Guidelines
//  Copyright: 2002-2014 Marcus Hudritsch
//             This software is provide under the GNU General Public License
//             Please visit: http://opensource.org/licenses/GPL-3.0
//#############################################################################

#include <stdafx.h>
#include <SLScene.h>
#include <SLSceneView.h>
#include <SLInputManager.h>


SLInputManager SLInputManager::_instance;

SLInputManager& SLInputManager::instance()
{
    return _instance;
}

SLInputManager::~SLInputManager()
{
    _devices.clear();
}

void SLInputManager::update()
{
    // process system events first
    processQueuedEvents();

    // process custom input devices
    for(SLint i = 0; i < _devices.size(); ++i)
        _devices[i]->poll();
}


void SLInputManager::queueEvent(const SLInputEvent* e)
{
    _systemEventQueue.push(e);
}

void SLInputManager::processQueuedEvents()
{
    SLInputEventPtrQueue& q = _systemEventQueue;
    while (!q.empty())
    {
        if (q.size() > 1)
            SL_LOG("processing %d events this frame.\n", q.size());
        const SLInputEvent* e = q.front();
        q.pop();

        SLSceneView* sv = SLScene::current->sv(e->svIndex);
        
        switch (e->type)
        {
        case SLInputEvent::SLCommand:          { const SLCommandEvent* ce = (const SLCommandEvent*)e; sv->onCommand(ce->cmd); } break;

        case SLInputEvent::MouseMove:          { const SLMouseEvent* me = (const SLMouseEvent*)e; sv->onMouseMove(me->x, me->y); } break;
        case SLInputEvent::MouseDown:          { const SLMouseEvent* me = (const SLMouseEvent*)e; sv->onMouseDown(me->button, me->x, me->y, me->modifier); } break;
        case SLInputEvent::MouseUp:            { const SLMouseEvent* me = (const SLMouseEvent*)e; sv->onMouseDown(me->button, me->x, me->y, me->modifier); } break;
        case SLInputEvent::MouseDoubleClick:   { const SLMouseEvent* me = (const SLMouseEvent*)e; sv->onMouseDown(me->button, me->x, me->y, me->modifier); } break;
        case SLInputEvent::MouseWheel:         { const SLMouseEvent* me = (const SLMouseEvent*)e; sv->onMouseWheel(me->y, me->modifier); } break;

        case SLInputEvent::Touch2Move:         { const SLTouchEvent* te = (const SLTouchEvent*)e; sv->onTouch2Move(te->x1, te->y1, te->x2, te->y2); } break;
        case SLInputEvent::Touch2Down:         { const SLTouchEvent* te = (const SLTouchEvent*)e; sv->onTouch2Down(te->x1, te->y1, te->x2, te->y2); } break;
        case SLInputEvent::Touch2Up:           { const SLTouchEvent* te = (const SLTouchEvent*)e; sv->onTouch2Up(te->x1, te->y1, te->x2, te->y2); } break;

        case SLInputEvent::KeyDown:            { const SLKeyEvent* ke = (const SLKeyEvent*)e; sv->onKeyPress(ke->key, ke->modifier); } break;
        case SLInputEvent::KeyUp:              { const SLKeyEvent* ke = (const SLKeyEvent*)e; sv->onKeyRelease(ke->key, ke->modifier); } break;

        case SLInputEvent::Resize:             { const SLResizeEvent* re = (const SLResizeEvent*)e; sv->onResize(re->width, re->height); } break;

        case SLInputEvent::DeviceRotationPYR:  { const SLRotationEvent* re = (const SLRotationEvent*)e; sv->onRotationPYR(re->x, re->y, re->z, 3.0f); } break;
        case SLInputEvent::DeviceRotationQUAT: { const SLRotationEvent* re = (const SLRotationEvent*)e; sv->onRotationQUAT(re->x, re->y, re->z, re->w); } break;
        }

        delete e;
    }
}