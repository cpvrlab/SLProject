//#############################################################################
//  File:      sl/SLEventHandler.h
//  Date:      July 2014
//  Codestyle: https://github.com/cpvrlab/SLProject/wiki/SLProject-Coding-Style
//  Authors:   Marcus Hudritsch
//  License:   This software is provided under the GNU General Public License
//             Please visit: http://opensource.org/licenses/GPL-3.0
//#############################################################################

#ifndef SLEVENTHANDLER_H
#define SLEVENTHANDLER_H

#include <SL.h>
#include <SLEnums.h>

//-----------------------------------------------------------------------------
//! Virtual Eventhandler class
/*!
SLEventHandler provides virtual methods for basic mouse and keyboard events.
The SLNode class is derived from the SLEventhandler class and therefore all
nodes can act as a eventhandler. For the moment only the camera class handles
the events and implements this way the trackball camera.
The scene instance has a pointer to the active eventhandler and forwards the
events that it gets from the user interface.
See also: SLSceneView and SLCamera classes.
*/
class SLEventHandler
{
public:
    SLEventHandler()
    {
        // todo anim
        _mouseRotationFactor = 0.1f;
        _keyboardDeltaPos    = 0.1f;
        //_dRot      = 15.0f;
    }
    virtual ~SLEventHandler() { ; }

    // Event handlers
    virtual SLbool onMouseDown(const SLMouseButton button,
                               const SLint         x,
                               const SLint         y,
                               const SLKey         mod)
    {
        (void)button;
        (void)x;
        (void)y;
        return false;
    }
    virtual SLbool onMouseUp(const SLMouseButton button,
                             const SLint         x,
                             const SLint         y,
                             const SLKey         mod)
    {
        (void)button;
        (void)x;
        (void)y;
        (void)mod;
        return false;
    }
    virtual SLbool onMouseMove(const SLMouseButton button,
                               const SLint         x,
                               const SLint         y,
                               const SLKey         mod)
    {
        (void)button;
        (void)x;
        (void)y;
        (void)mod;
        return false;
    }
    virtual SLbool onDoubleClick(const SLMouseButton button,
                                 const SLint         x,
                                 const SLint         y,
                                 const SLKey         mod)
    {
        (void)button;
        (void)x;
        (void)y;
        return false;
    }
    virtual SLbool onMouseWheel(const SLint delta, const SLKey mod)
    {
        (void)delta;
        (void)mod;
        return false;
    }
    virtual SLbool onTouch2Down(const SLint x1,
                                const SLint y1,
                                const SLint x2,
                                const SLint y2)
    {
        return false;
    }
    virtual SLbool onTouch2Move(const SLint x1,
                                const SLint y1,
                                const SLint x2,
                                const SLint y2)
    {
        return false;
    }
    virtual SLbool onTouch2Up(const SLint x1,
                              const SLint y1,
                              const SLint x2,
                              const SLint y2)
    {
        return false;
    }
    virtual SLbool onTouch3Down(const SLint x1,
                                const SLint y1)
    {
        return false;
    }
    virtual SLbool onTouch3Move(const SLint x1,
                                const SLint y1)
    {
        return false;
    }
    virtual SLbool onTouch3Up(const SLint x1,
                              const SLint y1)
    {
        return false;
    }
    virtual SLbool onKeyPress(const SLKey key,
                              const SLKey mod)
    {
        (void)key;
        (void)mod;
        return false;
    }
    virtual SLbool onKeyRelease(const SLKey key,
                                const SLKey mod)
    {
        (void)key;
        (void)mod;
        return false;
    }
    virtual SLbool onRotationPYR(const SLfloat pitchRAD,
                                 const SLfloat yawRAD,
                                 const SLfloat rollRAD)
    {
        return false;
    }

    // Setters
    void mouseRotationFactor(SLfloat rf) { _mouseRotationFactor = rf; }

    // Getters
    SLfloat mouseRotationFactor() { return _mouseRotationFactor; }

protected:
    SLfloat _mouseRotationFactor; //!< Mouse rotation sensibility
    SLfloat _keyboardDeltaPos;    //!< Delta dist. for keyboard translation
};
//-----------------------------------------------------------------------------
// STL list container of SLEventHandler pointers
typedef vector<SLEventHandler*> SLVEventHandler;
//-----------------------------------------------------------------------------
#endif
