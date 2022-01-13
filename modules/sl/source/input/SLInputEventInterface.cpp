#include <SLInputEventInterface.h>
#include <SLInputManager.h>

//-----------------------------------------------------------------------------
SLInputEventInterface::SLInputEventInterface(SLInputManager& inputManager)
  : _inputManager(inputManager)
{
}
//-----------------------------------------------------------------------------
/*! Global resize function that must be called whenever the OpenGL frame
changes it's size.
*/
void SLInputEventInterface::resize(int sceneViewIndex, int width, int height)
{
    SLResizeEvent* e = new SLResizeEvent;
    e->svIndex       = sceneViewIndex;
    e->width         = width;
    e->height        = height;
    _inputManager.queueEvent(e);
}
//-----------------------------------------------------------------------------
/*! Global event handler for mouse button down events.
 */
void SLInputEventInterface::mouseDown(int           sceneViewIndex,
                                      SLMouseButton button,
                                      int           xpos,
                                      int           ypos,
                                      SLKey         modifier)
{
    SLMouseEvent* e = new SLMouseEvent(SLInputEvent::MouseDown);
    e->svIndex      = sceneViewIndex;
    e->button       = button;
    e->x            = xpos;
    e->y            = ypos;
    e->modifier     = modifier;
    _inputManager.queueEvent(e);
}
//-----------------------------------------------------------------------------
/*! Global event handler for mouse move events.
 */
void SLInputEventInterface::mouseMove(int sceneViewIndex,
                                      int x,
                                      int y)
{
    SLMouseEvent* e = new SLMouseEvent(SLInputEvent::MouseMove);
    e->svIndex      = sceneViewIndex;
    e->x            = x;
    e->y            = y;
    _inputManager.queueEvent(e);
}
//-----------------------------------------------------------------------------
/*! Global event handler for mouse button up events.
 */
void SLInputEventInterface::mouseUp(int           sceneViewIndex,
                                    SLMouseButton button,
                                    int           xpos,
                                    int           ypos,
                                    SLKey         modifier)
{
    SLMouseEvent* e = new SLMouseEvent(SLInputEvent::MouseUp);
    e->svIndex      = sceneViewIndex;
    e->button       = button;
    e->x            = xpos;
    e->y            = ypos;
    e->modifier     = modifier;
    _inputManager.queueEvent(e);
}
//-----------------------------------------------------------------------------
/*! Global event handler for double click events.
 */
void SLInputEventInterface::doubleClick(int           sceneViewIndex,
                                        SLMouseButton button,
                                        int           xpos,
                                        int           ypos,
                                        SLKey         modifier)
{
    SLMouseEvent* e = new SLMouseEvent(SLInputEvent::MouseDoubleClick);
    e->svIndex      = sceneViewIndex;
    e->button       = button;
    e->x            = xpos;
    e->y            = ypos;
    e->modifier     = modifier;
    _inputManager.queueEvent(e);
}
//-----------------------------------------------------------------------------
/*! Global event handler for the two finger touch down events of touchscreen
devices.
*/
void SLInputEventInterface::touch2Down(int sceneViewIndex,
                                       int xpos1,
                                       int ypos1,
                                       int xpos2,
                                       int ypos2)
{
    SLTouchEvent* e = new SLTouchEvent(SLInputEvent::Touch2Down);
    e->svIndex      = sceneViewIndex;
    e->x1           = xpos1;
    e->y1           = ypos1;
    e->x2           = xpos2;
    e->y2           = ypos2;

    _inputManager.queueEvent(e);
}
//-----------------------------------------------------------------------------
/*! Global event handler for the two finger move events of touchscreen devices.
 */
void SLInputEventInterface::touch2Move(int sceneViewIndex,
                                       int xpos1,
                                       int ypos1,
                                       int xpos2,
                                       int ypos2)
{
    SLTouchEvent* e = new SLTouchEvent(SLInputEvent::Touch2Move);
    e->svIndex      = sceneViewIndex;
    e->x1           = xpos1;
    e->y1           = ypos1;
    e->x2           = xpos2;
    e->y2           = ypos2;
    _inputManager.queueEvent(e);
}
//-----------------------------------------------------------------------------
/*! Global event handler for the two finger touch up events of touchscreen
devices.
*/
void SLInputEventInterface::touch2Up(int sceneViewIndex,
                                     int xpos1,
                                     int ypos1,
                                     int xpos2,
                                     int ypos2)
{
    SLTouchEvent* e = new SLTouchEvent(SLInputEvent::Touch2Up);
    e->svIndex      = sceneViewIndex;
    e->x1           = xpos1;
    e->y1           = ypos1;
    e->x2           = xpos2;
    e->y2           = ypos2;
    _inputManager.queueEvent(e);
}
//-----------------------------------------------------------------------------
/*! Global event handler for mouse wheel events.
 */
void SLInputEventInterface::mouseWheel(int   sceneViewIndex,
                                       int   pos,
                                       SLKey modifier)
{
    SLMouseEvent* e = new SLMouseEvent(SLInputEvent::MouseWheel);
    e->svIndex      = sceneViewIndex;
    e->y            = pos;
    e->modifier     = modifier;
    _inputManager.queueEvent(e);
}
//-----------------------------------------------------------------------------
/*! Global event handler for keyboard key press events.
 */
void SLInputEventInterface::keyPress(int   sceneViewIndex,
                                     SLKey key,
                                     SLKey modifier)
{
    SLKeyEvent* e = new SLKeyEvent(SLInputEvent::KeyDown);
    e->svIndex    = sceneViewIndex;
    e->key        = key;
    e->modifier   = modifier;
    _inputManager.queueEvent(e);
}
//-----------------------------------------------------------------------------
/*! Global event handler for keyboard key release events.
 */
void SLInputEventInterface::keyRelease(int   sceneViewIndex,
                                       SLKey key,
                                       SLKey modifier)
{
    SLKeyEvent* e = new SLKeyEvent(SLInputEvent::KeyUp);
    e->svIndex    = sceneViewIndex;
    e->key        = key;
    e->modifier   = modifier;
    _inputManager.queueEvent(e);
}

//-----------------------------------------------------------------------------
/*! Global event handler for unicode character input.
 */
void SLInputEventInterface::charInput(int          sceneViewIndex,
                                      unsigned int character)
{
    SLCharInputEvent* e = new SLCharInputEvent();
    e->svIndex          = sceneViewIndex;
    e->character        = character;
    _inputManager.queueEvent(e);
}
//-----------------------------------------------------------------------------
/*! Global event handler to trigger a screenshot
 */
void SLInputEventInterface::scrCaptureRequest(int sceneViewIndex, std::string outputPath)
{
    SLScrCaptureRequestEvent* e = new SLScrCaptureRequestEvent();
    e->svIndex                  = sceneViewIndex;
    e->path                     = outputPath;
    _inputManager.queueEvent(e);
}
