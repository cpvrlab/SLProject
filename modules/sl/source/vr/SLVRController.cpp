//#############################################################################
//  File:      SLVRController.cpp
//  Author:    Marino von Wattenwyl
//  Date:      August 2021
//  Codestyle: https://github.com/cpvrlab/SLProject/wiki/SLProject-Coding-Style
//  Copyright: Marcus Hudritsch
//             This software is provide under the GNU General Public License
//             Please visit: http://opensource.org/licenses/GPL-3.0
//#############################################################################

#include <vr/SLVRController.h>
#include <vr/SLVRSystem.h>

//-----------------------------------------------------------------------------
SLVRController::SLVRController(SLVRTrackedDeviceIndex index)
  : SLVRTrackedDevice(index)
{
    _state.ulButtonPressed = 0;
    _state.ulButtonTouched = 0;

    for (int i = 0; i < vr::k_unControllerStateAxisCount; i++)
    {
        _state.rAxis[i].x = 0;
        _state.rAxis[i].y = 0;
    }
}
//-----------------------------------------------------------------------------
/*! Updates the state of the controller
 * The state carries information about the states of buttons (pressed, touched or neither)
 * and about the values of the axes
 */
void SLVRController::updateState()
{
    // Get the state of this controller and store it in the _state instance variable
    system()->GetControllerState(_index, &_state, sizeof(_state));
}
//-----------------------------------------------------------------------------
/*! Returns whether or not the specified button is pressed
 * @param button The button that will be tested
 * @return True when the button is pressed, false otherwise
 */
SLbool SLVRController::isButtonPressed(const SLVRControllerButton& button) const
{
    // Check whether the bit of this button is set in the bitfield of pressed buttons
    return (_state.ulButtonPressed & getButtonMask(button)) != 0ull;
}
//-----------------------------------------------------------------------------
/*! Returns whether or not the specified button is touched
 * @param button The button that will be tested
 * @return True when the button is touched, false otherwise
 */
SLbool SLVRController::isButtonTouched(const SLVRControllerButton& button) const
{
    // Check whether the bit of this button is set in the bitfield of touched buttons
    return (_state.ulButtonTouched & getButtonMask(button)) != 0ull;
}
//-----------------------------------------------------------------------------
/*! Returns the value of a trigger (a 1D axis) as an SLfloat
 * The value will be in the range -1.0 to 1.0
 * @param axis The axis whose value will be returned
 * @return The axis value as an SLfloat
 */
SLfloat SLVRController::getTriggerAxis(const SLVRControllerAxis& axis) const
{
    // Only return the x component of the axis since a trigger axis is one-dimensional
    return _state.rAxis[axis].x;
}
//-----------------------------------------------------------------------------
/*! Returns the value of a 2D axis SLVec2f
 * The components are in the range -1.0 to 1.0
 * @param axis The axis whose value will be returned
 * @return The axis value as an SLVec2f
 */
SLVec2f SLVRController::get2DAxis(const SLVRControllerAxis& axis) const
{
    // Return the x and y component of the vector in an SLVec2f
    return SLVec2f(_state.rAxis[axis].x, _state.rAxis[axis].y);
}
//-----------------------------------------------------------------------------
/*! Gets a button mask from a button that can be used to mask a bitfield
 * @param button The button whose mask will be returned
 * @return The button mask
 */
uint64_t SLVRController::getButtonMask(const SLVRControllerButton& button) const
{
    return vr::ButtonMaskFromId((vr::EVRButtonId)button);
}
//-----------------------------------------------------------------------------