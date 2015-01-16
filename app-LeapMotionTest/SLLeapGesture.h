//  File:      SLLeapGesture.h
//  Author:    Marc Wacker
//  Date:      January 2015
//  Codestyle: https://github.com/cpvrlab/SLProject/wiki/Coding-Style-Guidelines
//  Copyright: 2002-2015 Marcus Hudritsch
//             This software is provide under the GNU General Public License
//             Please visit: http://opensource.org/licenses/GPL-3.0
//#############################################################################

#ifndef SLLEAPGESTURE_H
#define SLLEAPGESTURE_H

#include <stdafx.h>
#include <Leap.h>


class SLLeapGesture
{
public:
    enum Type
    {
        Swipe =  Leap::Gesture::TYPE_SWIPE,
        Circle = Leap::Gesture::TYPE_CIRCLE,
        ScreenTap =  Leap::Gesture::TYPE_SCREEN_TAP,
        KeyTap =  Leap::Gesture::TYPE_KEY_TAP
    };

    SLLeapGesture(Type type)
        : _type(type)
    { }

    Type type() const { return _type; }
    void leapGesture(const Leap::Gesture& gesture) { _gesture = gesture; }

protected:
    Leap::Gesture  _gesture;
    Type _type;
};

class SLLeapCircleGesture : public SLLeapGesture
{
public:
    SLLeapCircleGesture()
        : SLLeapGesture(Circle)
    { }

    /// @todo   Add accessors to the special Leap::CircleGesture members
};

class SLLeapSwipeGesture : public SLLeapGesture
{
public:
    SLLeapSwipeGesture()
        : SLLeapGesture(Swipe)
    { }

    /// @todo   Add accessors to the special Leap::SwipeGesture members
};

class SLLeapScreenTapGesture : public SLLeapGesture
{
public:
    SLLeapScreenTapGesture()
        : SLLeapGesture(ScreenTap)
    { }
    
    /// @todo   Add accessors to the special Leap::ScreenTapGesture members
};

class SLLeapKeyTapGesture : public SLLeapGesture
{
public:
    SLLeapKeyTapGesture()
        : SLLeapGesture(KeyTap)
    { }
    
    /// @todo   Add accessors to the special Leap::KeyTapGesture members
};

#endif