//#############################################################################
//  File:      SLLeapController.h
//  Author:    Marc Wacker
//  Date:      January 2015
//  Codestyle: https://github.com/cpvrlab/SLProject/wiki/Coding-Style-Guidelines
//  Copyright: 2002-2015 Marcus Hudritsch
//             This software is provide under the GNU General Public License
//             Please visit: http://opensource.org/licenses/GPL-3.0
//#############################################################################

#ifndef SLLEAPHAND_H
#define SLLEAPHAND_H

#include <stdafx.h>
#include <Leap.h>
#include "SLLeapFinger.h"


class SLLeapHand
{
public:
                SLLeapHand      ();

    // getters
    SLVec3f     palmPosition    () const;
    SLQuat4f    palmRotation    () const;


    SLVec3f     wristPosition   () const;
    SLVec3f     elbowPosition   () const;
    
    SLVec3f     armCenter       () const;
    SLVec3f     armDirection    () const;
    SLQuat4f    armRotation     () const;


    SLbool      isLeft          () const { return _hand.isLeft(); }

    void        leapHand        (const Leap::Hand& hand);

    // return iterator over fingers
    const SLVLeapFinger& fingers() const { return _fingers; }

    float       pinchStrength   () const;
    float       grabStrength    () const;

protected:
    Leap::Hand      _hand;      //!< Leap hand object
    SLVLeapFinger   _fingers;   //!< Vector of all SLLeapFinger objects for this hand

};

#endif
