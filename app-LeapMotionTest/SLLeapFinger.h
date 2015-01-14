//  File:      SLLeapController.h
//  Author:    Marc Wacker
//  Date:      January 2015
//  Codestyle: https://github.com/cpvrlab/SLProject/wiki/Coding-Style-Guidelines
//  Copyright: 2002-2015 Marcus Hudritsch
//             This software is provide under the GNU General Public License
//             Please visit: http://opensource.org/licenses/GPL-3.0
//#############################################################################

#ifndef SLLEAPFINGER_H
#define SLLEAPFINGER_H

#include <stdafx.h>
#include <Leap.h>


/*

this class will hold a set of SLLeapFingers.
+ specialized versions of SLLeapFingers like SLLeapRiggedFinger  will hold references to up to 4 finger bones

    see the unity integration for inspiration


Unity lm architecture:

HandModel	// getters for positions and rotations of hand and arm
	Hand		// lm sdk class
	FingerModel[]	// list of finger models
	HandController

FingerModel	// getters for positions and rotaitons of fingers
	Hand 		// lm sdk class
	Finger 		// lm sdk class
	HandController

// knows graphical and physical meshes for hands
// has list of HandModels
HandController 	



// for future use
FINGER_NAMES = {"Thumb", "Index", "Middle", "Ring", "Pinky"};
*/

class SLLeapFinger
{
public:
    SLLeapFinger(Leap::Finger::Type type);
    
    void        leapHand(const Leap::Hand& hand);
    
    /// @todo define this constant somewhere
    SLint       numBones() const { return 4; }
    SLint       numJoints() const { return 5; }

    SLVec3f     tipPosition() const;
    SLVec3f     jointPosition(SLint joint) const;
    SLVec3f     boneCenter(SLint boneType) const;
    SLVec3f     boneDirection(SLint boneType) const;
    SLQuat4f    boneRotation(SLint boneType) const;

protected:
    Leap::Hand          _hand;
    Leap::Finger        _finger;
    Leap::Finger::Type  _fingerType;
};

typedef vector<SLLeapFinger> SLVLeapFinger;

#endif