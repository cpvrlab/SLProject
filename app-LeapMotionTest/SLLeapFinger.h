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

// temporary test function
class TestUtil {
public:
    static SLQuat4f QuaternionLookRotation(SLVec3f forward, SLVec3f up)
    {
        forward.normalize();

        SLVec3f vector = forward;
        SLVec3f vector2;
        vector2.cross(up, vector);
        vector2.normalize();
        SLVec3f vector3;
        vector3.cross(vector, vector2);
        vector3.normalize();
        float m00 = vector2.x;
        float m01 = vector2.y;
        float m02 = vector2.z;
        float m10 = vector3.x;
        float m11 = vector3.y;
        float m12 = vector3.z;
        float m20 = vector.x;
        float m21 = vector.y;
        float m22 = vector.z;
 
        float num8 = (m00 + m11) + m22;
        SLVec4f quaternion;
        if (num8 > 0.0f)
        {
            float num = (float)sqrt(num8 + 1.0f);
            quaternion.w = num * 0.5f;
            num = 0.5f / num;
            quaternion.x = (m12 - m21) * num;
            quaternion.y = (m20 - m02) * num;
            quaternion.z = (m01 - m10) * num;
        }
        else if ((m00 >= m11) && (m00 >= m22))
        {
            float num7 = (float)sqrt(((1.0f + m00) - m11) - m22);
            float num4 = 0.5f / num7;
            quaternion.x = 0.5f * num7;
            quaternion.y = (m01 + m10) * num4;
            quaternion.z = (m02 + m20) * num4;
            quaternion.w = (m12 - m21) * num4;
        }
        else if (m11 > m22)
        {
            float num6 = (float)sqrt(((1.0f + m11) - m00) - m22);
            float num3 = 0.5f / num6;
            quaternion.x = (m10+ m01) * num3;
            quaternion.y = 0.5f * num6;
            quaternion.z = (m21 + m12) * num3;
            quaternion.w = (m20 - m02) * num3;
        }
        else
        {
            float num5 = (float)sqrt(((1.0f + m22) - m00) - m11);
            float num2 = 0.5f / num5;
            quaternion.x = (m20 + m02) * num2;
            quaternion.y = (m21 + m12) * num2;
            quaternion.z = 0.5f * num5;
            quaternion.w = (m01 - m10) * num2;
        }
        return SLQuat4f(quaternion.x, quaternion.y, quaternion.z, quaternion.w);
    }
};
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