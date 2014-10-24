//#############################################################################
//  File:      SLKeyframe.h
//  Author:    Marcus Hudritsch
//  Date:      July 2014
//  Codestyle: https://code.google.com/p/slproject/wiki/CodingStyleGuidelines
//  Copyright: 2002-2014 Marcus Hudritsch
//             This software is provide under the GNU General Public License
//             Please visit: http://opensource.org/licenses/GPL-3.0
//#############################################################################

#ifndef SLKEYFRAME_H
#define SLKEYFRAME_H

#include <stdafx.h>

//-----------------------------------------------------------------------------
//! SLKeyframe implementes a keyframe for the keyframe animation in SLAnimation
/*
A keyframe is defines position (translation), rotation and scaling for a 
specific time point with in an animation. An animation interpolates between two
keyframes the position, rotation and scaling. See SLAnimation for more infos.
*/
class SLKeyframe
{
   public:        //! Constructor for zero transform keyframe
                  SLKeyframe     () :  
                                  _time(0),
                                  _position(SLVec3f::ZERO),
                                  _orientation(SLQuat4f(0,0,0,1)),
                                  _scaling(SLVec3f(1,1,1)) {}

                  //! Constructor for combined transform keyframe
                  SLKeyframe     (SLfloat time,
                                  SLVec3f position,
                                  SLfloat angleDEG,
                                  SLVec3f rotAxis,
                                  SLVec3f scale) :  
                                  _time(SL_abs(time)),
                                  _position(position),
                                  _orientation(SLQuat4f(angleDEG, rotAxis)),
                                  _scaling(scale) {}

                  //! Constructor for translation transform keyframe
                  SLKeyframe     (SLfloat time,
                                  SLVec3f position) :
                                  _time(SL_abs(time)),
                                  _position(position),
                                  _orientation(SLQuat4f(0,0,0,1)),
                                  _scaling(SLVec3f(1,1,1)) {}

                  //! Constructor for rotation transform keyframe
                  SLKeyframe     (SLfloat time,
                                  SLfloat rotAngleDEG,
                                  SLVec3f rotAxis) :  
                                  _time(SL_abs(time)),
                                  _position(SLVec3f::ZERO),
                                  _orientation(SLQuat4f(rotAngleDEG, rotAxis)),
                                  _scaling(SLVec3f(1,1,1)) {}

                  //! Constructor for rotation transform keyframe
                  SLKeyframe     (SLfloat time,
                                  SLfloat scaleX,
                                  SLfloat scaleY,
                                  SLfloat scaleZ) :  
                                  _time(SL_abs(time)),
                                  _position(SLVec3f::ZERO),
                                  _orientation(SLQuat4f(0,0,0,1)),
                                  _scaling(SLVec3f(scaleX, scaleY, scaleZ)) {}

                 ~SLKeyframe() {}

      // Setters
      void        time           (SLfloat t) {_time = t;}
      void        position       (SLVec3f pos) {_position.set(pos);}
      void        position       (SLfloat x, SLfloat y, SLfloat z)
                                 {_position.set(x, y, z);}
      void        orientation    (SLfloat angleDEG, SLfloat x, SLfloat y, SLfloat z)
                                 {_orientation.fromAngleAxis(angleDEG*SL_DEG2RAD, x,y,z);}
      void        orientation    (SLfloat angleDEG, SLVec3f axis)
                                 {_orientation.fromAngleAxis(angleDEG*SL_DEG2RAD, axis.x, axis.y, axis.z);}
      void        scaling        (SLfloat x, SLfloat y, SLfloat z)
                                 {_scaling.set(x, y, z);}
      void        scaling        (SLVec3f scale) {_scaling.set(scale);}

      // Getters
      SLfloat     time           () {return _time;}
      SLVec3f&    position       () {return _position;}
      SLQuat4f&   orientation    () {return _orientation;}
      SLVec3f&    scaling        () {return _scaling;}

   private:
      SLfloat     _time;         //! Time in seconds since last keyframe
      SLVec3f     _position;     //! position vector
      SLQuat4f    _orientation;  //! Angle-axis rotation as quaternion
      SLVec3f     _scaling;      //! Scaling factors
};
//-----------------------------------------------------------------------------
typedef std::vector<SLKeyframe>  SLVKeyframe;
//-----------------------------------------------------------------------------
#endif
