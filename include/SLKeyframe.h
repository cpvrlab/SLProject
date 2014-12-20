//#############################################################################
//  File:      SLKeyframe.h
//  Author:    Marcus Hudritsch
//  Date:      July 2014
//  Codestyle: https://github.com/cpvrlab/SLProject/wiki/Coding-Style-Guidelines
//  Copyright: 2002-2014 Marcus Hudritsch
//             This software is provide under the GNU General Public License
//             Please visit: http://opensource.org/licenses/GPL-3.0
//#############################################################################



#ifndef SLKEYFRAME_H
#define SLKEYFRAME_H

#include <stdafx.h>

class SLAnimationTrack;

//-----------------------------------------------------------------------------
// new keyframe
class SLKeyframe
{
public:
    SLKeyframe(const SLAnimationTrack* parent, SLfloat time);
    
    void        time(SLfloat t) { _time = t; }
    SLfloat     time() const { return _time; }
    SLbool      isValid() const { _isValid; }

    bool operator<(const SLKeyframe& other) const;

protected:
    const SLAnimationTrack* _parentTrack;	//!< owning animation track for this keyframe
	SLfloat		            _time;		    //!< temporal position in local time relative to the keyframes parent clip
    SLbool                  _isValid;       //!< is this keyframe in use inside its parent track
};
//-----------------------------------------------------------------------------
typedef std::vector<SLKeyframe*>  SLVKeyframe;

//-----------------------------------------------------------------------------
//! Special keyframe for node transform animations
class SLTransformKeyframe : public SLKeyframe
{
public:    
    SLTransformKeyframe(const SLAnimationTrack* parent, SLfloat time);

    void            translation(const SLVec3f& t) { _translation = t; }
    const SLVec3f&  translation() const { return _translation; }
    
    void            rotation(const SLQuat4f& r) { _rotation = r; }
    const SLQuat4f& rotation() const { return _rotation; }
    
    void            scale(const SLVec3f& s) { _scale = s; }
    const SLVec3f&  scale() const { return _scale; }

protected:
    SLVec3f     _translation;
    SLQuat4f    _rotation;
    SLVec3f     _scale;
};

//-----------------------------------------------------------------------------
/** Generic keyframe for special objects 
    @todo ...
*/
class SLNumericKeyframe : public SLKeyframe
{
public:
protected:
};
//-----------------------------------------------------------------------------
#endif

