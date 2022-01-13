//#############################################################################
//  File:      SLAnimKeyframe.h
//  Authors:   Marcus Hudritsch
//  Date:      July 2014
//  Codestyle: https://github.com/cpvrlab/SLProject/wiki/SLProject-Coding-Style
//  Authors:   Marcus Hudritsch
//  License:   This software is provided under the GNU General Public License
//             Please visit: http://opensource.org/licenses/GPL-3.0
//#############################################################################

#ifndef SLKEYFRAME_H
#define SLKEYFRAME_H

#include <SL.h>
#include <SLQuat4.h>
#include <SLVec3.h>

class SLAnimTrack;

//-----------------------------------------------------------------------------
//! Base class for all animation keyframes
class SLAnimKeyframe
{
public:
    SLAnimKeyframe(const SLAnimTrack* parent,
                   SLfloat            time);

    bool operator<(const SLAnimKeyframe& other) const;

    void    time(SLfloat t) { _time = t; }
    SLfloat time() const { return _time; }
    SLbool  isValid() const { return _isValid; }

protected:
    const SLAnimTrack* _parentTrack; //!< owning animation track for this keyframe
    SLfloat            _time;        //!< temporal position in local time relative to the keyframes parent clip in seconds
    SLbool             _isValid;     //!< is this keyframe in use inside its parent track
};

//-----------------------------------------------------------------------------
//! SLTransformKeyframe is a specialized SLKeyframe for node transformations
/*!
Keeps track of translation, rotation and scale.
*/
class SLTransformKeyframe : public SLAnimKeyframe
{
public:
    SLTransformKeyframe(const SLAnimTrack* parent,
                        SLfloat            time);

    // Setters
    void translation(const SLVec3f& t) { _translation = t; }
    void rotation(const SLQuat4f& r) { _rotation = r; }
    void scale(const SLVec3f& s) { _scale = s; }

    // Getters
    const SLVec3f&  translation() const { return _translation; }
    const SLQuat4f& rotation() const { return _rotation; }
    const SLVec3f&  scale() const { return _scale; }

protected:
    SLVec3f  _translation;
    SLQuat4f _rotation;
    SLVec3f  _scale;
};
//-----------------------------------------------------------------------------
typedef vector<SLAnimKeyframe*> SLVKeyframe;
//-----------------------------------------------------------------------------

#endif
