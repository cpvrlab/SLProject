//#############################################################################
//  File:      SLKeyframe.h
//  Author:    Marcus Hudritsch
//  Date:      July 2014
//  Codestyle: https://github.com/cpvrlab/SLProject/wiki/Coding-Style-Guidelines
//  Copyright: Marcus Hudritsch
//             This software is provide under the GNU General Public License
//             Please visit: http://opensource.org/licenses/GPL-3.0
//#############################################################################

#ifndef SLKEYFRAME_H
#define SLKEYFRAME_H

#include <stdafx.h>

class SLAnimTrack;

//-----------------------------------------------------------------------------
//! Base class for all keyframes
class SLKeyframe
{
public:
                            SLKeyframe  (const SLAnimTrack* parent,
                                         SLfloat time);

            bool            operator<   (const SLKeyframe& other) const;

            void            time        (SLfloat t) { _time = t; }
            SLfloat         time        () const { return _time; }
            SLbool          isValid     () const { return _isValid; }

protected:
    const   SLAnimTrack*    _parentTrack;	//!< owning animation track for this keyframe
            SLfloat         _time;		    //!< temporal position in local time relative to the keyframes parent clip in seconds
            SLbool          _isValid;       //!< is this keyframe in use inside its parent track
};

//-----------------------------------------------------------------------------
//! SLTransformKeyframe is a specialized SLKeyframe for node transformations
/*!     
Keeps track of translation, rotation and scale.
*/
class SLTransformKeyframe : public SLKeyframe
{
public:    
                        SLTransformKeyframe(const SLAnimTrack* parent,
                                            SLfloat time);

    // Setters
            void        translation (const SLVec3f& t) { _translation = t; }
            void        rotation    (const SLQuat4f& r) { _rotation = r; }
            void        scale       (const SLVec3f& s) { _scale = s; }

    // Getters
    const   SLVec3f&    translation () const { return _translation; }
    const   SLQuat4f&   rotation    () const { return _rotation; }
    const   SLVec3f&    scale       () const { return _scale; }

protected:
            SLVec3f     _translation;
            SLQuat4f    _rotation;
            SLVec3f     _scale;
};
//-----------------------------------------------------------------------------
typedef std::vector<SLKeyframe*>  SLVKeyframe;
//-----------------------------------------------------------------------------

#endif

