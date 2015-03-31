//#############################################################################
//  File:      SLAnimation.cpp
//  Author:    Marcus Hudritsch
//  Date:      July 2014
//  Codestyle: https://code.google.com/p/slproject/wiki/CodingStyleGuidelines
//  Copyright: Marcus Hudritsch
//             This software is provide under the GNU General Public License
//             Please visit: http://opensource.org/licenses/GPL-3.0
//#############################################################################

#include <stdafx.h>
#ifdef SL_MEMLEAKDETECT       // set in SL.h for debug config only
#include <debug_new.h>        // memory leak detector
#endif
#include <SLKeyframe.h>

//-----------------------------------------------------------------------------
/*! Constructor for default keyframes.
*/
SLKeyframe::SLKeyframe(const SLAnimTrack* parent, SLfloat time)
: _parentTrack(parent), _time(time)
{ }

//-----------------------------------------------------------------------------
/*! Comperator operator.
*/
bool SLKeyframe::operator<(const SLKeyframe& other) const
{
    return _time < other._time;
}

    
//-----------------------------------------------------------------------------
/*! Constructor for specialized transform keyframes.
*/
SLTransformKeyframe::SLTransformKeyframe(const SLAnimTrack* parent, SLfloat time)
                    : SLKeyframe(parent, time),
                    _translation(0, 0, 0),
                    _rotation(0, 0, 0, 1),
                    _scale(1, 1, 1)
{ }
//-----------------------------------------------------------------------------

