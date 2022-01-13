//#############################################################################
//  File:      SLAnimation.cpp
//  Authors:   Marcus Hudritsch
//  Date:      July 2014
//  Codestyle: https://code.google.com/p/slproject/wiki/CodingStyleGuidelines
//  Authors:   Marcus Hudritsch
//  License:   This software is provided under the GNU General Public License
//             Please visit: http://opensource.org/licenses/GPL-3.0
//#############################################################################

#include <SLAnimKeyframe.h>

//-----------------------------------------------------------------------------
/*! Constructor for default keyframes.
 */
SLAnimKeyframe::SLAnimKeyframe(const SLAnimTrack* parent, SLfloat time)
  : _parentTrack(parent), _time(time)
{
}
//-----------------------------------------------------------------------------
/*! Comperator operator.
 */
bool SLAnimKeyframe::operator<(const SLAnimKeyframe& other) const
{
    return _time < other._time;
}
//-----------------------------------------------------------------------------
/*! Constructor for specialized transform keyframes.
 */
SLTransformKeyframe::SLTransformKeyframe(const SLAnimTrack* parent,
                                         SLfloat            time)
  : SLAnimKeyframe(parent, time),
    _translation(0, 0, 0),
    _rotation(0, 0, 0, 1),
    _scale(1, 1, 1)
{
}
//-----------------------------------------------------------------------------
