//#############################################################################
//  File:      Depth.frag
//  Purpose:   Simple depth shader
//  Author:    Marcus Hudritsch, Michael Schertenleib
//  Date:      March 2020
//  Copyright: Marcus Hudritsch
//             This software is provide under the GNU General Public License
//             Please visit: http://opensource.org/licenses/GPL-3.0
//#############################################################################

#ifdef GL_ES
precision mediump float;
#endif

//-----------------------------------------------------------------------------
void main()
{
    gl_FragDepth = gl_FragCoord.z;
}
//-----------------------------------------------------------------------------
