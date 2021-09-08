//#############################################################################
//  File:      Depth.frag
//  Purpose:   Simple depth shader
//  Date:      March 2020
//  Authors:   Marcus Hudritsch, Michael Schertenleib
//  License:   This software is provided under the GNU General Public License
//             Please visit: http://opensource.org/licenses/GPL-3.0
//#############################################################################

precision highp float;

//-----------------------------------------------------------------------------
void main(void)
{
    gl_FragDepth = gl_FragCoord.z;
}
//-----------------------------------------------------------------------------
