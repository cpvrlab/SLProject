//#############################################################################
//  File:      Depth.frag
//  Purpose:   Simple depth shader
//  Author:    Marcus Hudritsch, Michael Schertenleib
//  Date:      March 2020
//  Authors:   Marcus Hudritsch
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
