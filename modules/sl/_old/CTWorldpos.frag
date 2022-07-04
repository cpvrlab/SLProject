//############################################################################
//  File:      CTWorldpos.frag
//  Purpose:   GLSL vertex shader calculating world-space point used in the
//             voxelization process.
//  Date:      September 2018
//  Authors:   Stefan Thoeni
//  License:   This software is provided under the GNU General Public License
//             Please visit: http://opensource.org/licenses/GPL-3.0
//############################################################################


precision highp float;

//-----------------------------------------------------------------------------
in vec3 a_P_WS;

out vec4 color;
//-----------------------------------------------------------------------------
void main()
{
    color.rgb = a_P_WS;
}
//-----------------------------------------------------------------------------
