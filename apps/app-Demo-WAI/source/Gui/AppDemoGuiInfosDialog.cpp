//#############################################################################
//  File:      AppDemoGuiInfosDialog.cpp
//  Author:    Michael Goettlicher
//  Date:      April 2018
//  Codestyle: https://github.com/cpvrlab/SLProject/wiki/Coding-Style-Guidelines
//  Copyright: Marcus Hudritsch
//             This software is provide under the GNU General Public License
//             Please visit: http://opensource.org/licenses/GPL-3.0
//#############################################################################

#include <stdafx.h>
#include <AppDemoGuiInfosDialog.h>
#include <SLGLImGui.h>

//-----------------------------------------------------------------------------
AppDemoGuiInfosDialog::AppDemoGuiInfosDialog(std::string name, bool* activator, ImFont* font)
  : _name(name),
    _activator(activator),
    _font(font)
{
}
