//#############################################################################
//  File:      SLImGuiInfosDialog.cpp
//  Author:    Michael Goettlicher
//  Date:      April 2018
//  Codestyle: https://github.com/cpvrlab/SLProject/wiki/Coding-Style-Guidelines
//  Copyright: Marcus Hudritsch
//             This software is provide under the GNU General Public License
//             Please visit: http://opensource.org/licenses/GPL-3.0
//#############################################################################

#include <stdafx.h>
#include <SLImGuiInfosDialog.h>

//-----------------------------------------------------------------------------
SLImGuiInfosDialog::SLImGuiInfosDialog(std::string name)
    : _name(name)
{
}
//-----------------------------------------------------------------------------
SLImGuiInfosDialog::SLImGuiInfosDialog(std::string name, SLSceneID sceneId)
    : _name(name)
{
    setActiveForSceneID(sceneId);
}
////-----------------------------------------------------------------------------
////! check, if the given dialog should be enabled for the scene with given id
//bool SLImGuiInfosDialog::getActiveForSceneID(SLSceneID sceneId)
//{
//    return (_dialogScenes.find(sceneId) != _dialogScenes.end());
//}
////-----------------------------------------------------------------------------
//void SLImGuiInfosDialog::setActiveForSceneID(SLSceneID sceneId)
//{
//    _dialogScenes.insert(sceneId);
//}