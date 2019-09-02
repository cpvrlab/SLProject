//#############################################################################
//  File:      AppDemoGui.cpp
//  Purpose:   UI with the ImGUI framework fully rendered in OpenGL 3+
//  Author:    Marcus Hudritsch
//  Date:      Summer 2017
//  Codestyle: https://github.com/cpvrlab/SLProject/wiki/Coding-Style-Guidelines
//  Copyright: Marcus Hudritsch
//             This software is provide under the GNU General Public License
//             Please visit: http://opensource.org/licenses/GPL-3.0
//#############################################################################

#include <stdafx.h> // Must be the 1st include followed by  an empty line

#ifdef SL_MEMLEAKDETECT    // set in SL.h for debug config only
#    include <debug_new.h> // memory leak detector
#endif

#include <AppDemoGui.h>
#include <SLAnimPlayback.h>
#include <SLApplication.h>
#include <SLInterface.h>
#include <SLAverageTiming.h>
#include <CVCapture.h>
#include <CVImage.h>
#include <CVTrackedFeatures.h>
#include <SLGLProgram.h>
#include <SLGLShader.h>
#include <SLGLTexture.h>
#include <AppDemoGuiInfosDialog.h>
#include <AppDemoGuiTrackedMapping.h>
#include <AppDemoGuiMapStorage.h>
#include <SLImporter.h>
#include <SLInterface.h>
#include <SLMaterial.h>
#include <SLMesh.h>
#include <SLNode.h>
#include <SLScene.h>
#include <SLSceneView.h>
#include <SLGLImGui.h>
#include <imgui.h>
#include <imgui_internal.h>

map<string, AppDemoGuiInfosDialog*> AppDemoGui::_infoDialogs;

//-----------------------------------------------------------------------------
void AppDemoGui::addInfoDialog(AppDemoGuiInfosDialog* dialog)
{
    string name = string(dialog->getName());
    if (_infoDialogs.find(name) == _infoDialogs.end())
    {
        _infoDialogs[name] = dialog;
    }
}
//-----------------------------------------------------------------------------
void AppDemoGui::clearInfoDialogs()
{
    _infoDialogs.clear();
}
//-----------------------------------------------------------------------------
void AppDemoGui::buildInfosDialogs(SLScene* s, SLSceneView* sv)
{
    for (auto dialog : _infoDialogs)
    {
        if (dialog.second->show())
        {
            dialog.second->buildInfos(s, sv);
        }
    }
}
