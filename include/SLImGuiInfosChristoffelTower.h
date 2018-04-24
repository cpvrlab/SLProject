//#############################################################################
//  File:      SLImGuiInfosChristoffelTower.h
//  Author:    Michael Goettlicher
//  Date:      April 2018
//  Codestyle: https://github.com/cpvrlab/SLProject/wiki/Coding-Style-Guidelines
//  Copyright: Marcus Hudritsch
//             This software is provide under the GNU General Public License
//             Please visit: http://opensource.org/licenses/GPL-3.0
//#############################################################################

#ifndef SL_IMGUI_INFOSCHRISTOFFELTOWER_H
#define SL_IMGUI_INFOSCHRISTOFFELTOWER_H

#include <string>
#include <SLImGuiInfosDialog.h>

class SLNode;

//-----------------------------------------------------------------------------
class SLImGuiInfosChristoffelTower : public SLImGuiInfosDialog
{
public:
    SLImGuiInfosChristoffelTower(std::string name, SLNode* bern);

    void buildInfos() override;

private:
    // Scene node for Christoffel objects
    SLNode* _bern = nullptr;
    SLNode* umgeb_dach = nullptr;
    SLNode* umgeb_fass = nullptr;
    SLNode* boden = nullptr;
    SLNode* balda_stahl = nullptr;
    SLNode* balda_glas = nullptr;
    SLNode* mauer_wand = nullptr;
    SLNode* mauer_dach = nullptr;
    SLNode* mauer_turm = nullptr;
    SLNode* mauer_weg = nullptr;
    SLNode* grab_mauern = nullptr;
    SLNode* grab_brueck = nullptr;
    SLNode* grab_grass = nullptr;
    SLNode* grab_t_dach = nullptr;
    SLNode* grab_t_fahn = nullptr;
    SLNode* grab_t_stein = nullptr;
};

#endif //SL_IMGUI_INFOSCHRISTOFFELTOWER_H