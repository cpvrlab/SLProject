//#############################################################################
//  File:      SLImGuiInfosChristoffelTower.cpp
//  Author:    Michael Goettlicher
//  Date:      April 2018
//  Codestyle: https://github.com/cpvrlab/SLProject/wiki/Coding-Style-Guidelines
//  Copyright: Marcus Hudritsch
//             This software is provide under the GNU General Public License
//             Please visit: http://opensource.org/licenses/GPL-3.0
//#############################################################################

#include <stdafx.h>
#include <SLImGuiInfosChristoffelTower.h>

#include <imgui.h>
#include <imgui_internal.h>

#include <SLNode.h>

//-----------------------------------------------------------------------------
SLImGuiInfosChristoffelTower::SLImGuiInfosChristoffelTower(std::string name, SLNode* bern)
    : SLImGuiInfosDialog(name),
    _bern(bern)
{
    boden = _bern->findChild<SLNode>("Boden", false);
    balda_stahl = _bern->findChild<SLNode>("Baldachin-Stahl", false);
    balda_glas = _bern->findChild<SLNode>("Baldachin-Glas", false);
    umgeb_dach = _bern->findChild<SLNode>("Umgebung-Daecher", false);
    umgeb_fass = _bern->findChild<SLNode>("Umgebung-Fassaden", false);
    mauer_wand = _bern->findChild<SLNode>("Mauer-Wand", false);
    mauer_dach = _bern->findChild<SLNode>("Mauer-Dach", false);
    mauer_turm = _bern->findChild<SLNode>("Mauer-Turm", false);
    mauer_weg = _bern->findChild<SLNode>("Mauer-Weg", false);
    grab_mauern = _bern->findChild<SLNode>("Graben-Mauern", false);
    grab_brueck = _bern->findChild<SLNode>("Graben-Bruecken", false);
    grab_grass = _bern->findChild<SLNode>("Graben-Grass", false);
    grab_t_dach = _bern->findChild<SLNode>("Graben-Turm-Dach", false);
    grab_t_fahn = _bern->findChild<SLNode>("Graben-Turm-Fahne", false);
    grab_t_stein = _bern->findChild<SLNode>("Graben-Turm-Stein", false);
}
//-----------------------------------------------------------------------------
void SLImGuiInfosChristoffelTower::buildInfos()
{
    SLbool umgebung = !umgeb_fass->drawBits()->get(SL_DB_HIDDEN);
    if (ImGui::Checkbox("Umgebung", &umgebung))
    {
        umgeb_fass->drawBits()->set(SL_DB_HIDDEN, !umgebung);
        umgeb_dach->drawBits()->set(SL_DB_HIDDEN, !umgebung);
    }

    SLbool bodenBool = !boden->drawBits()->get(SL_DB_HIDDEN);
    if (ImGui::Checkbox("Boden", &bodenBool))
    {
        boden->drawBits()->set(SL_DB_HIDDEN, !bodenBool);
    }

    SLbool baldachin = !balda_stahl->drawBits()->get(SL_DB_HIDDEN);
    if (ImGui::Checkbox("Baldachin", &baldachin))
    {
        balda_stahl->drawBits()->set(SL_DB_HIDDEN, !baldachin);
        balda_glas->drawBits()->set(SL_DB_HIDDEN, !baldachin);
    }

    SLbool mauer = !mauer_wand->drawBits()->get(SL_DB_HIDDEN);
    if (ImGui::Checkbox("Mauer", &mauer))
    {
        mauer_wand->drawBits()->set(SL_DB_HIDDEN, !mauer);
        mauer_dach->drawBits()->set(SL_DB_HIDDEN, !mauer);
        mauer_turm->drawBits()->set(SL_DB_HIDDEN, !mauer);
        mauer_weg->drawBits()->set(SL_DB_HIDDEN, !mauer);
    }

    SLbool graben = !grab_mauern->drawBits()->get(SL_DB_HIDDEN);
    if (ImGui::Checkbox("Graben", &graben))
    {
        grab_mauern->drawBits()->set(SL_DB_HIDDEN, !graben);
        grab_brueck->drawBits()->set(SL_DB_HIDDEN, !graben);
        grab_grass->drawBits()->set(SL_DB_HIDDEN, !graben);
        grab_t_dach->drawBits()->set(SL_DB_HIDDEN, !graben);
        grab_t_fahn->drawBits()->set(SL_DB_HIDDEN, !graben);
        grab_t_stein->drawBits()->set(SL_DB_HIDDEN, !graben);
    }
}
//-----------------------------------------------------------------------------