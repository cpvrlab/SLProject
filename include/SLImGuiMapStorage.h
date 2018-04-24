//#############################################################################
//  File:      SLImGuiMapStorage.h
//  Author:    Michael Goettlicher
//  Date:      April 2018
//  Codestyle: https://github.com/cpvrlab/SLProject/wiki/Coding-Style-Guidelines
//  Copyright: Marcus Hudritsch
//             This software is provide under the GNU General Public License
//             Please visit: http://opensource.org/licenses/GPL-3.0
//#############################################################################

#ifndef SL_IMGUI_MAPSTORAGE_H
#define SL_IMGUI_MAPSTORAGE_H

#include <SLImGuiInfosDialog.h>

class SLCVMap;
class SLCVMapNode;
class SLCVKeyFrameDB;
class SLCVMapTracking;

//-----------------------------------------------------------------------------
class SLImGuiMapStorage : public SLImGuiInfosDialog
{
public:
    SLImGuiMapStorage(const string& name, SLCVMap* map, SLCVMapNode* mapNode, 
        SLCVKeyFrameDB* kfDB, SLCVMapTracking* tracking);

    void buildInfos() override;

private:
    SLCVMap* _map;
    SLCVMapNode* _mapNode;
    SLCVKeyFrameDB* _kfDB;
    SLCVMapTracking* _tracking;

    ////!currently selected combobox item
    //static const char* _currItem;
    ////!currently selected combobox index
    //static int _currN;
};

#endif //SL_IMGUI_MAPSTORAGE_H