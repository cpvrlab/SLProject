//#############################################################################
//  File:      SLImGuiMapStorage.h
//  Author:    Michael Goettlicher
//  Date:      April 2018
//  Codestyle: https://github.com/cpvrlab/SLProject/wiki/SLProject-Coding-Style
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
    SLImGuiMapStorage(const string& name, SLCVMapTracking* tracking);

    void buildInfos() override;

private:
    SLCVMap*         _map;
    SLCVMapNode*     _mapNode;
    SLCVKeyFrameDB*  _kfDB;
    SLCVMapTracking* _tracking;
};

#endif //SL_IMGUI_MAPSTORAGE_H