//#############################################################################
//  File:      WAIMapStorage.h
//  Author:    Michael Goettlicher
//  Date:      April 2018
//  Codestyle: https://github.com/cpvrlab/SLProject/wiki/Coding-Style-Guidelines
//  Copyright: Marcus Hudritsch, Michael Goettlicher
//             This softwareis provide under the GNU General Public License
//             Please visit: http://opensource.org/licenses/GPL-3.0
//#############################################################################

#ifndef WAI_MAPSTORAGE
#define WAI_MAPSTORAGE

#include <WAIHelper.h>
#include <OrbSlam/ORBVocabulary.h>
#include <WAIModeOrbSlam2.h>
#include <WAIMapIO.h>

#include <Utils.h>

//-----------------------------------------------------------------------------
/* This class keeps track of the existing slam maps (WAIMap) in the storage.
It can read slam maps from this storage and fill the WAIMap with it. It also
can write WAIMaps into this storage. You have to call init()
*/
class WAI_API WAIMapStorage
{
    public:
    WAIMapStorage();

    //check if directory for map storage exists and read existing map names
    static void init(std::string externalDir);
    static void saveMap(int id, WAI::ModeOrbSlam2* orbSlamMode, bool saveImgs, cv::Mat nodeOm, std::string externalDir);
    static bool loadMap(const std::string& name, WAI::ModeOrbSlam2* orbSlamMode, ORBVocabulary* orbVoc, bool loadKfImgs, cv::Mat& nodeOm);
    //increase current id and maximum id in MapStorage
    static void newMap();

    static unsigned int getCurrentId() { return _currentId; }
    static string       mapsDir();

    static string externalDir() { return _externalDir; }

    //values used by imgui:
    //!currently existing names in storage
    static std::vector<std::string> existingMapNames;
    //!currently selected combobox item
    static const char* currItem;
    //!currently selected combobox index
    static int currN;

    private:
    static unsigned int _nextId;
    static unsigned int _currentId;
    static std::string  _mapPrefix;
    static std::string  _mapsDirName;
    static std::string  _mapsDir;
    static bool         _isInitialized;
    static std::string  _externalDir;
};
//-----------------------------------------------------------------------------

#endif //WAI_MAPSTORAGE
