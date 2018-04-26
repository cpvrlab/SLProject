//#############################################################################
//  File:      SLCVMapStorage.h
//  Author:    Michael Goettlicher
//  Date:      April 2018
//  Codestyle: https://github.com/cpvrlab/SLProject/wiki/Coding-Style-Guidelines
//  Copyright: Marcus Hudritsch, Michael Goettlicher
//             This softwareis provide under the GNU General Public License
//             Please visit: http://opensource.org/licenses/GPL-3.0
//#############################################################################

#ifndef SLCV_MAPSTORAGE
#define SLCV_MAPSTORAGE

class SLCVMapTracking;

//-----------------------------------------------------------------------------
/* This class keeps track of the existing slam maps (SLCVMap) in the storage.
It can read slam maps from this storage and fill the SLCVMap with it. It also
can write SLCVMaps into this storage. You have to call init()
*/
class SLCVMapStorage
{
public:
    SLCVMapStorage(ORBVocabulary* orbVoc, bool loadKfImgs=true);

    //check if directory for map storage exists and read existing map names
    static void init();
    static void saveMap(int id, SLCVMapTracking* mapTracking, bool saveImgs);
    void loadMap(const string& name, SLCVMapTracking* mapTracking);
    //increase current id and maximum id in MapStorage
    static void newMap();

    static unsigned int getCurrentId() { return _currentId; }

    //values used by imgui:
    //!currently existing names in storage
    static SLVstring existingMapNames;
    //!currently selected combobox item
    static const char* currItem;
    //!currently selected combobox index
    static int currN;

private:
    cv::FileStorage _fs;
    ORBVocabulary* _orbVoc;

    //load keyframe images
    bool _loadKfImgs = false;
    SLstring _currPathImgs;

    static unsigned int _nextId;
    static unsigned int _currentId;
    static SLstring _mapPrefix;
    static SLstring _mapsDirName;
    static SLstring _mapsDir;
    static SLbool _isInitialized;
};
//-----------------------------------------------------------------------------

#endif //SLCV_MAPSTORAGE
