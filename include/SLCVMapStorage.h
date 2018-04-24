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

//-----------------------------------------------------------------------------
/* This class keeps track of the existing slam maps (SLCVMap) in the storage.
*/
class SLCVMapStorage
{
public:
    SLCVMapStorage(ORBVocabulary* orbVoc, bool loadKfImgs=true);

    //check if directory for map storage exists and read existing map names
    static void init();
    static void saveMap(int id, SLCVMap& map, bool saveImgs);
    void loadMap(const string& name, SLCVMap& map, SLCVKeyFrameDB& kfDB);
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
    void loadKeyFrames(SLCVMap& map, SLCVKeyFrameDB& kfDB);
    void loadMapPoints(SLCVMap& map);

    //calculation of scaleFactors , levelsigma2, invScaleFactors and invLevelSigma2
    void calculateScaleFactors(float scaleFactor, int nlevels);
    void clear();

    cv::FileStorage _fs;
    ORBVocabulary* _orbVoc;

    //load keyframe images
    bool _loadKfImgs = false;

    //mapping of keyframe pointer by their id (used during map points loading)
    map<int, SLCVKeyFrame*> _kfsMap;
    //vectors for precalculation of scalefactors
    std::vector<float> _vScaleFactor;
    std::vector<float> _vInvScaleFactor;
    std::vector<float> _vLevelSigma2;
    std::vector<float> _vInvLevelSigma2;
    SLstring _currPathImgs;

    static unsigned int _nextId;
    static unsigned int _currentId;
    static SLstring _mapPrefix;
    static SLstring _mapsDirName;
    static SLstring _mapsDir;
};
//-----------------------------------------------------------------------------

#endif //SLCV_MAPSTORAGE
