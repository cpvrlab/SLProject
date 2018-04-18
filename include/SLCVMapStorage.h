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
    SLCVMapStorage(ORBVocabulary* orbVoc, bool loadKfImgs);

    //check if directory for map storage exists and read existing map names
    static void init();
    static void saveMap(SLCVMap& map, bool saveImgs);
    void loadMap(int id, SLCVMap& map, SLCVKeyFrameDB& kfDB);
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

    static unsigned int _highestId;
    static SLstring _mapPrefix;
    static SLstring _mapsDirName;
    static SLstring _mapsDir;
    static SLVstring existingMapNames;
};
//-----------------------------------------------------------------------------

#endif //SLCV_MAPSTORAGE
