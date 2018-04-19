//#############################################################################
//  File:      SLCVMapStorage.cpp
//  Author:    Michael Goettlicher
//  Date:      April 2018
//  Codestyle: https://github.com/cpvrlab/SLProject/wiki/Coding-Style-Guidelines
//  Copyright: Marcus Hudritsch, Michael Goettlicher
//             This softwareis provide under the GNU General Public License
//             Please visit: http://opensource.org/licenses/GPL-3.0
//#############################################################################

#include <stdafx.h>
#include <SLCVMapStorage.h>
#include <SLFileSystem.h>

//-----------------------------------------------------------------------------
unsigned int SLCVMapStorage::_nextId = 0;
unsigned int SLCVMapStorage::_currentId = 0;
SLstring SLCVMapStorage::_mapPrefix = "slam-map-";
SLstring SLCVMapStorage::_mapsDirName = "slam-maps";
SLstring SLCVMapStorage::_mapsDir = "";
//values used by imgui
SLVstring SLCVMapStorage::existingMapNames;
const char* SLCVMapStorage::currItem = NULL;
int SLCVMapStorage::currN = -1;
//-----------------------------------------------------------------------------
SLCVMapStorage::SLCVMapStorage( ORBVocabulary* orbVoc, bool loadKfImgs)
    : _orbVoc(orbVoc),
    _loadKfImgs(loadKfImgs)
{
}
//-----------------------------------------------------------------------------
void SLCVMapStorage::init()
{
    existingMapNames.clear();

    //setup file system and check for existing files
    if (SLFileSystem::externalDirExists())
    {
        _mapsDir = SLFileSystem::getExternalDir() + _mapsDirName;
        _mapsDir = SLUtils::unifySlashes(_mapsDir);

        //check if visual odometry maps directory exists
        if (!SLFileSystem::dirExists(_mapsDir))
        {
            SL_LOG("Making dir: %s\n", _mapsDir.c_str());
            SLFileSystem::makeDir(_mapsDir);
        }
        else
        {
            //parse content: we search for directories in mapsDir
            SLVstring content = SLUtils::getFileNamesInDir(_mapsDir);
            for (auto path : content)
            {
                SLstring name = SLUtils::getFileName(path);
                //find json files that contain mapPrefix and estimate highest used id
                if (SLUtils::contains(name, _mapPrefix))
                {
                    existingMapNames.push_back(name);
                    SL_LOG("VO-Map found: %s\n", name.c_str());
                    //estimate highest used id
                    SLVstring splitted;
                    SLUtils::split(name, '-', splitted);
                    if (splitted.size())
                    {
                        int id = atoi(splitted.back().c_str());
                        if (id >= _nextId)
                        {
                            _nextId = id + 1;
                            SL_LOG("New next id: %i\n", _nextId);
                        }
                    }
                }
            }
        }
    }
    else
    {
        SL_LOG("Failed to setup external map storage!\n");
        SL_EXIT_MSG("Exit in SLCVMapStorage::init()");
    }
}
//-----------------------------------------------------------------------------
void SLCVMapStorage::saveMap(int id, SLCVMap& map, bool saveImgs)
{
    //check if map exists
    string mapName = _mapPrefix + to_string(id);
    string path = SLUtils::unifySlashes(_mapsDir + mapName);
    string pathImgs = path + "imgs/";
    string filename = path + mapName + ".json";

    try {
        //if path exists, delete content
        if (SLFileSystem::fileExists(path))
        {
            //remove json file
            if (SLFileSystem::fileExists(filename))
                SLFileSystem::deleteFile(filename);
            //check if imgs dir exists and delete all containing files
            if (SLFileSystem::fileExists(pathImgs))
            {
                SLVstring content = SLUtils::getFileNamesInDir(pathImgs);
                for (auto path : content)
                {
                    SLFileSystem::deleteFile(path);
                }
            }
        }
        else
        {
            //create map directory and imgs directory
            SLFileSystem::makeDir(path);
            SLFileSystem::makeDir(pathImgs);
        }

        cv::FileStorage fs(filename, cv::FileStorage::WRITE);

        //save keyframes (without graph/neigbourhood information)
        auto kfs = map.GetAllKeyFrames();
        if (!kfs.size())
            return;

        //store levels and scaleFactor here and not for every keyframe
        if (kfs.size())
        {
            //scale factor
            fs << "scaleFactor" << kfs[0]->mfScaleFactor;
            //number of pyriamid scale levels
            fs << "nScaleLevels" << kfs[0]->mnScaleLevels;
            //store camera matrix
            fs << "K" << kfs[0]->mK;
        }

        //start sequence keyframes
        fs << "KeyFrames" << "[";
        for (int i = 0; i < kfs.size(); ++i)
        {
            SLCVKeyFrame* kf = kfs[i];
            if (kf->isBad())
                continue;

            fs << "{"; //new map keyFrame
                       //add id
            fs << "id" << (int)kf->mnId;

            // world w.r.t camera
            fs << "Tcw" << kf->GetPose();
            fs << "featureDescriptors" << kf->mDescriptors;
            fs << "keyPtsUndist" << kf->mvKeysUn;

            fs << "nMinX" << kf->mnMinX;
            fs << "nMinY" << kf->mnMinY;
            fs << "nMaxX" << kf->mnMaxX;
            fs << "nMaxY" << kf->mnMaxY;

            fs << "}"; //close map

            //save the original frame image for this keyframe
            if (saveImgs)
            {
                cv::Mat imgColor;
                if (saveImgs && !kf->imgGray.empty())
                {
                    std::stringstream ss; 
                    ss << pathImgs << "kf" << (int)kf->mnId << ".jpg";

                    cv::cvtColor(kf->imgGray, imgColor, cv::COLOR_GRAY2BGR);
                    cv::imwrite(ss.str(), imgColor);

                    //if this kf was never loaded, we still have to set the texture path
                    kf->setTexturePath(ss.str());
                }
            }
        }
        fs << "]"; //close sequence keyframes

        auto mpts = map.GetAllMapPoints();
        //start map points sequence
        fs << "MapPoints" << "[";
        for (int i = 0; i < mpts.size(); ++i)
        {
            SLCVMapPoint* mpt = mpts[i];
            if (mpt->isBad())
                continue;

            fs << "{"; //new map for MapPoint
                       //add id
            fs << "id" << (int)mpt->mnId;
            //add position
            fs << "mWorldPos" << mpt->GetWorldPos();
            //save keyframe observations
            auto observations = mpt->GetObservations();
            vector<int> observingKfIds;
            vector<int> corrKpIndices; //corresponding keypoint indices in observing keyframe
            for (auto it : observations)
            {
                if (!it.first->isBad()) {
                    observingKfIds.push_back(it.first->mnId);
                    corrKpIndices.push_back(it.second);
                }
            }
            fs << "observingKfIds" << observingKfIds;
            fs << "corrKpIndices" << corrKpIndices;
            //(we calculate mean descriptor and mean deviation after loading)

            //reference key frame (I think this is the keyframe from which this
            //map point was generated -> first reference?)
            fs << "refKfId" << (int)mpt->refKf()->mnId;

            fs << "}"; //close map
        }
        fs << "]";

        // explicit close
        fs.release();
        SL_LOG("Slam map storage successful.");

        //update list of existing maps
        SLCVMapStorage::init();
        //update current combobox item
        auto it = std::find(existingMapNames.begin(), existingMapNames.end(), mapName);
        if (it != existingMapNames.end()) {
            currN = it - existingMapNames.begin();
            currItem = existingMapNames[currN].c_str();
        }
    }
    catch (std::exception& e)
    {
        string msg = "Exception during slam map storage: " + filename + "\n" +
            e.what() + "\n";
        SL_WARN_MSG(msg.c_str());
    }
    catch (...)
    {
        string msg = "Exception during slam map storage: " + filename + "\n";
        SL_WARN_MSG(msg.c_str());
    }
}
//-----------------------------------------------------------------------------
void SLCVMapStorage::loadMap(const string& name, SLCVMap& map, SLCVKeyFrameDB& kfDB)
{
    clear();
    map.clear();
    kfDB.clear();

    //extract id from map name
    SLVstring splitted;
    int id = -1;
    SLUtils::split(name, '-', splitted);
    if (splitted.size())
        id = atoi(splitted.back().c_str());
    else {
        SL_LOG("Could not load map. Map id not found in name: %s\n", name.c_str());
    }

    //check if map exists
    string mapName = _mapPrefix + to_string(id);
    string path = SLUtils::unifySlashes(_mapsDir + mapName);
    _currPathImgs = path + "imgs/";
    string filename = path + mapName + ".json";

    //check if dir and file exist
    if (!SLFileSystem::dirExists(path)) {
        string msg = "Failed to load map. Path does not exist: " + path + "\n";
        SL_WARN_MSG(msg.c_str());
        return;
    }
    if (!SLFileSystem::fileExists(filename)) {
        string msg = "Failed to load map: " + filename + "\n";
        SL_WARN_MSG(msg.c_str());
        return;
    }

    try {
        _fs.open(filename, cv::FileStorage::READ);
        if (!_fs.isOpened()) {
            string msg = "Failed to open filestorage: " + filename + "\n";
            SL_WARN_MSG(msg.c_str());
            return;
        }

        //load keyframes
        loadKeyFrames(map, kfDB);
        //load map points
        loadMapPoints(map);

        //update the covisibility graph, when all keyframes and mappoints are loaded
        auto kfs = map.GetAllKeyFrames();
        for (auto kf : kfs)
        {
            // Update links in the Covisibility Graph
            kf->UpdateConnections();
        }

        //compute resulting values for map points
        auto mapPts = map.GetAllMapPoints();
        for (auto& mp : mapPts) {
            //mean viewing direction and depth
            mp->UpdateNormalAndDepth();
            mp->ComputeDistinctiveDescriptors();
        }

        SL_LOG("Slam map loading successful.");
    }
    catch (std::exception& e)
    {
        string msg = "Exception during slam map loading: " + filename + 
            e.what() + "\n";
        SL_WARN_MSG(msg.c_str());
    }
    catch (...)
    {
        string msg = "Exception during slam map loading: " + filename + "\n";
        SL_WARN_MSG(msg.c_str());
    }
}
//-----------------------------------------------------------------------------
void SLCVMapStorage::loadKeyFrames( SLCVMap& map, SLCVKeyFrameDB& kfDB)
{
    //calibration information
    //load camera matrix
    cv::Mat K;
    _fs["K"] >> K;
    float fx, fy, cx, cy;
    fx = K.at<float>(0, 0);
    fy = K.at<float>(1, 1);
    cx = K.at<float>(0, 2);
    cy = K.at<float>(1, 2);

    //ORB extractor information
    float scaleFactor;
    _fs["scaleFactor"] >> scaleFactor;
    //number of pyriamid scale levels
    int nScaleLevels = -1;
    _fs["nScaleLevels"] >> nScaleLevels;
    //calculation of scaleFactors , levelsigma2, invScaleFactors and invLevelSigma2
    calculateScaleFactors(scaleFactor, nScaleLevels);

    cv::FileNode n = _fs["KeyFrames"];
    if (n.type() != cv::FileNode::SEQ)
    {
        cerr << "strings is not a sequence! FAIL" << endl;
    }

    //mapping of keyframe pointer by their id (used during map points loading)
    _kfsMap.clear();

    //reserve space in kfs
    //kfs.reserve(n.size());
    bool first = true;
    for (auto it = n.begin(); it != n.end(); ++it)
    {
        first = false;

        int id = (*it)["id"];

        // Infos about the pose: https://github.com/raulmur/ORB_SLAM2/issues/249
        // world w.r.t. camera pose -> wTc
        cv::Mat Tcw; //has to be here!
        (*it)["Tcw"] >> Tcw;

        ////get inverse
        //cv::Mat Twc = Tcw.inv();
        //Twc.rowRange(0, 3).col(3) += _t;
        //Twc = _rot * Twc;
        //Tcw = Twc.inv();
        ////apply scale
        //Tcw.rowRange(0, 3).col(3) *= _s; //scheint gut aber bei s=200 irgendwie komisch

        cv::Mat featureDescriptors; //has to be here!
        (*it)["featureDescriptors"] >> featureDescriptors;

        //load undistorted keypoints in frame
        //todo: braucht man diese wirklich oder kann man das umgehen, indem zus�tzliche daten im MapPoint abgelegt werden (z.B. octave/level siehe UpdateNormalAndDepth)
        std::vector<cv::KeyPoint> keyPtsUndist;
        (*it)["keyPtsUndist"] >> keyPtsUndist;

        //image bounds
        float nMinX, nMinY, nMaxX, nMaxY;
        (*it)["nMinX"] >> nMinX;
        (*it)["nMinY"] >> nMinY;
        (*it)["nMaxX"] >> nMaxX;
        (*it)["nMaxY"] >> nMaxY;

        //SLCVKeyFrame* newKf = new SLCVKeyFrame(keyPtsUndist.size());
        SLCVKeyFrame* newKf = new SLCVKeyFrame(Tcw, id, fx, fy, cx, cy, keyPtsUndist.size(),
            keyPtsUndist, featureDescriptors, _orbVoc, nScaleLevels, scaleFactor, _vScaleFactor,
            _vLevelSigma2, _vInvLevelSigma2, nMinX, nMinY, nMaxX, nMaxY, K, &kfDB, &map);

        if (_loadKfImgs)
        {
            stringstream ss;
            ss << _currPathImgs << "kf" << id << ".jpg";
            //newKf->imgGray = kfImg;
            if (SLFileSystem::fileExists(ss.str()))
            {
                newKf->setTexturePath(ss.str());
                newKf->imgGray = cv::imread(ss.str());
            }
        }
        //kfs.push_back(newKf);
        map.AddKeyFrame(newKf);

        //Update keyframe database:
        //add to keyframe database
        kfDB.add(newKf);

        //pointer goes out of scope und wird invalid!!!!!!
        //map pointer by id for look-up
        _kfsMap[newKf->mnId] = newKf;
    }
}
//-----------------------------------------------------------------------------
void SLCVMapStorage::loadMapPoints(SLCVMap& map)
{
    cv::FileNode n = _fs["MapPoints"];
    if (n.type() != cv::FileNode::SEQ)
    {
        cerr << "strings is not a sequence! FAIL" << endl;
    }

    //reserve space in mapPts
    //mapPts.reserve(n.size());
    //read and add map points
    for (auto it = n.begin(); it != n.end(); ++it)
    {
        //newPt->id( (int)(*it)["id"]);
        int id = (int)(*it)["id"];

        cv::Mat mWorldPos; //has to be here!
        (*it)["mWorldPos"] >> mWorldPos;

        SLCVMapPoint* newPt = new SLCVMapPoint(id, mWorldPos, &map);
        //get observing keyframes
        vector<int> observingKfIds;
        (*it)["observingKfIds"] >> observingKfIds;
        //get corresponding keypoint indices in observing keyframe
        vector<int> corrKpIndices;
        (*it)["corrKpIndices"] >> corrKpIndices;

        map.AddMapPoint(newPt);

        //get reference keyframe id
        int refKfId = (int)(*it)["refKfId"];

        //find and add pointers of observing keyframes to map point
        {
            //SLCVMapPoint* mapPt = mapPts.back();
            SLCVMapPoint* mapPt = newPt;
            for (int i = 0; i<observingKfIds.size(); ++i)
            {
                const int kfId = observingKfIds[i];
                if (_kfsMap.find(kfId) != _kfsMap.end()) {
                    SLCVKeyFrame* kf = _kfsMap[kfId];
                    mapPt->AddObservation(kf, corrKpIndices[i]);
                    kf->AddMapPoint(mapPt, corrKpIndices[i]);
                }
                else {
                    cout << "keyframe with id " << i << " not found!";
                }
            }

            //todo: is the reference keyframe only a currently valid variable or has every keyframe a reference keyframe?? Is it necessary for tracking?
            //map reference key frame pointer
            if (_kfsMap.find(refKfId) != _kfsMap.end())
                mapPt->refKf(_kfsMap[refKfId]);
            else {
                cout << "no reference keyframe found!" << endl;
                if (observingKfIds.size()) {
                    //we use the first of the observing keyframes
                    int kfId = observingKfIds[0];
                    if (_kfsMap.find(kfId) != _kfsMap.end())
                        mapPt->refKf(_kfsMap[kfId]);
                }
                else
                    int stop = 0;
            }
        }
    }
}
//-----------------------------------------------------------------------------
//calculation of scaleFactors , levelsigma2, invScaleFactors and invLevelSigma2
void SLCVMapStorage::calculateScaleFactors(float scaleFactor, int nlevels)
{
    //(copied from ORBextractor ctor)
    _vScaleFactor.resize(nlevels);
    _vLevelSigma2.resize(nlevels);
    _vScaleFactor[0] = 1.0f;
    _vLevelSigma2[0] = 1.0f;
    for (int i = 1; i<nlevels; i++)
    {
        _vScaleFactor[i] = _vScaleFactor[i - 1] * scaleFactor;
        _vLevelSigma2[i] = _vScaleFactor[i] * _vScaleFactor[i];
    }

    _vInvScaleFactor.resize(nlevels);
    _vInvLevelSigma2.resize(nlevels);
    for (int i = 0; i<nlevels; i++)
    {
        _vInvScaleFactor[i] = 1.0f / _vScaleFactor[i];
        _vInvLevelSigma2[i] = 1.0f / _vLevelSigma2[i];
    }
}
//-----------------------------------------------------------------------------
void SLCVMapStorage::clear()
{
    _kfsMap.clear();
    //vectors for precalculation of scalefactors
    _vScaleFactor.clear();
    _vInvScaleFactor.clear();
    _vLevelSigma2.clear();
    _vInvLevelSigma2.clear();
}
//-----------------------------------------------------------------------------
void SLCVMapStorage::newMap()
{
    //assign next id to current id. The nextId will be increased after file save.
    _currentId = _nextId;
}