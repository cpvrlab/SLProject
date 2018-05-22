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
#include <SLCVMapIO.h>
#include <SLCVMapTracking.h>

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
SLbool SLCVMapStorage::_isInitialized = false;
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
        //mark storage as initialized
        _isInitialized = true;
    }
    else
    {
        SL_LOG("Failed to setup external map storage!\n");
        SL_EXIT_MSG("Exit in SLCVMapStorage::init()");
    }
}
//-----------------------------------------------------------------------------
void SLCVMapStorage::saveMap(int id, SLCVMapTracking* mapTracking, bool saveImgs )
{
    if (!_isInitialized) {
        SL_LOG("External map storage is not initialized, you have to call init() first!\n");
        return;
    }

    if (!mapTracking->isInitialized()) {
        SL_LOG("Map storage: System is not initialized. Map saving is not possible!\n");
        return;
    }

    bool errorOccured = false;
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

        //switch to idle, so that map does not change, while we are accessing keyframes
        mapTracking->sm.requestStateIdle();
        while (!mapTracking->sm.hasStateIdle())
        {
            std::this_thread::sleep_for(std::chrono::seconds(1));
        }

        //save the map
        SLCVMapIO::save(filename, *mapTracking->getMap(), saveImgs, pathImgs);

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
        errorOccured = true;
    }
    catch (...)
    {
        string msg = "Exception during slam map storage: " + filename + "\n";
        SL_WARN_MSG(msg.c_str());
        errorOccured = true;
    }

    //if an error occured, we delete the whole directory
    if (errorOccured)
    {
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
                SLFileSystem::deleteFile(pathImgs);
            }
            SLFileSystem::deleteFile(path);
        }
    }

    //switch back to initialized state and resume tracking
    mapTracking->sm.requestResume();
}
//-----------------------------------------------------------------------------
bool SLCVMapStorage::loadMap(const string& name, SLCVMapTracking* mapTracking)
{
    bool loadingSuccessful = false;
    if (!_isInitialized) {
        SL_LOG("External map storage is not initialized, you have to call init() first!\n");
        return loadingSuccessful;
    }
    if (!mapTracking) {
        SL_LOG("Map tracking not initialized!\n");
        return loadingSuccessful;
    }
    //reset tracking (and all dependent threads/objects like Map, KeyFrameDatabase, LocalMapping, loopClosing)
    mapTracking->sm.requestStateIdle();
    while (!mapTracking->sm.hasStateIdle())
    {
        std::this_thread::sleep_for(std::chrono::seconds(1));
    }

    mapTracking->Reset();
    //clear map and keyframe database
    SLCVMap& map = *mapTracking->getMap();
    SLCVKeyFrameDB& kfDB = *mapTracking->getKfDB();

    //extract id from map name
    SLVstring splitted;
    int id = -1;
    SLUtils::split(name, '-', splitted);
    if (splitted.size()) {
        id = atoi(splitted.back().c_str());
        _currentId = id;
    }
    else {
        SL_LOG("Could not load map. Map id not found in name: %s\n", name.c_str());
        return loadingSuccessful;
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
        return loadingSuccessful;
    }
    if (!SLFileSystem::fileExists(filename)) {
        string msg = "Failed to load map: " + filename + "\n";
        SL_WARN_MSG(msg.c_str());
        return loadingSuccessful;
    }

    try {

        SLCVMapIO mapIO(filename, _orbVoc, _loadKfImgs, _currPathImgs);
        mapIO.load(map, kfDB);

        //if map loading was successful, switch to initialized
        mapTracking->setInitialized(true);
        loadingSuccessful = true;
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

    map.getSizeOf();
    mapTracking->sm.requestResume();
    return loadingSuccessful;
}
//-----------------------------------------------------------------------------
void SLCVMapStorage::newMap()
{
    if (!_isInitialized) {
        SL_LOG("External map storage is not initialized, you have to call init() first!\n");
        return;
    }

    //assign next id to current id. The nextId will be increased after file save.
    _currentId = _nextId;
}