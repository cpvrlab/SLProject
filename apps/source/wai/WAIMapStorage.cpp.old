//#############################################################################
//  File:      WAIMapStorage.cpp
//  Author:    Michael Goettlicher
//  Date:      April 2018
//  Codestyle: https://github.com/cpvrlab/SLProject/wiki/Coding-Style-Guidelines
//  Copyright: Marcus Hudritsch, Michael Goettlicher
//             This softwareis provide under the GNU General Public License
//             Please visit: http://opensource.org/licenses/GPL-3.0
//#############################################################################

#include <WAIMapStorage.h>

//-----------------------------------------------------------------------------
unsigned int WAIMapStorage::_nextId      = 0;
unsigned int WAIMapStorage::_currentId   = 0;
std::string  WAIMapStorage::_mapPrefix   = "slam-map-";
std::string  WAIMapStorage::_mapsDirName = "slam-maps";
std::string  WAIMapStorage::_mapsDir     = "";
std::string  WAIMapStorage::_externalDir = "";
//values used by imgui
std::vector<std::string> WAIMapStorage::existingMapNames;
const char*              WAIMapStorage::currItem       = nullptr;
int                      WAIMapStorage::currN          = -1;
bool                     WAIMapStorage::_isInitialized = false;
//-----------------------------------------------------------------------------
WAIMapStorage::WAIMapStorage()
{
}
//-----------------------------------------------------------------------------
void WAIMapStorage::init(std::string externalDir)
{
    _externalDir = externalDir;
    existingMapNames.clear();
    vector<pair<int, string>> existingMapNamesSorted;

    //setup file system and check for existing files
    if (Utils::dirExists(_externalDir))
    {
        _mapsDir = Utils::unifySlashes(externalDir + _mapsDirName);

        //check if visual odometry maps directory exists
        if (!Utils::dirExists(_mapsDir))
        {
            WAI_LOG("Making dir: %s\n", _mapsDir.c_str());
            Utils::makeDir(_mapsDir);
        }
        else
        {
            //parse content: we search for directories in mapsDir
            std::vector<std::string> content = Utils::getFileNamesInDir(_mapsDir);
            for (auto path : content)
            {
                std::string name = Utils::getFileName(path);
                //find json files that contain mapPrefix and estimate highest used id
                if (Utils::containsString(name, _mapPrefix))
                {
                    WAI_LOG("VO-Map found: %s\n", name.c_str());
                    //estimate highest used id
                    std::vector<std::string> splitted;
                    Utils::splitString(name, '-', splitted);
                    if (splitted.size())
                    {
                        int id = atoi(splitted.back().c_str());
                        existingMapNamesSorted.push_back(make_pair(id, name));
                        if (id >= _nextId)
                        {
                            _nextId = id + 1;
                            WAI_LOG("New next id: %i\n", _nextId);
                        }
                    }
                }
            }
        }
        //sort existingMapNames
        std::sort(existingMapNamesSorted.begin(), existingMapNamesSorted.end(), [](const pair<int, string>& left, const pair<int, string>& right) { return left.first < right.first; });
        for (auto it = existingMapNamesSorted.begin(); it != existingMapNamesSorted.end(); ++it)
            existingMapNames.push_back(it->second);

        //mark storage as initialized
        _isInitialized = true;
    }
    else
    {
        WAI_LOG("Failed to setup external map storage!\n");
        WAI_LOG("Exit in WAIMapStorage::init()");
        std::exit(0);
    }
}
//-----------------------------------------------------------------------------
void WAIMapStorage::saveMap(int id, WAI::ModeOrbSlam2* orbSlamMode, bool saveImgs, cv::Mat nodeOm, std::string externalDir)
{
    if (!_isInitialized)
    {
        WAI_LOG("External map storage is not initialized, you have to call init() first!\n");
        return;
    }

    if (!orbSlamMode->isInitialized())
    {
        WAI_LOG("Map storage: System is not initialized. Map saving is not possible!\n");
        return;
    }

    bool errorOccured = false;
    //check if map exists
    string mapName  = _mapPrefix + to_string(id);
    string path     = Utils::unifySlashes(_mapsDir + mapName);
    string pathImgs = path + "imgs/";
    string filename = path + mapName + ".json";

    try
    {
        //if path exists, delete content
        if (Utils::fileExists(path))
        {
            //remove json file
            if (Utils::fileExists(filename))
            {
                Utils::deleteFile(filename);
                //check if imgs dir exists and delete all containing files
                if (Utils::fileExists(pathImgs))
                {
                    std::vector<std::string> content = Utils::getFileNamesInDir(pathImgs);
                    for (auto path : content)
                    {
                        Utils::deleteFile(path);
                    }
                }
            }
        }
        else
        {
            //create map directory and imgs directory
            Utils::makeDir(path);
        }

        if (!Utils::fileExists(pathImgs))
        {
            Utils::makeDir(pathImgs);
        }

        //switch to idle, so that map does not change, while we are accessing keyframes
        orbSlamMode->pause();
#if 0
        mapTracking->sm.requestStateIdle();
        while (!mapTracking->sm.hasStateIdle())
        {
            std::this_thread::sleep_for(std::chrono::seconds(1));
        }
#endif

        //save the map
        WAIMapIO::save(filename, *orbSlamMode->getMap(), saveImgs, pathImgs, nodeOm);

        //update list of existing maps
        WAIMapStorage::init(externalDir);
        //update current combobox item
        auto it = std::find(existingMapNames.begin(), existingMapNames.end(), mapName);
        if (it != existingMapNames.end())
        {
            currN    = it - existingMapNames.begin();
            currItem = existingMapNames[currN].c_str();
        }
    }
    catch (std::exception& e)
    {
        string msg = "Exception during slam map storage: " + filename + "\n" +
                     e.what() + "\n";
        WAI_LOG("%s\n", msg.c_str());
        errorOccured = true;
    }
    catch (...)
    {
        string msg = "Exception during slam map storage: " + filename + "\n";
        WAI_LOG("%s\n", msg.c_str());
        errorOccured = true;
    }

    //if an error occured, we delete the whole directory
    if (errorOccured)
    {
        //if path exists, delete content
        if (Utils::fileExists(path))
        {
            //remove json file
            if (Utils::fileExists(filename))
                Utils::deleteFile(filename);
            //check if imgs dir exists and delete all containing files
            if (Utils::fileExists(pathImgs))
            {
                std::vector<std::string> content = Utils::getFileNamesInDir(pathImgs);
                for (auto path : content)
                {
                    Utils::deleteFile(path);
                }
                Utils::deleteFile(pathImgs);
            }
            Utils::deleteFile(path);
        }
    }

    //switch back to initialized state and resume tracking
    orbSlamMode->resume();
}
//-----------------------------------------------------------------------------
bool WAIMapStorage::loadMap(const string& path, WAI::ModeOrbSlam2* orbSlamMode, ORBVocabulary* orbVoc, bool loadKfImgs, cv::Mat& nodeOm)
{
    bool loadingSuccessful = false;
    if (!_isInitialized)
    {
        WAI_LOG("External map storage is not initialized, you have to call init() first!\n");
        return loadingSuccessful;
    }
    if (!orbSlamMode)
    {
        WAI_LOG("Map tracking not initialized!\n");
        return loadingSuccessful;
    }

    //reset tracking (and all dependent threads/objects like Map, KeyFrameDatabase, LocalMapping, loopClosing)
    orbSlamMode->requestStateIdle();
    while (!orbSlamMode->hasStateIdle())
    {
        std::this_thread::sleep_for(std::chrono::seconds(1));
    }
    orbSlamMode->reset();

    //clear map and keyframe database
    WAIMap*        map  = orbSlamMode->getMap();
    WAIKeyFrameDB* kfDB = orbSlamMode->getKfDB();

    //extract id from map name
    size_t prefixIndex = path.find(_mapPrefix);
    if (prefixIndex != string::npos)
    {
        std::string name     = path.substr(prefixIndex);
        std::string idString = name.substr(_mapPrefix.length());
        _currentId           = atoi(idString.c_str());
    }
    else
    {
        WAI_LOG("Could not load map. Map id not found in name: %s\n", path.c_str());
        return loadingSuccessful;
    }

    //check if map exists
    string mapName      = _mapPrefix + to_string(_currentId);
    string mapPath      = Utils::unifySlashes(_mapsDir + mapName);
    string currPathImgs = mapPath + "imgs/";
    string filename     = mapPath + mapName + ".json";

    //check if dir and file exist
    if (!Utils::dirExists(mapPath))
    {
        string msg = "Failed to load map. Path does not exist: " + mapPath + "\n";
        WAI_LOG("%s\n", msg.c_str());
        return loadingSuccessful;
    }
    if (!Utils::fileExists(filename))
    {
        string msg = "Failed to load map: " + filename + "\n";
        WAI_LOG("%s\n", msg.c_str());
        return loadingSuccessful;
    }

    try
    {
        WAIMapIO mapIO(filename, orbVoc, loadKfImgs, currPathImgs);
        mapIO.load(nodeOm, *map, *kfDB);

        //if map loading was successful, switch to initialized
        orbSlamMode->setInitialized(true);
        loadingSuccessful = true;
    }
    catch (std::exception& e)
    {
        string msg = "Exception during slam map loading: " + filename +
                     e.what() + "\n";
        WAI_LOG("%s\n", msg.c_str());
    }
    catch (...)
    {
        string msg = "Exception during slam map loading: " + filename + "\n";
        WAI_LOG("%s\n", msg.c_str());
    }

    orbSlamMode->resume();
    return loadingSuccessful;
}
//-----------------------------------------------------------------------------
void WAIMapStorage::newMap()
{
    if (!_isInitialized)
    {
        WAI_LOG("External map storage is not initialized, you have to call init() first!\n");
        return;
    }

    //assign next id to current id. The nextId will be increased after file save.
    _currentId = _nextId;
}
//-----------------------------------------------------------------------------
string WAIMapStorage::mapsDir()
{
    return _mapsDir;
}
