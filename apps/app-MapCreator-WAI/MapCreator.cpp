#include "MapCreator.h"
#include <memory>
#include <CVCamera.h>
#include <GLSLextractor.h>
#include <FeatureExtractorFactory.h>

MapCreator::MapCreator(std::string erlebARDir, std::string configFile, std::string vocFile)
  : _erlebARDir(Utils::unifySlashes(erlebARDir))
{
    _calibrationsDir = _erlebARDir + "calibrations/";
    _vocFile         = vocFile;
    _outputDir       = _erlebARDir + "MapCreator/";
    if (!Utils::dirExists(_outputDir))
        Utils::makeDir(_outputDir);

    _mpUL = nullptr;
    _mpUR = nullptr;
    _mpLL = nullptr;
    _mpLR = nullptr;

    //scan erlebar directory and config file, collect everything that is enabled in the config file and
    //check that all files (video and calibration) exist.
    loadSites(erlebARDir, configFile);

    //init keypoint extractors
    //TODO(lulu) create extractor depending on video resolution especially if different for each video!
    FeatureExtractorFactory factory;
    _kpExtractor = factory.make(7, {640, 320});
    //_kpIniExtractor    = factory.make(8, {640, 360});
    //_kpMarkerExtractor = factory.make(8, {640, 360});
}

MapCreator::~MapCreator()
{
}

void MapCreator::loadSites(const std::string& erlebARDir, const std::string& configFile)
{
    try
    {
        WAI_DEBUG("MapCreator: loading sites:");
        //parse config file
        cv::FileStorage fs;
        std::cout << "erlebBarDir " << erlebARDir << std::endl;
        std::cout << "configFile " << configFile << std::endl
                  << std::endl;

        fs.open(configFile, cv::FileStorage::READ);
        if (!fs.isOpened())
            throw std::runtime_error("MapCreator::loadSites: Could not open configFile: " + configFile);

        //helper for areas that have been enabled
        std::set<Area> enabledAreas;

        std::string erlebARDirUnified = Utils::unifySlashes(erlebARDir);

        //setup for enabled areas
        cv::FileNode locsNode = fs["locationsEnabling"];
        for (auto itLocs = locsNode.begin(); itLocs != locsNode.end(); ++itLocs)
        {
            std::string  location  = (*itLocs)["location"];
            cv::FileNode areasNode = (*itLocs)["areas"];
            Areas        areas;
            for (auto itAreas = areasNode.begin(); itAreas != areasNode.end(); ++itAreas)
            {
                std::string area    = (*itAreas)["area"];
                bool        enabled = false;
                (*itAreas)["enabled"] >> enabled;
                if (enabled)
                {
                    WAI_DEBUG("MapCreator::loadSites: enabling %s %s", location.c_str(), area.c_str());
                    Areas&     areas      = _erlebAR[location];
                    AreaConfig areaConfig = {};
                    areaConfig.videos     = Videos();

                    std::string markerFileName;
                    (*itAreas)["marker"] >> markerFileName;

                    if (!markerFileName.empty())
                    {
                        areaConfig.markerFile = erlebARDirUnified + "locations/" + location + "/" + area + "/" + "markers/" + markerFileName;

                        if (!Utils::fileExists(areaConfig.markerFile))
                            throw std::runtime_error("Marker file does not exist: " + areaConfig.markerFile);

                        WAI_DEBUG("MapCreator::loadSites: %s %s uses markerfile %s", location.c_str(), area.c_str(), areaConfig.markerFile.c_str());
                    }

                    //insert empty Videos vector
                    areas.insert(std::pair<Area, AreaConfig>(area, areaConfig));
                    enabledAreas.insert(area);
                }
            }
        }

        //try to find corresponding files in sites directory and add full file paths to _sites
        cv::FileNode videoAreasNode = fs["mappingVideos"];
        for (auto itVideoAreas = videoAreasNode.begin(); itVideoAreas != videoAreasNode.end(); ++itVideoAreas)
        {
            Location location = (*itVideoAreas)["location"];
            Area     area     = (*itVideoAreas)["area"];
            if (enabledAreas.find(area) != enabledAreas.end())
            {
                cv::FileNode videosNode = (*itVideoAreas)["videos"];
                for (auto itVideos = videosNode.begin(); itVideos != videosNode.end(); ++itVideos)
                {
                    //check if this is enabled
                    std::string   name = *itVideos;
                    VideoAndCalib videoAndCalib;
                    videoAndCalib.videoFile = erlebARDirUnified + "locations/" + location + "/" + area + "/" + "videos/" + name;

                    if (!Utils::fileExists(videoAndCalib.videoFile))
                        throw std::runtime_error("MapCreator::loadSites: Video file does not exist: " + videoAndCalib.videoFile);

                    //check if calibration file exists
                    SlamVideoInfos slamVideoInfos;

                    if (!extractSlamVideoInfosFromFileName(name, &slamVideoInfos))
                        throw std::runtime_error("MapCreator::loadSites: Could not extract slam video infos: " + name);

                    // construct calibrations file name and check if it exists
                    std::string calibFile = "camCalib_" + slamVideoInfos.deviceString + "_main.xml";

                    //videoAndCalib.calibFile = erlebARDirUnified + "../calibrations/" + "camCalib_" + slamVideoInfos.deviceString + "_main.xml";
                    if (!Utils::fileExists(_calibrationsDir + calibFile))
                        throw std::runtime_error("MapCreator::loadSites: Calibration file does not exist: " + _calibrationsDir + calibFile);

                    //load calibration file and check for aspect ratio
                    if (!videoAndCalib.calibration.load(_calibrationsDir, calibFile, true))
                        throw std::runtime_error("MapCreator::loadSites: Could not load calibration file: " + _calibrationsDir + calibFile);

                    std::vector<std::string> size;
                    Utils::splitString(slamVideoInfos.resolution, 'x', size);
                    if (size.size() == 2)
                    {
                        int width  = std::stoi(size[0]);
                        int height = std::stoi(size[1]);
                        if (videoAndCalib.calibration.imageSize().width != width ||
                            videoAndCalib.calibration.imageSize().height != height)
                        {
                            videoAndCalib.calibration.adaptForNewResolution(CVSize(width, height), true);
                            //throw std::runtime_error("Resolutions of video and calibration do not fit together. Using: " + calibFile + " and " + name);
                        }
                    }
                    else
                    {
                        throw std::runtime_error("MapCreator::loadSites: Could not estimate resolution string: " + calibFile);
                    }

                    //add video to videos vector
                    _erlebAR[location][area].videos.push_back(videoAndCalib);
                }
            }
        }
    }
    catch (std::exception& e)
    {
        throw std::runtime_error("Exception in MapCreator::loadSites: " + std::string(e.what()));
    }
    catch (...)
    {
        throw std::runtime_error("Unknown exception catched in MapCreator::loadSites!");
    }
}

bool MapCreator::createMarkerMap(AreaConfig&        areaConfig,
                                 const std::string& mapFile,
                                 const std::string& mapDir,
                                 const float        cullRedundantPerc)
{
    //wai mode config
    WAI::ModeOrbSlam2::Params modeParams;
    modeParams.cullRedundantPerc = cullRedundantPerc;
    modeParams.serial            = true;
    modeParams.fixOldKfs         = false;
    modeParams.retainImg         = true;

    ORBVocabulary* voc = new ORBVocabulary();
    if (!voc->loadFromBinaryFile(_vocFile))
    {
        std::cout << "Can't open vocabulary file!!! " << _vocFile << std::endl;
        exit(1);
    }
    WAIKeyFrameDB* kfDB          = new WAIKeyFrameDB(*voc);
    WAIMap*        map           = new WAIMap(kfDB);
    SLNode         mapNode       = SLNode();
    cv::Mat        nodeTransform = cv::Mat::eye(4, 4, CV_32F);

    bool mapLoadingSuccess = WAIMapStorage::loadMap(map,
                                                    &mapNode,
                                                    voc,
                                                    mapDir + "/" + mapFile,
                                                    false,
                                                    modeParams.fixOldKfs);

    if (!mapLoadingSuccess)
    {
        std::cout << ("MapCreator::createMarkerMap: Could not load map from file " + mapDir + "/" + mapFile) << std::endl;
        return false;
    }

    cv::Mat markerImgGray = cv::imread(areaConfig.markerFile, cv::IMREAD_GRAYSCALE);

    FeatureExtractorFactory      factory;
    std::unique_ptr<KPextractor> kpExtractor = factory.make(7, {markerImgGray.cols, markerImgGray.rows});

    bool result = WAISlamTools::doMarkerMapPreprocessing(areaConfig.markerFile,
                                                         nodeTransform,
                                                         0.355f,
                                                         kpExtractor.get(),
                                                         map,
                                                         areaConfig.videos.front().calibration.cameraMat(), // TODO(dgj1): use actual calibration for marker image
                                                         voc);

    return result;
}

void MapCreator::createNewWaiMap(const Location& location, const Area& area, AreaConfig& areaConfig)
{
    WAI_INFO("MapCreator::createNewWaiMap: Starting map creation for area: %s", area.c_str());
    //the lastly saved map file (only valid if initialized is true)
    std::string mapFile     = constructSlamMapFileName(location, area, _kpExtractor->GetName(), Utils::getDateTime2String());
    std::string mapDir      = _outputDir + area + "/";
    bool        initialized = false;
    std::string currentMapFileName;

    const float cullRedundantPerc = 0.99f;
    initialized                   = createNewDenseWaiMap(areaConfig.videos, mapFile, mapDir, cullRedundantPerc, currentMapFileName);

    if (areaConfig.videos.size() && initialized)
    {
        if (!areaConfig.markerFile.empty())
        {
            if (!createMarkerMap(areaConfig, currentMapFileName, mapDir, cullRedundantPerc))
            {
                WAI_WARN("MapCreator::createNewWaiMap: Could not do marker map preprocessing for %s %s", location.c_str(), area.c_str());
            }
        }

        const float cullRedundantPerc = 0.95f;
        //select one calibration (we need one to instantiate mode and we need mode to load map)
        thinOutNewWaiMap(mapDir, currentMapFileName, mapFile, areaConfig.videos.front().calibration, cullRedundantPerc);
    }
    else
    {
        WAI_WARN("MapCreator::createNewWaiMap: No map created for area: %s", area.c_str());
    }

    WAI_INFO("MapCreator::createNewWaiMap: Finished map creation for area: %s", area.c_str());
}

bool MapCreator::createNewDenseWaiMap(Videos&            videos,
                                      const std::string& mapFile,
                                      const std::string& mapDir,
                                      const float        cullRedundantPerc,
                                      std::string&       currentMapFileName)
{
    bool initialized = false;
    //wai mode config
    WAI::ModeOrbSlam2::Params modeParams;
    modeParams.cullRedundantPerc = cullRedundantPerc;
    modeParams.serial            = true;
    modeParams.fixOldKfs         = false;
    modeParams.retainImg         = true;

    //map creation parameter:
    int         videoIndex = 0;
    std::string lastMapFileName;

    //We want to repeat videos that did not initialize or relocalize after processing all other videos once.
    //alreadyRepeatedVideos is a helper set to track which videos are already repeated to not repeat endlessly.
    //We use the video file name as identifier.
    std::set<std::string> alreadyRepeatedVideos;

    //use all videos to create a new map
    for (Videos::size_type videoIdx = 0; videoIdx < videos.size(); ++videoIdx)
    {
        WAI_DEBUG("MapCreator::createNewDenseWaiMap: Starting video %s", videos[videoIdx].videoFile.c_str());
        lastMapFileName    = currentMapFileName;
        currentMapFileName = std::to_string(videoIndex) + "_" + mapFile;

        //initialze capture
        CVCapture* cap = CVCapture::instance();
        cap->videoType(CVVideoType::VT_FILE);
        cap->videoFilename             = videos[videoIdx].videoFile;
        cap->activeCamera->calibration = videos[videoIdx].calibration;
        cap->videoLoops                = true;
        cv::Size capturedSize          = cap->openFile();
        //check if resolution of captured frame fits to calibration
        if (capturedSize.width != cap->activeCamera->calibration.imageSize().width ||
            capturedSize.height != cap->activeCamera->calibration.imageSize().height)
            throw std::runtime_error("MapCreator::createNewDenseWaiMap: Resolution of captured frame does not fit to calibration: " + videos[videoIdx].videoFile);

        FeatureExtractorFactory factory;
        _kpExtractor = factory.make(7, {capturedSize.width, capturedSize.height});

        ORBVocabulary* voc = new ORBVocabulary();
        if (!voc->loadFromBinaryFile(_vocFile))
        {
            std::cout << "MapCreator::createNewDenseWaiMap: Can't open vocabulary file!!! " << _vocFile << std::endl;
            exit(1);
        }
        WAIMap* map = nullptr;

        //if we have an active map from one of the previously processed videos for this area then load it
        SLNode mapNode = SLNode();
        if (initialized)
        {
            WAIKeyFrameDB* kfdb    = new WAIKeyFrameDB(*voc);
            map                    = new WAIMap(kfdb);
            bool mapLoadingSuccess = WAIMapStorage::loadMap(map,
                                                            &mapNode,
                                                            voc,
                                                            mapDir + lastMapFileName,
                                                            false,
                                                            modeParams.fixOldKfs);
            if (!mapLoadingSuccess)
            {
                std::cout << ("MapCreator::createNewDenseWaiMap: Could not load map from file " + mapDir + "/" + mapFile) << std::endl;
                return;
            }
        }
        else
        {
            std::cout << "MapCreator::createNewDenseWaiMap: not initialized" << std::endl;
        }

        //instantiate wai mode
        std::unique_ptr<WAISlam> waiMode =
          std::make_unique<WAISlam>(cap->activeCamera->calibration.cameraMat(),
                                    cap->activeCamera->calibration.distortion(),
                                    voc,
                                    _kpExtractor.get(),
                                    map,
                                    modeParams.onlyTracking,
                                    modeParams.serial,
                                    modeParams.retainImg,
                                    modeParams.cullRedundantPerc);

        int firstRun = true;

        //frame with which map was initialized (we want to run the previous frames again)
        int  videoLength     = cap->videoLength();
        int  finalFrameIndex = videoLength;
        bool relocalizedOnce = false;

        while (cap->isOpened())
        {
            int currentFrameIndex = cap->nextFrameIndex();
            if (finalFrameIndex == currentFrameIndex)
            {
                if (!relocalizedOnce)
                {
                    //If this is not the last video or we already repeated it add it at the end of videos so that is
                    //can be processed again later when there is more information in the map
                    if (videoIdx != videos.size() - 1 && alreadyRepeatedVideos.find(videos[videoIdx].videoFile) == alreadyRepeatedVideos.end())
                    {
                        alreadyRepeatedVideos.insert(videos[videoIdx].videoFile);
                        videos.push_back(videos[videoIdx]);
                    }
                }
                break;
            }

            if (!cap->grabAndAdjustForSL(cap->activeCamera->calibration.imageAspectRatio()))
                break;

            //update wai
            waiMode->update(cap->lastFrameGray);
            if (firstRun)
            {
                firstRun = false;
            }
            //check if it relocalized once
            if (!relocalizedOnce && waiMode->getTrackingState() == WAI::TrackingState::TrackingState_TrackingOK)
            {
                relocalizedOnce = true;
                //if it relocalized once we will store the current index and repeat video up to this index
                finalFrameIndex = currentFrameIndex;
                WAI_DEBUG("Relocalized once for video %s at index %i", videos[videoIdx].videoFile.c_str(), finalFrameIndex);
            }

            decorateDebug(waiMode.get(), cap, currentFrameIndex, videoLength, waiMode->getNumKeyFrames());
        }

        //save map if it was initialized
        if (waiMode->isInitialized())
        {
            initialized = true;
            saveMap(waiMode.get(), mapDir, currentMapFileName, &mapNode);
        }
        else
        {
            std::cout << "Mode return not initialized!!" << std::endl;
        }

        //increment video index for map saving
        videoIndex++;
    }

    return initialized;
}

void MapCreator::thinOutNewWaiMap(const std::string& mapDir,
                                  const std::string& inputMapFile,
                                  const std::string  outputMapFile,
                                  CVCalibration&     calib,
                                  const float        cullRedundantPerc)
{
    //wai mode config
    WAI::ModeOrbSlam2::Params modeParams;
    modeParams.cullRedundantPerc = cullRedundantPerc;
    modeParams.serial            = true;
    modeParams.fixOldKfs         = false;
    modeParams.retainImg         = true;

    ORBVocabulary* voc = new ORBVocabulary();
    if (!voc->loadFromBinaryFile(_vocFile))
    {
        std::cout << "Can't open vocabulary file!!! " << _vocFile << std::endl;
        exit(1);
    }
    WAIKeyFrameDB* kfdb = new WAIKeyFrameDB(*voc);
    WAIMap*        map  = new WAIMap(kfdb);

    //load the map (currentMapFileName is valid if initialized is true)
    SLNode mapNode = SLNode();

    bool mapLoadingSuccess = WAIMapStorage::loadMap(map,
                                                    &mapNode,
                                                    voc,
                                                    mapDir + "/" + inputMapFile,
                                                    false,
                                                    modeParams.fixOldKfs);
    if (!mapLoadingSuccess)
    {
        std::cout << ("MapCreator::thinOutNewWaiMap: Could not load map from file " + mapDir + "/" + inputMapFile) << std::endl;
        return;
    }
    //instantiate wai mode
    std::unique_ptr<WAISlam> waiMode =
      std::make_unique<WAISlam>(calib.cameraMat(),
                                calib.distortion(),
                                voc,
                                _kpExtractor.get(),
                                map,
                                modeParams.onlyTracking,
                                modeParams.serial,
                                modeParams.retainImg,
                                modeParams.cullRedundantPerc);

    //cull keyframes
    std::vector<WAIKeyFrame*> kfs = waiMode->getMap()->GetAllKeyFrames();
    cullKeyframes(waiMode.get(), kfs, modeParams.cullRedundantPerc);

    //save map again (we use the map file name without index because this is the final map)
    saveMap(waiMode.get(), mapDir, outputMapFile, &mapNode);
}

void MapCreator::cullKeyframes(WAISlam* waiMode, std::vector<WAIKeyFrame*>& kfs, const float cullRedundantPerc)
{
    for (auto itKF = kfs.begin(); itKF != kfs.end(); ++itKF)
    {
        vector<WAIKeyFrame*> vpLocalKeyFrames = (*itKF)->GetVectorCovisibleKeyFrames();

        for (vector<WAIKeyFrame*>::iterator vit = vpLocalKeyFrames.begin(), vend = vpLocalKeyFrames.end(); vit != vend; vit++)
        {
            WAIKeyFrame* pKF = *vit;
            //do not cull the first keyframe
            if (pKF->mnId == 0)
                continue;
            //do not cull fixed keyframes
            if (pKF->isFixed())
                continue;

            const vector<WAIMapPoint*> vpMapPoints = pKF->GetMapPointMatches();

            //int       nObs                   = 3;
            const int thObs                  = 3;
            int       nRedundantObservations = 0;
            int       nMPs                   = 0;
            for (size_t i = 0, iend = vpMapPoints.size(); i < iend; i++)
            {
                WAIMapPoint* pMP = vpMapPoints[i];
                if (pMP)
                {
                    if (!pMP->isBad())
                    {
                        nMPs++;
                        if (pMP->Observations() > thObs)
                        {
                            const int&                           scaleLevel   = pKF->mvKeysUn[i].octave;
                            const std::map<WAIKeyFrame*, size_t> observations = pMP->GetObservations();
                            int                                  nObs         = 0;
                            for (std::map<WAIKeyFrame*, size_t>::const_iterator mit = observations.begin(), mend = observations.end(); mit != mend; mit++)
                            {
                                WAIKeyFrame* pKFi = mit->first;
                                if (pKFi == pKF)
                                    continue;
                                const int& scaleLeveli = pKFi->mvKeysUn[mit->second].octave;

                                if (scaleLeveli <= scaleLevel + 1)
                                {
                                    nObs++;
                                    if (nObs >= thObs)
                                    {
                                        break;
                                    }
                                }
                            }
                            if (nObs >= thObs)
                            {
                                nRedundantObservations++;
                            }
                        }
                    }
                }
            }

            if (nRedundantObservations > cullRedundantPerc * nMPs)
            {
                pKF->SetBadFlag();
                waiMode->getMap()->EraseKeyFrame(pKF);
            }
        }
    }
}

void MapCreator::decorateDebug(WAISlam* waiMode, CVCapture* cap, const int currentFrameIndex, const int videoLength, const int numOfKfs)
{
    //#ifdef _DEBUG
    if (!cap->lastFrame.empty())
    {
        cv::Mat     decoImg = cap->lastFrame.clone();
        std::string state   = waiMode->getPrintableState();

        waiMode->drawInfo(decoImg, true, true, true);

        double     fontScale = 0.5;
        cv::Point  stateOff(10, 25);
        cv::Point  idxOff = stateOff + cv::Point(0, 20);
        cv::Point  kfsOff = idxOff + cv::Point(0, 20);
        cv::Scalar color  = CV_RGB(255, 0, 0);
        cv::putText(decoImg, state, stateOff, 0, fontScale, color);
        cv::putText(decoImg, "FrameId: (" + std::to_string(currentFrameIndex) + "/" + std::to_string(videoLength) + ")", idxOff, 0, fontScale, color);
        cv::putText(decoImg, "Num Kfs: " + std::to_string(numOfKfs), kfsOff, 0, fontScale, color);
        cv::imshow("lastFrame", decoImg);
        cv::waitKey(1);
    }
    //#endif
}

void MapCreator::saveMap(WAISlam*           waiMode,
                         const std::string& mapDir,
                         const std::string& currentMapFileName,
                         SLNode*            mapNode)
{
    if (!Utils::dirExists(mapDir))
        Utils::makeDir(mapDir);

    std::string imgDir = constructSlamMapImgDir(mapDir, currentMapFileName);

    if (waiMode->retainImage())
    {
        if (!Utils::dirExists(imgDir))
            Utils::makeDir(imgDir);
    }

    waiMode->requestStateIdle();
    while (!waiMode->hasStateIdle())
    {
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }

    if (!WAIMapStorage::saveMap(waiMode->getMap(),
                                mapNode,
                                mapDir + currentMapFileName,
                                imgDir))
    {
        throw std::runtime_error("MapCreator::saveMap: Could not save map file: " + mapDir + currentMapFileName);
    }

    waiMode->resume();
}

void MapCreator::execute()
{
    try
    {
        for (auto itLocations = _erlebAR.begin(); itLocations != _erlebAR.end(); ++itLocations)
        {
            Areas& areas = itLocations->second;
            for (auto itAreas = areas.begin(); itAreas != areas.end(); ++itAreas)
            {
                createNewWaiMap(itLocations->first, itAreas->first, itAreas->second);
            }
        }
    }
    catch (std::exception& e)
    {
        throw std::runtime_error("Exception in MapCreator::execute: " + std::string(e.what()));
    }
    catch (...)
    {
        throw std::runtime_error("Unknown exception catched in MapCreator::execute!");
    }
}
