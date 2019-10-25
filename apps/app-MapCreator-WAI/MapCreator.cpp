#include "MapCreator.h"
#include <memory>

MapCreator::MapCreator(std::string erlebARDir, std::string configFile)
  : _erlebARDir(Utils::unifySlashes(erlebARDir))
{
    _calibrationsDir = _erlebARDir + "../calibrations/";
    _vocFile         = _erlebARDir + "../voc/ORBvoc.bin";
    _outputDir       = _erlebARDir + "MapCreator/";
    if (!Utils::dirExists(_outputDir))
        Utils::makeDir(_outputDir);

    //scan erlebar directory and config file, collect everything that is enabled in the config file and
    //check that all files (video and calibration) exist.
    loadSites(erlebARDir, configFile);
}

void MapCreator::loadSites(const std::string& erlebARDir, const std::string& configFile)
{
    try
    {
        WAI_DEBUG("MapCreator: loading sites:");
        //parse config file
        cv::FileStorage fs;
        fs.open(configFile, cv::FileStorage::READ);
        if (!fs.isOpened())
            throw std::runtime_error("Could not open configFile: " + configFile);

        //helper for areas that have been enabled
        std::set<Area> enabledAreas;

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
                    WAI_DEBUG("enabling %s %s", location.c_str(), area.c_str());
                    Areas& areas = _erlebAR[location];
                    //insert empty Videos vector
                    areas.insert(std::pair<std::string, std::vector<VideoAndCalib>>(area, Videos()));
                    enabledAreas.insert(area);
                }
            }
        }

        std::string erlebARDirUnified = Utils::unifySlashes(erlebARDir);
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
                        throw std::runtime_error("Video file does not exist: " + videoAndCalib.videoFile);

                    //check if calibration file exists
                    SlamVideoInfos slamVideoInfos;

                    if (!extractSlamVideoInfosFromFileName(name, &slamVideoInfos))
                        throw std::runtime_error("Could not extract slam video infos: " + name);

                    // construct calibrations file name and check if it exists
                    std::string calibFile = "camCalib_" + slamVideoInfos.deviceString + "_main.xml";

                    //videoAndCalib.calibFile = erlebARDirUnified + "../calibrations/" + "camCalib_" + slamVideoInfos.deviceString + "_main.xml";
                    if (!Utils::fileExists(_calibrationsDir + calibFile))
                        throw std::runtime_error("Calibration file does not exist: " + _calibrationsDir + calibFile);

                    //load calibration file and check for aspect ratio
                    if (!videoAndCalib.calibration.load(_calibrationsDir, calibFile, false, false))
                        throw std::runtime_error("Could not load calibration file: " + _calibrationsDir + calibFile);

                    std::vector<std::string> size;
                    Utils::splitString(slamVideoInfos.resolution, 'x', size);
                    if (size.size() == 2)
                    {
                        int width  = std::stoi(size[0]);
                        int height = std::stoi(size[1]);
                        if (videoAndCalib.calibration.imageSize().width != width ||
                            videoAndCalib.calibration.imageSize().height != height)
                        {
                            throw std::runtime_error("Resolutions of video and calibration do not fit together. Using: " + calibFile + " and " + name);
                        }
                    }
                    else
                    {
                        throw std::runtime_error("Could not estimate resolution string: " + calibFile);
                    }

                    //add video to videos vector
                    _erlebAR[location][area].push_back(videoAndCalib);
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

void MapCreator::createNewWaiMap(const Location& location, const Area& area, Videos& videos)
{
    WAI_INFO("Starting map creation for area: %s", area.c_str());

    //the lastly saved map file (only valid if initialized is true)
    std::string mapFile     = constructSlamMapFileName(location, area, Utils::getDateTime2String());
    std::string mapDir      = _outputDir + area + "/";
    bool        initialized = false;
    std::string currentMapFileName;

    const float cullRedundantPerc = 0.99f;
    initialized                   = createNewDenseWaiMap(videos, mapFile, mapDir, cullRedundantPerc, currentMapFileName);

    if (videos.size() && initialized)
    {
        const float cullRedundantPerc = 0.90f;
        //select one calibration (we need one to instantiate mode and we need mode to load map)
        thinOutNewWaiMap(mapDir, currentMapFileName, mapFile, videos.front().calibration, cullRedundantPerc);
    }
    else
    {
        WAI_WARN("No map created for area: %s", area.c_str());
    }

    WAI_INFO("Finished map creation for area: %s", area.c_str());
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

    //use all videos to create a new map
    for (auto itVideos = videos.begin(); itVideos != videos.end(); ++itVideos)
    {
        WAI_DEBUG("Starting video %s", itVideos->videoFile.c_str());
        lastMapFileName    = currentMapFileName;
        currentMapFileName = std::to_string(videoIndex) + "_" + mapFile;

        //initialze capture
        CVCapture* cap = CVCapture::instance();
        cap->videoType(CVVideoType::VT_FILE);
        cap->videoFilename    = itVideos->videoFile;
        cap->activeCalib      = &itVideos->calibration;
        cap->videoLoops       = true;
        cv::Size capturedSize = cap->openFile();
        //check if resolution of captured frame fits to calibration
        if (capturedSize.width != cap->activeCalib->imageSize().width ||
            capturedSize.height != cap->activeCalib->imageSize().height)
            throw std::runtime_error("MapCreator::createWaiMap: Resolution of captured frame does not fit to calibration: " + itVideos->videoFile);

        //instantiate wai mode
        std::unique_ptr<WAI::ModeOrbSlam2> waiMode =
          std::make_unique<WAI::ModeOrbSlam2>(cap->activeCalib->cameraMat(),
                                              cap->activeCalib->distortion(),
                                              modeParams,
                                              _vocFile);

        //if we have an active map from one of the previously processed videos for this area then load it
        if (initialized)
        {
            loadMap(waiMode.get(), mapDir, lastMapFileName, modeParams.fixOldKfs);
        }

        //frame with which map was initialized (we want to run the previous frames again)
        int  finalFrameIndex = 0;
        bool relocalizedOnce = false;

        while (cap->isOpened())
        {
            int currentFrameIndex = cap->nextFrameIndex();
            if (finalFrameIndex == currentFrameIndex && relocalizedOnce)
            {
                break;
            }

            if (!cap->grabAndAdjustForSL(cap->activeCalib->imageAspectRatio()))
                break;

            //update wai
            waiMode->update(cap->lastFrameGray, cap->lastFrame);

            //check if it relocalized once
            if (!relocalizedOnce && waiMode->getTrackingState() == WAI::TrackingState::TrackingState_TrackingOK)
            {
                relocalizedOnce = true;
                //if it relocalized once we will store the current index and repeat video up to this index
                finalFrameIndex = currentFrameIndex;
                WAI_DEBUG("Relocalized once for video %s at index %2", itVideos->videoFile.c_str(), std::to_string(finalFrameIndex).c_str());
            }

            decorateDebug(waiMode.get(), cap, currentFrameIndex, waiMode->getNumKeyFrames());
        }

        //save map if it was initialized
        if (waiMode->isInitialized())
        {
            initialized = true;
            saveMap(waiMode.get(), mapDir, currentMapFileName);
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

    //instantiate wai mode
    std::unique_ptr<WAI::ModeOrbSlam2> waiMode =
      std::make_unique<WAI::ModeOrbSlam2>(calib.cameraMat(),
                                          calib.distortion(),
                                          modeParams,
                                          _vocFile);

    //load the map (currentMapFileName is valid if initialized is true)
    loadMap(waiMode.get(), mapDir, inputMapFile, modeParams.fixOldKfs);

    //cull keyframes
    std::vector<WAIKeyFrame*> kfs = waiMode->getMap()->GetAllKeyFrames();
    cullKeyframes(kfs, modeParams.cullRedundantPerc);

    //save map again (we use the map file name without index because this is the final map)
    saveMap(waiMode.get(), mapDir, outputMapFile);
}

void MapCreator::cullKeyframes(std::vector<WAIKeyFrame*>& kfs, const float cullRedundantPerc)
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
            }
        }
    }
}

void MapCreator::decorateDebug(WAI::ModeOrbSlam2* waiMode, CVCapture* cap, const int currentFrameIndex, const int numOfKfs)
{
    //#ifdef _DEBUG
    if (!cap->lastFrame.empty())
    {
        cv::Mat            decoImg      = cap->lastFrame.clone();
        WAI::TrackingState waiModeState = waiMode->getTrackingState();

        double     fontScale = 0.5;
        cv::Point  stateOff(10, 25);
        cv::Point  idxOff = stateOff + cv::Point(0, 20);
        cv::Point  kfsOff = idxOff + cv::Point(0, 20);
        cv::Scalar color  = CV_RGB(255, 0, 0);
        if (waiModeState == WAI::TrackingState::TrackingState_Initializing)
            cv::putText(decoImg, "Initializing", stateOff, 0, fontScale, color);
        else if (waiModeState == WAI::TrackingState::TrackingState_TrackingLost)
            cv::putText(decoImg, "Relocalizing", stateOff, 0, fontScale, color);
        else if (waiModeState == WAI::TrackingState::TrackingState_TrackingOK)
            cv::putText(decoImg, "Tracking", stateOff, 0, fontScale, color);

        cv::putText(decoImg, "FrameId: " + std::to_string(currentFrameIndex), idxOff, 0, fontScale, color);
        cv::putText(decoImg, "Num Kfs: " + std::to_string(numOfKfs), kfsOff, 0, fontScale, color);
        cv::imshow("lastFrame", decoImg);
        cv::waitKey(1);
    }
    //#endif
}

void MapCreator::saveMap(WAI::ModeOrbSlam2* waiMode, const std::string& mapDir, const std::string& currentMapFileName)
{
    if (!Utils::dirExists(mapDir))
        Utils::makeDir(mapDir);

    std::string imgDir = constructSlamMapImgDir(mapDir, currentMapFileName);

    if (waiMode->retainImage())
    {
        if (!Utils::dirExists(imgDir))
            Utils::makeDir(imgDir);
    }

    if (!WAIMapStorage::saveMap(waiMode->getMap(),
                                nullptr,
                                mapDir + currentMapFileName,
                                imgDir))
    {
        throw std::runtime_error("Could not save map file: " + mapDir + currentMapFileName);
    }
}

void MapCreator::loadMap(WAI::ModeOrbSlam2* waiMode, const std::string& mapDir, const std::string& currentMapFileName, bool fixKfsForLBA)
{
    waiMode->requestStateIdle();
    while (!waiMode->hasStateIdle())
    {
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    waiMode->reset();

    bool mapLoadingSuccess = WAIMapStorage::loadMap(waiMode->getMap(),
                                                    waiMode->getKfDB(),
                                                    nullptr,
                                                    mapDir + currentMapFileName,
                                                    waiMode->retainImage(),
                                                    fixKfsForLBA);

    if (!mapLoadingSuccess)
    {
        throw std::runtime_error("Could not load map from file: " + mapDir + currentMapFileName);
    }

    waiMode->resume();
    waiMode->setInitialized(true);
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
