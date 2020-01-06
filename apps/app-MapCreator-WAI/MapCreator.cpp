#include "MapCreator.h"
#include <memory>
#include <CVCamera.h>

MapCreator::MapCreator(std::string erlebARDir, std::string configFile)
  : _erlebARDir(Utils::unifySlashes(erlebARDir))
{
    _calibrationsDir = _erlebARDir + "calibrations/";
    _vocFile         = _erlebARDir + "../voc/ORBvoc.bin";
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
                    WAI_DEBUG("enabling %s %s", location.c_str(), area.c_str());
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

                        WAI_DEBUG("%s %s uses markerfile %s", location.c_str(), area.c_str(), areaConfig.markerFile.c_str());
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
                    if (!videoAndCalib.calibration.load(_calibrationsDir, calibFile))
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
                            videoAndCalib.calibration.adaptForNewResolution(CVSize(width, height));
                            //throw std::runtime_error("Resolutions of video and calibration do not fit together. Using: " + calibFile + " and " + name);
                        }
                    }
                    else
                    {
                        throw std::runtime_error("Could not estimate resolution string: " + calibFile);
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

void MapCreator::createNewWaiMap(const Location& location, const Area& area, AreaConfig& areaConfig)
{
    WAI_INFO("Starting map creation for area: %s", area.c_str());

    //the lastly saved map file (only valid if initialized is true)
    std::string mapFile     = constructSlamMapFileName(location, area, "SURF", Utils::getDateTime2String()); // TODO(dgj1): replace SURF with actual extractor type
    std::string mapDir      = _outputDir + area + "/";
    bool        initialized = false;
    std::string currentMapFileName;

    const float cullRedundantPerc = 0.99f;
    initialized                   = createNewDenseWaiMap(areaConfig.videos, mapFile, mapDir, cullRedundantPerc, currentMapFileName);

    if (areaConfig.videos.size() && initialized)
    {
        if (!areaConfig.markerFile.empty())
        {
            if (!doMarkerMapPreprocessing(mapDir, currentMapFileName, areaConfig.markerFile, 0.355f, areaConfig.videos.front().calibration, cullRedundantPerc))
            {
                WAI_WARN("Could not do marker map preprocessing for %s %s", location.c_str(), area.c_str());
            }
        }

        const float cullRedundantPerc = 0.95f;
        //select one calibration (we need one to instantiate mode and we need mode to load map)
        thinOutNewWaiMap(mapDir, currentMapFileName, mapFile, areaConfig.videos.front().calibration, cullRedundantPerc);
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

    //We want to repeat videos that did not initialize or relocalize after processing all other videos once.
    //alreadyRepeatedVideos is a helper set to track which videos are already repeated to not repeat endlessly.
    //We use the video file name as identifier.
    std::set<std::string> alreadyRepeatedVideos;

    //use all videos to create a new map
    for (Videos::size_type videoIdx = 0; videoIdx < videos.size(); ++videoIdx)
    {
        WAI_DEBUG("Starting video %s", videos[videoIdx].videoFile.c_str());
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
            throw std::runtime_error("MapCreator::createWaiMap: Resolution of captured frame does not fit to calibration: " + videos[videoIdx].videoFile);

        //instantiate wai mode
        std::unique_ptr<WAI::ModeOrbSlam2> waiMode =
          std::make_unique<WAI::ModeOrbSlam2>(cap->activeCamera->calibration.cameraMat(),
                                              cap->activeCamera->calibration.distortion(),
                                              modeParams,
                                              _vocFile);

        //if we have an active map from one of the previously processed videos for this area then load it
        SLNode mapNode = SLNode();
        if (initialized)
        {
            loadMap(waiMode.get(), mapDir, lastMapFileName, modeParams.fixOldKfs, &mapNode);
        }

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
            waiMode->update(cap->lastFrameGray, cap->lastFrame);

            //check if it relocalized once
            if (!relocalizedOnce && waiMode->getTrackingState() == WAI::TrackingState::TrackingState_TrackingOK)
            {
                relocalizedOnce = true;
                //if it relocalized once we will store the current index and repeat video up to this index
                finalFrameIndex = currentFrameIndex;
                WAI_DEBUG("Relocalized once for video %s at index %2", videos[videoIdx].videoFile.c_str(), std::to_string(finalFrameIndex).c_str());
            }

            decorateDebug(waiMode.get(), cap, currentFrameIndex, videoLength, waiMode->getNumKeyFrames());
        }

        //save map if it was initialized
        if (waiMode->isInitialized())
        {
            initialized = true;
            saveMap(waiMode.get(), mapDir, currentMapFileName, &mapNode);
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
    SLNode mapNode = SLNode();
    loadMap(waiMode.get(), mapDir, inputMapFile, modeParams.fixOldKfs, &mapNode);

    //cull keyframes
    std::vector<WAIKeyFrame*> kfs = waiMode->getMap()->GetAllKeyFrames();
    cullKeyframes(kfs, modeParams.cullRedundantPerc);

    //save map again (we use the map file name without index because this is the final map)
    saveMap(waiMode.get(), mapDir, outputMapFile, &mapNode);
}

bool MapCreator::findMarkerHomography(WAIFrame&    markerFrame,
                                      WAIKeyFrame* kfCand,
                                      cv::Mat&     homography,
                                      int          minMatches)
{
    bool result = false;

    ORBmatcher matcher(0.9, true);

    std::vector<int> markerMatchesToCurrentFrame;
    int              nmatches = matcher.SearchForMarkerMap(markerFrame, *kfCand, markerMatchesToCurrentFrame);

    if (nmatches > minMatches)
    {
        std::vector<cv::Point2f> markerPoints;
        std::vector<cv::Point2f> framePoints;

        for (int j = 0; j < markerMatchesToCurrentFrame.size(); j++)
        {
            if (markerMatchesToCurrentFrame[j] >= 0)
            {
                markerPoints.push_back(markerFrame.mvKeysUn[j].pt);
                framePoints.push_back(kfCand->mvKeysUn[markerMatchesToCurrentFrame[j]].pt);
            }
        }

        homography = cv::findHomography(markerPoints,
                                        framePoints,
                                        cv::RANSAC);

        if (!homography.empty())
        {
            homography.convertTo(homography, CV_32F);

            result = true;
        }
    }

    return result;
}

bool MapCreator::doMarkerMapPreprocessing(const std::string& mapDir,
                                          const std::string& mapFile,
                                          std::string        markerFile,
                                          float              markerWidthInM,
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

    SLNode mapNode = SLNode();
    loadMap(waiMode.get(), mapDir, mapFile, modeParams.fixOldKfs, &mapNode);

    // Additional steps to save marker map
    // 1. Find matches to marker on two keyframes
    // 1.a Extract features from marker image
    KPextractor* markerExtractor = new ORB_SLAM2::SURFextractor(800); //TODO(dgj1): delete this somewhere
    WAIFrame     markerFrame     = waiMode->createMarkerFrame(markerFile, markerExtractor);

    // 1.b Find keyframes with enough matches to marker image
    WAIMap*                   map = waiMode->getMap();
    std::vector<WAIKeyFrame*> kfs = map->GetAllKeyFrames();

    WAIKeyFrame* matchedKf1 = nullptr;
    WAIKeyFrame* matchedKf2 = nullptr;

    cv::Mat ul = cv::Mat(cv::Point3f(0, 0, 1));
    cv::Mat ur = cv::Mat(cv::Point3f(markerFrame.imgGray.cols, 0, 1));
    cv::Mat ll = cv::Mat(cv::Point3f(0, markerFrame.imgGray.rows, 1));
    cv::Mat lr = cv::Mat(cv::Point3f(markerFrame.imgGray.cols, markerFrame.imgGray.rows, 1));

    cv::Mat ulKf1, urKf1, llKf1, lrKf1, ulKf2, urKf2, llKf2, lrKf2;
    cv::Mat ul3D, ur3D, ll3D, lr3D;
    cv::Mat AC, AB, n;

    for (int i1 = 0; i1 < kfs.size() - 1; i1++)
    {
        WAIKeyFrame* kfCand1 = kfs[i1];

        if (kfCand1->isBad()) continue;

        // 2. Calculate homography between the keyframes and marker
        cv::Mat homography1;
        if (findMarkerHomography(markerFrame, kfCand1, homography1, 50))
        {
            // 3.a Calculate position of the markers cornerpoints on first keyframe in 2D
            // NOTE(dgj1): assumption that intrinsic camera parameters are the same
            // TODO(dgj1): think about this assumption
            ulKf1 = homography1 * ul;
            ulKf1 /= ulKf1.at<float>(2, 0);
            urKf1 = homography1 * ur;
            urKf1 /= urKf1.at<float>(2, 0);
            llKf1 = homography1 * ll;
            llKf1 /= llKf1.at<float>(2, 0);
            lrKf1 = homography1 * lr;
            lrKf1 /= lrKf1.at<float>(2, 0);

            for (int i2 = i1 + 1; i2 < kfs.size(); i2++)
            {
                WAIKeyFrame* kfCand2 = kfs[i2];

                if (kfCand2->isBad()) continue;

                cv::Mat homography2;
                if (findMarkerHomography(markerFrame, kfCand2, homography2, 50))
                {
                    // 3.b Calculate position of the markers cornerpoints on second keyframe in 2D
                    // NOTE(dgj1): assumption that intrinsic camera parameters are the same
                    // TODO(dgj1): think about this assumption
                    ulKf2 = homography2 * ul;
                    ulKf2 /= ulKf2.at<float>(2, 0);
                    urKf2 = homography2 * ur;
                    urKf2 /= urKf2.at<float>(2, 0);
                    llKf2 = homography2 * ll;
                    llKf2 /= llKf2.at<float>(2, 0);
                    lrKf2 = homography2 * lr;
                    lrKf2 /= lrKf2.at<float>(2, 0);

                    // 4. Triangulate position of the markers cornerpoints
                    cv::Mat Rcw1 = kfCand1->GetRotation();
                    cv::Mat Rwc1 = Rcw1.t();
                    cv::Mat tcw1 = kfCand1->GetTranslation();
                    cv::Mat Tcw1(3, 4, CV_32F);
                    Rcw1.copyTo(Tcw1.colRange(0, 3));
                    tcw1.copyTo(Tcw1.col(3));

                    const float& fx1    = kfCand1->fx;
                    const float& fy1    = kfCand1->fy;
                    const float& cx1    = kfCand1->cx;
                    const float& cy1    = kfCand1->cy;
                    const float& invfx1 = kfCand1->invfx;
                    const float& invfy1 = kfCand1->invfy;

                    cv::Mat Rcw2 = kfCand2->GetRotation();
                    cv::Mat Rwc2 = Rcw2.t();
                    cv::Mat tcw2 = kfCand2->GetTranslation();
                    cv::Mat Tcw2(3, 4, CV_32F);
                    Rcw2.copyTo(Tcw2.colRange(0, 3));
                    tcw2.copyTo(Tcw2.col(3));

                    const float& fx2    = kfCand2->fx;
                    const float& fy2    = kfCand2->fy;
                    const float& cx2    = kfCand2->cx;
                    const float& cy2    = kfCand2->cy;
                    const float& invfx2 = kfCand2->invfx;
                    const float& invfy2 = kfCand2->invfy;

                    {
                        cv::Mat ul1 = (cv::Mat_<float>(3, 1) << (ulKf1.at<float>(0, 0) - cx1) * invfx1, (ulKf1.at<float>(1, 0) - cy1) * invfy1, 1.0);
                        cv::Mat ul2 = (cv::Mat_<float>(3, 1) << (ulKf2.at<float>(0, 0) - cx2) * invfx2, (ulKf2.at<float>(1, 0) - cy2) * invfy2, 1.0);

                        // Linear Triangulation Method
                        cv::Mat A(4, 4, CV_32F);
                        A.row(0) = ul1.at<float>(0) * Tcw1.row(2) - Tcw1.row(0);
                        A.row(1) = ul1.at<float>(1) * Tcw1.row(2) - Tcw1.row(1);
                        A.row(2) = ul2.at<float>(0) * Tcw2.row(2) - Tcw2.row(0);
                        A.row(3) = ul2.at<float>(1) * Tcw2.row(2) - Tcw2.row(1);

                        cv::Mat w, u, vt;
                        cv::SVD::compute(A, w, u, vt, cv::SVD::MODIFY_A | cv::SVD::FULL_UV);

                        ul3D = vt.row(3).t();

                        if (ul3D.at<float>(3) != 0)
                        {
                            // Euclidean coordinates
                            ul3D = ul3D.rowRange(0, 3) / ul3D.at<float>(3);
                        }
                    }

                    {
                        cv::Mat ur1 = (cv::Mat_<float>(3, 1) << (urKf1.at<float>(0, 0) - cx1) * invfx1, (urKf1.at<float>(1, 0) - cy1) * invfy1, 1.0);
                        cv::Mat ur2 = (cv::Mat_<float>(3, 1) << (urKf2.at<float>(0, 0) - cx2) * invfx2, (urKf2.at<float>(1, 0) - cy2) * invfy2, 1.0);

                        // Linear Triangulation Method
                        cv::Mat A(4, 4, CV_32F);
                        A.row(0) = ur1.at<float>(0) * Tcw1.row(2) - Tcw1.row(0);
                        A.row(1) = ur1.at<float>(1) * Tcw1.row(2) - Tcw1.row(1);
                        A.row(2) = ur2.at<float>(0) * Tcw2.row(2) - Tcw2.row(0);
                        A.row(3) = ur2.at<float>(1) * Tcw2.row(2) - Tcw2.row(1);

                        cv::Mat w, u, vt;
                        cv::SVD::compute(A, w, u, vt, cv::SVD::MODIFY_A | cv::SVD::FULL_UV);

                        ur3D = vt.row(3).t();

                        if (ur3D.at<float>(3) != 0)
                        {
                            // Euclidean coordinates
                            ur3D = ur3D.rowRange(0, 3) / ur3D.at<float>(3);
                        }
                    }

                    {
                        cv::Mat ll1 = (cv::Mat_<float>(3, 1) << (llKf1.at<float>(0, 0) - cx1) * invfx1, (llKf1.at<float>(1, 0) - cy1) * invfy1, 1.0);
                        cv::Mat ll2 = (cv::Mat_<float>(3, 1) << (llKf2.at<float>(0, 0) - cx2) * invfx2, (llKf2.at<float>(1, 0) - cy2) * invfy2, 1.0);

                        // Linear Triangulation Method
                        cv::Mat A(4, 4, CV_32F);
                        A.row(0) = ll1.at<float>(0) * Tcw1.row(2) - Tcw1.row(0);
                        A.row(1) = ll1.at<float>(1) * Tcw1.row(2) - Tcw1.row(1);
                        A.row(2) = ll2.at<float>(0) * Tcw2.row(2) - Tcw2.row(0);
                        A.row(3) = ll2.at<float>(1) * Tcw2.row(2) - Tcw2.row(1);

                        cv::Mat w, u, vt;
                        cv::SVD::compute(A, w, u, vt, cv::SVD::MODIFY_A | cv::SVD::FULL_UV);

                        ll3D = vt.row(3).t();

                        if (ll3D.at<float>(3) != 0)
                        {
                            // Euclidean coordinates
                            ll3D = ll3D.rowRange(0, 3) / ll3D.at<float>(3);
                        }
                    }

                    {
                        cv::Mat lr1 = (cv::Mat_<float>(3, 1) << (lrKf1.at<float>(0, 0) - cx1) * invfx1, (lrKf1.at<float>(1, 0) - cy1) * invfy1, 1.0);
                        cv::Mat lr2 = (cv::Mat_<float>(3, 1) << (lrKf2.at<float>(0, 0) - cx2) * invfx2, (lrKf2.at<float>(1, 0) - cy2) * invfy2, 1.0);

                        // Linear Triangulation Method
                        cv::Mat A(4, 4, CV_32F);
                        A.row(0) = lr1.at<float>(0) * Tcw1.row(2) - Tcw1.row(0);
                        A.row(1) = lr1.at<float>(1) * Tcw1.row(2) - Tcw1.row(1);
                        A.row(2) = lr2.at<float>(0) * Tcw2.row(2) - Tcw2.row(0);
                        A.row(3) = lr2.at<float>(1) * Tcw2.row(2) - Tcw2.row(1);

                        cv::Mat w, u, vt;
                        cv::SVD::compute(A, w, u, vt, cv::SVD::MODIFY_A | cv::SVD::FULL_UV);

                        lr3D = vt.row(3).t();

                        if (lr3D.at<float>(3) != 0)
                        {
                            // Euclidean coordinates
                            lr3D = lr3D.rowRange(0, 3) / lr3D.at<float>(3);
                        }
                    }

                    AC = ll3D - ul3D;
                    AB = ur3D - ul3D;

                    cv::Vec3f vAC = AC;
                    cv::Vec3f vAB = AB;

                    cv::Vec3f vn = vAB.cross(vAC);
                    n            = cv::Mat(vn);

                    cv::Mat   AD  = lr3D - ul3D;
                    cv::Vec3f vAD = AD;

                    float d = cv::norm(vn.dot(vAD)) / cv::norm(vn);
                    if (d < 0.01f)
                    {
                        matchedKf1 = kfCand1;
                        matchedKf2 = kfCand2;

                        break;
                    }
                }
            }
        }

        if (matchedKf2) break;
    }

    if (!matchedKf1 || !matchedKf2)
    {
        return false;
    }

    // 5. Cull mappoints outside of marker
    std::vector<WAIMapPoint*> mapPoints = map->GetAllMapPoints();

    cv::Mat system = cv::Mat::zeros(3, 3, CV_32F);
    AC.copyTo(system.rowRange(0, 3).col(0));
    AB.copyTo(system.rowRange(0, 3).col(1));
    n.copyTo(system.rowRange(0, 3).col(2));

    cv::Mat systemInv = system.inv();

    for (int i = 0; i < mapPoints.size(); i++)
    {
        WAIMapPoint* mp = mapPoints[i];

        if (mp->isBad()) continue;

        cv::Mat sol = systemInv * (mp->GetWorldPos() - ul3D);

        if (sol.at<float>(0, 0) < 0 || sol.at<float>(0, 0) > 1 ||
            sol.at<float>(1, 0) < 0 || sol.at<float>(1, 0) > 1 ||
            sol.at<float>(2, 0) < -0.1f || sol.at<float>(2, 0) > 0.1f)
        {
            mp->SetBadFlag();
        }
    }

    for (int i = 0; i < kfs.size(); i++)
    {
        WAIKeyFrame* kf = kfs[i];

        if (kf->mnId == 0 || kf->isBad()) continue;

        int mpCount = 0;

        std::vector<WAIMapPoint*> mps = kf->GetMapPointMatches();
        for (int j = 0; j < mps.size(); j++)
        {
            WAIMapPoint* mp = mps[j];

            if (!mp || mp->isBad()) continue;

            mpCount++;
        }

        if (mpCount <= 0)
        {
            kf->SetBadFlag();
        }
    }

    cv::Mat systemNorm               = cv::Mat::zeros(3, 3, CV_32F);
    systemNorm.rowRange(0, 3).col(0) = system.rowRange(0, 3).col(1) / cv::norm(AB);
    systemNorm.rowRange(0, 3).col(1) = system.rowRange(0, 3).col(0) / cv::norm(AC);
    systemNorm.rowRange(0, 3).col(2) = system.rowRange(0, 3).col(2) / cv::norm(n);

    cv::Mat systemNormInv = systemNorm.inv();

    cv::Mat nodeTransform = cv::Mat::eye(4, 4, CV_32F);
    cv::Mat ul3Dinv       = -systemNormInv * ul3D;
    ul3Dinv.copyTo(nodeTransform.rowRange(0, 3).col(3));
    systemNormInv.copyTo(nodeTransform.rowRange(0, 3).colRange(0, 3));

    cv::Mat scaleMat         = cv::Mat::eye(4, 4, CV_32F);
    float   markerWidthInRef = cv::norm(ul3D - ur3D);
    float   scaleFactor      = markerWidthInM / markerWidthInRef;
    scaleMat.at<float>(0, 0) = scaleFactor;
    scaleMat.at<float>(1, 1) = scaleFactor;
    scaleMat.at<float>(2, 2) = scaleFactor;

    nodeTransform = scaleMat * nodeTransform;

    if (_mpUL)
    {
        delete _mpUL;
        _mpUL = nullptr;
    }
    if (_mpUR)
    {
        delete _mpUR;
        _mpUR = nullptr;
    }
    if (_mpLL)
    {
        delete _mpLL;
        _mpLL = nullptr;
    }
    if (_mpLR)
    {
        delete _mpLR;
        _mpLR = nullptr;
    }

    _mpUL = new WAIMapPoint(0, ul3D, nullptr, false);
    _mpUR = new WAIMapPoint(0, ur3D, nullptr, false);
    _mpLL = new WAIMapPoint(0, ll3D, nullptr, false);
    _mpLR = new WAIMapPoint(0, lr3D, nullptr, false);

    mapNode.om(WAIMapStorage::convertToSLMat(nodeTransform));

    saveMap(waiMode.get(), mapDir, mapFile, &mapNode);

    return true;
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

void MapCreator::decorateDebug(WAI::ModeOrbSlam2* waiMode, CVCapture* cap, const int currentFrameIndex, const int videoLength, const int numOfKfs)
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

        cv::putText(decoImg, "FrameId: (" + std::to_string(currentFrameIndex) + "/" + std::to_string(videoLength) + ")", idxOff, 0, fontScale, color);
        cv::putText(decoImg, "Num Kfs: " + std::to_string(numOfKfs), kfsOff, 0, fontScale, color);
        cv::imshow("lastFrame", decoImg);
        cv::waitKey(1);
    }
    //#endif
}

void MapCreator::saveMap(WAI::ModeOrbSlam2* waiMode,
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
        throw std::runtime_error("Could not save map file: " + mapDir + currentMapFileName);
    }

    waiMode->resume();
}

void MapCreator::loadMap(WAI::ModeOrbSlam2* waiMode,
                         const std::string& mapDir,
                         const std::string& currentMapFileName,
                         bool               fixKfsForLBA,
                         SLNode*            mapNode)
{
    waiMode->requestStateIdle();
    while (!waiMode->hasStateIdle())
    {
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    waiMode->reset();

    bool mapLoadingSuccess = WAIMapStorage::loadMap(waiMode->getMap(),
                                                    waiMode->getKfDB(),
                                                    mapNode,
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
