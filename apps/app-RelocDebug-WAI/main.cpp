
#include <GL/gl3w.h>    // OpenGL headers
#include <GLFW/glfw3.h> // GLFW GUI library
#include <GLFW/glfw3.h>
#include <string>
#include <iostream>
#include <Utils.h>
#include <memory>
#include <vector>
#include <GLFW/glfw3.h>
#include <SENSVideoStream.h>
#include <WAISlamTools.h>
#include <FeatureExtractorFactory.h>
#include <AppWAISlamParamHelper.h>
#include <Utils.h>
#include <WAIMapStorage.h>
#include <cv/CVCamera.h>
#include <GLSLextractor.h>
#include <FeatureExtractorFactory.h>

//app parameter
struct Config
{
    std::string   erlebARDir;
    std::string   calibrationsDir;
    std::string   calibrationFile;
    std::string   vocFile;
    std::string   videoFile;
    std::string   map1File;
    std::string   map2File;
    ExtractorType extractorType;
    int           nLevels;
};

void printHelp()
{
    std::stringstream ss;
    ss << "app-RelocDebug for creation of Erleb-AR maps!" << std::endl;
    ss << "Example1 (win):  app-MapCreator.exe -erlebARDir C:/Erleb-AR -configFile MapCreatorConfig.json -mapOutputDir output" << std::endl;
    ss << "Example2 (unix): ./app-MapCreator -erlebARDir C:/Erleb-AR -configFile MapCreatorConfig.json -mapOutputDir output" << std::endl;
    ss << "" << std::endl;
    ss << "Options: " << std::endl;
    ss << "  -h/-help              print this help, e.g. -h" << std::endl;
    ss << "  -erlebARDir           Path to Erleb-AR root directory (Optional. If not specified, <AppsWritableDir>/erleb-AR/ is used)" << std::endl;
    ss << "  -calibDir             Path to directory containing camera calibrations (Optional. If not specified, <AppsWritableDir>/calibrations/ is used)" << std::endl;
    ss << "  -vocFile              Path and name to Vocabulary file (Optional. If not specified, <AppsWritableDir>/voc/voc_fbow.bin is used)" << std::endl;
    ss << "  -level                Number of pyramid levels" << std::endl;

    std::cout << ss.str() << std::endl;
}

void readArgs(int argc, char* argv[], Config& config)
{
    config.extractorType   = ExtractorType_FAST_ORBS_1000;
    config.erlebARDir      = Utils::getAppsWritableDir() + "erleb-AR/";
    config.calibrationsDir = Utils::getAppsWritableDir() + "erleb-AR/calibrations/";
    config.nLevels         = -1;

#if USE_FBOW
    config.vocFile = Utils::getAppsWritableDir() + "voc/voc_fbow.bin";
#else
    config.vocFile = Utils::getAppsWritableDir() + "voc/ORBvoc.bin";
#endif

    for (int i = 1; i < argc; ++i)
    {
        if (!strcmp(argv[i], "-calibFile"))
        {
            config.calibrationFile = argv[++i];
            //config.calibrationsDir = Utils::
            //config.calibrationsFile = Utils::
        }
        else if (!strcmp(argv[i], "-vocFile"))
        {
            config.vocFile = argv[++i];
        }
        else if (!strcmp(argv[i], "-level"))
        {
            config.nLevels = std::stoi(argv[++i]);
        }
        else if (!strcmp(argv[i], "-feature"))
        {
            i++;
            if (!strcmp(argv[i], "FAST_BRIEF_1000"))
                config.extractorType = ExtractorType_FAST_BRIEF_1000;
            else if (!strcmp(argv[i], "FAST_BRIEF_2000"))
                config.extractorType = ExtractorType_FAST_BRIEF_2000;
            else if (!strcmp(argv[i], "FAST_BRIEF_3000"))
                config.extractorType = ExtractorType_FAST_BRIEF_3000;
            else if (!strcmp(argv[i], "FAST_BRIEF_4000"))
                config.extractorType = ExtractorType_FAST_BRIEF_4000;
            else if (!strcmp(argv[i], "FAST_ORBS_1000"))
                config.extractorType = ExtractorType_FAST_ORBS_1000;
            else if (!strcmp(argv[i], "FAST_ORBS_2000"))
                config.extractorType = ExtractorType_FAST_ORBS_2000;
            else if (!strcmp(argv[i], "FAST_ORBS_3000"))
                config.extractorType = ExtractorType_FAST_ORBS_3000;
            else if (!strcmp(argv[i], "FAST_ORBS_4000"))
                config.extractorType = ExtractorType_FAST_ORBS_4000;
            else
                std::cout << "deduce feature type from map name" << std::endl;
        }
        else if (!strcmp(argv[i], "-video"))
        {
            config.videoFile = argv[++i];
        }
        else if (!strcmp(argv[i], "-map1"))
        {
            config.map1File = argv[++i];
        }
        else if (!strcmp(argv[i], "-map2"))
        {
            config.map2File = argv[++i];
        }
        else if (!strcmp(argv[i], "-h") || !strcmp(argv[i], "-help"))
        {
            printHelp();
        }
    }
    if (config.nLevels == -1)
    {
        std::cerr << "pyramid level not specified" << std::endl;
        exit(1);
    }

    SlamVideoInfos vi;
    //SlamMapInfos map1Infos;
    //SlamMapInfos map2Infos;

    extractSlamVideoInfosFromFileName(config.videoFile, &vi);
    if (config.calibrationFile.empty())
        config.calibrationFile = "camCalib_" + vi.deviceString + "_main.xml";

    //extractSlamMapInfosFromFileName(config.map1File, &map1Infos);
    //extractSlamMapInfosFromFileName(config.map2File, &map2Infos);

    std::cout << "=====" << std::endl;
    std::cout << "calibration path : " << config.calibrationsDir << config.calibrationFile << std::endl;
    std::cout << "map1 : " << config.map1File << std::endl;
    std::cout << "map2 : " << config.map2File << std::endl;
    std::string s = FeatureExtractorFactory().getExtractorIdToNames()[config.extractorType];
    std::cout << "feature type = " << s << " with " << config.nLevels << " levels" << std::endl;
    std::cout << "voc path : " << config.vocFile << std::endl;
    std::cout << "video path : " << config.videoFile << std::endl;
    std::cout << "=====" << std::endl;
}

struct InfoDebug
{
    int     nSharingWordKF    = 0;
    int     maxCommonWords    = 0;
    int     nKFMinCommonWords = 0;
    int     vpCandidates      = 0;
    int     nCandidates       = 0;
    int     nGood1            = 0;
    int     nadditional1      = 0;
    int     nGood2            = 0;
    int     nadditional2      = 0;
    int     nGood3            = 0;
    int     bMatch            = 0;
    int     commonWidCounter  = 0;
    cv::Mat pose1;
    cv::Mat pose2;
    cv::Mat pose3;

    std::vector<int>          nmatches;
    std::vector<WAIKeyFrame*> cKF;
};

bool relocalization(WAIFrame&         frame,
                    WAIKeyFrameDB*    kfdb,
                    WAIOrbVocabulary* voc,
                    LocalMap&         localMap,
                    int&              inliers,
                    InfoDebug*        info);

std::vector<WAIKeyFrame*> DetectRelocalizationCandidates(WAIKeyFrameDB* kfdb, WAIFrame* F, WAIOrbVocabulary* voc, bool applyMinAccScoreFilter, InfoDebug* info);

int SearchByBoW(WAIKeyFrame* pKF, WAIFrame& F, vector<WAIMapPoint*>& vpMapPointMatches, float nnratio, InfoDebug* info);

int main(int argc, char* argv[])
{
    Config config;

    //parse arguments
    readArgs(argc, argv, config);

    //initialize logger
    std::string cwd = Utils::getCurrentWorkingDir();

    SENSVideoStream vStream = SENSVideoStream(config.videoFile, false, false, false);
    {
        SENSCalibration calibration = SENSCalibration(config.calibrationsDir, Utils::getFileName(config.calibrationFile), true);
        vStream.setCalibration(calibration, true);
    }
    const SENSCalibration* calibration    = vStream.calibration();
    cv::Size2i             videoFrameSize = vStream.getFrameSize();

    FeatureExtractorFactory      extractorF        = FeatureExtractorFactory();
    std::unique_ptr<KPextractor> trackingExtractor = extractorF.make(config.extractorType, videoFrameSize, config.nLevels);

    WAIOrbVocabulary* voc = new WAIOrbVocabulary();
    voc->loadFromFile(config.vocFile);

    WAIKeyFrame::nNextId            = 0;
    WAIFrame::nNextId               = 0;
    WAIFrame::mbInitialComputations = true;
    WAIMapPoint::nNextId            = 0;

    WAIKeyFrameDB* kfdb1 = new WAIKeyFrameDB(voc);
    WAIKeyFrameDB* kfdb2 = new WAIKeyFrameDB(voc);
    WAIMap*        map1  = new WAIMap(kfdb1);
    WAIMap*        map2  = new WAIMap(kfdb2);

    cv::Mat map1nodeOm, map2nodeOm;

    WAIMapStorage::loadMap(map1,
                           map1nodeOm,
                           voc,
                           config.map1File,
                           false,
                           true);

    WAIMapStorage::loadMap(map2,
                           map2nodeOm,
                           voc,
                           config.map2File,
                           false,
                           true);

    for (int i = 0; i < map1->GetAllKeyFrames().size(); i++)
    {
        WAIKeyFrame* kf = map1->GetAllKeyFrames()[i];
    }

    int frameCounter = 0;

    while (1)
    {
        SENSFramePtr sensFrame = vStream.grabNextFrame();
        if (sensFrame == nullptr)
            break;

        cv::Mat intrinsic  = calibration->cameraMat();
        cv::Mat distortion = calibration->distortion();

        WAIFrame frame = WAIFrame(sensFrame.get()->imgManip,
                                  0.0,
                                  trackingExtractor.get(),
                                  intrinsic,
                                  distortion,
                                  voc,
                                  false);

        frame.ComputeBoW();

        int             inliers1 = 0;
        int             inliers2 = 0;
        struct LocalMap localMap1;
        struct LocalMap localMap2;

        InfoDebug info1;
        InfoDebug info2;


        DUtils::Random::SeedRand(42);
        bool b1 = relocalization(frame, kfdb1, voc, localMap1, inliers1, &info1);

        DUtils::Random::SeedRand(42);
        bool b2 = relocalization(frame, kfdb2, voc, localMap2, inliers2, &info2);

        //if (b1 != b2 || inliers1 != inliers2)
        {
            std::cout << std::endl;
            std::cout << std::endl;
            std::cout << "map1 : " << std::endl;
            std::cout << "  nSharingWordKF " << info1.nSharingWordKF << std::endl;
            std::cout << "  maxCommonWords " << info1.maxCommonWords << std::endl;
            std::cout << "  nKFMinCommonWords " << info1.nKFMinCommonWords << std::endl;
            std::cout << "  vpCandidates " << info1.vpCandidates << std::endl;
            std::cout << "  commonWidCounter " << info1.commonWidCounter << std::endl;
            std::cout << "  nCandidates " << info1.nCandidates << std::endl;
            std::cout << "  nGood1 " << info1.nGood1 << std::endl;
            std::cout << "  nadditional1 " << info1.nadditional1 << std::endl;
            std::cout << "  nGood2 " << info1.nGood2 << std::endl;
            std::cout << "  nadditional2 " << info1.nadditional2 << std::endl;
            std::cout << "  nGood3 " << info1.nGood3 << std::endl;
            std::cout << "  bMatch " << info1.bMatch << std::endl;
            std::cout << "  pose 1 " << std::endl
                      << info1.pose1 << std::endl;
            std::cout << "  pose 2 " << std::endl
                      << info1.pose2 << std::endl;
            std::cout << "  pose 3 " << std::endl
                      << info1.pose3 << std::endl;

            std::cout << "  nmatches " << std::endl;
            std::cout << "    ";
            for (int i = 0; i < info1.nmatches.size(); i++)
                std::cout << info1.nmatches[i] << " ";
            std::cout << std::endl;
            std::cout << "  ckf ptrs " << std::endl;
            std::cout << "    ";
            for (int i = 0; i < info1.cKF.size(); i++)
                std::cout << info1.cKF[i]->mnId << " ";
            std::cout << std::endl;

            std::cout << std::endl;
            std::cout << "map2 : " << std::endl;
            std::cout << "  nSharingWordKF " << info2.nSharingWordKF << std::endl;
            std::cout << "  maxCommonWords " << info2.maxCommonWords << std::endl;
            std::cout << "  nKFMinCommonWords " << info2.nKFMinCommonWords << std::endl;
            std::cout << "  vpCandidates " << info2.vpCandidates << std::endl;
            std::cout << "  commonWidCounter " << info2.commonWidCounter << std::endl;
            std::cout << "  nCandidates " << info2.nCandidates << std::endl;
            std::cout << "  nGood1 " << info2.nGood1 << std::endl;
            std::cout << "  nadditional1 " << info2.nadditional1 << std::endl;
            std::cout << "  nGood2 " << info2.nGood2 << std::endl;
            std::cout << "  nadditional2 " << info2.nadditional2 << std::endl;
            std::cout << "  nGood3 " << info2.nGood3 << std::endl;
            std::cout << "  bMatch " << info2.bMatch << std::endl;
            std::cout << "  pose 1 " << std::endl
                      << info2.pose1 << std::endl;
            std::cout << "  pose 2 " << std::endl
                      << info2.pose2 << std::endl;
            std::cout << "  pose 3 " << std::endl
                      << info2.pose3 << std::endl;
            std::cout << "  nmatches " << std::endl;
            std::cout << "    ";
            for (int i = 0; i < info2.nmatches.size(); i++)
                std::cout << info2.nmatches[i] << " ";
            std::cout << std::endl;
            std::cout << "  ckf ptrs " << std::endl;
            std::cout << "    ";
            for (int i = 0; i < info2.cKF.size(); i++)
                std::cout << info2.cKF[i]->mnId << " ";
            std::cout << std::endl;
        }
    }

    return 0;
}

bool relocalization(WAIFrame&         frame,
                    WAIKeyFrameDB*    kfdb,
                    WAIOrbVocabulary* voc,
                    LocalMap&         localMap,
                    int&              inliers,
                    InfoDebug*        info)
{
    vector<WAIKeyFrame*> vpCandidateKFs;
    vpCandidateKFs = DetectRelocalizationCandidates(kfdb, &frame, voc, true, info);
    info->cKF      = vpCandidateKFs;

    //INFO
    info->vpCandidates = vpCandidateKFs.size();

    if (vpCandidateKFs.empty())
        return false;

    const int nKFs = (int)vpCandidateKFs.size();

    ORBmatcher matcher(0.75, true);

    vector<PnPsolver*> vpPnPsolvers;
    vpPnPsolvers.resize(nKFs);

    vector<vector<WAIMapPoint*>> vvpMapPointMatches;
    vvpMapPointMatches.resize(nKFs);

    std::vector<bool> outliers;

    vector<bool> vbDiscarded;
    vbDiscarded.resize(nKFs);

    int nCandidates = 0;

    for (int i = 0; i < nKFs; i++)
    {
        WAIKeyFrame* pKF = vpCandidateKFs[i];
        if (pKF->isBad())
            vbDiscarded[i] = true;
        else
        {
            int nmatches = SearchByBoW(pKF, frame, vvpMapPointMatches[i], 0.75, info);
            //INFO
            info->nmatches.push_back(nmatches);

            if (nmatches < 15)
            {
                vbDiscarded[i] = true;
                continue;
            }
            else
            {
                PnPsolver* pSolver = new PnPsolver(frame, vvpMapPointMatches[i]);
                pSolver->SetRansacParameters(0.99, 10, 300, 4, 0.5f, 5.991f);
                vpPnPsolvers[i] = pSolver;
                nCandidates++;
            }
        }
    }

    //INFO
    info->nCandidates = nCandidates;

    bool       bMatch = false;
    ORBmatcher matcher2(0.9f, true);

    while (nCandidates > 0 && !bMatch)
    {
        for (int i = 0; i < nKFs; i++)
        {
            if (vbDiscarded[i])
                continue;

            // Perform 5 Ransac Iterations
            vector<bool> vbInliers;
            int          nInliers;
            bool         bNoMore;

            PnPsolver* pSolver = vpPnPsolvers[i];
            cv::Mat    Tcw     = pSolver->iterate(150, bNoMore, vbInliers, nInliers);

            // If Ransac reachs max. iterations discard keyframe
            if (bNoMore)
            {
                vbDiscarded[i] = true;
                nCandidates--;
            }

            // If a Camera Pose is computed, optimize
            if (!Tcw.empty())
            {
                Tcw.copyTo(frame.mTcw);

                set<WAIMapPoint*> sFound;

                const int np = (int)vbInliers.size();

                for (int j = 0; j < np; j++)
                {
                    if (vbInliers[j])
                    {
                        frame.mvpMapPoints[j] = vvpMapPointMatches[i][j];
                        sFound.insert(vvpMapPointMatches[i][j]);
                    }
                    else
                        frame.mvpMapPoints[j] = NULL;
                }

                int nGood = Optimizer::PoseOptimization(&frame, outliers);

                //INFO
                info->nGood1 = nGood;
                info->pose1  = frame.mTcw;

                if (nGood < 10)
                    continue;

                if (nGood < 50)
                {
                    int nadditional = matcher2.SearchByProjection(frame, vpCandidateKFs[i], sFound, 10, 100);

                    //INFO
                    info->nadditional1 = nadditional;

                    if (nadditional + nGood >= 50)
                    {
                        nGood = Optimizer::PoseOptimization(&frame, outliers);
                        //INFO
                        info->nGood2 = nGood;
                        info->pose2  = frame.mTcw;

                        if (nGood > 30 && nGood < 50)
                        {
                            sFound.clear();
                            for (int ip = 0; ip < frame.N; ip++)
                                if (frame.mvpMapPoints[ip] && !outliers[ip])
                                    sFound.insert(frame.mvpMapPoints[ip]);
                            nadditional = matcher2.SearchByProjection(frame, vpCandidateKFs[i], sFound, 3, 64);

                            //INFO
                            info->nadditional2 = nadditional;

                            // Final optimization
                            if (nGood + nadditional >= 50)
                            {
                                nGood = Optimizer::PoseOptimization(&frame);
                                //INFO
                                info->nGood3 = nGood;
                                info->pose3  = frame.mTcw;
                            }
                        }
                    }
                }
                if (nGood >= 50)
                {
                    bMatch = WAISlamTools::trackLocalMap(localMap, frame, frame.mnId, inliers);
                    break;
                }
            }
        }
    }

    //INFO
    info->bMatch = bMatch;

    return bMatch;
}

std::vector<WAIKeyFrame*> DetectRelocalizationCandidates(WAIKeyFrameDB* kfdb, WAIFrame* F, WAIOrbVocabulary* voc, bool applyMinAccScoreFilter, InfoDebug* info)
{
    std::list<WAIKeyFrame*> lKFsSharingWords;

    // Search all keyframes that share a word with current frame
    {
        int vitCounter = 0;
        for (auto vit = F->mBowVec.getWordScoreMapping().begin(), vend = F->mBowVec.getWordScoreMapping().end(); vit != vend; vit++)
        {
            std::list<WAIKeyFrame*>& lKFs = kfdb->getInvertedFile()[vit->first];

            for (std::list<WAIKeyFrame*>::iterator lit = lKFs.begin(), lend = lKFs.end(); lit != lend; lit++)
            {
                WAIKeyFrame* pKFi = *lit;
                if (pKFi->mnRelocQuery != F->mnId)
                {
                    pKFi->mnRelocWords = 0;
                    pKFi->mRelocScore  = 0.f;
                    pKFi->mnRelocQuery = F->mnId;
                    lKFsSharingWords.push_back(pKFi);
                }
                pKFi->mnRelocWords++;
            }
        }
    }

    //INFO
    info->nSharingWordKF = lKFsSharingWords.size();

    if (lKFsSharingWords.empty())
        return std::vector<WAIKeyFrame*>();

    // Only compare against those keyframes that share enough words
    int maxCommonWords = 0;
    for (std::list<WAIKeyFrame*>::iterator lit = lKFsSharingWords.begin(), lend = lKFsSharingWords.end(); lit != lend; lit++)
    {
        if ((*lit)->mnRelocWords > maxCommonWords)
            maxCommonWords = (*lit)->mnRelocWords;
    }

    //INFO
    info->maxCommonWords = maxCommonWords;

    int minCommonWords = (int)(maxCommonWords * 0.8f);

    {
        std::list<std::pair<float, WAIKeyFrame*>> lScoreAndMatch;
        int                                       nscores = 0;

        // Compute similarity score.
        for (std::list<WAIKeyFrame*>::iterator lit = lKFsSharingWords.begin(), lend = lKFsSharingWords.end(); lit != lend; lit++)
        {
            WAIKeyFrame* pKFi = *lit;

            if (pKFi->mnRelocWords > minCommonWords)
            {
                nscores++;
                float si          = (float)voc->score(F->mBowVec, pKFi->mBowVec);
                pKFi->mRelocScore = si;
                lScoreAndMatch.push_back(std::make_pair(si, pKFi));
            }
        }

        //INFO
        info->nKFMinCommonWords = lScoreAndMatch.size();

        if (lScoreAndMatch.empty())
            return std::vector<WAIKeyFrame*>();

        std::list<std::pair<float, WAIKeyFrame*>> lAccScoreAndMatch;
        float                                     bestAccScore = 0;

        for (std::list<std::pair<float, WAIKeyFrame*>>::iterator it = lScoreAndMatch.begin(), itend = lScoreAndMatch.end(); it != itend; it++)
        {
            WAIKeyFrame*              pKFi     = it->second;
            std::vector<WAIKeyFrame*> vpNeighs = pKFi->GetBestCovisibilityKeyFrames(10);

            float        bestScore = it->first;
            float        accScore  = bestScore;
            WAIKeyFrame* pBestKF   = pKFi;
            for (std::vector<WAIKeyFrame*>::iterator vit = vpNeighs.begin(), vend = vpNeighs.end(); vit != vend; vit++)
            {
                WAIKeyFrame* pKF2 = *vit;
                if (pKF2->mnRelocQuery != F->mnId && pKF2->mnRelocWords <= minCommonWords)
                    continue;

                accScore += pKF2->mRelocScore;
                if (pKF2->mRelocScore > bestScore)
                {
                    pBestKF   = pKF2;
                    bestScore = pKF2->mRelocScore;
                }
            }
            lAccScoreAndMatch.push_back(std::make_pair(accScore, pBestKF));
            if (accScore > bestAccScore)
                bestAccScore = accScore;
        }

        float                     minScoreToRetain = 0.75f * bestAccScore;
        std::set<WAIKeyFrame*>    spAlreadyAddedKF;
        std::vector<WAIKeyFrame*> vpRelocCandidates;
        vpRelocCandidates.reserve(lAccScoreAndMatch.size());
        for (std::list<std::pair<float, WAIKeyFrame*>>::iterator it = lAccScoreAndMatch.begin(), itend = lAccScoreAndMatch.end(); it != itend; it++)
        {
            const float& si = it->first;
            if (si > minScoreToRetain)
            {
                WAIKeyFrame* pKFi = it->second;
                if (!spAlreadyAddedKF.count(pKFi))
                {
                    vpRelocCandidates.push_back(pKFi);
                    spAlreadyAddedKF.insert(pKFi);
                }
            }
        }

        return vpRelocCandidates;
    }
}
const int TH_HIGH      = 100;
const int TH_LOW       = 50;
const int HISTO_LENGTH = 30;

int DescriptorDistance(const cv::Mat& a, const cv::Mat& b)
{
    const int* pa = a.ptr<int32_t>();
    const int* pb = b.ptr<int32_t>();

    int dist = 0;

    for (int i = 0; i < 8; i++, pa++, pb++)
    {
        unsigned int v = *pa ^ *pb;
        v              = v - ((v >> 1) & 0x55555555);
        v              = (v & 0x33333333) + ((v >> 2) & 0x33333333);
        dist += (((v + (v >> 4)) & 0xF0F0F0F) * 0x1010101) >> 24;
    }

    return dist;
}

void ComputeThreeMaxima(vector<int>* histo, const int L, int& ind1, int& ind2, int& ind3)
{
    int max1 = 0;
    int max2 = 0;
    int max3 = 0;

    for (int i = 0; i < L; i++)
    {
        const int s = (int)histo[i].size();
        if (s > max1)
        {
            max3 = max2;
            max2 = max1;
            max1 = s;
            ind3 = ind2;
            ind2 = ind1;
            ind1 = i;
        }
        else if (s > max2)
        {
            max3 = max2;
            max2 = s;
            ind3 = ind2;
            ind2 = i;
        }
        else if (s > max3)
        {
            max3 = s;
            ind3 = i;
        }
    }

    if (max2 < 0.1f * (float)max1)
    {
        ind2 = -1;
        ind3 = -1;
    }
    else if (max3 < 0.1f * (float)max1)
    {
        ind3 = -1;
    }
}

int SearchByBoW(WAIKeyFrame* pKF, WAIFrame& F, vector<WAIMapPoint*>& vpMapPointMatches, float nnratio, InfoDebug* info)
{
    const vector<WAIMapPoint*> vpMapPointsKF = pKF->GetMapPointMatches();

    vpMapPointMatches = vector<WAIMapPoint*>(F.N, static_cast<WAIMapPoint*>(NULL));

    WAIFeatVector& vFeatVecKF = pKF->mFeatVec;

    int nmatches = 0;

    vector<int> rotHist[HISTO_LENGTH];

    if (true)
    {
        for (int i = 0; i < HISTO_LENGTH; i++)
            rotHist[i].reserve(500);
    }
    const float factor = 1.0f / HISTO_LENGTH;

    // We perform the matching over ORB that belong to the same vocabulary node (at a certain level)
    auto KFit  = vFeatVecKF.getFeatMapping().begin();
    auto Fit   = F.mFeatVec.getFeatMapping().begin();
    auto KFend = vFeatVecKF.getFeatMapping().end();
    auto Fend  = F.mFeatVec.getFeatMapping().end();

    while (KFit != KFend && Fit != Fend)
    {
        if (KFit->first == Fit->first)
        {
            const vector<unsigned int> vIndicesKF = KFit->second;
            const vector<unsigned int> vIndicesF  = Fit->second;

            for (size_t iKF = 0; iKF < vIndicesKF.size(); iKF++)
            {
                const unsigned int realIdxKF = vIndicesKF[iKF];

                WAIMapPoint* pMP = vpMapPointsKF[realIdxKF];

                if (!pMP)
                    continue;

                if (pMP->isBad())
                    continue;

                info->commonWidCounter++;

                const cv::Mat& dKF = pKF->mDescriptors.row(realIdxKF);

                int bestDist1 = 256;
                int bestIdxF  = -1;
                int bestDist2 = 256;

                for (size_t iF = 0; iF < vIndicesF.size(); iF++)
                {
                    const unsigned int realIdxF = vIndicesF[iF];

                    if (vpMapPointMatches[realIdxF])
                        continue;

                    const cv::Mat& dF = F.mDescriptors.row(realIdxF);

                    const int dist = DescriptorDistance(dKF, dF);

                    if (dist < bestDist1)
                    {
                        bestDist2 = bestDist1;
                        bestDist1 = dist;
                        bestIdxF  = realIdxF;
                    }
                    else if (dist < bestDist2)
                    {
                        bestDist2 = dist;
                    }
                }

                if (bestDist1 <= TH_LOW)
                {
                    if (static_cast<float>(bestDist1) < nnratio * static_cast<float>(bestDist2))
                    {
                        vpMapPointMatches[bestIdxF] = pMP;

                        const cv::KeyPoint& kp = pKF->mvKeysUn[realIdxKF];

                        if (true)
                        {
                            float rot = kp.angle - F.mvKeys[bestIdxF].angle;
                            if (rot < 0.0)
                                rot += 360.0f;
                            int bin = (int)round(rot * factor);
                            if (bin == HISTO_LENGTH)
                                bin = 0;
                            assert(bin >= 0 && bin < HISTO_LENGTH);
                            rotHist[bin].push_back(bestIdxF);
                        }
                        nmatches++;
                    }
                }
            }

            KFit++;
            Fit++;
        }
        else if (KFit->first < Fit->first)
        {
            KFit = vFeatVecKF.getFeatMapping().lower_bound(Fit->first);
        }
        else
        {
            Fit = F.mFeatVec.getFeatMapping().lower_bound(KFit->first);
        }
    }

    if (true)
    {
        int ind1 = -1;
        int ind2 = -1;
        int ind3 = -1;

        ComputeThreeMaxima(rotHist, HISTO_LENGTH, ind1, ind2, ind3);

        for (int i = 0; i < HISTO_LENGTH; i++)
        {
            if (i == ind1 || i == ind2 || i == ind3)
                continue;
            for (size_t j = 0, jend = rotHist[i].size(); j < jend; j++)
            {
                vpMapPointMatches[rotHist[i][j]] = static_cast<WAIMapPoint*>(NULL);
                nmatches--;
            }
        }
    }

    return nmatches;
}