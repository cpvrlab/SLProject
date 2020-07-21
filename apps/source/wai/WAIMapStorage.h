#ifndef MAP_STORAGE
#define MAP_STORAGE

#include <SLSceneView.h>
#include <WAIHelper.h>
#include <WAISlam.h>
#include <fbow.h>
#include <Utils.h>

class WAI_API WAIMapStorage
{
public:
    static bool saveMap(WAIMap*     waiMap,
                        SLNode*     mapNode,
                        std::string fileName,
                        std::string imgDir = "");

    static bool saveMapRaw(WAIMap*     waiMap,
                           SLNode*     mapNode,
                           std::string fileName,
                           std::string imgDir = "");

    static bool loadMap(WAIMap*           waiMap,
                        SLNode*           mapNode,
                        WAIOrbVocabulary* voc,
                        std::string       path,
                        bool              loadImgs,
                        bool              fixKfsAndMPts);

    static cv::Mat convertToCVMat(const SLMat4f slMat);
    static SLMat4f convertToSLMat(const cv::Mat& cvMat);


    static void saveKeyFrameVideoMatching(std::vector<int>& keyFrameVideoMatching, std::vector<std::string> vidname, const std::string& mapDir, const std::string outputKFMatchingFile);
    static void loadKeyFrameVideoMatching(std::vector<int>& keyFrameVideoMatching, std::vector<std::string> &vidname, const std::string& mapDir, const std::string outputKFMatchingFile);
};

#endif
