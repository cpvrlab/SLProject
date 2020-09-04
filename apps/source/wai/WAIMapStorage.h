#ifndef MAP_STORAGE
#define MAP_STORAGE

#include <SLSceneView.h>
#include <WAIHelper.h>
#include <WAISlam.h>
#include <fbow.h>
#include <Utils.h>

class WAI_API WAIMapStorage
{
    struct MapInfo
    {
        int32_t version;
        int32_t kfCount, mpCount;
        // nodeOmSaved is used as boolean... sizeof(bool) is not defined in the c++ standard, thats why we have to use an int
        int32_t nodeOmSaved;
    };

    struct KeyFrameInfo
    {
        int32_t id;
        int32_t parentId;

        float   scaleFactor;
        int32_t scaleLevels;

        int32_t minX, minY, maxX, maxY;

        int32_t loopEdgesCount;
        int32_t kpCount;
        int32_t covisiblesCount;

        int32_t bowVecSize;
    };

    struct MapPointInfo
    {
        int32_t id;
        int32_t refKfId;

        int32_t nObervations;

        float minDistance, maxDistance;
    };

    struct KeyPointData
    {
        float   x, y;
        float   size;
        float   angle;
        float   response;
        int32_t octave;
        int32_t classId;
    };

public:
    static bool saveMap(WAIMap*     waiMap,
                        SLNode*     mapNode,
                        std::string fileName,
                        std::string imgDir  = "",
                        bool        saveBOW = true);

    static bool saveMapRaw(WAIMap*     waiMap,
                           SLNode*     mapNode,
                           std::string fileName,
                           std::string imgDir = "");

    static bool saveMapBinary(WAIMap*     waiMap,
                              SLNode*     mapNode,
                              std::string fileName,
                              std::string imgDir  = "",
                              bool        saveBOW = true);

    static bool loadMap(WAIMap*           waiMap,
                        cv::Mat&          mapNodeOm,
                        WAIOrbVocabulary* voc,
                        std::string       path,
                        bool              loadImgs,
                        bool              fixKfsAndMPts);

    static bool loadMapBinary(WAIMap*           waiMap,
                              cv::Mat&          mapNodeOm,
                              WAIOrbVocabulary* voc,
                              std::string       path,
                              bool              loadImgs,
                              bool              fixKfsAndMPts);

    static cv::Mat              convertToCVMat(const SLMat4f slMat);
    static SLMat4f              convertToSLMat(const cv::Mat& cvMat);
    static std::vector<uint8_t> convertCVMatToVector(const cv::Mat& mat);
    static void                 saveKeyFrameVideoMatching(std::vector<int>& keyFrameVideoMatching, std::vector<std::string> vidname, const std::string& mapDir, const std::string outputKFMatchingFile);
    static void                 loadKeyFrameVideoMatching(std::vector<int>& keyFrameVideoMatching, std::vector<std::string>& vidname, const std::string& mapDir, const std::string outputKFMatchingFile);

    template<typename T>
    static void writeVectorToBinaryFile(FILE* f, const std::vector<T> vec);
    template<typename T>
    static std::vector<T> loadVectorFromBinaryStream(uint8_t** data, int count);
    static void           writeCVMatToBinaryFile(FILE* f, const cv::Mat& mat);
    static cv::Mat        loadCVMatFromBinaryStream(uint8_t** data, int rows, int cols, int type);
};

#endif
