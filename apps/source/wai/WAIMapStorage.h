#ifndef MAP_STORAGE
#define MAP_STORAGE

#include <SLSceneView.h>
#include <WAIHelper.h>
#include <OrbSlam/ORBVocabulary.h>
#include <WAIModeOrbSlam2.h>
#include <Utils.h>

class WAI_API WAIMapStorage
{
public:
    static bool saveMap(WAIMap*     waiMap,
                        SLNode*     mapNode,
                        std::string fileName,
                        std::string imgDir = "");

    static bool loadMap(WAIMap*        waiMap,
                        WAIKeyFrameDB* kfDB,
                        SLNode*        mapNode,
                        std::string    path,
                        bool           loadImgs,
                        bool           fixKfsForLBA);

private:
    //static SLMat4f loadObjectMatrix(const cv::FileNode& n);
};

#endif
