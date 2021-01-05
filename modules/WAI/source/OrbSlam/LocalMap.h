#ifndef LOCALMAP_H
#define LOCALMAP_H

#include <vector>
#include <WAIKeyFrame.h>
#include <WAIMapPoint.h>

struct LocalMap
{
    WAIKeyFrame*              refKF;
    std::vector<WAIKeyFrame*> keyFrames;
    std::vector<WAIMapPoint*> mapPoints;
    std::vector<WAIKeyFrame*> secondNeighbors;
};

#endif
