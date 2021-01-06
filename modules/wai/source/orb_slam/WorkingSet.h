#ifndef WORKING_SET
#define WORKING_SET

#include <list>
#include <condition_variable>
#include <mutex>
#include <queue>

#include <opencv2/core.hpp>
#include <orb_slam/ORBVocabulary.h>
#include <WAIKeyFrame.h>

struct WorkingSet 
{
    unsigned int mMaxBufferSize;
    WAIKeyFrame ** inUse;
    std::mutex useSetMutex;
    std::mutex toLocalAdjustmentMutex;
    std::queue<WAIKeyFrame*> toLocalAdjustment;

    WorkingSet(unsigned int maxBufferSize);

    void reset();

    void addToLocalAdjustment(WAIKeyFrame* kf);

    int popFromLocalAdjustment(WAIKeyFrame ** kf);

    void addToUseSet(WAIKeyFrame* kf);

    void removeFromUseSet(WAIKeyFrame* kf);

    bool isInUseSet(WAIKeyFrame* kf);
};

#endif
