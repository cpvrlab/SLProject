#include <orb_slam/WorkingSet.h>
#include <Utils.h>

WorkingSet::WorkingSet(unsigned int maxBufferSize)
{
    mMaxBufferSize = maxBufferSize;
    inUse = new WAIKeyFrame*[maxBufferSize];
    for (unsigned int i = 0; i < maxBufferSize; i++)
    {
        inUse[i] = nullptr;
    }
}

void WorkingSet::reset()
{
    std::unique_lock<std::mutex> lock(toLocalAdjustmentMutex);
    std::unique_lock<std::mutex> lock2(useSetMutex);
    while (!toLocalAdjustment.empty())
    {
        toLocalAdjustment.pop();
    }

    for (unsigned int i = 0; i < mMaxBufferSize; i++)
    {
        inUse[i] = nullptr;
    }
}

void WorkingSet::addToLocalAdjustment(WAIKeyFrame* kf)
{
    std::unique_lock<std::mutex> lock(toLocalAdjustmentMutex);
    toLocalAdjustment.push(kf);
    lock.unlock();
}

int WorkingSet::popFromLocalAdjustment(WAIKeyFrame** kf)
{
    int kfInQueue;
    std::unique_lock<std::mutex> lock(toLocalAdjustmentMutex);
    kfInQueue = (int)toLocalAdjustment.size();
    if (kfInQueue == 0)
        return 0;
    *kf = toLocalAdjustment.front();
    toLocalAdjustment.pop();
    return kfInQueue;
}

void WorkingSet::addToUseSet(WAIKeyFrame* kf)
{
    std::unique_lock<std::mutex> lock(useSetMutex);
    for (unsigned int i = 0; i < mMaxBufferSize; i++)
    {
        if (inUse[i] == nullptr)
        {
            inUse[i] = kf;
            lock.unlock();
            return;
        }
    }
    Utils::log("Info", "AAAAA use set full");
}

void WorkingSet::removeFromUseSet(WAIKeyFrame* kf)
{
    std::unique_lock<std::mutex> lock(useSetMutex);
    for (unsigned int i = 0; i < mMaxBufferSize; i++)
    {
        if (inUse[i] == kf)
        {
            inUse[i] = nullptr;
            lock.unlock();
            return;
        }
    }
}

bool WorkingSet::isInUseSet(WAIKeyFrame* kf)
{
    std::unique_lock<std::mutex> lock(useSetMutex);

    for (unsigned int i = 0; i < mMaxBufferSize; i++)
    {
        if (inUse[i] == kf)
        {
            lock.unlock();
            return true;
        }
    }
    return false;
}
