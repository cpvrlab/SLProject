#include <WorkingSet.h>

WorkingSet::WorkingSet()
{
    for (int i = 0; i < 5; i++)
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
    kfInQueue = toLocalAdjustment.size();
    if (kfInQueue <= 0)
        return 0;
    *kf = toLocalAdjustment.front();
    toLocalAdjustment.pop();
    return kfInQueue;
}

void WorkingSet::addToUseSet(WAIKeyFrame* kf)
{
    std::unique_lock<std::mutex> lock(useSetMutex);
    for (int i = 0; i < 5; i++)
    {
        if (inUse[i] == nullptr)
        {
            inUse[i] = kf;
            lock.unlock();
            return;
        }
    }
    std::cout << "use set full" << std::endl;
}

void WorkingSet::removeFromUseSet(WAIKeyFrame* kf)
{
    std::unique_lock<std::mutex> lock(useSetMutex);
    for (int i = 0; i < 5; i++)
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

    for (int i = 0; i < 5; i++)
    {
        if (inUse[i] == kf)
        {
            lock.unlock();
            return true;
        }
    }
    return false;
}
