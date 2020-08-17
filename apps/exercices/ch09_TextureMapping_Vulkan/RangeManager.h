#ifndef RANGEMANAGER_H
#define RANGEMANAGER_H

#include <memory>

class RangeManager
{
public:
    RangeManager(int size);
    ~RangeManager();

    void add(const int index);
    void add(const int index, const int amount);

    const int getMin(const int index) { return _min[index]; }
    const int getMax(const int index) { return _max[index]; }
    const int getDiff(const int index) { return _max[index] - _min[index]; }

private:
    int* _min  = nullptr;
    int* _max  = nullptr;
    int  _size = 0;
};

#endif
