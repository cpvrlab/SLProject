#include "RangeManager.h"
#include <assert.h>

RangeManager::RangeManager(const int size) : _size(size)
{
    _min = new int[_size];
    _max = new int[_size];

    for (int i = 0; i < _size; i++)
    {
        _min[i] = 0;
        _max[i] = 0;
    }
}

RangeManager::~RangeManager()
{
    delete[] _min;
    delete[] _max;

    _min = nullptr;
    _max = nullptr;
}

void RangeManager::add(const int index)
{
    add(index, 1);
}

void RangeManager::add(const int index, const int amount)
{
    assert(index < _size);

    _max[index] += amount;

    for (int i = (index + 1); i < _size; i++)
    {
        _min[i] += amount;
        _max[i] += amount;
    }
}
