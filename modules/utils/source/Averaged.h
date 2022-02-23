//#############################################################################
//  File:      Averaged.h
//  Authors:   Marcus Hudritsch
//  Date:      July 2014
//  Codestyle: https://github.com/cpvrlab/SLProject/wiki/SLProject-Coding-Style
//  Authors:   Marcus Hudritsch
//  License:   This software is provided under the GNU General Public License
//             Please visit: http://opensource.org/licenses/GPL-3.0
//#############################################################################

#ifndef AVERAGED_H
#define AVERAGED_H

#include <string>
#include <vector>
#include <assert.h>

using std::string;
using std::vector;

namespace Utils
{
//-----------------------------------------------------------------------------
//! Averaged template class provides an average value from a fixed size array.
/*!The Average template class provides a simple moving average value
 continuously averaged from a fixed size vector. The template class can be
 used for any template type T that provides the following operators:
 =, -, +, T* float
*/
template<class T>
class Averaged
{
public:
    Averaged() : _currentValueIndex(0), _sum(0), _average(0) {}
    Averaged(int numValues, T initValue = 0)
    {
        init(numValues, initValue);
    }

    //! Initializes the average value array to a given value
    void init(int numValues, T initValue)
    {
        assert(numValues > 0 && "Num. of values must be greater than zero");
        _values.clear();
        _values.resize(numValues, initValue);
        _oneOverNumValues  = 1.0f / (float)_values.size();
        _sum               = initValue * numValues;
        _average           = initValue;
        _currentValueIndex = 0;
    }

    //! Sets the current value in the value array and builds the average
    void set(T value)
    {
        assert(_values.size() > 0 && "_value vector not initialized");

        // Shortcut for no averaging
        if (_values.size() == 1)
            _sum = _average = value;
        else
        {
            if (_currentValueIndex == _values.size())
                _currentValueIndex = 0;

            // Correct the sum continuously
            _sum                        = _sum - _values[_currentValueIndex];
            _values[_currentValueIndex] = value;
            _sum                        = _sum + _values[_currentValueIndex];
            _average                    = _sum * _oneOverNumValues; // avoid division
            _currentValueIndex++;
        }
    }

    T      average() { return _average; }
    size_t size() { return _values.size(); }

private:
    float     _oneOverNumValues{};  //!< multiplier instead of divider
    vector<T> _values;              //!< value array
    int       _currentValueIndex{}; //!< current value index within _values
    T         _sum;                 //!< sum of all values
    T         _average;             //!< average value
};
//-----------------------------------------------------------------------------
typedef Utils::Averaged<float> AvgFloat;
//-----------------------------------------------------------------------------
};
#endif
