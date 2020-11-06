//#############################################################################
//  File:      Averaged.h
//  Author:    Marcus Hudritsch
//  Date:      July 2014
//  Codestyle: https://github.com/cpvrlab/SLProject/wiki/SLProject-Coding-Style
//  Copyright: Marcus Hudritsch
//             This software is provide under the GNU General Public License
//             Please visit: http://opensource.org/licenses/GPL-3.0
//#############################################################################

#ifndef AVERAGED_H
#define AVERAGED_H

#include <vector>
#include <assert.h>

using namespace std;

//-----------------------------------------------------------------------------
//!Averaged template class provides an average value from a fixed size array.
/*!The Average template class provides an average value continuously averaged
from a fixed size vector. The template class can be used for any template type
T that provides the following operators: =,-,+,T*float
*/
template<class T>
class Averaged
{
public:
    //! Ctor with a default value array size
    explicit Averaged(int numValues = 60, T initValue = 0)
    {
        init(numValues, initValue);
    }

    //! Initializes the average value array to a given value
    void init(int numValues, T initValue)
    {
        assert(numValues > 0 && "Num. of values must be greater than zero");
        _values.clear();
        _values.resize(numValues, initValue);
        _oneOverNumValues = 1.0f / (float)_values.size();
        _sum              = initValue * numValues;
        _average          = initValue;
        _currentValueNo   = 0;
    }

    //! Sets the current value in the value array and builds the average
    void set(T value)
    {
        if (_currentValueNo == _values.size())
            _currentValueNo = 0;

        // Correct the sum continuously
        _sum                     = _sum - _values[_currentValueNo];
        _values[_currentValueNo] = value;
        _sum                     = _sum + _values[_currentValueNo];
        _average                 = _sum * _oneOverNumValues; // avoid division
        _currentValueNo++;
    }

    //! Gets the averaged value
    T average() { return _average; }

    //! Get the last entry
    T last() { return _currentValueNo > 0 ? _currentValueNo - 1 : _values.size() - 1; }

private:
    float     _oneOverNumValues{}; //!< multiplier instead of divider
    vector<T> _values;             //!< value array
    int       _currentValueNo{};   //!< current value index within _values
    T         _sum;                //!< sum of all values
    T         _average;            //!< average value
};
//-----------------------------------------------------------------------------
typedef Averaged<float> AvgFloat;
//-----------------------------------------------------------------------------
#endif
