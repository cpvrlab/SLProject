//#############################################################################
//  File:      SL/SLAverage.h
//  Author:    Marcus Hudritsch
//  Date:      July 2014
//  Codestyle: https://github.com/cpvrlab/SLProject/wiki/Coding-Style-Guidelines
//  Copyright: Marcus Hudritsch
//             This software is provide under the GNU General Public License
//             Please visit: http://opensource.org/licenses/GPL-3.0
//#############################################################################


#ifndef SLAVERAGE_H
#define SLAVERAGE_H

#include <stdafx.h>

//-----------------------------------------------------------------------------
//!SLAverage template class provides an average value from a fixed size array.
/*!The SLAverage template class provides an average value continuously averaged 
from a fixed size vector. The template class can be used for any template type
T that provides the following operators: =,-=,+=,T*float
*/
template<class T>
class SLAverage
{
    public:
            //! Ctor with a default value array size
            SLAverage(SLint numValues, T zeroValue)
            {   
                init(numValues, zeroValue);
            }

            //! Initializes the average value array to a given value
            void init(SLint numValues, T zeroValue)
            {  
                _values.clear();
                _values.resize(numValues, zeroValue);
                _oneOverNumValues = 1.0f / (SLfloat)_values.size();
                _sum = zeroValue;
                _average = zeroValue;
                _currentValueNo = 0;
            }

            //! Sets the current value in the value array and builds the average
            void set(T value)
            {
                if (_currentValueNo==_values.size()) 
                    _currentValueNo = 0;

                // Correct the sum continuosly
                _sum -= _values[_currentValueNo];
                _values[_currentValueNo] = value;
                _sum += _values[_currentValueNo];
                _average = _sum * _oneOverNumValues; // avoid division
                _currentValueNo++;
            }

            //! Gets the avaraged value
            T average() {return _average;}
    
            //! Get the last entry
            T last()    {return _currentValueNo>0? _currentValueNo-1:_values.size()-1;}

   private:
            SLfloat     _oneOverNumValues;   //!< multiplier instead of devider
            vector<T>   _values;             //!< value array
            SLint       _currentValueNo;     //!< current value index
            T           _sum;                //!< sum of all values
            T           _average;            //!< average value
};
//-----------------------------------------------------------------------------
typedef SLAverage<SLfloat>  SLAvgFloat;
typedef SLAverage<SLVec2f>  SLAvgVec2f;

typedef vector<SLAvgFloat>  SLVAvgFloat;
typedef vector<SLAvgVec2f>  SLVAvgVec2f;
//-----------------------------------------------------------------------------
#endif



