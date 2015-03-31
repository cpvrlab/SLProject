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
from a fixed size array. It is used as SLAvgFloat for averaging time keeping
in SLSceneView.
*/
template<class T>
class SLAverage
{
    public:
            //! Ctor with a default value array size
            SLAverage(SLint numValues=60)
            {   _value = 0;
                init(numValues, 0);
            }

            //! Deletes the array on the heap
            ~SLAverage() {delete[] _value;}

            //! Initializes the average value array to a given value
            void init(SLint numValue=60, T initValue = 0.0f)
            {  
                _numValue = numValue;
                _oneOverNumValues = (T)1 / (T)_numValue;
                if (_value) delete[] _value;
                _value = new T[_numValue];
                _sum = (T)0;
                for (SLint i=0; i<_numValue; ++i)
                {   _value[i] = initValue;
                    _sum += initValue;
                }
                _currentValueNo = 0;
                _average = initValue;
            }

            //! Sets the current value in the value array and builds the average
            void set(T value)
            {
                if (_currentValueNo==_numValue) _currentValueNo = 0;

                // Correct the sum continuosly
                _sum -= _value[_currentValueNo];
                _value[_currentValueNo] = value;
                _sum += _value[_currentValueNo];
                _average = _sum * _oneOverNumValues; // avoid division
                _currentValueNo++;
            }

            //! Gets the avaraged value
            T average() {return _average;}

   private:
            SLint    _numValue;           //!< size of value array
            T        _oneOverNumValues;   //!< multiplier instead of devider
            T*       _value;              //!< value array
            SLint    _currentValueNo;     //!< current value index
            T        _sum;                //!< sum of all values
            T        _average;            //!< average value
};
//-----------------------------------------------------------------------------
typedef SLAverage<SLfloat> SLAvgFloat;
//-----------------------------------------------------------------------------
#endif



