//#############################################################################
//  File:      SL/SLVector.h
//  Author:    Jonas Lottner, Marcus Hudritsch
//  Date:      July 2014
//  Codestyle: https://github.com/cpvrlab/SLProject/wiki/Coding-Style-Guidelines
//  Copyright: Institut fr Informatik, FHBB Muttenz, Switzerland
//             This software is provide under the GNU General Public License
//             Please visit: http://opensource.org/licenses/GPL-3.0
//  Changes:   jlottner    09-JAN-2003
//             mhudritsch  08-FEB-2003
//             jlottner    10-FEB-2003
//#############################################################################

#ifndef SLVECTOR_H
#define SLVECTOR_H

#include <stdafx.h>


//-----------------------------------------------------------------------------
#define MAXCAPACITY (SLint)(pow(2.0f, (SLfloat)(sizeof(U)*8))) - 1
//-----------------------------------------------------------------------------
//! Template class for dynamic vector
/*!
Implements a minimal dynamic sized array like the STL std::vector.
The array can be of a class type T and can have the max. size of type U.
Compatibility is given as long no iterators are used. Bounds checks are only
done in _DEBUG mode within the access methods and operators.
*/
template<class T, class U> 
class SLVector
{  public:
                     SLVector     ();                 //!< creates empty array
                     SLVector     (SLuint size);      //!< creates array w. size
                     SLVector     (const SLVector& a);//!< creates a copy of array a
      virtual       ~SLVector     ();                 //!< standard destructor

      SLVector<T,U>& operator =  (const SLVector& a); //!< assignment operator
      SLVector<T,U>& operator =  (const SLVector* a); //!< assignment operator
      inline T&      operator[]  (SLuint i);          //!< access operator

      void           set         (const SLVector& a); //!< set array with another
      U              size        (){return _size;}    //!< returns size
      U              capacity    (){return _capacity;}//!< returns internal size
      void           push_back   (const T element);   //!< appends element at end
      void           pop_back    ();                  //!< deletes element at end
      void           erase       (U i);               //!< delete element at pos i
      inline T&      at          (SLuint i);          //!< returns element at pos i
      void           reverse     ();                  //!< reverses the order
      void           clear       () {resize(0);}      //!< deletes all
      void           resize      (SLuint64 size=0);   //!< deletes all, sets _size=size
      void           reserve     (SLuint64 newSize);  //!< set capacity = newSize
                                            
   private:                                 
      U              _size;      //!< real size of array of type U
      U              _capacity;  //!< internal size of array of type U
      T*             _contents;  //!< pointer to the array of type T
};

//-----------------------------------------------------------------------------
template<class T, class U>
SLVector<T,U>::SLVector()
{  _size     = 0;
   _capacity = 0;
   _contents = 0;
}
//-----------------------------------------------------------------------------
template<class T, class U>
SLVector<T,U>::SLVector(SLuint size)
{  _size     = size;
   _capacity = size;
   _contents = new T[_capacity];
}
//-----------------------------------------------------------------------------
template<class T, class U>
SLVector<T,U>::SLVector(const SLVector& a)
{  _size     = 0;
   _capacity = 0;
   _contents = 0;
   set(a);
}
//-----------------------------------------------------------------------------
template<class T, class U> 
SLVector<T,U>::~SLVector()
{  
    delete[] _contents;
}
//-----------------------------------------------------------------------------
/*!
The bracket operator as used in arrays. You can use it on the left or right 
side of =. Overrun is checked in _DEBUG mode and causes Warning but returns
a value so that the caller can be reached.
*/
template<class T, class U>
inline T& SLVector<T,U>::operator[](SLuint i)
{  
   #ifdef _DEBUG
   if (i >= _size)
   {  SL_LOG("SLVector::operator[]: Index >= size! Overflow! Argh!");
      return _contents[_size-1]; // return something for debug
   }
   #endif
   return _contents[i];
}
//-----------------------------------------------------------------------------
/*!
Returns the element at position i.
Overrun is checked in _DEBUG mode and causes Warning but returns
a value so that the caller can be reached.
*/
template<class T, class U>
inline T& SLVector<T,U>::at(SLuint i)
{  
   #ifdef _DEBUG
   if (i >= _size)
   {  SL_LOG("SLVector::operator[]: Index >= size! Overflow! Argh!");
      return _contents[_size-1]; // return something for debug
   }
   #endif
   return _contents[i];
}
//-----------------------------------------------------------------------------
template<class T, class U>
SLVector<T,U>& SLVector<T,U>::operator = (const SLVector& a)
{  this->set(a);
   return(*this);
}
//-----------------------------------------------------------------------------
template<class T, class U>
SLVector<T,U>& SLVector<T,U>::operator = (const SLVector* a)
{  this->set(*a);
   return(*this);
}
//-----------------------------------------------------------------------------
template<class T, class U>
void SLVector<T,U>::set(const SLVector &a)
{  SLuint i;
   if (&a != this)
   {  T* temp= 0;
      
      if (_capacity > 0) 
         delete[] _contents;

      _size = a._size;
      _capacity = _size;
      _contents = 0; 
      
      if (_capacity > 0)
      {  temp = new T[_capacity];
         for (i=0; i<_capacity; ++i) temp[i] = a._contents[i];
         _contents = temp;
      }
   }
}
//-----------------------------------------------------------------------------
/*!
Internal, the SLVector is representet by an c++Array not of size _size, but of 
an internal size. The function reserve changes the internal representation and 
can make adding much more faster. If s is smaller than the actual size, it 
will be ignored.
*/
template<class T, class U>
void SLVector<T,U>::reserve(SLuint64 newSize)
{  T* temp;
   SLuint i;
   
   #ifdef _DEBUG
   if (newSize > MAXCAPACITY)
   {  SL_LOG("SLVector::reserve: newSize > Max. capacity\n");
      newSize = MAXCAPACITY;
   }
   #endif
   
   if ((U)newSize >= _size)
   {  _capacity = (U)newSize;
      temp = new T[_capacity];
      for (i=0; i<_size; i++) temp[i] = _contents[i];
      delete[] _contents;
      _contents = temp;
   }
}
//-----------------------------------------------------------------------------
template<class T, class U>
void SLVector<T,U>::push_back(const T element)
{  if (_capacity != 0)
   {  if (_size >= _capacity)
      {  reserve(_capacity * 2);
      }
      _contents[_size] = element;
      _size++;
   } else
   {  _capacity = 2;
      _size = 1;
      _contents = new T[_capacity];
      _contents[0] = element;
   }
}
//-----------------------------------------------------------------------------
template<class T, class U>
void SLVector<T,U>::erase(U index)
{  if (index >= _size)
   {  SL_EXIT_MSG("SLVector::erase(SLuint index): Index out of range!");
   }
   else if (index == _size-1)
   {  pop_back();
   }
   else
   {  SLuint j;
      _size--;
      _capacity= _size;
      T *temp = new T[_capacity];
      for (j=0; j<_size; ++j)
      {  if (j<index) temp[j] = _contents[j];
         else temp[j] = _contents[j+1];
      }
      delete[] _contents;
      _contents = temp;
   }
}
//-----------------------------------------------------------------------------
template<class T, class U>
void SLVector<T,U>::reverse()
{  SLuint i;
   T temp;

   for (i=0; i<(_size/2); i++)
   {  temp = _contents[i];
      _contents[i] = _contents[_size-i-1];
      _contents[_size-i-1] = temp;
   }
}
//-----------------------------------------------------------------------------
template<class T, class U> 
void SLVector<T,U>::resize(SLuint64 newSize)
{  
   #ifdef _DEBUG
   if (newSize > MAXCAPACITY)
   {  SL_LOG("SLVector::reserve: newSize > Max. capacity\n");
      newSize = MAXCAPACITY;
   }
   #endif
   _size = (U)newSize;
   _capacity = (U)newSize;
   delete[] _contents;
   _contents = 0;
   if (_capacity) _contents = new T[_capacity];
}
//-----------------------------------------------------------------------------
template<class T, class U>
void SLVector<T,U>::pop_back()
{  _size--;
}
//-----------------------------------------------------------------------------
#endif
