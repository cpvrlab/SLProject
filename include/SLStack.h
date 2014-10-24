//#############################################################################
//  File:      SL/SLStack.h
//  Purpose:   Simple & fast fixed size LIFO stack template
//  Date:      July 2014
//  Codestyle: https://code.google.com/p/slproject/wiki/CodingStyleGuidelines
//  Copyright: 2002-2014 Marcus Hudritsch
//             This software is provide under the GNU General Public License
//             Please visit: http://opensource.org/licenses/GPL-3.0
//#############################################################################

#ifndef SLSTACK_H
#define SLSTACK_H

#include <SLMat4.h>

//-----------------------------------------------------------------------------
//! Simple & fast fixed size LIFO stack template
template<class T>
class SLStack 
{
    public:
        SLStack(int size=32)
        {   _items = 0;
            init(size);
        }

        //! The destructor deletes the array of items
        ~SLStack() {delete[] _items;}

        //! adds a new item on top of the stack
        void push_back(T c)
        {   if(full())
            {   cout << "Stack Full!" << endl;
                exit(1);
            }
            _items[++_top] = c;
        }

        //! returns the top item of the stack
        T pop_back()
        {   if(empty())
            {   cout << "Stack Empty!" << endl;
                exit(1);
            }
            return _items[_top--];
        }

        //! Initializes the stack to the given size
        void init(int size)   
        {   _size = size;
            _top = -1;
            if (_items) delete[] _items;
            _items = new T[_size];
        }
      
        void    clear()   {init(_size);}          //!< initializes to an empty stack
        bool    empty()   {return _top==-1;}      //!< returns true if stack is full
        bool    full()    {return _top+1==_size;} //!< returns false if stack is empty
        int     size()    {return _size;}         //!< returns the max. size of the stack
        int     top()     {return _top;}          //!< returns the index of the top item

    private:
        int     _size;   //!< max. size of the stack
        int     _top;    //!< index of the top item 
        T*      _items;  //!< pointer to the item array
};
//-----------------------------------------------------------------------------
typedef SLStack<SLMat4f>  SLSMat4f;
//-----------------------------------------------------------------------------
#endif