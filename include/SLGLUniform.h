//#############################################################################
//  File:      SLGLUniform.h
//  Author:    Marcus Hudritsch
//  Date:      July 2014
//  Codestyle: https://github.com/cpvrlab/SLProject/wiki/Coding-Style-Guidelines
//             This software is provide under the GNU General Public License
//             Please visit: http://opensource.org/licenses/GPL-3.0
//#############################################################################

#ifndef SLGLUNIFORM_H
#define SLGLUNIFORM_H

#include <stdafx.h>
#include <SLEventHandler.h>

//-----------------------------------------------------------------------------
//! Template for a single GLSL uniform variable.
/*! Class for GLSL uniform variables that change per frame. An SLGLProgram
holds a list of this type of uniform variables that are applied within the 
beginUse method. 
*/
template<class T>
class SLGLUniform : public SLEventHandler
{
    public:
        SLGLUniform(SLUF1Type type,
                    const SLchar* name,
                    T value,
                    T inc=0.0f,
                    T min=0.0f,
                    T max=0.0f,
                    SLKey   keyInc=KeyNone)
        {   _name   = name;
            _value  = value;
            _min    = min;
            _max    = max;
            _inc    = inc;
            _type   = type;
            _keyInc = keyInc;
        }

        const SLchar* name() {return _name.c_str();}
      
        /*! 
        calculates the current value & returns it. 
        This method is called on every frame.
        */
        T value()
        {  
            if (_type==UF1Const) 
            return _value;
            if (_type==UF1IncDec)
            {   if ((_inc>0.0f && _value>=_max) || 
                (_inc<0.0f && _value<=_min)) _inc*=-1; 
                _value+=_inc;
                return _value;
            }
            if (_type==UF1IncInc)
            {   if (_inc>0 && _value>=_max) _value=_min; 
                if (_inc<0 && _value<=_min) _value=_max; 
                _value+=_inc;
                return _value;
            }
            if (_type==UF1Inc)
            {   _value+=_inc;
                return _value;
            }
            if (_type==UF1Random)
            {   _value = _min + ((T)rand()/(T)RAND_MAX)*(_max-_min);
                return _value;
            }
            if (_type==UF1Seconds)
            {   _value = (T)clock()/CLOCKS_PER_SEC;
                return _value;
            }
            else return _value;
        }
   
        //! Keypress eventhandler
        SLbool onKeyPress(const SLKey key, const SLKey mod)
        {   if (_keyInc!=KeyNone)
            {   if (key==_keyInc) 
                {   if (mod==KeyNone) 
                    {   if (_value <_max)
                        {   _value+=_inc;
                            cout << "Uniform: " << _name.c_str() << " = " << _value << endl;
                            return true;
                        } else
                        if (_inc == _max) // Toggle between min & max
                        {   _value = _min;
                            cout << "Uniform: " << _name.c_str() << " = " << _value << endl;
                            return true;
                        }
                    } else
                    if (mod==KeyShift) 
                    {   if (_value>_min)
                        {   _value-=_inc;
                            cout << "Uniform: " << _name.c_str() << " = " << _value << endl;
                            return true;
                        }
                    }
                }
            }
            return false; 
        }
      
    private:
        SLstring    _name;      //!< Name of the variable
        T           _value;     //!< Current value
        T           _max;       //!< Max. value for IncInc, IncDec & random types
        T           _min;       //!< Min. value for IncInc, IncDec & random types
        T           _inc;       //!< Increment value for IncInc, IncDec & Inc types
        SLUF1Type   _type;      //!< Uniform1f type
        SLKey       _keyInc;    //!< keyboard key incrementing const values
};
//-----------------------------------------------------------------------------
typedef SLGLUniform<SLfloat> SLGLUniform1f;
typedef SLGLUniform<SLint>   SLGLUniform1i;
typedef SLGLUniform<SLuint>  SLGLUniform1u;
//-----------------------------------------------------------------------------
//! STL vector of SLGLShaderUniform1f pointers
typedef std::vector<SLGLUniform1f*>  SLVUniform1f;
typedef std::vector<SLGLUniform1i*>  SLVUniform1i;
//-----------------------------------------------------------------------------
#endif
