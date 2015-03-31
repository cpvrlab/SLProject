//#############################################################################
//  File:      SL/SLDrawBits.h
//  Author:    Marcus Hudritsch
//  Date:      July 2014
//  Codestyle: https://github.com/cpvrlab/SLProject/wiki/Coding-Style-Guidelines
//  Copyright: Marcus Hudritsch
//             This software is provide under the GNU General Public License
//             Please visit: http://opensource.org/licenses/GPL-3.0
//#############################################################################

#ifndef SLDRAWBITS_H
#define SLDRAWBITS_H

#include <stdafx.h>

//-----------------------------------------------------------------------------
/*!
Drawing Bits control some visual states of the scene and are applied per scene 
view or per single node object. Not all are used from the beginning
*/
#define SL_DB_HIDDEN      1   //!< Flags an object as hidden  
#define SL_DB_SELECTED    2   //!< Flags an object as selected
#define SL_DB_WIREMESH    4   //!< Draw polygons as wired mesh 
#define SL_DB_NORMALS     8   //!< Draw the vertex normals
#define SL_DB_BBOX       16   //!< Draw the bounding boxes of a node  
#define SL_DB_AXIS       32   //!< Draw the coordinate axis of a node
#define SL_DB_VOXELS     64   //!< Draw the voxels of the uniform grid 
#define SL_DB_CULLOFF   128   //!< Turn off face culling
#define SL_DB_TEXOFF    256   //!< Turn off texture mapping

//-----------------------------------------------------------------------------
//! Drawing states stored in the bits of an unsigned int
/*! The drawing bits can be applied to the entire scene, a group or a mesh. The
default value is 0 signifying the default state. See the defines above for the
different drawing bit flags. 
*/
class SLDrawBits
{     
   public:           
                        SLDrawBits (){_bits=0;}
                       ~SLDrawBits (){;}
            
            //! Turns all bits off
            void        allOff  () {_bits=0;}

            //! Turns the specified bit on
            void        on(SLuint bit){ SL_SETBIT(_bits, bit); }

            //! Turns the specified bit off
            void        off(SLuint bit){ SL_DELBIT(_bits, bit); }
            
            //! Sets the specified bit to the passed state
            void        set     (SLuint bit, SLbool state){if(state) SL_SETBIT(_bits, bit);
                                                           else SL_DELBIT(_bits, bit);}
            //! Toggles the specified bit
            void        toggle  (SLuint bit){SL_TOGBIT(_bits, bit);}
            
            //! Returns the specified bit
            SLbool      get     (SLuint bit){return (_bits & bit)?true:false;}
            
            //! Returns the all bits
            SLuint      bits    () {return _bits;}

   private:            
            SLuint      _bits; //!< Drawing flags as a unsigned 32-bit register
};
//-----------------------------------------------------------------------------
#endif
