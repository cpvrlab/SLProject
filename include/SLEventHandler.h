//#############################################################################
//  File:      SL/SLEventHandler.h
//  Author:    Marcus Hudritsch
//  Date:      July 2014
//  Codestyle: https://github.com/cpvrlab/SLProject/wiki/Coding-Style-Guidelines
//  Copyright: Marcus Hudritsch
//             This software is provide under the GNU General Public License
//             Please visit: http://opensource.org/licenses/GPL-3.0
//#############################################################################

#ifndef SLEVENTHANDLER_H
#define SLEVENTHANDLER_H

#include <stdafx.h>
#include <SLEnums.h>

//-----------------------------------------------------------------------------
//! Virtual Eventhandler class
/*!
SLEventHandler profides virtual methods for basic mouse and keyboard events.
The SLNode class is derived from the SLEventhandler class and therefore all
nodes can act as a eventhandler. For the moment only the camera class handles
the events and implements this way the trackball camera.
The scene instance has a pointer to the active eventhandler and forwards the
events that it gets from the user interface.
See also: SLSceneView and SLCamera classes.
*/
class SLEventHandler
{  public:           
                     SLEventHandler()
                     {  _rotFactor = 0.5f;
                        _dPos = 0.1f;
                        _dRot = 15.0f;
                     }
                    ~SLEventHandler(){;}

            // Event handlers
   virtual  SLbool   onMouseDown    (const SLMouseButton button, 
                                     const SLint x, const SLint y,
                                     const SLKey mod)
                                    {(void)button; (void)x; (void)y; 
                                      return false;}  
   virtual  SLbool   onMouseUp      (const SLMouseButton button, 
                                     const SLint x, const SLint y,
                                     const SLKey mod)
                                    {(void)button; (void)x; (void)y; (void)mod; 
                                      return false;}  
   virtual  SLbool   onMouseMove    (const SLMouseButton button, 
                                     const SLint x, const SLint y,
                                     const SLKey mod)
                                    {(void)button; (void)x; (void)y; (void)mod;
                                     return false;}   
   virtual  SLbool   onDoubleClick  (const SLMouseButton button, 
                                     const SLint x, const SLint y,
                                     const SLKey mod)
                                    {(void)button; (void)x; (void)y; 
                                     return false;}   
   virtual  SLbool   onMouseWheel   (const SLint delta, const SLKey mod)
                                    {(void)delta; (void)mod;
                                      return false;}
   virtual  SLbool   onTouch2Down   (const SLint x1, const SLint y1,
                                     const SLint x2, const SLint y2)
                                    {return false;}
   virtual  SLbool   onTouch2Move   (const SLint x1, const SLint y1,
                                     const SLint x2, const SLint y2)
                                    {return false;}
   virtual  SLbool   onTouch2Up     (const SLint x1, const SLint y1,
                                     const SLint x2, const SLint y2)
                                    {return false;}
   virtual  SLbool   onKeyPress     (const SLKey key, const SLKey mod)
                                    {(void)key; (void)mod; return false;} 
   virtual  SLbool   onKeyRelease   (const SLKey key, const SLKey mod)
                                    {(void)key; (void)mod; return false;}  

            // Setters
            void     rotFactor      (SLfloat rf){_rotFactor=rf;}
            void     dRot           (SLfloat dr){_dRot=dr;}
            void     dPos           (SLfloat dp){_dPos=dp;}
            
            // Getters
            SLfloat  rotFactor      (){return _rotFactor;}
            SLfloat  dRot           (){return _dRot;}
            SLfloat  dPos           (){return _dPos;}
            
   protected:
            SLfloat  _rotFactor;    //!< Mouse rotation sensibility
            SLfloat  _dRot;         //!< Delta angle for keyb. rot.
            SLfloat  _dPos;         //!< Delta dist. for keyb. transl.
};
//-----------------------------------------------------------------------------
// STL list containter of SLEventHandler pointers
typedef std::vector<SLEventHandler*>  SLVEventHandler;
//-----------------------------------------------------------------------------
#endif
