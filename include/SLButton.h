//#############################################################################
//  File:      SLButton.h
//  Author:    Marcus Hudritsch
//  Date:      July 2014
//  Codestyle: https://github.com/cpvrlab/SLProject/wiki/Coding-Style-Guidelines
//  Copyright: Marcus Hudritsch
//             This software is provide under the GNU General Public License
//             Please visit: http://opensource.org/licenses/GPL-3.0
//#############################################################################

#ifndef SLBUTTON_H
#define SLBUTTON_H

#include <stdafx.h>
#include <SLNode.h>
#include <SLTexFont.h>
#include <SLText.h>
#include <SLSceneView.h>
#include <SLGLBuffer.h>

//-----------------------------------------------------------------------------
//! Defines a 2D-GUI button with optional submenus as children of the group.
/*! 
SLButton defines a GUI button with optional submenus as children of the group.
It serves as a minimal hierarchical GUI element for implementing a menu.
It has a SLText instance for text rendering. The button can be
either a click-, check- or radio-button. For check and radio button there is a
cross drawn in checked state. A button can have a command (SLCmd) associated.
A button without command is expected to open its children button as a submenu.
The button tree is created in SLSceneView::build2DMenus and attached to 
SLScene._menu2D. Opening, closing & command execution is handled in
onMouseDown and onMouseUp.
*/  
class SLButton: public SLNode
{  public:                 
                        SLButton(SLSceneView* sv,
                                SLstring     text,
                                SLTexFont*   txtFont,
                                SLCmd        cmd = cmdMenu,
                                SLbool       isCheckable = false,
                                SLbool       isChecked = false,
                                SLButton*    radioParent = 0,
                                SLbool       closeOnClick = true,
                                SLfloat      btnWidth = 0.0f, 
                                SLfloat      btnHeight = 0.0f, 
                                SLCol3f      btnColor = SLCol3f::COLBFH,
                                SLfloat      btnAlpha = 0.8f,
                                SLTextAlign  txtAlign = centerLeft,
                                SLCol4f      txtColor = SLCol4f::WHITE);

                       ~SLButton    ();
                        
            void        drawRec     (SLSceneView* sv);
            SLAABBox&   updateAABBRec();
            SLbool      hitRec      (SLRay* ray){(void)ray; return false;}
            
            // Mouse down handler
            SLbool      onMouseDown    (const SLMouseButton button, 
                                        const SLint x, const SLint y, 
                                        const SLKey mod); 
            SLbool      onMouseUp      (const SLMouseButton button, 
                                        const SLint x, const SLint y, 
                                        const SLKey mod); 
            SLbool      onDoubleClick  (const SLMouseButton button, 
                                        const SLint x, const SLint y, 
                                        const SLKey mod){return false;}               
            // Setters
            void        isDown      (SLbool down)  {_isDown=down;}
            void        isChecked   (SLbool check) {if (_isCheckable)
                                                      _isChecked=check;
                                                    else _isChecked=false;}
            void        btnW        (SLfloat w) {_btnW = w;}
            void        btnH        (SLfloat h) {_btnH = h;}

            // Getters
            SLCmd       command     () {return _command;}
            SLText*     text        () {return _text;}
            SLbool      isDown      () {return _isDown;}
            SLbool      isCheckable () {return _isCheckable;}
            SLbool      isChecked   () {return _isChecked;}
            SLbool      isRadio     () {return _radioParent ? true : false;}
            SLfloat     btnW        () {return _btnW;}
            SLfloat     btnH        () {return _btnH;}
            SLfloat     minX        () {return _minX;}
            SLfloat     minY        () {return _minY;}
               
            // Misc.
            void        buildBuffers      ();
            void        closeAll          ();
            void        hideAndReleaseRec ();
            void        checkRadioRec     ();
            SLVec2f     setSizeRec        ();
            void        setPosRec         (SLfloat x, SLfloat y);
               
    static  SLButton*   buttonDown;    //!< The button of the last mouse down
    static  SLButton*   buttonParent;  //!< The parent btn of an open subMenu
    static  SLVec2f     minMenuPos;    //!< Lower left corner of unshifted root menu
    static  SLVec2f     newMenuPos;    //!< New transition pos on menu click
    static  SLVec2f     oldMenuPos;    //!< Old transition pos on menu click

   private:
            SLSceneView* _sv;          //!< SV on which the button appears   
            SLCmd       _command;      //!< Cmd id that is issued on click
            SLbool      _isCheckable;  //!< Flag if button is checkable
            SLbool      _isChecked;    //!< Flag if button has a check cross
            SLButton*   _radioParent;  //!< parent button group for radio buttons
            SLbool      _closeOnClick; //!< Flag if all menues are closed on click
               
            SLfloat     _minX;         //!< X-coord. from bottom left corner
            SLfloat     _minY;         //!< Y-coord. from bottom left corner
            SLfloat     _btnW;         //!< Width in pixels. if 0 then autom.
            SLfloat     _btnH;         //!< Height in pixels. if 0 then autom.

            SLbool      _isDown;       //!< Flag if button is pressed
            SLTextAlign _txtAlign;     //!< Text alignment 
            SLCol3f     _btnCol;       //!< RGB-Color of the button
            SLfloat     _btnAlpha;     //!< Alpha value for the button
               
            SLText*     _text;         //!< Text object
            SLVec2f     _textSize;     //!< Width and height of text string
            SLint       _updateSteps;  //!< NO. of update steps for transition
               
            SLGLBuffer  _bufP;         //!< Buffer for vertex positions
            SLGLBuffer  _bufC;         //!< Buffer for vertex colors
            SLGLBuffer  _bufI;         //!< Buffer for vertex indexes
};
//-----------------------------------------------------------------------------
#endif
