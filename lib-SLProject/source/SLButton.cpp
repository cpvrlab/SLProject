//#############################################################################
//  File:      SLButton.cpp
//  Author:    Marcus Hudritsch
//  Date:      July 2014
//  Codestyle: https://github.com/cpvrlab/SLProject/wiki/Coding-Style-Guidelines
//  Copyright: Marcus Hudritsch
//             This software is provide under the GNU General Public License
//             Please visit: http://opensource.org/licenses/GPL-3.0
//#############################################################################

#include <stdafx.h>           // precompiled headers
#ifdef SL_MEMLEAKDETECT       // set in SL.h for debug config only
#include <debug_new.h>        // memory leak detector
#endif

#include <SLButton.h>
#include <SLCamera.h>

//-----------------------------------------------------------------------------
const SLfloat BTN_TXT2BTN_H_FACTOR = 1.0f;   // Button height factor from text hight
const SLfloat BTN_BORDER_W_MM = 3.0f;        // Horizontal border in mm
const SLfloat BTN_GAP_W_MM = 0.7f;           // Horizontal gap in mm
const SLfloat BTN_GAP_H_MM = 0.7f;           // vertical gap in mm   
//-----------------------------------------------------------------------------
SLButton* SLButton::buttonDown = 0;
SLButton* SLButton::buttonParent = 0;
SLVec2f   SLButton::minMenuPos = SLVec2f(10,10); //!< Lower left corner of menu
SLVec2f   SLButton::newMenuPos = SLVec2f(0,0);
SLVec2f   SLButton::oldMenuPos = SLVec2f(0,0);
//-----------------------------------------------------------------------------
/*! SLButton Constructor: 
If command is cmdMenu the button is supposed to have sub menu buttons.
If isCheckable is true the button gets a checkbox
If isChecked is true the checkbox gets a little cross
If radioParent is a parent button all of it checkable children act as radio buttons.
If closeOnClick is true the entire menu gets closed after command execution.
*/
SLButton::SLButton(SLSceneView* sv,
                   SLstring     text,  
                   SLTexFont*   txtFont,
                   SLCmd        command, 
                   SLbool       isCheckable, 
                   SLbool       isChecked, 
                   SLButton*    radioParent, 
                   SLbool       closeOnClick,
                   SLfloat      btnWidth, 
                   SLfloat      btnHeight, 
                   SLCol3f      btnColor, 
                   SLfloat      btnAlpha,
                   SLTextAlign  txtAlign,  
                   SLCol4f      txtColor) : SLNode(text)
{
    assert(txtFont);
    assert(text!="");
   
    _sv           = sv;
    _command      = command;
    _isCheckable  = isCheckable;
    _isChecked    = isChecked;
    _radioParent  = radioParent;
    _closeOnClick = closeOnClick;
    _minX         = 0.0f;
    _minY         = 0.0f;
    _btnW         = btnWidth;
    _btnH         = btnHeight;
    _isDown       = false;
    _txtAlign     = txtAlign;
    _btnCol       = btnColor;
    _btnAlpha     = btnAlpha;
    _drawBits.on(SL_DB_HIDDEN);

    _text = new SLText(text, txtFont, txtColor, 
                        _btnW-2*_sv->dpmm()*BTN_BORDER_W_MM);
}
//-----------------------------------------------------------------------------
SLButton::~SLButton()
{  
    delete _text;
}
//-----------------------------------------------------------------------------
/*! 
SLButton::shapeDraw draws everything of the button
*/
void SLButton::drawRec(SLSceneView* sv)
{ 
    bool rootButton = (_name == ">");
    bool loadSceneButton = (_name == "Load Scene >");
    if (_drawBits.get(SL_DB_HIDDEN)) return;
    
    _stateGL->pushModelViewMatrix();
    _stateGL->modelViewMatrix.multiply(_om.m());
    _stateGL->buildInverseAndNormalMatrix();

    // Create the buffer objects the first time
    if (!_bufP.id() && !_bufC.id() && !_bufI.id())
        buildBuffers();
   
    // draw root menu always, raw sliding menus below parent and
    // draw menu from parent upwards
    if (!buttonParent || (buttonParent && depth() >= buttonParent->depth()))
    {
        // Setup shader
        SLMaterial::current = 0;
        SLGLProgram* sp = SLScene::current->programs(ColorAttribute);
        SLGLState* state = SLGLState::getInstance();
        sp->useProgram();
        sp->uniformMatrix4fv("u_mvpMatrix", 1, (SLfloat*)state->mvpMatrix());
        SLint indexP = sp->getAttribLocation("a_position");
        SLint indexC = sp->getAttribLocation("a_color");

        // draw antialiased 
        state->multiSample(true);
   
        // Draw button rectangle
        SLint colorIndex = _isDown ? buttonDown==this ? 24 : 0 : 4;
        _bufP.bindAndEnableAttrib(indexP);
        _bufC.bindAndEnableAttrib(indexC, colorIndex * sizeof(SLCol4f));
        _bufI.bindAndDrawElementsAs(SL_TRIANGLES, 6);
   
        // Draw border line
        colorIndex = _isDown ? 8 : 12;
        _bufC.bindAndEnableAttrib(indexC, colorIndex * sizeof(SLCol4f));
        _bufI.bindAndDrawElementsAs(SL_LINE_LOOP, 4, 6*sizeof(SLushort));
   
        // Draw checkbox
        if (_isCheckable)
        {   _bufP.bindAndEnableAttrib(indexP,  8 * sizeof(SLVec3f));
            _bufC.bindAndEnableAttrib(indexC, 20 * sizeof(SLCol4f));
            _bufI.bindAndDrawElementsAs(SL_LINE_LOOP, 4, 14*sizeof(SLushort));
        }
   
        // Draw checkmark
        if (_isChecked)
        {   _bufP.bindAndEnableAttrib(indexP,  4 * sizeof(SLVec3f));
            _bufC.bindAndEnableAttrib(indexC, 16 * sizeof(SLCol4f));
            _bufI.bindAndDrawElementsAs(SL_TRIANGLE_FAN, 4, 10*sizeof(SLushort));
        }

        _bufP.disableAttribArray();
        _bufC.disableAttribArray();    
      
        // Set start position at bottom left corner of the text
        SLfloat x = (SLfloat)_minX; 
        SLfloat y = (SLfloat)_minY;
        switch (_txtAlign)
        {   case topLeft:      x+=4*sv->dpmm()*BTN_GAP_W_MM;  y+= _btnH-_textSize.y;       break;
            case topCenter:    x+=(_btnW-_textSize.x)*0.5f;   y+= _btnH-_textSize.y;       break;
            case topRight:     x+= _btnW-_textSize.x;         y+= _btnH-_textSize.y;       break;
            case centerLeft:   x+=4*sv->dpmm()*BTN_GAP_W_MM;  y+=(_btnH-_textSize.y)*0.5f; break;
            case centerCenter: x+=(_btnW-_textSize.x)*0.5f;   y+=(_btnH-_textSize.y)*0.5f; break;
            case centerRight:  x+= _btnW-_textSize.x;         y+=(_btnH-_textSize.y)*0.5f; break;
            case bottomLeft:   x+=4*sv->dpmm()*BTN_GAP_W_MM;                               break;
            case bottomCenter: x+=(_btnW-_textSize.x)*0.5f;                                break;
            case bottomRight:  x+= _btnW-_textSize.x;                                      break;
        }
   
        // position & draw the text
        if (_text)
        {   state->pushModelViewMatrix();
            state->modelViewMatrix.translate(x, y, 0.0f);
            _text->drawRec(sv);
            state->popModelViewMatrix();
        }
    }
    
    for (auto child : _children)
        child->drawRec(sv);

    _stateGL->popModelViewMatrix();
}
//-----------------------------------------------------------------------------
/*! 
SLButton::buildBuffers creates VBOs for position, texture and color
*/
void SLButton::buildBuffers()
{
    SLint numP = 4+4+4;  // 4 vertices for btn, 4 for checkmark, 4 for check rect
    SLint numC = 7*4;    // 4+4 colors for btn & border each down & up + 4 for checkmark
    SLint numI = 6+4+4+4;// 6 indexes for btn, 4 for border, 4 for checkmark, 4 for checkbox
      
    SLVec3f*  P  = new SLVec3f[numP];   // vertex positions
    SLCol4f*  C  = new SLCol4f[numC];   // colors
    SLushort* I  = new SLushort[numI];  // triangle index
    SLfloat 	 x = _minX;
    SLfloat 	 y = _minY;
    SLfloat 	 w = _btnW;
    SLfloat 	 h = _btnH;
    SLfloat 	 mx = x + 2*_sv->dpmm()*BTN_GAP_W_MM; // center x of check mark
    SLfloat 	 my = y + h*0.5f; // center y of check mark
    SLfloat   diff1 = 0.3f;    // button color difference upper to lower border
    SLfloat   diff2 = 0.6f;    // border color difference upper to lower border  
    
    // vertices
    P[0].set(x,   y  );        // button bottom left corner
    P[1].set(x+w, y  );        // button bottem right corner
    P[2].set(x+w, y+h);        // button top right corner
    P[3].set(x,   y+h);        // button top left corner
      
    P[4].set(mx-4, my+4);      // 1st point of checkmark
    P[5].set(mx+4, my-4);      // 2nd point of checkmark
    P[6].set(mx-4, my-4);      // 3rd point of checkmark
    P[7].set(mx+4, my+4);      // 4th point of checkmark
      
    P[8].set(mx-5, my-5);      // 1st point of checkbox rect
    P[9].set(mx+5, my-5);      // 2nd point of checkbox rect
    P[10].set(mx+5, my+5);     // 3rd point of checkbox rect
    P[11].set(mx-5, my+5);     // 4th point of checkbox rect
      
    // down checked button colors
    C[0].set(_btnCol.r+diff1, _btnCol.g+diff1, _btnCol.b+diff1, _btnAlpha);
    C[1].set(_btnCol.r+diff1, _btnCol.g+diff1, _btnCol.b+diff1, _btnAlpha); 
    C[2].set(_btnCol.r+diff1, _btnCol.g+diff1, _btnCol.b+diff1, _btnAlpha);
    C[3].set(_btnCol.r+diff1, _btnCol.g+diff1, _btnCol.b+diff1, _btnAlpha);
      
    // up button colors
    C[4].set(_btnCol.r-diff1, _btnCol.g-diff1, _btnCol.b-diff1, _btnAlpha);
    C[5].set(_btnCol.r-diff1, _btnCol.g-diff1, _btnCol.b-diff1, _btnAlpha); 
    C[6].set(_btnCol.r+diff1, _btnCol.g+diff1, _btnCol.b+diff1, _btnAlpha);
    C[7].set(_btnCol.r+diff1, _btnCol.g+diff1, _btnCol.b+diff1, _btnAlpha);
      
    // down border colors
    C[8 ].set(_btnCol.r+diff2, _btnCol.g+diff2, _btnCol.b+diff2, 1.0f);
    C[9 ].set(_btnCol.r+diff2, _btnCol.g+diff2, _btnCol.b+diff2, 1.0f); 
    C[10].set(_btnCol.r-diff2, _btnCol.g-diff2, _btnCol.b-diff2, 1.0f);
    C[11].set(_btnCol.r-diff2, _btnCol.g-diff2, _btnCol.b-diff2, 1.0f);
      
    // up border colors
    C[12].set(_btnCol.r-diff2, _btnCol.g-diff2, _btnCol.b-diff2, 1.0f);
    C[13].set(_btnCol.r-diff2, _btnCol.g-diff2, _btnCol.b-diff2, 1.0f); 
    C[14].set(_btnCol.r+diff2, _btnCol.g+diff2, _btnCol.b+diff2, 1.0f);
    C[15].set(_btnCol.r+diff2, _btnCol.g+diff2, _btnCol.b+diff2, 1.0f);
      
    // Color of check mark
    C[16].set(1,1,1,0.8f);
    C[17].set(1,1,1,0.8f);
    C[18].set(1,1,1,0.8f);
    C[19].set(1,1,1,0.8f);
      
    // Color of check box
    C[20].set(_btnCol.r-diff2, _btnCol.g-diff2, _btnCol.b-diff2, 1.0f);
    C[21].set(_btnCol.r-diff2, _btnCol.g-diff2, _btnCol.b-diff2, 1.0f);
    C[22].set(_btnCol.r-diff2, _btnCol.g-diff2, _btnCol.b-diff2, 1.0f);
    C[23].set(_btnCol.r-diff2, _btnCol.g-diff2, _btnCol.b-diff2, 1.0f);

    // pressed button colors
    C[24].set(_btnCol.r+diff1, _btnCol.g+diff1, _btnCol.b+diff1, _btnAlpha);
    C[25].set(_btnCol.r+diff1, _btnCol.g+diff1, _btnCol.b+diff1, _btnAlpha); 
    C[26].set(_btnCol.r-diff1, _btnCol.g-diff1, _btnCol.b-diff1, _btnAlpha);
    C[27].set(_btnCol.r-diff1, _btnCol.g-diff1, _btnCol.b-diff1, _btnAlpha);
      
    // indexes for 2 button triangles
    I[0]=0; I[1]=1; I[2]=3;  I[3]=1; I[4]=2; I[5]=3;
      
    // indexes for 2 button border line loop
    I[6]=0; I[7]=1; I[8]=2; I[9]=3;
      
    // indexes for checkmark
    I[10]=0; I[11]=2; I[12]=1; I[13]=3;
      
    // indexes for checkbox lines
    I[14]=0; I[15]=1; I[16]=2; I[17]=3;
      
    // create buffers on GPU
    _bufP.generate(P, numP, 3);
    _bufC.generate(C, numC, 4);
    _bufI.generate(I, numI, 1, SL_UNSIGNED_SHORT, SL_ELEMENT_ARRAY_BUFFER);
      
    // delete data on CPU     
    delete[] P;
    delete[] C;
    delete[] I;
}
//-----------------------------------------------------------------------------
/*! 
SLButton::buildAABB builds and returns the axis-aligned bounding box.
*/
SLAABBox& SLButton::updateAABBRec()
{  
    // build AABB of subMenus
    SLNode::updateAABBRec();
   
    // calculate min & max in object space
    SLVec3f minOS((SLfloat)_minX, (SLfloat)_minY, -0.01f);
    SLVec3f maxOS((SLfloat)(_minX+_btnW), (SLfloat)(_minY+_btnH), 0.01f);
   
    // enlarge aabb for avoiding rounding errors 
    minOS -= FLT_EPSILON;
    maxOS += FLT_EPSILON;
   
    // apply world matrix: this overwrites the AABB of the group
    _aabb.fromOStoWS(minOS, maxOS, updateAndGetWM());
      
    return _aabb;
}
//-----------------------------------------------------------------------------
/*!
SLButton::onMouseDown handles events and returns true if refresh is needed 
*/
SLbool SLButton::onMouseDown(const SLMouseButton button, 
                             const SLint x, const SLint y, const SLKey mod)
{     
    // GUI space is bottom left!
    SLint h = _sv->scrH()-y;
   
    // Compare x and h with min and max of AABB in WS
    if (x > _aabb.minWS().x && x < _aabb.maxWS().x &&
        h > _aabb.minWS().y && h < _aabb.maxWS().y)
    {  _isDown = true;
        buttonDown = this;
        return true;
    }
   
    // check sub menus
    for (auto child : _children)
    {   if (!child->drawBits()->get(SL_DB_HIDDEN))
        if (child->onMouseDown(button, x, y, mod))
            return true;
    }
   
    buttonDown = 0;
    return false;
}
//-----------------------------------------------------------------------------
/*!
SLButton::onMouseUp handles events and returns true if refresh is needed. This
method holds the main functionality for the buttons command execution as well
as the hiding and showing of the sub menues.
*/
SLbool SLButton::onMouseUp(const SLMouseButton button, 
                           const SLint x, const SLint y, const SLKey mod)
{  
    SLScene* s = SLScene::current;
    SLButton* mnu2D = s->menu2D();
    SLButton* btn = 0; // button pointer for various loops
    if (!mnu2D) return false;

    // GUI space is bottom left!
    SLint h = _sv->scrH()-y;
   
    if (x > _aabb.minWS().x && x < _aabb.maxWS().x &&
        h > _aabb.minWS().y && h < _aabb.maxWS().y && 
        !_drawBits.get(SL_DB_HIDDEN))
    {
        if (buttonDown==this)
        { 
            // For a command execute it
            if (_command) 
            {  _isDown = false;
            
                // Toggle checkable buttons
                if (_isCheckable && !_radioParent)
                    _isChecked = !_isChecked;
            
                // For radio buttons uncheck others
                if (_radioParent)
                    _radioParent->checkRadioRec();
            
                // Hide all menus again
                if (_closeOnClick)
                    closeAll();

                /////////////////////////
                _sv->onCommand(_command);
                /////////////////////////

                if (_closeOnClick)
                    return true;
            }
            else if (_children.size()>0)
            {  
                // if another menu on the same or higher level is open hide it first
                if (buttonParent && buttonParent!=this && buttonParent->depth()>=_depth)
                {  
                    while (buttonParent && buttonParent->depth() >= depth())
                    {   for (auto child : buttonParent->children())
                            child->drawBits()->set(SL_DB_HIDDEN, true);
                        buttonParent->isDown(false);
                        buttonParent = (SLButton*)buttonParent->parent();
                    }
                }
         
                // show my submenu buttons         
                btn = (SLButton*)_children[0];
                if (btn && btn->drawBits()->get(SL_DB_HIDDEN))
                {  
                    for (auto child : _children)
                        child->drawBits()->set(SL_DB_HIDDEN, false);
                    buttonParent = this;
                } 
                else // submenu is already open so close everything
                {   if (buttonDown==mnu2D)
                    {   mnu2D->hideAndReleaseRec();
                        mnu2D->drawBits()->off(SL_DB_HIDDEN);
                        buttonParent = 0;
                    } else
                    {   if (buttonParent)
                        {  
                            for (auto child : buttonParent->children())
                                child->drawBits()->set(SL_DB_HIDDEN, true);
                            buttonParent->isDown(false);
                            buttonParent = (SLButton*)(buttonParent->parent());
                        }
                    }
                }
            }
         
            // Move menu left or right
            if (buttonParent)
            {   newMenuPos.set(-buttonParent->minX()+minMenuPos.x, 0);
                if (newMenuPos != oldMenuPos) 
                {  
                    mnu2D->translate(newMenuPos.x - oldMenuPos.x, 0, 0, TS_Object);
               
                    // update AABB's
                    _stateGL->pushModelViewMatrix();
                    _stateGL->modelViewMatrix.identity();
                    oldMenuPos.set(newMenuPos);
                    mnu2D->updateAABBRec();
                    _stateGL->popModelViewMatrix();
                }
            }
         
            buttonDown = 0;
            return true;
        }
        else // mouse up on a different button, so release it
        {  if (buttonDown)
            {  buttonDown->_isDown = false;
            buttonDown = 0;
            return true;
            }
        }
    }
   
    // check sub menus
    for (auto child : _children)
    {   if (child->onMouseUp(button, x, y, mod))
            return true;
    }
   
    // check if mouse down was on this button and the up was on the scene
    if (_isDown && buttonDown==this)
    {   closeAll();
        return true;
    }
    return false;
}
//-----------------------------------------------------------------------------
/*!
SLButton::closeAll closes all menus except the root menu
*/
void SLButton::closeAll()
{  
    SLScene* s = SLScene::current;
    SLButton* mnu2D = s->menu2D();
    if (!mnu2D) return;
    mnu2D->hideAndReleaseRec();
    mnu2D->drawBits()->off(SL_DB_HIDDEN);

    // update world matrices & AABBs
    _stateGL->pushModelViewMatrix();
    _stateGL->modelViewMatrix.identity();
    mnu2D->translation(0, 0, 0);
    mnu2D->updateAABBRec();

    _stateGL->popModelViewMatrix();
   
    newMenuPos.set(0,0);
    oldMenuPos.set(0,0);
    buttonDown = 0;
    buttonParent = 0;
}
//-----------------------------------------------------------------------------
/*!
SLButton::hidden: hides or shows the button recursively
*/
void SLButton::hideAndReleaseRec()
{
    _drawBits.set(SL_DB_HIDDEN, true);
    _isDown = false;
   
    // hide/show sub menus
    for (auto child : _children)
        ((SLButton*)child)->hideAndReleaseRec();
}
//-----------------------------------------------------------------------------
/*!
SLButton::checkRadioRecursive checks the clicked button and recursively 
unchecks all other checkable children buttons.
*/
void SLButton::checkRadioRec()
{  
    for (auto child : _children)
    {   SLButton* btn = (SLButton*)child;
        btn->isChecked(btn->isCheckable() && btn==buttonDown);
        btn->checkRadioRec();
    }
}
//-----------------------------------------------------------------------------
/*!
SLButton::setSizeRecursive set the text and button size recursively
*/
SLVec2f SLButton::setSizeRec()
{  
    _textSize = _text->size();
    if (_btnW==0.0f) _btnW = _textSize.x + 2*_sv->dpmm()*BTN_BORDER_W_MM;
    if (_btnH==0.0f) _btnH = _textSize.y + _text->fontHeightPX()*BTN_TXT2BTN_H_FACTOR;
   
    if (_children.size()>0)
    {   SLfloat maxW = FLT_MIN;
        SLfloat maxH = FLT_MIN;
      
        // Loop through children & get max text size
        for (auto child : _children)
        {   SLButton* btn = (SLButton*)child;
            SLVec2f textSize = btn->setSizeRec();
            if (textSize.x > maxW) maxW = textSize.x;
            if (textSize.y > maxH) maxH = textSize.y;
        }
      
        // Loop through children & set all button to max. size
        for (auto child : _children)
        {   SLButton* btn = (SLButton*)child;
            btn->btnW(maxW + 2*_sv->dpmm()*BTN_BORDER_W_MM);
            btn->btnH(maxH + _text->fontHeightPX()*BTN_TXT2BTN_H_FACTOR);
        }
    }
   
    return _textSize;
}
//-----------------------------------------------------------------------------
/*!
SLButton::setPositionRecursive set the button position recursive
*/
void SLButton::setPosRec(SLfloat x, SLfloat y)
{  
    _minX = x;
    _minY = y;
   
    if (_children.size()>0)
    {   // set children Y-pos. according to its parent
        SLfloat curChildY = _minY;
        SLfloat childH = ((SLButton*)_children[0])->btnH();
        if (_children.size()>2)
            curChildY -= (_children.size()/2) * (childH + _sv->dpmm()*BTN_GAP_H_MM);
        curChildY = SL_max(curChildY, minMenuPos.y);
             
        // Loop through children & set their position
        for (auto child : _children)
        {   SLButton* btn = (SLButton*)child;
            btn->setPosRec(_minX + _btnW + _sv->dpmm()*BTN_GAP_W_MM, curChildY);
            curChildY += btn->btnH() + _sv->dpmm()*BTN_GAP_H_MM;
        }
    }
}

//-----------------------------------------------------------------------------
