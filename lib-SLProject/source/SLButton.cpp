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
const SLfloat BTN_TXT2BTN_H_FACTOR = 1.0f;  // Button height factor from text hight
const SLfloat BTN_BORDER_W_MM = 3.0f;       // Horizontal border in mm
const SLfloat BTN_GAP_W_MM = 0.7f;          // Horizontal gap in mm
const SLfloat BTN_GAP_H_MM = 0.7f;          // vertical gap in mm
//-----------------------------------------------------------------------------
SLButton* SLButton::buttonDown = 0;
SLButton* SLButton::buttonParent = 0;
SLVec2f   SLButton::minMenuPos = SLVec2f(10,10); //!< Lower left corner of menu
SLVec2f   SLButton::newMenuPos = SLVec2f(0,0);
SLVec2f   SLButton::oldMenuPos = SLVec2f(0,0);
//-----------------------------------------------------------------------------
/*! SLButton Constructor: 
If command is C_menu the button is supposed to have sub menu buttons.
If isCheckable is true the button gets a check box
If isChecked is true the check box gets a little cross
If radioParent is a parent button all of it checkable children act as radio buttons.
If closeOnClick is true the entire menu gets closed after command execution.
*/
SLButton::SLButton(SLSceneView* sv,
                   SLstring     text,  
                   SLTexFont*   txtFont,
                   SLCommand    command,
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

    _text = new SLText(text,
                       txtFont,
                       txtColor,
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
    if (_drawBits.get(SL_DB_HIDDEN)) return;
    
    _stateGL->pushModelViewMatrix();
    _stateGL->modelViewMatrix.multiply(_om.m());
    _stateGL->buildInverseAndNormalMatrix();

    // Create the vertex array objects the first time
    if (!_vao.id())
        buildBuffers();
   
    // draw root menu always, raw sliding menus below parent and
    // draw menu from parent upwards
    if (!buttonParent || (buttonParent && depth() >= buttonParent->depth()))
    {
        // Setup shader
        SLMaterial::current = 0;
        SLGLProgram* sp = SLScene::current->programs(SP_colorAttribute);
        SLGLState* state = SLGLState::getInstance();
        sp->useProgram();
        sp->uniformMatrix4fv("u_mvpMatrix", 1, (SLfloat*)state->mvpMatrix());

        // draw antialiased 
        state->multiSample(true);
   
        // Draw button rectangle
        _vao.drawArrayAs(PT_triangleStrip, _isDown ? buttonDown==this ? 8:4:0, 4);
   
        // Draw border line
        _vao.drawArrayAs(PT_lineLoop, _isDown ? 12 : 16, 4);
   
        // Draw check box
        if (_isCheckable) 
            _vao.drawArrayAs(PT_lineLoop, 20, 4);
   
        // Draw check mark
        if (_isChecked)
            _vao.drawArrayAs(PT_lines, 24, 4);  
      
        // Set start position at bottom left corner of the text
        SLfloat x = (SLfloat)_minX; 
        SLfloat y = (SLfloat)_minY;
        switch (_txtAlign)
        {   case TA_topLeft:      x+=4*sv->dpmm()*BTN_GAP_W_MM;  y+= _btnH-_textSize.y;       break;
            case TA_topCenter:    x+=(_btnW-_textSize.x)*0.5f;   y+= _btnH-_textSize.y;       break;
            case TA_topRight:     x+= _btnW-_textSize.x;         y+= _btnH-_textSize.y;       break;
            case TA_centerLeft:   x+=4*sv->dpmm()*BTN_GAP_W_MM;  y+=(_btnH-_textSize.y)*0.5f; break;
            case TA_centerCenter: x+=(_btnW-_textSize.x)*0.5f;   y+=(_btnH-_textSize.y)*0.5f; break;
            case TA_centerRight:  x+= _btnW-_textSize.x;         y+=(_btnH-_textSize.y)*0.5f; break;
            case TA_bottomLeft:   x+=4*sv->dpmm()*BTN_GAP_W_MM;                               break;
            case TA_bottomCenter: x+=(_btnW-_textSize.x)*0.5f;                                break;
            case TA_bottomRight:  x+= _btnW-_textSize.x;                                      break;
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
SLButton::buildBuffers creates the VAO for rendering
*/
void SLButton::buildBuffers()
{
    SLVVec2f  P;   // vertex positions
    SLVCol4f  C;   // colors
    SLfloat   x = _minX;
    SLfloat   y = _minY;
    SLfloat   w = _btnW;
    SLfloat   h = _btnH;
    SLfloat   mx = x + 2*_sv->dpmm()*BTN_GAP_W_MM; // center x of check mark
    SLfloat   my = y + h*0.5f; // center y of check mark
    SLfloat   diff1 = 0.3f;    // button color difference upper to lower border
    SLfloat   diff2 = 0.6f;    // border color difference upper to lower border  
    SLint     nP = 0;
    SLint     nC = 0;
    
    // up button 
    P.push_back(SLVec2f(x,   y+h));      // button top left corner
    P.push_back(SLVec2f(x,   y  ));      // button bottom left corner
    P.push_back(SLVec2f(x+w, y+h));      // button top right corner
    P.push_back(SLVec2f(x+w, y  ));      // button bottom right corner
    C.push_back(SLCol4f(_btnCol.r+diff1, _btnCol.g+diff1, _btnCol.b+diff1, _btnAlpha));
    C.push_back(SLCol4f(_btnCol.r-diff1, _btnCol.g-diff1, _btnCol.b-diff1, _btnAlpha));
    C.push_back(SLCol4f(_btnCol.r+diff1, _btnCol.g+diff1, _btnCol.b+diff1, _btnAlpha));
    C.push_back(SLCol4f(_btnCol.r-diff1, _btnCol.g-diff1, _btnCol.b-diff1, _btnAlpha)); 

    // down button
    P.push_back(SLVec2f(x,   y+h));      // button top left corner
    P.push_back(SLVec2f(x,   y  ));      // button bottom left corner
    P.push_back(SLVec2f(x+w, y+h));      // button top right corner
    P.push_back(SLVec2f(x+w, y  ));      // button bottom right corner
    C.push_back(SLCol4f(_btnCol.r+diff1, _btnCol.g+diff1, _btnCol.b+diff1, _btnAlpha));
    C.push_back(SLCol4f(_btnCol.r+diff1, _btnCol.g+diff1, _btnCol.b+diff1, _btnAlpha)); 
    C.push_back(SLCol4f(_btnCol.r+diff1, _btnCol.g+diff1, _btnCol.b+diff1, _btnAlpha));
    C.push_back(SLCol4f(_btnCol.r+diff1, _btnCol.g+diff1, _btnCol.b+diff1, _btnAlpha));
    
    // pressed button
    P.push_back(SLVec2f(x,   y+h));      // button top left corner
    P.push_back(SLVec2f(x,   y  ));      // button bottom left corner
    P.push_back(SLVec2f(x+w, y+h));      // button top right corner
    P.push_back(SLVec2f(x+w, y  ));      // button bottom right corner
    C.push_back(SLCol4f(_btnCol.r-diff1, _btnCol.g-diff1, _btnCol.b-diff1, _btnAlpha));
    C.push_back(SLCol4f(_btnCol.r+diff1, _btnCol.g+diff1, _btnCol.b+diff1, _btnAlpha));
    C.push_back(SLCol4f(_btnCol.r-diff1, _btnCol.g-diff1, _btnCol.b-diff1, _btnAlpha));
    C.push_back(SLCol4f(_btnCol.r+diff1, _btnCol.g+diff1, _btnCol.b+diff1, _btnAlpha)); 

    // down border
    P.push_back(SLVec2f(x,   y  ));      // button bottom left corner
    P.push_back(SLVec2f(x+w, y  ));      // button bottom right corner
    P.push_back(SLVec2f(x+w, y+h));      // button top right corner
    P.push_back(SLVec2f(x,   y+h));      // button top left corner
    C.push_back(SLCol4f(_btnCol.r+diff2, _btnCol.g+diff2, _btnCol.b+diff2, 1.0f));
    C.push_back(SLCol4f(_btnCol.r+diff2, _btnCol.g+diff2, _btnCol.b+diff2, 1.0f)); 
    C.push_back(SLCol4f(_btnCol.r-diff2, _btnCol.g-diff2, _btnCol.b-diff2, 1.0f));
    C.push_back(SLCol4f(_btnCol.r-diff2, _btnCol.g-diff2, _btnCol.b-diff2, 1.0f));
      
    // up border
    P.push_back(SLVec2f(x,   y  ));      // button bottom left corner
    P.push_back(SLVec2f(x+w, y  ));      // button bottom right corner
    P.push_back(SLVec2f(x+w, y+h));      // button top right corner
    P.push_back(SLVec2f(x,   y+h));      // button top left corner
    C.push_back(SLCol4f(_btnCol.r-diff2, _btnCol.g-diff2, _btnCol.b-diff2, 1.0f));
    C.push_back(SLCol4f(_btnCol.r-diff2, _btnCol.g-diff2, _btnCol.b-diff2, 1.0f)); 
    C.push_back(SLCol4f(_btnCol.r+diff2, _btnCol.g+diff2, _btnCol.b+diff2, 1.0f));
    C.push_back(SLCol4f(_btnCol.r+diff2, _btnCol.g+diff2, _btnCol.b+diff2, 1.0f));
      
    // White check box
    P.push_back(SLVec2f(mx-5, my-5));    // 1st point of check box rect
    P.push_back(SLVec2f(mx+5, my-5));    // 2nd point of check box rect
    P.push_back(SLVec2f(mx+5, my+5));    // 3rd point of check box rect
    P.push_back(SLVec2f(mx-5, my+5));    // 4th point of check box rect
    C.push_back(SLCol4f(_btnCol.r-diff2, _btnCol.g-diff2, _btnCol.b-diff2, 1.0f));
    C.push_back(SLCol4f(_btnCol.r-diff2, _btnCol.g-diff2, _btnCol.b-diff2, 1.0f));
    C.push_back(SLCol4f(_btnCol.r-diff2, _btnCol.g-diff2, _btnCol.b-diff2, 1.0f));
    C.push_back(SLCol4f(_btnCol.r-diff2, _btnCol.g-diff2, _btnCol.b-diff2, 1.0f));

    // White check mark
    P.push_back(SLVec2f(mx-4, my+4));    // 1st point of check mark
    P.push_back(SLVec2f(mx+4, my-4));    // 2nd point of check mark
    P.push_back(SLVec2f(mx-4, my-4));    // 3rd point of check mark
    P.push_back(SLVec2f(mx+4, my+4));    // 4th point of check mark
    C.push_back(SLCol4f(1,1,1,0.8f));
    C.push_back(SLCol4f(1,1,1,0.8f));
    C.push_back(SLCol4f(1,1,1,0.8f));
    C.push_back(SLCol4f(1,1,1,0.8f));
      
    // create buffers on GPU
    SLGLProgram* sp = SLScene::current->programs(SP_colorAttribute);
    sp->useProgram();
    _vao.setAttrib(AT_position, sp->getAttribLocation("a_position"), &P);
    _vao.setAttrib(AT_color, sp->getAttribLocation("a_color"), &C);
    _vao.generate((SLuint)P.size());
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
as the hiding and showing of the sub menus.
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
            if (_command != C_menu) 
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
                    mnu2D->translate(newMenuPos.x - oldMenuPos.x, 0, 0, TS_object);
               
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
