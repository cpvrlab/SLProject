//#############################################################################
//  File:      qtProperty.h
//  Purpose:   Serves as Qt tree widget item representing a property of SLNode
//  Author:    Marcus Hudritsch
//  Date:      July 2014
//  Copyright: Marcus Hudritsch
//             This software is provide under the GNU General Public License
//             Please visit: http://opensource.org/licenses/GPL-3.0
//#############################################################################

#include "qtProperty.h"
#include "qtPropertyTreeWidget.h"
#include "qtUtils.h"
#include <QColorDialog>
#include <QFile>
#include <QFileInfo>
#include <QUrl>
#include <QDesktopServices>
#include <QMessageBox>

//-----------------------------------------------------------------------------
qtProperty::qtProperty(QString column0,
                       QString column1,
                       ActionOnDblClick onDblClick)
{   
    init(column0, column1, onDblClick);
}
//-----------------------------------------------------------------------------
void qtProperty::init(QString column0, 
                      QString column1, 
                      ActionOnDblClick onDblClick)
{   _column0 = column0;
    _onDblClick = onDblClick;

    setText(0, column0);
    if (column1!="") setText(1, column1);
    if (onDblClick > none) 
    {   setTextColor(1, Qt::darkGreen);
        if (onDblClick == edit)
            setFlags(flags() | Qt::ItemIsEditable);
    }
    setTextAlignment(0, Qt::AlignLeft);
}
//-----------------------------------------------------------------------------                 
void qtProperty::setGetString(function<const string&(void)> getString,
                              function<void(const string&)> setString)
{   _getString = getString;
    _setString = setString;
    setText(1, QString::fromStdString(_getString()));
}
//-----------------------------------------------------------------------------
void qtProperty::setGetBool(function<bool(void)> getBool,
                            function<void(bool)> setBool)
{   _getBool = getBool;
    _setBool = setBool;
    setFlags(flags() | Qt::ItemIsUserCheckable);
    setCheckState(1, _getBool() ? Qt::Checked : Qt::Unchecked);
}
//-----------------------------------------------------------------------------
void qtProperty::setGetFloat(function<float(void)> getFloat,
                             function<void(float)> setFloat)
{   _getFloat = getFloat;
    _setFloat = setFloat;
    setText(1, QString::number(_getFloat()));
}
//-----------------------------------------------------------------------------
void qtProperty::setGetVec3f(function<SLVec3f(void)> getVec3f,
                             function<void(SLVec3f)> setVec3f)
{   _getVec3f = getVec3f;
    _setVec3f = setVec3f;
    setText(1, QString::fromStdString(_getVec3f().toString()));
}
//-----------------------------------------------------------------------------
void qtProperty::setGetVec4f(function<SLVec4f(void)> getVec4f,
                             function<void(SLVec4f)> setVec4f)
{   _getVec4f = getVec4f;
    _setVec4f = setVec4f;
    setText(1, QString::fromStdString(_getVec4f().toString()));
}
//-----------------------------------------------------------------------------                 
void qtProperty::getNameAndURL(function<const string&(void)> getName,
                               function<const string&(void)> getURL)
{   _getName = getName;
    _getURL = getURL;
    setText(1, QString::fromStdString(_getName()));
}
//-----------------------------------------------------------------------------
void qtProperty::onItemChanged(int column)
{
    if (qtPropertyTreeWidget::isBeingBuilt || _onDblClick==none) return;

    if (column==0)
    {
        setText(0, _column0); // reset property column
    }
    else if (column==1)
    {
        if (_setString)
        {   _setString(text(1).toStdString());
        }
        else if (_setBool)
        {   _setBool(checkState(1) == Qt::Checked);
        }
        else if (_setFloat)
        {   _setFloat(text(1).toFloat());
            setText(1, QString::number(_getFloat()));
        }
        else if (_setVec3f)
        {   _setVec3f(SLVec3f(text(1).toStdString()));
            setText(1, QString::fromStdString(_getVec3f().toString()));
        }
        else if (_setVec4f)
        {   _setVec4f(SLVec4f(text(1).toStdString()));
            setText(1, QString::fromStdString(_getVec4f().toString()));
        }
    }
}
//-----------------------------------------------------------------------------
void qtProperty::onItemDblClicked(int column)
{
    if (qtPropertyTreeWidget::isBeingBuilt || _onDblClick==none) return;

    if (column==0)
    {
        setText(0, _column0); // reset property column
    }
    else if (column==1)
    {
        if (_onDblClick == colorPick)
        {   QColorDialog dlg;
            SLCol4f c(_getVec4f());
            QColor newCol = dlg.getColor(QColor(c.r*255,c.g*255,c.b*255,c.a*255), 
                                         nullptr, 
                                         _column0, 
                                         QColorDialog::ShowAlphaChannel |
                                         QColorDialog::DontUseNativeDialog);
            if (newCol.isValid())
            {   SLCol4f newSLCol(newCol.redF(), 
                                 newCol.greenF(), 
                                 newCol.blueF(), 
                                 newCol.alphaF());
                setText(1, QString::fromStdString(newSLCol.toString()));
            }
        }
        if (_onDblClick == openFile)
        {   QFileInfo fileToOpen(_getURL().c_str());
            
            cout << _getURL() << endl;
            cout << fileToOpen.canonicalFilePath().toStdString() << endl;

            if (fileToOpen.exists() && fileToOpen.isFile())
                QDesktopServices::openUrl(fileToOpen.canonicalFilePath());
            else
            {   SLstring msg = "File not found: \n" + 
                               fileToOpen.canonicalFilePath().toStdString();
                qtUtils::showMsgBox(msg.c_str());
            }
        }
    }
}
//-----------------------------------------------------------------------------
