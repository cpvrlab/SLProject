//#############################################################################
//  File:      qtPropertyTreeItem.h
//  Purpose:   Serves as Qt tree widget item representing a property of SLNode
//  Author:    Marcus Hudritsch
//  Date:      July 2014
//  Copyright: Marcus Hudritsch
//             This software is provide under the GNU General Public License
//             Please visit: http://opensource.org/licenses/GPL-3.0
//#############################################################################

#include "qtPropertyTreeItem.h"
#include "qtPropertyTreeWidget.h"

//-----------------------------------------------------------------------------
void qtPropertyTreeItem::onItemChanged(int column)
{
    if (qtPropertyTreeWidget::isBeingBuilt ||
        !(flags() & Qt::ItemIsEditable)) return;

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
