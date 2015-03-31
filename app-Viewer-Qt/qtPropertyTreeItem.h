//#############################################################################
//  File:      qtPropertyTreeItem.h
//  Purpose:   Serves as Qt tree widget item representing a property of SLNode
//  Author:    Marcus Hudritsch
//  Date:      July 2014
//  Copyright: Marcus Hudritsch
//             This software is provide under the GNU General Public License
//             Please visit: http://opensource.org/licenses/GPL-3.0
//#############################################################################

#ifndef QTPROPERTYTREEITEM_H
#define QTPROPERTYTREEITEM_H

#include "../include/SLNode.h"
#include <QTreeWidget>
#include <QTreeWidgetItem>
//----------------------------------------------------------------------------
//! qtPropertyTreeItem: QTreeWidgetItem derived class for property tree item.
class qtPropertyTreeItem : public QTreeWidgetItem
{
    public:
                        // Constructor for item in the tree widget
                        // Don't add the item by passing the parent to the 
                        // constructor of QTreeWidget or QTreeWidgetItem.
                        // This is extremely slow. Add after all settings.
                        qtPropertyTreeItem(QString column0,
                                           QString column1 = "",
                                           bool editable = false)
                        {   init(column0, column1, editable);
                        }

            void        init(QString column0, QString column1, bool editable)
                        {   _column0 = column0;
                            setText(0, column0);
                            if (column1!="") setText(1, column1);
                            if (editable) 
                            {   setFlags(flags() | Qt::ItemIsEditable);
                                setTextColor(1, Qt::darkGreen);
                            }
                            setTextAlignment(0, Qt::AlignLeft);
                        }
                     
            void        setGetString(function<const string&(void)> getString,
                                    function<void(const string&)> setString)
                        {   _getString = getString;
                            _setString = setString;
                            setText(1, QString::fromStdString(_getString()));
                        }
            void        setGetBool(function<bool(void)> getBool,
                                   function<void(bool)> setBool)
                        {   _getBool = getBool;
                            _setBool = setBool;
                            setFlags(flags() | Qt::ItemIsUserCheckable);
                            setCheckState(1, _getBool() ? Qt::Checked : Qt::Unchecked);
                        }
            void        setGetFloat(function<float(void)> getFloat,
                                    function<void(float)> setFloat)
                        {   _getFloat = getFloat;
                            _setFloat = setFloat;
                            setText(1, QString::number(_getFloat()));
                        }
            void        setGetVec3f(function<SLVec3f(void)> getVec3f,
                                    function<void(SLVec3f)> setVec3f)
                        {   _getVec3f = getVec3f;
                            _setVec3f = setVec3f;
                            setText(1, QString::fromStdString(_getVec3f().toString()));
                        }
            void        setGetVec4f(function<SLVec4f(void)> getVec4f,
                                    function<void(SLVec4f)> setVec4f)
                        {   _getVec4f = getVec4f;
                            _setVec4f = setVec4f;
                            setText(1, QString::fromStdString(_getVec4f().toString()));
                        }

        void           onItemChanged(int column);

    private:
        QString                         _column0;   //!< string for property name
        function<const string&(void)>   _getString; //!< string getter member function
        function<void(const string&)>   _setString; //!< string setter member function
        function<bool(void)>            _getBool;   //!< boolean getter member function
        function<void(bool)>            _setBool;   //!< boolean setter member function
        function<float(void)>           _getFloat;  //!< float getter member function
        function<void(float)>           _setFloat;  //!< float setter member function
        function<SLVec3f(void)>         _getVec3f;  //!< SLVec3f getter member function
        function<void(SLVec3f)>         _setVec3f;  //!< SLVec3f setter member function
        function<SLVec4f(void)>         _getVec4f;  //!< SLVec4f getter member function
        function<void(SLVec4f)>         _setVec4f;  //!< SLVec4f setter member function
};
//----------------------------------------------------------------------------
#endif
