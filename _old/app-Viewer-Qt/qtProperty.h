//#############################################################################
//  File:      qtProperty.h
//  Purpose:   Serves as Qt tree widget item representing a property of SLNode
//  Author:    Marcus Hudritsch
//  Date:      July 2014
//  Copyright: Marcus Hudritsch
//             This software is provide under the GNU General Public License
//             Please visit: http://opensource.org/licenses/GPL-3.0
//#############################################################################

#ifndef QTPROPERTY_H
#define QTPROPERTY_H

#include "../include/SLNode.h"
#include <QTreeWidget>
#include <QTreeWidgetItem>
//----------------------------------------------------------------------------
//! qtPropertyTreeItem: QTreeWidgetItem derived class for property tree item.
class qtProperty : public QTreeWidgetItem
{
    public:
        //!< Possible actions on double click
        enum ActionOnDblClick {none, edit, colorPick, openFile};

                        // Constructor for item in the tree widget
                        // Don't add the item by passing the parent to the 
                        // constructor of QTreeWidget or QTreeWidgetItem.
                        // This is extremely slow. Add after all settings.
                        qtProperty      (QString column0,
                                         QString column1 = "",
                                         ActionOnDblClick onDblClick = none);
        void            init            (QString column0, 
                                         QString column1, 
                                         ActionOnDblClick onDblClick);
        void            setGetString    (function<const string&(void)> getString,
                                         function<void(const string&)> setString);
        void            setGetBool      (function<bool(void)> getBool,
                                         function<void(bool)> setBool);
        void            setGetFloat     (function<float(void)> getFloat,
                                         function<void(float)> setFloat);
        void            setGetVec3f     (function<SLVec3f(void)> getVec3f,
                                         function<void(SLVec3f)> setVec3f);
        void            setGetVec4f     (function<SLVec4f(void)> getVec4f,
                                         function<void(SLVec4f)> setVec4f);
        void            getNameAndURL   (function<const string&(void)> getName,
                                         function<const string&(void)> getURL);
        void            onItemChanged   (int column);
        void            onItemDblClicked(int column);

        // Getters
        QString         column0         () {return _column0;}
        ActionOnDblClick onDblClick     () {return _onDblClick;}

    private:
        QString                         _column0;   //!< string for property name
        ActionOnDblClick                _onDblClick;//!< action to do on double click
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
        function<const string&(void)>   _getName;   //!< name getter member function for files
        function<const string&(void)>   _getURL;    //!< URL getter member function for files
};
//----------------------------------------------------------------------------
#endif
