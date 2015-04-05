//#############################################################################
//  File:      SL/SLUtils.h
//  Author:    Marcus Hudritsch
//  Date:      July 2014
//  Codestyle: https://github.com/cpvrlab/SLProject/wiki/Coding-Style-Guidelines
//  Copyright: Marcus Hudritsch
//             This software is provide under the GNU General Public License
//             Please visit: http://opensource.org/licenses/GPL-3.0
//#############################################################################

#include <stdafx.h>
#include <dirent.h>

#ifndef SLUTILS_H
#define SLUTILS_H

//-----------------------------------------------------------------------------
//! SLUtils provides static utility functions string handling
class SLUtils
{
    public:

        //! SLUtils::toString returns a string from a float with max. one trailing zero
        static SLstring toString(float f)
        {   char cstr[32];
            sprintf(cstr, "%f", f);
            for (SLint i = (SLint)strlen(cstr); i > 0; i--)
            if (cstr[i]=='0' && cstr[i-1]>='0' && cstr[i-1]<='9') cstr[i] = 0;
            SLstring num = cstr;
            if (num == "-0.0") num = "0.0";
            return num;
        }

        //! SLUtils::toString returns a string from a double with max. one trailing zero
        static SLstring toString(double d)
        {   char cstr[32];
            sprintf(cstr, "%f", d);
            for (SLint i = (SLint)strlen(cstr); i > 0; i--)
            if (cstr[i]=='0' && cstr[i-1]>='0' && cstr[i-1]<='9') cstr[i] = 0;
            SLstring num = cstr;
            if (num == "-0.0") num = "0.0";
            return num;
        }

        //! SLUtils::toLower returns a string in lower case
        static SLstring toLower(SLstring s)
        {   SLstring cpy(s);
            transform(cpy.begin(), cpy.end(), cpy.begin(),::tolower);
            return cpy;
        }
      
        //! SLUtils::toUpper returns a string in upper case
        static SLstring toUpper(SLstring s)
        {   SLstring cpy(s);
            transform(cpy.begin(), cpy.end(), cpy.begin(),::toupper);
            return cpy;
        }
      
        //! SLUtils::getPath returns the path w. '\\' of path-filename string
        static SLstring getPath(const SLstring& pathFilename) 
        {
            size_t i;
            i = pathFilename.rfind('\\', pathFilename.length( ));
            if (i != string::npos) 
            {  return(pathFilename.substr(0, i+1));
            }
         
            i = pathFilename.rfind('/', pathFilename.length( ));
            if (i != string::npos) 
            {  return(pathFilename.substr(0, i+1));
            }
            return pathFilename;
        }
      
        //! SLUtils::getFileName returns the filename of path-filename string
        static SLstring getFileName(const SLstring& pathFilename) 
        {
            size_t i;
            i = pathFilename.rfind('\\', pathFilename.length( ));
            if (i != string::npos) 
                return(pathFilename.substr(i+1, pathFilename.length( ) - i));
            i = pathFilename.rfind('/', pathFilename.length( ));
            if (i != string::npos) 
                return(pathFilename.substr(i+1, pathFilename.length( ) - i));
            return pathFilename;
        }
      
        //! SLUtils::getFileNameWOExt returns the filename without extension
        static SLstring getFileNameWOExt(const SLstring& pathFilename) 
        {
            SLstring filename = getFileName(pathFilename);
            size_t i;
            i = filename.rfind('.', filename.length( ));
            if (i != string::npos) 
            {  return(filename.substr(0, i));
            }
         
            return(filename);
        }    
      
        //! SLUtils::getFileExt returns the file extension without dot in lower case
        static SLstring getFileExt(SLstring filename) 
        {
            size_t i;
            i = filename.rfind('.', filename.length( ));
            if (i != string::npos) 
            return toLower(filename.substr(i+1, filename.length() - i));
            return("");
        }

        //! SLUtils::getFileNamesinDir returns a vector of storted filesname with path within a directory
        static SLVstring getFileNamesInDir(SLstring dirName)
        {
            SLVstring fileNames;
            DIR* dir;
            struct dirent *dirContent;
            int i=0;
            dir = opendir(dirName.c_str());

            if (dir)
            {   while ((dirContent = readdir(dir)) != NULL)
                {   i++;
                    SLstring name(dirContent->d_name);
                    if(name != "." && name != "..")
                        fileNames.push_back(dirName+"/"+name);
                }
                closedir(dir);
            }
            return fileNames;
        }
            
        //! SLUtils::trims a string at the end
        static SLstring trim(SLstring& s, const SLstring& drop = " ")
        {   SLstring r=s.erase(s.find_last_not_of(drop)+1);
            return r.erase(0,r.find_first_not_of(drop));
        }
      
        //! SLUtils::splits an input string at a delimeter character into a string vector
        static void split(const string& s, char delimiter, vector<string>& splits) 
        {   string::size_type i = 0;
            string::size_type j = s.find(delimiter);

            while (j != string::npos) 
            {   splits.push_back(s.substr(i, j - i));
                i = ++j;
                j = s.find(delimiter, j);
                if (j == string::npos)
                    splits.push_back(s.substr(i, s.length()));
            }
        }

        //! Replaces in the subject string the search string by the replace string
        static void replaceString(SLstring& subject, 
                                  const SLstring& search,
                                  const SLstring& replace)
        {   size_t pos = 0;
            while ((pos = subject.find(search, pos)) != std::string::npos) 
            {   subject.replace(pos, search.length(), replace);
                pos += replace.length();
            }
        }

        //! SLUtils::removeComments for C/C++ comments removal from shader code
        static SLstring removeComments(SLstring src)
        {  
            SLstring dst;
            SLint len = (SLint)src.length();
            SLint i = 0;

            while (i < len)
            {   if (src[i]=='/' && src[i+1]=='/')
                {   dst += '\n';
                    while (i<len && src[i] != '\n') i++;
                    i++; 
                } 
                else if (src[i]=='/' && src[i+1]=='*')
                {   while (i<len && !(src[i]=='*' && src[i+1]=='/'))
                    { 
                        if (src[i]=='\n') dst += '\n';
                        i++; 
                    }
                    i+=2;
                } 
                else
                {  dst += src[i++];
                } 
            }
            //cout << dst << "|" << endl;
            return dst;
        }
};
//-----------------------------------------------------------------------------
#endif



