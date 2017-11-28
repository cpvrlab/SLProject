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
        static SLstring toString(float f, int roundedDecimals = 1)
        {   
            SLint magnitude = SL_pow(10, roundedDecimals);
            SLfloat rf = (SLfloat)(round(f * magnitude) / magnitude);

            char cstr[32];
            sprintf(cstr, "%f", rf);
            for (SLint i = (SLint)strlen(cstr); i > 0; i--)
                if (cstr[i]=='0' && cstr[i-1]>='0' && cstr[i-1]<='9') cstr[i] = 0;
            SLstring num = cstr;
            if (num == "-0.0") num = "0.0";
            return num;
        }

        //! SLUtils::toString returns a string from a double with max. one trailing zero
        static SLstring toString(double d, int roundedDecimals = 1)
        {   
            SLint magnitude = SL_pow(10, roundedDecimals);
            SLfloat rd = (SLfloat)(round(d * magnitude) / magnitude);

            char cstr[32];
            sprintf(cstr, "%f", rd);
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
            size_t i1, i2;
            i1 = pathFilename.rfind('\\', pathFilename.length());
            i2 = pathFilename.rfind('/', pathFilename.length());
            if ((i1 != string::npos && i2 == string::npos) ||
                (i1 != string::npos && i1 > i2)) 
            {  return(pathFilename.substr(0, i1+1));
            }
         
            if ((i2 != string::npos && i1 == string::npos) ||
                (i2 != string::npos && i2 > i1))
            {  return(pathFilename.substr(0, i2+1));
            }
            return pathFilename;
        }
      
        //! SLUtils::getFileName returns the filename of path-filename string
        static SLstring getFileName(const SLstring& pathFilename) 
        {
            size_t i=0, i1, i2;
            i1 = pathFilename.rfind('\\', pathFilename.length( ));
            i2 = pathFilename.rfind('/', pathFilename.length( ));

            if (i1 != string::npos && i2 != string::npos)
                i = SL_max(i1, i2);
            else if (i1 != string::npos)
                i = i1;
            else if (i2 != string::npos)
                i = i2;

            return pathFilename.substr(i+1, pathFilename.length( ) - i);
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
        static SLVstring getFileNamesInDir(const SLstring dirName)
        {
            SLVstring fileNames;
            DIR* dir;
            dir = opendir(dirName.c_str());

            if (dir)
            {
                struct dirent *dirContent;
                int i=0;

                while ((dirContent = readdir(dir)) != NULL)
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
        static void replaceString(string& source,
                                  const string& from,
                                  const string& to)
        {   
            // Code from: http://stackoverflow.com/questions/2896600/how-to-replace-all-occurrences-of-a-character-in-string
            string newString;
            newString.reserve(source.length());  // avoids a few memory allocations

            string::size_type lastPos = 0;
            string::size_type findPos;

            while( string::npos != ( findPos = source.find( from, lastPos )))
            {
                newString.append( source, lastPos, findPos - lastPos );
                newString += to;
                lastPos = findPos + from.length();
            }

            // Care for the rest after last occurrence
            newString += source.substr( lastPos );
            source.swap( newString );
        }

        //! Returns local time as string
        static SLstring getLocalTimeString()
        {
            time_t tm;
            time(&tm);
            struct tm *t2 = localtime(&tm);
            char buf[1024];
            strftime(buf, sizeof(buf), "%c", t2);
            return SLstring(buf);
        }

        //! Returns a formatted string as sprintf
        static SLstring formatString(const SLstring fmt_str, ...) 
        {
            // Reserve two times as much as the length of the fmt_str
            int final_n, n = ((int)fmt_str.size()) * 2; 
            std::string str;
            std::unique_ptr<char[]> formatted;
            va_list ap;
            while(1) 
            {
                formatted.reset(new char[n]); /* Wrap the plain char array into the unique_ptr */
                strcpy(&formatted[0], fmt_str.c_str());
                va_start(ap, fmt_str);
                final_n = vsnprintf(&formatted[0], n, fmt_str.c_str(), ap);
                va_end(ap);
                if (final_n < 0 || final_n >= n)
                    n += abs(final_n - n + 1);
                else
                    break;
            }
            return SLstring(formatted.get());
        }

        //! contains returns true if container contains the search string
        static SLbool contains(const SLstring container, const SLstring search)
        {
            return (container.find(search) != string::npos);
        }
};
//-----------------------------------------------------------------------------
#endif
