//#############################################################################
//  File:      SLFileSystem.mm
//  Author:    Marcus Hudritsch
//  Date:      September 2011 (HS11)
//  Copyright: M. Hudritsch, Fachhochschule Nordwestschweiz
//             THIS SOFTWARE IS PROVIDED FOR EDUCATIONAL PURPOSE ONLY AND
//             WITHOUT ANY WARRANTIES WHETHER EXPRESSED OR IMPLIED.
//#############################################################################

#include <stdafx.h>
#ifdef SL_MEMLEAKDETECT
#include <nvwa/debug_new.h>   // memory leak detector
#endif

#include "SLFileSystem.h"

//-----------------------------------------------------------------------------
/*! SLFileSystem::fileExists returns true if the file exists. This code works
only Apple OSX and iOS. If no file matches, it checks all files of the same
directory and compares them case insensitive. If now one file matches the 
passed filename is converted to the existing casesensitive filename.
Because I was not able to do this task in C++, I have to do this with a
C++/ObjectiveC mix.
*/
SLbool SLFileSystem::fileExists(SLstring& pathfilename) 
{  
   // This stat compare is done casesensitive only on ARM hardware
   struct stat stFileInfo;
   if (stat(pathfilename.c_str(), &stFileInfo) == 0)
      return true;
   
   // Get path and file name seperately and as NSString
   SLstring path = SLUtils::getPath(pathfilename);
   SLstring file = SLUtils::getFileName(pathfilename);
   NSString *nsPath = [NSString stringWithCString:path.c_str() 
                       encoding:[NSString defaultCStringEncoding]];
   NSString *nsFile = [NSString stringWithCString:file.c_str() 
                       encoding:[NSString defaultCStringEncoding]];
               
   NSFileManager *fileManager = [NSFileManager defaultManager];
   if ([fileManager fileExistsAtPath:nsPath]) 
   {  BOOL isDir = NO;
      [fileManager fileExistsAtPath:nsPath isDirectory:(&isDir)];
      if (isDir == YES) 
      {  NSArray *contents;
         contents = [fileManager contentsOfDirectoryAtPath:nsPath error:nil];
         
         // Loop over all files of directory and compare caseinsensitive
         for (NSString *entity in contents) 
         {
            //NSLog(@"filesystemname = %@, searchname = %@", entity, nsFile);
                    
            if ([entity length] == [nsFile length])
            {  
               if ([entity caseInsensitiveCompare: nsFile] == NSOrderedSame) 
               {
                  // update the pathfilename with the real filename
                  pathfilename = path + [entity UTF8String];
                  return true;
               }
            }
         }
      }
   }
   return false;
}
//-----------------------------------------------------------------------------






