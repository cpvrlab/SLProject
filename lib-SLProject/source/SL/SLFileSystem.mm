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
SLbool SLFileSystem::dirExists(SLstring& path)
{
    struct stat info;
    if(stat(path.c_str(), &info ) != 0)
        return false;
    else if(info.st_mode & S_IFDIR)
        return true;
    else
        return false;
}
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
SLstring SLFileSystem::getAppsWritableDir()
{
    // Get library directory for config file
    NSArray *paths = NSSearchPathForDirectoriesInDomains(NSLibraryDirectory,
                                                         NSUserDomainMask,
                                                         YES);
    NSString *libraryDirectory = [paths objectAtIndex:0];
    string configDir = [libraryDirectory UTF8String];
    configDir += "/SLProject";
    NSString* configPath = [NSString stringWithUTF8String:configDir.c_str()];
    
    // Create if it does not exist
    NSError *error;
    if (![[NSFileManager defaultManager] fileExistsAtPath:configPath])
        [[NSFileManager defaultManager] createDirectoryAtPath:configPath
                                        withIntermediateDirectories:NO
                                        attributes:nil
                                        error:&error];
    
    return configDir + "/";
}
//-----------------------------------------------------------------------------
SLstring SLFileSystem::getCurrentWorkingDir()
{
    // Get the main bundle path and pass it the SLTexture and SLShaderProg
    // This will be the default storage location for textures and shaders
    NSString* bundlePath =[[NSBundle mainBundle] resourcePath];
    string cwd = [bundlePath UTF8String];
    return cwd + "/";
}
//-----------------------------------------------------------------------------





