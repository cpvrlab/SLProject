//#############################################################################
//  File:      Utils_iOS.mm
//  Date:      September 2011 (HS11)
//  Authors:   Marcus Hudritsch
//  License:   This software is provided under the GNU General Public License
//             Please visit: http://opensource.org/licenses/GPL-3.0
//#############################################################################

#include "Utils_iOS.h"
#include <Utils.h>
#include <sys/stat.h> //dirent

#import <Foundation/Foundation.h>
//-----------------------------------------------------------------------------
/*! Utils_iOS::fileExists returns true if the file exists. This code works
only Apple OSX and iOS. If no file matches, it checks all files of the same
directory and compares them case insensitive. If now one file matches the
passed filename is converted to the existing casesensitive filename.
Because I was not able to do this task in C++, I have to do this with a
C++/ObjectiveC mix.
*/
bool Utils_iOS::fileExists(string& pathfilename)
{
    // This stat compare is done casesensitive only on ARM hardware
    struct stat stFileInfo;
    if (stat(pathfilename.c_str(), &stFileInfo) == 0)
        return true;

    // Get path and file name seperately and as NSString
    std::string path   = Utils::getPath(pathfilename);
    std::string file   = Utils::getFileName(pathfilename);
    NSString*   nsPath = [NSString stringWithCString:path.c_str()
                                          encoding:[NSString defaultCStringEncoding]];
    NSString*   nsFile = [NSString stringWithCString:file.c_str()
                                          encoding:[NSString defaultCStringEncoding]];

    NSFileManager* fileManager = [NSFileManager defaultManager];
    if ([fileManager fileExistsAtPath:nsPath])
    {
        BOOL isDir = NO;
        [fileManager fileExistsAtPath:nsPath isDirectory:(&isDir)];
        if (isDir == YES)
        {
            NSArray* contents;
            contents = [fileManager contentsOfDirectoryAtPath:nsPath error:nil];

            // Loop over all files of directory and compare caseinsensitive
            for (NSString* entity in contents)
            {
                // NSLog(@"filesystemname = %@, searchname = %@", entity, nsFile);

                if ([entity length] == [nsFile length])
                {
                    if ([entity caseInsensitiveCompare:nsFile] == NSOrderedSame)
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
vector<string> Utils_iOS::getAllNamesInDir(const string& dirName, bool fullPath)
{
    vector<string> folderContent;

    // Get path and file name seperately and as NSString
    std::string path   = Utils::getPath(dirName);
    std::string folder = Utils::getFileName(dirName);

    NSString* nsPath   = [NSString stringWithCString:path.c_str()
                                          encoding:[NSString defaultCStringEncoding]];
    NSString* nsFolder = [NSString stringWithCString:folder.c_str()
                                            encoding:[NSString defaultCStringEncoding]];

    NSFileManager* fileManager = [NSFileManager defaultManager];

    if ([fileManager fileExistsAtPath:nsPath])
    {
        BOOL isDir = NO;
        [fileManager fileExistsAtPath:nsPath isDirectory:(&isDir)];

        if (isDir == YES)
        {
            NSArray* contents;
            contents = [fileManager contentsOfDirectoryAtPath:nsPath error:nil];

            for (NSString* entity in contents)
            {
                if (fullPath)
                    folderContent.emplace_back(path + [entity UTF8String]);
                else
                    folderContent.emplace_back([entity UTF8String]);
            }
        }
    }
    return folderContent;
}
//-----------------------------------------------------------------------------
std::string Utils_iOS::getAppsWritableDir()
{
    // Get library directory for config file
    NSArray*  paths            = NSSearchPathForDirectoriesInDomains(NSLibraryDirectory,
                                                         NSUserDomainMask,
                                                         YES);
    NSString* libraryDirectory = [paths objectAtIndex:0];
    string    configDir        = [libraryDirectory UTF8String];
    configDir += "/SLProject";
    NSString* configPath = [NSString stringWithUTF8String:configDir.c_str()];

    // Create if it does not exist
    NSError* error;
    if (![[NSFileManager defaultManager] fileExistsAtPath:configPath])
        [[NSFileManager defaultManager] createDirectoryAtPath:configPath
                                  withIntermediateDirectories:NO
                                                   attributes:nil
                                                        error:&error];

    return configDir + "/";
}
//-----------------------------------------------------------------------------
std::string Utils_iOS::getCurrentWorkingDir()
{
    // Get the main bundle path and pass it the SLTexture and SLShaderProg
    // This will be the default storage location for textures and shaders
    NSString* bundlePath = [[NSBundle mainBundle] resourcePath];
    string    cwd        = [bundlePath UTF8String];
    return cwd + "/";
}
//-----------------------------------------------------------------------------
bool Utils_iOS::deleteFile(std::string& pathfilename)
{
    if (Utils_iOS::fileExists(pathfilename))
        return remove(pathfilename.c_str()) != 0;
    return false;
}
//-----------------------------------------------------------------------------
