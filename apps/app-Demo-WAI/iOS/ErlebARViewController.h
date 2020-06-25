//#############################################################################
//  File:      ErlebARViewController.h
//  Purpose:
//  Author:    Michael Göttlicher
//  Date:      June 2020
//  Codestyle: https://github.com/cpvrlab/SLProject/wiki/SLProject-Coding-Style
//  Copyright: Marcus Hudritsch
//             This software is provide under the GNU General Public License
//             Please visit: http://opensource.org/licenses/GPL-3.0
//#############################################################################

#import <GLKit/GLKit.h>

@interface ErlebARViewController : GLKViewController

- (id)init:(NSString*)nibNameOrNil;

- (void)appWillResignActive;
- (void)appDidEnterBackground;
- (void)appWillEnterForeground;
- (void)appDidBecomeActive;
- (void)appWillTerminate;

@end
