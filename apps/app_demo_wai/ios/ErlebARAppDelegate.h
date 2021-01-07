//#############################################################################
//  File:      ErlebARAppDelegate.h
//  Purpose:
//  Author:    Michael Göttlicher
//  Date:      June 2020
//  Codestyle: https://github.com/cpvrlab/SLProject/wiki/SLProject-Coding-Style
//  Copyright: Marcus Hudritsch
//             This software is provide under the GNU General Public License
//             Please visit: http://opensource.org/licenses/GPL-3.0
//#############################################################################

#import <UIKit/UIKit.h>

@class ErlebARViewController;

@interface ErlebARAppDelegate : UIResponder<UIApplicationDelegate>

@property (strong, nonatomic) UIWindow*              window;
@property (strong, nonatomic) ErlebARViewController* viewController;

@end
