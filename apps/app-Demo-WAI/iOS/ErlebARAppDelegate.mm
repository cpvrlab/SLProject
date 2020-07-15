//#############################################################################
//  File:      ErlebARAppDelegate.mm
//  Purpose:
//  Author:    Michael Göttlicher
//  Date:      June 2020
//  Codestyle: https://github.com/cpvrlab/SLProject/wiki/SLProject-Coding-Style
//  Copyright: Marcus Hudritsch
//             This software is provide under the GNU General Public License
//             Please visit: http://opensource.org/licenses/GPL-3.0
//#############################################################################

#import "ErlebARAppDelegate.h"
#import "ErlebARViewController.h"

@implementation ErlebARAppDelegate

- (BOOL)application:(UIApplication*)application didFinishLaunchingWithOptions:(NSDictionary*)launchOptions
{
    self.window = [[UIWindow alloc] initWithFrame:[[UIScreen mainScreen] bounds]];
    // Override point for customization after application launch.
    if ([[UIDevice currentDevice] userInterfaceIdiom] == UIUserInterfaceIdiomPhone)
    {
        self.viewController = [[ErlebARViewController alloc] init:@"ViewController_iPhone"];
    }
    else
    {
        self.viewController = [[ErlebARViewController alloc] init:@"ViewController_iPad"];
    }
    self.viewController.view.hidden = NO;
    self.window.rootViewController  = self.viewController;
    [self.window makeKeyAndVisible];
    return YES;
}

- (void)applicationWillResignActive:(UIApplication*)application
{
    /*
    Sent when the application is about to move from active to inactive state.
    This can occur for certain types of temporary interruptions (such as an 
    incoming phone call or SMS message) or when the user quits the application 
    and it begins the transition to the background state.
    Use this method to pause ongoing tasks, disable timers, and throttle down 
    OpenGL ES frame rates. Games should use this method to pause the game.
    */
    printf("applicationWillResignActive\n");
    [[self viewController] appWillResignActive];
}

- (void)applicationDidEnterBackground:(UIApplication*)application
{
    /*
    Use this method to release shared resources, save user data, invalidate timers,
    and store enough application state information to restore your application to 
    its current state in case it is terminated later.
    If your application supports background execution, this method is called instead 
    of applicationWillTerminate: when the user quits.
    */
    printf("applicationDidEnterBackground\n");
    [[self viewController] appDidEnterBackground];
}

- (void)applicationWillEnterForeground:(UIApplication*)application
{
    /*
    Called as part of the transition from the background to the inactive state;
    here you can undo many of the changes made on entering the background.
    */
    printf("applicationWillEnterForeground\n");
    [[self viewController] appWillEnterForeground];
}

- (void)applicationDidBecomeActive:(UIApplication*)application
{
    /*
    Restart any tasks that were paused (or not yet started) while the application
    was inactive. If the application was previously in the background, 
    optionally refresh the user interface.
    */
    printf("applicationDidBecomeActive\n");
    [[self viewController] appDidBecomeActive];
}

- (void)applicationWillTerminate:(UIApplication*)application
{
    /*
    Called when the application is about to terminate.
    Save data if appropriate.
    See also applicationDidEnterBackground:.
    */
    printf("applicationWillTerminate\n");
    [[self viewController] appWillTerminate];
}

@end
