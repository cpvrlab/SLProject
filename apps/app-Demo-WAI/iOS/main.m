//
//  main.m
//  comgr
//
//  Created by Marcus Hudritsch on 30.11.11.
//  Copyright (c) 2011 __MyCompanyName__. All rights reserved.
//

#import <UIKit/UIKit.h>

#import "ErlebARAppDelegate.h"

int main(int argc, char* argv[])
{
    NSString* appDelegateClassName;
    @autoreleasepool
    {
        appDelegateClassName = NSStringFromClass([ErlebARAppDelegate class]);
    }
    return UIApplicationMain(argc, argv, nil, appDelegateClassName);
}
