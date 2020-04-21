//
//  Scene.hpp
//  PathTracer
//
//  Created by Federico Saldarini on 4/16/20.
//  Copyright © 2020 Federico Saldarini. All rights reserved.
//

#ifndef Scene_hpp
#define Scene_hpp

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "HittableVector.hpp"

HittableVector generateSimpleScene();
HittableVector generateComplexScene();

#endif /* Scene_hpp */
