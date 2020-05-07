//
//  Scene.hpp
//  PathTracer
//
//  Created by Federico Saldarini on 4/16/20.
//  Copyright © 2020 Federico Saldarini. All rights reserved.
//

#ifndef Scene_hpp
#define Scene_hpp

#include "HittableVector.hpp"
#include "Camera.hpp"
#include <utility>

std::pair<HittableVector*, Camera*> generateSimpleScene(float aspect);
//HittableVector generateComplexScene();

#endif /* Scene_hpp */
