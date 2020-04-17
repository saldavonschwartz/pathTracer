//
//  Scene.hpp
//  PathTracer
//
//  Created by Federico Saldarini on 4/16/20.
//  Copyright © 2020 Federico Saldarini. All rights reserved.
//

#ifndef Scene_hpp
#define Scene_hpp

#include <memory>

class Hittable;

std::shared_ptr<Hittable> generateSimpleScene();
std::shared_ptr<Hittable> generateComplexScene();

#endif /* Scene_hpp */
