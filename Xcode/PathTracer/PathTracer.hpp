//
//  PathTracer.hpp
//  PathTracer
//
//  Created by Federico Saldarini on 4/16/20.
//  Copyright © 2020 Federico Saldarini. All rights reserved.
//

#ifndef PathTracer_hpp
#define PathTracer_hpp

#include <string>

int renderScene(int sceneId, std::string path, int width, int height, int raysPerPixel, int maxBouncesPerRay);


#endif /* PathTracer_hpp */
