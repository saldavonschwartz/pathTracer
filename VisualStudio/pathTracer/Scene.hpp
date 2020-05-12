//
//  Scene.hpp
//  PathTracer
//
//  Created by Federico Saldarini on 4/16/20.
//  Copyright © 2020 Federico Saldarini. All rights reserved.
//

#ifndef Scene_hpp
#define Scene_hpp

class Camera;
class BVHNode;

void generateScene(BVHNode**& bvh, Camera*& cam, float aspect);
void freeScene(BVHNode**& bvh, Camera*& cam);

#endif /* Scene_hpp */
