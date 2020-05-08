//
//  main.cpp
//  PathTracer
//
//  Created by Federico Saldarini on 4/16/20.
//  Copyright © 2020 Federico Saldarini. All rights reserved.
//

#include <iostream>
#include "PathTracer.hpp"

int main(int argc, const char * argv[]) {
  // 1 = simple scene; 2 = complex scene (takes longer to render):
  int sceneId = 2;
  renderScene(sceneId, "", 320, 200, 100, 50);
  return 0;
}
