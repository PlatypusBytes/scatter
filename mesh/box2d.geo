// Gmsh project: created with gmsh-3.0.6-Windows64
boxdim = 120;
column_height = 1.8;
gridsize = 0.6;

// Create 2D square mesh
Point(1) = {0, 0, 0, gridsize};
Point(2) = {boxdim, 0, 0, gridsize};
Point(3) = {boxdim, column_height, 0, gridsize};
Point(4) = {0, column_height, 0, gridsize};

Line(5) = {1, 2};
Line(6) = {2, 3};
Line(7) = {3, 4};
Line(8) = {4, 1};

Line Loop(9) = {5, 6, 7, 8};
Plane Surface(10) = 9;

// divide de lines for the elements
Transfinite Line{6, 8} = column_height / gridsize + 1;
Transfinite Line{5, 7} = boxdim / gridsize + 1;

Transfinite Surface{10};
Recombine Surface{10};

// Make 3D by extrusion
//newEntities[] = Extrude {0, 0, -boxdim}{Surface{10};
  //                                      Layers{boxdim / gridsize};
    //                                    Recombine;
      //                                  };

// Define the volume

Physical Surface("solid") = {10};
 Physical Line("rose") = {7};
//Physical Volume("solid") = {newEntities[1]};