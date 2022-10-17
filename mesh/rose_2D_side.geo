// Gmsh project: created with gmsh-3.0.6-Windows64
boxdim =90;
column_height = 1.8;
gridsize_x = 0.6;
gidsize_y = 0.6;

ballast_h = 0.5;
layer_1_h = 1;
layer_2_h =2;

// Create ballast
Point(1) = {0, 0, 0, gridsize_x};
Point(2) = {boxdim, 0, 0, gridsize_x};
Point(3) = {boxdim, ballast_h, 0, gridsize_x};
Point(4) = {0, ballast_h, 0, gridsize_x};

Line(5) = {1, 2};
Line(6) = {2, 3};
Line(7) = {3, 4};
Line(8) = {4, 1};


// Create layer 1

Point(9) = {boxdim, -layer_1_h, 0, gridsize_x};
Point(10) = {0, -layer_1_h, 0, gridsize_x};

Line(11) = {2, 9};
Line(12) = {9, 10};
Line(13) = {10, 1};

// Create layer 2

Point(14) = {boxdim, -layer_1_h - layer_2_h, 0, gridsize_x};
Point(15) = {0, -layer_1_h - layer_2_h , 0, gridsize_x};


Line(16) = {9, 14};
Line(17) = {14, 15};
Line(18) = {15, 10};


// ballast surface
Curve Loop(19) = {5, 6, 7, 8};
Plane Surface(20) = {19};

// layer 1  surface
Curve Loop(21) = {11, 12, 13, 5};
Plane Surface(22) = {21};

// layer 2  surface
Curve Loop(23) = {16, 17, 18, -12};
Plane Surface(24) = {23};

Transfinite Line{7} = boxdim / gridsize_x + 1;

Physical Surface("embankment") = {20};
Physical Surface("soil1") = {-22};
Physical Surface("soil2") = {-24};
Physical Line("rose") = {-7};
