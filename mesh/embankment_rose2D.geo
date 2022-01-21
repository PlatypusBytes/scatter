// Gmsh project: created with gmsh-3.0.6-Windows64

emb_top_w_1 = 0.75;
emb_top_w_2 = 1.25;
emb_h = 0.5;
emb_bot_w = 2.5;


x_size = 10;
z_size = 78;
y_size_1 = 1;
y_size_2 = 4;

grid_size_z_emb = 0.6;
grid_size_x_emb = 0.75;
grid_size_y_emb = 1.0;

grid_size_z_soil = 1.0;

// create 2D embankement
Point(1) = {0.0, emb_h, 0, grid_size_x_emb};
Point(2) = {emb_top_w_1, emb_h, 0, grid_size_x_emb};
Point(3) = {emb_top_w_1 + emb_top_w_2, emb_h, 0, grid_size_x_emb};
Point(4) = {emb_bot_w, 0.0, 0, grid_size_x_emb};
Point(5) = {0, 0, 0, grid_size_x_emb};

Line(6) = {1, 2};
Line(7) = {2, 3};
Line(8) = {3, 4};
Line(9) = {4, 5};
Line(10) = {5, 1};

// create 2D soil layering
// layer 1
Point(11) = {x_size, 0, 0, grid_size_x_emb};
Point(12) = {x_size, -y_size_1, 0, grid_size_x_emb};
Point(13) = {0, -y_size_1, 0, grid_size_x_emb};

Line(14) = {4, 11};
Line(15) = {11, 12};
Line(16) = {12, 13};
Line(17) = {13, 5};

// layer 2
Point(18) = {x_size, -y_size_1 -y_size_2, 0, grid_size_x_emb};
Point(19) = {0, -y_size_1 -y_size_2, 0, grid_size_x_emb};

Line(20) = {12, 18};
Line(21) = {18, 19};
Line(22) = {19, 13};

// embankment surface
Curve Loop(23) = {6, 7, 8, 9, 10};
Plane Surface(24) = {23};

// layer 1  surface
Curve Loop(25) = {14, 15, 16, 17, -9};
Plane Surface(26) = {25};

// layer 2  surface
Curve Loop(27) = {20, 21, 22, -16};
Plane Surface(28) = {27};

//Transfinite Surface{24};

//Transfinite Surface{26};

//Transfinite Surface{28};



//Physical Line("rose") = {newRail[1]};
Physical Surface("embankment") = {24};
Physical Surface("soil1") = {26};
Physical Surface("soil2") = {28};

//Recombine Surface{28};
//Recombine Surface{24};
//Recombine Surface{26};



