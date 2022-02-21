meshsize2 = meshsize/6;

Point(1) = {0,0,0,meshsize};
Point(2) = {1,0,0,meshsize};
Point(3) = {0,1,0,meshsize};
Point(4) = {-1,0,0,meshsize};
Point(5) = {0,-1,0,meshsize};

Point(6) = {0,0,0.3+theta4,meshsize2};
Point(7) = {0,1,0.3+theta4,meshsize2};
Point(8) = {-1,0,0.3+theta4,meshsize};
Point(9) = {0,-1,0.3+theta4,meshsize2};
Point(29) = {-0.25,0,0.3+theta4,meshsize};
Point(30) = {0.25,0,0.3+theta4,meshsize};

Point(10) = {0,0,0.7+theta3,meshsize2};
Point(11) = {0,1,0.7+theta3,meshsize2};
Point(12) = {-1,0,0.7+theta3-theta2,meshsize};
Point(13) = {0,-1,0.7+theta3,meshsize2};
Point(28) = {0.225,0,0.7+theta3,meshsize};

Point(21) = {0,0,1,meshsize};
Point(22) = {1,0,1,meshsize};
Point(23) = {0,1,1,meshsize};
Point(24) = {-1,0,1,meshsize};
Point(25) = {0,-1,1,meshsize};

Point(26) = {1,0,0.3+theta4,meshsize};
Point(27) = {1,0,0.7+theta3,meshsize};

Circle(1) = {2,1,3};
Circle(2) = {3,1,4};
Circle(3) = {4,1,5};
Circle(4) = {5,1,2};

Circle(5) = {7,6,8};
Circle(6) = {8,6,9};


Ellipsis(8) = {11,10,12};
Ellipsis(9) = {12,10,13};


Circle(11) = {22,21,23};
Circle(12) = {23,21,24};
Circle(13) = {24,21,25};
Circle(14) = {25,21,22};

//+
Line(15) = {22, 27};
//+
Line(16) = {27, 26};
//+
Line(17) = {26, 2};
//+
Line(18) = {5, 1};
//+
Line(20) = {1, 3};
//+
Line(22) = {9, 6};


//+
Line(23) = {6, 7};
//+
Line(24) = {13, 10};
//+
Line(25) = {10, 11};
//+
Line(26) = {25, 21};
//+
Line(27) = {21, 23};
//+
Ellipse(28) = {13, 10, 10, 27};
//+
Ellipse(29) = {9, 6, 6, 26};
//+
Ellipse(30) = {7, 6, 6, 26};
//+
Ellipse(31) = {11, 10, 10, 27};
//+
Line(32) = {25, 13};
//+
Line(33) = {13, 9};
//+
Line(34) = {9, 5};
//+
Line(35) = {7, 3};
//+
Line(36) = {11, 7};
//+
Line(37) = {23, 11};
//+
Line(38) = {24, 12};
//+
Line(39) = {12, 8};
//+
Line(40) = {8, 4};
//+
Curve Loop(1) = {24, 25, 36, -23, -22, -33};
//+
Plane Surface(1) = {1};
//+
Curve Loop(2) = {26, 27, 37, -25, -24, -32};
//+
Plane Surface(2) = {2};
//+
Curve Loop(3) = {34, 18, 20, -35, -23, -22};
//+
Plane Surface(3) = {3};
//+
Curve Loop(4) = {13, 26, 27, 12};
//+
Plane Surface(4) = {4};
//+
Curve Loop(5) = {9, 24, 25, 8};
//+
Plane Surface(5) = {5};
//+
Curve Loop(6) = {6, 22, 23, 5};
//+
Plane Surface(6) = {6};
Point{29} In Surface{6};
//+
Curve Loop(7) = {3, 18, 20, 2};
//+
Plane Surface(7) = {7};
//+
Curve Loop(8) = {11, -27, -26, 14};
//+
Plane Surface(8) = {8};
//+
Curve Loop(9) = {1, -20, -18, 4};
//+
Plane Surface(9) = {9};
//+
Curve Loop(10) = {28, -31, -25, -24};
//+
Plane Surface(10) = {10};
Point {28} In Surface{10};
//+
Curve Loop(11) = {30, -29, 22, 23};
//+
Plane Surface(11) = {11};

Point{30} In Surface{11};
//+
Curve Loop(12) = {16, -30, -36, 31};
//+
Surface(12) = {12};
//+
Curve Loop(13) = {30, 17, 1, -35};
//+
Surface(13) = {13};
//+
Curve Loop(14) = {11, 37, 31, -15};
//+
Surface(14) = {14};
//+
Curve Loop(15) = {5, 40, -2, -35};
//+
Surface(15) = {15};
//+
Curve Loop(16) = {6, 34, -3, -40};
//+
Surface(16) = {16};
//+
Curve Loop(17) = {29, 17, -4, -34};
//+
Surface(17) = {17};
//+
Curve Loop(18) = {39, 6, -33, -9};
//+
Surface(18) = {18};
//+
Curve Loop(19) = {28, 16, -29, -33};
//+
Surface(19) = {19};
//+
Curve Loop(20) = {8, 39, -5, -36};
//+
Surface(20) = {20};
//+
Curve Loop(21) = {38, 9, -32, -13};
//+
Surface(21) = {21};
//+
Curve Loop(22) = {12, 38, -8, -37};
//+
Surface(22) = {22};
//+
Curve Loop(23) = {14, 15, -28, -32};
//+
Surface(23) = {23};
//+
Surface Loop(1) = {17, 13, 9, 11, 3};
//+
Volume(1) = {1};
//+
Surface Loop(2) = {6, 7, 16, 15, 3};
//+
Volume(2) = {2};
//+
Surface Loop(3) = {18, 20, 1, 6, 5};
//+
Volume(3) = {3};
//+
Surface Loop(4) = {4, 21, 22, 5, 2};
//+
Volume(4) = {4};
//+
Surface Loop(5) = {8, 14, 23, 10, 2};
//+
Volume(5) = {5};
//+
Surface Loop(6) = {1, 10, 11, 19, 12};
//+
Volume(6) = {6};
//+
Physical Surface("top", 41) = {4, 8};
//+
Physical Surface("bottom", 42) = {7, 9};
//+
Physical Surface("side", 43) = {18, 21, 16, 17, 19, 23, 14, 13, 12, 20, 22, 15};
//+
Physical Volume("filling", 44) = {3};
//+
Physical Volume("rest", 45) = {4, 5, 6, 1, 2};









