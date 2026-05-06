// Builds six extruded L-sections at hexagon vertices.
// Inputs expected from main file: hexRadius, lc, targetArea

lThickness = 0.03;
// L area is 4*lHalf*lThickness - lThickness^2.
lHalf = (targetArea + lThickness^2) / (4 * lThickness);

Macro MakeLSection
	p1 = newp; Point(p1) = {xc - lHalf, yc - lHalf, 0, lc};
	p2 = newp; Point(p2) = {xc + lHalf, yc - lHalf, 0, lc};
	p3 = newp; Point(p3) = {xc + lHalf, yc - lHalf + lThickness, 0, lc};
	p4 = newp; Point(p4) = {xc - lHalf + lThickness, yc - lHalf + lThickness, 0, lc};
	p5 = newp; Point(p5) = {xc - lHalf + lThickness, yc + lHalf, 0, lc};
	p6 = newp; Point(p6) = {xc - lHalf, yc + lHalf, 0, lc};

	l1 = newl; Line(l1) = {p1, p2};
	l2 = newl; Line(l2) = {p2, p3};
	l3 = newl; Line(l3) = {p3, p4};
	l4 = newl; Line(l4) = {p4, p5};
	l5 = newl; Line(l5) = {p5, p6};
	l6 = newl; Line(l6) = {p6, p1};

	cl = newll; Curve Loop(cl) = {l1, l2, l3, l4, l5, l6};
	sL = news; Plane Surface(sL) = {cl};
Return

xc = hexRadius * Cos(0 * Pi / 3); yc = hexRadius * Sin(0 * Pi / 3); Call MakeLSection;
out1[] = Extrude {0, 0, 1} { Surface{sL}; };
xc = hexRadius * Cos(1 * Pi / 3); yc = hexRadius * Sin(1 * Pi / 3); Call MakeLSection;
out2[] = Extrude {0, 0, 1} { Surface{sL}; };
xc = hexRadius * Cos(2 * Pi / 3); yc = hexRadius * Sin(2 * Pi / 3); Call MakeLSection;
out3[] = Extrude {0, 0, 1} { Surface{sL}; };
xc = hexRadius * Cos(3 * Pi / 3); yc = hexRadius * Sin(3 * Pi / 3); Call MakeLSection;
out4[] = Extrude {0, 0, 1} { Surface{sL}; };
xc = hexRadius * Cos(4 * Pi / 3); yc = hexRadius * Sin(4 * Pi / 3); Call MakeLSection;
out5[] = Extrude {0, 0, 1} { Surface{sL}; };
xc = hexRadius * Cos(5 * Pi / 3); yc = hexRadius * Sin(5 * Pi / 3); Call MakeLSection;
out6[] = Extrude {0, 0, 1} { Surface{sL}; };

cs1 = out1[1];
cs2 = out2[1];
cs3 = out3[1];
cs4 = out4[1];
cs5 = out5[1];
cs6 = out6[1];
