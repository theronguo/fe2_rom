// Builds six extruded hollow disks (annuli) at hexagon vertices.
// Inputs expected from main file: hexRadius, targetArea

innerRadius = 0.03;
outerRadius = Sqrt(innerRadius^2 + targetArea / Pi);

For i In {0:5}
	xc = hexRadius * Cos(i * Pi / 3);
	yc = hexRadius * Sin(i * Pi / 3);

	// Outer and inner circles for annulus profile at each location.
	cOut = newc; Circle(cOut) = {xc, yc, 0, outerRadius, 0, 2 * Pi};
	cIn = newc; Circle(cIn) = {xc, yc, 0, innerRadius, 0, 2 * Pi};

	clOut = newll; Curve Loop(clOut) = {cOut};
	clIn = newll; Curve Loop(clIn) = {cIn};
	sA = news; Plane Surface(sA) = {clOut, clIn};

	out[] = Extrude {0, 0, 1} { Surface{sA}; };
	If (i == 0)
		cs1 = out[1];
	ElseIf (i == 1)
		cs2 = out[1];
	ElseIf (i == 2)
		cs3 = out[1];
	ElseIf (i == 3)
		cs4 = out[1];
	ElseIf (i == 4)
		cs5 = out[1];
	Else
		cs6 = out[1];
	EndIf
EndFor
