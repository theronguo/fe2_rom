// Builds six extruded disks at hexagon vertices.
// Inputs expected from main file: hexRadius, targetArea

holeRadius = Sqrt(targetArea / Pi);

Disk(1) = {hexRadius * Cos(0 * Pi / 3), hexRadius * Sin(0 * Pi / 3), 0, holeRadius, holeRadius};
Disk(2) = {hexRadius * Cos(1 * Pi / 3), hexRadius * Sin(1 * Pi / 3), 0, holeRadius, holeRadius};
Disk(3) = {hexRadius * Cos(2 * Pi / 3), hexRadius * Sin(2 * Pi / 3), 0, holeRadius, holeRadius};
Disk(4) = {hexRadius * Cos(3 * Pi / 3), hexRadius * Sin(3 * Pi / 3), 0, holeRadius, holeRadius};
Disk(5) = {hexRadius * Cos(4 * Pi / 3), hexRadius * Sin(4 * Pi / 3), 0, holeRadius, holeRadius};
Disk(6) = {hexRadius * Cos(5 * Pi / 3), hexRadius * Sin(5 * Pi / 3), 0, holeRadius, holeRadius};

out1[] = Extrude {0, 0, 1} { Surface{1}; };
out2[] = Extrude {0, 0, 1} { Surface{2}; };
out3[] = Extrude {0, 0, 1} { Surface{3}; };
out4[] = Extrude {0, 0, 1} { Surface{4}; };
out5[] = Extrude {0, 0, 1} { Surface{5}; };
out6[] = Extrude {0, 0, 1} { Surface{6}; };

cs1 = out1[1];
cs2 = out2[1];
cs3 = out3[1];
cs4 = out4[1];
cs5 = out5[1];
cs6 = out6[1];
