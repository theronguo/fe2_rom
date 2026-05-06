SetFactory("OpenCASCADE");

// Global characteristic mesh size.
lc = 0.03;
Mesh.CharacteristicLengthMin = lc;
Mesh.CharacteristicLengthMax = lc;

// Hexagon radius for cross-section placement.
hexRadius = 0.5;

// Target area for each cross-section type.
targetArea = Pi * 0.05^2;

// Choose one cross-section generator file.
// Include "cross_sections/l_sections.geo";
// Include "cross_sections/disks.geo";
Include "cross_sections/hollow_disks.geo";

// Two slab volumes: z in [0, 0.1] and z in [0.9, 1.0].
Box(100) = {-0.6, -0.6, 0.0, 1.2, 1.2, 0.1};
Box(101) = {-0.6, -0.6, 0.9, 1.2, 1.2, 0.1};

// Merge slabs and extruded cross-sections into one OCC boolean union.
mergedVolumes[] = BooleanUnion {
	Volume{100, 101}; Delete;
} {
	Volume{cs1, cs2, cs3, cs4, cs5, cs6}; Delete;
};

Physical Volume("domain") = {mergedVolumes[]};
