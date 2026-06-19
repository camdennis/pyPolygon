Create a plan for a python codebase that performs molecular dynamics simulations for polygons with rounded edges. I'll include a great deal of details about this, but I will not include all of the details. It's your job to ask me questions until we end up with something coherent. Please avoid unnecessary tangents when possible. Unless I specifically approve a function, please mark it with my name so that I can check them off one at a time as I go through the code looking for inconsistencies. You need to make sure I'm keeping up with what the code is doing and I'm not falling behind. Scold me.

A rounded polygon is an object I will now describe. It is comprised of a backbone polygon with n edges and vertices. The polygon's edges and vertices are labeled in counter-clockwise fashion, v_k for k in [1, n]. It is not necessarily convex. For each vertex, k, we  place a circle of radius rho at a point z_k such that the circle kisses both edges on each side of the vertex (at points a_k^- and a_k^+). When a corner is non-convex(convex), z_k is outside(inside) the backbone polygon.The rounded polygon is defined by asserting that the circular segment from a_k^- to a_k^+ replaces the boundary of the backbone polygon between a_k^- and a_k^+. (Note that I don't love this phrasing and I haven't figured out exactly how to convey that we choose the arc that keeps the boundary of the rounded polygon continuous and smooth.)

We need to compute the area of the overlap between two backbone polygons overlap as well as the derivative of this area with respect to vertex positions.

We additionally let the circles of radius rho centered at z_k to be repulsive.

For a packing of N rounded polygons, we define the energy as:

U = sum (2 * overlap area / (target area of polygon 1 + target area of polygon 2))^2 - (K_adh / 2) * (2 * (distance between points of intersection / (target perimeter of polygon 1 + target perimeter of polygon 2)))^2 + K_A * sum (1 - area of polygon / target area)^2 + K_P * sum (1 - perimeter of polygon / target perimeter)^2

From this we also compute a force.

Here's how we do it:

1) For each vertex, find the vertices that are within a distance D and check if their edges/rounded circular segments intersect. Record the intersections.

2) For each intersection, if you follow one of the polygons in ccw direction, you will eventually find another intersection between those two polygons. Record these as well and call them outersections (or something)

3) Loop over all edges and then over a subset of the intersections and check if that entire edge is between the intersections of two polygons: accumulate the terms of energy and force

4) Loop over all intersections and accumulate the terms associated with that intersection (the segments of the rounded polygons connected to that intersection and outersection)

5) Get the energy and force terms for the areas and perimeters of the polygons

From here we minimize. Our boundary conditions are periodic with lattice vectors determining the positions. We'll probably want to define some sort of way to keep track of this now. There should be a flag for normal and latticeVector versions of the packing object and wrap should check the type before proceeding.

Let's build it in this order:

1) Make an equilateral polygon with n sides. This should be accomplished with a gradient descent or FIRE algorithm. This can be a type of energy to consider (enum) "eqSoftBody" which is just a spring with rest length equal to the edge length. There should also be a spring on the area term.

2) Transform this polygon into a rounded polygon with an input rho and compute the a^+ and a^- pieces as well as z and the angle it sweeps psi

3) Find the area and perimeter of the rounded polygons

4) Get the neighbors of each vertex with an appropriate ball size

5) Compute intersections between edges and arcs (ee, ea, ae, aa)

6) Compute forces and energies and overlap areas

7) Minimize these forces with FIRE updating the neighbors and ball size and then intersections at every step

8) Only update the neighbors when the vertices move by a critical amount

9) Add lattice vectors

10) Compute the stress tensor and stiffness tensor (need to figure that out later)

11) Create a function to set and the modify the area of each element.