// Task 8: closed-vesicle detection by outside flood fill, the GPU counterpart of
// engine/src/closure.ts's occupancy()/enclosedVolume() -- see that file's module doc for the
// periodic-flood decision this kernel deliberately makes the SAME way (all six box faces are
// OPEN boundary for the flood itself; only occupancy_main wraps x,y, matching where a bead's
// influence actually reaches).
//
// Pipeline, run once per call from closure.ts's enclosedVolumeGpuDetailed:
//   clear_main       -- zero occ[] and outside[]
//   occupancy_main   -- one atomicOr pass over beads, walking only the cells within `radius` of
//                        each bead (not every cell against every bead)
//   seed_main        -- marks every EMPTY cell on one of the box's six faces as outside=1
//   (clear_changed_main, propagate_main) -- repeated from the host until a pass changes nothing:
//     propagate_main spreads outside=1 to any empty, not-yet-outside cell with an outside
//     neighbour (6-connected, bounds-checked, no periodic wrap) and raises `changed` whenever it
//     flips a cell -- in place on one buffer via atomics, which is safe here because the flip is
//     monotonic (0 -> 1 only, never reset), so a read seeing a slightly stale or slightly fresher
//     neighbour value within the same dispatch still yields a correct eventual result, only
//     possibly converging in fewer host-side iterations.
// The enclosed volume itself (count of empty, never-reached cells * cell^3) is computed on the
// CPU side after the final readback of occ[]/outside[] -- WGSL has no cross-workgroup reduction
// primitive as simple as a host-side sum over a small (thousands to ~1e6 cell) array.

struct Grid {
  dims: vec4<u32>, // nx, ny, nz, reach (reach = ceil(radius/cell))
  geom: vec4<f32>, // Lx, Ly, Lz, cell
  rad: vec4<f32>,  // radius, radius^2, unused, unused
};

@group(0) @binding(0) var<uniform> G: Grid;
@group(0) @binding(1) var<storage, read> pos: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read_write> occ: array<atomic<u32>>;
@group(0) @binding(3) var<storage, read_write> outside: array<atomic<u32>>;
@group(0) @binding(4) var<storage, read_write> changed: array<atomic<u32>>;

fn idx3(x: u32, y: u32, z: u32) -> u32 {
  return x + G.dims.x * (y + G.dims.y * z);
}

@compute @workgroup_size(64)
fn clear_main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let ncells = G.dims.x * G.dims.y * G.dims.z;
  let i = gid.x;
  if (i >= ncells) { return; }
  atomicStore(&occ[i], 0u);
  atomicStore(&outside[i], 0u);
}

@compute @workgroup_size(1)
fn clear_changed_main() {
  atomicStore(&changed[0], 0u);
}

// One pass over beads: for each bead, walk the (2*reach+1)^3 cells around it (bounded by `reach`
// = ceil(radius/cell), not the whole grid) and mark any whose CENTRE lies within `radius` -- x,y
// wrap periodically (matching the box's own periodicity, mi()-style minimum image), z does not.
@compute @workgroup_size(64)
fn occupancy_main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  if (i >= arrayLength(&pos)) { return; }
  let p = pos[i].xyz;
  let cell = G.geom.w;
  let reach = i32(G.dims.w);
  let nx = i32(G.dims.x);
  let ny = i32(G.dims.y);
  let nz = i32(G.dims.z);
  let Lx = G.geom.x;
  let Ly = G.geom.y;
  let r2 = G.rad.y;

  let cx = i32(floor(p.x / cell));
  let cy = i32(floor(p.y / cell));
  let cz = i32(floor(p.z / cell));

  for (var dz = -reach; dz <= reach; dz = dz + 1) {
    let gz = cz + dz;
    if (gz < 0 || gz >= nz) { continue; }
    let ddz = (f32(gz) + 0.5) * cell - p.z;
    for (var dy = -reach; dy <= reach; dy = dy + 1) {
      var gy = (cy + dy) % ny;
      if (gy < 0) { gy = gy + ny; }
      var ddy = (f32(gy) + 0.5) * cell - p.y;
      ddy = ddy - round(ddy / Ly) * Ly;
      for (var dx = -reach; dx <= reach; dx = dx + 1) {
        var gx = (cx + dx) % nx;
        if (gx < 0) { gx = gx + nx; }
        var ddx = (f32(gx) + 0.5) * cell - p.x;
        ddx = ddx - round(ddx / Lx) * Lx;
        let d2 = ddx * ddx + ddy * ddy + ddz * ddz;
        if (d2 <= r2) {
          atomicOr(&occ[idx3(u32(gx), u32(gy), u32(gz))], 1u);
        }
      }
    }
  }
}

// Marks every EMPTY cell on one of the box's six faces as outside=1. Open boundary: a cell at
// x=0 and a cell at x=nx-1 are two independent faces here, never connected to each other -- see
// the periodic-flood note in closure.ts.
@compute @workgroup_size(64)
fn seed_main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let ncells = G.dims.x * G.dims.y * G.dims.z;
  let i = gid.x;
  if (i >= ncells) { return; }
  let nx = G.dims.x;
  let ny = G.dims.y;
  let nz = G.dims.z;
  let z = i / (nx * ny);
  let rem = i - z * nx * ny;
  let y = rem / nx;
  let x = rem - y * nx;
  let boundary = x == 0u || x == nx - 1u || y == 0u || y == ny - 1u || z == 0u || z == nz - 1u;
  if (boundary && atomicLoad(&occ[i]) == 0u) {
    atomicStore(&outside[i], 1u);
  }
}

// One flood layer: any empty, not-yet-outside cell with a 6-connected outside neighbour (bounds-
// checked, no periodic wrap in the flood itself) becomes outside too, and raises `changed`.
// Dispatched repeatedly from the host until a pass raises no change -- see the module doc above
// for why in-place atomics on one buffer are safe for this monotonic update.
@compute @workgroup_size(64)
fn propagate_main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let ncells = G.dims.x * G.dims.y * G.dims.z;
  let i = gid.x;
  if (i >= ncells) { return; }
  if (atomicLoad(&occ[i]) != 0u) { return; }
  if (atomicLoad(&outside[i]) != 0u) { return; }
  let nx = G.dims.x;
  let ny = G.dims.y;
  let nz = G.dims.z;
  let z = i / (nx * ny);
  let rem = i - z * nx * ny;
  let y = rem / nx;
  let x = rem - y * nx;

  var found = false;
  if (x > 0u) { if (atomicLoad(&outside[idx3(x - 1u, y, z)]) != 0u) { found = true; } }
  if (!found && x + 1u < nx) { if (atomicLoad(&outside[idx3(x + 1u, y, z)]) != 0u) { found = true; } }
  if (!found && y > 0u) { if (atomicLoad(&outside[idx3(x, y - 1u, z)]) != 0u) { found = true; } }
  if (!found && y + 1u < ny) { if (atomicLoad(&outside[idx3(x, y + 1u, z)]) != 0u) { found = true; } }
  if (!found && z > 0u) { if (atomicLoad(&outside[idx3(x, y, z - 1u)]) != 0u) { found = true; } }
  if (!found && z + 1u < nz) { if (atomicLoad(&outside[idx3(x, y, z + 1u)]) != 0u) { found = true; } }

  if (found) {
    let old = atomicOr(&outside[i], 1u);
    if (old == 0u) {
      atomicOr(&changed[0], 1u);
    }
  }
}
