// Uniform-grid neighbor list. Cell size is chosen by the caller (engine/src/sim.ts) to be at
// least r_c + w_c (largest interaction range), so any pair within range of each other lands in
// the same cell or one of the 26 surrounding cells — a purely geometric guarantee that holds for
// whatever positions are current at rebuild time. Periodic in x,y (cell_of clamps rather than
// wraps; wrapping across the boundary happens in the 3x3x3 neighbor walk in forces.wgsl, not here).
//
// Build is entirely on the GPU (count -> prefix sum -> fill) so a full step never needs a
// CPU<->GPU readback round trip: with the drift test alone re-binning 200 lipids for 2000 steps,
// a per-step mapAsync stall would dominate wall time far more than the physics itself.
struct Grid { dims: vec4<u32>, box: vec4<f32> };

@group(0) @binding(0) var<uniform> G: Grid;
@group(0) @binding(1) var<storage, read> pos: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read_write> counts: array<atomic<u32>>;
@group(0) @binding(3) var<storage, read_write> cells: array<u32>;
@group(0) @binding(4) var<storage, read_write> cellStart: array<u32>;
@group(0) @binding(5) var<storage, read_write> cursor: array<atomic<u32>>;

fn cell_of(p: vec3<f32>) -> u32 {
  let n = vec3<f32>(f32(G.dims.x), f32(G.dims.y), f32(G.dims.z));
  let f = floor((p / G.box.xyz) * n);
  let c = clamp(vec3<u32>(max(f, vec3<f32>(0.0))), vec3<u32>(0u), G.dims.xyz - vec3<u32>(1u));
  return c.x + G.dims.x * (c.y + G.dims.y * c.z);
}

@compute @workgroup_size(64)
fn clear_counts_main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  let ncells = G.dims.x * G.dims.y * G.dims.z;
  if (i >= ncells) { return; }
  atomicStore(&counts[i], 0u);
}

@compute @workgroup_size(64)
fn count_main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  if (i >= arrayLength(&pos)) { return; }
  atomicAdd(&counts[cell_of(pos[i].xyz)], 1u);
}

// Serial exclusive prefix sum over cell counts. Runs as a single invocation: the cell count
// (a few hundred to a few thousand for the box sizes this engine targets) makes an O(ncells)
// sequential loop cheap next to the O(N) passes around it, and it keeps the whole rebuild on the
// GPU timeline with no CPU readback.
@compute @workgroup_size(1)
fn prefix_main() {
  let ncells = G.dims.x * G.dims.y * G.dims.z;
  var acc = 0u;
  for (var c = 0u; c < ncells; c = c + 1u) {
    cellStart[c] = acc;
    atomicStore(&cursor[c], acc);
    acc = acc + atomicLoad(&counts[c]);
  }
  cellStart[ncells] = acc;
}

@compute @workgroup_size(64)
fn fill_main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  if (i >= arrayLength(&pos)) { return; }
  let c = cell_of(pos[i].xyz);
  let slot = atomicAdd(&cursor[c], 1u);
  cells[slot] = i;
}
