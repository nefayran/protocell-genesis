
## File size: 400–600 lines

A hard project rule, set by the project owner on 2026-08-18 after `soup/src/sim.ts` grew to 2351
lines and `viewer/run.ts` to 1483.

- The target file size is **400–600 lines**. A file that goes past 600 is split by responsibility,
  not by line count: buffers and their ownership, kernels and their bindings, the integration step,
  bond chemistry, measurements, state input/output, each in its own module.
- New code is not added to a file that is already over the limit: split first, then add.
- A split is checked by the existing tests and must be a **pure move**: valence, binding of growth
  to the catalyst, the force check against brute force, the closure detector and the viewer tests
  must give the same numbers before and after.

Size check: `find soup engine chem viewer verify tests -name '*.ts' -o -name '*.wgsl' | xargs wc -l | sort -rn | head`
