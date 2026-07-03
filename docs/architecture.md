# Architecture

The canonical runtime is in `src/`.

- `bogvm`: ISA, assembler, machine, receipts, and strict object formats.
- `tsos`: process model, bootloader, execution drivers, kernel tick, allocator, and IPC records.
- `tscli`: trustworthy command-line surface.
- `tsfield`: shared Allen-Cahn runtime.
- `tstopology`: typed topology classifier and graph serialization.

The kernel tick order is:

1. step the canonical field runtime when enabled;
2. sample measured field values at process positions;
3. ask the configured execution driver how many VM instructions are allowed;
4. execute eligible process VM instructions in PID order;
5. collect process output events from VM receipts;
6. synchronize process status from halted/faulted machine state;
7. update spatial allocation using `WeightedSpatialAllocator`;
8. clear and recompute IPC records;
9. emit a kernel receipt with field measurements and process status.

Wave timing is separated from VM semantics by the execution-driver interface. The default `ClockedExecutionDriver` permits one instruction per eligible process per tick. `WaveThresholdExecutionDriver` reads the measured field sample at each process position and permits execution only when the sample is above its threshold.
