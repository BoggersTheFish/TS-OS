# Architecture

The canonical runtime is in `src/`.

- `bogvm`: ISA, assembler, machine, receipts, and strict object formats.
- `tsos`: process model, bootloader, execution drivers, kernel tick, allocator, and IPC records.
- `tscli`: trustworthy command-line surface.
- `tsfield`: shared Allen-Cahn runtime.
- `tstopology`: typed topology classifier and graph serialization.

The kernel tick order is:

1. execute eligible process VM instructions in PID order;
2. collect process output events from VM receipts;
3. synchronize process status from halted/faulted machine state;
4. update spatial allocation using `WeightedSpatialAllocator`;
5. clear and recompute IPC records;
6. emit a kernel receipt.

Wave timing is separated from VM semantics by the execution-driver interface. The default `ClockedExecutionDriver` permits one instruction per eligible process per tick.
