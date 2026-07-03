# Determinism v0

For identical executable bytes, initial registers, memory, machine configuration, and package metadata, BOGVM v0 produces identical final machine state and receipt hashes.

Deterministic guarantees:

- canonical JSON serialization uses sorted keys and compact separators;
- program hashes are derived only from entry point and canonical instruction stream;
- VM arithmetic is signed 32-bit integer arithmetic with defined wraparound;
- division by zero, invalid PC, and memory bounds produce typed faults;
- kernel process execution order is sorted by PID;
- the default execution driver permits one instruction per process per kernel tick;
- receipt hashes form a SHA-256 hash chain.

Non-guarantees in this milestone:

- floating-point Allen-Cahn evolution is centralized and tested for boundary semantics, but cross-platform bitwise equality is not claimed;
- wave-triggered execution is experimental timing only and does not redefine instruction meanings;
- prior cube/topology scripts are archived as experiments and are not claimed as autonomous discovery.
