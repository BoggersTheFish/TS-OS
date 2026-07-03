import numpy as np

from bogvm.formats import BogPkg, read_json
from bogvm.receipts import verify_receipt_chain
from tsos import Kernel, ProcessStatus
from tsos.drivers import WaveThresholdExecutionDriver
from tsos.persistence import kernel_from_bogstate, kernel_to_bogstate


def boot_addition():
    kernel = Kernel(N=6)
    kernel.boot_package(BogPkg.from_json(read_json("examples/addition_process.bogpkg")))
    return kernel


def test_package_boot_creates_process_and_ticks_execute_vm():
    kernel = boot_addition()
    assert 0 in kernel.processes
    assert kernel.processes[0].status == ProcessStatus.READY
    kernel.tick()
    assert kernel.processes[0].machine.registers[0] == 13


def test_addition_end_to_end_halts_and_verifies():
    kernel = boot_addition()
    kernel.run(20)
    p = kernel.processes[0]
    assert p.machine.registers[2] == 23
    assert p.output_events == [23]
    assert p.status == ProcessStatus.HALTED
    assert verify_receipt_chain(p.machine.receipts)


def test_suspend_resume():
    kernel = boot_addition()
    kernel.tick()
    kernel.suspend(0)
    pc = kernel.processes[0].machine.pc
    kernel.tick()
    assert kernel.processes[0].machine.pc == pc
    kernel.resume(0)
    kernel.tick()
    assert kernel.processes[0].machine.pc == pc + 1


def test_zero_active_processes_unowned_space_and_no_stale_ipc():
    kernel = boot_addition()
    kernel.kill(0)
    kernel.tick()
    assert np.all(kernel.voxel_ownership == -1)
    assert kernel.ipc_records == {}


def test_ipc_cleared_when_pair_disappears():
    package = BogPkg.from_json(read_json("examples/addition_process.bogpkg"))
    kernel = Kernel(N=6)
    kernel.boot_package(package)
    p2 = package.processes[0]
    from tsos.kernel import Process
    from bogvm.machine import BOGMachine

    m = BOGMachine()
    m.load_program(p2.executable.program())
    kernel.processes[1] = Process(1, (1, 0, 0), 1, 1.0, 0.0, m, ProcessStatus.READY)
    kernel.tick()
    assert kernel.ipc_records
    kernel.kill(1)
    kernel.tick()
    assert kernel.ipc_records == {}


def test_field_runtime_steps_and_is_receipted():
    kernel = Kernel(N=6, field_enabled=True)
    kernel.boot_package(BogPkg.from_json(read_json("examples/addition_process.bogpkg")))
    receipt = kernel.tick()
    assert receipt["field"] is not None
    assert kernel.field_runtime is not None
    assert kernel.field_runtime.step_counter == 1
    assert receipt["field_samples"]["0"] > 0


def test_wave_threshold_driver_blocks_below_measured_field_threshold():
    kernel = Kernel(N=6, field_enabled=True, driver=WaveThresholdExecutionDriver(threshold=0.5))
    kernel.boot_package(BogPkg.from_json(read_json("examples/addition_process.bogpkg")))
    kernel.tick()
    assert kernel.processes[0].machine.pc == 0
    assert kernel.kernel_receipts[-1]["field_samples"]["0"] < 0.5


def test_wave_threshold_driver_executes_from_measured_field():
    kernel = Kernel(N=6, field_enabled=True, driver=WaveThresholdExecutionDriver(threshold=0.01))
    kernel.boot_package(BogPkg.from_json(read_json("examples/addition_process.bogpkg")))
    kernel.tick()
    assert kernel.processes[0].machine.pc == 1
    assert kernel.processes[0].machine.registers[0] == 13


def test_bogstate_persists_field_and_wave_driver():
    kernel = Kernel(N=6, field_enabled=True, driver=WaveThresholdExecutionDriver(threshold=0.01))
    kernel.boot_package(BogPkg.from_json(read_json("examples/addition_process.bogpkg")))
    kernel.tick()
    state = kernel_to_bogstate(kernel, "examples/addition_process.bogpkg")
    _, restored = kernel_from_bogstate(state)
    assert restored.field_runtime is not None
    assert restored.field_runtime.step_counter == 1
    assert isinstance(restored.driver, WaveThresholdExecutionDriver)
    assert restored.processes[0].machine.pc == 1
