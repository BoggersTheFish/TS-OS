import numpy as np

from bogvm.formats import BogPkg, read_json
from bogvm.receipts import verify_receipt_chain
from tsos import Kernel, ProcessStatus


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
