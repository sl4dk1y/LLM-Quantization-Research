from quantbench import monitor


def test_macos_vm_stat_keeps_page_categories_disjoint(monkeypatch):
    vm_stat = """Mach Virtual Memory Statistics: (page size of 16384 bytes)
Pages free:                               100.
Pages active:                             200.
Pages inactive:                           300.
Pages speculative:                         20.
Pages wired down:                         400.
Pages purgeable:                            5.
Pages occupied by compressor:             50.
File-backed pages:                         70.
"""

    def command(argv, timeout=3.0):
        if argv[-1].endswith("vm_stat"):
            return 0, vm_stat
        if argv[-1:] == ["-Q"] and argv[-2].endswith("memory_pressure"):
            return 0, "System-wide memory free percentage: 48%\n"
        return 1, "not used in this test"

    monkeypatch.setattr(monitor, "_command", command)
    monkeypatch.setattr("quantbench.platforms.macos.process_metrics", lambda _pid: {
        "physical_memory_bytes": 16 * 1024**3, "physical_footprint_bytes": None,
        "thermal_state": "nominal", "swap_used_bytes": 0,
    })
    result = monitor.snapshot()

    assert result["vm_page_size_bytes"] == 16384
    assert result["free_pages_bytes"] == 100 * 16384
    assert result["active_pages_bytes"] == 200 * 16384
    assert result["inactive_pages_bytes"] == 300 * 16384
    assert result["wired_pages_bytes"] == 400 * 16384
    assert result["compressor_physical_bytes"] == 50 * 16384
    assert result["purgeable_pages_bytes"] == 5 * 16384
    assert result["speculative_pages_bytes"] == 20 * 16384
    assert result["available_memory_estimate_bytes"] == 620 * 16384
    assert result["file_backed_pages_bytes"] == 70 * 16384
    assert result["reclaimable_memory_estimate_bytes"] == 175 * 16384
    assert result["system_used_bytes"] == 16 * 1024**3 - 100 * 16384
    assert result["memory_pressure"] == {"free_percent": 48, "category": None}


def test_swap_delta_accumulates_from_single_block_baseline(monkeypatch):
    from quantbench.platforms import macos

    swap = iter((110, 175))
    monkeypatch.setattr(monitor, "snapshot", lambda _pid=None: {
        "swap_used_bytes": next(swap), "runtime_active": True, "platform": "macOS",
        "safety_model": "apple_fully_reclaimable_estimate", "pressure_notifications_available": True,
        "memory_pressure": {}, "missing": {}})
    events = type("Pressure", (), {"available": True, "state": "normal", "last_error": None})()
    adapter = macos.MacOSAdapter()
    first = adapter.safety_snapshot(pid=123, swap_baseline_bytes=100, pressure_events=events)
    second = adapter.safety_snapshot(pid=123, swap_baseline_bytes=100, pressure_events=events)
    assert first["swap_delta_bytes"] == 10
    assert second["swap_delta_bytes"] == 75
