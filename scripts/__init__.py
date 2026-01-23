"""
Green AI Power Monitoring System
================================

Article 4: Green AI Research
Dr. Park - Systems Engineering

A high-frequency GPU power monitoring system with:
- NVIDIA NVML integration for accurate power measurement
- 10Hz sampling (100ms intervals)
- Thread-safe concurrent access
- Energy attribution for overlapping requests
- Comprehensive edge case handling

Usage:
------
    from green_ai.power_monitor import PowerMonitor, EnergyAttributionAlgorithm

    # Create and start monitor
    monitor = PowerMonitor(sample_rate_hz=10)
    monitor.start()

    # Track inference requests
    request_id = monitor.begin_request(metadata={"model": "llama-7b"})
    # ... run inference ...
    attribution = monitor.end_request(request_id)
    print(f"Energy consumed: {attribution.attributed_energy_joules:.2f}J")

    # Export data
    monitor.export_json("experiment_data.json")
    monitor.stop()

Context Manager:
---------------
    with PowerMonitor(sample_rate_hz=10) as monitor:
        req = monitor.begin_request()
        # ... inference ...
        attr = monitor.end_request(req)
"""

from .power_monitor import (
    PowerMonitor,
    PowerSample,
    RequestEnergyAttribution,
    EnergyAttributionAlgorithm,
    MonitorState,
    PowerMonitorValidator
)

from .edge_cases import (
    EdgeCaseManager,
    IdlePowerTracker,
    OverlapResolver,
    MeasurementGapHandler,
    ThermalThrottleDetector,
    PowerLimitDetector,
    PowerState
)

from .schemas import (
    JSON_SCHEMA,
    CSV_POWER_SAMPLES_SCHEMA,
    CSV_ATTRIBUTIONS_SCHEMA,
    validate_json_export,
    get_json_schema
)

__version__ = "1.0.0"
__author__ = "Dr. Park - Green AI Research Team"

__all__ = [
    # Core monitoring
    "PowerMonitor",
    "PowerSample",
    "RequestEnergyAttribution",
    "EnergyAttributionAlgorithm",
    "MonitorState",
    "PowerMonitorValidator",

    # Edge case handling
    "EdgeCaseManager",
    "IdlePowerTracker",
    "OverlapResolver",
    "MeasurementGapHandler",
    "ThermalThrottleDetector",
    "PowerLimitDetector",
    "PowerState",

    # Schemas
    "JSON_SCHEMA",
    "CSV_POWER_SAMPLES_SCHEMA",
    "CSV_ATTRIBUTIONS_SCHEMA",
    "validate_json_export",
    "get_json_schema",
]
