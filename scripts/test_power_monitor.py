"""
Test Suite for Power Monitoring System
======================================

Dr. Park - Systems Engineering Validation
Article 4: Green AI Research

Comprehensive tests for:
1. Basic power monitoring functionality
2. Energy attribution accuracy
3. Edge case handling
4. Thread safety
5. Data export validation
"""

import json
import os
import tempfile
import threading
import time
import unittest
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import MagicMock, patch

from power_monitor import (
    PowerMonitor,
    PowerSample,
    RequestEnergyAttribution,
    EnergyAttributionAlgorithm,
    PowerMonitorValidator,
    MonitorState
)
from edge_cases import (
    IdlePowerTracker,
    OverlapResolver,
    MeasurementGapHandler,
    ThermalThrottleDetector,
    PowerLimitDetector,
    EdgeCaseManager,
    PowerState
)


class TestPowerSample(unittest.TestCase):
    """Test PowerSample dataclass."""

    def test_sample_creation(self):
        """Test creating a power sample."""
        sample = PowerSample(
            timestamp=1000.0,
            power_watts=150.5,
            gpu_utilization=75.0,
            memory_used_bytes=4 * 1024**3,  # 4GB
            memory_total_bytes=16 * 1024**3,  # 16GB
            temperature_c=65.0,
            gpu_index=0
        )

        self.assertEqual(sample.power_watts, 150.5)
        self.assertEqual(sample.gpu_utilization, 75.0)
        self.assertEqual(sample.temperature_c, 65.0)

    def test_sample_to_dict(self):
        """Test sample serialization."""
        sample = PowerSample(
            timestamp=1000.0,
            power_watts=150.5,
            gpu_utilization=75.0,
            memory_used_bytes=4 * 1024**3,
            memory_total_bytes=16 * 1024**3,
            temperature_c=65.0,
            gpu_index=0
        )

        d = sample.to_dict()

        self.assertIn("timestamp", d)
        self.assertIn("timestamp_iso", d)
        self.assertIn("power_watts", d)
        self.assertIn("memory_used_mb", d)
        self.assertIn("memory_utilization", d)
        self.assertAlmostEqual(d["memory_utilization"], 25.0, places=1)


class TestPowerMonitor(unittest.TestCase):
    """Test PowerMonitor class."""

    def setUp(self):
        """Set up test fixtures."""
        self.monitor = PowerMonitor(
            sample_rate_hz=10,
            auto_calibrate_idle=False  # Skip calibration for faster tests
        )

    def tearDown(self):
        """Clean up after tests."""
        if self.monitor._state == MonitorState.RUNNING:
            self.monitor.stop()

    def test_monitor_start_stop(self):
        """Test starting and stopping the monitor."""
        self.assertEqual(self.monitor._state, MonitorState.STOPPED)

        result = self.monitor.start()
        self.assertTrue(result)
        self.assertEqual(self.monitor._state, MonitorState.RUNNING)

        self.monitor.stop()
        self.assertEqual(self.monitor._state, MonitorState.STOPPED)

    def test_monitor_context_manager(self):
        """Test using monitor as context manager."""
        with PowerMonitor(sample_rate_hz=10, auto_calibrate_idle=False) as monitor:
            self.assertEqual(monitor._state, MonitorState.RUNNING)
            time.sleep(0.2)

        self.assertEqual(monitor._state, MonitorState.STOPPED)

    def test_sample_collection(self):
        """Test that samples are collected at expected rate."""
        self.monitor.start()
        time.sleep(0.5)  # Collect ~5 samples
        self.monitor.stop()

        samples = self.monitor.get_samples()
        # Should have approximately 5 samples (allow some tolerance)
        self.assertGreaterEqual(len(samples), 3)
        self.assertLessEqual(len(samples), 8)

    def test_request_tracking(self):
        """Test request begin/end tracking."""
        self.monitor.start()
        time.sleep(0.1)

        # Begin a request
        request_id = self.monitor.begin_request(metadata={"model": "test"})
        self.assertIsNotNone(request_id)
        self.assertIn(request_id, self.monitor._active_requests)

        time.sleep(0.3)

        # End the request
        attribution = self.monitor.end_request(request_id)

        self.assertIsInstance(attribution, RequestEnergyAttribution)
        self.assertEqual(attribution.request_id, request_id)
        self.assertGreater(attribution.duration_seconds, 0.2)
        self.assertNotIn(request_id, self.monitor._active_requests)

        self.monitor.stop()

    def test_duplicate_request_id_raises(self):
        """Test that duplicate request IDs raise an error."""
        self.monitor.start()

        request_id = self.monitor.begin_request(request_id="test-123")

        with self.assertRaises(ValueError):
            self.monitor.begin_request(request_id="test-123")

        self.monitor.end_request(request_id)
        self.monitor.stop()

    def test_unknown_request_id_raises(self):
        """Test that ending unknown request raises an error."""
        self.monitor.start()

        with self.assertRaises(ValueError):
            self.monitor.end_request("nonexistent-id")

        self.monitor.stop()

    def test_energy_integration(self):
        """Test energy integration from samples."""
        # Create mock samples
        samples = [
            PowerSample(timestamp=0.0, power_watts=100, gpu_utilization=50,
                       memory_used_bytes=1e9, memory_total_bytes=16e9, temperature_c=50),
            PowerSample(timestamp=1.0, power_watts=100, gpu_utilization=50,
                       memory_used_bytes=1e9, memory_total_bytes=16e9, temperature_c=50),
            PowerSample(timestamp=2.0, power_watts=100, gpu_utilization=50,
                       memory_used_bytes=1e9, memory_total_bytes=16e9, temperature_c=50),
        ]

        # 100W for 2 seconds = 200 Joules
        energy = self.monitor._integrate_energy(samples)
        self.assertAlmostEqual(energy, 200.0, places=1)

    def test_statistics(self):
        """Test statistics collection."""
        self.monitor.start()
        time.sleep(0.3)
        self.monitor.stop()

        stats = self.monitor.get_statistics()

        self.assertIn("state", stats)
        self.assertIn("total_samples", stats)
        self.assertIn("runtime_seconds", stats)
        self.assertGreater(stats["runtime_seconds"], 0)

    def test_csv_export(self):
        """Test CSV export functionality."""
        self.monitor.start()
        time.sleep(0.3)
        self.monitor.stop()

        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            filepath = f.name

        try:
            self.monitor.export_csv(filepath)

            with open(filepath, 'r') as f:
                content = f.read()

            self.assertIn("timestamp", content)
            self.assertIn("power_watts", content)
            self.assertIn("gpu_utilization", content)
        finally:
            os.unlink(filepath)

    def test_json_export(self):
        """Test JSON export functionality."""
        self.monitor.start()

        request_id = self.monitor.begin_request()
        time.sleep(0.2)
        self.monitor.end_request(request_id)

        self.monitor.stop()

        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            filepath = f.name

        try:
            self.monitor.export_json(filepath)

            with open(filepath, 'r') as f:
                data = json.load(f)

            self.assertIn("metadata", data)
            self.assertIn("statistics", data)
            self.assertIn("samples", data)
            self.assertIn("attributions", data)

            self.assertEqual(len(data["attributions"]), 1)
        finally:
            os.unlink(filepath)


class TestEnergyAttribution(unittest.TestCase):
    """Test energy attribution algorithms."""

    def test_equal_share_attribution(self):
        """Test equal share attribution algorithm."""
        monitor = PowerMonitor(
            sample_rate_hz=10,
            attribution_algorithm=EnergyAttributionAlgorithm.EQUAL_SHARE,
            auto_calibrate_idle=False
        )

        monitor.start()
        time.sleep(0.1)

        # Start two overlapping requests
        req1 = monitor.begin_request()
        time.sleep(0.1)
        req2 = monitor.begin_request()
        time.sleep(0.2)
        attr1 = monitor.end_request(req1)
        time.sleep(0.1)
        attr2 = monitor.end_request(req2)

        monitor.stop()

        # Both should have energy attributed
        self.assertGreater(attr1.attributed_energy_joules, 0)
        self.assertGreater(attr2.attributed_energy_joules, 0)

        # First request should have lower confidence due to overlap
        self.assertLess(attr1.confidence_score, 1.0)

    def test_marginal_cost_attribution(self):
        """Test marginal cost attribution algorithm."""
        monitor = PowerMonitor(
            sample_rate_hz=10,
            attribution_algorithm=EnergyAttributionAlgorithm.MARGINAL_COST,
            auto_calibrate_idle=False
        )
        monitor._idle_power_watts = 50.0  # Set known baseline

        monitor.start()
        time.sleep(0.1)

        req = monitor.begin_request()
        time.sleep(0.3)
        attr = monitor.end_request(req)

        monitor.stop()

        # Should have energy above idle baseline
        self.assertGreater(attr.attributed_energy_joules, 0)
        self.assertEqual(attr.attribution_method, "marginal_cost")


class TestThreadSafety(unittest.TestCase):
    """Test thread safety of power monitor."""

    def test_concurrent_requests(self):
        """Test handling many concurrent requests."""
        monitor = PowerMonitor(sample_rate_hz=10, auto_calibrate_idle=False)
        monitor.start()
        time.sleep(0.1)

        request_ids = []
        attributions = []
        errors = []

        def worker(i):
            try:
                req_id = monitor.begin_request(metadata={"worker": i})
                request_ids.append(req_id)
                time.sleep(0.1 + (i % 3) * 0.05)
                attr = monitor.end_request(req_id)
                attributions.append(attr)
            except Exception as e:
                errors.append(str(e))

        # Run 10 concurrent workers
        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(worker, i) for i in range(10)]
            for f in futures:
                f.result()

        monitor.stop()

        # Should have no errors
        self.assertEqual(len(errors), 0, f"Errors: {errors}")

        # Should have 10 attributions
        self.assertEqual(len(attributions), 10)

        # All should have positive energy
        for attr in attributions:
            self.assertGreater(attr.attributed_energy_joules, 0)

    def test_concurrent_access_to_samples(self):
        """Test concurrent read/write to sample buffer."""
        monitor = PowerMonitor(sample_rate_hz=100, auto_calibrate_idle=False)
        monitor.start()

        read_count = [0]
        errors = []

        def reader():
            for _ in range(100):
                try:
                    _ = monitor.get_samples()
                    _ = monitor.get_current_power()
                    _ = monitor.get_total_energy()
                    read_count[0] += 1
                    time.sleep(0.001)
                except Exception as e:
                    errors.append(str(e))

        # Multiple readers while sampling is active
        with ThreadPoolExecutor(max_workers=5) as executor:
            futures = [executor.submit(reader) for _ in range(5)]
            for f in futures:
                f.result()

        monitor.stop()

        self.assertEqual(len(errors), 0, f"Errors: {errors}")
        self.assertEqual(read_count[0], 500)


class TestIdlePowerTracker(unittest.TestCase):
    """Test idle power baseline tracking."""

    def test_idle_baseline_calculation(self):
        """Test idle baseline is calculated correctly."""
        tracker = IdlePowerTracker()

        # Add idle samples
        base_time = time.time()
        for i in range(100):
            tracker.add_sample(
                power_watts=50 + (i % 5),  # 50-55W
                gpu_utilization=2.0,  # Low utilization
                timestamp=base_time + i
            )

        # Force recalculation
        tracker._recalculate_baseline(base_time + 100)

        baseline = tracker.get_baseline()
        # Should be around 50-52W (lower percentile)
        self.assertGreater(baseline, 48)
        self.assertLess(baseline, 55)

    def test_ignores_high_utilization(self):
        """Test that high utilization samples are ignored."""
        tracker = IdlePowerTracker()

        base_time = time.time()

        # Add high utilization samples
        for i in range(50):
            tracker.add_sample(
                power_watts=200,  # High power
                gpu_utilization=80.0,  # High utilization
                timestamp=base_time + i
            )

        # Add low utilization samples
        for i in range(50):
            tracker.add_sample(
                power_watts=55,
                gpu_utilization=3.0,
                timestamp=base_time + 50 + i
            )

        tracker._recalculate_baseline(base_time + 100)

        baseline = tracker.get_baseline()
        # Should be around 55W, not 200W
        self.assertLess(baseline, 60)


class TestOverlapResolver(unittest.TestCase):
    """Test overlap resolution algorithms."""

    def test_time_sliced_no_overlap(self):
        """Test time-sliced attribution without overlap."""
        resolver = OverlapResolver()

        samples = [(i * 0.1, 100.0) for i in range(10)]  # 1 second, 100W
        requests = {
            "req1": (0.0, 1.0)
        }

        energy, confidence, overlap = resolver.resolve_time_sliced(
            target_request_id="req1",
            target_start=0.0,
            target_end=1.0,
            all_requests=requests,
            samples=samples,
            idle_power=50.0
        )

        # 100W for 0.9 seconds = 90 Joules (trapezoidal integration)
        self.assertGreater(energy, 80)
        self.assertLess(energy, 100)
        self.assertEqual(confidence, 1.0)

    def test_time_sliced_with_overlap(self):
        """Test time-sliced attribution with overlap."""
        resolver = OverlapResolver()

        samples = [(i * 0.1, 100.0) for i in range(20)]
        requests = {
            "req1": (0.0, 1.5),
            "req2": (0.5, 2.0)
        }

        energy1, conf1, _ = resolver.resolve_time_sliced(
            target_request_id="req1",
            target_start=0.0,
            target_end=1.5,
            all_requests=requests,
            samples=samples,
            idle_power=50.0
        )

        energy2, conf2, _ = resolver.resolve_time_sliced(
            target_request_id="req2",
            target_start=0.5,
            target_end=2.0,
            all_requests=requests,
            samples=samples,
            idle_power=50.0
        )

        # Both should have energy
        self.assertGreater(energy1, 0)
        self.assertGreater(energy2, 0)

        # Confidence should be reduced due to overlap
        self.assertLess(conf1, 1.0)
        self.assertLess(conf2, 1.0)


class TestMeasurementGapHandler(unittest.TestCase):
    """Test measurement gap handling."""

    def test_detect_gaps(self):
        """Test gap detection."""
        handler = MeasurementGapHandler()

        # Samples with a gap at index 5
        samples = []
        for i in range(10):
            if i == 5:
                # Create a 0.5 second gap
                samples.append((i * 0.1 + 0.4, 100.0))
            else:
                samples.append((i * 0.1, 100.0))

        gaps = handler.detect_gaps(samples, expected_interval=0.1)

        self.assertEqual(len(gaps), 1)
        self.assertEqual(gaps[0]["index"], 5)
        self.assertAlmostEqual(gaps[0]["duration_seconds"], 0.5, places=1)

    def test_interpolate_gap(self):
        """Test gap interpolation."""
        handler = MeasurementGapHandler()

        before = (0.0, 100.0)
        after = (0.5, 150.0)

        interpolated = handler.interpolate_gap(before, after, expected_interval=0.1)

        # Should have 4 interpolated samples (0.1, 0.2, 0.3, 0.4)
        self.assertEqual(len(interpolated), 4)

        # Values should be linearly interpolated
        for t, p in interpolated:
            expected_p = 100 + (150 - 100) * (t / 0.5)
            self.assertAlmostEqual(p, expected_p, places=1)

    def test_estimate_gap_energy(self):
        """Test gap energy estimation."""
        handler = MeasurementGapHandler()

        before = (0.0, 100.0)
        after = (1.0, 100.0)

        energy, uncertainty = handler.estimate_gap_energy(before, after, method="linear")

        # 100W for 1 second = 100 Joules
        self.assertAlmostEqual(energy, 100.0, places=1)
        self.assertAlmostEqual(uncertainty, 0.0, places=1)


class TestThermalThrottleDetector(unittest.TestCase):
    """Test thermal throttle detection."""

    def test_detect_throttling(self):
        """Test thermal throttling detection."""
        detector = ThermalThrottleDetector(
            throttle_temp_threshold_c=80.0,
            power_drop_threshold_percent=15.0
        )

        base_time = time.time()

        # Normal operation
        for i in range(20):
            detector.add_sample(base_time + i, 60.0, 200.0)

        # Temperature increase with power drop (throttling)
        for i in range(20, 30):
            event = detector.add_sample(base_time + i, 82.0, 160.0)

        # Should detect throttling
        self.assertTrue(detector.is_throttling())
        self.assertGreater(len(detector.get_throttle_events()), 0)


class TestPowerLimitDetector(unittest.TestCase):
    """Test power limit detection."""

    def test_detect_power_limit(self):
        """Test power limit detection."""
        detector = PowerLimitDetector()
        detector.set_power_limit(250.0)

        # Add samples below limit
        for _ in range(50):
            detector.add_sample(200.0)

        # Add samples at limit
        for _ in range(50):
            detector.add_sample(248.0)

        ratio = detector.get_limit_hit_ratio()
        self.assertGreater(ratio, 0.4)


class TestEdgeCaseManager(unittest.TestCase):
    """Test unified edge case manager."""

    def test_process_samples(self):
        """Test processing samples through manager."""
        manager = EdgeCaseManager(sample_interval=0.1, power_limit_watts=300.0)

        base_time = time.time()

        for i in range(100):
            status = manager.process_sample(
                timestamp=base_time + i * 0.1,
                power_watts=100 + i % 20,
                gpu_utilization=50 + i % 30,
                temperature_c=55 + i % 10
            )

            self.assertIn("state", status)
            self.assertIn("issues", status)

    def test_health_report(self):
        """Test health report generation."""
        manager = EdgeCaseManager(sample_interval=0.1)

        base_time = time.time()

        for i in range(100):
            manager.process_sample(
                timestamp=base_time + i * 0.1,
                power_watts=100,
                gpu_utilization=50,
                temperature_c=55
            )

        report = manager.get_health_report()

        self.assertIn("status", report)
        self.assertIn("sample_count", report)
        self.assertEqual(report["sample_count"], 100)


class TestValidator(unittest.TestCase):
    """Test PowerMonitorValidator."""

    def test_energy_conservation(self):
        """Test energy conservation validation."""
        monitor = PowerMonitor(sample_rate_hz=10, auto_calibrate_idle=False)
        monitor.start()

        # Create non-overlapping requests
        for _ in range(3):
            req = monitor.begin_request()
            time.sleep(0.2)
            monitor.end_request(req)
            time.sleep(0.1)

        time.sleep(0.1)
        monitor.stop()

        attributions = monitor.get_attributions()
        result = PowerMonitorValidator.validate_energy_conservation(monitor, attributions)

        # Attribution should not exceed total
        self.assertTrue(result["passed"])
        self.assertLessEqual(result["attribution_ratio"], 1.1)  # Allow 10% tolerance


if __name__ == "__main__":
    unittest.main(verbosity=2)
