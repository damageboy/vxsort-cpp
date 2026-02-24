from unittest.mock import MagicMock, patch

import psutil

from src.memory_monitor import MemorySnapshot, _fmt_bytes, collect_memory_snapshot


class TestFmtBytes:
    def test_megabytes(self):
        assert _fmt_bytes(100 * 1024 * 1024) == "100 MB"

    def test_megabytes_small(self):
        assert _fmt_bytes(1 * 1024 * 1024) == "1 MB"

    def test_gigabytes(self):
        assert _fmt_bytes(1024 * 1024 * 1024) == "1.0 GB"

    def test_gigabytes_fractional(self):
        assert _fmt_bytes(int(1.5 * 1024 * 1024 * 1024)) == "1.5 GB"

    def test_zero(self):
        assert _fmt_bytes(0) == "0 MB"


class TestMemorySnapshot:
    def test_total_rss(self):
        snap = MemorySnapshot(
            main_rss_bytes=100 * 1024 * 1024,
            worker_rss_bytes=400 * 1024 * 1024,
            worker_count=4,
        )
        assert snap.total_rss_bytes == 500 * 1024 * 1024

    def test_worker_avg(self):
        snap = MemorySnapshot(
            main_rss_bytes=100 * 1024 * 1024,
            worker_rss_bytes=400 * 1024 * 1024,
            worker_count=4,
        )
        assert snap.worker_avg_rss_bytes == 100 * 1024 * 1024

    def test_worker_avg_zero_workers(self):
        snap = MemorySnapshot(
            main_rss_bytes=100 * 1024 * 1024,
            worker_rss_bytes=0,
            worker_count=0,
        )
        assert snap.worker_avg_rss_bytes == 0

    def test_format_with_workers(self):
        snap = MemorySnapshot(
            main_rss_bytes=245 * 1024 * 1024,
            worker_rss_bytes=int(1.2 * 1024 * 1024 * 1024),
            worker_count=8,
        )
        result = snap.format()
        assert "Main 245 MB" in result
        assert "Workers 1.2 GB" in result
        assert "8 x ~" in result
        assert "Total" in result

    def test_format_no_workers(self):
        snap = MemorySnapshot(
            main_rss_bytes=245 * 1024 * 1024,
            worker_rss_bytes=0,
            worker_count=0,
        )
        result = snap.format()
        assert "Main 245 MB" in result
        assert "Workers: none" in result
        assert "Total 245 MB" in result


class TestCollectMemorySnapshot:
    @patch("src.memory_monitor.psutil.Process")
    def test_basic_collection(self, mock_process_cls):
        main_proc = MagicMock()
        main_mem = MagicMock()
        main_mem.rss = 200 * 1024 * 1024
        main_proc.memory_info.return_value = main_mem

        child1 = MagicMock()
        child1_mem = MagicMock()
        child1_mem.rss = 150 * 1024 * 1024
        child1.memory_info.return_value = child1_mem

        child2 = MagicMock()
        child2_mem = MagicMock()
        child2_mem.rss = 150 * 1024 * 1024
        child2.memory_info.return_value = child2_mem

        main_proc.children.return_value = [child1, child2]
        mock_process_cls.return_value = main_proc

        snap = collect_memory_snapshot()
        assert snap is not None
        assert snap.main_rss_bytes == 200 * 1024 * 1024
        assert snap.worker_rss_bytes == 300 * 1024 * 1024
        assert snap.worker_count == 2

    @patch("src.memory_monitor.psutil.Process")
    def test_dead_child_skipped(self, mock_process_cls):
        main_proc = MagicMock()
        main_mem = MagicMock()
        main_mem.rss = 200 * 1024 * 1024
        main_proc.memory_info.return_value = main_mem

        alive_child = MagicMock()
        alive_mem = MagicMock()
        alive_mem.rss = 100 * 1024 * 1024
        alive_child.memory_info.return_value = alive_mem

        dead_child = MagicMock()
        dead_child.memory_info.side_effect = psutil.NoSuchProcess(999)

        main_proc.children.return_value = [alive_child, dead_child]
        mock_process_cls.return_value = main_proc

        snap = collect_memory_snapshot()
        assert snap is not None
        assert snap.worker_count == 1
        assert snap.worker_rss_bytes == 100 * 1024 * 1024

    @patch("src.memory_monitor.psutil.Process")
    def test_main_process_gone(self, mock_process_cls):
        mock_process_cls.side_effect = psutil.NoSuchProcess(999)

        snap = collect_memory_snapshot()
        assert snap is None

    @patch("src.memory_monitor.psutil.Process")
    def test_no_children(self, mock_process_cls):
        main_proc = MagicMock()
        main_mem = MagicMock()
        main_mem.rss = 200 * 1024 * 1024
        main_proc.memory_info.return_value = main_mem
        main_proc.children.return_value = []
        mock_process_cls.return_value = main_proc

        snap = collect_memory_snapshot()
        assert snap is not None
        assert snap.worker_count == 0
        assert snap.worker_rss_bytes == 0
