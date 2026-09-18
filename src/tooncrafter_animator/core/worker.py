from __future__ import annotations

import logging
import traceback
from pathlib import Path

import numpy as np
from PySide6.QtCore import QObject, QThread, Signal, Slot

from tooncrafter_animator.core.errors import InferenceCancelled
from tooncrafter_animator.core.images import fit_or_cover, load_rgb, sample_border_color
from tooncrafter_animator.core.models import (
    AspectMode,
    CancellationToken,
    GenerationParams,
    LoadPlan,
    PassRequest,
    ProgressEvent,
)
from tooncrafter_animator.core.pipeline import FRAMES_PER_PASS, plan_intermediates
from tooncrafter_animator.memory import release_torch_memory
from tooncrafter_animator import copy as t

log = logging.getLogger("tooncrafter")


class InferenceWorker(QObject):
    progressed = Signal(object)
    finished = Signal(object)  # list[np.ndarray]
    failed = Signal(str)
    cancelled = Signal()
    staged = Signal(str)

    def __init__(
        self,
        start_path: Path,
        end_path: Path,
        params: GenerationParams,
        load: LoadPlan,
        parent: QObject | None = None,
    ) -> None:
        super().__init__(parent)
        self.start_path = Path(start_path)
        self.end_path = Path(end_path)
        self.params = params
        self.load = load
        self.token = CancellationToken()

    def cancel(self) -> None:
        self.token.cancel()

    @Slot()
    def run(self) -> None:
        adapter = None
        try:
            from tooncrafter_animator.inference.adapter import ToonCrafterAdapter

            start = load_rgb(self.start_path)
            end = load_rgb(self.end_path)
            fill = sample_border_color(start)
            plan = plan_intermediates(self.params.intermediates)
            self.staged.emit(plan.description)
            if plan.n_passes == 0:
                frames = [start, end]
                self.finished.emit(_resize_all(frames, self.params, fill))
                return

            adapter = ToonCrafterAdapter()
            self.staged.emit(t.STATUS_LOADING_CKPT)
            adapter.load(self.load, on_stage=lambda msg: self.staged.emit(msg))
            self.token.raise_if_cancelled()

            def make_request(a: np.ndarray, b: np.ndarray) -> PassRequest:
                return PassRequest(
                    start=a,
                    end=b,
                    prompt=self.params.prompt,
                    seed=self.params.seed,
                    steps=self.params.steps,
                    cfg_scale=self.params.cfg_scale,
                    eta=self.params.eta,
                    frame_stride=self.params.motion_stride,
                    precision=self.params.precision,
                    device=self.params.device,
                    gen_height=self.params.gen_height,
                    gen_width=self.params.gen_width,
                    vram_strategy=self.params.vram_strategy,
                )

            def on_progress(event: ProgressEvent) -> None:
                self.progressed.emit(event)

            native_start_end = None
            if not plan.scout:
                spec = plan.passes[0]
                clip = adapter.interpolate(
                    make_request(start, end),
                    progress=on_progress,
                    cancel=self.token,
                    pass_index=1,
                    n_passes=1,
                )
                mids = [clip[i] for i in spec.take_indices]
                native_start_end = (clip[0], clip[-1])
                assembled = [start] + mids + [end]
            else:
                self.staged.emit(t.STATUS_SCOUT)
                scout = adapter.interpolate(
                    make_request(start, end),
                    progress=on_progress,
                    cancel=self.token,
                    pass_index=1,
                    n_passes=plan.n_passes,
                )
                if scout.shape[0] != FRAMES_PER_PASS:
                    raise RuntimeError(t.ERR_SCOUT_FRAMES.format(got=scout.shape[0], expected=FRAMES_PER_PASS))
                anchors = [scout[i] for i in plan.anchor_indices]
                mids: list[np.ndarray] = []
                for i, spec in enumerate(plan.passes):
                    self.token.raise_if_cancelled()
                    a = anchors[i]
                    b = anchors[i + 1]
                    clip = adapter.interpolate(
                        make_request(a, b),
                        progress=on_progress,
                        cancel=self.token,
                        pass_index=i + 2,
                        n_passes=plan.n_passes,
                    )
                    picked = [clip[j] for j in spec.take_indices]
                    mids.extend(picked)
                    if spec.include_end_anchor:
                        mids.append(b)
                assembled = [start] + mids + [end]

            # Drop the raw keyframe copies at the ends for output? Spec is start/end keyframes
            # plus intermediates. Keep user keyframes as endpoints (they asked for interpolation
            # between those images) and generated mids in between.
            resized = _resize_all(assembled, self.params, fill)
            self.finished.emit(resized)
        except InferenceCancelled:
            log.info("inference_cancelled")
            self.cancelled.emit()
        except Exception as exc:  # noqa: BLE001
            log.error("inference_failed error=%s\n%s", exc, traceback.format_exc())
            self.failed.emit(str(exc))
        finally:
            if adapter is not None:
                try:
                    adapter.unload()
                except Exception as exc:  # pragma: no cover
                    log.warning("adapter_unload_failed error=%s", exc)
            release_torch_memory()


def _resize_all(frames: list[np.ndarray], params: GenerationParams, fill: tuple[int, int, int]) -> list[np.ndarray]:
    return [
        fit_or_cover(frame, params.output_width, params.output_height, params.aspect, fill=fill)
        for frame in frames
    ]


class WorkerHost:
    """Owns the QThread so the UI can cancel and join without killing it."""

    def __init__(self) -> None:
        self.thread: QThread | None = None
        self.worker: InferenceWorker | None = None

    def start(self, worker: InferenceWorker) -> None:
        self.stop()
        self.worker = worker
        thread = QThread()
        worker.moveToThread(thread)
        thread.started.connect(worker.run)
        worker.finished.connect(thread.quit)
        worker.failed.connect(thread.quit)
        worker.cancelled.connect(thread.quit)
        thread.finished.connect(worker.deleteLater)
        self.thread = thread
        thread.start()

    def cancel(self) -> None:
        if self.worker is not None:
            self.worker.cancel()

    def is_running(self) -> bool:
        return self.thread is not None and self.thread.isRunning()

    def stop(self) -> None:
        if self.thread is not None:
            self.cancel()
            self.thread.quit()
            self.thread.wait(2_000)
        self.thread = None
        self.worker = None
