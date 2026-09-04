"""ONNX-only inference for the router+experts pipeline -- no torch/lightgbm/
mlflow import needed at runtime, just onnxruntime + the 4 .onnx files
fetch_models.py's ModelFetcher exports. One flow in, one final class out.

Usage:
    python run_onnx.py
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import onnxruntime as ort

from consts import CLASSES, EXPERT_CLASSES, EXPERT_NAMES, FEATURE_NAMES

ONNX_MODEL_DIR = Path("onnx_models/")


class OnnxPipeline:
    """Loads the router + every group expert's ONNX model once, and
    classifies flows one at a time -- the real deployment shape."""

    def __init__(self, onnx_model_dir: Path = ONNX_MODEL_DIR,
                 intra_op_num_threads: int = 1) -> None:
        """Load the router and every expert session up front.

        Input:
            onnx_model_dir: directory holding router.onnx and
                expert_<group>.onnx for every group in EXPERT_NAMES.
            intra_op_num_threads: onnxruntime SessionOptions thread cap.
        Output:
            None.
        """
        so = ort.SessionOptions()
        so.intra_op_num_threads = intra_op_num_threads

        self.router: ort.InferenceSession = ort.InferenceSession(
            str(onnx_model_dir / "router.onnx"), so, providers=["CPUExecutionProvider"])
        self._router_in: str = self.router.get_inputs()[0].name
        self._router_out: str = self.router.get_outputs()[0].name

        # each entry: (session, input_name, probabilities_output_name,
        # class_names) -- LightGBM->ONNX with zipmap=False emits
        # (label, probabilities), so the probabilities output is index 1.
        self.experts: dict[str, tuple[ort.InferenceSession, str, str, list[str]]] = {}
        for name in EXPERT_NAMES:
            sess = ort.InferenceSession(
                str(onnx_model_dir / f"expert_{name}.onnx"), so,
                providers=["CPUExecutionProvider"])
            self.experts[name] = (
                sess, sess.get_inputs()[0].name, sess.get_outputs()[1].name,
                EXPERT_CLASSES[name])

    def classify(self, row: np.ndarray,
                 profile: bool = False) -> str | tuple[str, float, float, str | None]:
        """Classify one flow: the router first, then the matching group
        expert if the router routed it to one of EXPERT_NAMES.

        Input:
            row: one flow's raw feature vector, shape (n_features,),
                ordered per consts.FEATURE_NAMES.
            profile: if True, also time each stage separately from the
                scale/reshape glue around it.
        Output:
            The predicted class name; or, if profile=True, a tuple
            (label, router_ms, expert_ms, expert_group) where expert_group
            is the name of the expert that was consulted (one of
            EXPERT_NAMES), or None when the router's own answer was final
            (a hard class or BenignTraffic, no expert involved) -- in
            which case expert_ms is 0.0.
        """
        x_router = row.reshape(1, -1).astype(np.float32)

        t0 = time.perf_counter()
        logits = self.router.run([self._router_out], {self._router_in: x_router})[0]
        t1 = time.perf_counter()
        group = CLASSES[int(np.argmax(logits[0]))]

        if group not in self.experts:
            return (group, (t1 - t0) * 1e3, 0.0, None) if profile else group

        sess, in_name, proba_name, names = self.experts[group]
        xe = row.reshape(1, -1).astype(np.float32)
        t2 = time.perf_counter()
        proba = sess.run([proba_name], {in_name: xe})[0]
        t3 = time.perf_counter()
        label = names[int(np.argmax(proba[0]))]
        return (label, (t1 - t0) * 1e3, (t3 - t2) * 1e3, group) if profile else label


if __name__ == "__main__":
    import pandas as pd

    from metrics.classification_report import ClassificationReporter
    from metrics.energy_consumption import EnergyMonitor
    from metrics.hardware_consumption import HardwareMonitor
    from metrics.report_formatter import format_pipeline_report

    TEST_CSV = "/home/papi/projects/diplwmatikh/code/ids_pipeline/datasets/grouped_test.csv"
    RUN_NAME = "run_onnx_grouped_test"

    with HardwareMonitor() as hw, EnergyMonitor(project_name=RUN_NAME) as energy:
        pipeline = OnnxPipeline()

        df = pd.read_csv(TEST_CSV, usecols=FEATURE_NAMES + ["fine_label"])
        x_raw = df[FEATURE_NAMES].to_numpy(dtype=np.float32)
        n = len(df)

        final, router_ms = [], []
        expert_ms_by_group: dict[str, list[float]] = {name: [] for name in EXPERT_NAMES}
        for row in x_raw:
            label, t_router, t_expert, expert_group = pipeline.classify(row, profile=True)
            final.append(label)
            router_ms.append(t_router)
            if expert_group is not None:
                expert_ms_by_group[expert_group].append(t_expert)
        final = np.array(final)

    reporter = ClassificationReporter(df["fine_label"].to_numpy(), final)
    metrics = reporter.report_dict()
    reporter.save(RUN_NAME)

    report_text = format_pipeline_report(
        dataset_name=Path(TEST_CSV).name,
        n_rows=n,
        router_ms=router_ms,
        expert_ms_by_group=expert_ms_by_group,
        hardware_summary=hw.summary(),
        energy_summary=energy.summary(),
        accuracy=metrics["accuracy"],
        macro_f1=metrics["macro avg"]["f1-score"],
        weighted_f1=metrics["weighted avg"]["f1-score"],
    )
    print(report_text)
    print(f"\nfull classification report + confusion matrix saved under "
          f"{reporter.output_dir} (prefix '{RUN_NAME}')")
