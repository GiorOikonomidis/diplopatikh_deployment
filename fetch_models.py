"""Model loading (from MLflow) and ONNX export for the deployed router +
G1/G3/G2 experts. ONNX export makes the pipeline loadable with just
onnxruntime -- no torch/lightgbm/mlflow needed at inference time (see
run_onnx.py).
"""

from __future__ import annotations

from pathlib import Path

import joblib
import lightgbm as lgb
import torch
import torch.nn as nn
from onnxmltools import convert_lightgbm
from onnxmltools.convert.common.data_types import FloatTensorType

MLFLOW_TRACKING_URI = "http://192.168.1.57:5000"
ROUTER_MODEL_URI = "models:/layer1_64x64_ovn25_indep10x150k_unweighted/latest"
EXPERT_MODEL_URIS = {
    "G1": "models:/singlestage_g1_optuna/latest",
    "G3": "models:/singlestage_g3_optuna/latest",
    "G2": "models:/singlestage_g2_optuna/latest",
}

META_DIR = Path("meta/")
SCALER_PATH = META_DIR / "scaler_tile_knn_25_OVN_09.joblib"
LABEL_MAP_PATH = META_DIR / "label_map_tile_knn_25_OVN_09.joblib"
ONNX_MODEL_DIR = Path("onnx_models/")

RUN_ON_GPU = False


class ScaledRouter(nn.Module):
    """Wraps the router MLP with its feature-scaling step folded in, so
    the ONNX export takes raw (unscaled) features directly."""

    def __init__(self, router: nn.Module, scaler: object) -> None:
        """Store the router and bake the scaler's mean/std in as buffers.

        Input:
            router: the trained router MLP module.
            scaler: a fitted sklearn StandardScaler (uses .mean_/.scale_).
        Output:
            None.
        """
        super().__init__()
        self.router = router
        self.register_buffer("mean", torch.as_tensor(scaler.mean_, dtype=torch.float32))
        self.register_buffer("std", torch.as_tensor(scaler.scale_, dtype=torch.float32))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Scale the input and run it through the wrapped router.

        Input:
            x: raw (unscaled) feature batch, shape (batch, n_features).
        Output:
            Router logits, shape (batch, n_classes).
        """
        x = (x - self.mean) / self.std
        return self.router(x)


class ModelFetcher:
    """Loads the deployed router + G1/G3/G2 experts from the MLflow
    registry, and exports them to ONNX."""

    def __init__(self, tracking_uri: str = MLFLOW_TRACKING_URI,
                 run_on_gpu: bool = RUN_ON_GPU) -> None:
        """Configure where to load models from and which device to place
        the router on.

        Input:
            tracking_uri: MLflow tracking server URL.
            run_on_gpu: whether to place the loaded router on CUDA (falls
                back to CPU if no GPU is visible).
        Output:
            None.
        """
        self.tracking_uri: str = tracking_uri
        self.device: str = "cuda" if run_on_gpu and torch.cuda.is_available() else "cpu"

    def load_models(self) -> tuple[nn.Module, dict[str, lgb.Booster]]:
        """Load the router and every group expert from MLflow.

        Input:
            None.
        Output:
            (router, experts) where experts maps group name ("G1"/"G3"/"G2")
            to its trained LightGBM Booster.
        """
        import mlflow.lightgbm
        import mlflow.pytorch

        mlflow.set_tracking_uri(self.tracking_uri)

        router = mlflow.pytorch.load_model(model_uri=ROUTER_MODEL_URI, map_location=self.device)
        router.eval()

        experts = {name: mlflow.lightgbm.load_model(uri)
                   for name, uri in EXPERT_MODEL_URIS.items()}
        return router, experts

    @staticmethod
    def load_meta() -> tuple[object | None, dict | None]:
        """Load the fitted scaler and label map used at training time.

        Input:
            None.
        Output:
            (scaler, label_map); either is None if its file is missing.
        """
        scaler = joblib.load(SCALER_PATH) if SCALER_PATH.is_file() else None
        label_map = joblib.load(LABEL_MAP_PATH) if LABEL_MAP_PATH.is_file() else None
        return scaler, label_map

    def export_router_to_onnx(self, router: nn.Module, scaler: object,
                              output_path: Path, num_of_features: int) -> None:
        """Export the router (with scaling folded in) to ONNX.

        Input:
            router: the trained router MLP module.
            scaler: fitted sklearn StandardScaler for the router's raw
                input features.
            output_path: destination .onnx file path.
            num_of_features: number of raw input features the router
                takes, used to build the dummy trace input.
        Output:
            None (writes output_path).
        """
        scaled_router = ScaledRouter(router, scaler)
        scaled_router.eval()
        dummy_input = torch.randn(1, num_of_features, device=self.device)
        torch.onnx.export(
            scaled_router, dummy_input, str(output_path),
            input_names=["input"], output_names=["logits"],
            dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}},
            opset_version=17,
        )

    @staticmethod
    def export_expert_to_onnx(expert: lgb.Booster, output_path: Path,
                              num_of_features: int) -> None:
        """Export one LightGBM expert to ONNX.

        Input:
            expert: a trained LightGBM Booster.
            output_path: destination .onnx file path.
            num_of_features: number of raw input features the expert was
                trained on.
        Output:
            None (writes output_path).
        """
        onx = convert_lightgbm(
            expert, initial_types=[("input", FloatTensorType([None, num_of_features]))],
            zipmap=False,
        )
        with open(output_path, "wb") as fh:
            fh.write(onx.SerializeToString())

    def export_all(self, onnx_model_dir: Path = ONNX_MODEL_DIR) -> None:
        """Load every model + the fitted scaler, then export the full
        set (router + every group expert) to ONNX.

        Input:
            onnx_model_dir: directory to write the .onnx files into
                (created if missing).
        Output:
            None (writes router.onnx and expert_<group>.onnx for every
            group in EXPERT_MODEL_URIS under onnx_model_dir).
        """
        router, experts = self.load_models()
        scaler, _ = self.load_meta()
        onnx_model_dir.mkdir(parents=True, exist_ok=True)
        num_of_features = len(scaler.mean_)

        self.export_router_to_onnx(router, scaler, onnx_model_dir / "router.onnx",
                                   num_of_features)
        for group, expert in experts.items():
            self.export_expert_to_onnx(expert, onnx_model_dir / f"expert_{group}.onnx",
                                       num_of_features)


if __name__ == "__main__":
    ModelFetcher().export_all()
