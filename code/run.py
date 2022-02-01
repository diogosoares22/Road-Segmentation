import sys
from matplotlib import pyplot as plt
from code.models.baseline_cnn import BaselineCNN
from code.models.drn import DRN
from code.models.unet import Unet
from code.models.transunet import TransUNet
from model_runner import ModelRunner
from visualize_results import save_history_to_csv


def train_model(model_name):
    """
    Train model
    @param model_name: Model's name
    """
    runner = ModelRunner()

    if model_name == "BaselineCNN":
        model_params = {
            "seed": runner.SEED,
            "patch_size": 16,  # must be divisible by 16 and not conflict with image size
            "num_channels": runner.NUM_CHANNELS,
            "num_labels": runner.NUM_LABELS,
        }
        runner_params = {
            "batch_size": 32,
            "epochs": 40,
            "lr": 0.0005
        }
        model = BaselineCNN(model_params)
    elif model_name == "DRN":
        model_params = {
            "seed": runner.SEED,
            "patch_size": 16,  # must be divisible by 16 and not conflict with image size
            "num_channels": runner.NUM_CHANNELS,
            "num_labels": runner.NUM_LABELS,
        }
        runner_params = {
            "batch_size": 32,
            "epochs": 40,
            "lr": 0.0005
        }
        model = DRN(model_params)
    elif model_name == "Unet":
        model_params = {
            "seed": runner.SEED,
            "patch_size": 400,  # must be divisible by 16 and not conflict with image size
            "num_channels": runner.NUM_CHANNELS,
            "num_labels": runner.NUM_LABELS,
        }
        runner_params = {
            "batch_size": 8,
            "epochs": 40,
            "lr": 0.0003
        }
        model = Unet(model_params)
    else:  # model_name == "TransUNet"
        model_params = {
            "seed": runner.SEED,
            "patch_size": 320,  # must be divisible by 16 and not conflict with image size
            "num_channels": runner.NUM_CHANNELS,
            "num_labels": runner.NUM_LABELS,
        }
        runner_params = {
            "batch_size": 8,
            "epochs": 40,
            "lr": 0.0003
        }
        model = TransUNet(model_params)

    model, history = runner.train(runner_params, model, extended_dataset=True)
    print(history.params)
    print(history.history.keys())
    save_history_to_csv(history, "../histories/", model.name)
    plt.plot(history.history["f1_m"])
    plt.show()


def predict_model(model_name, version):
    """
    Predict model
    @param model_name: Model's name
    @param version: Model's version
    """
    runner = ModelRunner()
    models_path = "../models/"

    if model_name == "BaselineCNN":
        model_params = {
            "seed": runner.SEED,
            "patch_size": 16,  # must be divisible by 16 and not conflict with image size
            "num_channels": runner.NUM_CHANNELS,
            "num_labels": runner.NUM_LABELS,
        }
        model = BaselineCNN(model_params, load=True, load_params={"path": models_path, "version": version})
    elif model_name == "DRN":
        model_params = {
            "seed": runner.SEED,
            "patch_size": 16,  # must be divisible by 16 and not conflict with image size
            "num_channels": runner.NUM_CHANNELS,
            "num_labels": runner.NUM_LABELS,
        }
        model = DRN(model_params, load=True, load_params={"path": models_path, "version": version})
    elif model_name == "Unet":
        model_params = {
            "seed": runner.SEED,
            "patch_size": 400,  # must be divisible by 16 and not conflict with image size
            "num_channels": runner.NUM_CHANNELS,
            "num_labels": runner.NUM_LABELS,
        }
        model = Unet(model_params, load=True, load_params={"path": models_path, "version": version})
    elif model_name == "TransUNet":
        model_params = {
            "seed": runner.SEED,
            "patch_size": 320,  # must be divisible by 16 and not conflict with image size
            "num_channels": runner.NUM_CHANNELS,
            "num_labels": runner.NUM_LABELS,
        }
        model = TransUNet(model_params, load=True, load_params={"path": models_path, "version": version})

    else:
        model_params_1 = {
            "seed": runner.SEED,
            "patch_size": 320,  # must be divisible by 16 and not conflict with image size
            "num_channels": runner.NUM_CHANNELS,
            "num_labels": runner.NUM_LABELS,
        }
        model_1 = TransUNet(model_params_1, load=True, load_params={"path": models_path, "version": version})

        model_params_2 = {
            "seed": runner.SEED,
            "patch_size": 400,  # must be divisible by 16 and not conflict with image size
            "num_channels": runner.NUM_CHANNELS,
            "num_labels": runner.NUM_LABELS,
        }
        model_2 = Unet(model_params_2, load=True, load_params={"path": models_path, "version": version})

        runner.predict_ensemble(model_1, model_2, post_processing=True)
        return

    runner.predict(model, post_processing=True)


def run_tuning(model_name, folds=4):
    """
    Tunes model
    @param model_name: Respective Model Name
    @param folds: Number of splits
    """
    runner = ModelRunner()
    if model_name == "BaselineCNN":
        to_tune = ["batch_size", "lr"]
        fixed = {
                 "epochs": 40,
                 "patch_size": 16}
        placeholder = {"lr": 0.005}
        model_params = {
            "seed": runner.SEED,
            "patch_size": 16,  # must be divisible by 16 and not conflict with image size
            "num_channels": runner.NUM_CHANNELS,
            "num_labels": runner.NUM_LABELS,
        }
        print(runner.tune_hyperparameters(to_tune, fixed, placeholder, folds, model_name, model_params, batch_size=(32, 64, 128)))

    elif model_name == "DRN":
        to_tune = ["batch_size", "lr"]
        fixed = {
                 "epochs": 40,
                 "patch_size": 16}
        placeholder = {"lr": 0.0005}
        model_params = {
            "seed": runner.SEED,
            "patch_size": 16,  # must be divisible by 16 and not conflict with image size
            "num_channels": runner.NUM_CHANNELS,
            "num_labels": runner.NUM_LABELS,
        }
        print(runner.tune_hyperparameters(to_tune, fixed, placeholder, folds, model_name, model_params, batch_size=(32, 64, 128)))

    elif model_name == "Unet":
        to_tune = ["patch_size", "batch_size", "lr"]
        fixed = {
                 "epochs": 40,
            }
        placeholder = {"batch_size": 8, "lr": 0.001}
        model_params = {
            "seed": runner.SEED,
            "patch_size": 256,  # must be divisible by 16 and not conflict with image size
            "num_channels": runner.NUM_CHANNELS,
            "num_labels": runner.NUM_LABELS,
        }
        print(runner.tune_hyperparameters(to_tune, fixed, placeholder, folds, model_name, model_params))
    else:
        to_tune = ["patch_size", "batch_size", "lr"]
        fixed = {
                 "epochs": 40,
        }
        placeholder = {"batch_size": 8, "lr": 0.001}
        model_params = {
            "seed": runner.SEED,
            "patch_size": 256,  # must be divisible by 16 and not conflict with image size
            "num_channels": runner.NUM_CHANNELS,
            "num_labels": runner.NUM_LABELS,
        }
        print(runner.tune_hyperparameters(to_tune, fixed, placeholder, folds, model_name, model_params))


if __name__ == "__main__":
    assert len(sys.argv) >= 3
    execution_type = sys.argv[1]
    model_name = sys.argv[2]
    if execution_type == "tune":
        folds = sys.argv[3]
        run_tuning(model_name, folds=int(folds))
    elif execution_type == "train":
        train_model(model_name)
    elif execution_type == "predict":
        version = sys.argv[3]
        predict_model(model_name, version)


