import os
import time
from utils import set_seed, Logger
import matplotlib.pyplot as plt
import pandas as pd  # 导入 pandas 用于处理 CSV


class ATBase(object):

    def __init__(
        self, model, eps=8.0 / 255, log_dir="./log/", name="", device=None, seed=0
    ):
        set_seed(seed)
        self.model = model
        self.eps = eps
        output_dir = os.path.join(log_dir, name)
        output_dir = os.path.join(
            output_dir,
            time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime()) + "-seed-" + str(seed),
        )

        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        self.output_dir = output_dir
        log_file = os.path.join(output_dir, "output.log")
        self.logger = Logger(log_file)

        if device is None:
            self.device = next(model.parameters()).device
        else:
            self.device = device

    def train(self, opt, scheduler, train_loader, test_loader, epoch):
        raise NotImplementedError

    def save_metrics_results(self, metrics_data, eval_start):
        """
        保存训练指标和测试指标到各自的 CSV 文件，并绘制曲线图保存为 PNG。

        Args:
            metrics_data (dict): 包含所有指标列表的字典。
            eval_start (int): 开始评估测试集和鲁棒性的 epoch 编号（从 0 开始）。
        """
        # 1. 创建两个独立的 DataFrame：训练指标和测试指标
        train_df, test_df = self._create_metrics_dataframes(metrics_data, eval_start)

        # 2. 分别保存指标到 CSV 文件
        self._save_metrics_to_csv(train_df, "train_metrics.csv")
        # 只有测试数据非空时才保存测试 CSV
        if not test_df.empty:
            self._save_metrics_to_csv(test_df, "test_metrics.csv")

        # 3. 绘制并保存指标曲线图
        self._save_metric_plots(train_df, test_df, eval_start)

    def _create_metrics_dataframes(self, metrics_data, eval_start):
        """创建训练和测试指标的两个独立的 Pandas DataFrame。"""

        num_epochs = len(metrics_data["train_acc"])

        # --- 训练指标 DataFrame ---
        train_data = {
            "Epoch": list(range(1, num_epochs + 1)),
            "Train_Loss": metrics_data["train_loss"],
            "Train_Accuracy": metrics_data["train_acc"],
        }
        train_df = pd.DataFrame(train_data)

        # --- 测试指标 DataFrame ---
        # 测试指标只包含 eval_start 之后的 epoch 数据
        test_epochs = list(range(eval_start + 1, num_epochs + 1))

        if metrics_data["test_acc"]:
            test_data = {
                "Epoch": test_epochs,
                "Test_Accuracy_Clean": metrics_data["test_acc"],
                "Test_Accuracy_PGD": metrics_data["pgd_acc"],
                "Test_Accuracy_FGSM": metrics_data["fgsm_acc"],
            }
            test_df = pd.DataFrame(test_data)
        else:
            test_df = pd.DataFrame()  # 创建空 DataFrame

        return train_df, test_df

    def _save_metrics_to_csv(self, df, filename):
        """将 DataFrame 保存为 CSV 文件。"""
        csv_path = os.path.join(self.output_dir, filename)
        df.to_csv(csv_path, index=False)
        self.logger.log(f"Training metrics CSV file saved to: {csv_path}")

    def _save_metric_plots(self, train_df, test_df, eval_start):
        """绘制并保存训练指标曲线图。"""

        epochs = train_df["Epoch"].tolist()

        plt.figure(figsize=(14, 10))

        # 绘制训练损失
        plt.subplot(2, 1, 1)
        plt.plot(epochs, train_df["Train_Loss"], label="Train Loss", color="red")
        plt.title("Training Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        plt.grid(True)

        # 绘制准确率
        plt.subplot(2, 1, 2)
        plt.plot(
            epochs,
            train_df["Train_Accuracy"],
            label="Train Accuracy",
            color="blue",
            linestyle="--",
        )

        # 绘制测试指标
        if not test_df.empty:
            test_epochs_slice = test_df["Epoch"].tolist()

            plt.plot(
                test_epochs_slice,
                test_df["Test_Accuracy_Clean"],
                label="Test Accuracy (Clean)",
                color="green",
            )
            plt.plot(
                test_epochs_slice,
                test_df["Test_Accuracy_PGD"],
                label="Test Accuracy (PGD)",
                color="orange",
            )
            plt.plot(
                test_epochs_slice,
                test_df["Test_Accuracy_FGSM"],
                label="Test Accuracy (FGSM)",
                color="purple",
                linestyle=":",
            )

        plt.title("Accuracy Metrics over Epochs")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.legend(loc="lower right")
        plt.grid(True)

        plt.tight_layout()

        # 保存图像到 output_dir
        plot_path = os.path.join(self.output_dir, "training_metrics.png")
        plt.savefig(plot_path)
        self.logger.log(f"Training metrics plot saved to: {plot_path}")
        plt.close()
