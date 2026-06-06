import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from typing import Literal, Dict, Optional, Any
import io

from model_complex import Calibration, FactoryModel, ModelParams


def resize_image_bytes(img_bytes: bytes, future_size: tuple = (900, 600)) -> bytes:
    """Изменяет размер изображения из байтов и возвращает байты"""
    img = Image.open(io.BytesIO(img_bytes))
    img_resized = img.resize(future_size, Image.Resampling.LANCZOS)
    buf = io.BytesIO()
    img_resized.save(buf, format="png")
    return buf.getvalue()


def save_plot_to_bytes(dpi: int = 600) -> Dict[str, bytes]:
    """Сохраняет текущий график в BytesIO и возвращает словарь с байтами"""
    # Сохраняем в PNG в память
    buf_png = io.BytesIO()
    plt.savefig(buf_png, format='png', dpi=dpi, bbox_inches="tight")
    buf_png.seek(0)
    
    # Сохраняем в PDF в память
    buf_pdf = io.BytesIO()
    plt.savefig(buf_pdf, format='pdf', bbox_inches="tight")
    buf_pdf.seek(0)
    
    return {
        "png": buf_png.getvalue(),
        "pdf": buf_pdf.getvalue()
    }


def save_plot_to_files(save_path: str, suffix: str, dpi: int = 600) -> Dict[str, str]:
    """Сохраняет график в файлы и возвращает пути"""
    img_path = save_path + suffix
    plt.savefig(img_path + ".pdf", bbox_inches="tight")
    plt.savefig(img_path + ".png", dpi=dpi, bbox_inches="tight")
    
    return {
        "png": img_path + ".png",
        "pdf": img_path + ".pdf"
    }


# ========== ОСНОВНАЯ ФУНКЦИЯ ==========

def interval_estimation_plot(
    epid_data,
    city: str,
    method: str,
    type: str,
    save_path: Optional[str] = None,
    epsilon: int = 3000,
    output_mode: Literal["local", "bytes", "both"] = "local"
) -> Dict[str, Any]:
    """
    Создает графики интервальной оценки параметров (alpha и beta)
    
    Args:
        epid_data: данные эпидемии
        city: название города
        method: метод калибровки ("abc" или "mcmc")
        type: тип данных ("total" или "age")
        save_path: путь для сохранения (нужен при output_mode="local" или "both")
        epsilon: параметр epsilon для калибровки
        output_mode: режим вывода
            - "local": сохранить локально (требует save_path)
            - "bytes": вернуть байты (без сохранения на диск)
            - "both": и сохранить локально, и вернуть байты
    
    Returns:
        Dict с результатами в зависимости от режима
    """
    
    # Проверка save_path для локального сохранения
    if output_mode in ["local", "both"] and not save_path:
        raise ValueError(f"save_path is required when output_mode='{output_mode}'")
    
    # 1. Подготовка данных
    epid_data.get_wave_data(type=type)
    data = epid_data.get_data()
    
    model_params = ModelParams(
        alpha=[0],
        beta=[0],
        population_size=epid_data.get_rho() // 10,
        initial_infectious=[100],
    )
    
    model = FactoryModel.get_model(type)
    
    # при добавлении новых моделей, нужно эту часть обновлять
    if type == "age":
        model_params.initial_infectious = [100, 100]
        label_alpha = {0: "0-14 years", 1: "15+ years"}
    else:
        label_alpha = {0: "total"}
    
    # 2. Калибровка
    calibration = Calibration(model, data, model_params)
    
    if method.lower() == "abc":
        calibration.abc_calibration(epsilon=epsilon)
    else:
        calibration.mcmc_calibration(epsilon=epsilon)
    
    # 3. Сбор параметров из доверительных интервалов
    alpha = []
    beta = []
    
    for params in model.get_ci_params():
        alpha.append(params.alpha)
        beta.append(params.beta)
    
    alpha, beta = np.array(alpha), np.array(beta)
    
    # Результаты для возврата
    results = {}
    
    # 4. График для alpha
    if type == "age":
        fig, axes = plt.subplots(1, 2, figsize=(10, 5))
        for group in range(len(alpha[0])):
            sns.histplot(alpha[:, group], ax=axes[group], kde=True)
            axes[group].set_title(f"{label_alpha[group]} alpha")
    else:
        fig, axes = plt.subplots(1, 1, figsize=(5, 5))
        sns.histplot(alpha[:, 0], ax=axes, kde=True)
        axes.set_title(f"{label_alpha[0]} alpha")
    
    # Сохранение alpha графика
    alpha_filename = f"IE_alpha_{city}_{method}_{type}"
    
    if output_mode in ["local", "both"]:
        alpha_paths = save_plot_to_files(save_path, alpha_filename)
        results["alpha_local"] = alpha_paths
    
    if output_mode in ["bytes", "both"]:
        alpha_bytes = save_plot_to_bytes(dpi=600)
        results["alpha_bytes"] = alpha_bytes
    
    plt.clf()
    
    # 5. График для beta
    if type == "age":
        fig, axes = plt.subplots(2, 2, figsize=(10, 10))
        for group in range(len(beta[0])):
            sns.histplot(beta[:, group], ax=axes[group // 2][group % 2], kde=True)
    else:
        fig, axes = plt.subplots(1, 1, figsize=(5, 5))
        sns.histplot(beta[:, 0], ax=axes, kde=True)
    
    fig.suptitle("beta interval estimation")
    
    # Сохранение beta графика
    beta_filename = f"IE_beta_{city}_{method}_{type}"
    
    if output_mode in ["local", "both"]:
        beta_paths = save_plot_to_files(save_path, beta_filename)
        results["beta_local"] = beta_paths
    
    if output_mode in ["bytes", "both"]:
        beta_bytes = save_plot_to_bytes(dpi=600)
        results["beta_bytes"] = beta_bytes
    
    plt.clf()
    
    # 6. Формирование результата в зависимости от режима
    if output_mode == "local":
        return {
            "alpha": results["alpha_local"],
            "beta": results["beta_local"]
        }
    elif output_mode == "bytes":
        return {
            "alpha": results["alpha_bytes"],
            "beta": results["beta_bytes"]
        }
    else:  # both
        return {
            "local": {
                "alpha": results["alpha_local"],
                "beta": results["beta_local"]
            },
            "bytes": {
                "alpha": results["alpha_bytes"],
                "beta": results["beta_bytes"]
            }
        }
