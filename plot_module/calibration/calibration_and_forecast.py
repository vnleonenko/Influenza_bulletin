import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

from model_complex import Calibration, FactoryModel, ModelParams
from PIL import Image

# from ..epid_results import prevalence_plot, 

def week_to_label(week_num, year):
    """Конвертирует номер недели в метку с учетом нового года"""
    while week_num > 52:
        week_num -= 52
        year += 1
    
    return f"{week_num}"
    # return f"{week_num}нед,\n{year:02d}"

def prepare_calibration_data(epid_data, type):
    epid_data.get_wave_data(type=type)
    data = epid_data.get_data()
    dur = epid_data.get_duration()
    plot_data = epid_data.prepare_for_plot()
    return data, dur, plot_data

def resize_img(file_path: str, future_size: tuple = (900, 330), prefix_name: str = "small", format: str = "png" ):
    # filename = "method_annealing_eps_10_iter_0_time_175907_russia_annealing_total_2025-2025.png"
    img = Image.open(file_path)

    # Ресайз до точного размера
    img_resized = img.resize(future_size, Image.Resampling.LANCZOS)

    new_file_path = file_path.replace(format, "_" + prefix_name + "." +format)
    # # Перезапись файла
    img_resized.save(new_file_path)

    return new_file_path


def calibration_forecast_plot(
    epid_data,
    city,
    method,
    type,
    forecast_duration,
    title,
    save_path,
    epsilon=3000,
    is_prevalence_plot=False,
    is_recovered_plot=False
):

    data, dur, plot_data = prepare_calibration_data(epid_data, type)
    model_params = ModelParams(
        alpha=[0],
        beta=[0],
        population_size=epid_data.get_rho() // 10,
        initial_infectious=[100],
    )

    model = FactoryModel.get_model(type)

    if type == "age":
        model_params.initial_infectious = [100, 100]
        label = {0: "0-14 years", 1: "15+ years"}

    else:
        label = {0: "total"}

    color = {0: "blue", 1: "orange"}

    if data.attrs["time_step"] == "week":
        func_to_get_newly_data = model.get_weekly_newly_infected_by_group

    else:
        func_to_get_newly_data = model.get_daily_newly_infected_by_group

    calibration = Calibration(model, data, model_params)

    if method.lower() == "annealing":
        calibration.annealing_calibration()
    elif method.lower() == "abc":
        calibration.abc_calibration(sample=200, epsilon=epsilon)
    elif method.lower() == "mcmc":
        calibration.mcmc_calibration(sample=200, epsilon=epsilon)
    else:
        calibration.optuna_calibration()

    # if is_prevalence_plot:
    #     prevalence_plot(
    #         st_time,
    #         end_time,
    #         city,
    #         method,
    #         type,
    #         save_path,
    #         model,
    #         data.attrs["time_step"],
    #     )

    # if is_recovered_plot:
    #     recovered_plot(
    #         st_time,
    #         end_time,
    #         city,
    #         method,
    #         type,
    #         save_path,
    #         model,
    #         data.attrs["time_step"],
    #     )

    
    
    
    dur += forecast_duration * 7
    # print(dur)
    array = list(range(epid_data.begin_week, epid_data.begin_week + round(dur / 7)))
    x_labels = [week_to_label(week, epid_data.begin_year) for week in array]

    ci_params = list(model.get_ci_params())

    # magic 10: (10 000)/N but better (10 000 / N) 
    coef = 10000 / epid_data.returned_df["total_population"].iloc[0] 

    # Желаемый размер в пикселях
    desired_width_px = 900
    desired_height_px = 330
    dpi = 600

    # Конвертируем в дюймы (для matplotlib)
    width_inches = desired_width_px / dpi
    height_inches = desired_height_px / dpi

    # Создаем фигуру нужного размера
    plt.figure(figsize=(width_inches*8, height_inches*8))

    for line_num, ci_par in enumerate(ci_params):
        print(f"Line {line_num}: {ci_par}")
        
        model.simulate(params=ci_par, modeling_duration=dur)

        res = func_to_get_newly_data()

        for i in range(len(res)):
            # plt.plot(array, res[i], lw=0.3, alpha=0.5, color=color[i])
            plt.plot(array, res[i] * coef, lw=0.3, alpha=0.5, color='lightblue')


    model.simulate(params=model.get_best_params(), modeling_duration=dur)

    res = func_to_get_newly_data()
    print(res)
    for i in range(len(res)):
        # print(res[i])
        plt.plot(
            array[:len(plot_data[:, i])],
            res[i][:len(plot_data[:, i])] * coef,
            label=f"Лучшая модель, $R^2$: {round(r2_score(plot_data[:, i], res[i][:len(plot_data[:, i])]), 2)}",
            lw = 1.0,
            color='royalblue',
        )
        last_known_idx = len(plot_data[:, i]) - 1
        plt.plot(
                array[last_known_idx:],  # Start from last known point
                res[i][last_known_idx:] * coef,
                '--', color='lightcoral', alpha=1.0,
                label='Прогноз'
            )
        # plt.plot(
        #     array[(len(plot_data[:, i])):],
        #     res[i][(len(plot_data[:, i])):]* coef,
        #      '--', color='lightcoral', alpha=1.0, label='Future data')
        print(plot_data[:, i])
        plt.scatter(array[:len(plot_data[:, i])], plot_data[:, i]* coef, marker= "o", color='blue', zorder=5, label = "Данные")
        print(plot_data[:, i]* coef)


    # plt.title(f"{method.capitalize()}, {type.capitalize()}")
    # plt.title("Заболеваемость по РФ", fontsize=14, fontweight='bold')
    # plt.tight_layout(pad=2.5, h_pad=2.5, w_pad=2.5)
    plt.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.7)
    y_min, y_max = plt.ylim()  # Получаем текущие границы оси Y
    y_ticks = np.arange(0, np.ceil(y_max) + 0.5, 5.0)
    plt.yticks(y_ticks)
    plt.tight_layout(rect=[0.03, 0.05, 0.97, 1]) 
    plt.legend(fontsize=10)
    plt.xlabel('Недели', fontsize=12)
    plt.ylabel('Заболеваемость гриппом на 10 тыс. населения', fontsize=10)
    plt.xticks(array, x_labels, rotation=0, fontsize=10)
    img_path = save_path + f"_{city}_{method}_{type}_{epid_data.begin_year}-{epid_data.end_year}"
    plt.savefig(img_path + ".pdf", bbox_inches="tight")
    img_path = img_path + ".png"
    plt.savefig(img_path, dpi=dpi)

    resize_img(img_path, 
               future_size=(900, 330), 
               prefix_name="small",
               format="png")
    # plt.savefig(save_path + f"_{city}_{method}_{type}_{epid_data.begin_year}-{epid_data.end_year}.png",
    #             dpi=600, bbox_inches="tight")
    # plt.title(title, fontsize=14, fontweight='bold')
    
    plt.clf()
