"""Optional monthly overview renderer."""
import hashlib
from pathlib import Path
import pandas as pd
from .palette import PALETTE, apply_plot_theme
BLUE, RED, GRAY, PURPLE = (PALETTE[k] for k in ("fact", "anomaly", "reference", "quality"))

def generate_monthly_plots(context, monthly_events, output_folder):
    """Overview keeps missing-month gaps and displays coverage beside traffic."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    from .detectors import monthly_summary

    output_folder=Path(output_folder)
    output_folder.mkdir(parents=True,exist_ok=True)
    apply_plot_theme(plt)
    summary=monthly_summary(context)
    for section,g in summary.groupby("Участок дороги"):
        g=g.sort_values("Период")
        dates=pd.to_datetime(g["Период"]+"-01")
        fig,axes=plt.subplots(3,1,figsize=(12,9),sharex=True)
        axes[0].plot(dates,g["ССИ"],color=BLUE,marker="o",label="ССИ полных пригодных суток")
        traffic=monthly_events[monthly_events["Участок дороги"].eq(section)&monthly_events["Категория"].eq("Трафик")&monthly_events["_metric"].eq("vol_total")]
        highlighted=dates.isin(traffic["Начало"])
        axes[0].scatter(dates[highlighted],g.loc[highlighted,"ССИ"],color=RED,zorder=4,label="Отклонение ССИ YoY" if highlighted.any() else None)
        axes[0].set_ylabel("Интенсивность\nавт./сут")
        axes[0].set_ylim(bottom=0)
        axes[1].plot(dates,g["Скорость, прямое"],color=BLUE,marker=".",label="Прямое направление")
        axes[1].plot(dates,g["Скорость, обратное"],color=GRAY,ls="--",marker=".",label="Обратное направление")
        axes[1].set_ylabel("Взвешенная скорость\nкм/ч")
        axes[1].set_ylim(bottom=0)
        axes[2].bar(dates,g["Полнота (%)"],width=20,color=GRAY,alpha=.2,label="Измерено часов")
        axes[2].plot(dates,g["Пригодно часов (%)"],color=PURPLE,marker="s",label="Пригодно часов")
        axes[2].axhline(context.config.min_month_coverage,color=RED,ls=":",label=f"Порог {context.config.min_month_coverage:g}%")
        axes[2].set_ylim(0,105)
        axes[2].set_ylabel("Покрытие месяца\n% календарных часов")
        for ax in axes:
            ax.grid(axis="y",alpha=.2)
            ax.legend(loc="best",fontsize=9,frameon=False)
            ax.spines[["top","right"]].set_visible(False)
        axes[2].xaxis.set_major_locator(mdates.MonthLocator(interval=max(1,len(g)//12)))
        axes[2].xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
        axes[2].tick_params(axis="x",rotation=35)
        fig.suptitle(section+"\nМесячный обзор: трафик и качество измерений",fontsize=13,color=PALETTE["heading"])
        note="Нет достаточных данных для YoY: отсутствие меток не означает отсутствие отклонений." if not g["Статус YoY"].eq("Доступно сравнение").any() else "YoY оценивается только для пригодных месяцев; опорные годы указаны в Excel."
        fig.text(.5,.025,note,ha="center",fontsize=9)
        fig.subplots_adjust(left=.10,right=.97,top=.90,bottom=.12,hspace=.2)
        digest=hashlib.sha256(section.encode("utf-8")).hexdigest()[:10]
        try:
            fig.savefig(output_folder/f"monthly_{digest}.png",dpi=110)
        finally:
            plt.close(fig)
