"""Diagnostics only for months with confirmed repeated accounting discrepancies."""
from pathlib import Path
import hashlib

import numpy as np
import pandas as pd

from .balance import monthly_balance_summary,CHECK_NAMES
from .palette import PALETTE,apply_plot_theme


def generate_systematic_plots(context,folder):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    apply_plot_theme(plt)
    details=context.prepared.balance_details
    summary=monthly_balance_summary(details,context.config)
    selected=summary[summary["Систематическое расхождение"].eq("Да")]
    records=[]
    for (section,period),group in selected.groupby(["Участок дороги","Период"]):
        fig,axes=plt.subplots(len(group),1,figsize=(14,3*len(group)+2),squeeze=False)
        for ax,(_,stat) in zip(axes[:,0],group.iterrows()):
            rows=details[details.road_section.eq(section)&details.timestamp.dt.to_period("M").eq(pd.Period(period))&details["check"].eq(stat["_check"])].sort_values("timestamp")
            times=rows.timestamp
            difference=rows.difference.where(rows.valid)
            ax.fill_between(times,-rows.tolerance,rows.tolerance,color=PALETTE["reference"],alpha=.12,label="Допустимый диапазон")
            ax.plot(times,difference,color=PALETTE["quality"],lw=1,label="Объявлено − сумма")
            significant=rows.significant
            ax.scatter(times[significant],difference[significant],color=PALETTE["anomaly"],s=12,label="Выше допуска")
            ax.axhline(0,color=PALETTE["reference"],lw=.8)
            ax.set_title(CHECK_NAMES[stat["_check"]]+f" · {stat['Часов с расхождением']} ч / {stat['Дней с расхождениями']} дней · знак {stat['Преобладающий знак']} в {stat['Доля преобладающего знака (%)']:.1f}%")
            ax.set_ylabel("Разность, авто")
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%d.%m"))
            ax.xaxis.set_major_locator(mdates.AutoDateLocator(minticks=4,maxticks=9))
            ax.grid(axis="y",alpha=.5)
            ax.legend(loc="best",fontsize=9,frameon=False)
            ax.spines[["top","right"]].set_visible(False)
        fig.suptitle(f"{section} · {period}\nСистематическое расхождение учёта",color=PALETTE["heading"],fontsize=13)
        fig.text(.5,.025,"Показаны только проверки, прошедшие пороги часов, дней и одинакового знака. Малые расхождения сохранены; связанные проверки не означают независимые ошибки.",ha="center",fontsize=9)
        fig.tight_layout(rect=(.02,.08,.99,.91))
        path=Path(folder)/f"{period}_{hashlib.sha256(section.encode('utf-8')).hexdigest()[:10]}.png"
        path.parent.mkdir(parents=True,exist_ok=True)
        try:
            fig.savefig(path,dpi=110)
        finally:
            plt.close(fig)
        records.append({"section":section,"period":period,"file":str(path.name),"kind":"Систематическое расхождение учёта"})
    return records
